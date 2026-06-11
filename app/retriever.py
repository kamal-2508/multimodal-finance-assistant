# ─────────────────────────────────────────────────────────────────────────────
# app/retriever.py
#
# WHY THIS FILE EXISTS:
#   After extracting fields from a document, we need to:
#     1. Store it in a vector database (ChromaDB) so we can search it later
#     2. Retrieve relevant past documents when answering a question
#     3. Build a RAG (Retrieval-Augmented Generation) chain that passes
#        retrieved context to the LLM so it gives grounded answers
#
# WHAT IS RAG?
#   Without RAG: User asks "Why was I charged $320?"
#               → LLM guesses based on training data (unreliable)
#
#   With RAG:   User asks "Why was I charged $320?"
#               → We search ChromaDB for similar past documents
#               → Find: "AWS EC2 usage charges typically appear as..."
#               → Pass that context to LLM
#               → LLM gives a grounded, specific answer
#
# CHROMADB vs FAISS vs PINECONE:
#   ChromaDB: runs locally, no API key needed, persistent storage, good for dev
#   FAISS: faster but no metadata filtering, harder to update
#   Pinecone: cloud, scalable, but costs money — overkill for this project
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import logging
from typing import Optional
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Embedding model setup ────────────────────────────────────────────────────

def get_embeddings():
    """
    Load the sentence-transformer embedding model.

    WHY all-MiniLM-L6-v2:
    - Completely FREE (runs locally, no API calls)
    - Small (80MB) but surprisingly good for semantic search
    - Returns 384-dimensional vectors — fast to compare
    - Great for financial text: understands "charge", "deduction", "fee" etc.

    Alternative: OpenAI's text-embedding-ada-002 is better but costs money.
    For this project, MiniLM is more than sufficient.
    """
    logger.info("Loading embedding model (all-MiniLM-L6-v2)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},     # Use CPU (no GPU needed for this model)
        encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine similarity
    )

    logger.info("✓ Embedding model loaded")
    return embeddings


# ─── ChromaDB setup ───────────────────────────────────────────────────────────

def get_chroma_client(persist_dir: Optional[str] = None):
    """
    Create a ChromaDB client.

    WHY TWO MODES (persistent vs in-memory):
    - Persistent (local dev): saves embeddings to disk so you don't
      re-embed everything every time you restart the app
    - In-memory (Hugging Face Spaces): HF Spaces free tier has no
      persistent disk, so we use in-memory storage (resets on restart)

    The mode is chosen based on whether persist_dir is provided.
    """
    if persist_dir:
        # Persistent mode: save embeddings to disk
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"Using persistent ChromaDB at: {persist_dir}")
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        # In-memory mode: for Hugging Face Spaces or testing
        logger.info("Using in-memory ChromaDB (data will reset on restart)")
        client = chromadb.Client()

    return client


# ─── Document store ───────────────────────────────────────────────────────────

class FinancialDocumentStore:
    """
    Manages storing and retrieving financial documents using ChromaDB.

    This class wraps ChromaDB with financial-domain-specific logic:
    - Splits documents into chunks the LLM can handle
    - Adds metadata (document type, vendor, amount) for filtered search
    - Provides a simple add/search interface for the Streamlit app
    """

    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize the document store.

        persist_dir: path to save ChromaDB data (None = in-memory for HF Spaces)
        """
        self.embeddings = get_embeddings()
        self.persist_dir = persist_dir or os.getenv("CHROMA_DB_PATH", "./data/chroma_store")

        # Determine if we're in HF Spaces (no persistent storage)
        self.is_hf_spaces = os.getenv("SPACE_ID") is not None

        # Text splitter: breaks long documents into overlapping chunks
        # WHY OVERLAP: If a charge description spans a chunk boundary,
        # overlap ensures neither chunk loses the full context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # Each chunk ~500 characters
            chunk_overlap=50,      # 50-char overlap between chunks
            separators=["\n\n", "\n", ". ", " "],  # Split on natural boundaries first
        )

        # Initialize ChromaDB
        chroma_client = get_chroma_client(
            None if self.is_hf_spaces else self.persist_dir
        )

        # Create or load the LangChain Chroma wrapper
        self.vector_store = Chroma(
            client=chroma_client,
            collection_name="financial_documents",  # Name of our collection
            embedding_function=self.embeddings,
        )

        logger.info("✓ FinancialDocumentStore initialized")

    def add_document(self, extracted_doc: dict) -> int:
        """
        Add an extracted document to the vector store.

        WHY CHUNK METADATA:
        We store vendor name, amount, document type as metadata on each chunk.
        This lets us do filtered searches like:
        "Find all AWS charges over $100" by filtering metadata, not just text.

        Returns: number of chunks added
        """
        # Get the text content to embed
        # raw_text contains the full document content (JSON, markdown, or plain text)
        text = extracted_doc.get("raw_text", "")

        if not text:
            logger.warning("Document has no text content, skipping")
            return 0

        # Split into manageable chunks
        chunks = self.text_splitter.split_text(text)

        if not chunks:
            return 0

        # Build metadata for each chunk
        # This metadata is stored alongside the vector and can be used for filtering
        metadata = {
            "document_type": extracted_doc.get("document_type", "unknown"),
            "vendor_name":   str(extracted_doc.get("vendor_name") or "unknown"),
            "total_amount":  str(extracted_doc.get("total_amount") or "0"),
            "invoice_date":  str(extracted_doc.get("invoice_date") or ""),
            "source_file":   str(extracted_doc.get("source_file") or ""),
            "added_at":      str(time.time()),
        }

        # Create LangChain Document objects (required by Chroma wrapper)
        documents = [
            Document(page_content=chunk, metadata=metadata)
            for chunk in chunks
        ]

        # Add to ChromaDB — this embeds each chunk and stores the vector
        self.vector_store.add_documents(documents)

        logger.info(f"✓ Added {len(chunks)} chunks from document to ChromaDB")
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[Document]:
        """
        Find the k most relevant document chunks for a given query.

        HOW IT WORKS:
        1. Embed the query using the same MiniLM model
        2. Compare query vector to all stored chunk vectors (cosine similarity)
        3. Return top-k most similar chunks

        k=3 means we return the 3 most relevant chunks.
        These chunks become the "context" fed to the LLM.
        """
        if self.get_document_count() == 0:
            logger.warning("ChromaDB is empty — no documents to search")
            return []

        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} relevant chunks for query: '{query[:50]}...'")
        return results

    def get_document_count(self) -> int:
        """Return total number of chunks stored in ChromaDB."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0


# ─── RAG Chain ────────────────────────────────────────────────────────────────

class FinancialRAGChain:
    """
    The main question-answering chain.

    Combines:
    - FinancialDocumentStore (retrieves relevant context)
    - ChatGroq LLM (generates the answer)
    - A carefully designed prompt (guides the LLM to be factual)

    This is the core AI logic of our application.
    """

    def __init__(self, doc_store: FinancialDocumentStore):
        self.doc_store = doc_store
        self.llm = self._create_llm()
        self.chain = self._build_chain()

    def _create_llm(self):
        """
        Initialize the Groq LLM.

        WHY GROQ:
        - Free tier available (generous limits for development)
        - Very fast inference (much faster than OpenAI)
        - Supports Llama 3.1 and other open-source models

        WHY llama-3.1-8b-instant:
        - Fast and free on Groq
        - 8B parameters is enough for document Q&A
        - "instant" variant is optimized for quick responses
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Copy .env.example to .env and add your key from console.groq.com"
            )

        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,       # Low temperature = more factual, less creative
            max_tokens=1024,       # Keep answers concise
            api_key=api_key,
        )

    def _build_chain(self):
        """
        Build the LangChain RAG chain.

        CHAIN FLOW:
        User question
            ↓
        Retrieve top-3 relevant chunks from ChromaDB
            ↓
        Format prompt: [system instructions] + [retrieved context] + [question]
            ↓
        Send to Groq LLM
            ↓
        Parse and return the text answer

        WHY THIS PROMPT:
        We explicitly tell the LLM to:
        1. Only answer from the provided context (prevents hallucination)
        2. Mention specific amounts, dates, vendors (encourages specificity)
        3. Say "I don't know" if context doesn't cover the question (honesty)
        """
        # This is the system prompt — sets the LLM's behavior
        prompt = ChatPromptTemplate.from_template("""
You are a helpful financial document assistant. Your job is to explain 
charges, transactions, and financial entries from uploaded documents.

RULES:
1. Only answer based on the CONTEXT provided below
2. Always mention specific amounts, dates, and vendor names when available
3. If the context doesn't contain enough information, say so clearly
4. Keep answers concise — 2-4 sentences unless more detail is asked for
5. If asked about a charge, explain: what it is, why it occurred, and if it seems normal

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{question}

ANSWER:""")

        # Build the chain using LangChain's pipe operator (|)
        # Each | passes the output of one step as input to the next
        chain = (
            {
                # Step 1: Retrieve relevant chunks and format as a single string
                "context": lambda x: self._format_context(
                    self.doc_store.search(x["question"])
                ),
                # Step 2: Pass through the question unchanged
                "question": RunnablePassthrough() | (lambda x: x["question"]),
            }
            | prompt          # Step 3: Format the prompt template
            | self.llm        # Step 4: Send to Groq LLM
            | StrOutputParser()  # Step 5: Extract just the text from the response
        )

        return chain

    def _format_context(self, documents: list[Document]) -> str:
        """
        Format retrieved document chunks into a readable context string.

        WHY FORMAT IT:
        The LLM receives a single string, so we need to combine all
        retrieved chunks into one coherent block of text. We add
        separators so the LLM can tell where one chunk ends and another begins.
        """
        if not documents:
            return "No relevant documents found in the database."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            # Include metadata as headers so the LLM knows the source
            header = f"[Document {i} | Type: {meta.get('document_type', 'unknown')} | Vendor: {meta.get('vendor_name', 'N/A')}]"
            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def ask(self, question: str, current_doc: Optional[dict] = None) -> dict:
        """
        Main method: ask a question about financial documents.

        current_doc: if provided, add it to the context even if not in ChromaDB yet
        Returns: dict with 'answer', 'sources', 'response_time'

        WHY RETURN SOURCES:
        Showing which documents were used to answer builds user trust.
        It also helps debug cases where the answer seems wrong.
        """
        start_time = time.time()

        # If a current document is provided (just uploaded), add it first
        if current_doc:
            self.doc_store.add_document(current_doc)

        # Retrieve relevant context
        sources = self.doc_store.search(question, k=3)

        # Run the full chain
        try:
            answer = self.chain.invoke({"question": question})
        except Exception as e:
            logger.error(f"LLM chain failed: {e}")
            answer = f"I encountered an error while processing your question. Please try again. Error: {str(e)}"

        elapsed = round(time.time() - start_time, 2)

        return {
            "answer": answer,
            "sources": sources,            # Retrieved chunks used for context
            "response_time": elapsed,      # How long the query took
            "doc_count": self.doc_store.get_document_count(),
        }

    def ask_vision(self, question: str, image_doc: dict) -> dict:
        """
        Special handler for image uploads.

        WHY SEPARATE VISION METHOD:
        When a user uploads an IMAGE (not a PDF), we can't extract
        structured text first. Instead, we send the image directly to
        a vision-capable LLM that can "see" the document.

        NOTE: Groq's free models are text-only. For vision, you'd need
        GPT-4V or Claude. This method shows the pattern but falls back
        to text if vision model is unavailable.
        """
        # For the free Groq tier, describe what we'd do with a vision model
        # In production, replace this with an actual vision API call

        vision_prompt = f"""
I have uploaded an image of a financial document.
Please analyze it and answer: {question}

The document appears to be a: {image_doc.get('document_type', 'financial document')}
Image dimensions: {image_doc.get('dimensions', {}).get('width')}x{image_doc.get('dimensions', {}).get('height')}px

Note: Full vision analysis requires a vision-capable model (e.g., GPT-4V).
Based on the document type, here is what I can tell you:
"""
        # Add image doc to store anyway (for future text-based queries)
        self.doc_store.add_document(image_doc)

        return self.ask(question)
