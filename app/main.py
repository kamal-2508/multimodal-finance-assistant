# ─────────────────────────────────────────────────────────────────────────────
# app/main.py
#
# WHY THIS FILE EXISTS:
#   This is the Streamlit frontend — the user-facing part of the app.
#   Streamlit turns Python scripts into interactive web apps with no HTML/CSS.
#
# HOW TO RUN:
#   streamlit run app/main.py
#
# WHAT THE USER SEES:
#   - Sidebar: file uploader + sample documents + stats
#   - Main area: extracted fields from the uploaded document
#   - Chat interface: ask questions about the document
#   - Expandable section: see which document chunks were used for the answer
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Our custom modules
from extractor import extract_document
from retriever import FinancialDocumentStore, FinancialRAGChain

# Load .env file so GROQ_API_KEY is available
load_dotenv()

# ─── Page config (must be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state initialization ─────────────────────────────────────────────
# WHY SESSION STATE:
# Streamlit re-runs the entire script on every user interaction.
# Session state persists data across re-runs (like a global variable that
# survives page refreshes within the same user session).

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # List of {role, content} dicts

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None      # Currently loaded document

if "doc_store" not in st.session_state:
    st.session_state.doc_store = None        # ChromaDB store (initialized once)

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None        # RAG chain (initialized once)

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0       # Count of questions asked

if "total_response_time" not in st.session_state:
    st.session_state.total_response_time = 0.0


# ─── Initialize RAG components (only once per session) ────────────────────────

@st.cache_resource   # Cache this so it doesn't reload on every interaction
def initialize_rag():
    """
    Initialize ChromaDB and the RAG chain.

    WHY @st.cache_resource:
    Loading the embedding model (MiniLM) takes ~5 seconds.
    Without caching, it would reload every time the user types or clicks.
    cache_resource keeps the initialized objects in memory for the session.
    """
    try:
        # Check if we're on Hugging Face Spaces (in-memory mode)
        is_hf = os.getenv("SPACE_ID") is not None
        persist_dir = None if is_hf else os.getenv("CHROMA_DB_PATH", "./data/chroma_store")

        doc_store = FinancialDocumentStore(persist_dir=persist_dir)
        rag_chain = FinancialRAGChain(doc_store)
        return doc_store, rag_chain, None   # None = no error

    except ValueError as e:
        # This happens if GROQ_API_KEY is missing
        return None, None, str(e)

    except Exception as e:
        return None, None, f"Initialization failed: {str(e)}"


# ─── Helper functions ─────────────────────────────────────────────────────────

def load_sample_document(filepath: str):
    """Load a sample document and add it to the RAG store."""
    with st.spinner(f"Loading {Path(filepath).name}..."):
        try:
            doc = extract_document(filepath)
            st.session_state.current_doc = doc

            if st.session_state.doc_store:
                chunks_added = st.session_state.doc_store.add_document(doc)
                st.success(f"✓ Loaded document ({chunks_added} chunks indexed)")

        except Exception as e:
            st.error(f"Failed to load document: {e}")


def format_currency(amount) -> str:
    """Format a number as currency string."""
    if amount is None:
        return "N/A"
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return str(amount)


def display_extracted_fields(doc: dict):
    """
    Show extracted document fields in a clean table.

    WHY DISPLAY EXTRACTED FIELDS:
    Users can verify the AI correctly read their document before asking
    questions. If the extraction is wrong, the AI's answers will also be wrong.
    This transparency builds trust and helps debugging.
    """
    st.subheader("📋 Extracted Document Fields")

    doc_type = doc.get("document_type", "unknown").replace("_", " ").title()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Document Type", doc_type)
        st.metric("Vendor / Account", doc.get("vendor_name") or "N/A")

    with col2:
        st.metric("Total Amount", format_currency(doc.get("total_amount")))
        st.metric("Tax Amount", format_currency(doc.get("tax_amount")))

    with col3:
        st.metric("Date", doc.get("invoice_date") or "N/A")
        st.metric("Invoice #", doc.get("invoice_number") or "N/A")
        st.metric("Extraction Method", doc.get("extraction_method", "N/A"))

    # Show line items if available
    line_items = doc.get("line_items", [])
    if line_items:
        st.subheader(f"📦 Line Items / Transactions ({len(line_items)} entries)")
        # Convert to a format Streamlit can display as a table
        table_data = []
        for item in line_items[:10]:   # Show max 10 items to keep UI clean
            if isinstance(item, dict):
                table_data.append({
                    "Description": item.get("description") or item.get("merchant", ""),
                    "Amount": format_currency(item.get("amount")),
                    "Date": item.get("date", ""),
                    "Category": item.get("category", ""),
                })
        if table_data:
            st.table(table_data)

        if len(line_items) > 10:
            st.caption(f"... and {len(line_items) - 10} more items")


# ─── Main app layout ──────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🤖 AI Financial Assistant")
    st.caption("Upload a financial document (PDF, image, or JSON) and ask questions about it")

    # ── Initialize RAG (cached) ───────────────────────────────────────────────
    doc_store, rag_chain, init_error = initialize_rag()

    if init_error:
        st.error(f"⚠️ Setup Error: {init_error}")
        st.info("💡 Add your GROQ_API_KEY to the .env file. Get a free key at: https://console.groq.com")
        st.stop()   # Don't render the rest of the app if setup failed

    # Store in session state so other parts of the app can use them
    st.session_state.doc_store = doc_store
    st.session_state.rag_chain = rag_chain

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📁 Document Upload")

        # File uploader — accepts PDF, images, and our sample JSON files
        uploaded_file = st.file_uploader(
            "Upload a financial document",
            type=["pdf", "png", "jpg", "jpeg", "json"],
            help="Supported: PDF invoices, receipt images, credit card statement screenshots",
        )

        if uploaded_file:
            # Save uploaded file to temp location so extractor can read it
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Show image preview if it's an image file
            if uploaded_file.type.startswith("image/"):
                st.image(uploaded_file, caption="Uploaded document", use_container_width=True)

            # Extract and load the document
            load_sample_document(temp_path)

        st.divider()

        # ── Sample documents ─────────────────────────────────────────────────
        st.subheader("📂 Sample Documents")
        st.caption("Try these pre-generated examples")

        sample_dir = "./data/sample_docs"
        if os.path.exists(sample_dir):
            sample_files = sorted(Path(sample_dir).glob("*.json"))[:6]  # Show first 6

            if sample_files:
                for sample_file in sample_files:
                    # Clean up filename for display
                    display_name = sample_file.stem.replace("_", " ").title()
                    if st.button(f"📄 {display_name}", key=str(sample_file)):
                        load_sample_document(str(sample_file))
            else:
                st.warning("No samples found. Run: python data/generate_samples.py")
        else:
            st.info("Generate samples first:\n```\npython data/generate_samples.py\n```")

        st.divider()

        # ── Stats panel ───────────────────────────────────────────────────────
        # WHY STATS: Shows the user the system is working.
        # Developers can use these to spot performance issues.
        st.subheader("📊 Session Stats")

        doc_count = doc_store.get_document_count() if doc_store else 0
        st.metric("Chunks in ChromaDB", doc_count)
        st.metric("Questions Asked", st.session_state.total_queries)

        if st.session_state.total_queries > 0:
            avg_time = st.session_state.total_response_time / st.session_state.total_queries
            st.metric("Avg Response Time", f"{avg_time:.1f}s")

        st.caption("Model: llama-3.1-8b-instant (Groq)")
        st.caption("Embeddings: all-MiniLM-L6-v2 (local)")

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Main content area ─────────────────────────────────────────────────────

    # Show extracted fields if a document is loaded
    if st.session_state.current_doc:
        display_extracted_fields(st.session_state.current_doc)
        st.divider()

    else:
        # Show a welcome message with instructions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Step 1:** Upload a financial document using the sidebar, or select a sample")
        with col2:
            st.info("**Step 2:** Review the extracted fields above to verify accuracy")
        with col3:
            st.info("**Step 3:** Ask questions in the chat below")

    # ── Chat interface ────────────────────────────────────────────────────────
    st.subheader("💬 Ask Questions About Your Document")

    # Display existing chat messages
    # WHY LOOP THROUGH HISTORY: Streamlit re-renders everything from scratch
    # on each interaction, so we must re-draw all past messages each time
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show retrieved sources if it's an assistant message
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📎 Retrieved context (sources used for this answer)"):
                    for i, source in enumerate(message["sources"], 1):
                        st.caption(f"Source {i}: {source.metadata.get('document_type', 'unknown')} | {source.metadata.get('vendor_name', 'N/A')}")
                        st.text(source.page_content[:200] + "...")

    # ── Chat input ────────────────────────────────────────────────────────────
    # Suggested questions to help users get started
    st.caption("💡 Try asking: 'Why was I charged $320.45?' or 'What are the total taxes?' or 'Summarize this invoice'")

    user_input = st.chat_input(
        "Ask a question about your financial document...",
        disabled=(rag_chain is None),
    )

    if user_input:
        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your documents..."):
                try:
                    result = rag_chain.ask(
                        question=user_input,
                        current_doc=None,   # Already added when document was loaded
                    )

                    answer = result["answer"]
                    sources = result["sources"]
                    response_time = result["response_time"]

                    # Display the answer
                    st.write(answer)

                    # Show response metadata
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        st.caption(f"⏱️ {response_time}s")

                    # Show retrieved sources in expander
                    if sources:
                        with st.expander("📎 Retrieved context (sources used for this answer)"):
                            for i, source in enumerate(sources, 1):
                                st.caption(
                                    f"Source {i}: {source.metadata.get('document_type', 'unknown')} "
                                    f"| {source.metadata.get('vendor_name', 'N/A')}"
                                )
                                st.text(source.page_content[:200] + "...")

                    # Update session stats
                    st.session_state.total_queries += 1
                    st.session_state.total_response_time += response_time

                    # Save to chat history (with sources for display later)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
