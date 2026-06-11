---
title: Multimodal Finance Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.1
app_file: app/main.py
pinned: false
---

# 🤖 Multimodal AI Financial Assistant

> Upload a financial document (invoice, credit card statement, receipt) and ask natural language questions about it. Powered by LangChain + ChromaDB + Groq (free).

[![CI/CD](https://github.com/YOUR_USERNAME/multimodal-finance-assistant/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/multimodal-finance-assistant/actions)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/YOUR_USERNAME/multimodal-finance-assistant)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)

## What it does

| Upload | Ask | Get |
|--------|-----|-----|
| PDF invoice / credit card screenshot / JSON export | "Why was I charged $320?" | Grounded answer with source documents shown |

## Architecture

```
User uploads document
        ↓
   extractor.py          ← Docling (PDFs) / base64 (images) / JSON parser
        ↓
  Extracted fields        ← vendor, amount, date, line items
        ↓
   retriever.py           ← ChromaDB stores chunks + similarity search
        ↓
  RAG chain               ← LangChain prompt + Groq LLM (llama-3.1-8b)
        ↓
  Grounded answer         ← with cited source chunks
```

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Document extraction | Docling + PyPDF2 | Docling preserves table layout |
| Vector store | ChromaDB | Free, runs locally, persistent |
| Embeddings | all-MiniLM-L6-v2 | Free, runs on CPU, 80MB |
| LLM | Groq / llama-3.1-8b-instant | Free tier, very fast |
| Frontend | Streamlit | Python-only, no HTML needed |
| CI/CD | GitHub Actions + HF Spaces | Auto-deploy on main merge |
| Evaluation | Ragas | Measures RAG quality |

## Quick Start

### 1. Clone and set up environment
```bash
git clone https://github.com/YOUR_USERNAME/multimodal-finance-assistant
cd multimodal-finance-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get free key at: https://console.groq.com
```

### 3. Generate sample documents
```bash
python data/generate_samples.py
# Creates 20 invoices + 10 credit card statements in data/sample_docs/
```

### 4. Run the app
```bash
streamlit run app/main.py
# Opens at http://localhost:8501
```

### 5. Run tests
```bash
pytest tests/ -v --cov=app
```

## Project Structure

```
multimodal-finance-assistant/
├── app/
│   ├── main.py          # Streamlit frontend
│   ├── extractor.py     # PDF/image/JSON document parser
│   └── retriever.py     # ChromaDB + LangChain RAG chain
├── data/
│   ├── generate_samples.py   # Creates fake invoices for testing
│   └── sample_docs/          # Generated sample documents
├── tests/
│   └── test_extractor.py     # pytest test suite
├── .github/workflows/
│   └── ci-cd.yml             # GitHub Actions CI/CD
├── .env.example              # Template for environment variables
└── requirements.txt
```

## Deploy to Hugging Face Spaces

1. Create a new Space at huggingface.co → Streamlit SDK
2. Add `GROQ_API_KEY` to Space Secrets (Settings → Variables and secrets)
3. Add `HF_TOKEN` to GitHub Secrets (repo Settings → Secrets)
4. Push to main branch → GitHub Actions auto-deploys

## Evaluation (Ragas)

Run the RAG evaluation script to measure quality:
```bash
python app/evaluate.py
# Outputs: faithfulness, answer_relevancy, context_precision scores
```
