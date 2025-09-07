# RAG: Document-based Chatbot (FAISS + Sentence-Transformers + FastAPI)

A minimal Retrieval-Augmented Generation (RAG) stack that:
- Chunks PDFs/TXT/MD docs
- Embeds with `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- Indexes with **FAISS**
- Serves a **FastAPI** endpoint `/ask` that retrieves top-k chunks and prompts a local LLM or OpenAI-compatible endpoint (pluggable).

> Out of the box, the answer is formed by **extractive synthesis** (concise summary from retrieved chunks) without calling an LLM. You can plug an LLM later via the `llm.py` adapter.

## Layout
```
.
├── data/                   # place your docs here
├── src/
│   ├── ingest.py           # build/update FAISS index
│   ├── retriever.py        # search
│   ├── llm.py              # optional LLM adapter
│   └── api.py              # FastAPI app
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ingest documents
python src/ingest.py --docs_dir data --index_path outputs/index.faiss --db_path outputs/meta.jsonl

# Run API
uvicorn src.api:app --reload --port 8000

# Ask
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"query":"What is the policy?"}'
```
