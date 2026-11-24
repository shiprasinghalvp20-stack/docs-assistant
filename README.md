# Docs Assistant

Upload PDFs, ask questions in plain English, and get answers with **file and page citations**.
Built with Retrieval-Augmented Generation (RAG).

## Architecture

```
            INGESTION (once per PDF)                       QUERY (every question)
  ┌───────┐   ┌─────────┐   ┌────────┐   ┌──────────┐    ┌──────────┐   ┌──────────┐
  │  PDF  │──▶│  Load   │──▶│ Chunk  │──▶│  Embed   │    │ Question │──▶│  Embed   │
  └───────┘   │ (pages) │   │        │   │ (vectors)│    └──────────┘   └────┬─────┘
              └─────────┘   └────────┘   └────┬─────┘                        │
                                              ▼                              ▼
                                       ┌─────────────────────────────────────────┐
                                       │        ChromaDB (vector database)       │
                                       │  text + vector + {file, page} metadata  │
                                       └──────────────────┬──────────────────────┘
                                                          │ top-k similar chunks
                                                          ▼
                                       ┌─────────────────────────────────────────┐
                                       │ LLM (Groq / Gemini): answer ONLY from   │
                                       │ these chunks, cite [file, page]         │
                                       └─────────────────────────────────────────┘
```

## Tech stack

Python · FastAPI · LangChain · ChromaDB · Groq / Gemini · pytest · Docker

## Project structure

```
app/
  main.py          FastAPI app entry point
  api/             HTTP routes (thin layer)
  core/            configuration (reads .env)
  services/        ingestion, retrieval, generation logic
tests/             pytest tests
eval/              retrieval accuracy evaluation
data/uploads/      uploaded PDFs (git-ignored)
data/chroma/       vector database files (git-ignored)
```

## Setup

Requires **Python 3.11 or 3.12**.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your API keys in .env
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API.

## Test

```bash
pytest -v
```

## Roadmap

- [x] Step 1 – Architecture and project skeleton
- [ ] Step 2 – Configuration from `.env`
- [ ] Step 3 – PDF loading (text + page numbers)
- [ ] Step 4 – Chunking
- [ ] Step 5 – Embeddings + ChromaDB storage
- [ ] Step 6 – Retrieval (vector search)
- [ ] Step 7 – Answer generation with citations
- [ ] Step 8 – Upload and Ask API endpoints
- [ ] Step 9 – Evaluation (retrieval accuracy)
- [ ] Step 10 – Dockerfile
