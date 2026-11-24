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
scripts/          helper scripts (e.g. make_sample_pdf.py)
samples/          small fictional PDF for demos and tests
tests/             pytest tests
eval/             labelled questions + retrieval evaluation (see eval/README.md)
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

| Method | Path | What it does |
|---|---|---|
| `POST` | `/documents` | Upload a PDF (multipart form field `file`); validates, saves to `data/uploads`, indexes it |
| `GET` | `/documents` | List indexed files and their chunk counts |
| `POST` | `/ask` | `{"question": "..."}` -> answer, citations, source chunks, model used |
| `GET` | `/health` | Liveness check |

Status codes: `200` OK · `400` not a valid PDF · `413` file too large · `422` invalid input ·
`429` LLM rate limit (see `Retry-After`) · `502` LLM rejected our config (e.g. bad key) ·
`503` LLM temporarily unavailable.

```bash
curl -X POST http://127.0.0.1:8000/documents -F "file=@samples/company_handbook.pdf"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json"      -d '{"question": "How many holidays do I get?"}'
```

## Preview PDF extraction

See exactly what text is extracted from each page of a PDF:

```bash
python -m app.services.pdf_loader samples/company_handbook.pdf
```

Pages with no text (usually scanned images) are reported as warnings; OCR is not supported yet.

## Preview chunking

See how a PDF is split into chunks (defaults come from `CHUNK_SIZE` / `CHUNK_OVERLAP` in `.env`):

```bash
python -m app.services.chunker samples/company_handbook.pdf
python -m app.services.chunker samples/company_handbook.pdf --size 200 --overlap 40
```

## Index PDFs into the vector store

```bash
python -m app.services.ingestion samples/company_handbook.pdf
```

Vectors are saved in `data/chroma`. Re-running on an unchanged file is skipped;
a changed file with the same name replaces its old chunks.
The first run downloads the embedding model (~90 MB) once; after that it loads offline.

## Search the indexed documents

```bash
python -m app.services.retrieval "How many holidays do I get?"
python -m app.services.retrieval "Can I work from home?" -k 2 --min-score 0
```

Returns the top `TOP_K` chunks with file, page and similarity score. Chunks scoring
below `MIN_SCORE` are dropped, so off-topic questions return nothing.

## Ask a question

```bash
python -m app.services.generation "How many holidays do I get?"
```

Needs `GOOGLE_API_KEY` and `LLM_PROVIDER=gemini` in `.env`. Answers cite `[file, p.N]`;
citations not matching a retrieved chunk are removed, and answers with no valid citation
are replaced by "I couldn't find this in the documents." Off-topic questions never call the LLM.

## Evaluate retrieval

```bash
python -m eval.run_eval
```

Compares chunk sizes and `MIN_SCORE` values on a labelled question set (Hit Rate@k, MRR,
off-topic rejection). Retrieval only, so no LLM calls and no cost. Results and findings: [eval/README.md](eval/README.md).

## Test

```bash
pytest -v
```

## Roadmap

- [x] Step 1 – Architecture and project skeleton
- [x] Step 2 – Configuration from `.env` (pydantic-settings)
- [x] Step 3 – PDF loading (text + page numbers)
- [x] Step 4 – Chunking (RecursiveCharacterTextSplitter)
- [x] Step 5 – Embeddings + ChromaDB storage (local all-MiniLM-L6-v2)
- [x] Step 6 – Retrieval (vector search + score threshold)
- [x] Step 7 – Answer generation with citations (Gemini, citation validation, fallback model)
- [x] Step 8 – Upload and Ask API endpoints
- [x] Step 9 – Evaluation (Hit Rate@k, MRR, off-topic rejection)
- [ ] Step 10 – Dockerfile
