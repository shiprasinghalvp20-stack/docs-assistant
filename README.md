# Docs Assistant

Upload PDFs, ask questions in plain English, and get answers with **file and page citations**.
Built with Retrieval-Augmented Generation (RAG). Comes with a REST API (FastAPI) and a chat UI (Streamlit).

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

Python 3.12 · FastAPI · LangChain · ChromaDB · sentence-transformers (local embeddings) · Google Gemini (answers) · Streamlit (UI) · pytest · Docker

## Project structure

```
app/
  main.py              FastAPI app entry point
  api/                 HTTP routes + Pydantic request/response models
  core/config.py       settings, read from environment / .env
  services/            pdf_loader, chunker, embeddings, vector_store, ingestion,
                       retrieval, llm, generation, uploads
  evaluation.py        retrieval metrics (Hit Rate@k, MRR, rejection)
tests/                 pytest tests (fake embeddings + fake LLM, no network)
eval/                  labelled questions + evaluation script (see eval/README.md)
ui/                    Streamlit chat UI (app.py) + API client (api_client.py) + its Dockerfile
scripts/               helper scripts (make_sample_pdf.py)
samples/               small fictional PDF for demos and tests
data/                  uploads + vector DB at runtime (git-ignored)
Dockerfile, docker-compose.yml, .dockerignore
requirements.txt       runtime dependencies (used by the Docker image)
requirements-dev.txt   runtime + test/dev tools
```

## Setup

Requires **Python 3.11 or 3.12**.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt
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

## Screenshots

<!-- Take these from the running UI (see docs/screenshots/README.md), save them in
     docs/screenshots/, then remove the comment markers around the two image lines below. -->
<!--
![Chat: answer with citation and sources](docs/screenshots/ui-chat.png)
![Sidebar: upload status and indexed files](docs/screenshots/ui-upload.png)
-->

*Screenshots coming soon. See [docs/screenshots/README.md](docs/screenshots/README.md).*

## Web UI (Streamlit)

A chat page that talks to the API over HTTP. It needs no API key and no ML libraries.

- **Sidebar:** API status, upload a PDF (with status and chunk count), list of indexed files
- **Chat:** ask questions, get answers with `[file, p.N]` citations, and expand **Sources** to see the retrieved chunks with page and similarity score
- **Friendly messages** for "not found", rate limits (429), the model being overloaded (503) and the API being down
- **Chat history** is kept while the browser tab is open

Run locally (the API must be running on port 8000):

```bash
cd ui
streamlit run app.py        # opens http://localhost:8501
```

The API address comes from the `API_URL` environment variable (default `http://localhost:8000`).

## Run with Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/). Create `.env` first
(`cp .env.example .env`) and add your `GOOGLE_API_KEY`. The key is read when the container
**starts**: it is never built into the image, and `.dockerignore` keeps `.env` out of it.

```bash
docker compose up --build -d     # build the image and start in the background
docker compose ps                # STATUS should become "healthy"
docker compose logs -f api       # follow logs (Ctrl+C to stop following)
docker compose down              # stop (uploads + vectors are kept in the docs-data volume)
docker compose down -v           # stop AND delete all uploaded data
```

This starts **two services**:

| Service | URL | What it is |
|---|---|---|
| `ui` | http://localhost:8501 | Streamlit chat UI |
| `api` | http://localhost:8000/docs | FastAPI backend (interactive API docs) |

The UI waits until the API is healthy, then reaches it at `http://api:8000`, using the Compose
service name. Uploaded PDFs and the vector database live in the named volume `docs-data`, so
they survive restarts and image upgrades. Only the `api` container receives `.env`.

Without Compose:

```bash
docker build -t docs-assistant .
docker run --env-file .env -p 8000:8000 -v docs-data:/app/data docs-assistant
```

The image is CPU-only. The embedding model is downloaded at build time, so the container
starts offline, and the app runs as a non-root user with a `/health` healthcheck.

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
- [x] Step 10 – Dockerfile + docker-compose (CPU-only, model baked in, volume, healthcheck)
- [x] Step 11 – Streamlit chat UI (second Compose service)
