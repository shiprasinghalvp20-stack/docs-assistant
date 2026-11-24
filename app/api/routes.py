"""HTTP endpoints. Thin layer: validate input, call services, map errors to status codes.

Endpoints are plain `def` (not `async def`) on purpose: PDF parsing, embedding and the
LLM call are blocking work, and FastAPI runs `def` endpoints in a thread pool so one
slow request doesn't freeze the server.
"""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.schemas import (
    AskRequest, AskResponse, CitationOut, DocumentInfo, DocumentsResponse,
    ErrorResponse, SourceOut, UploadResponse,
)
from app.core.config import get_settings
from app.services.generation import answer_question
from app.services.ingestion import IngestResult, get_vector_store, ingest_pdf
from app.services.llm import LLMError, LLMRateLimitError, LLMUnavailableError
from app.services.pdf_loader import PDFLoadError
from app.services.uploads import InvalidUploadError, UploadTooLargeError, save_pdf_upload
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------- dependencies (tests replace these with fakes) ----------

def get_store() -> VectorStore:
    return get_vector_store()


def get_llms() -> list | None:
    """None = use the configured Gemini models. Tests override this with a fake LLM."""
    return None


# ---------- documents ----------

UPLOAD_MESSAGES = {
    "indexed": "Indexed {chunks} chunks from {pages} pages.",
    "replaced": "File changed: replaced old chunks with {chunks} new chunks from {pages} pages.",
    "unchanged": "File already indexed and unchanged - nothing to do ({chunks} chunks).",
    "no_text": "No extractable text found (scanned PDF?). Nothing was indexed.",
}


@router.post(
    "/documents",
    response_model=UploadResponse,
    tags=["documents"],
    summary="Upload and index a PDF",
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
def upload_document(
    file: UploadFile = File(description="A PDF file"),
    store: VectorStore = Depends(get_store),
) -> UploadResponse:
    settings = get_settings()
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    # Cheap early reject when the client tells us the size; the streaming check still applies.
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                            f"File is larger than the {settings.max_upload_mb:g} MB limit.")
    try:
        path = save_pdf_upload(file.file, file.filename, settings.upload_dir, max_bytes)
    except UploadTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except InvalidUploadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    try:
        result: IngestResult = ingest_pdf(path, store)
    except PDFLoadError as exc:
        path.unlink(missing_ok=True)  # don't keep files we couldn't read
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    if result.status == "no_text":
        path.unlink(missing_ok=True)
    logger.info("Upload %s: %s (%d chunks)", result.source, result.status, result.chunks)
    return UploadResponse(
        filename=result.source,
        status=result.status,
        pages_with_text=result.pages_with_text,
        chunks=result.chunks,
        message=UPLOAD_MESSAGES[result.status].format(chunks=result.chunks, pages=result.pages_with_text),
    )


@router.get("/documents", response_model=DocumentsResponse, tags=["documents"], summary="List indexed files")
def list_documents(store: VectorStore = Depends(get_store)) -> DocumentsResponse:
    counts = store.source_counts()
    return DocumentsResponse(
        documents=[DocumentInfo(filename=name, chunks=n) for name, n in counts.items()],
        total_chunks=sum(counts.values()),
    )


# ---------- questions ----------

@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["questions"],
    summary="Ask a question about the indexed documents",
    responses={
        429: {"model": ErrorResponse, "description": "LLM rate limit reached"},
        502: {"model": ErrorResponse, "description": "LLM provider rejected the request (e.g. bad API key)"},
        503: {"model": ErrorResponse, "description": "LLM temporarily unavailable"},
    },
)
def ask(
    body: AskRequest,
    store: VectorStore = Depends(get_store),
    llms: list | None = Depends(get_llms),
) -> AskResponse:
    try:
        result = answer_question(body.question, k=body.k, store=store, llms=llms)
    except LLMRateLimitError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc), headers={"Retry-After": "60"}) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except ValueError as exc:  # e.g. missing API key in settings -> server misconfiguration
        logger.error("Configuration problem: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        found=result.found,
        citations=[CitationOut(source=c.source, page=c.page) for c in result.citations],
        sources=[SourceOut(source=s.source, page=s.page, score=s.score, text=s.text) for s in result.sources],
        model=result.model,
    )
