"""Ingestion pipeline: PDF -> pages -> chunks -> vectors in ChromaDB.

Re-ingesting is safe:
  * same file + same chunk settings  -> skipped (no embedding work)
  * same file name, new content      -> old chunks deleted, new ones stored
  * chunk ids are deterministic       -> a chunk is never stored twice

Try it:
    python -m app.services.ingestion samples/company_handbook.pdf
"""
import argparse
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.core.config import get_settings
from app.services.chunker import chunk_documents
from app.services.pdf_loader import PDFLoadError, load_pdf
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

Status = Literal["indexed", "replaced", "unchanged", "no_text"]


@dataclass
class IngestResult:
    source: str
    status: Status
    pages_with_text: int
    chunks: int


def file_sha256(path: Path) -> str:
    """Fingerprint of the file's bytes: identical files always give the same hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(
    path: str | Path,
    store: VectorStore,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestResult:
    """Load, chunk, embed and store one PDF. Raises PDFLoadError for unreadable files."""
    path = Path(path)
    settings = get_settings()
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    pages = load_pdf(path)  # validates the file first; raises PDFLoadError if unusable
    source = path.name
    new_info = {"file_hash": file_sha256(path), "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    old_info = store.get_index_info(source)

    if old_info == new_info:
        count = store.count_source(source)
        logger.info("%s unchanged - skipped (%d chunks already stored).", source, count)
        return IngestResult(source, "unchanged", len(pages), count)

    if old_info is not None:
        removed = store.delete_source(source)
        logger.info("%s changed - removed %d old chunks.", source, removed)

    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    if not chunks:
        logger.warning("%s has no extractable text - nothing stored.", source)
        return IngestResult(source, "no_text", 0, 0)

    store.add_chunks(chunks, extra_metadata=new_info)
    status: Status = "replaced" if old_info is not None else "indexed"
    logger.info("%s %s: %d chunks from %d pages.", source, status, len(chunks), len(pages))
    return IngestResult(source, status, len(pages), len(chunks))


@lru_cache
def get_vector_store() -> VectorStore:
    """The app's shared vector store, configured from settings (created once)."""
    from app.services.embeddings import get_embeddings

    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return VectorStore(
        persist_dir=settings.chroma_dir,
        embeddings=get_embeddings(),
        embedding_model=settings.embedding_model,
        collection_name=settings.collection_name,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Index PDFs into the vector store.")
    parser.add_argument("pdfs", nargs="+", help="one or more PDF files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide model-hub HTTP request logs

    store = get_vector_store()
    for pdf in args.pdfs:
        try:
            r = ingest_pdf(pdf, store)
            print(f"{r.source}: {r.status} - {r.chunks} chunks from {r.pages_with_text} pages")
        except PDFLoadError as exc:
            print(f"Error: {exc}")

    print(f"\nTotal vectors in '{get_settings().collection_name}': {store.count()}")
    print(f"Indexed files: {', '.join(store.list_sources()) or '(none)'}")


if __name__ == "__main__":
    _main()
