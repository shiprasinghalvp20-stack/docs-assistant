"""Split page Documents into smaller, overlapping chunks for embedding and search.

Try it:
    python -m app.services.chunker samples/company_handbook.pdf --size 200 --overlap 40
"""
import argparse
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Tried in order: paragraph -> sentence -> line -> word -> character (last resort).
# Sentence comes BEFORE line on purpose: in PDF text a single "\n" is usually just a
# visual line wrap in the middle of a sentence, not a real break in meaning.
SEPARATORS = ["\n\n", ". ", "\n", " ", ""]


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split each page into chunks. Every chunk keeps its page's metadata, plus:

        chunk_index  position of the chunk within its page (0, 1, 2...)
        start_index  character offset where the chunk starts in the page text
        chunk_id     stable unique id, e.g. "handbook.pdf::p4::c0" (used as the DB id later)

    Pages are split separately, so a chunk never spans two pages -> one exact page per citation.
    chunk_size / chunk_overlap default to CHUNK_SIZE / CHUNK_OVERLAP from settings.
    """
    settings = get_settings()
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"Need chunk_size > 0 and 0 <= chunk_overlap < chunk_size "
            f"(got size={chunk_size}, overlap={chunk_overlap})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",   # keep the "." with its sentence, not at the start of the next chunk
        add_start_index=True,
    )

    chunks: list[Document] = []
    for page_doc in documents:
        for i, chunk in enumerate(splitter.split_documents([page_doc])):
            meta = chunk.metadata
            meta["chunk_index"] = i
            meta["chunk_id"] = f"{meta['source']}::p{meta['page']}::c{i}"
            chunks.append(chunk)

    logger.info(
        "Split %d page(s) into %d chunk(s) (size=%d, overlap=%d).",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def _main() -> None:
    """Command-line preview: load a PDF, chunk it, and print every chunk."""
    from app.services.pdf_loader import PDFLoadError, load_pdf

    parser = argparse.ArgumentParser(description="Preview how a PDF is chunked.")
    parser.add_argument("pdf", help="path to a PDF file")
    parser.add_argument("--size", type=int, help="chunk size in characters (default: CHUNK_SIZE)")
    parser.add_argument("--overlap", type=int, help="chunk overlap (default: CHUNK_OVERLAP)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        chunks = chunk_documents(load_pdf(args.pdf), args.size, args.overlap)
    except (PDFLoadError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    for c in chunks:
        m = c.metadata
        print(f"\n[{m['chunk_id']}]  page {m['page']} | start {m['start_index']} "
              f"| {len(c.page_content)} chars")
        print(c.page_content)


if __name__ == "__main__":
    _main()
