"""Load a PDF into LangChain Documents: one Document per page, with file + page metadata.

Try it on any PDF:
    python -m app.services.pdf_loader path/to/file.pdf
"""
import logging
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be read (missing, corrupt, encrypted, not a PDF)."""


def clean_text(text: str) -> str:
    """Tidy raw PDF text without changing its meaning."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # "infor-\nmation" -> "information"
    text = re.sub(r"[ \t]+", " ", text)              # many spaces/tabs -> one space
    text = re.sub(r" *\n *", "\n", text)             # trim spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)           # max one blank line in a row
    return text.strip()


def load_pdf(path: str | Path) -> list[Document]:
    """Extract text from every page of a PDF.

    Returns one Document per page that has text. Each Document's metadata:
        source       file name only (e.g. "handbook.pdf") - not the full local path
        page         human page number, starting at 1 (pypdf counts from 0)
        total_pages  number of pages in the file

    Pages with no extractable text (usually scanned images) are skipped with a warning.
    Raises PDFLoadError for missing, corrupt, encrypted or non-PDF files.
    """
    path = Path(path)
    if not path.is_file():
        raise PDFLoadError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Not a PDF file: {path.name}")

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise PDFLoadError(f"PDF is password-protected: {path.name}")
        pages = reader.pages
        total_pages = len(pages)
    except PdfReadError as exc:
        raise PDFLoadError(f"Could not read PDF (corrupt or invalid): {path.name}") from exc

    documents: list[Document] = []
    empty_pages: list[int] = []

    for index, page in enumerate(pages):
        page_number = index + 1
        text = clean_text(page.extract_text() or "")
        if not text:
            empty_pages.append(page_number)
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_number, "total_pages": total_pages},
            )
        )

    if empty_pages:
        logger.warning(
            "%s: no text on page(s) %s - probably scanned images (OCR not supported yet).",
            path.name,
            empty_pages,
        )
    logger.info("Loaded %s: %d of %d pages have text.", path.name, len(documents), total_pages)
    return documents


def _main() -> None:
    """Command-line preview: show what was extracted from each page."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) != 2:
        print("Usage: python -m app.services.pdf_loader path/to/file.pdf")
        sys.exit(1)
    try:
        docs = load_pdf(sys.argv[1])
    except PDFLoadError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    for doc in docs:
        m = doc.metadata
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"\n--- {m['source']} | page {m['page']}/{m['total_pages']} "
              f"| {len(doc.page_content)} chars ---")
        print(preview + ("..." if len(doc.page_content) > 200 else ""))


if __name__ == "__main__":
    _main()
