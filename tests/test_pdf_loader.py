import logging
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from app.services.pdf_loader import PDFLoadError, clean_text, load_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "company_handbook.pdf"


# ---------- happy path ----------

def test_one_document_per_page_with_metadata(make_pdf):
    path = make_pdf(["Leave policy text", "Remote work text"], name="handbook.pdf")
    docs = load_pdf(path)

    assert len(docs) == 2
    assert docs[0].page_content == "Leave policy text"
    assert docs[0].metadata == {"source": "handbook.pdf", "page": 1, "total_pages": 2}
    assert docs[1].metadata["page"] == 2


def test_source_is_file_name_not_full_path(make_pdf):
    docs = load_pdf(make_pdf(["hello"], name="report.pdf"))
    assert docs[0].metadata["source"] == "report.pdf"


def test_accepts_string_path(make_pdf):
    assert len(load_pdf(str(make_pdf(["hello"])))) == 1


# ---------- tricky PDFs ----------

def test_blank_page_is_skipped_but_page_numbers_stay_correct(make_pdf, caplog):
    path = make_pdf(["first page", None, "third page"])
    with caplog.at_level(logging.WARNING):
        docs = load_pdf(path)

    assert [d.metadata["page"] for d in docs] == [1, 3]  # page 3 is still "page 3"
    assert "no text on page(s) [2]" in caplog.text


def test_pdf_with_only_blank_pages_returns_empty_list(make_pdf):
    assert load_pdf(make_pdf([None, None])) == []


# ---------- errors ----------

def test_missing_file_raises(tmp_path):
    with pytest.raises(PDFLoadError, match="File not found"):
        load_pdf(tmp_path / "nope.pdf")


def test_non_pdf_extension_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(PDFLoadError, match="Not a PDF"):
        load_pdf(path)


def test_corrupt_pdf_raises(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not really a pdf")
    with pytest.raises(PDFLoadError, match="corrupt or invalid"):
        load_pdf(path)


def test_password_protected_pdf_raises(make_pdf, tmp_path):
    writer = PdfWriter(clone_from=PdfReader(make_pdf(["secret"])))
    writer.encrypt("pass123")
    locked = tmp_path / "locked.pdf"
    with open(locked, "wb") as f:
        writer.write(f)

    with pytest.raises(PDFLoadError, match="password-protected"):
        load_pdf(locked)


# ---------- text cleaning ----------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("infor-\nmation", "information"),          # word split across lines
        ("too    many \t spaces", "too many spaces"),
        ("line one   \n   line two", "line one\nline two"),
        ("para one\n\n\n\n\npara two", "para one\n\npara two"),
        ("   padded   ", "padded"),
        ("well-known fact", "well-known fact"),     # real hyphens are kept
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


# ---------- the committed sample ----------

def test_sample_handbook_loads_with_correct_pages():
    docs = load_pdf(SAMPLE_PDF)
    pages = {d.metadata["page"]: d.page_content for d in docs}

    assert sorted(pages) == [1, 2, 4]          # page 3 is blank by design
    assert "24 days of paid annual leave" in pages[1]
    assert "3 days per week" in pages[2]
    assert "within 30 days" in pages[4]
