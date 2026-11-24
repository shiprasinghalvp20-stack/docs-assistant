import io

import pytest

from app.services.uploads import InvalidUploadError, UploadTooLargeError, safe_filename, save_pdf_upload

PDF_BYTES = b"%PDF-1.4\n" + b"x" * 100


@pytest.mark.parametrize("raw, safe", [
    ("report.pdf", "report.pdf"),
    ("My Report (final).PDF", "My_Report_final.pdf"),
    ("../../app/main.pdf", "main.pdf"),                 # path traversal, Unix style
    ("..\\..\\Windows\\evil.pdf", "evil.pdf"),          # path traversal, Windows style
    ("/etc/passwd.pdf", "passwd.pdf"),
    ("Résumé ñandú.pdf", "Resume_nandu.pdf"),           # accents -> plain letters
    (".hidden.pdf", "hidden.pdf"),                      # no hidden files
    ("CON.pdf", "_CON.pdf"),                            # Windows reserved name
    ("<script>.pdf", "script.pdf"),
    ("....pdf", "document.pdf"),
    ("", "document"),
    (None, "document"),
    ("notes.txt", "notes"),                             # not a pdf -> no .pdf extension
])
def test_safe_filename(raw, safe):
    assert safe_filename(raw) == safe


def test_safe_filename_limits_length():
    assert len(safe_filename("a" * 500 + ".pdf")) == 100 + len(".pdf")


def test_valid_pdf_is_saved_inside_upload_dir(tmp_path):
    path = save_pdf_upload(io.BytesIO(PDF_BYTES), "../../escape.pdf", tmp_path, max_bytes=10_000)
    assert path == tmp_path / "escape.pdf"
    assert path.read_bytes() == PDF_BYTES
    assert list(tmp_path.iterdir()) == [path]          # no leftover temp files


def test_non_pdf_extension_rejected(tmp_path):
    with pytest.raises(InvalidUploadError, match="Only PDF"):
        save_pdf_upload(io.BytesIO(PDF_BYTES), "virus.exe", tmp_path, 10_000)


def test_fake_pdf_content_rejected(tmp_path):
    with pytest.raises(InvalidUploadError, match="not a real PDF"):
        save_pdf_upload(io.BytesIO(b"MZ\x90\x00 windows exe"), "virus.pdf", tmp_path, 10_000)
    assert list(tmp_path.iterdir()) == []


def test_empty_file_rejected(tmp_path):
    with pytest.raises(InvalidUploadError, match="empty"):
        save_pdf_upload(io.BytesIO(b""), "empty.pdf", tmp_path, 10_000)


def test_too_large_rejected_and_nothing_left_behind(tmp_path):
    big = io.BytesIO(b"%PDF-" + b"x" * (3 * 1024 * 1024))
    with pytest.raises(UploadTooLargeError, match="limit"):
        save_pdf_upload(big, "big.pdf", tmp_path, max_bytes=1024 * 1024)
    assert list(tmp_path.iterdir()) == []


def test_rejected_upload_does_not_replace_existing_good_file(tmp_path):
    good = save_pdf_upload(io.BytesIO(PDF_BYTES), "doc.pdf", tmp_path, 10_000)
    with pytest.raises(InvalidUploadError):
        save_pdf_upload(io.BytesIO(b"garbage"), "doc.pdf", tmp_path, 10_000)
    assert good.read_bytes() == PDF_BYTES
