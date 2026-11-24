"""Safe handling of user-uploaded files.

Never trust anything the client sends: not the file name, not the Content-Type
header, not the claimed size. We check the actual bytes and enforce our own limits.
"""
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import BinaryIO

PDF_MAGIC = b"%PDF-"          # every real PDF starts with these bytes
COPY_BLOCK = 1024 * 1024      # copy 1 MB at a time
MAX_NAME_LENGTH = 100
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}


class InvalidUploadError(Exception):
    """The upload is not an acceptable PDF (wrong type, empty, ...)."""


class UploadTooLargeError(Exception):
    """The upload exceeds the size limit."""


def safe_filename(name: str | None) -> str:
    """Turn an untrusted client file name into a safe one like 'my_report.pdf'.

    - drops any folder part (blocks "../../app/main.py" path traversal, / and \\ alike)
    - keeps only letters, digits, '.', '_' and '-' (accents become plain letters)
    - no leading dots (hidden files), no Windows reserved names, limited length
    """
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # Check the ending directly: os.path.splitext("....pdf") finds no extension at all.
    if name.lower().endswith(".pdf"):
        stem, ext = name[: -len(".pdf")], ".pdf"
    else:
        stem, ext = os.path.splitext(name)[0], ""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    stem = re.sub(r"_{2,}", "_", stem)[:MAX_NAME_LENGTH] or "document"
    if stem.split(".")[0].upper() in WINDOWS_RESERVED:
        stem = f"_{stem}"
    return stem + ext


def save_pdf_upload(source: BinaryIO, original_name: str | None, upload_dir: Path, max_bytes: int) -> Path:
    """Validate and save an uploaded PDF into upload_dir. Returns the saved path.

    The file is streamed to a temporary name first and only renamed into place once
    it passed every check, so a rejected upload never replaces a good file.
    """
    filename = safe_filename(original_name)
    if not filename.endswith(".pdf"):
        raise InvalidUploadError("Only PDF files are accepted (the file name must end in .pdf).")

    upload_dir.mkdir(parents=True, exist_ok=True)
    final_path = upload_dir / filename
    temp_path = upload_dir / f".{uuid.uuid4().hex}.part"
    size = 0
    try:
        with open(temp_path, "wb") as out:
            first = source.read(len(PDF_MAGIC))
            if not first:
                raise InvalidUploadError("The uploaded file is empty.")
            if first != PDF_MAGIC:
                raise InvalidUploadError("This file is not a real PDF (its content doesn't start with %PDF-).")
            out.write(first)
            size = len(first)
            while block := source.read(COPY_BLOCK):
                size += len(block)
                if size > max_bytes:
                    raise UploadTooLargeError(
                        f"File is larger than the {max_bytes / (1024 * 1024):g} MB limit."
                    )
                out.write(block)
        os.replace(temp_path, final_path)  # atomic: either the old file or the new one
        return final_path
    finally:
        temp_path.unlink(missing_ok=True)
