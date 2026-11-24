"""Shared pytest fixtures."""
from pathlib import Path

import pytest
from fpdf import FPDF


@pytest.fixture
def make_pdf(tmp_path):
    """Factory: make_pdf(["page 1 text", None, "page 3 text"]) -> Path to a new PDF.

    None creates a blank page (behaves like a scanned page with no text layer).
    """

    def _make(pages: list[str | None], name: str = "test.pdf") -> Path:
        pdf = FPDF()
        pdf.set_font("Helvetica", size=12)
        for text in pages:
            pdf.add_page()
            if text:
                pdf.multi_cell(0, 8, text)
        path = tmp_path / name
        pdf.output(str(path))
        return path

    return _make
