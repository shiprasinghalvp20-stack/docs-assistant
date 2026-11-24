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


class CountingFakeEmbeddings:
    """Stand-in for the real model: deterministic vectors, no download, no API.

    Counts how many texts were embedded so tests can prove work was (or wasn't) done.
    """

    def __init__(self, size: int = 16):
        from langchain_core.embeddings import DeterministicFakeEmbedding

        self._fake = DeterministicFakeEmbedding(size=size)
        self.texts_embedded = 0

    def embed_documents(self, texts):
        self.texts_embedded += len(texts)
        return self._fake.embed_documents(texts)

    def embed_query(self, text):
        return self._fake.embed_query(text)


@pytest.fixture
def fake_embeddings():
    return CountingFakeEmbeddings()


@pytest.fixture
def store(tmp_path, fake_embeddings):
    """A fresh, empty vector store in a temp folder, using fake embeddings."""
    from app.services.vector_store import VectorStore

    return VectorStore(tmp_path / "chroma", fake_embeddings, embedding_model="fake-model")
