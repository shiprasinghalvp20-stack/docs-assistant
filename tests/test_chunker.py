from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.chunker import chunk_documents
from app.services.pdf_loader import load_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "company_handbook.pdf"

SENTENCES = " ".join(f"This is sentence number {i} about the leave policy." for i in range(30))
WORDS = " ".join(f"word{i}" for i in range(200))  # no full stops -> forces word-level splits


def page(text: str, source: str = "doc.pdf", page_no: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page_no, "total_pages": 9})


# ---------- metadata ----------

def test_every_chunk_keeps_source_and_page():
    chunks = chunk_documents([page(SENTENCES, "hr.pdf", 7)], chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source"] == "hr.pdf"
        assert c.metadata["page"] == 7
        assert c.metadata["total_pages"] == 9


def test_chunk_index_and_id_are_added():
    chunks = chunk_documents([page(SENTENCES, "hr.pdf", 7)], chunk_size=200, chunk_overlap=40)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0].metadata["chunk_id"] == "hr.pdf::p7::c0"
    assert chunks[1].metadata["chunk_id"] == "hr.pdf::p7::c1"


def test_chunk_ids_are_unique_across_pages():
    docs = [page(SENTENCES, "a.pdf", 1), page(SENTENCES, "a.pdf", 2), page(SENTENCES, "b.pdf", 1)]
    ids = [c.metadata["chunk_id"] for c in chunk_documents(docs, 200, 40)]
    assert len(ids) == len(set(ids))


def test_start_index_points_to_chunk_text_in_page():
    """start_index lets us find each chunk inside the original page (useful for highlighting)."""
    chunks = chunk_documents([page(SENTENCES)], chunk_size=200, chunk_overlap=40)
    for c in chunks:
        start = c.metadata["start_index"]
        assert SENTENCES[start : start + len(c.page_content)] == c.page_content


def test_input_documents_are_not_modified():
    original = page(SENTENCES)
    chunk_documents([original], chunk_size=200, chunk_overlap=40)
    assert original.metadata == {"source": "doc.pdf", "page": 1, "total_pages": 9}


# ---------- size and boundaries ----------

@pytest.mark.parametrize("size, overlap", [(100, 0), (200, 40), (500, 100)])
def test_no_chunk_exceeds_chunk_size(size, overlap):
    for c in chunk_documents([page(SENTENCES)], chunk_size=size, chunk_overlap=overlap):
        assert len(c.page_content) <= size


def test_short_page_stays_one_chunk():
    chunks = chunk_documents([page("Short page.")], chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short page."


def test_chunks_end_at_sentence_boundaries_when_possible():
    for c in chunk_documents([page(SENTENCES)], chunk_size=200, chunk_overlap=0):
        assert c.page_content.endswith(".")


def test_line_wraps_inside_a_sentence_are_not_preferred_cut_points():
    """PDF text has '\\n' at visual line wraps; we must not cut there before sentence ends."""
    text = ("First sentence starts here and\ncontinues on the next line. " * 6).strip()
    for c in chunk_documents([page(text)], chunk_size=130, chunk_overlap=0):
        assert c.page_content.endswith(".")


def test_chunks_never_span_two_pages():
    docs = [page("Page one text. " * 20, page_no=1), page("Page two text. " * 20, page_no=2)]
    for c in chunk_documents(docs, chunk_size=100, chunk_overlap=20):
        expected = "Page one" if c.metadata["page"] == 1 else "Page two"
        other = "Page two" if c.metadata["page"] == 1 else "Page one"
        assert expected in c.page_content and other not in c.page_content


# ---------- overlap ----------

def test_overlap_repeats_text_between_neighbouring_chunks():
    chunks = chunk_documents([page(WORDS)], chunk_size=100, chunk_overlap=30)
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_end = prev.metadata["start_index"] + len(prev.page_content)
        assert nxt.metadata["start_index"] < prev_end  # next chunk starts BEFORE previous ends


def test_zero_overlap_means_no_repeated_text():
    chunks = chunk_documents([page(WORDS)], chunk_size=100, chunk_overlap=0)
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_end = prev.metadata["start_index"] + len(prev.page_content)
        assert nxt.metadata["start_index"] >= prev_end


# ---------- settings and errors ----------

def test_defaults_come_from_settings(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    get_settings.cache_clear()
    try:
        chunks = chunk_documents([page(SENTENCES)])
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 100 for c in chunks)
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_size_or_overlap_raises(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([page("text")], chunk_size=size, chunk_overlap=overlap)


def test_empty_input_gives_empty_output():
    assert chunk_documents([], chunk_size=100, chunk_overlap=10) == []


# ---------- the committed sample ----------

def test_sample_handbook_chunks_keep_correct_pages():
    chunks = chunk_documents(load_pdf(SAMPLE_PDF), chunk_size=200, chunk_overlap=40)
    by_page = {}
    for c in chunks:
        by_page.setdefault(c.metadata["page"], []).append(c.page_content)

    assert sorted(by_page) == [1, 2, 4]
    assert any("medical certificate" in t for t in by_page[1])
    assert any("VPN" in t for t in by_page[2])
    assert any("25,000 require approval" in t for t in by_page[4])
