import pytest
from langchain_core.documents import Document

from app.services.vector_store import EmbeddingModelMismatchError, VectorStore


def chunk(source: str, page: int, idx: int, text: str = "some text") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "total_pages": 5, "chunk_index": idx,
                  "start_index": 0, "chunk_id": f"{source}::p{page}::c{idx}"},
    )


def test_new_store_is_empty(store):
    assert store.count() == 0
    assert store.list_sources() == []


def test_add_chunks_stores_text_and_metadata(store):
    store.add_chunks([chunk("hr.pdf", 7, 0, "Annual leave is 24 days.")], {"file_hash": "abc"})
    [doc] = store.get_chunks("hr.pdf")
    assert doc.page_content == "Annual leave is 24 days."
    assert doc.metadata["source"] == "hr.pdf"
    assert doc.metadata["page"] == 7
    assert doc.metadata["chunk_id"] == "hr.pdf::p7::c0"
    assert doc.metadata["file_hash"] == "abc"


def test_adding_same_chunk_id_twice_does_not_duplicate(store):
    store.add_chunks([chunk("a.pdf", 1, 0, "old")])
    store.add_chunks([chunk("a.pdf", 1, 0, "new")])
    assert store.count() == 1
    assert store.get_chunks("a.pdf")[0].page_content == "new"  # upsert overwrote it


def test_delete_source_removes_only_that_file(store):
    store.add_chunks([chunk("a.pdf", 1, 0), chunk("a.pdf", 2, 0), chunk("b.pdf", 1, 0)])
    assert store.delete_source("a.pdf") == 2
    assert store.list_sources() == ["b.pdf"]
    assert store.delete_source("missing.pdf") == 0


def test_get_chunks_returns_page_then_chunk_order(store):
    store.add_chunks([chunk("a.pdf", 2, 0), chunk("a.pdf", 1, 1), chunk("a.pdf", 1, 0)])
    order = [(d.metadata["page"], d.metadata["chunk_index"]) for d in store.get_chunks("a.pdf")]
    assert order == [(1, 0), (1, 1), (2, 0)]


def test_index_info(store):
    assert store.get_index_info("a.pdf") is None
    store.add_chunks([chunk("a.pdf", 1, 0)], {"file_hash": "h1", "chunk_size": 800, "chunk_overlap": 120})
    assert store.get_index_info("a.pdf") == {"file_hash": "h1", "chunk_size": 800, "chunk_overlap": 120}


def test_large_batches_are_split(store, fake_embeddings, monkeypatch):
    monkeypatch.setattr("app.services.vector_store.BATCH_SIZE", 10)
    store.add_chunks([chunk("big.pdf", 1, i) for i in range(25)])
    assert store.count() == 25
    assert fake_embeddings.texts_embedded == 25


def test_data_survives_reopening(tmp_path, fake_embeddings):
    path = tmp_path / "chroma"
    VectorStore(path, fake_embeddings, "fake-model").add_chunks([chunk("a.pdf", 1, 0)])
    reopened = VectorStore(path, fake_embeddings, "fake-model")
    assert reopened.count() == 1


def test_different_embedding_model_is_refused(tmp_path, fake_embeddings):
    path = tmp_path / "chroma"
    VectorStore(path, fake_embeddings, "model-A").add_chunks([chunk("a.pdf", 1, 0)])
    with pytest.raises(EmbeddingModelMismatchError, match="model-A"):
        VectorStore(path, fake_embeddings, "model-B")
