import shutil
from pathlib import Path

import pytest

from app.services.ingestion import file_sha256, ingest_pdf
from app.services.pdf_loader import PDFLoadError

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "company_handbook.pdf"


def test_sample_handbook_is_indexed(store):
    result = ingest_pdf(SAMPLE_PDF, store, chunk_size=200, chunk_overlap=40)

    assert result.status == "indexed"
    assert result.pages_with_text == 3
    assert result.chunks == 6
    assert store.count() == 6
    assert store.list_sources() == ["company_handbook.pdf"]
    assert {d.metadata["page"] for d in store.get_chunks("company_handbook.pdf")} == {1, 2, 4}


def test_reingesting_unchanged_file_is_skipped(store, fake_embeddings):
    ingest_pdf(SAMPLE_PDF, store, 200, 40)
    embedded_before = fake_embeddings.texts_embedded

    result = ingest_pdf(SAMPLE_PDF, store, 200, 40)

    assert result.status == "unchanged"
    assert result.chunks == 6
    assert store.count() == 6                                   # no duplicates
    assert fake_embeddings.texts_embedded == embedded_before    # no embedding work


def test_changed_file_replaces_old_chunks(store, make_pdf):
    path = make_pdf(["Page one. " * 30, "Page two. " * 30, "Page three. " * 30], name="policy.pdf")
    first = ingest_pdf(path, store, 100, 0)

    # Same file name, new (shorter) content -> fewer chunks; old leftovers must be gone.
    path = make_pdf(["Only one short page."], name="policy.pdf")
    second = ingest_pdf(path, store, 100, 0)

    assert second.status == "replaced"
    assert second.chunks < first.chunks
    assert store.count() == second.chunks
    assert {d.metadata["page"] for d in store.get_chunks("policy.pdf")} == {1}


def test_changed_chunk_settings_trigger_reindex(store):
    ingest_pdf(SAMPLE_PDF, store, 800, 120)
    result = ingest_pdf(SAMPLE_PDF, store, 200, 40)
    assert result.status == "replaced"
    assert store.count() == 6


def test_identical_copy_under_new_name_is_a_separate_file(store, tmp_path):
    copy = tmp_path / "copy.pdf"
    shutil.copy(SAMPLE_PDF, copy)
    ingest_pdf(SAMPLE_PDF, store, 200, 40)
    ingest_pdf(copy, store, 200, 40)
    assert store.list_sources() == ["company_handbook.pdf", "copy.pdf"]
    assert store.count() == 12


def test_pdf_without_text_stores_nothing(store, make_pdf):
    result = ingest_pdf(make_pdf([None, None], name="scan.pdf"), store, 200, 40)
    assert result.status == "no_text"
    assert store.count() == 0


def test_unreadable_pdf_raises_and_stores_nothing(store, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(PDFLoadError):
        ingest_pdf(bad, store)
    assert store.count() == 0


def test_file_hash_is_stable_and_content_based(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    assert file_sha256(a) == file_sha256(b)
    b.write_bytes(b"different")
    assert file_sha256(a) != file_sha256(b)
