"""API tests: real FastAPI app, fake embeddings, fake LLM, temp folders. No network."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_llms, get_store
from app.core.config import get_settings
from app.main import app
from app.services.generation import NOT_FOUND_MESSAGE
from app.services.vector_store import VectorStore
from tests.fakes import BagOfWordsEmbeddings, FakeAPIError, FakeLLM

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "company_handbook.pdf"


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Point uploads at a temp folder and pin settings, so the real .env can't affect tests."""
    folder = tmp_path / "uploads"
    monkeypatch.setenv("UPLOAD_DIR", str(folder))
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "40")
    monkeypatch.setenv("MIN_SCORE", "0.25")
    monkeypatch.setenv("TOP_K", "4")
    monkeypatch.setenv("MAX_UPLOAD_MB", "20")
    get_settings.cache_clear()
    yield folder
    get_settings.cache_clear()


@pytest.fixture
def llm():
    return FakeLLM("You get 24 days of paid annual leave [company_handbook.pdf, p.1].")


@pytest.fixture
def client(tmp_path, upload_dir, llm):
    store = VectorStore(tmp_path / "chroma", BagOfWordsEmbeddings(), "bow-model")
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_llms] = lambda: [("fake-model", llm)]
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, name="company_handbook.pdf", content=None, mime="application/pdf"):
    data = content if content is not None else SAMPLE_PDF.read_bytes()
    return client.post("/documents", files={"file": (name, data, mime)})


# ---------- POST /documents ----------

def test_upload_indexes_pdf(client, upload_dir):
    r = upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "company_handbook.pdf"
    assert body["status"] == "indexed"
    assert body["pages_with_text"] == 3
    assert body["chunks"] == 6
    assert (upload_dir / "company_handbook.pdf").exists()


def test_reupload_same_file_is_unchanged(client):
    upload(client)
    r = upload(client)
    assert r.status_code == 200
    assert r.json()["status"] == "unchanged"


def test_unsafe_filename_is_cleaned(client, upload_dir):
    r = upload(client, name="../../Hand Book (v2).pdf")
    assert r.json()["filename"] == "Hand_Book_v2.pdf"
    assert (upload_dir / "Hand_Book_v2.pdf").exists()


def test_non_pdf_name_rejected_with_400(client):
    r = upload(client, name="notes.txt", content=b"hello", mime="text/plain")
    assert r.status_code == 400
    assert "Only PDF" in r.json()["detail"]


def test_fake_pdf_rejected_with_400_even_with_pdf_content_type(client):
    r = upload(client, name="evil.pdf", content=b"<html>not a pdf</html>", mime="application/pdf")
    assert r.status_code == 400
    assert "not a real PDF" in r.json()["detail"]


def test_corrupt_pdf_rejected_with_400_and_not_kept(client, upload_dir):
    r = upload(client, name="broken.pdf", content=b"%PDF-1.4 this is garbage")
    assert r.status_code == 400
    assert not (upload_dir / "broken.pdf").exists()


def test_too_large_upload_rejected_with_413(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")   # ~1 KB limit
    get_settings.cache_clear()
    r = upload(client)
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]


def test_missing_file_gives_422(client):
    assert client.post("/documents").status_code == 422


def test_scanned_pdf_reports_no_text(client, make_pdf, upload_dir):
    r = upload(client, name="scan.pdf", content=make_pdf([None, None]).read_bytes())
    assert r.status_code == 200
    assert r.json()["status"] == "no_text"
    assert r.json()["chunks"] == 0
    assert not (upload_dir / "scan.pdf").exists()


# ---------- GET /documents ----------

def test_list_documents_empty_then_filled(client):
    assert client.get("/documents").json() == {"documents": [], "total_chunks": 0}
    upload(client)
    assert client.get("/documents").json() == {
        "documents": [{"filename": "company_handbook.pdf", "chunks": 6}],
        "total_chunks": 6,
    }


# ---------- POST /ask ----------

def test_ask_returns_answer_citations_sources_and_model(client, llm):
    upload(client)
    r = client.post("/ask", json={"question": "How many annual leave days do I get?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert "24 days" in body["answer"]
    assert body["citations"] == [{"source": "company_handbook.pdf", "page": 1}]
    assert body["sources"][0]["page"] == 1 and "24 days" in body["sources"][0]["text"]
    assert body["model"] == "fake-model"
    assert len(llm.calls) == 1


def test_ask_off_topic_returns_not_found_without_llm(client, llm):
    upload(client)
    r = client.post("/ask", json={"question": "Who won the cricket world cup?"})
    assert r.status_code == 200
    assert r.json() == {"question": "Who won the cricket world cup?", "answer": NOT_FOUND_MESSAGE,
                        "found": False, "citations": [], "sources": [], "model": None}
    assert llm.calls == []


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "   "},
                                  {"question": "x" * 1001}, {"question": "ok", "k": 0}])
def test_invalid_question_gives_422(client, body):
    assert client.post("/ask", json=body).status_code == 422


@pytest.mark.parametrize("error, code, phrase", [
    (FakeAPIError(429, "RESOURCE_EXHAUSTED"), 429, "Rate limit"),
    (FakeAPIError(503, "UNAVAILABLE"), 503, "temporarily unavailable"),
    (FakeAPIError(400, "API key not valid"), 502, "API key"),
])
def test_llm_errors_map_to_status_codes(client, llm, error, code, phrase):
    upload(client)
    llm.error = error
    r = client.post("/ask", json={"question": "annual leave days"})
    assert r.status_code == code
    assert phrase in r.json()["detail"]
    if code == 429:
        assert r.headers["retry-after"] == "60"


def test_openapi_documents_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/documents", "/ask", "/health"} <= set(paths)
    assert {"get", "post"} <= set(paths["/documents"])
