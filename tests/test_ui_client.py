"""Tests for the UI's API client. httpx.MockTransport fakes the API - no server, no network."""
import json

import httpx
import pytest

from ui.api_client import (
    ApiUnreachable, DocsApiClient, InvalidRequest, LLMUnavailable, RateLimited, ServerError,
    citation_label, escape_markdown, get_api_url,
)

ASK_OK = {
    "question": "How many holidays do I get?",
    "answer": "You get 24 days of paid annual leave [handbook.pdf, p.1].",
    "found": True,
    "citations": [{"source": "handbook.pdf", "page": 1}],
    "sources": [{"source": "handbook.pdf", "page": 1, "score": 0.375, "text": "24 days of paid annual leave"}],
    "model": "gemini-x",
}


def client_with(handler) -> DocsApiClient:
    """A client whose HTTP calls go to `handler(request) -> httpx.Response` instead of the network."""
    return DocsApiClient("http://api.test", transport=httpx.MockTransport(handler))


def reply(status: int, body=None, headers=None):
    return lambda request: httpx.Response(status, json=body, headers=headers)


# ---------- settings and helpers ----------

def test_api_url_default_and_env(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    assert get_api_url() == "http://localhost:8000"
    monkeypatch.setenv("API_URL", "http://api:8000/")
    assert get_api_url() == "http://api:8000"          # trailing slash removed


def test_citation_label_and_markdown_escape():
    assert citation_label("hr.pdf", 4) == "[hr.pdf, p.4]"
    assert escape_markdown("costs $5 or $10") == "costs \\$5 or \\$10"


# ---------- successful calls ----------

def test_health():
    assert client_with(reply(200, {"status": "ok"})).health() is True
    assert client_with(reply(500)).health() is False


def test_health_false_when_api_down():
    def down(request):
        raise httpx.ConnectError("refused")
    assert client_with(down).health() is False


def test_list_documents():
    body = {"documents": [{"filename": "a.pdf", "chunks": 6}, {"filename": "b.pdf", "chunks": 2}],
            "total_chunks": 8}
    assert client_with(reply(200, body)).list_documents() == [("a.pdf", 6), ("b.pdf", 2)]


def test_upload_sends_multipart_file_and_parses_result():
    seen = {}

    def handler(request: httpx.Request):
        seen["method"], seen["path"] = request.method, request.url.path
        seen["body"] = request.read()
        return httpx.Response(200, json={"filename": "hr.pdf", "status": "indexed", "pages_with_text": 3,
                                         "chunks": 6, "message": "Indexed 6 chunks from 3 pages."})

    result = client_with(handler).upload("hr.pdf", b"%PDF-1.4 fake")
    assert (seen["method"], seen["path"]) == ("POST", "/documents")
    assert b'name="file"; filename="hr.pdf"' in seen["body"]
    assert b"%PDF-1.4 fake" in seen["body"]
    assert (result.filename, result.status, result.chunks) == ("hr.pdf", "indexed", 6)


def test_ask_sends_question_and_parses_answer():
    seen = {}

    def handler(request: httpx.Request):
        seen["path"], seen["json"] = request.url.path, json.loads(request.read())
        return httpx.Response(200, json=ASK_OK)

    result = client_with(handler).ask("How many holidays do I get?")
    assert seen == {"path": "/ask", "json": {"question": "How many holidays do I get?"}}
    assert result.found is True
    assert result.citations == [("handbook.pdf", 1)]
    assert result.sources[0].page == 1 and result.sources[0].score == 0.375
    assert result.model == "gemini-x"


def test_ask_not_found():
    body = {**ASK_OK, "answer": "I couldn't find this in the documents.", "found": False,
            "citations": [], "sources": [], "model": None}
    result = client_with(reply(200, body)).ask("cricket?")
    assert (result.found, result.citations, result.sources, result.model) == (False, [], [], None)


# ---------- errors become friendly messages ----------

def test_rate_limit_uses_retry_after():
    with pytest.raises(RateLimited) as err:
        client_with(reply(429, {"detail": "Rate limit"}, {"Retry-After": "60"})).ask("q")
    assert err.value.retry_after == 60
    assert "60 seconds" in err.value.message


def test_rate_limit_without_header():
    with pytest.raises(RateLimited, match="a minute"):
        client_with(reply(429, {"detail": "Rate limit"})).ask("q")


def test_llm_unavailable_503():
    with pytest.raises(LLMUnavailable, match="overloaded"):
        client_with(reply(503, {"detail": "unavailable"})).ask("q")


@pytest.mark.parametrize("status, body, message", [
    (400, {"detail": "This file is not a real PDF."}, "This file is not a real PDF."),
    (413, {"detail": "File is larger than the 20 MB limit."}, "File is larger than the 20 MB limit."),
    (422, {"detail": [{"loc": ["body", "question"], "msg": "String should have at least 1 character"}]},
     "String should have at least 1 character"),
])
def test_rejected_input_shows_api_detail(status, body, message):
    with pytest.raises(InvalidRequest) as err:
        client_with(reply(status, body)).upload("x.pdf", b"x")
    assert err.value.message == message


def test_502_explains_server_side_problem():
    with pytest.raises(ServerError, match="rejected the API key"):
        client_with(reply(502, {"detail": "The LLM provider rejected the API key."})).ask("q")


def test_other_server_error_without_json():
    def handler(request):
        return httpx.Response(500, text="Internal Server Error")
    with pytest.raises(ServerError, match="HTTP 500"):
        client_with(handler).ask("q")


def test_api_down_is_reported_with_url():
    def down(request):
        raise httpx.ConnectError("connection refused")
    with pytest.raises(ApiUnreachable, match="http://api.test"):
        client_with(down).list_documents()


def test_timeout_is_reported_as_busy():
    def slow(request):
        raise httpx.ReadTimeout("timed out")
    with pytest.raises(LLMUnavailable, match="took too long"):
        client_with(slow).ask("q")
