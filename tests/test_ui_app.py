"""Smoke tests for the Streamlit page using Streamlit's AppTest (runs the script headlessly).

The API client's methods are replaced with fakes, so no server or network is needed.
"""
import sys
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
# Appended (not inserted first): ui/ contains app.py, which must not shadow the real `app` package.
sys.path.append(str(UI_DIR))   # ui/app.py does "from api_client import ..."

import api_client  # noqa: E402  (the same module object the app imports)
from streamlit.testing.v1 import AppTest  # noqa: E402

ANSWER = api_client.AskResult(
    question="How many holidays do I get?",
    answer="You get 24 days of paid annual leave [handbook.pdf, p.1].",
    found=True,
    citations=[("handbook.pdf", 1)],
    sources=[api_client.Source("handbook.pdf", 1, 0.375, "Employees receive 24 days of paid annual leave.")],
    model="gemini-x",
)


@pytest.fixture
def api_up(monkeypatch):
    monkeypatch.setattr(api_client.DocsApiClient, "health", lambda self: True)
    monkeypatch.setattr(api_client.DocsApiClient, "list_documents", lambda self: [("handbook.pdf", 6)])


def run_app() -> AppTest:
    at = AppTest.from_file(str(UI_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning]
    parts += [i.value for i in at.info] + [s.value for s in at.success]
    return "\n".join(str(p) for p in parts)


def test_page_renders_with_api_down(monkeypatch):
    monkeypatch.setattr(api_client.DocsApiClient, "health", lambda self: False)
    at = run_app()
    assert "API not reachable" in all_text(at)
    assert at.chat_input[0].disabled            # can't ask while the API is down


def test_sidebar_lists_indexed_files(api_up):
    at = run_app()
    text = all_text(at)
    assert "API connected" in text
    assert "handbook.pdf" in text and "6 chunks" in text


def test_asking_shows_answer_citation_and_sources(api_up, monkeypatch):
    monkeypatch.setattr(api_client.DocsApiClient, "ask", lambda self, q: ANSWER)
    at = run_app()
    at.chat_input[0].set_value("How many holidays do I get?").run()
    assert not at.exception, at.exception

    text = all_text(at)
    assert "24 days of paid annual leave" in text
    assert "[handbook.pdf, p.1]" in text
    assert at.expander[0].label == "Sources (1)"
    assert len(at.session_state.messages) == 2   # question + answer kept in history


def test_rate_limit_shows_friendly_warning(api_up, monkeypatch):
    def limited(self, q):
        raise api_client.RateLimited("The AI model's free-tier limit was reached. Please wait 60 seconds.", 60)
    monkeypatch.setattr(api_client.DocsApiClient, "ask", limited)
    at = run_app()
    at.chat_input[0].set_value("anything").run()
    assert "free-tier limit" in all_text(at)


def test_not_found_shows_info(api_up, monkeypatch):
    not_found = api_client.AskResult("cricket?", "I couldn't find this in the documents.", False)
    monkeypatch.setattr(api_client.DocsApiClient, "ask", lambda self, q: not_found)
    at = run_app()
    at.chat_input[0].set_value("Who won the cricket world cup?").run()
    assert "couldn't find this" in all_text(at)
    assert at.info  # shown as an info box, not as an answer
