import math
import re

import pytest
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.retrieval import RetrievedChunk, retrieve
from app.services.vector_store import VectorStore


class BagOfWordsEmbeddings:
    """Fake model where texts sharing words get similar vectors (no download, no API).

    Real models understand meaning; this one only counts words - but that's enough
    to test ranking, k and the score threshold deterministically.
    """

    VOCAB = ["leave", "days", "annual", "sick", "remote", "home", "work", "vpn",
             "expense", "meal", "allowance", "travel", "receipts", "cricket", "cup"]

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        v = [float(words.count(w)) for w in self.VOCAB]
        norm = math.sqrt(sum(x * x for x in v))
        # Unknown words only -> a vector that's orthogonal to everything (score 0).
        return [x / norm for x in v] if norm else [0.0] * (len(v) - 1) + [1e-9]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


CHUNKS = [
    ("hr.pdf", 1, "Annual leave is 24 days. Sick leave is 10 days."),
    ("hr.pdf", 2, "Remote work from home is allowed with VPN."),
    ("hr.pdf", 4, "Expense receipts and meal allowance during travel."),
]


@pytest.fixture
def handbook_store(tmp_path):
    store = VectorStore(tmp_path / "chroma", BagOfWordsEmbeddings(), "bow-model")
    store.add_chunks([
        Document(page_content=text, metadata={
            "source": src, "page": page, "chunk_index": 0, "start_index": 0,
            "chunk_id": f"{src}::p{page}::c0"})
        for src, page, text in CHUNKS
    ])
    return store


# ---------- results ----------

def test_returns_source_page_and_score(handbook_store):
    [best, *_] = retrieve("How many annual leave days?", k=3, min_score=0, store=handbook_store)
    assert isinstance(best, RetrievedChunk)
    assert (best.source, best.page, best.chunk_id) == ("hr.pdf", 1, "hr.pdf::p1::c0")
    assert "24 days" in best.text
    assert 0 < best.score <= 1


@pytest.mark.parametrize("question, page", [
    ("annual leave days", 1),
    ("work from home with vpn", 2),
    ("meal allowance when I travel", 4),
])
def test_most_relevant_chunk_ranks_first(handbook_store, question, page):
    assert retrieve(question, k=3, min_score=0, store=handbook_store)[0].page == page


def test_results_are_sorted_best_first(handbook_store):
    scores = [r.score for r in retrieve("leave work expense", k=3, min_score=-1, store=handbook_store)]
    assert scores == sorted(scores, reverse=True)


def test_identical_text_scores_about_one(handbook_store):
    [best, *_] = retrieve(CHUNKS[1][2], k=1, min_score=0, store=handbook_store)
    assert best.score == pytest.approx(1.0, abs=1e-3)


# ---------- k ----------

def test_k_limits_number_of_results(handbook_store):
    assert len(retrieve("leave work expense", k=2, min_score=-1, store=handbook_store)) == 2


def test_k_larger_than_store_returns_everything(handbook_store):
    assert len(retrieve("leave work expense", k=50, min_score=-1, store=handbook_store)) == 3


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(handbook_store, k):
    with pytest.raises(ValueError, match="k must be positive"):
        retrieve("leave", k=k, store=handbook_store)


# ---------- threshold ----------

def test_off_topic_question_returns_nothing(handbook_store):
    assert retrieve("Who won the cricket world cup?", k=3, min_score=0.25, store=handbook_store) == []


def test_threshold_drops_weak_matches_but_keeps_strong_ones(handbook_store):
    everything = retrieve("annual leave days", k=3, min_score=-1, store=handbook_store)
    strong = retrieve("annual leave days", k=3, min_score=0.5, store=handbook_store)
    assert len(strong) < len(everything)
    assert all(r.score >= 0.5 for r in strong)
    assert strong[0].page == 1


# ---------- edge cases and settings ----------

def test_empty_store_returns_nothing(tmp_path):
    empty = VectorStore(tmp_path / "chroma", BagOfWordsEmbeddings(), "bow-model")
    assert retrieve("anything", k=3, min_score=-1, store=empty) == []


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_empty_question_raises(handbook_store, question):
    with pytest.raises(ValueError, match="must not be empty"):
        retrieve(question, store=handbook_store)


def test_defaults_come_from_settings(handbook_store, monkeypatch):
    monkeypatch.setenv("TOP_K", "1")
    monkeypatch.setenv("MIN_SCORE", "-1")
    get_settings.cache_clear()
    try:
        assert len(retrieve("leave work expense", store=handbook_store)) == 1
    finally:
        get_settings.cache_clear()
