"""Retrieval evaluation: Hit Rate@k, MRR and off-topic rejection on a labelled question set.

Retrieval only - no LLM calls, so it's free and fully repeatable.

Each question is searched ONCE per chunk setting (no threshold, top max_k results);
every MIN_SCORE value is then scored from those same results, so sweeping thresholds
costs nothing extra.

Run it:  python -m eval.run_eval
"""
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.embeddings import Embeddings

from app.services.ingestion import ingest_pdf
from app.services.vector_store import VectorStore

Category = Literal["answerable", "off_topic", "unanswerable"]


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    category: Category
    expected: frozenset[tuple[str, int]]   # {(file, page)}; empty for off_topic / unanswerable


@dataclass
class QuestionRun:
    """Ranked retrieval results for one question, before any threshold: [(file, page, score)]."""
    question: EvalQuestion
    results: list[tuple[str, int, float]] = field(default_factory=list)

    def kept(self, k: int, min_score: float) -> list[tuple[str, int, float]]:
        """What retrieve() would return with this k and threshold."""
        return [r for r in self.results[:k] if r[2] >= min_score]

    def first_relevant_rank(self, k: int, min_score: float) -> int | None:
        """1-based rank of the first chunk from an expected page, or None if missed."""
        for rank, (source, page, _) in enumerate(self.kept(k, min_score), start=1):
            if (source, page) in self.question.expected:
                return rank
        return None


# ---------- dataset ----------

def load_dataset(path: str | Path) -> tuple[list[Path], list[EvalQuestion]]:
    """Read the JSON question set. Document paths are relative to the project root."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent
    documents = [root / d for d in data["documents"]]
    questions = [
        EvalQuestion(
            id=q["id"],
            question=q["question"],
            category=q["category"],
            expected=frozenset((e["source"], e["page"]) for e in q["expected"]),
        )
        for q in data["questions"]
    ]
    return documents, questions


# ---------- metrics ----------

def _answerable(runs: list[QuestionRun]) -> list[QuestionRun]:
    return [r for r in runs if r.question.category == "answerable"]


def hit_rate(runs: list[QuestionRun], k: int, min_score: float = -1.0) -> float:
    """Share of answerable questions with an expected page in the kept top-k results."""
    answerable = _answerable(runs)
    hits = sum(r.first_relevant_rank(k, min_score) is not None for r in answerable)
    return hits / len(answerable) if answerable else 0.0


def mrr(runs: list[QuestionRun], k: int, min_score: float = -1.0) -> float:
    """Mean Reciprocal Rank over answerable questions: average of 1/rank (0 if missed)."""
    answerable = _answerable(runs)
    total = 0.0
    for r in answerable:
        rank = r.first_relevant_rank(k, min_score)
        total += 1.0 / rank if rank else 0.0
    return total / len(answerable) if answerable else 0.0


def rejection_rate(runs: list[QuestionRun], category: Category, min_score: float) -> float:
    """Share of questions in `category` for which retrieval returns nothing."""
    selected = [r for r in runs if r.question.category == category]
    rejected = sum(not r.kept(1, min_score) for r in selected)
    return rejected / len(selected) if selected else 0.0


# ---------- running ----------

def build_index(documents: list[Path], embeddings: Embeddings, model_name: str,
                chunk_size: int, chunk_overlap: int, persist_dir: Path) -> VectorStore:
    """Index the documents into a fresh, throw-away vector store (never touches data/chroma)."""
    store = VectorStore(persist_dir, embeddings, embedding_model=model_name, collection_name="eval")
    for doc in documents:
        ingest_pdf(doc, store, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return store


def run_questions(questions: list[EvalQuestion], store: VectorStore, max_k: int) -> list[QuestionRun]:
    """Search every question once, without a threshold."""
    runs = []
    for q in questions:
        hits = store.search(q.question, max_k)
        runs.append(QuestionRun(q, [(d.metadata["source"], d.metadata["page"], round(s, 4)) for d, s in hits]))
    return runs


@dataclass
class ConfigResult:
    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    runs: list[QuestionRun]


def evaluate_chunk_configs(documents: list[Path], questions: list[EvalQuestion], embeddings: Embeddings,
                           model_name: str, chunk_configs: list[tuple[int, int]], max_k: int) -> list[ConfigResult]:
    results = []
    for size, overlap in chunk_configs:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = build_index(documents, embeddings, model_name, size, overlap, Path(tmp))
            results.append(ConfigResult(size, overlap, store.count(), run_questions(questions, store, max_k)))
    return results


def score_gap(runs: list[QuestionRun]) -> tuple[float, float]:
    """(weakest correct match, strongest off-topic match), ignoring any threshold.

    weakest correct = lowest score that an expected page reached across answerable questions
    (-1 if some expected page wasn't retrieved at all). A threshold between the two values
    separates every answerable question from every off-topic one.
    """
    weakest = min(
        (max((s for src, p, s in r.results if (src, p) in r.question.expected), default=-1.0)
         for r in _answerable(runs)),
        default=-1.0,
    )
    strongest_off = max(
        (r.results[0][2] for r in runs if r.question.category == "off_topic" and r.results),
        default=-1.0,
    )
    return weakest, strongest_off


def score(runs: list[QuestionRun], k: int, min_score: float) -> dict[str, float]:
    """All headline metrics for one (chunk setting, k, threshold) combination."""
    hit_k = hit_rate(runs, k, min_score)
    off = rejection_rate(runs, "off_topic", min_score)
    weakest, strongest_off = score_gap(runs)
    return {
        "hit@1": hit_rate(runs, 1, min_score),
        f"hit@{k}": hit_k,
        "mrr": mrr(runs, k, min_score),
        "off_topic_rejected": off,
        "unanswerable_rejected": rejection_rate(runs, "unanswerable", min_score),
        # One number to compare settings: answerable found AND off-topic rejected, weighted equally.
        "balanced": (hit_k + off) / 2,
        # Robustness: distance from the threshold to the nearest mistake (negative = already wrong).
        # Bigger = new phrasings are less likely to fall on the wrong side.
        "margin": min(weakest - min_score, min_score - strongest_off),
    }


# ---------- reporting ----------

def format_table(rows: list[dict], columns: list[str]) -> str:
    """Plain-text table; rates are shown as percentages, MRR and score margins as decimals."""
    def fmt(col: str, value) -> str:
        if isinstance(value, float):
            return f"{value:.3f}" if col in ("mrr", "margin") else f"{value * 100:.0f}%"
        return str(value)

    cells = [[fmt(c, row[c]) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    line = "  ".join(c.rjust(w) for c, w in zip(columns, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in cells]
    return "\n".join([line, sep, *body])
