"""Retrieval: find the chunks most relevant to a question.

Try it:
    python -m app.services.retrieval "How many holidays do I get?"
    python -m app.services.retrieval "Can I work from home?" -k 2 --min-score 0
"""
import argparse
import logging
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str     # file name, e.g. "company_handbook.pdf"
    page: int       # human page number (starts at 1)
    chunk_id: str
    score: float    # cosine similarity, -1..1 (higher = more relevant)


def retrieve(
    question: str,
    k: int | None = None,
    min_score: float | None = None,
    store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Return up to k chunks with score >= min_score, best first.

    An empty list means "nothing relevant in the documents" - callers should then
    answer "I couldn't find that" instead of letting the LLM guess.
    k / min_score default to TOP_K / MIN_SCORE from settings.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    settings = get_settings()
    k = k if k is not None else settings.top_k
    min_score = min_score if min_score is not None else settings.min_score
    if k <= 0:
        raise ValueError(f"k must be positive (got {k}).")
    if store is None:
        from app.services.ingestion import get_vector_store  # loads the model only when needed

        store = get_vector_store()

    results = [
        RetrievedChunk(
            text=doc.page_content,
            source=doc.metadata["source"],
            page=doc.metadata["page"],
            chunk_id=doc.metadata["chunk_id"],
            score=round(score, 4),
        )
        for doc, score in store.search(question, k)
    ]
    kept = [r for r in results if r.score >= min_score]
    logger.info(
        "Retrieved %d/%d chunks for %r (min_score=%.2f, best=%s).",
        len(kept), len(results), question, min_score,
        f"{results[0].score:.3f}" if results else "n/a",
    )
    return kept


def _main() -> None:
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, help="max results (default: TOP_K)")
    parser.add_argument("--min-score", type=float, help="score threshold (default: MIN_SCORE)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        chunks = retrieve(args.question, args.k, args.min_score)
    except ValueError as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Q: {args.question}")
    if not chunks:
        print("   -> No relevant chunks found (all below the score threshold).")
    for i, c in enumerate(chunks, 1):
        preview = c.text.replace("\n", " ")[:110]
        print(f"   {i}. [{c.source}, p.{c.page}]  score={c.score:.3f}\n      {preview}...")


if __name__ == "__main__":
    _main()
