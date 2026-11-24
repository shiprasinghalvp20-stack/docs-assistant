"""Compare chunk sizes and MIN_SCORE values on the labelled question set (retrieval only, free).

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --chunks 100:20,200:40,800:120 --thresholds 0.15,0.2,0.25,0.3 -k 4
"""
import argparse
import logging
from pathlib import Path

from app.core.config import get_settings
from app.evaluation import evaluate_chunk_configs, format_table, load_dataset, score, score_gap

DEFAULT_DATASET = Path(__file__).resolve().parent / "handbook_questions.json"


def parse_pairs(text: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in pair.split(":")) for pair in text.split(",")]


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--chunks", default="100:20,200:40,800:120", help="size:overlap pairs")
    parser.add_argument("--thresholds", default="0.15,0.20,0.25,0.30,0.35")
    parser.add_argument("-k", type=int, default=settings.top_k)
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    from app.services.embeddings import get_embeddings  # loads the real model (local, free)

    documents, questions = load_dataset(args.dataset)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    counts = {c: sum(q.category == c for q in questions) for c in ("answerable", "off_topic", "unanswerable")}
    print(f"Dataset: {len(questions)} questions ({counts['answerable']} answerable, "
          f"{counts['off_topic']} off-topic, {counts['unanswerable']} unanswerable) | "
          f"model: {settings.embedding_model} | k={args.k}\n")

    configs = evaluate_chunk_configs(documents, questions, get_embeddings(), settings.embedding_model,
                                     parse_pairs(args.chunks), max_k=args.k)
    cols = ["chunks", "min_score", "hit@1", f"hit@{args.k}", "mrr", "off_topic_rejected",
            "unanswerable_rejected", "balanced", "margin"]

    # 1) Pure ranking quality: no threshold.
    print("1) RANKING QUALITY (no threshold)")
    rows = [{"chunks": f"{c.chunk_size}/{c.chunk_overlap} ({c.num_chunks})", "min_score": "none",
             **score(c.runs, args.k, -1.0)} for c in configs]
    print(format_table(rows, cols[:5]) + "\n")

    # 1b) How well do scores separate answerable from off-topic questions?
    print("1b) SCORE GAP (a MIN_SCORE inside the gap separates every answerable from every off-topic question)")
    gap_rows = []
    for c in configs:
        weakest, strongest_off = score_gap(c.runs)
        gap_rows.append({"chunks": f"{c.chunk_size}/{c.chunk_overlap} ({c.num_chunks})",
                         "weakest_correct": f"{weakest:.3f}", "strongest_off_topic": f"{strongest_off:.3f}",
                         "gap": f"{weakest - strongest_off:.3f}",
                         "ideal_min_score": f"{(weakest + strongest_off) / 2:.3f}"})
    print(format_table(gap_rows, list(gap_rows[0])) + "\n")

    # 2) Every chunk setting x threshold.
    print("2) CHUNK SIZE x MIN_SCORE")
    all_rows = []
    for c in configs:
        for t in thresholds:
            all_rows.append({"chunks": f"{c.chunk_size}/{c.chunk_overlap} ({c.num_chunks})",
                             "min_score": f"{t:.2f}", "_config": c, "_t": t, **score(c.runs, args.k, t)})
    print(format_table(all_rows, cols) + "\n")

    # 3) Best combination: highest balanced score; ties broken by the safest margin, then MRR.
    best = max(all_rows, key=lambda r: (round(r["balanced"], 6), round(r["margin"], 6), round(r["mrr"], 6)))
    print(f"BEST: chunks {best['chunks']} with MIN_SCORE={best['min_score']} "
          f"(balanced {best['balanced'] * 100:.0f}%, margin {best['margin']:.3f}, MRR {best['mrr']:.3f})\n")

    # 4) What went wrong with the current .env settings - the most useful part for improving.
    current = next((c for c in configs if (c.chunk_size, c.chunk_overlap) ==
                    (settings.chunk_size, settings.chunk_overlap)), configs[-1])
    t = settings.min_score
    print(f"4) MISTAKES with chunks {current.chunk_size}/{current.chunk_overlap}, MIN_SCORE={t:.2f}")
    mistakes = 0
    for run in current.runs:
        q, top = run.question, run.results[0] if run.results else None
        top_txt = f"top: p{top[1]} {top[2]:.3f}" if top else "no results"
        if q.category == "answerable" and run.first_relevant_rank(args.k, t) is None:
            mistakes += 1
            print(f"   MISSED     {q.id} {q.question!r:65} expected p{sorted(q.expected)[0][1]}, {top_txt}")
        elif q.category == "answerable" and run.first_relevant_rank(1, t) is None:
            print(f"   NOT FIRST  {q.id} {q.question!r:65} expected p{sorted(q.expected)[0][1]}, {top_txt}")
        elif q.category == "off_topic" and run.kept(1, t):
            mistakes += 1
            print(f"   LET IN     {q.id} {q.question!r:65} {top_txt}")
    print(f"   {mistakes} mistake(s).")
    passed = [r for r in current.runs if r.question.category == "unanswerable" and r.kept(1, t)]
    if passed:
        print(f"   ({len(passed)}/{counts['unanswerable']} unanswerable questions reached the LLM - "
              f"expected; the LLM's 'not found' rule must handle these)")


if __name__ == "__main__":
    main()
