from pathlib import Path

import pytest

from app.evaluation import (
    EvalQuestion, QuestionRun, evaluate_chunk_configs, format_table, hit_rate,
    load_dataset, mrr, rejection_rate, score, score_gap,
)
from app.services.pdf_loader import load_pdf
from tests.fakes import BagOfWordsEmbeddings

DATASET = Path(__file__).resolve().parent.parent / "eval" / "handbook_questions.json"


def q(id, category="answerable", page=None):
    expected = frozenset({("hb.pdf", page)}) if page else frozenset()
    return EvalQuestion(id, f"question {id}", category, expected)


def run(question, *pages_scores):
    return QuestionRun(question, [("hb.pdf", p, s) for p, s in pages_scores])


# The worked example from the explanation: correct page at rank 1, 2, 3, and missed.
EXAMPLE = [
    run(q("q1", page=1), (1, 0.6), (4, 0.3), (2, 0.1)),
    run(q("q2", page=2), (4, 0.5), (2, 0.4), (1, 0.2)),
    run(q("q3", page=4), (1, 0.5), (2, 0.4), (4, 0.3)),
    run(q("q4", page=4), (1, 0.5), (2, 0.4)),
]


# ---------- metrics ----------

def test_hit_rate_worked_example():
    assert hit_rate(EXAMPLE, k=1) == 0.25
    assert hit_rate(EXAMPLE, k=3) == 0.75


def test_mrr_worked_example():
    assert mrr(EXAMPLE, k=3) == pytest.approx((1 + 1 / 2 + 1 / 3 + 0) / 4)


def test_k_cuts_off_lower_ranks():
    assert mrr(EXAMPLE, k=2) == pytest.approx((1 + 1 / 2) / 4)


def test_threshold_turns_low_scoring_hit_into_miss():
    # q3's correct page is at rank 3 with score 0.3; a 0.35 threshold drops it.
    assert hit_rate(EXAMPLE, k=3, min_score=0.35) == 0.5


def test_only_answerable_questions_count_for_hit_rate():
    runs = EXAMPLE[:1] + [run(q("o1", "off_topic"), (1, 0.9))]
    assert hit_rate(runs, k=1) == 1.0


def test_rejection_rate():
    runs = [run(q("o1", "off_topic"), (1, 0.10)), run(q("o2", "off_topic"), (2, 0.30)),
            run(q("u1", "unanswerable"), (1, 0.40))]
    assert rejection_rate(runs, "off_topic", min_score=0.25) == 0.5
    assert rejection_rate(runs, "unanswerable", min_score=0.25) == 0.0
    assert rejection_rate(runs, "off_topic", min_score=0.5) == 1.0


def test_score_gap_and_margin():
    runs = [run(q("a1", page=1), (1, 0.40)), run(q("a2", page=2), (1, 0.5), (2, 0.30)),
            run(q("o1", "off_topic"), (1, 0.10))]
    assert score_gap(runs) == (0.30, 0.10)       # weakest correct, strongest off-topic
    s = score(runs, k=2, min_score=0.20)
    assert s["margin"] == pytest.approx(0.10)    # 0.20 sits 0.10 from both sides
    assert s["balanced"] == 1.0
    assert score(runs, k=2, min_score=0.35)["margin"] < 0   # threshold already loses a2


def test_format_table():
    table = format_table([{"name": "x", "hit@1": 0.5, "mrr": 0.25, "margin": -0.03}],
                         ["name", "hit@1", "mrr", "margin"])
    header, sep, row = table.splitlines()
    assert header.split() == ["name", "hit@1", "mrr", "margin"]
    assert row.split() == ["x", "50%", "0.250", "-0.030"]


# ---------- the dataset itself (bad labels would silently corrupt every score) ----------

def test_dataset_is_well_formed():
    documents, questions = load_dataset(DATASET)
    assert all(d.exists() for d in documents)
    assert len({x.id for x in questions}) == len(questions)          # unique ids
    assert len({x.question for x in questions}) == len(questions)    # no duplicate questions
    for x in questions:
        assert x.category in ("answerable", "off_topic", "unanswerable")
        assert bool(x.expected) == (x.category == "answerable")


def test_every_expected_page_exists_and_has_text():
    documents, questions = load_dataset(DATASET)
    real_pages = {(d.metadata["source"], d.metadata["page"]) for doc in documents for d in load_pdf(doc)}
    for x in questions:
        assert x.expected <= real_pages, f"{x.id} points at a page with no text: {x.expected}"


def test_dataset_covers_every_page_and_category():
    documents, questions = load_dataset(DATASET)
    real_pages = {(d.metadata["source"], d.metadata["page"]) for doc in documents for d in load_pdf(doc)}
    labelled = set().union(*(x.expected for x in questions))
    assert labelled == real_pages
    assert sum(x.category == "off_topic" for x in questions) >= 5
    assert sum(x.category == "unanswerable" for x in questions) >= 3


# ---------- end to end with a fake model ----------

def test_evaluate_chunk_configs_runs_every_question_per_setting():
    documents, questions = load_dataset(DATASET)
    results = evaluate_chunk_configs(documents, questions, BagOfWordsEmbeddings(), "bow-model",
                                     chunk_configs=[(200, 40), (800, 120)], max_k=2)
    assert [(r.chunk_size, r.num_chunks) for r in results] == [(200, 6), (800, 3)]
    for r in results:
        assert len(r.runs) == len(questions)
        assert all(len(x.results) <= 2 for x in r.runs)
        scores = [s for x in r.runs for _, _, s in x.results]
        assert all(-1.0 <= s <= 1.0001 for s in scores)
