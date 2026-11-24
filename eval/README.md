# Evaluation

Measures **retrieval** quality on a labelled question set. No LLM calls, so it is free
and gives the same result every run.

```bash
python -m eval.run_eval
python -m eval.run_eval --chunks 200:40,400:80 --thresholds 0.2,0.25,0.3 -k 4
python -m eval.run_eval --dataset eval/my_questions.json
```

## Metrics

| Metric | Meaning |
|---|---|
| **Hit Rate@k** | % of answerable questions whose expected page is in the top-k results (after `MIN_SCORE`) |
| **MRR** | Mean of 1/rank of the first correct chunk (1.0 = always first, 0 if missed) |
| **Off-topic rejected** | % of off-topic questions for which retrieval returns nothing |
| **Unanswerable rejected** | Same, for questions on a related topic that the documents don't answer. Informational only: no threshold can reliably reject these, so the LLM's "not found" rule handles them |
| **Balanced** | (Hit Rate@k + off-topic rejected) / 2 |
| **Margin** | Distance from `MIN_SCORE` to the nearest mistake. Larger is more robust to new wording; negative means the threshold already causes errors |

## Dataset format

`handbook_questions.json` has 24 answerable, 8 off-topic and 6 unanswerable questions.
They were written fresh, not reused from the step 6 calibration questions, to avoid overfitting.

```json
{"documents": ["samples/company_handbook.pdf"],
 "questions": [{"id": "a01", "category": "answerable", "question": "...",
                "expected": [{"source": "company_handbook.pdf", "page": 1}]}]}
```

To evaluate your own PDFs, copy the format, list your PDFs under `documents`, and write
30 or more questions phrased the way real users would ask them.

## Results: 2026-11-24, all-MiniLM-L6-v2, k=4

| Chunks (count) | Weakest correct | Strongest off-topic | Gap | Ideal MIN_SCORE |
|---|---|---|---|---|
| 100/20 (12) | 0.394 | 0.133 | **0.261** | 0.263 |
| 200/40 (6) | 0.309 | 0.171 | 0.138 | 0.240 |
| 800/120 (3) | 0.220 | 0.139 | 0.081 | 0.179 |

| Chunks | MIN_SCORE | Hit@1 | Hit@4 | MRR | Off-topic rejected | Unanswerable rejected | Margin |
|---|---|---|---|---|---|---|---|
| 100/20 | **0.25** | 100% | 100% | 1.000 | 100% | 33% | **0.117** |
| 200/40 | 0.25 | 100% | 100% | 1.000 | 100% | 50% | 0.059 |
| 800/120 | 0.20 | 100% | 100% | 1.000 | 100% | 50% | 0.020 |
| 800/120 | 0.25 | 96% | 96% | 0.958 | 100% | 50% | −0.030 |

### Findings
- **Ranking is perfect for every chunk size.** Without a threshold the right page is always ranked first.
  The handbook has three very distinct topics, so ranking is easy. Real documents will be harder.
- **Smaller chunks separate scores far better.** The gap grows from 0.081 to 0.261 as chunks shrink from 800 to 100.
  Focused chunks match focused questions more strongly, while off-topic scores stay low.
- **The old default (800/120, 0.25) misses one question.** "When is a medical certificate required?" scores 0.220
  because the sentence is diluted by the rest of the leave page.
- **MIN_SCORE=0.25 holds up on held-out questions** for 100/20 and 200/40.
- **Unanswerable questions often get past retrieval** (50–67%), as expected. They are the LLM's job.

### Caveat
The sample has only three short pages and 38 questions, and each question carries 2.6 percentage points.
Re-run this evaluation on your real documents before trusting any setting.
