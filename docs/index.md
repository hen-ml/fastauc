# fastauc

Exact, fast classification, calibration and ranking metrics for Python,
written in Rust.

```
pip install pyfastauc
```

The package installs as `pyfastauc` and imports as `fastauc`. NumPy is the
only dependency. Wheels cover CPython 3.9+ on Linux, macOS and Windows,
including free-threaded 3.14t.

## Why another metrics library

Most evaluation code sorts by score and breaks ties by input order. When a
model gives several items the same score (a constant fallback, a quantized or
bfloat16 output, a cold-start default), the metric then depends on how the
rows happened to be laid out:

```python
import numpy as np, fastauc

rel   = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])   # clicked items happen to be logged first
score = np.zeros(10)                                # a model that knows nothing

order = np.argsort(-score, kind="stable")           # what most evaluation code does
rel[order][:2].mean()                               # precision@2 = 1.0, a perfect score

fastauc.ranking_metrics(rel, score, group_sizes=[10], ks=[2])["precision@2"]   # 0.2, the base rate
```

fastauc never breaks ties by position. Every top-k metric is the exact
average over all orderings of the tied items, in closed form, and every
threshold metric cuts only between distinct scores. No randomness, no seed,
the same number on every run and for every row order. See
[Exact tie handling](ties.md).

"Exact" runs through the whole library:

- **Ties** are averaged in closed form, never broken by position or at random.
- **String ids** are grouped by an exact hash table: the hash picks a bucket and ids are compared byte for byte, so two different ids are never merged.
- **Streaming** keeps the full table of distinct scores, not a sketch: the result after a million batches is the result on all rows at once.
- **scikit-learn parity** is tested function by function, warnings and errors included.

## What you get

| Area | Functions |
|---|---|
| Binary classification | `binary_metrics`, `roc_auc`, `average_precision`, `pr_auc`, `partial_roc_auc`, `ks_statistic`, `max_f1`, operating points |
| Multiclass | `roc_auc_multiclass` (one-vs-rest, one-vs-one) |
| Calibration | `calibration_metrics`, `log_loss`, `brier_score`, `normalized_entropy`, `calibration_ratio`, `expected_calibration_error`, `calibration_curve` |
| Ranking per query | `ranking_metrics`: GAUC, NDCG, MRR, MAP, recall, precision, hit rate at any cutoffs, with optional complete judgments (qrels) |
| Many models at once | `roc_auc_ap_batch`, `roc_auc_ap_grouped` |
| Uncertainty | `bootstrap_roc_auc_ap`, `bootstrap_ranking_metrics` (by row or by group), `delong_roc_auc` |
| Significance | `paired_test` (randomization or paired t-test), `holm` |
| Streaming | `BinaryAccumulator` (exact, mergeable across workers) |
| scikit-learn drop-in | `fastauc.sklearn`: `roc_auc_score`, `average_precision_score`, `ndcg_score`, `dcg_score`, `roc_curve`, `precision_recall_curve`, `auc`, `log_loss`, `brier_score_loss` |

## Thirty-second tour

```python
import fastauc

fastauc.binary_metrics(y, p)                       # ROC-AUC, AP, PR-AUC, KS, F1, log loss, ECE, ...
fastauc.ranking_metrics(rel, s, query_id, ks=(10,))  # GAUC, NDCG@10, MRR@10, MAP@10, ...

b = fastauc.bootstrap_ranking_metrics(rel, np.stack([s_a, s_b]), query_id)
b.compare(0, 1, metric="ndcg@10")                  # paired, query-level bootstrap

from fastauc import sklearn as metrics             # same calls as sklearn.metrics
metrics.roc_auc_score(y, p)
```
