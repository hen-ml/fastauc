# fastauc

[![CI](https://github.com/hen-ml/fastauc/actions/workflows/CI.yml/badge.svg)](https://github.com/hen-ml/fastauc/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Fast, correct classification, calibration and ranking metrics for Python, written in Rust.

## Your offline ranking metrics may be inflated by row order

```python
import numpy as np, fastauc

rel   = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])   # clicked items happen to be logged first
score = np.zeros(10)                                # a model that knows nothing

order = np.argsort(-score, kind="stable")           # what most evaluation code does
rel[order][:2].mean()                               # precision@2 = 1.0, a perfect score

fastauc.ranking_metrics(rel, score, group_sizes=[10], ks=[2])["precision@2"]   # 0.2, the base rate
```

Whenever a model gives several items the same score (a constant fallback, a
quantized or clipped output, a cold-start default), a sort has to break the tie
somehow, and most code breaks it by input order. If the input lists positives
first, NDCG, MRR, MAP and precision@k all look better than they are.

fastauc never breaks ties by position. Tied items are treated as
interchangeable and every top-k metric is the exact average over all their
orderings, in closed form: no random tie-breaking, no seed, the same answer on
every run. See [Tied scores](#tied-scores-never-leak-row-order).

## What you get

```
pip install fastauc
```

| Function | Returns |
|---|---|
| `binary_metrics` | ROC-AUC, average precision, KS, max F1, log loss, normalized entropy, Brier, calibration ratio, ECE |
| `ranking_metrics` | GAUC, NDCG, MRR, MAP, recall, precision, hit rate at any cutoffs, per query |
| `roc_pr_auc_batch` | ROC-AUC and average precision for thousands of models or targets at once |
| `roc_pr_auc_grouped` | ROC-AUC and average precision within each group |
| `bootstrap_roc_pr_auc`, `bootstrap_ranking_metrics` | confidence intervals and paired model comparisons, by row or by group |
| `delong_roc_auc` | analytic AUC confidence intervals and the DeLong test for two models, no resampling |
| `roc_curve`, `precision_recall_curve`, `calibration_curve` | curve points for plotting, same output as scikit-learn |
| `precision_at_recall`, `recall_at_precision`, `tpr_at_fpr`, `lift_at`, `ks_statistic`, `max_f1` | operating points with their thresholds |
| `log_loss`, `brier_score`, `normalized_entropy`, `calibration_ratio`, `expected_calibration_error` | individual calibration metrics |

- Same numbers as scikit-learn where scikit-learn has the metric, including tied scores and sample weights.
- Multithreaded, GIL released, no JIT warm-up. NumPy is the only dependency.
- One wheel per platform covers every CPython from 3.9.
- Results are bit-identical regardless of thread count and of input row order.

## Quick start

### One binary model

```python
fastauc.binary_metrics(y, p)
# {'roc_auc', 'average_precision', 'ks', 'ks_threshold', 'f1_max', 'f1_threshold',
#  'prevalence', 'log_loss', 'normalized_entropy', 'brier', 'calibration_ratio', 'ece'}
```

One sort gives the discrimination metrics; one extra linear pass gives the
calibration block. The calibration keys are `NaN` when scores are not
probabilities in `[0, 1]` (for example raw logits).

Each piece is available on its own. A threshold `t` always means "predict
positive when `score >= t`".

```python
fastauc.roc_auc(y, s)                   # == sklearn.metrics.roc_auc_score
fastauc.average_precision(y, s)         # == sklearn.metrics.average_precision_score
fastauc.ks_statistic(y, s)              # (ks, threshold)
fastauc.max_f1(y, s)                    # (f1, threshold)
fastauc.precision_at_recall(y, s, 0.8)  # (precision, threshold)
fastauc.recall_at_precision(y, s, 0.9)  # (recall, threshold)
fastauc.tpr_at_fpr(y, s, 0.01)          # (tpr, threshold)
fastauc.lift_at(y, s, 0.1)              # top-decile lift

fastauc.calibration_metrics(y, p, n_bins=10, strategy="uniform")  # or "quantile"
fastauc.normalized_entropy(y, p)        # log loss / entropy of the base rate; RIG = 1 - NE
fastauc.calibration_ratio(y, p)         # sum(p) / sum(y), also called COPC or E/O
```

All of these accept `sample_weight` and `pos_label`.

### Ranking metrics per query

```python
fastauc.ranking_metrics(relevance, s, query_id, ks=(1, 5, 10, None))
# {'gauc', 'gauc_macro',
#  'ndcg@1', 'mrr@1', 'map@1', 'recall@1', 'precision@1', 'hit_rate@1', ... '@5', '@10',
#  'ndcg', 'mrr', 'map', 'recall', 'precision', 'hit_rate',      # k=None: whole list
#  'n_groups', 'n_groups_ranked', 'n_groups_auc'}
```

- `relevance` is binary or graded (non-negative). An item counts as relevant when it is `> 0`; grades only change NDCG (`gain="exponential"` or `"linear"`).
- Pass ids via `groups` (any row order, any dtype), or `group_sizes=` for data already laid out query by query, as LightGBM and XGBoost expect. Integer ids in a compact range and `group_sizes` are the fast paths; string or sparse ids are sorted first.
- `gauc` is weighted by group size; `gauc_macro` is the plain mean over groups.
- Queries with no relevant item are left out of the averages (`no_relevant="zero"` counts them as 0 instead). `n_groups_ranked` tells you how many were used.
- `sample_weight` weights the average over queries: a query's weight is the mean weight of its rows (the LightGBM convention).
- `per_group=True` also returns every per-query value.

### Curves for plotting

```python
fpr, tpr, thresholds = fastauc.roc_curve(y, s)                  # == sklearn.metrics.roc_curve
precision, recall, thresholds = fastauc.precision_recall_curve(y, s)
prob_true, prob_pred = fastauc.calibration_curve(y, p, n_bins=10)  # reliability diagram
```

Same arrays as scikit-learn (including `drop_intermediate` and
`sample_weight`), 3 to 4x faster. `calibration_curve` also takes
`sample_weight` and `return_weights=True` (points per bin), and its
`strategy="quantile"` makes truly equal-weight bins even with tied
probabilities.

### Many models or many targets

```python
S = rng.random((1000, 100_000))            # 1000 models, one row each
roc, ap = fastauc.roc_pr_auc_batch(y, S)   # arrays of shape (1000,)

Y = rng.integers(0, 2, (1000, 100_000))    # 1000 targets, e.g. label permutations
roc, ap = fastauc.roc_pr_auc_batch(Y, s)   # sorts s once, then O(N) per target

roc, ap = fastauc.roc_pr_auc_batch(Y, S)   # row i against row i
```

2-D inputs are `(n_tasks, n_samples)` by default. For the scikit-learn layout
`(n_samples, n_tasks)` pass `axis=0`.

### AUC within groups

```python
res = fastauc.roc_pr_auc_grouped(y, s, user_id)
res.groups, res.roc_auc, res.average_precision, res.n_pos, res.n_neg  # aligned arrays
res.mean("roc_auc")                      # size-weighted GAUC
res.mean("roc_auc", weight="uniform")    # macro average over groups
```

### Confidence intervals

```python
b = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=2000, seed=0)
b.roc_auc_estimate            # point estimate
b.ci("roc_auc", level=0.95)   # (low, high) percentile interval
b.std("average_precision")    # bootstrap standard error
```

Scores are sorted once and each replicate is a linear scan, so thousands of
replicates cost about as much as a few dozen plain evaluations.

**Resample groups, not rows, when rows are correlated.** Impressions from the
same user, query or session are not independent, and a row-level bootstrap
then reports intervals that are too narrow. Pass the ids:

```python
b = fastauc.bootstrap_roc_pr_auc(y, s, groups=user_id, n_bootstraps=2000)
```

Ranking metrics always resample whole queries:

```python
r = fastauc.bootstrap_ranking_metrics(relevance, s, query_id, ks=(10,), n_bootstraps=2000)
r.estimate["ndcg@10"], r.ci("ndcg@10"), r.std("mrr@10")
```

Other options: `stratified=True` keeps the class balance fixed (use it when
positives are rare); `sample_weight` is supported.

### Is model A better than model B?

For ROC-AUC on independent rows, the DeLong test answers in one pass, with
no resampling:

```python
d = fastauc.delong_roc_auc(y, np.stack([scores_a, scores_b]))
d.roc_auc, d.ci(), d.std()                       # per model
diff, (low, high), p_value = d.compare(0, 1)     # paired test of a vs b
```

It uses the fast algorithm of Sun and Xu (2014), one sort per model:
two models on 1,000,000 rows take 0.09 s, against 16 s for a 1,000-replicate
bootstrap. It assumes independent rows and no sample weights; otherwise use
the bootstrap below.

For other metrics, clustered rows or weights, give a bootstrap several score
vectors. Every replicate uses the same resample
for all models, so the comparison is paired and far more sensitive than
comparing two separate intervals:

```python
b = fastauc.bootstrap_roc_pr_auc(y, np.stack([scores_a, scores_b]), groups=user_id)
diff, (low, high), p_value = b.compare(0, 1, metric="roc_auc")

r = fastauc.bootstrap_ranking_metrics(relevance, np.stack([scores_a, scores_b]), query_id)
diff, (low, high), p_value = r.compare(0, 1, metric="ndcg@10")
```

## Use it inside your training loop

Runnable scripts are in [`examples/`](examples).

**LightGBM**

```python
def ranking_feval(preds, data):
    m = fastauc.ranking_metrics(data.get_label(), preds, group_sizes=data.get_group(), ks=(10,))
    return [("ndcg@10", m["ndcg@10"], True), ("mrr@10", m["mrr@10"], True)]

lgb.train(params, train, valid_sets=[valid], feval=ranking_feval)
```

**XGBoost**

```python
def ndcg_at_10(preds, dmatrix):
    m = fastauc.ranking_metrics(dmatrix.get_label(), preds,
                                group_sizes=dmatrix.get_group().astype(np.int64), ks=(10,))
    return "fastauc-ndcg@10", m["ndcg@10"]

xgb.train(params, dtrain, evals=[(dvalid, "valid")], custom_metric=ndcg_at_10)
```

**PyTorch** (or anything that can hand over a NumPy array)

```python
p = torch.sigmoid(logits).cpu().numpy()
metrics = fastauc.binary_metrics(labels.cpu().numpy(), p)
metrics.update(fastauc.ranking_metrics(labels.cpu().numpy(), p, query_ids.cpu().numpy()))
```

**Threads.** fastauc uses all cores by default. Inside data-loader workers,
joblib jobs or any code that is already parallel, call
`fastauc.set_num_threads(1)` to avoid oversubscription
(`set_num_threads(None)` restores the default; the `RAYON_NUM_THREADS`
environment variable also works). Results do not depend on the thread count.

## Tied scores never leak row order

fastauc treats items with equal scores as interchangeable, everywhere:

- **Top-k ranking metrics** (NDCG, MRR, MAP, recall, precision, hit rate) return the exact expected value over all orderings of the tied items, in closed form. It is what you would get by averaging over infinitely many random tie-breaks, without the randomness.
- **Threshold metrics** (ROC-AUC, average precision, KS, F1, operating points) only place thresholds between distinct scores, so a tied block is one point on the curve. For ROC-AUC this gives a tied positive/negative pair the usual half credit.
- **Lift and quantile-binned ECE** split a tied block that straddles a cut in proportion.

So a constant score gives ROC-AUC 0.5, lift 1.0, and precision@k equal to the
query's base rate, however the rows are sorted. The test suite checks the tie
formulas against brute-force enumeration of every ordering, and checks that
every metric is unchanged under row permutation, including the adversarial
"positives first" layout.

With no ties these are the textbook definitions, and NDCG matches
`sklearn.metrics.ndcg_score` (which also averages ties).

## How it compares

**Tie handling.** `python benchmarks/tie_leak.py` runs other libraries on
1,000 queries of 10 documents (2 relevant each) scored by a model that gives
every document the same score. The two layouts hold identical data; only the
input order and document ids differ. A tie-correct metric returns the same
chance-level value for both.

| Library | Metrics checked | Relevant first | Relevant last | Correct |
|---|---|---:|---:|---:|
| **fastauc** | NDCG@3, MAP@3, MRR@3, P@1 | 0.26, 0.20, 0.34, 0.20 | same | same |
| scikit-learn 1.9.1 `ndcg_score`, `roc_auc_score` | NDCG@3, AUC | 0.26, 0.50 | same | same |
| ranx 0.3.21 | NDCG@3, MAP@3, MRR@3, P@1 | 1.0 | 0.0 | 0.26, 0.20, 0.34, 0.20 |
| pytrec_eval 0.5.11 / ir-measures 0.4.3 | NDCG@3, P@1 | 0.0 | 1.0 | 0.26, 0.20 |
| LightGBM 4.7.0 built-in eval | NDCG@3, MAP@3 | 1.0 | 0.0 | 0.26, 0.20 |
| XGBoost 3.4.1 built-in eval | NDCG@3, MAP@3 | 1.0 | 0.0 | 0.26, 0.20 |
| fastroc 1.1.3 | AUC | 0.5005 | 0.4995 | 0.5 |

trec_eval (and so pytrec_eval and ir-measures) breaks ties by document id on
purpose; it is a documented convention, but it still means a tied model's
score depends on how documents happen to be named.

From reading the source of libraries not run above: torchmetrics 1.9.0
averages ties for NDCG (a port of scikit-learn's method) but its retrieval
MRR, MAP, precision, recall and hit rate take `topk`/`argsort` with no tie
handling; TorchRec 1.9.2 computes AUC, GAUC and NDCG from an `argsort` with
no tie grouping. TF-Ranking breaks ties randomly, which is unbiased but noisy
and seed-dependent; fastauc returns the exact average of that randomization.

**Coverage.** fastauc puts classification, calibration and per-query
ranking metrics, curves, and confidence intervals (DeLong and bootstrap) in
one NumPy-only package:
scikit-learn has no GAUC, MRR or MAP@k; ranx has no AUC or calibration
metrics; torchmetrics and TorchRec need PyTorch. As far as we know, none of
them bootstraps by user or query.

**What they have that fastauc does not.** GPU and distributed evaluation
(torchmetrics, TorchRec); multiclass and partial AUC (scikit-learn); TREC
file formats, run fusion, per-query significance tests and many more IR
metrics (ranx).

## Behaviour to know about

| Situation | fastauc | scikit-learn |
|---|---|---|
| Only one class present | returns `NaN` (average precision is `1.0` if all positive) | `roc_auc_score` raises |
| `NaN` in scores | raises `ValueError` | raises |
| `±inf` in scores | ranked normally | raises |
| Labels other than `0/1` or `bool` | raises; pass `pos_label=` | infers the positive class |
| Multiclass, `max_fpr`, `average=` | not supported | supported |
| 2-D layout | `(n_tasks, n_samples)`; `axis=0` for the other | `(n_samples, n_tasks)` |

Returning `NaN` rather than raising is deliberate: in a batch, a group-by, or
a bootstrap, one degenerate slice should not abort the whole computation.

Other notes:

- Inputs are converted to `uint8` labels and `float64` scores; pass those dtypes, C-contiguous, to avoid a copy.
- Bootstrap results depend only on `seed`. Each replicate has its own independently seeded xoshiro256++ stream.
- MAP@k divides by `min(n_relevant, k)`; precision@k divides by `k`.
- Up to 2^32 - 1 samples per vector.
- ERR (expected reciprocal rank) is not included.

## Benchmarks

All numbers below are from a **2-core** x86-64 Linux VM, Python 3.13. The
parallel paths scale with cores, so expect larger gaps on real hardware; run
the scripts yourself before quoting numbers.

Against scikit-learn (`python benchmarks/bench.py`; ROC-AUC + average
precision unless noted):

| Task | scikit-learn | fastauc | Speedup |
|---|---:|---:|---:|
| Single pair, N = 1,000 | 2.1 ms | 0.03 ms | 83x |
| Single pair, N = 100,000 | 29 ms | 3.2 ms | 9x |
| Single pair, N = 10,000,000 | 5.9 s | 0.55 s | 11x |
| 1,000 models, 1 target, N = 10,000 | 4.46 s | 0.20 s | 22x |
| 1 model, 1,000 targets, N = 10,000 | 4.42 s | 0.039 s | 113x |
| 1,000 bootstrap replicates, N = 10,000 | 4.61 s | 0.049 s | 95x |
| Grouped AUC, N = 1,000,000, 20,000 groups | 37.3 s | 0.034 s | 1093x |
| NDCG@10, N = 1,000,000, 20,000 groups | 6.8 s | 0.051 s | 132x |
| `binary_metrics`, N = 1,000,000 | 0.79 s | 0.059 s | 13x |

All ranking metrics at four cutoffs on the same 1M rows take 0.052 s, the same
as NDCG@10 alone. The `binary_metrics` baseline is scikit-learn's ROC-AUC,
average precision, log loss and Brier, plus KS and max F1 from its curve
functions.

Against ranx 0.3.21, a Numba-based ranking evaluation library
(`python benchmarks/bench_vs_others.py`; NDCG@10 + MRR@10 + MAP@10 on
1,000,000 rows in 20,000 queries, no tied scores, results identical to six
decimals):

| | Time |
|---|---:|
| fastauc, straight from NumPy arrays | 0.036 s |
| ranx `evaluate`, after JIT warm-up | 0.93 s |
| ranx, building `Qrels` and `Run` from the same arrays | + 5.7 s |

Against fastroc 1.1.3, a C implementation of ROC-AUC, with 2 threads each:
10,000,000 samples take 2.5 s in fastroc and 0.53 s in fastauc (which also
computes average precision), and 1,000 models on 10,000 samples take 0.65 s
versus 0.18 s.

The script also times torchmetrics when it is installed; that comparison has
not been run for this table.

### End-to-end evidence on synthetic data

`python benchmarks/evidence.py` builds two synthetic ranking workloads with
bfloat16-tied scores (short lists of 10, and ragged pools of 8 to 95
documents) and reports, in one file: nDCG@10 against a step-by-step reference
and scikit-learn, ROC-AUC and average precision against scikit-learn, how far
position-based tie-breaking moves the result, a paired query-level bootstrap
comparison of two models, and timings against scikit-learn, a Python loop and
(if installed) a Numba kernel. It exits with an error if any accuracy check
fails. The [Evidence (macOS)](https://github.com/hen-ml/fastauc/actions/workflows/evidence.yml)
workflow runs it on Apple silicon and Intel Macs and shows the report in each
run's summary.

## Development

```
pip install maturin pytest scikit-learn scipy
maturin develop --release
pytest
```

Requires a Rust toolchain (1.85+). The native code is in `src/lib.rs`
(no `unsafe`); input validation and the public API are in
`python/fastauc/__init__.py`.

## Releasing

1. On PyPI, add a trusted publisher: owner `hen-ml`, repository `fastauc`, workflow `CI.yml`, environment `pypi`. For a brand-new project use a "pending publisher" (PyPI account settings, Publishing).
2. Bump the version in `Cargo.toml`, `pyproject.toml` and `python/fastauc/__init__.py`.
3. `git tag v0.1.0 && git push --tags`. CI builds wheels for Linux (glibc and musl, x86-64 and aarch64), macOS (Intel and Apple silicon) and Windows, runs the tests, and publishes.

## License

MIT
