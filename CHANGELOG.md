# Changelog

## Unreleased

- Docs: [case study](https://hen-ml.github.io/fastauc/case-study/) on Istella-S LETOR. LightGBM's own NDCG@10 after one round ranges from 0.465 to 0.694 with row order alone, and short-patience early stopping can keep the round-1 model. Script: `benchmarks/case_study.py`, charts: `benchmarks/plot_case_study.py`.

## 0.3.0

- `fastauc.lightgbm` and `fastauc.xgboost`: one-line evaluation metrics, e.g. `feval=fastauc.lightgbm.ranking("ndcg@10", "mrr@10", "gauc")` or `custom_metric=fastauc.xgboost.binary("roc_auc", "log_loss")`, with exact ties and early-stopping directions; XGBoost's per-query weights are honoured.
- `BinaryAccumulator.all_gather()`: exact merge across `torch.distributed` ranks (DDP, PyTorch Lightning); a copy outside a process group.
- Torch tensors are accepted by every core function, on any device, with or without grad, bfloat16 included (widened exactly to float32). Polars and PyArrow columns (including string query ids) are tested inputs.
- `CITATION.cff`.

## 0.2.0

Everything stays exact: no binning, no sketching, no random tie-breaking, no
hashing that can merge two ids.

- `fastauc.sklearn`: drop-in replacements for `roc_auc_score` (binary, `max_fpr`, multiclass `ovr`/`ovo`, multilabel), `average_precision_score`, `ndcg_score`, `dcg_score`, `roc_curve`, `precision_recall_curve`, `auc`, `log_loss` and `brier_score_loss`, with scikit-learn's values, warnings and error messages.
- `BinaryAccumulator`: exact streaming binary metrics with `update`, `merge` (or `+`), `compute`, `reset`, pickling and `state()`/`from_state()` for distributed gathers. Bit-identical to the one-shot result for unweighted data.
- `paired_test`: paired randomization (sign-flip) test, exact by enumeration for small samples, and paired t-test without scipy. `holm` for Holm-Bonferroni adjusted p-values.
- `roc_auc_multiclass` (one-vs-rest and Hand-Till one-vs-one) and `partial_roc_auc` (McClish-standardized, as scikit-learn's `max_fpr`).
- `ranking_metrics(judgments=...)`: complete relevance judgments (qrels) for incomplete lists, so recall, MAP and ideal DCG count unretrieved relevant items; matches trec_eval. `map_denominator="total"` divides AP@k by the number of relevant items (trec_eval, ranx); the default `"min"` keeps `min(R, k)`.
- String query ids are grouped by an exact hash table in Rust (ids compared byte for byte), 4 to 6 times faster than `np.unique` with identical output.
- Faster ranking kernel: reused per-thread buffers, early exit past the deepest cutoff, partial selection for the ideal DCG. About 2x faster on short lists; on one thread, computing every ranking metric now takes less time than a warm Numba kernel computing NDCG@10 alone.
- Free-threaded CPython 3.14t wheels; concurrent calls from Python threads are tested to match serial results.
- Documentation site.

## 0.1.0

Initial release, published on PyPI as `pyfastauc` (imported as `fastauc`).

- `roc_auc`, `average_precision`, `roc_auc_ap`: single-pair metrics matching scikit-learn, with `sample_weight` and `pos_label`.
- `pr_auc`: trapezoidal area under the precision-recall curve, matching `sklearn.metrics.auc(recall, precision)`, kept distinct from average precision.
- `roc_auc_ap_batch`: many models vs one target, one model vs many targets (single sort), or row-wise pairs, in parallel.
- `roc_auc_ap_grouped`: per-group (per-query, per-user) metrics and GAUC.
- `bootstrap_roc_auc_ap`: sort-once bootstrap with optional stratification, percentile CIs, and paired multi-model comparison.
- `binary_metrics`: ROC-AUC, average precision, trapezoidal PR-AUC, KS, max F1 and calibration in one call.
- `ks_statistic`, `max_f1`, `precision_at_recall`, `recall_at_precision`, `tpr_at_fpr`, `lift_at`.
- `calibration_metrics`, `log_loss`, `brier_score`, `normalized_entropy`, `calibration_ratio`, `expected_calibration_error`.
- `ranking_metrics`: GAUC, NDCG, MRR, MAP, recall, precision and hit rate at several cutoffs, graded relevance, exact tie handling.
- Every metric is invariant to input row order; tied scores are never broken by position.
- Cluster bootstrap: `bootstrap_roc_auc_ap(groups=...)` resamples whole groups; `bootstrap_ranking_metrics` resamples queries, with paired multi-model comparison.
- `sample_weight` in `bootstrap_roc_auc_ap` and `ranking_metrics`.
- `axis` argument for 2-D inputs (`axis=0` for the scikit-learn layout).
- `set_num_threads` / `num_threads`; results are identical for any thread count.
- O(N) fast path for integer group ids.
- Type stub for the native module; LightGBM, XGBoost and PyTorch examples.
- `delong_roc_auc`: DeLong confidence intervals and paired test for ROC-AUC (Sun and Xu's fast algorithm).
- `roc_curve`, `precision_recall_curve`, `calibration_curve`, matching scikit-learn's output.
- Uniform ECE bins now follow scikit-learn's edge convention (a value exactly on an edge goes to the lower bin).
