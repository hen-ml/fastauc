# Changelog

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
