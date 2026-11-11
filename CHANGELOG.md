# Changelog

## 0.1.0

Initial release.

- `roc_auc`, `average_precision`, `roc_pr_auc`: single-pair metrics matching scikit-learn, with `sample_weight` and `pos_label`.
- `roc_pr_auc_batch`: many models vs one target, one model vs many targets (single sort), or row-wise pairs, in parallel.
- `roc_pr_auc_grouped`: per-group (per-query, per-user) metrics and GAUC.
- `bootstrap_roc_pr_auc`: sort-once bootstrap with optional stratification, percentile CIs, and paired multi-model comparison.
- `binary_metrics`: ROC-AUC, average precision, KS, max F1 and calibration in one call.
- `ks_statistic`, `max_f1`, `precision_at_recall`, `recall_at_precision`, `tpr_at_fpr`, `lift_at`.
- `calibration_metrics`, `log_loss`, `brier_score`, `normalized_entropy`, `calibration_ratio`, `expected_calibration_error`.
- `ranking_metrics`: GAUC, NDCG, MRR, MAP, recall, precision and hit rate at several cutoffs, graded relevance, exact tie handling.
- Every metric is invariant to input row order; tied scores are never broken by position.
- Cluster bootstrap: `bootstrap_roc_pr_auc(groups=...)` resamples whole groups; `bootstrap_ranking_metrics` resamples queries, with paired multi-model comparison.
- `sample_weight` in `bootstrap_roc_pr_auc` and `ranking_metrics`.
- `axis` argument for 2-D inputs (`axis=0` for the scikit-learn layout).
- `set_num_threads` / `num_threads`; results are identical for any thread count.
- O(N) fast path for integer group ids.
- Type stub for the native module; LightGBM, XGBoost and PyTorch examples.
- `delong_roc_auc`: DeLong confidence intervals and paired test for ROC-AUC (Sun and Xu's fast algorithm).
- `roc_curve`, `precision_recall_curve`, `calibration_curve`, matching scikit-learn's output.
- Uniform ECE bins now follow scikit-learn's edge convention (a value exactly on an edge goes to the lower bin).
