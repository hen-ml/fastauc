# Binary and multiclass metrics

## One model, everything at once

```python
fastauc.binary_metrics(y, p)
# {'roc_auc', 'average_precision', 'pr_auc', 'ks', 'ks_threshold', 'f1_max', 'f1_threshold',
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
fastauc.pr_auc(y, s)                    # == sklearn.metrics.auc(recall, precision)
fastauc.partial_roc_auc(y, s, 0.1)      # == roc_auc_score(..., max_fpr=0.1), McClish-standardized
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

## Average precision or PR-AUC?

Both summarise the same precision-recall curve, one point per distinct score,
and they are not the same number:

| | formula | scikit-learn equivalent |
|---|---|---|
| `average_precision` | $\sum_k (R_k - R_{k-1}) P_k$ | `average_precision_score` |
| `pr_auc` | $\sum_k (R_k - R_{k-1}) (P_k + P_{k-1}) / 2$, from $(R, P) = (0, 1)$ | `auc(recall, precision)` |

Average precision credits each step in recall with the precision actually
reached there. The trapezoidal rule draws straight lines between points,
which no classifier can achieve in precision-recall space (Davis and
Goadrich, 2006), and it credits the starting point (0, 1) even when the
top-scored item is a negative. With three items scored 0.9, 0.8, 0.7 and
labels 1, 0, 1, average precision is 0.833 and PR-AUC is 0.792.

Report average precision. Use `pr_auc` only to reproduce numbers from a
pipeline that computes "PR-AUC" with the trapezoidal rule.

## Curves

```python
fpr, tpr, thresholds = fastauc.roc_curve(y, s)
precision, recall, thresholds = fastauc.precision_recall_curve(y, s)
prob_true, prob_pred = fastauc.calibration_curve(y, p, n_bins=10)
```

Same arrays as scikit-learn, including `drop_intermediate` and
`sample_weight`. `calibration_curve` also takes `return_weights=True`, and
its `strategy="quantile"` makes truly equal-weight bins even with tied
probabilities.

## Multiclass ROC-AUC

```python
fastauc.roc_auc_multiclass(y, P)                       # one-vs-rest, macro average
fastauc.roc_auc_multiclass(y, P, average="weighted")   # by class prevalence
fastauc.roc_auc_multiclass(y, P, average=None)         # one value per class
fastauc.roc_auc_multiclass(y, P, multi_class="ovo")    # Hand and Till (2001)
```

`P` has shape `(n_samples, n_classes)`. One-vs-rest evaluates all classes in
one parallel call. Unlike scikit-learn, scores do not have to sum to one
(use the `fastauc.sklearn` drop-in for scikit-learn's exact checks).

## Many models or many targets

```python
S = rng.random((1000, 100_000))            # 1000 models, one row each
roc, ap = fastauc.roc_auc_ap_batch(y, S)   # arrays of shape (1000,)

Y = rng.integers(0, 2, (1000, 100_000))    # 1000 targets, e.g. label permutations
roc, ap = fastauc.roc_auc_ap_batch(Y, s)   # sorts s once, then O(N) per target
```

2-D inputs are `(n_tasks, n_samples)` by default; pass `axis=0` for the
scikit-learn layout.

## AUC within groups

```python
res = fastauc.roc_auc_ap_grouped(y, s, user_id)
res.groups, res.roc_auc, res.average_precision, res.n_pos, res.n_neg
res.mean("roc_auc")                      # size-weighted GAUC
res.mean("roc_auc", weight="uniform")    # macro average over groups
```

## Undefined cases

Undefined metrics return `NaN` instead of raising, so one degenerate slice
never aborts a batch, a group-by or a bootstrap: ROC-AUC needs both classes,
average precision needs a positive. `NaN` scores raise `ValueError`;
`±inf` scores are ranked normally.
