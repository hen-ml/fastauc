# scikit-learn drop-in

Change one import and keep your code:

```python
from sklearn import metrics             # before
from fastauc import sklearn as metrics  # after

metrics.roc_auc_score(y, p)
metrics.roc_auc_score(y_multi, P, multi_class="ovr", average="weighted")
metrics.average_precision_score(Y_multilabel, S, average="samples")
metrics.ndcg_score(relevance_matrix, score_matrix, k=10)
```

Each function takes the same arguments as its scikit-learn namesake (as of
scikit-learn 1.9) and returns the same value:

| Function | Covers |
|---|---|
| `roc_auc_score` | binary, `max_fpr` (McClish), multiclass `ovr` and `ovo`, multilabel with `average` = `macro`, `weighted`, `micro`, `samples`, `None`, `labels`, `sample_weight` |
| `average_precision_score` | binary with `pos_label`, multiclass, multilabel, all averages |
| `roc_curve`, `precision_recall_curve` | `pos_label`, `sample_weight`, `drop_intermediate` |
| `ndcg_score`, `dcg_score` | `k`, `log_base`, `sample_weight`, rows with no relevant item |
| `auc` | increasing or decreasing `x` |
| `log_loss` | binary and multiclass, `labels`, `normalize`, the warning for rows that do not sum to one (no renormalisation, as in scikit-learn 1.5+) |
| `brier_score_loss` | `pos_label`, 2-D probabilities, `scale_by_half` |

The test suite compares every function with scikit-learn on tied scores,
weights, string labels, `{-1, 1}` labels, and the degenerate cases: the
same values, the same warnings (`UndefinedMetricWarning` is scikit-learn's
own class when scikit-learn is installed, so warning filters keep working)
and the same error messages.

## Deliberate differences

- `ndcg_score` and `dcg_score` accept `ignore_ties` but always average ties
  exactly. With `ignore_ties=True` scikit-learn keeps the input order inside
  tied blocks, so its result depends on that order; fastauc's does not.
- `log_loss(y_pred=...)` and `brier_score_loss(y_prob=...)`, the arguments'
  old names, still work, as does `log_loss(eps=...)`, which scikit-learn
  removed in 1.5. Old code runs unchanged.
- scikit-learn is not a dependency.

## Speed

See [Performance](performance.md#against-scikit-learn). The gain is largest
where scikit-learn loops in Python (NDCG over many queries, one-vs-one
multiclass, multilabel) and smallest for a single unweighted ROC-AUC.
