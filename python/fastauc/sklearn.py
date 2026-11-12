"""Drop-in replacements for ``sklearn.metrics``.

Change one import and keep your code::

    from sklearn import metrics           # before
    from fastauc import sklearn as metrics  # after

Each function takes the same arguments as its scikit-learn namesake (as of
scikit-learn 1.9) and returns the same value, including for multiclass and
multilabel targets, sample weights, and the cases scikit-learn warns about.
The computation runs in fastauc's multithreaded core, so it is faster, and
tied scores are always handled exactly.

Differences, all deliberate:

* ``ndcg_score`` and ``dcg_score`` accept ``ignore_ties`` but always average
  ties exactly. With ``ignore_ties=True`` scikit-learn breaks ties in an
  arbitrary order, so its result then depends on that order.
* scikit-learn is not needed; when it is installed, warnings use its
  ``UndefinedMetricWarning`` class so existing warning filters keep working.
"""
from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

from . import (
    _core,
    average_precision,
    partial_roc_auc,
    roc_auc,
    roc_auc_multiclass,
)
from . import precision_recall_curve as _pr_curve
from . import ranking_metrics as _ranking_metrics
from . import roc_curve as _roc_curve

try:  # keep users' warning filters working when scikit-learn is installed
    from sklearn.exceptions import UndefinedMetricWarning
except ImportError:  # pragma: no cover - exercised only without scikit-learn

    class UndefinedMetricWarning(UserWarning):
        """A metric is undefined for the given input (mirrors scikit-learn's)."""


__all__ = [
    "auc",
    "average_precision_score",
    "brier_score_loss",
    "dcg_score",
    "log_loss",
    "ndcg_score",
    "precision_recall_curve",
    "roc_auc_score",
    "roc_curve",
    "UndefinedMetricWarning",
]

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Target types
# ---------------------------------------------------------------------------


def _target_type(y: np.ndarray) -> str:
    """'binary', 'multiclass' or 'multilabel-indicator', as scikit-learn names them."""
    if y.ndim == 2 and y.shape[1] > 1:
        if y.dtype.kind in "biuf" and np.isin(y, (0, 1)).all():
            return "multilabel-indicator"
        raise ValueError("2-D y_true must be a binary indicator matrix")
    if y.ndim == 2:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"y_true has unsupported shape {y.shape}")
    if y.dtype.kind == "f" and y.size and np.any(y != np.round(y)):
        raise ValueError("continuous format is not supported")
    return "multiclass" if len(np.unique(y)) > 2 else "binary"


def _one_d(y_score: np.ndarray) -> np.ndarray:
    if y_score.ndim == 2 and y_score.shape[1] == 1:
        return y_score.ravel()
    if y_score.ndim != 1:
        raise ValueError(f"y should be a 1d array, got an array of shape {y_score.shape} instead.")
    return y_score


def _default_pos_label(y: np.ndarray, pos_label: Any) -> Any:
    """scikit-learn's rule: pos_label may be omitted only for {0, 1} or {-1, 1} labels."""
    if pos_label is not None:
        return pos_label
    present = np.unique(y)
    if present.dtype.kind in "biuf" and (
        np.isin(present, (0, 1)).all() or np.isin(present, (-1, 1)).all()
    ):
        return 1
    raise ValueError(
        f"y_true takes value in {set(present.tolist())} and pos_label is not specified: either "
        "make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly."
    )


def _check_option(func: str, name: str, value, allowed) -> None:
    if value not in allowed:
        shown = ", ".join(repr(a) for a in allowed if a is not None)
        extra = " or None" if None in allowed else ""
        raise ValueError(f"The {name!r} parameter of {func} must be a str among "
                         f"{{{shown}}}{extra}. Got {value!r} instead.")


_AVERAGES = ("macro", "micro", "weighted", "samples", None)


def _average(values: np.ndarray, weights: Optional[np.ndarray], average: Optional[str]):
    if average is None:
        return values
    if average == "weighted":
        if weights is None or float(np.sum(weights)) == 0.0:
            return 0.0
        return float(np.average(values, weights=weights))
    return float(np.average(values))


# ---------------------------------------------------------------------------
# ROC-AUC and average precision
# ---------------------------------------------------------------------------


def roc_auc_score(
    y_true,
    y_score,
    *,
    average: Optional[str] = "macro",
    sample_weight=None,
    max_fpr: Optional[float] = None,
    multi_class: str = "raise",
    labels=None,
):
    """Same as ``sklearn.metrics.roc_auc_score``."""
    _check_option("roc_auc_score", "average", average, _AVERAGES)
    _check_option("roc_auc_score", "multi_class", multi_class, ("raise", "ovr", "ovo"))
    yt, ys = np.asarray(y_true), np.asarray(y_score)
    kind = _target_type(yt)
    if kind == "binary" and ys.ndim == 2 and ys.shape[1] > 2:
        kind = "multiclass"
    if kind == "multiclass":
        if max_fpr is not None and max_fpr != 1:
            raise ValueError(
                "Partial AUC computation not available in multiclass setting, 'max_fpr' "
                f"must be set to `None`, received `max_fpr={max_fpr}` instead"
            )
        if multi_class == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        if ys.ndim != 2:
            raise ValueError(
                "`y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` "
                f"contains multiple classes. Got `y_score.shape={ys.shape}`."
            )
        if not np.allclose(1, ys.sum(axis=1)):
            raise ValueError(
                "Target scores need to be probabilities for multiclass roc_auc, i.e. "
                "they should sum up to 1.0 over classes"
            )
        if labels is not None:
            lab = np.asarray(labels)
            if not np.array_equal(np.unique(lab), lab):
                raise ValueError("Parameter 'labels' must be unique and ordered")
        if multi_class == "ovo" and average is None:
            raise NotImplementedError("average=None is not implemented for multi_class='ovo'.")
        return roc_auc_multiclass(yt, ys, multi_class=multi_class, average=average,
                                  labels=labels, sample_weight=sample_weight)

    def binary(y, s, w):
        present = np.unique(y)
        if len(present) < 2:
            warnings.warn("Only one class is present in y_true. ROC AUC score is not defined "
                          "in that case.", UndefinedMetricWarning, stacklevel=3)
            return _NAN
        pos = present[-1]
        if max_fpr is None or max_fpr == 1:
            return roc_auc(y, s, sample_weight=w, pos_label=pos)
        return partial_roc_auc(y, s, max_fpr, sample_weight=w, pos_label=pos)

    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if kind == "binary":
        return binary(yt.ravel(), _one_d(ys), w)
    fast = "roc" if max_fpr is None or max_fpr == 1 else None
    return _multilabel(binary, yt, ys, w, average, fast)


def _multilabel(binary, Y: np.ndarray, S: np.ndarray, w, average, fast: Optional[str] = None):
    """Average a binary metric over the columns (or rows, for 'samples') of an
    indicator matrix. ``fast`` ('roc' or 'ap') evaluates every column in one
    parallel call; columns where the metric is undefined go through ``binary``
    so they warn (and score) exactly as scikit-learn does."""
    if S.shape != Y.shape:
        raise ValueError(f"y_true {Y.shape} and y_score {S.shape} have different shapes")
    if average == "micro":
        ww = None if w is None else np.repeat(w, Y.shape[1])
        return binary(Y.ravel(), S.ravel(), ww)
    by_row = average == "samples"
    if fast is not None:
        A, B = (Y, S) if by_row else (Y.T, S.T)
        roc, ap = _core.pairwise(np.ascontiguousarray(A, dtype=np.uint8),
                                 np.ascontiguousarray(B, dtype=np.float64),
                                 None if by_row else w)
        vals = roc if fast == "roc" else ap
        for j in np.flatnonzero(np.isnan(vals)):
            vals[j] = binary(A[j], B[j], None if by_row else w)
    elif by_row:
        vals = np.array([binary(Y[i], S[i], None) for i in range(len(Y))])
    else:
        vals = np.array([binary(Y[:, j], S[:, j], w) for j in range(Y.shape[1])])
    if by_row:
        return float(np.average(vals, weights=w))
    weights = None
    if average == "weighted":
        weights = Y.sum(axis=0).astype(np.float64) if w is None else w @ Y
    return _average(vals, weights, average)


def average_precision_score(
    y_true,
    y_score,
    *,
    average: Optional[str] = "macro",
    pos_label: Any = 1,
    sample_weight=None,
):
    """Same as ``sklearn.metrics.average_precision_score``."""
    _check_option("average_precision_score", "average", average, _AVERAGES)
    yt, ys = np.asarray(y_true), np.asarray(y_score)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    kind = _target_type(yt)

    def binary(y, s, ww):
        y = np.asarray(y, dtype=bool) if y.dtype.kind == "b" else y
        if not np.any(y == pos_label):
            warnings.warn("No positive class found in y_true, recall is set to be zero, which "
                          "makes average precision 0.", UserWarning, stacklevel=3)
            return 0.0
        return average_precision(y, s, sample_weight=ww, pos_label=pos_label)

    if kind == "binary":
        present = np.unique(yt)
        if len(present) == 2 and pos_label not in present.tolist():
            raise ValueError(f"pos_label={pos_label} is not a valid label. It should be one of {present}")
        return binary(yt.ravel(), _one_d(ys), w)
    if kind == "multiclass":
        if pos_label != 1:
            raise ValueError("Parameter pos_label is fixed to 1 for multiclass y_true. Do not set "
                             "pos_label or set pos_label to 1.")
        classes = np.unique(yt)
        if ys.ndim != 2 or ys.shape[1] != len(classes):
            raise ValueError("y_score must have one column per class in y_true")
        yt = (yt[:, None] == classes[None, :]).astype(np.int64)
    pos_label = 1
    return _multilabel(binary, yt, ys, w, average, "ap")


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True):
    """Same as ``sklearn.metrics.roc_curve``."""
    yt = np.asarray(y_true)
    pos = _default_pos_label(yt, pos_label)
    fpr, tpr, thr = _roc_curve(yt, y_score, sample_weight=sample_weight, pos_label=pos,
                               drop_intermediate=drop_intermediate)
    if np.isnan(fpr).all() and len(fpr):
        warnings.warn("No negative samples in y_true, false positive value should be "
                      "meaningless", UndefinedMetricWarning, stacklevel=2)
    if np.isnan(tpr).all() and len(tpr):
        warnings.warn("No positive samples in y_true, true positive value should be "
                      "meaningless", UndefinedMetricWarning, stacklevel=2)
    return fpr, tpr, thr


def precision_recall_curve(y_true, y_score, *, pos_label=None, sample_weight=None,
                           drop_intermediate=False):
    """Same as ``sklearn.metrics.precision_recall_curve``."""
    yt = np.asarray(y_true)
    pos = _default_pos_label(yt, pos_label)
    precision, recall, thr = _pr_curve(yt, y_score, sample_weight=sample_weight, pos_label=pos,
                                       drop_intermediate=drop_intermediate)
    if np.isnan(recall).any():
        warnings.warn("No positive class found in y_true, recall is set to be one for all "
                      "thresholds.", UserWarning, stacklevel=2)
        recall = np.where(np.isnan(recall), 1.0, recall)
    return precision, recall, thr


def auc(x, y) -> float:
    """Same as ``sklearn.metrics.auc``: trapezoidal area under any curve."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1-D arrays of the same length")
    if len(x) < 2:
        raise ValueError(f"At least 2 points are needed to compute area under curve, but "
                         f"x.shape = {len(x)}")
    dx = np.diff(x)
    direction = 1.0
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1.0
        else:
            raise ValueError(f"x is neither increasing nor decreasing : {x}.")
    return float(direction * np.sum(dx * (y[1:] + y[:-1]) / 2.0))


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _ranking_inputs(y_true, y_score, sample_weight):
    G, S = np.asarray(y_true), np.asarray(y_score)
    if G.ndim != 2:
        raise ValueError(
            "Only ('multilabel-indicator', 'continuous-multioutput', 'multiclass-multioutput') "
            "formats are supported. Got binary instead"
        )
    if G.shape != S.shape:
        raise ValueError(f"Input contains different shapes: {G.shape} and {S.shape}")
    if G.shape[1] < 2:
        raise ValueError("Computing NDCG is only meaningful when there is more than 1 document. "
                         f"Got {G.shape[1]} instead.")
    if G.size and G.min() < 0:
        raise ValueError("ndcg_score should not be used on negative y_true values.")
    n, m = G.shape
    w = None if sample_weight is None else np.repeat(np.asarray(sample_weight, dtype=np.float64), m)
    return G.astype(np.float64), S, n, m, w


def ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False) -> float:
    """Same as ``sklearn.metrics.ndcg_score`` (ties are always averaged exactly)."""
    G, S, n, m, w = _ranking_inputs(y_true, y_score, sample_weight)
    out = _ranking_metrics(G.ravel(), S.ravel(), group_sizes=np.full(n, m), ks=(k,),
                           gain="linear", no_relevant="zero", sample_weight=w)
    return out["ndcg" if k is None else f"ndcg@{k}"]


def dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None,
              ignore_ties=False) -> float:
    """Same as ``sklearn.metrics.dcg_score`` (ties are always averaged exactly)."""
    G, S, n, m, _ = _ranking_inputs(y_true, y_score, None)
    key = "ndcg" if k is None else f"ndcg@{k}"
    _, d = _ranking_metrics(G.ravel(), S.ravel(), group_sizes=np.full(n, m), ks=(k,),
                            gain="linear", per_group=True)
    cut = m if k is None else min(k, m)
    ideal = (-np.sort(-G, axis=1)[:, :cut] / np.log2(np.arange(2, cut + 2))).sum(axis=1)
    # NDCG x ideal DCG; a row with nothing relevant has DCG 0.
    dcg = np.where(ideal > 0, np.nan_to_num(d[key]) * ideal, 0.0)
    # 1 / log_b(i) = log2(b) / log2(i), so any base is a rescaled base-2 DCG.
    dcg = dcg * np.log2(log_base)
    return float(np.average(dcg, weights=sample_weight))


# ---------------------------------------------------------------------------
# Probabilistic losses
# ---------------------------------------------------------------------------


def _renamed(new, old, new_name: str, old_name: str):
    if old is not None:
        if new is not None:
            raise ValueError(f"Pass {new_name} only; {old_name} is its old name.")
        return old
    if new is None:
        raise TypeError(f"missing required argument: {new_name!r}")
    return new


def log_loss(y_true, y_proba=None, *, normalize=True, sample_weight=None, labels=None,
             y_pred=None, eps="auto") -> float:
    """Same as ``sklearn.metrics.log_loss``.

    ``y_pred`` (the argument's old name) and ``eps`` (removed from
    scikit-learn in 1.5) are still accepted, so older code runs unchanged.
    """
    y_proba = _renamed(y_proba, y_pred, "y_proba", "y_pred")
    yt = np.asarray(y_true)
    P = np.asarray(y_proba, dtype=np.float64)
    classes = np.unique(yt if labels is None else np.asarray(labels))
    if len(classes) == 1:
        raise ValueError(f"y_true contains only one label ({classes[0]}). Please provide the list "
                         "of all expected class labels explicitly through the labels argument.")
    if P.ndim == 1 or (P.ndim == 2 and P.shape[1] == 1):
        P = P.reshape(-1)
        P = np.column_stack([1.0 - P, P])
    if P.shape != (len(yt), len(classes)):
        raise ValueError(f"y_true and y_proba contain different numbers of classes: "
                         f"{len(classes)} vs {P.shape[1]}")
    idx = np.searchsorted(classes, yt)
    if np.any(idx >= len(classes)) or np.any(classes[np.minimum(idx, len(classes) - 1)] != yt):
        raise ValueError("y_true contains values not in labels")
    if not np.allclose(P.sum(axis=1), 1.0):
        warnings.warn("The y_proba values do not sum to one. Make sure to pass probabilities.",
                      UserWarning, stacklevel=2)
    if eps == "auto":
        eps = np.finfo(P.dtype).eps
    loss = -np.log(np.clip(P[np.arange(len(yt)), idx], eps, 1.0 - eps))
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if normalize:
        return float(np.average(loss, weights=w))
    return float(np.sum(loss if w is None else loss * w))


def brier_score_loss(y_true, y_proba=None, *, sample_weight=None, pos_label=None, labels=None,
                     scale_by_half="auto", y_prob=None) -> float:
    """Same as ``sklearn.metrics.brier_score_loss``.

    ``y_prob`` (the argument's old name) is still accepted.
    """
    y_proba = _renamed(y_proba, y_prob, "y_proba", "y_prob")
    yt = np.asarray(y_true)
    P = np.asarray(y_proba, dtype=np.float64)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if P.ndim == 1 or (P.ndim == 2 and P.shape[1] == 1):
        pos = _default_pos_label(yt, pos_label)
        y = (yt == pos).astype(np.float64)
        sq = (P.reshape(-1) - y) ** 2
        return float(np.average(sq, weights=w))
    classes = np.unique(yt if labels is None else np.asarray(labels))
    if P.shape[1] != len(classes):
        raise ValueError("y_proba must have one column per class")
    Y = (yt[:, None] == classes[None, :]).astype(np.float64)
    half = (P.shape[1] == 2) if scale_by_half == "auto" else bool(scale_by_half)
    sq = ((P - Y) ** 2).sum(axis=1) * (0.5 if half else 1.0)
    return float(np.average(sq, weights=w))

