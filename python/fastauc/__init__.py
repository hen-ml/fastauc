"""fastauc: fast, multithreaded ROC-AUC and PR-AUC (average precision).

Results match ``sklearn.metrics.roc_auc_score`` and
``sklearn.metrics.average_precision_score`` for binary targets, including
tied scores and sample weights.

Conventions
-----------
* Labels are binary. Pass ``bool`` or ``{0, 1}`` arrays, or name the positive
  class with ``pos_label``.
* Batched inputs are laid out as ``(n_tasks, n_samples)``: one **row** per
  model / target. (scikit-learn uses columns; pass ``arr.T`` if needed.)
* Undefined metrics are returned as ``NaN`` instead of raising, so one
  degenerate row never kills a batch: ROC-AUC needs both classes, average
  precision needs at least one positive.
* ``NaN`` scores raise ``ValueError``.
* A threshold ``t`` always means "predict positive when ``score >= t``".
* No result depends on the order of the input rows. Samples with equal
  scores are treated as interchangeable everywhere: threshold metrics only
  cut between distinct scores, and top-k ranking metrics are averaged over
  all orderings of the tied items (exactly, in closed form, with no random
  tie-breaking).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from . import _core

__version__ = "0.1.0"

__all__ = [
    "roc_auc",
    "average_precision",
    "roc_pr_auc",
    "roc_pr_auc_batch",
    "roc_pr_auc_grouped",
    "bootstrap_roc_pr_auc",
    "binary_metrics",
    "ks_statistic",
    "max_f1",
    "precision_at_recall",
    "recall_at_precision",
    "tpr_at_fpr",
    "lift_at",
    "roc_curve",
    "precision_recall_curve",
    "calibration_curve",
    "delong_roc_auc",
    "DeLongResult",
    "calibration_metrics",
    "log_loss",
    "brier_score",
    "normalized_entropy",
    "calibration_ratio",
    "expected_calibration_error",
    "ranking_metrics",
    "bootstrap_ranking_metrics",
    "RankingBootstrapResult",
    "set_num_threads",
    "BootstrapResult",
    "GroupedResult",
    "num_threads",
    "__version__",
]

ArrayLike = Any


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _labels(y_true: ArrayLike, pos_label: Any = None) -> np.ndarray:
    a = np.asarray(y_true)
    if pos_label is not None:
        return np.ascontiguousarray(a == pos_label, dtype=np.uint8)
    if a.dtype == np.bool_:
        return np.ascontiguousarray(a).view(np.uint8)
    if a.dtype.kind not in "iuf":
        raise TypeError(
            f"y_true has dtype {a.dtype}; pass bool or 0/1 values, or set pos_label"
        )
    if a.size and not ((a == 0) | (a == 1)).all():
        raise ValueError(
            "y_true must contain only 0 and 1 (or bool); set pos_label to "
            "choose the positive class for other label encodings"
        )
    return np.ascontiguousarray(a, dtype=np.uint8)


def _scores(y_score: ArrayLike) -> np.ndarray:
    a = np.asarray(y_score)
    if a.dtype.kind not in "iufb":
        raise TypeError(f"y_score has non-numeric dtype {a.dtype}")
    return np.ascontiguousarray(a, dtype=np.float64)


def _weights(sample_weight: Optional[ArrayLike]) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    w = np.ascontiguousarray(sample_weight, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("sample_weight must be 1-D")
    return w


def _require_1d(name: str, a: np.ndarray) -> None:
    if a.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got shape {a.shape}; "
            "use roc_pr_auc_batch for 2-D inputs"
        )


def _samples_last(a: np.ndarray, axis: int) -> np.ndarray:
    """Return 2-D ``a`` as C-contiguous (n_tasks, n_samples)."""
    if axis not in (0, 1, -1, -2):
        raise ValueError("axis must be 0 or 1 (-1)")
    if a.ndim == 2 and axis in (0, -2):
        return np.ascontiguousarray(a.T)
    return a


def _group_codes(groups: ArrayLike, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Map group ids to dense codes. Returns (sorted unique ids, int64 codes)."""
    g = np.asarray(groups)
    _require_1d("groups", g)
    if len(g) != n:
        raise ValueError(f"groups has length {len(g)}, expected {n}")
    if g.dtype.kind in "iu" and g.size:
        # Integer ids in a compact range: O(N), no sort.
        lo, hi = int(g.min()), int(g.max())
        width = hi - lo + 1
        if width <= 2 * g.size + 1024:
            raw = (g - lo).astype(np.int64)
            present = np.bincount(raw, minlength=width) > 0
            if present.all():
                return np.arange(lo, hi + 1).astype(g.dtype), np.ascontiguousarray(raw)
            remap = np.cumsum(present) - 1
            ids = (np.flatnonzero(present) + lo).astype(g.dtype)
            return ids, np.ascontiguousarray(remap[raw], dtype=np.int64)
    ids, codes = np.unique(g, return_inverse=True)
    return ids, np.ascontiguousarray(codes.reshape(-1), dtype=np.int64)


def _percentile_ci(s: np.ndarray, level: float) -> Any:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.nanpercentile(s, [tail, 100.0 - tail], axis=-1)
    if s.ndim == 1:
        return float(lo), float(hi)
    return np.stack([lo, hi], axis=-1)


def _paired_compare(s: np.ndarray, est: Any, a: int, b: int, level: float):
    if s.ndim != 2:
        raise ValueError("compare() needs a result computed from 2-D scores")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    est = np.asarray(est)
    d = s[a] - s[b]
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float(est[a] - est[b]), (float("nan"), float("nan")), float("nan")
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(d, [tail, 100.0 - tail])
    # Ties at zero count half; +1 smoothing keeps p > 0.
    below = (d < 0).sum() + 0.5 * (d == 0).sum()
    p = 2.0 * min(below + 1.0, d.size - below + 1.0) / (d.size + 1.0)
    return float(est[a] - est[b]), (float(lo), float(hi)), float(min(p, 1.0))


# ---------------------------------------------------------------------------
# Single pair
# ---------------------------------------------------------------------------


def roc_pr_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> Tuple[float, float]:
    """ROC-AUC and average precision from a single sort.

    Returns
    -------
    (roc_auc, average_precision) : tuple of float
    """
    yt, ys = _labels(y_true, pos_label), _scores(y_score)
    _require_1d("y_true", yt)
    _require_1d("y_score", ys)
    return _core.roc_pr_auc(yt, ys, _weights(sample_weight))


def roc_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> float:
    """Area under the ROC curve. Equivalent to ``sklearn.metrics.roc_auc_score``."""
    return roc_pr_auc(y_true, y_score, sample_weight=sample_weight, pos_label=pos_label)[0]


def average_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> float:
    """Average precision (step-wise PR-AUC).

    Equivalent to ``sklearn.metrics.average_precision_score``.
    """
    return roc_pr_auc(y_true, y_score, sample_weight=sample_weight, pos_label=pos_label)[1]


# ---------------------------------------------------------------------------
# Batched
# ---------------------------------------------------------------------------


def roc_pr_auc_batch(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    axis: int = -1,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate many (labels, scores) pairs in parallel.

    The shapes decide what is computed:

    ======================  ======================  =================================
    ``y_true``              ``y_score``             meaning
    ======================  ======================  =================================
    ``(n_samples,)``        ``(m, n_samples)``      m models, one target
    ``(m, n_samples)``      ``(n_samples,)``        one model, m targets (sorts once)
    ``(m, n_samples)``      ``(m, n_samples)``      row i against row i
    ======================  ======================  =================================

    ``axis`` is the axis of 2-D inputs that runs over samples. The default
    ``-1`` is the layout above; pass ``axis=0`` for the scikit-learn layout
    ``(n_samples, m)``.

    ``sample_weight``, if given, is ``(n_samples,)`` and shared by all tasks.

    Returns
    -------
    (roc_auc, average_precision) : tuple of ndarray, each of shape ``(m,)``
    """
    yt, ys = _labels(y_true, pos_label), _scores(y_score)
    yt, ys = _samples_last(yt, axis), _samples_last(ys, axis)
    w = _weights(sample_weight)
    if yt.ndim > 2 or ys.ndim > 2 or (yt.ndim == 1 and ys.ndim == 1):
        raise ValueError(
            f"unsupported shapes: y_true {yt.shape}, y_score {ys.shape}; "
            "at least one must be 2-D and neither more than 2-D"
        )
    if yt.shape[-1] != ys.shape[-1]:
        other = 0 if axis in (1, -1) else -1
        raise ValueError(
            f"length mismatch along the sample axis: y_true {np.shape(y_true)}, "
            f"y_score {np.shape(y_score)}. If samples run along the other axis, "
            f"pass axis={other}"
        )
    if yt.ndim == 1:
        return _core.many_scores(yt, ys, w)
    if ys.ndim == 1:
        return _core.many_targets(yt, ys, w)
    return _core.pairwise(yt, ys, w)


# ---------------------------------------------------------------------------
# Grouped
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupedResult:
    """Per-group metrics from :func:`roc_pr_auc_grouped`.

    All arrays are aligned with ``groups`` (the sorted unique group ids).
    """

    groups: np.ndarray
    roc_auc: np.ndarray
    average_precision: np.ndarray
    n_pos: np.ndarray
    n_neg: np.ndarray

    def mean(self, metric: str = "roc_auc", *, weight: str = "size") -> float:
        """Average a metric over groups where it is defined.

        Parameters
        ----------
        metric : ``"roc_auc"`` or ``"average_precision"``
        weight : ``"size"`` weights each group by its sample count (the usual
            "GAUC"); ``"uniform"`` gives every group equal weight.
        """
        if metric not in ("roc_auc", "average_precision"):
            raise ValueError("metric must be 'roc_auc' or 'average_precision'")
        values = getattr(self, metric)
        if weight == "size":
            w = (self.n_pos + self.n_neg).astype(np.float64)
        elif weight == "uniform":
            w = np.ones(len(values))
        else:
            raise ValueError("weight must be 'size' or 'uniform'")
        ok = ~np.isnan(values)
        if not ok.any():
            return float("nan")
        return float(np.average(values[ok], weights=w[ok]))


def roc_pr_auc_grouped(
    y_true: ArrayLike,
    y_score: ArrayLike,
    groups: ArrayLike,
    *,
    pos_label: Any = None,
) -> GroupedResult:
    """ROC-AUC and average precision within each group, in parallel.

    Typical use: per-query or per-user AUC for ranking and recommendation
    models. ``groups`` is any 1-D array of ids (ints, strings, ...); samples
    need not be sorted by group.
    """
    yt, ys = _labels(y_true, pos_label), _scores(y_score)
    _require_1d("y_true", yt)
    _require_1d("y_score", ys)
    uniq, codes = _group_codes(groups, len(yt))
    roc, ap = _core.grouped(yt, ys, codes, len(uniq))
    n_pos = np.bincount(codes, weights=yt, minlength=len(uniq)).astype(np.int64)
    n_all = np.bincount(codes, minlength=len(uniq)).astype(np.int64)
    return GroupedResult(uniq, roc, ap, n_pos, n_all - n_pos)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distribution from :func:`bootstrap_roc_pr_auc`.

    Attributes
    ----------
    roc_auc, average_precision : ndarray
        Replicates. Shape ``(n_bootstraps,)`` for one model, or
        ``(n_models, n_bootstraps)`` when 2-D scores were passed. Column ``b``
        uses the same resample for every model, so differences between rows
        are paired.
    roc_auc_estimate, average_precision_estimate : float or ndarray
        The metrics on the original (un-resampled) data.
    """

    roc_auc: np.ndarray
    average_precision: np.ndarray
    roc_auc_estimate: Any
    average_precision_estimate: Any

    def _samples(self, metric: str) -> np.ndarray:
        if metric not in ("roc_auc", "average_precision"):
            raise ValueError("metric must be 'roc_auc' or 'average_precision'")
        return getattr(self, metric)

    def ci(self, metric: str = "roc_auc", level: float = 0.95) -> Any:
        """Percentile confidence interval.

        Returns ``(low, high)`` floats for one model, or an array of shape
        ``(n_models, 2)``. Replicates where the metric is undefined are ignored.
        """
        return _percentile_ci(self._samples(metric), level)

    def std(self, metric: str = "roc_auc") -> Any:
        """Bootstrap standard error."""
        s = np.nanstd(self._samples(metric), axis=-1, ddof=1)
        return float(s) if np.ndim(s) == 0 else s

    def compare(self, a: int, b: int, metric: str = "roc_auc", level: float = 0.95):
        """Paired comparison of model ``a`` minus model ``b``.

        Returns ``(difference, (low, high), p_value)`` where the interval is a
        percentile CI of the paired difference and ``p_value`` is a two-sided
        bootstrap p-value for "no difference".
        """
        return _paired_compare(
            self._samples(metric), getattr(self, metric + "_estimate"), a, b, level
        )


def bootstrap_roc_pr_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    n_bootstraps: int = 1000,
    seed: int = 0,
    stratified: bool = False,
    groups: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    axis: int = -1,
    pos_label: Any = None,
) -> BootstrapResult:
    """Bootstrap ROC-AUC and average precision without re-sorting.

    Scores are sorted once; every replicate is then a linear scan with
    resampling multiplicities as weights, run across all cores.

    Parameters
    ----------
    y_true : (n_samples,)
    y_score : (n_samples,) or (n_models, n_samples)
        With 2-D scores, every replicate uses the same resample for all
        models, which is what you want for comparing them (see
        :meth:`BootstrapResult.compare`). Pass ``axis=0`` if the array is
        ``(n_samples, n_models)``.
    n_bootstraps : number of replicates.
    seed : results are reproducible for a given seed, independent of the
        number of threads.
    stratified : resample positives and negatives separately, keeping the
        class balance fixed. Recommended for rare positives, where plain
        resampling can produce replicates with no positives (``NaN``).
    groups : (n_samples,), optional
        Cluster bootstrap: resample whole groups (users, queries, sessions)
        instead of rows. Use this whenever rows within a group are
        correlated, which is the norm for search, ads and recommendation
        logs; row-level resampling then gives intervals that are too narrow.
        Cannot be combined with ``stratified``.
    sample_weight : (n_samples,), optional
    """
    yt, ys = _labels(y_true, pos_label), _scores(y_score)
    _require_1d("y_true", yt)
    if ys.ndim not in (1, 2):
        raise ValueError(f"y_score must be 1-D or 2-D, got shape {ys.shape}")
    ys = _samples_last(ys, axis)
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be >= 1")
    single = ys.ndim == 1
    ys2 = ys[None, :] if single else ys
    k = ys2.shape[0]
    w = _weights(sample_weight)
    codes, n_groups = None, 0
    if groups is not None:
        if stratified:
            raise ValueError("stratified resampling cannot be combined with groups")
        ids, codes = _group_codes(groups, len(yt))
        n_groups = len(ids)

    roc_est, ap_est = _core.many_scores(yt, ys2, w)
    roc, ap = _core.bootstrap(
        yt, ys2, int(n_bootstraps), int(seed) % (1 << 64), bool(stratified),
        w, codes, n_groups,
    )
    roc = np.ascontiguousarray(roc.reshape(n_bootstraps, k).T)
    ap = np.ascontiguousarray(ap.reshape(n_bootstraps, k).T)
    if single:
        return BootstrapResult(roc[0], ap[0], float(roc_est[0]), float(ap_est[0]))
    return BootstrapResult(roc, ap, roc_est, ap_est)


# ---------------------------------------------------------------------------
# Threshold metrics (global sort)
# ---------------------------------------------------------------------------

_NAN = float("nan")
_INF = float("inf")


def _pair(y_true, y_score, sample_weight, pos_label):
    yt, ys = _labels(y_true, pos_label), _scores(y_score)
    _require_1d("y_true", yt)
    _require_1d("y_score", ys)
    return yt, ys, _weights(sample_weight)


def _curve(y_true, y_score, sample_weight, pos_label):
    return _core.curve(*_pair(y_true, y_score, sample_weight, pos_label))


def _target(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def ks_statistic(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> Tuple[float, float]:
    """Kolmogorov-Smirnov statistic, ``max |TPR - FPR|``.

    Returns ``(ks, threshold)``.
    """
    r = _core.binary_summary(*_pair(y_true, y_score, sample_weight, pos_label))
    return r[2], r[3]


def max_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> Tuple[float, float]:
    """Best F1 over all thresholds. Returns ``(f1, threshold)``."""
    r = _core.binary_summary(*_pair(y_true, y_score, sample_weight, pos_label))
    return r[4], r[5]


def precision_at_recall(
    y_true: ArrayLike,
    y_score: ArrayLike,
    recall: float,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> Tuple[float, float]:
    """Highest precision among thresholds with recall >= ``recall``.

    Returns ``(precision, threshold)``; ``(nan, nan)`` without positives.
    """
    recall = _target("recall", recall)
    thr, tps, fps = _curve(y_true, y_score, sample_weight, pos_label)
    if len(thr) == 0 or tps[-1] <= 0:
        return _NAN, _NAN
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tps + fps > 0, tps / (tps + fps), 0.0)
    ok = tps / tps[-1] >= recall
    i = int(np.argmax(np.where(ok, prec, -1.0)))
    return float(prec[i]), float(thr[i])


def recall_at_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    precision: float,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> Tuple[float, float]:
    """Highest recall among thresholds with precision >= ``precision``.

    Returns ``(recall, threshold)``; ``(0.0, inf)`` if no threshold qualifies.
    """
    precision = _target("precision", precision)
    thr, tps, fps = _curve(y_true, y_score, sample_weight, pos_label)
    if len(thr) == 0 or tps[-1] <= 0:
        return _NAN, _NAN
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tps + fps > 0, tps / (tps + fps), 0.0)
    ok = np.flatnonzero((prec >= precision) & (tps > 0))
    if len(ok) == 0:
        return 0.0, _INF
    i = int(ok[-1])  # recall is non-decreasing along the curve
    return float(tps[i] / tps[-1]), float(thr[i])


def tpr_at_fpr(
    y_true: ArrayLike,
    y_score: ArrayLike,
    fpr: float,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> Tuple[float, float]:
    """Highest true positive rate among thresholds with FPR <= ``fpr``.

    Returns ``(tpr, threshold)``; ``(0.0, inf)`` if only the empty
    prediction set qualifies.
    """
    fpr = _target("fpr", fpr)
    thr, tps, fps = _curve(y_true, y_score, sample_weight, pos_label)
    if len(thr) == 0 or tps[-1] <= 0 or fps[-1] <= 0:
        return _NAN, _NAN
    ok = np.flatnonzero(fps / fps[-1] <= fpr)
    if len(ok) == 0:
        return 0.0, _INF
    i = int(ok[-1])
    return float(tps[i] / tps[-1]), float(thr[i])


def lift_at(
    y_true: ArrayLike,
    y_score: ArrayLike,
    fraction: float = 0.1,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
) -> float:
    """Lift in the top ``fraction`` of scores (0.1 = top-decile lift).

    Positive rate among the highest-scored ``fraction`` of samples divided by
    the overall positive rate. Ties across the cut are split proportionally.
    """
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    thr, tps, fps = _curve(y_true, y_score, sample_weight, pos_label)
    if len(thr) == 0 or tps[-1] <= 0:
        return _NAN
    cum = tps + fps
    cut = fraction * cum[-1]
    tp_cut = np.interp(cut, np.concatenate([[0.0], cum]), np.concatenate([[0.0], tps]))
    return float((tp_cut / cut) / (tps[-1] / cum[-1]))


def _nonzero_curve(y_true, y_score, sample_weight, pos_label):
    yt, ys, w = _pair(y_true, y_score, sample_weight, pos_label)
    if w is not None:
        keep = w != 0  # as scikit-learn: zero-weight samples add no thresholds
        if not keep.all():
            yt, ys, w = yt[keep], ys[keep], w[keep]
    return _core.curve(yt, ys, w)


def roc_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
    drop_intermediate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve. Same output as ``sklearn.metrics.roc_curve``.

    Returns ``(fpr, tpr, thresholds)`` with thresholds decreasing and
    ``thresholds[0] = inf`` (the empty prediction set). Tied scores form a
    single point. ``drop_intermediate`` removes points that lie on a straight
    segment, which does not change the curve's shape or area. ``fpr`` (or
    ``tpr``) is all ``NaN`` when there are no negatives (or positives).
    """
    thr, tps, fps = _nonzero_curve(y_true, y_score, sample_weight, pos_label)
    if drop_intermediate and len(fps) > 2:
        keep = np.flatnonzero(
            np.concatenate([[True], (np.diff(fps, 2) != 0) | (np.diff(tps, 2) != 0), [True]])
        )
        thr, tps, fps = thr[keep], tps[keep], fps[keep]
    tps = np.concatenate([[0.0], tps])
    fps = np.concatenate([[0.0], fps])
    thr = np.concatenate([[np.inf], thr])
    fpr = fps / fps[-1] if fps[-1] > 0 else np.full(len(fps), _NAN)
    tpr = tps / tps[-1] if tps[-1] > 0 else np.full(len(tps), _NAN)
    return fpr, tpr, thr


def precision_recall_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
    drop_intermediate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision-recall curve. Same output as
    ``sklearn.metrics.precision_recall_curve``.

    Returns ``(precision, recall, thresholds)``: thresholds increasing, and
    the arrays ending in the point ``precision=1, recall=0`` (which has no
    threshold). Tied scores form a single point. ``drop_intermediate`` keeps
    only the end points of runs with equal recall. Without positives, recall
    is ``NaN`` (scikit-learn sets it to 1 and warns).
    """
    thr, tps, fps = _nonzero_curve(y_true, y_score, sample_weight, pos_label)
    if drop_intermediate and len(fps) > 2:
        keep = np.flatnonzero(
            np.concatenate([[True], (np.diff(tps[:-1]) != 0) | (np.diff(tps[1:]) != 0), [True]])
        )
        thr, tps, fps = thr[keep], tps[keep], fps[keep]
    ps = tps + fps
    precision = np.divide(tps, ps, out=np.zeros_like(tps), where=ps != 0)
    if len(tps) and tps[-1] > 0:
        recall = tps / tps[-1]
    else:
        recall = np.full(len(tps), _NAN)
    return (
        np.concatenate([precision[::-1], [1.0]]),
        np.concatenate([recall[::-1], [0.0]]),
        np.ascontiguousarray(thr[::-1]),
    )


def calibration_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = 5,
    strategy: str = "uniform",
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
    return_weights: bool = False,
):
    """Reliability diagram data: observed positive rate against mean
    predicted probability, per bin.

    Returns ``(prob_true, prob_pred)`` for non-empty bins, like
    ``sklearn.calibration.calibration_curve`` (identical output for
    ``strategy="uniform"``). With ``return_weights=True`` also returns each
    bin's total weight, for sizing points or error bars.

    ``strategy="quantile"`` makes equal-weight bins; tied probabilities that
    straddle a bin edge are split proportionally, so every bin has the same
    weight (scikit-learn instead cuts at percentile values, which leaves
    bins uneven when there are ties). The weighted mean of
    ``|prob_true - prob_pred|`` over bins is :func:`expected_calibration_error`.
    """
    yt, ps, w = _pair(y_true, y_prob, sample_weight, pos_label)
    _core.calibration_sums(yt, ps, w, 1e-15)  # validates the [0, 1] range
    tot, sum_p, sum_y = _calibration_bins(yt, ps, w, n_bins, strategy)
    ok = tot > 0
    prob_true, prob_pred = sum_y[ok] / tot[ok], sum_p[ok] / tot[ok]
    if return_weights:
        return prob_true, prob_pred, tot[ok]
    return prob_true, prob_pred


# ---------------------------------------------------------------------------
# Calibration (no sort)
# ---------------------------------------------------------------------------


def _calibration_bins(yt, ps, w, n_bins: int, strategy: str):
    """Per-bin (total weight, weighted sum of p, weighted sum of y)."""
    if int(n_bins) != n_bins or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    n_bins = int(n_bins)
    ww = np.ones(len(ps)) if w is None else w
    if strategy == "uniform":
        # Same edges as sklearn.calibration.calibration_curve: a value exactly
        # on an inner edge goes to the lower bin.
        bins = np.searchsorted(np.linspace(0.0, 1.0, n_bins + 1)[1:-1], ps)
        tot = np.bincount(bins, weights=ww, minlength=n_bins)
        sum_p = np.bincount(bins, weights=ww * ps, minlength=n_bins)
        sum_y = np.bincount(bins, weights=ww * yt, minlength=n_bins)
    elif strategy == "quantile":
        # Equal-weight bins. Samples sharing a probability are interchangeable,
        # so a run of tied values that straddles a bin edge is split across the
        # bins in proportion, carrying its average label. Summing each run and
        # interpolating the cumulative sums at the bin edges does exactly that,
        # and makes the result independent of row order.
        values, inv = np.unique(ps, return_inverse=True)
        inv = inv.reshape(-1)
        run_w = np.bincount(inv, weights=ww, minlength=len(values))
        run_y = np.bincount(inv, weights=ww * yt, minlength=len(values))
        cum_w = np.concatenate([[0.0], np.cumsum(run_w)])
        cum_p = np.concatenate([[0.0], np.cumsum(run_w * values)])
        cum_y = np.concatenate([[0.0], np.cumsum(run_y)])
        edges = np.linspace(0.0, cum_w[-1], n_bins + 1)
        tot = np.diff(edges)
        sum_p = np.diff(np.interp(edges, cum_w, cum_p))
        sum_y = np.diff(np.interp(edges, cum_w, cum_y))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    return tot, sum_p, sum_y


def _ece(yt, ps, w, n_bins: int, strategy: str) -> float:
    tot, sum_p, sum_y = _calibration_bins(yt, ps, w, n_bins, strategy)
    if len(ps) == 0 or tot.sum() <= 0:
        return _NAN
    # sum_b (n_b / N) * |mean_p_b - mean_y_b|
    return float(np.abs(sum_p - sum_y).sum() / tot.sum())


def _calibration(yt, ps, w, eps, n_bins, strategy) -> Dict[str, float]:
    sw, sy, sp, sll, sbr = _core.calibration_sums(yt, ps, w, eps)
    out = dict.fromkeys(
        ("log_loss", "normalized_entropy", "brier", "calibration_ratio", "ece"), _NAN
    )
    if sw <= 0:
        return out
    out["log_loss"] = sll / sw
    out["brier"] = sbr / sw
    rate = sy / sw
    if 0.0 < rate < 1.0:
        base = -(rate * np.log(rate) + (1.0 - rate) * np.log(1.0 - rate))
        out["normalized_entropy"] = float(out["log_loss"] / base)
    if sy > 0:
        out["calibration_ratio"] = sp / sy
    out["ece"] = _ece(yt, ps, w, n_bins, strategy)
    return out


def calibration_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    eps: float = 1e-15,
) -> Dict[str, float]:
    """Pointwise calibration metrics for probabilities in ``[0, 1]``.

    Returns a dict with

    * ``log_loss``: binary cross-entropy (probabilities clipped to
      ``[eps, 1 - eps]``).
    * ``normalized_entropy``: log loss divided by the entropy of the base
      rate. Below 1 beats predicting the base rate; ``1 - NE`` is the
      relative information gain (RIG).
    * ``brier``: mean squared error.
    * ``calibration_ratio``: sum of predictions over sum of labels (also
      called COPC or E/O). 1.0 is calibrated on average; above 1
      over-predicts.
    * ``ece``: expected calibration error over ``n_bins`` bins;
      ``strategy`` is ``"uniform"`` (equal width) or ``"quantile"``
      (equal weight; tied probabilities that straddle a bin edge are split
      proportionally).
    """
    yt, ps, w = _pair(y_true, y_prob, sample_weight, pos_label)
    return _calibration(yt, ps, w, eps, n_bins, strategy)


def log_loss(y_true, y_prob, *, sample_weight=None, pos_label=None, eps: float = 1e-15) -> float:
    """Binary cross-entropy. Equivalent to ``sklearn.metrics.log_loss``."""
    yt, ps, w = _pair(y_true, y_prob, sample_weight, pos_label)
    sw, _, _, sll, _ = _core.calibration_sums(yt, ps, w, eps)
    return sll / sw if sw > 0 else _NAN


def brier_score(y_true, y_prob, *, sample_weight=None, pos_label=None) -> float:
    """Brier score. Equivalent to ``sklearn.metrics.brier_score_loss``."""
    yt, ps, w = _pair(y_true, y_prob, sample_weight, pos_label)
    sw, _, _, _, sbr = _core.calibration_sums(yt, ps, w, 1e-15)
    return sbr / sw if sw > 0 else _NAN


def normalized_entropy(
    y_true, y_prob, *, sample_weight=None, pos_label=None, eps: float = 1e-15
) -> float:
    """Log loss divided by the entropy of the empirical positive rate."""
    return calibration_metrics(
        y_true, y_prob, sample_weight=sample_weight, pos_label=pos_label, n_bins=1, eps=eps
    )["normalized_entropy"]


def calibration_ratio(y_true, y_prob, *, sample_weight=None, pos_label=None) -> float:
    """Sum of predicted probabilities over the number of positives (COPC, E/O)."""
    yt, ps, w = _pair(y_true, y_prob, sample_weight, pos_label)
    sw, sy, sp, _, _ = _core.calibration_sums(yt, ps, w, 1e-15)
    return sp / sy if sy > 0 else _NAN


def expected_calibration_error(
    y_true,
    y_prob,
    *,
    sample_weight=None,
    pos_label=None,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Expected calibration error with equal-width or equal-count bins."""
    yt, ps, w = _pair(y_true, y_prob, sample_weight, pos_label)
    _core.calibration_sums(yt, ps, w, 1e-15)  # validates the [0, 1] range
    return _ece(yt, ps, w, n_bins, strategy)


# ---------------------------------------------------------------------------
# Everything for one binary model
# ---------------------------------------------------------------------------


def binary_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    pos_label: Any = None,
    calibration: Optional[bool] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    eps: float = 1e-15,
) -> Dict[str, float]:
    """Discrimination and calibration metrics for one model, one sort.

    Returns a dict with ``roc_auc``, ``average_precision``, ``ks``,
    ``ks_threshold``, ``f1_max``, ``f1_threshold``, ``prevalence`` and the
    keys of :func:`calibration_metrics`.

    ``calibration`` controls the calibration block: ``None`` (default)
    computes it when every score lies in ``[0, 1]`` and returns ``NaN`` for
    those keys otherwise; ``True`` requires probabilities and raises if they
    are not; ``False`` skips it.
    """
    yt, ys, w = _pair(y_true, y_score, sample_weight, pos_label)
    roc, ap, ks, ks_thr, f1, f1_thr = _core.binary_summary(yt, ys, w)
    out: Dict[str, float] = {
        "roc_auc": roc,
        "average_precision": ap,
        "ks": ks,
        "ks_threshold": ks_thr,
        "f1_max": f1,
        "f1_threshold": f1_thr,
    }
    if w is None:
        out["prevalence"] = float(yt.mean()) if len(yt) else _NAN
    else:
        out["prevalence"] = float(np.dot(w, yt) / w.sum()) if w.sum() > 0 else _NAN
    if calibration is None:
        calibration = bool(len(ys)) and bool(ys.min() >= 0.0 and ys.max() <= 1.0)
    if calibration:
        out.update(_calibration(yt, ys, w, eps, n_bins, strategy))
    else:
        out.update(
            dict.fromkeys(
                ("log_loss", "normalized_entropy", "brier", "calibration_ratio", "ece"), _NAN
            )
        )
    return out


# ---------------------------------------------------------------------------
# Grouped ranking metrics
# ---------------------------------------------------------------------------

_RANK_NAMES = ("ndcg", "mrr", "map", "recall", "precision", "hit_rate")


def _ranking_setup(y_true, n, groups, group_sizes, ks, gain, no_relevant, sample_weight):
    """Shared input handling for ranking_metrics and its bootstrap."""
    rel = np.asarray(y_true)
    _require_1d("y_true", rel)
    if len(rel) != n:
        raise ValueError(f"y_true has length {len(rel)}, y_score has {n} samples")
    if rel.dtype.kind not in "iufb":
        raise TypeError(f"y_true has non-numeric dtype {rel.dtype}")
    rel = rel.astype(np.float64)
    if rel.size and not (np.isfinite(rel).all() and rel.min() >= 0):
        raise ValueError("relevance must be finite and non-negative")
    yt = np.ascontiguousarray(rel > 0, dtype=np.uint8)
    if gain == "exponential":
        if rel.size and rel.max() > 1000:
            raise ValueError("relevance too large for gain='exponential'; use gain='linear'")
        gains = np.exp2(rel) - 1.0
    elif gain == "linear":
        gains = rel
    else:
        raise ValueError("gain must be 'exponential' or 'linear'")
    if no_relevant not in ("skip", "zero"):
        raise ValueError("no_relevant must be 'skip' or 'zero'")

    if (groups is None) == (group_sizes is None):
        raise ValueError("pass exactly one of groups or group_sizes")
    if groups is not None:
        ids, codes = _group_codes(groups, n)
    else:
        sizes = np.asarray(group_sizes)
        _require_1d("group_sizes", sizes)
        if sizes.dtype.kind not in "iu" or (sizes.size and sizes.min() < 0):
            raise ValueError("group_sizes must be non-negative integers")
        if int(sizes.sum()) != n:
            raise ValueError(f"group_sizes sums to {int(sizes.sum())}, y_score has {n} samples")
        ids = np.arange(len(sizes))
        codes = np.ascontiguousarray(np.repeat(ids, sizes), dtype=np.int64)
    size = np.bincount(codes, minlength=len(ids))

    ks = list(ks)
    if not ks:
        raise ValueError("ks must not be empty")
    core_ks, suffixes = [], []
    for k in ks:
        if k is None:
            core_ks.append(int(np.iinfo(np.uintp).max))
            suffixes.append("")
        else:
            if int(k) != k or k < 1:
                raise ValueError("cutoffs must be positive integers or None")
            core_ks.append(int(k))
            suffixes.append(f"@{int(k)}")
    if len(set(suffixes)) != len(suffixes):
        raise ValueError("duplicate cutoffs in ks")
    names = ["roc_auc"] + [m + s for s in suffixes for m in _RANK_NAMES]

    group_weight = None
    w = _weights(sample_weight)
    if w is not None:
        if len(w) != n:
            raise ValueError(f"sample_weight has length {len(w)}, expected {n}")
        if w.size and not (np.isfinite(w).all() and w.min() >= 0):
            raise ValueError("sample_weight must be finite and non-negative")
        # A query's weight is the mean weight of its rows.
        group_weight = np.bincount(codes, weights=w, minlength=len(ids)) / np.maximum(size, 1)
    return yt, np.ascontiguousarray(gains), ids, codes, size, core_ks, names, group_weight


def _ranking_summary(sums, wts, extra, names, no_relevant) -> Dict[str, float]:
    auc_sum, auc_w, n_nonempty, n_ranked, n_auc, w_nonempty = extra
    summary: Dict[str, float] = {
        "gauc": float(auc_sum / auc_w) if auc_w > 0 else _NAN,
        "gauc_macro": float(sums[0] / wts[0]) if wts[0] > 0 else _NAN,
    }
    for c in range(1, len(names)):
        denom = wts[c] if no_relevant == "skip" else w_nonempty
        summary[names[c]] = float(sums[c] / denom) if denom > 0 else _NAN
    summary["n_groups"] = int(n_nonempty)
    summary["n_groups_ranked"] = int(n_ranked)
    summary["n_groups_auc"] = int(n_auc)
    return summary


def ranking_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    groups: Optional[ArrayLike] = None,
    *,
    group_sizes: Optional[ArrayLike] = None,
    ks: Sequence[Optional[int]] = (1, 5, 10),
    gain: str = "exponential",
    no_relevant: str = "skip",
    sample_weight: Optional[ArrayLike] = None,
    per_group: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    """GAUC, NDCG, MRR, MAP, recall, precision and hit rate at several
    cutoffs, with one sort per group, in parallel.

    Parameters
    ----------
    y_true : (n_samples,)
        Relevance. Binary labels, or non-negative graded relevance. An item
        is "relevant" when its value is > 0; grades only affect NDCG.
    y_score : (n_samples,)
    groups : (n_samples,), optional
        Query / user / session id per row, in any order. Integer ids in a
        compact range take an O(N) fast path; other ids are sorted first.
    group_sizes : (n_groups,), optional
        Alternative to ``groups`` for data already laid out group by group
        (the LightGBM / XGBoost convention). Exactly one of the two.
    ks : cutoffs. ``None`` means the whole list and produces keys without a
        suffix (``"ndcg"``, ``"map"``, ...).
    gain : NDCG gain, ``"exponential"`` (``2**rel - 1``) or ``"linear"``.
    no_relevant : groups without any relevant item have undefined ranking
        metrics. ``"skip"`` (default) leaves them out of the averages;
        ``"zero"`` counts them as 0.
    sample_weight : (n_samples,), optional
        Weights the *averages over groups*: each group's weight is the mean
        weight of its rows (the LightGBM convention), so give every row of a
        query that query's weight. Values inside a group are not reweighted.
    per_group : also return the per-group values.

    Returns
    -------
    summary : dict
        ``gauc`` (ROC-AUC per group, weighted by group size), ``gauc_macro``
        (unweighted), ``<metric>@<k>`` means over groups, and the counts
        ``n_groups``, ``n_groups_ranked`` (at least one relevant item),
        ``n_groups_auc`` (both classes present).
    per_group : dict of arrays, only if ``per_group=True``
        ``group``, ``size``, ``roc_auc`` and every ``<metric>@<k>``.

    Notes
    -----
    Tied scores are averaged exactly over all orderings of the tied items,
    so a model that outputs a constant score gets chance-level metrics no
    matter how the rows are ordered. MAP@k divides by ``min(n_relevant, k)``;
    precision@k divides by ``k``.
    """
    ys = _scores(y_score)
    _require_1d("y_score", ys)
    yt, gains, ids, codes, size, core_ks, names, gw = _ranking_setup(
        y_true, len(ys), groups, group_sizes, ks, gain, no_relevant, sample_weight
    )
    sums, wts, extra, rows = _core.ranking(
        yt, gains, ys, codes, len(ids), core_ks, bool(per_group), gw
    )
    summary = _ranking_summary(sums, wts, extra, names, no_relevant)
    if not per_group:
        return summary

    mat = rows.reshape(len(ids), len(names))
    detail: Dict[str, np.ndarray] = {"group": ids, "size": size}
    for c, name in enumerate(names):
        detail[name] = np.ascontiguousarray(mat[:, c])
    return summary, detail


@dataclass(frozen=True)
class RankingBootstrapResult:
    """Bootstrap distribution from :func:`bootstrap_ranking_metrics`.

    Attributes
    ----------
    estimate : dict
        Metric name -> value on the original data (float, or an array of
        shape ``(n_models,)``).
    samples : dict
        Metric name -> replicates, shape ``(n_bootstraps,)`` or
        ``(n_models, n_bootstraps)``. Column ``b`` uses the same resampled
        set of groups for every metric and every model.
    """

    estimate: Dict[str, Any]
    samples: Dict[str, np.ndarray]

    def _samples(self, metric: str) -> np.ndarray:
        if metric not in self.samples:
            raise ValueError(f"unknown metric {metric!r}; available: {sorted(self.samples)}")
        return self.samples[metric]

    def ci(self, metric: str, level: float = 0.95) -> Any:
        """Percentile confidence interval: ``(low, high)`` or ``(n_models, 2)``."""
        return _percentile_ci(self._samples(metric), level)

    def std(self, metric: str) -> Any:
        """Bootstrap standard error."""
        s = np.nanstd(self._samples(metric), axis=-1, ddof=1)
        return float(s) if np.ndim(s) == 0 else s

    def compare(self, a: int, b: int, metric: str, level: float = 0.95):
        """Paired comparison of model ``a`` minus model ``b``.

        Returns ``(difference, (low, high), p_value)``.
        """
        return _paired_compare(self._samples(metric), self.estimate[metric], a, b, level)


def bootstrap_ranking_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    groups: Optional[ArrayLike] = None,
    *,
    group_sizes: Optional[ArrayLike] = None,
    ks: Sequence[Optional[int]] = (1, 5, 10),
    gain: str = "exponential",
    no_relevant: str = "skip",
    sample_weight: Optional[ArrayLike] = None,
    n_bootstraps: int = 1000,
    seed: int = 0,
    axis: int = -1,
) -> RankingBootstrapResult:
    """Confidence intervals for :func:`ranking_metrics` by resampling groups.

    Queries (not rows) are the resampling unit. Per-query metrics are
    computed once; each replicate is then a weighted mean over the drawn
    queries, so thousands of replicates cost little more than one evaluation.

    ``y_score`` may be ``(n_models, n_samples)`` (or ``(n_samples,
    n_models)`` with ``axis=0``) to compare rankers on identical resamples;
    see :meth:`RankingBootstrapResult.compare`. Other parameters are as in
    :func:`ranking_metrics`.
    """
    ys = _scores(y_score)
    if ys.ndim not in (1, 2):
        raise ValueError(f"y_score must be 1-D or 2-D, got shape {ys.shape}")
    ys = _samples_last(ys, axis)
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be >= 1")
    single = ys.ndim == 1
    ys2 = ys[None, :] if single else ys
    yt, gains, ids, codes, size, core_ks, names, gw = _ranking_setup(
        y_true, ys2.shape[1], groups, group_sizes, ks, gain, no_relevant, sample_weight
    )
    n_groups, n_cols = len(ids), len(names)
    w_group = np.ones(n_groups) if gw is None else gw
    w_auc = np.ascontiguousarray(w_group * size)
    seed = int(seed) % (1 << 64)
    keys = ["gauc", "gauc_macro"] + names[1:]

    est_rows, sample_rows = [], []
    for row in ys2:
        sums, wts, extra, flat = _core.ranking(
            yt, gains, np.ascontiguousarray(row), codes, n_groups, core_ks, True, gw
        )
        summary = _ranking_summary(sums, wts, extra, names, no_relevant)
        est_rows.append([summary[k] for k in keys])
        mat = flat.reshape(n_groups, n_cols)
        if no_relevant == "zero":
            mat = mat.copy()
            block = mat[:, 1:]
            block[np.isnan(block) & (size > 0)[:, None]] = 0.0
        # Same seed and group count -> the same draws in both calls.
        means = _core.bootstrap_group_means(
            np.ascontiguousarray(mat).reshape(-1), n_cols, w_group, int(n_bootstraps), seed
        ).reshape(n_bootstraps, n_cols)
        gauc = _core.bootstrap_group_means(
            np.ascontiguousarray(mat[:, 0]), 1, w_auc, int(n_bootstraps), seed
        )
        sample_rows.append(np.column_stack([gauc, means]))  # (B, 1 + n_cols)

    est = np.array(est_rows)  # (K, n_keys)
    smp = np.stack(sample_rows)  # (K, B, n_keys)
    if single:
        return RankingBootstrapResult(
            {k: float(est[0, j]) for j, k in enumerate(keys)},
            {k: np.ascontiguousarray(smp[0, :, j]) for j, k in enumerate(keys)},
        )
    return RankingBootstrapResult(
        {k: est[:, j].copy() for j, k in enumerate(keys)},
        {k: np.ascontiguousarray(smp[:, :, j]) for j, k in enumerate(keys)},
    )


@dataclass(frozen=True)
class DeLongResult:
    """ROC-AUC with DeLong's analytic covariance, from :func:`delong_roc_auc`.

    Attributes
    ----------
    roc_auc : float, or ndarray of shape ``(n_models,)``
    covariance : ndarray of shape ``(n_models, n_models)``
        Estimated covariance of the AUC estimates (``NaN`` with fewer than
        two samples in either class).
    n_pos, n_neg : int
    """

    roc_auc: Any
    covariance: np.ndarray
    n_pos: int
    n_neg: int

    @property
    def _single(self) -> bool:
        return np.ndim(self.roc_auc) == 0

    def std(self) -> Any:
        """Standard error of the AUC."""
        se = np.sqrt(np.maximum(np.diag(self.covariance), 0.0))
        return float(se[0]) if self._single else se

    def ci(self, level: float = 0.95) -> Any:
        """Normal-approximation confidence interval, clipped to ``[0, 1]``.

        Returns ``(low, high)`` for one model, or an array of shape
        ``(n_models, 2)``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        auc, se = np.atleast_1d(self.roc_auc), np.atleast_1d(self.std())
        out = np.clip(np.stack([auc - z * se, auc + z * se], axis=-1), 0.0, 1.0)
        return (float(out[0, 0]), float(out[0, 1])) if self._single else out

    def compare(self, a: int, b: int, level: float = 0.95):
        """DeLong test of model ``a`` against model ``b`` on the same samples.

        Returns ``(difference, (low, high), p_value)``: the AUC difference
        ``a - b``, its confidence interval, and the two-sided p-value for
        "equal AUC".
        """
        if self._single:
            raise ValueError("compare() needs a result computed from 2-D scores")
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        auc, c = np.asarray(self.roc_auc), self.covariance
        diff = float(auc[a] - auc[b])
        var = c[a, a] + c[b, b] - 2.0 * c[a, b]
        if math.isnan(var) or math.isnan(diff):
            return diff, (_NAN, _NAN), _NAN
        se = math.sqrt(max(var, 0.0))
        if se == 0.0:
            return diff, (diff, diff), 1.0 if diff == 0.0 else 0.0
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        p = math.erfc(abs(diff) / se / math.sqrt(2.0))
        return diff, (diff - z * se, diff + z * se), float(p)


def delong_roc_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    axis: int = -1,
    pos_label: Any = None,
) -> DeLongResult:
    """ROC-AUC with DeLong's analytic variance, confidence intervals and
    paired test, with no resampling.

    Uses the fast algorithm of Sun and Xu (2014): one sort per model, then
    linear work. Ties count half, as in :func:`roc_auc`.

    Parameters
    ----------
    y_true : (n_samples,)
    y_score : (n_samples,) or (n_models, n_samples)
        Several models evaluated on the same samples give the full covariance
        matrix, so :meth:`DeLongResult.compare` is the classic DeLong test for
        two correlated ROC curves. Pass ``axis=0`` for ``(n_samples, n_models)``.

    Notes
    -----
    DeLong assumes independent samples. When rows are clustered (several
    impressions per user or query), use
    :func:`bootstrap_roc_pr_auc` with ``groups=`` instead. Sample weights are
    not supported here.
    """
    yt, ys = _labels(y_true, pos_label), _scores(y_score)
    _require_1d("y_true", yt)
    if ys.ndim not in (1, 2):
        raise ValueError(f"y_score must be 1-D or 2-D, got shape {ys.shape}")
    ys = _samples_last(ys, axis)
    single = ys.ndim == 1
    ys2 = ys[None, :] if single else ys
    if ys2.shape[1] != len(yt):
        raise ValueError(f"y_true / y_score: length mismatch ({len(yt)} vs {ys2.shape[1]})")
    k = ys2.shape[0]
    m = int(np.count_nonzero(yt))
    n = len(yt) - m
    if m == 0 or n == 0:
        auc = np.full(k, _NAN)
        cov = np.full((k, k), _NAN)
    else:
        v10, v01 = _core.delong_components(yt, ys2)
        v10, v01 = v10.reshape(k, m), v01.reshape(k, n)
        auc = v10.mean(axis=1)
        if m >= 2 and n >= 2:
            cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
        else:
            cov = np.full((k, k), _NAN)
    return DeLongResult(float(auc[0]) if single else auc, cov, m, n)


def num_threads() -> int:
    """Number of worker threads fastauc will use."""
    return _core.num_threads()


def set_num_threads(n: Optional[int]) -> None:
    """Limit fastauc to ``n`` threads, process-wide.

    Use ``1`` inside data-loader workers, joblib jobs or any code that is
    already parallel, to avoid oversubscribing cores. ``None`` restores the
    default: all cores, or the ``RAYON_NUM_THREADS`` environment variable.
    Results never depend on the thread count.
    """
    if n is None:
        _core.set_num_threads(0)
        return
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer or None")
    _core.set_num_threads(int(n))
