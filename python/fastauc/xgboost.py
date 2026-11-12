"""fastauc metrics as XGBoost ``custom_metric`` functions.

::

    import xgboost as xgb
    from fastauc import xgboost as fa_xgb

    xgb.train({"objective": "rank:ndcg", "disable_default_eval_metric": True}, dtrain,
              evals=[(dvalid, "valid")], custom_metric=fa_xgb.ranking("ndcg@10", "mrr@10"))

    xgb.train({"objective": "binary:logistic", "disable_default_eval_metric": True}, dtrain,
              evals=[(dvalid, "valid")], custom_metric=fa_xgb.binary("roc_auc", "log_loss"))

Ranking metrics take query groups from the evaluation ``DMatrix`` (``qid=``
or ``set_group``) and its per-query weights, if set. Tied predictions are handled exactly, unlike XGBoost's
built-in NDCG and MAP, which depend on row order within a query. For early
stopping, XGBoost uses the last metric listed; pass ``maximize=True`` to
``xgb.train`` unless that metric is a loss. XGBoost itself is not imported.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from . import _integrations as _i

__all__ = ["binary", "ranking"]

CustomMetric = Callable[[np.ndarray, object], List[Tuple[str, float]]]


def _group_sizes(dmatrix) -> np.ndarray:
    ptr = np.asarray(dmatrix.get_uint_info("group_ptr"), dtype=np.int64)
    return np.diff(ptr) if ptr.size > 1 else ptr[:0]


def ranking(*metrics: str, gain: str = "exponential", no_relevant: str = "skip") -> CustomMetric:
    """A ``custom_metric`` computing per-query ranking metrics.

    Same metric names and options as :func:`fastauc.lightgbm.ranking`
    (default ``"ndcg@10"``). All metrics come from one sort per query.
    """
    names, ks = _i.parse_ranking(metrics)

    def custom_metric(preds, dmatrix):
        sizes = _group_sizes(dmatrix)
        weights = _i.weights_or_none(dmatrix.get_weight())
        if weights is not None and len(sizes) and len(weights) == len(sizes):
            # XGBoost ranking weights are per query; give each row its query's weight.
            weights = np.repeat(weights, sizes)
        return _i.ranking_values(names, ks, dmatrix.get_label(), preds, sizes, weights,
                                 gain, no_relevant)

    return _i.describe(custom_metric, "ranking", names)


def binary(*metrics: str) -> CustomMetric:
    """A ``custom_metric`` computing binary classification metrics.

    Same metric names as :func:`fastauc.lightgbm.binary` (default
    ``"roc_auc"``). With ``binary:logistic`` XGBoost passes probabilities, so
    the calibration metrics are available.
    """
    names = _i.parse_binary(metrics)

    def custom_metric(preds, dmatrix):
        return _i.binary_values(names, dmatrix.get_label(), preds,
                                _i.weights_or_none(dmatrix.get_weight()))

    return _i.describe(custom_metric, "binary", names)
