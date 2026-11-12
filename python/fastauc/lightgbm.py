"""fastauc metrics as LightGBM ``feval`` functions.

::

    import lightgbm as lgb
    from fastauc import lightgbm as fa_lgb

    lgb.train({"objective": "lambdarank", "metric": "None"}, train,
              valid_sets=[valid], feval=fa_lgb.ranking("ndcg@10", "mrr@10", "gauc"))

    lgb.train({"objective": "binary", "metric": "None"}, train,
              valid_sets=[valid], feval=fa_lgb.binary("roc_auc", "log_loss"))

Ranking metrics take query groups from the evaluation ``Dataset``
(``group=``), and rows' weights, if set, weight the average over queries.
Tied predictions are handled exactly, unlike LightGBM's built-in NDCG and
MAP, which depend on row order within a query. Each metric reports its own
direction, so ``early_stopping`` works. LightGBM itself is not imported.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

from . import _integrations as _i

__all__ = ["binary", "ranking"]

Feval = Callable[[object, object], List[Tuple[str, float, bool]]]


def ranking(*metrics: str, gain: str = "exponential", no_relevant: str = "skip") -> Feval:
    """A ``feval`` computing per-query ranking metrics.

    ``metrics`` are names such as ``"ndcg@10"``, ``"mrr@5"``, ``"map@10"``,
    ``"recall@20"``, ``"precision@1"``, ``"hit_rate@3"``, the whole-list
    ``"ndcg"`` (no cutoff), ``"gauc"`` or ``"gauc_macro"``; the default is
    ``"ndcg@10"``. ``gain`` and ``no_relevant`` are passed to
    :func:`fastauc.ranking_metrics`. All metrics come from one sort per query.
    """
    names, ks = _i.parse_ranking(metrics)

    def feval(preds, eval_data):
        values = _i.ranking_values(
            names, ks, eval_data.get_label(), preds, eval_data.get_group(),
            _i.weights_or_none(eval_data.get_weight()), gain, no_relevant,
        )
        return [(name, value, True) for name, value in values]

    return _i.describe(feval, "ranking", names)


def binary(*metrics: str) -> Feval:
    """A ``feval`` computing binary classification metrics.

    ``metrics``: ``"roc_auc"`` (default), ``"average_precision"``,
    ``"pr_auc"``, ``"ks"``, ``"f1_max"``, and the lower-is-better
    ``"log_loss"``, ``"normalized_entropy"``, ``"brier"``, ``"ece"``. The
    calibration metrics need probabilities: with LightGBM's built-in
    ``binary`` objective ``feval`` receives them; with a custom objective it
    receives raw scores, which suit only the ranking-type metrics.
    """
    names = _i.parse_binary(metrics)

    def feval(preds, eval_data):
        values = _i.binary_values(names, eval_data.get_label(), preds,
                                  _i.weights_or_none(eval_data.get_weight()))
        return [(name, value, _i.higher_is_better(name)) for name, value in values]

    return _i.describe(feval, "binary", names)
