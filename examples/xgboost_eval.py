"""fastauc as an XGBoost custom metric.

    python examples/xgboost_eval.py
"""
import numpy as np
import xgboost as xgb

import fastauc


def ndcg_at_10(preds, dmatrix):
    """Tie-exact NDCG@10 for a ranking model."""
    m = fastauc.ranking_metrics(
        dmatrix.get_label(), preds, group_sizes=dmatrix.get_group().astype(np.int64), ks=(10,)
    )
    return "fastauc-ndcg@10", m["ndcg@10"]


def normalized_entropy(preds, dmatrix):
    """Log loss relative to predicting the base rate (lower is better)."""
    return "fastauc-ne", fastauc.normalized_entropy(dmatrix.get_label(), preds)


def main():
    rng = np.random.default_rng(0)
    sizes = np.full(400, 25)
    x = rng.normal(size=(sizes.sum(), 8))
    rel = np.clip(np.round(x[:, 0] + 0.5 * x[:, 1] + rng.normal(size=len(x))), 0, 3)
    half = len(x) // 2

    dtrain = xgb.DMatrix(x[:half], rel[:half])
    dtrain.set_group(sizes[:200])
    dvalid = xgb.DMatrix(x[half:], rel[half:])
    dvalid.set_group(sizes[200:])
    history = {}
    xgb.train(
        {"objective": "rank:ndcg", "disable_default_eval_metric": True},
        dtrain, num_boost_round=20, evals=[(dvalid, "valid")],
        custom_metric=ndcg_at_10, evals_result=history, verbose_eval=False,
    )
    print("rank:ndcg ndcg@10:", round(history["valid"]["fastauc-ndcg@10"][-1], 4))

    y = (rng.random(len(x)) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    dtrain, dvalid = xgb.DMatrix(x[:half], y[:half]), xgb.DMatrix(x[half:], y[half:])
    history = {}
    xgb.train(
        {"objective": "binary:logistic", "disable_default_eval_metric": True},
        dtrain, num_boost_round=20, evals=[(dvalid, "valid")],
        custom_metric=normalized_entropy, evals_result=history, verbose_eval=False,
    )
    ne = history["valid"]["fastauc-ne"][-1]
    print("binary:logistic normalized entropy:", round(ne, 4))
    assert ne < 1.0


if __name__ == "__main__":
    main()
