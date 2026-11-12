"""fastauc as XGBoost evaluation metrics.

    python examples/xgboost_eval.py
"""
import numpy as np
import xgboost as xgb

from fastauc import xgboost as fa_xgb


def main():
    rng = np.random.default_rng(0)
    sizes = np.full(400, 25)
    x = rng.normal(size=(sizes.sum(), 8))
    rel = np.clip(np.round(x[:, 0] + 0.5 * x[:, 1] + rng.normal(size=len(x))), 0, 3)
    qid = np.repeat(np.arange(len(sizes)), sizes)
    half = len(x) // 2

    dtrain = xgb.DMatrix(x[:half], rel[:half], qid=qid[:half])
    dvalid = xgb.DMatrix(x[half:], rel[half:], qid=qid[half:])
    history = {}
    xgb.train(
        {"objective": "rank:ndcg", "disable_default_eval_metric": True},
        dtrain, num_boost_round=20, evals=[(dvalid, "valid")],
        custom_metric=fa_xgb.ranking("mrr@10", "ndcg@10"),  # early stopping uses the last one
        maximize=True, evals_result=history, verbose_eval=False,
    )
    print("rank:ndcg:", {k: round(v[-1], 4) for k, v in history["valid"].items()})

    y = (rng.random(len(x)) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    dtrain, dvalid = xgb.DMatrix(x[:half], y[:half]), xgb.DMatrix(x[half:], y[half:])
    history = {}
    xgb.train(
        {"objective": "binary:logistic", "disable_default_eval_metric": True},
        dtrain, num_boost_round=20, evals=[(dvalid, "valid")],
        custom_metric=fa_xgb.binary("roc_auc", "normalized_entropy"),
        evals_result=history, verbose_eval=False,
    )
    ne = history["valid"]["normalized_entropy"][-1]
    print("binary:logistic:", {k: round(v[-1], 4) for k, v in history["valid"].items()})
    assert ne < 1.0


if __name__ == "__main__":
    main()
