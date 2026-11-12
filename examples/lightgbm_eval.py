"""fastauc as LightGBM evaluation metrics.

    python examples/lightgbm_eval.py
"""
import lightgbm as lgb
import numpy as np

from fastauc import lightgbm as fa_lgb


def main():
    rng = np.random.default_rng(0)

    # Ranking: 400 queries of 25 documents.
    sizes = np.full(400, 25)
    x = rng.normal(size=(sizes.sum(), 8))
    rel = np.clip(np.round(x[:, 0] + 0.5 * x[:, 1] + rng.normal(size=len(x))), 0, 3)
    half = len(x) // 2
    train = lgb.Dataset(x[:half], rel[:half], group=sizes[:200])
    valid = lgb.Dataset(x[half:], rel[half:], group=sizes[200:], reference=train)
    history = {}
    lgb.train(
        {"objective": "lambdarank", "metric": "None", "verbose": -1},
        train, num_boost_round=20, valid_sets=[valid], valid_names=["valid"],
        feval=fa_lgb.ranking("ndcg@10", "mrr@10", "gauc"),  # ties handled exactly
        callbacks=[lgb.record_evaluation(history)],
    )
    print("lambdarank:", {k: round(v[-1], 4) for k, v in history["valid"].items()})
    assert history["valid"]["ndcg@10"][-1] > history["valid"]["ndcg@10"][0] - 1e-9

    # Binary (CTR / CVR): discrimination and calibration.
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    train = lgb.Dataset(x[:half], y[:half])
    valid = lgb.Dataset(x[half:], y[half:], reference=train)
    history = {}
    lgb.train(
        {"objective": "binary", "metric": "None", "verbose": -1},
        train, num_boost_round=20, valid_sets=[valid], valid_names=["valid"],
        feval=fa_lgb.binary("roc_auc", "ks", "normalized_entropy"),
        callbacks=[lgb.record_evaluation(history)],
    )
    print("binary:", {k: round(v[-1], 4) for k, v in history["valid"].items()})
    assert history["valid"]["roc_auc"][-1] > 0.7


if __name__ == "__main__":
    main()
