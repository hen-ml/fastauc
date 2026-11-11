"""fastauc as a LightGBM evaluation metric.

    python examples/lightgbm_eval.py
"""
import lightgbm as lgb
import numpy as np

import fastauc


def ranking_feval(preds, data):
    """NDCG / MRR / GAUC for a lambdarank model. Ties are handled exactly."""
    m = fastauc.ranking_metrics(
        data.get_label(), preds, group_sizes=data.get_group(), ks=(10,)
    )
    return [
        ("ndcg@10", m["ndcg@10"], True),
        ("mrr@10", m["mrr@10"], True),
        ("gauc", m["gauc"], True),
    ]


def binary_feval(preds, data):
    """Discrimination and calibration for a binary (CTR / CVR) model."""
    m = fastauc.binary_metrics(data.get_label(), preds, sample_weight=data.get_weight())
    return [
        ("roc_auc", m["roc_auc"], True),
        ("ks", m["ks"], True),
        ("normalized_entropy", m["normalized_entropy"], False),
        ("calibration_ratio", m["calibration_ratio"], True),
    ]


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
        feval=ranking_feval, callbacks=[lgb.record_evaluation(history)],
    )
    print("lambdarank ndcg@10:", round(history["valid"]["ndcg@10"][-1], 4))
    assert history["valid"]["ndcg@10"][-1] > history["valid"]["ndcg@10"][0] - 1e-9

    # Binary.
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    train = lgb.Dataset(x[:half], y[:half])
    valid = lgb.Dataset(x[half:], y[half:], reference=train)
    history = {}
    lgb.train(
        {"objective": "binary", "metric": "None", "verbose": -1},
        train, num_boost_round=20, valid_sets=[valid], valid_names=["valid"],
        feval=binary_feval, callbacks=[lgb.record_evaluation(history)],
    )
    print("binary:", {k: round(v[-1], 4) for k, v in history["valid"].items()})
    assert history["valid"]["roc_auc"][-1] > 0.7


if __name__ == "__main__":
    main()
