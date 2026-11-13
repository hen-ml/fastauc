"""Case study: do LightGBM's built-in ranking metrics depend on row order?

Trains LambdaMART on MSLR-WEB10K (Fold1) and reports, per boosting round,
LightGBM's own validation NDCG@10 on three copies of the validation set that
differ only in the order of rows within each query:

* ``original``: as distributed,
* ``relevant_first``: rows sorted by label, highest first,
* ``relevant_last``: rows sorted by label, lowest first.

The model and its predictions are identical across the three copies, so any
difference comes from how tied predictions are broken. fastauc's NDCG@10
(ties averaged exactly) is reported alongside. The script then simulates
early stopping on each curve, measures how often ties reach the top 10, and
repeats the comparison for bfloat16-rounded predictions.

Queries with no relevant document are dropped from validation and test:
LightGBM scores them 1.0 and fastauc leaves them out by default, so dropping
them makes the two agree exactly when there are no ties.

    python benchmarks/case_study_mslr.py --data /path/to/MSLR-WEB10K --out results.json

``--data`` holds ``Fold1/{train,vali,test}.txt`` (SVMlight format with qid).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_svmlight_file

import fastauc
from fastauc import lightgbm as fa_lgb

K = 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load(path: str, max_queries: int = 0):
    X, y, qid = load_svmlight_file(path, query_id=True, dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, qid[1:] != qid[:-1]])
    if len(np.unique(qid)) != len(starts):
        raise ValueError(f"{path}: rows of a query are not contiguous")
    if max_queries and len(starts) > max_queries:
        end = starts[max_queries]
        X, y, qid, starts = X[:end], y[:end], qid[:end], starts[:max_queries]
    sizes = np.diff(np.r_[starts, len(y)])
    return X, y.astype(np.float64), sizes


def drop_unjudged(X, y, sizes):
    """Keep queries with at least one relevant document."""
    q = np.repeat(np.arange(len(sizes)), sizes)
    has_rel = np.bincount(q, weights=(y > 0).astype(float), minlength=len(sizes)) > 0
    rows = has_rel[q]
    return X[rows], y[rows], sizes[has_rel]


def reorder(y, sizes, how: str) -> np.ndarray:
    """Row permutation that keeps queries in place and reorders rows inside each."""
    q = np.repeat(np.arange(len(sizes)), sizes)
    if how == "original":
        return np.arange(len(y))
    key = -y if how == "relevant_first" else y
    return np.lexsort((np.arange(len(y)), key, q))  # stable within equal labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def ndcg_by_position(y, s, sizes, k=K) -> float:
    """NDCG@k as most libraries compute it: stable sort, ties keep input order."""
    q = np.repeat(np.arange(len(sizes)), sizes)
    order = np.lexsort((np.arange(len(y)), -s, q))
    ideal = np.lexsort((-y, q))
    start = np.repeat(np.r_[0, np.cumsum(sizes)[:-1]], sizes)
    rank = np.arange(len(y)) - start  # 0-based rank within query, in sorted order
    disc = np.where(rank < k, 1.0 / np.log2(rank + 2.0), 0.0)
    gain = 2.0 ** y - 1.0
    dcg = np.bincount(q, weights=gain[order] * disc, minlength=len(sizes))
    idcg = np.bincount(q, weights=gain[ideal] * disc, minlength=len(sizes))
    return float(np.mean(dcg / idcg))


def ties_in_top_k(y, s, sizes, k=K) -> Dict[str, float]:
    """How often a tie can change NDCG@k: a tied block that reaches the top k and
    mixes documents of different relevance."""
    q = np.repeat(np.arange(len(sizes)), sizes)
    order = np.lexsort((-s, q))
    qs, ss, ys = q[order], s[order], y[order]
    start = np.repeat(np.r_[0, np.cumsum(sizes)[:-1]], sizes)
    rank = np.arange(len(y)) - start
    new_block = np.r_[True, (qs[1:] != qs[:-1]) | (ss[1:] != ss[:-1])]
    block = np.cumsum(new_block) - 1
    n_blocks = block[-1] + 1
    b_size = np.bincount(block, minlength=n_blocks)
    b_first_rank = rank[new_block]
    b_ymin = np.full(n_blocks, np.inf)
    b_ymax = np.full(n_blocks, -np.inf)
    np.minimum.at(b_ymin, block, ys)
    np.maximum.at(b_ymax, block, ys)
    harmful = (b_size > 1) & (b_first_rank < k) & (b_ymin != b_ymax)
    b_query = qs[new_block]
    q_harm = np.bincount(b_query[harmful], minlength=len(sizes)) > 0
    distinct_share = n_blocks / len(y)
    return {
        "queries_with_mixed_tie_in_top_k": float(q_harm.mean()),
        "docs_in_tied_blocks": float((b_size[block] > 1).mean()),
        "distinct_scores_per_doc": float(distinct_share),
    }


def to_bfloat16(a: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round to nearest even) and back."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def early_stop(curve: List[float], patience: int) -> int:
    """1-based best iteration chosen by LightGBM-style early stopping."""
    best, best_i = -np.inf, 0
    for i, v in enumerate(curve):
        if v > best:
            best, best_i = v, i
        elif i - best_i >= patience:
            break
    return best_i + 1


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="directory with Fold1/{train,vali,test}.txt")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--patience", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0, help="limit queries per split (smoke tests)")
    ap.add_argument("--out", default="case_study_mslr.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    fold = os.path.join(args.data, "Fold1")
    Xtr, ytr, gtr = load(os.path.join(fold, "train.txt"), args.max_queries)
    Xva, yva, gva = drop_unjudged(*load(os.path.join(fold, "vali.txt"), args.max_queries))
    Xte, yte, gte = drop_unjudged(*load(os.path.join(fold, "test.txt"), args.max_queries))
    t_load = time.time() - t0
    print(f"loaded in {t_load:.0f}s: train {len(gtr)} queries / {len(ytr)} docs, "
          f"valid {len(gva)} / {len(yva)}, test {len(gte)} / {len(yte)}", flush=True)

    orders = ("original", "relevant_first", "relevant_last")
    perms = {o: reorder(yva, gva, o) for o in orders}
    train = lgb.Dataset(Xtr, ytr, group=gtr, free_raw_data=False)
    valids = [lgb.Dataset(Xva[perms[o]], yva[perms[o]], group=gva, reference=train) for o in orders]
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [K],
        "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
        "seed": args.seed, "deterministic": True, "verbose": -1,
    }
    fastauc_ndcg = fa_lgb.ranking(f"ndcg@{K}")

    def feval(preds, data):  # renamed so it does not collide with LightGBM's own ndcg@10
        return [(f"fastauc_{name}", value, up) for name, value, up in fastauc_ndcg(preds, data)]

    hist: Dict[str, Dict[str, List[float]]] = {}
    t0 = time.time()
    booster = lgb.train(params, train, num_boost_round=args.rounds, valid_sets=valids,
                        valid_names=list(orders), feval=feval,
                        callbacks=[lgb.record_evaluation(hist)])
    t_train = time.time() - t0
    print(f"trained {args.rounds} rounds in {t_train:.0f}s", flush=True)

    builtin = {o: [float(v) for v in hist[o][f"ndcg@{K}"]] for o in orders}
    fa_by_order = {o: np.asarray(hist[o][f"fastauc_ndcg@{K}"], dtype=float) for o in orders}
    for o in orders:  # same predictions, so fastauc must give one curve for every row order
        assert np.array_equal(fa_by_order[o], fa_by_order["original"]), o
    fa = [float(v) for v in fa_by_order["original"]]

    # At checkpoints: recompute both definitions from the predictions, and check
    # the by-position NDCG reproduces LightGBM's built-in numbers.
    checkpoints = sorted({i for i in (1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, args.rounds)
                          if 1 <= i <= args.rounds})
    rows = []
    for it in checkpoints:
        s = booster.predict(Xva, num_iteration=it)
        exact = fastauc.ranking_metrics(yva, s, group_sizes=gva, ks=(K,))[f"ndcg@{K}"]
        per_order = {}
        for o in orders:
            p = perms[o]
            per_order[o] = ndcg_by_position(yva[p], s[p], gva)
            v = fastauc.ranking_metrics(yva[p], s[p], group_sizes=gva, ks=(K,))[f"ndcg@{K}"]
            assert abs(v - exact) < 1e-12, (it, o, v, exact)
        # Our by-position NDCG must reproduce LightGBM's built-in metric exactly.
        gap = max(abs(per_order[o] - builtin[o][it - 1]) for o in orders)
        assert gap < 1e-9, (it, gap)
        rows.append({"iteration": it, "fastauc": exact, **{f"by_position_{o}": per_order[o] for o in orders},
                     "lightgbm_builtin": {o: builtin[o][it - 1] for o in orders},
                     **ties_in_top_k(yva, s, gva)})
        print(f"iter {it:>4}: lightgbm first {builtin['relevant_first'][it-1]:.4f} "
              f"orig {builtin['original'][it-1]:.4f} last {builtin['relevant_last'][it-1]:.4f} | "
              f"fastauc {exact:.4f} | queries with a mixed tie in top {K}: "
              f"{rows[-1]['queries_with_mixed_tie_in_top_k']:.1%}", flush=True)

    # Early stopping on each curve, and what it costs on the test set.
    stops = []
    for patience in args.patience:
        entry = {"patience": patience}
        for name, curve in [*((f"lightgbm_{o}", builtin[o]) for o in orders), ("fastauc", fa)]:
            best = early_stop(curve, patience)
            s_te = booster.predict(Xte, num_iteration=best)
            entry[name] = {
                "best_iteration": best,
                "test_ndcg@10": fastauc.ranking_metrics(yte, s_te, group_sizes=gte, ks=(K,))[f"ndcg@{K}"],
            }
        stops.append(entry)
        print(f"early stopping, patience {patience}: " + ", ".join(
            f"{k} -> iter {v['best_iteration']} (test {v['test_ndcg@10']:.4f})"
            for k, v in entry.items() if k != "patience"), flush=True)

    # Low-precision scores: the final model's test predictions in bfloat16.
    s32 = booster.predict(Xte).astype(np.float32).astype(np.float64)
    s16 = to_bfloat16(s32)
    lowp = {}
    for name, s in (("float32", s32), ("bfloat16", s16)):
        res = {"fastauc": fastauc.ranking_metrics(yte, s, group_sizes=gte, ks=(K,))[f"ndcg@{K}"]}
        for o in orders:
            p = reorder(yte, gte, o)
            res[f"by_position_{o}"] = ndcg_by_position(yte[p], s[p], gte)
        res.update(ties_in_top_k(yte, s, gte))
        lowp[name] = res
        print(f"test scores in {name}: first {res['by_position_relevant_first']:.4f} "
              f"orig {res['by_position_original']:.4f} last {res['by_position_relevant_last']:.4f} | "
              f"fastauc {res['fastauc']:.4f} | queries with a mixed tie in top {K}: "
              f"{res['queries_with_mixed_tie_in_top_k']:.1%}", flush=True)

    result = {
        "dataset": "MSLR-WEB10K Fold1",
        "split_sizes": {"train": [len(gtr), len(ytr)], "valid": [len(gva), len(yva)],
                        "test": [len(gte), len(yte)]},
        "params": {**params, "rounds": args.rounds},
        "versions": {"lightgbm": lgb.__version__, "fastauc": fastauc.__version__,
                     "numpy": np.__version__, "python": platform.python_version()},
        "seconds": {"load": t_load, "train": t_train},
        "curves": {"lightgbm_builtin": builtin, "fastauc": fa},
        "checkpoints": rows,
        "early_stopping": stops,
        "low_precision_test": lowp,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("CASE_STUDY_JSON_BEGIN")
    print(json.dumps(result))
    print("CASE_STUDY_JSON_END")
    return result


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
