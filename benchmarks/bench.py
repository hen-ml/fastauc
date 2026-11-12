"""Benchmark fastauc against scikit-learn.  Usage: python benchmarks/bench.py"""
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import fastauc


def best(fn, repeat=3):
    out = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out


def sk_pair(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)


def row(name, t_sk, t_fa):
    print(f"{name:<44} {t_sk * 1e3:>10.1f} ms {t_fa * 1e3:>9.2f} ms {t_sk / t_fa:>7.0f}x")


def main():
    rng = np.random.default_rng(42)
    print(f"threads: {fastauc.num_threads()}")
    print(f"{'task':<44} {'scikit-learn':>13} {'fastauc':>12} {'speedup':>8}")

    for n in (1_000, 100_000, 10_000_000):
        y = rng.integers(0, 2, n, dtype=np.uint8)
        s = rng.random(n)
        rep = 1 if n >= 10_000_000 else 3
        row(f"single pair, N={n:,}", best(lambda: sk_pair(y, s), rep),
            best(lambda: fastauc.roc_auc_ap(y, s), rep))

    n, m = 10_000, 1_000
    y = rng.integers(0, 2, n, dtype=np.uint8)
    S = rng.random((m, n))
    row(f"{m:,} models x 1 target, N={n:,}",
        best(lambda: [sk_pair(y, r) for r in S], 1),
        best(lambda: fastauc.roc_auc_ap_batch(y, S)))

    Y = rng.integers(0, 2, (m, n), dtype=np.uint8)
    s = rng.random(n)
    row(f"1 model x {m:,} targets, N={n:,}",
        best(lambda: [sk_pair(r, s) for r in Y], 1),
        best(lambda: fastauc.roc_auc_ap_batch(Y, s)))

    def sk_boot():
        r = np.random.default_rng(0)
        for _ in range(m):
            i = r.integers(0, n, n)
            sk_pair(y[i], s[i])

    row(f"{m:,} bootstrap replicates, N={n:,}", best(sk_boot, 1),
        best(lambda: fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=m)))

    n, g = 1_000_000, 20_000
    y = rng.integers(0, 2, n, dtype=np.uint8)
    s = rng.random(n)
    groups = rng.integers(0, g, n)

    def sk_grouped():
        order = np.argsort(groups, kind="stable")
        bounds = np.flatnonzero(np.diff(groups[order])) + 1
        for yy, ss in zip(np.split(y[order], bounds), np.split(s[order], bounds)):
            if yy.min() != yy.max():
                sk_pair(yy, ss)

    row(f"grouped, N={n:,}, {g:,} groups", best(sk_grouped, 1),
        best(lambda: fastauc.roc_auc_ap_grouped(y, s, groups)))

    def sk_ndcg():
        order = np.argsort(groups, kind="stable")
        bounds = np.flatnonzero(np.diff(groups[order])) + 1
        for yy, ss in zip(np.split(y[order], bounds), np.split(s[order], bounds)):
            if len(yy) > 1:
                ndcg_score(yy[None, :], ss[None, :], k=10)

    row(f"NDCG@10 only, N={n:,}, {g:,} groups", best(sk_ndcg, 1),
        best(lambda: fastauc.ranking_metrics(y, s, groups, ks=[10])))
    t = best(lambda: fastauc.ranking_metrics(y, s, groups, ks=[1, 5, 10, 50]))
    print(f"{'  all ranking metrics at 4 cutoffs':<44} {'':>13} {t * 1e3:>9.2f} ms")

    def sk_binary():
        sk_pair(y, s)
        log_loss(y, s)
        brier_score_loss(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        (2 * prec * rec / np.maximum(prec + rec, 1e-300)).max()
        fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
        np.abs(tpr - fpr).max()

    row(f"binary_metrics, N={n:,}", best(sk_binary, 1),
        best(lambda: fastauc.binary_metrics(y, s)))


if __name__ == "__main__":
    main()
