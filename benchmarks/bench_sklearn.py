"""fastauc.sklearn against sklearn.metrics, same calls, same results.

    python benchmarks/bench_sklearn.py

Every row checks that both libraries return the same value before timing.
"""
import time
import warnings

import numpy as np
from sklearn import metrics as sm

import fastauc
from fastauc import sklearn as fm


def best(fn, repeat=5):
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def row(label, name, *args, **kw):
    ours = getattr(fm, name)(*args, **kw)
    theirs = getattr(sm, name)(*args, **kw)
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-12)
    t_sk = best(lambda: getattr(sm, name)(*args, **kw))
    t_fa = best(lambda: getattr(fm, name)(*args, **kw))
    print(f"| {label} | {t_sk * 1e3:,.1f} ms | {t_fa * 1e3:,.1f} ms | {t_sk / t_fa:,.0f}x |")


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    print(f"scikit-learn {__import__('sklearn').__version__}, fastauc {fastauc.__version__}, "
          f"{fastauc.num_threads()} threads\n")
    print("| Call | scikit-learn | fastauc.sklearn | Speedup |")
    print("|---|---:|---:|---:|")

    n = 1_000_000
    y = rng.integers(0, 2, n)
    s = rng.normal(size=n) + y
    p = 1 / (1 + np.exp(-s))
    w = rng.random(n)
    row("`roc_auc_score`, N = 1M", "roc_auc_score", y, s)
    row("`roc_auc_score`, N = 1M, weighted", "roc_auc_score", y, s, sample_weight=w)
    row("`roc_auc_score(max_fpr=0.1)`, N = 1M", "roc_auc_score", y, s, max_fpr=0.1)
    row("`average_precision_score`, N = 1M", "average_precision_score", y, s)
    row("`log_loss`, N = 1M", "log_loss", y, p)

    k = 10
    yc = rng.integers(0, k, 200_000)
    P = rng.dirichlet(np.ones(k), 200_000)
    row("`roc_auc_score(multi_class='ovr')`, 10 classes, N = 200k", "roc_auc_score", yc, P,
        multi_class="ovr")
    row("`roc_auc_score(multi_class='ovo')`, 10 classes, N = 200k", "roc_auc_score", yc, P,
        multi_class="ovo")

    Y = rng.integers(0, 2, (200_000, 20))
    S = rng.normal(size=(200_000, 20)) + Y
    row("`roc_auc_score`, multilabel, 20 labels, N = 200k", "roc_auc_score", Y, S)

    G = rng.integers(0, 4, (10_000, 100))
    R = np.round(rng.normal(size=(10_000, 100)), 1)
    row("`ndcg_score(k=10)`, 10,000 queries x 100 docs", "ndcg_score", G, R, k=10)


if __name__ == "__main__":
    main()
