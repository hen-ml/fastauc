"""Calls from many Python threads at once must give the single-threaded answers.

On free-threaded CPython (3.14t) these calls really run in parallel;
on regular CPython the core releases the GIL, so they overlap there too.
"""
import os
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import fastauc as fa


def _work(seed):
    rng = np.random.default_rng(seed)
    n = 20_000
    y = rng.integers(0, 2, n)
    s = np.round(rng.normal(size=n) + y, 2)
    g = rng.integers(0, 300, n)
    acc = fa.BinaryAccumulator()
    for i in range(0, n, 3_000):
        acc.update(y[i:i + 3_000], s[i:i + 3_000])
    return (
        fa.binary_metrics(y, s)["roc_auc"],
        fa.ranking_metrics(y, s, g, ks=(5,))["ndcg@5"],
        float(fa.bootstrap_roc_auc_ap(y, s, n_bootstraps=50, seed=seed).ci()[0]),
        acc.compute()["average_precision"],
        fa.paired_test(s[:500], s[500:1000], n_resamples=500, seed=seed).p_value,
    )


def test_concurrent_calls_match_serial():
    seeds = list(range(16))
    serial = [_work(s) for s in seeds]
    with ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(_work, seeds))
    assert parallel == serial


def test_import_keeps_free_threading():
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("regular (GIL) CPython build")
    if os.environ.get("PYTHON_GIL") == "1":
        pytest.skip("GIL forced on by PYTHON_GIL=1")
    assert not sys._is_gil_enabled(), "importing fastauc re-enabled the GIL"
