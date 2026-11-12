"""Exact streaming accumulator and paired significance tests."""
import itertools
import math
import pickle

import numpy as np
import pytest

import fastauc as fa


def _same(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


@pytest.fixture(scope="module")
def stream():
    rng = np.random.default_rng(3)
    n = 60_000
    y = rng.integers(0, 2, n)
    # float16 scores: heavy ties, the case a streaming sketch gets wrong
    s = (rng.normal(size=n) + y).astype(np.float16).astype(np.float64)
    return y, 1 / (1 + np.exp(-s)), rng.random(n)


# ------------------------------------------------------------ accumulator


EXACT = ("roc_auc", "average_precision", "pr_auc", "ks", "ks_threshold", "f1_max",
         "f1_threshold", "prevalence")


@pytest.mark.parametrize("batch", [1, 997, 10_000, 60_000])
def test_accumulator_is_bit_identical_to_one_shot(stream, batch):
    y, p, _ = stream
    if batch == 1:
        y, p = y[:3000], p[:3000]
    full = fa.binary_metrics(y, p)
    acc = fa.BinaryAccumulator()
    for i in range(0, len(y), batch):
        acc.update(y[i:i + batch], p[i:i + batch])
    got = acc.compute()
    assert acc.n_samples == len(y)
    for k in EXACT:
        assert _same(got[k], full[k]), k
    for k in full:
        assert got[k] == pytest.approx(full[k], rel=1e-12, nan_ok=True), k
    assert acc.roc_auc() == full["roc_auc"]
    assert acc.average_precision() == full["average_precision"]
    assert acc.pr_auc() == full["pr_auc"]


def test_accumulator_curves(stream):
    y, p, _ = stream
    acc = fa.BinaryAccumulator().update(y[:30_000], p[:30_000]).update(y[30_000:], p[30_000:])
    for ours, ref in ((acc.roc_curve(), fa.roc_curve(y, p)),
                      (acc.precision_recall_curve(), fa.precision_recall_curve(y, p)),
                      (acc.roc_curve(drop_intermediate=True),
                       fa.roc_curve(y, p, drop_intermediate=True))):
        for a, b in zip(ours, ref):
            np.testing.assert_array_equal(a, b)


def test_accumulator_merge_in_any_order(stream):
    y, p, _ = stream
    parts = [fa.BinaryAccumulator().update(y[i::4], p[i::4]) for i in range(4)]
    full = fa.binary_metrics(y, p)
    for perm in itertools.permutations(range(4)):
        acc = fa.BinaryAccumulator()
        for i in perm:
            acc += parts[i]
        assert acc.compute()["roc_auc"] == full["roc_auc"]
    total = parts[0] + parts[1] + parts[2] + parts[3]
    assert total.n_samples == len(y)
    assert parts[0].n_samples == len(y[0::4])  # + does not mutate its operands


def test_accumulator_weighted(stream):
    y, p, w = stream
    full = fa.binary_metrics(y, p, sample_weight=w)
    acc = fa.BinaryAccumulator()
    for i in range(0, len(y), 7_000):
        acc.update(y[i:i + 7_000], p[i:i + 7_000], sample_weight=w[i:i + 7_000])
    got = acc.compute()
    for k in full:
        assert got[k] == pytest.approx(full[k], rel=1e-11, nan_ok=True), k


def test_accumulator_state_pickle_and_reset(stream):
    y, p, _ = stream
    acc = fa.BinaryAccumulator().update(y, p)
    clone = pickle.loads(pickle.dumps(acc))
    assert clone.compute()["roc_auc"] == acc.compute()["roc_auc"]
    rebuilt = fa.BinaryAccumulator.from_state(acc.state())
    assert rebuilt.n_distinct == acc.n_distinct and rebuilt.n_samples == acc.n_samples
    st = acc.state()
    st["score"] = st["score"][::-1].copy()
    with pytest.raises(ValueError, match="descending"):
        fa.BinaryAccumulator.from_state(st)
    acc.reset()
    assert acc.n_samples == 0 and acc.n_distinct == 0
    assert math.isnan(acc.roc_auc())


def test_accumulator_memory_is_distinct_scores():
    acc = fa.BinaryAccumulator()
    rng = np.random.default_rng(0)
    for _ in range(20):
        acc.update(rng.integers(0, 2, 10_000), rng.integers(0, 100, 10_000))
    assert acc.n_samples == 200_000
    assert acc.n_distinct == 100


def test_accumulator_pos_label_and_signed_zero():
    acc = fa.BinaryAccumulator(pos_label="spam")
    acc.update(["spam", "ham", "ham"], [0.0, -0.0, 1.0])
    assert acc.n_distinct == 2  # -0.0 and 0.0 are the same score
    assert acc.roc_auc() == fa.roc_auc(["spam", "ham", "ham"], [0.0, -0.0, 1.0], pos_label="spam")
    with pytest.raises(ValueError, match="NaN"):
        acc.update(["spam"], [float("nan")])


def test_accumulator_rejects_other_types():
    with pytest.raises(TypeError):
        fa.BinaryAccumulator().merge(object())


def test_accumulator_rejects_different_pos_label():
    a = fa.BinaryAccumulator(pos_label=1).update([0, 1, 1], [0.1, 0.2, 0.3])
    b = fa.BinaryAccumulator(pos_label=0).update([0, 1, 1], [0.1, 0.2, 0.3])
    before = a.state()
    with pytest.raises(ValueError, match="pos_label"):
        a.merge(b)
    with pytest.raises(ValueError, match="pos_label"):
        a + fa.BinaryAccumulator()  # default None differs from 1
    after = a.state()  # a failed merge leaves the accumulator untouched
    assert after["n_samples"] == before["n_samples"]
    np.testing.assert_array_equal(after["pos"], before["pos"])
    c = fa.BinaryAccumulator(pos_label=np.int64(1)).update([1, 0], [0.5, 0.4])
    assert a.copy().merge(c).n_samples == 5  # equal labels of different types merge


# ------------------------------------------------------------ paired tests


def _brute_sign_flip(d, alternative):
    obs = d.sum()
    tol = 1e-12 * np.abs(d).sum()
    hits = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        s = float(np.dot(signs, d))
        hits += {"two-sided": abs(s) >= abs(obs) - tol,
                 "greater": s >= obs - tol,
                 "less": s <= obs + tol}[alternative]
    return hits / 2 ** len(d)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_randomization_exact_matches_enumeration(alternative):
    rng = np.random.default_rng(5)
    a = rng.normal(size=10)
    b = a + rng.normal(0.4, 1.0, 10)
    r = fa.paired_test(a, b, alternative=alternative)
    assert r.exact and r.n == 10
    assert r.p_value == pytest.approx(_brute_sign_flip(a - b, alternative), abs=1e-15)


def test_randomization_monte_carlo_is_deterministic_and_close():
    rng = np.random.default_rng(6)
    a = rng.normal(0.5, 0.1, 400)
    b = a + rng.normal(0.01, 0.05, 400)
    r1 = fa.paired_test(a, b, n_resamples=20_000, seed=7)
    fa.set_num_threads(1)
    try:
        r2 = fa.paired_test(a, b, n_resamples=20_000, seed=7)
    finally:
        fa.set_num_threads(None)
    assert not r1.exact
    assert r1.p_value == r2.p_value
    # A large-sample sign-flip test agrees with the t-test to Monte Carlo error.
    t = fa.paired_test(a, b, method="t").p_value
    assert abs(r1.p_value - t) < 4 * math.sqrt(t * (1 - t) / 20_000) + 2e-3


def test_ties_with_the_observed_sum_count():
    # All differences equal: every pattern but the identity and its mirror is
    # less extreme, so the exact two-sided p-value is 2 / 2**n.
    r = fa.paired_test(np.full(8, 0.3), np.zeros(8))
    assert r.p_value == pytest.approx(2 / 256)


def test_nan_pairs_are_dropped_and_degenerate_cases():
    a = np.array([0.5, np.nan, 0.7, 0.2])
    b = np.array([0.4, 0.1, np.nan, 0.1])
    r = fa.paired_test(a, b)
    assert r.n == 2
    assert math.isnan(fa.paired_test([np.nan], [1.0]).p_value)
    assert math.isnan(fa.paired_test([1.0], [0.0], method="t").p_value)
    assert fa.paired_test([1.0, 1.0], [1.0, 1.0]).p_value == 1.0
    with pytest.raises(ValueError):
        fa.paired_test([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        fa.paired_test([1, 2], [1, 2], method="wilcoxon")


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_t_test_matches_scipy(alternative):
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(8)
    for n in (3, 12, 300):
        a = rng.normal(size=n)
        b = a + rng.normal(0.2, 1.0, n)
        ours = fa.paired_test(a, b, method="t", alternative=alternative)
        ref = stats.ttest_rel(a, b, alternative=alternative)
        assert ours.statistic == pytest.approx(ref.statistic, rel=1e-12)
        assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-10, abs=1e-300)


def test_t_cdf_tails():
    stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 5, 30, 1000):
        for t in (-40.0, -3.0, -0.5, 0.0, 0.5, 3.0, 40.0):
            assert fa._t_sf(t, df) == pytest.approx(stats.t.sf(t, df), rel=1e-10, abs=1e-300)


def test_holm():
    # Reference values from R: p.adjust(c(0.01, 0.04, 0.03, 0.005), "holm")
    np.testing.assert_allclose(fa.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    out = fa.holm([0.5, np.nan, 0.01])
    assert math.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [0.5, 0.02])
    np.testing.assert_allclose(fa.holm([0.9, 0.8]), [1.0, 1.0])
    assert fa.holm([]).shape == (0,)


def test_paired_test_on_ranking_output():
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(200), 10)
    y = rng.integers(0, 2, len(groups))
    good = y + rng.normal(0, 1.0, len(y))
    bad = y + rng.normal(0, 3.0, len(y))
    _, pa = fa.ranking_metrics(y, good, groups, ks=(10,), per_group=True)
    _, pb = fa.ranking_metrics(y, bad, groups, ks=(10,), per_group=True)
    r = fa.paired_test(pa["ndcg@10"], pb["ndcg@10"], alternative="greater")
    assert r.mean_difference > 0 and r.p_value < 0.01
