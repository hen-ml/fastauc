import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

import fastauc

RTOL = 1e-10


def make(n, seed, ties=False, prevalence=0.5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.uint8)
    s = rng.random(n) + 0.3 * y
    if ties:
        s = np.round(s, 1)
    return y, s


def sk(y, s, w=None):
    return (
        roc_auc_score(y, s, sample_weight=w),
        average_precision_score(y, s, sample_weight=w),
    )


# --------------------------------------------------------------------------
# Parity with scikit-learn
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 10, 257, 5000, 200_000])
@pytest.mark.parametrize("ties", [False, True])
def test_parity(n, ties):
    for seed in range(5):
        y, s = make(n, seed, ties)
        if y.min() == y.max():
            continue
        np.testing.assert_allclose(fastauc.roc_pr_auc(y, s), sk(y, s), rtol=RTOL)


@pytest.mark.parametrize("ties", [False, True])
def test_parity_weighted(ties):
    rng = np.random.default_rng(0)
    y, s = make(3000, 1, ties)
    w = rng.random(3000)
    w[rng.random(3000) < 0.1] = 0.0
    got = fastauc.roc_pr_auc(y, s, sample_weight=w)
    np.testing.assert_allclose(got, sk(y, s, w), rtol=RTOL)


def test_parity_imbalanced():
    y, s = make(100_000, 3, prevalence=0.001)
    np.testing.assert_allclose(fastauc.roc_pr_auc(y, s), sk(y, s), rtol=RTOL)


def test_single_metric_functions():
    y, s = make(1000, 0)
    roc, ap = sk(y, s)
    assert fastauc.roc_auc(y, s) == pytest.approx(roc, rel=RTOL)
    assert fastauc.average_precision(y, s) == pytest.approx(ap, rel=RTOL)


def test_all_tied_scores():
    y = np.array([0, 1, 1, 0, 1])
    s = np.zeros(5)
    roc, ap = fastauc.roc_pr_auc(y, s)
    assert roc == pytest.approx(0.5)
    assert ap == pytest.approx(0.6)


def test_signed_zero_and_infinities():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.0, -0.0, np.inf, -np.inf, 1.0, 1.0])
    # scikit-learn rejects infinities, so compare against a rank-equivalent vector.
    finite = np.array([0.0, 0.0, 2.0, -1.0, 1.0, 1.0])
    np.testing.assert_allclose(fastauc.roc_pr_auc(y, s), sk(y, finite), rtol=RTOL)


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def test_input_types():
    y, s = make(500, 0)
    ref = fastauc.roc_pr_auc(y, s)
    assert fastauc.roc_pr_auc(y.astype(bool), s) == ref
    assert fastauc.roc_pr_auc(y.astype(np.int64), s) == ref
    assert fastauc.roc_pr_auc(y.astype(float), s) == ref
    assert fastauc.roc_pr_auc(y.tolist(), s.tolist()) == ref
    # float32 scores, non-contiguous views
    s32 = s.astype(np.float32)
    np.testing.assert_allclose(fastauc.roc_pr_auc(y, s32), sk(y, s32), rtol=RTOL)
    np.testing.assert_allclose(
        fastauc.roc_pr_auc(y[::2], s[::2]), sk(y[::2], s[::2]), rtol=RTOL
    )


def test_pos_label():
    y, s = make(500, 0)
    ref = fastauc.roc_pr_auc(y, s)
    signed = np.where(y == 1, 1, -1)
    assert fastauc.roc_pr_auc(signed, s, pos_label=1) == ref
    names = np.where(y == 1, "spam", "ham")
    assert fastauc.roc_pr_auc(names, s, pos_label="spam") == ref


def test_errors():
    y, s = make(100, 0)
    with pytest.raises(ValueError, match="0 and 1"):
        fastauc.roc_pr_auc(np.where(y == 1, 1, -1), s)
    with pytest.raises(ValueError, match="0 and 1"):
        fastauc.roc_pr_auc(np.arange(100) % 3, s)
    with pytest.raises(TypeError):
        fastauc.roc_pr_auc(np.array(["a", "b"] * 50), s)
    with pytest.raises(ValueError, match="mismatch"):
        fastauc.roc_pr_auc(y[:-1], s)
    bad = s.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fastauc.roc_pr_auc(y, bad)
    with pytest.raises(ValueError, match="sample_weight"):
        fastauc.roc_pr_auc(y, s, sample_weight=-np.ones(100))
    with pytest.raises(ValueError, match="sample_weight"):
        fastauc.roc_pr_auc(y, s, sample_weight=np.ones(99))
    with pytest.raises(ValueError, match="1-D"):
        fastauc.roc_pr_auc(y, np.stack([s, s]))


def test_degenerate_returns_nan():
    s = np.array([0.1, 0.4, 0.3])
    roc, ap = fastauc.roc_pr_auc(np.zeros(3), s)
    assert math.isnan(roc) and math.isnan(ap)
    roc, ap = fastauc.roc_pr_auc(np.ones(3), s)
    assert math.isnan(roc) and ap == 1.0
    roc, ap = fastauc.roc_pr_auc(np.array([]), np.array([]))
    assert math.isnan(roc) and math.isnan(ap)


# --------------------------------------------------------------------------
# Batched
# --------------------------------------------------------------------------


def test_batch_many_scores():
    rng = np.random.default_rng(0)
    y, _ = make(2000, 0)
    S = np.round(rng.random((37, 2000)) + 0.2 * y, 2)
    w = rng.random(2000)
    roc, ap = fastauc.roc_pr_auc_batch(y, S)
    ref = np.array([sk(y, row) for row in S])
    np.testing.assert_allclose(roc, ref[:, 0], rtol=RTOL)
    np.testing.assert_allclose(ap, ref[:, 1], rtol=RTOL)
    roc, ap = fastauc.roc_pr_auc_batch(y, S, sample_weight=w)
    ref = np.array([sk(y, row, w) for row in S])
    np.testing.assert_allclose(roc, ref[:, 0], rtol=RTOL)
    np.testing.assert_allclose(ap, ref[:, 1], rtol=RTOL)


def test_batch_many_targets():
    rng = np.random.default_rng(1)
    s = np.round(rng.random(2000), 2)
    Y = rng.random((41, 2000)) < 0.3
    roc, ap = fastauc.roc_pr_auc_batch(Y, s)
    ref = np.array([sk(row, s) for row in Y])
    np.testing.assert_allclose(roc, ref[:, 0], rtol=RTOL)
    np.testing.assert_allclose(ap, ref[:, 1], rtol=RTOL)


def test_batch_pairwise_and_views():
    rng = np.random.default_rng(2)
    S = rng.random((2000, 13))  # (n_samples, n_tasks): the scikit-learn layout
    Y = rng.random((2000, 13)) < 0.4
    roc, ap = fastauc.roc_pr_auc_batch(Y.T, S.T)  # non-contiguous transposes
    ref = np.array([sk(Y[:, j], S[:, j]) for j in range(13)])
    np.testing.assert_allclose(roc, ref[:, 0], rtol=RTOL)
    np.testing.assert_allclose(ap, ref[:, 1], rtol=RTOL)


def test_batch_degenerate_row_is_nan_not_error():
    rng = np.random.default_rng(3)
    Y = rng.random((3, 100)) < 0.5
    Y[1] = False
    roc, ap = fastauc.roc_pr_auc_batch(Y, rng.random(100))
    assert np.isnan(roc[1]) and np.isnan(ap[1])
    assert not np.isnan(roc[[0, 2]]).any()


def test_batch_shape_errors():
    y, s = make(100, 0)
    with pytest.raises(ValueError, match="unsupported shapes"):
        fastauc.roc_pr_auc_batch(y, s)
    with pytest.raises(ValueError, match="mismatch"):
        fastauc.roc_pr_auc_batch(y, np.zeros((4, 99)))
    with pytest.raises(ValueError, match="mismatch"):
        fastauc.roc_pr_auc_batch(np.zeros((3, 100)), np.zeros((4, 100)))
    with pytest.raises(ValueError, match="NaN"):
        fastauc.roc_pr_auc_batch(y, np.full((2, 100), np.nan))


# --------------------------------------------------------------------------
# Grouped
# --------------------------------------------------------------------------


def test_grouped():
    rng = np.random.default_rng(0)
    n = 20_000
    y, s = make(n, 0, ties=True)
    g = rng.integers(0, 300, n) * 7 - 50  # arbitrary, unsorted ids
    res = fastauc.roc_pr_auc_grouped(y, s, g)
    assert len(res.groups) == len(np.unique(g))
    for i, gid in enumerate(res.groups):
        m = g == gid
        assert res.n_pos[i] == y[m].sum()
        assert res.n_neg[i] == (1 - y[m]).sum()
        np.testing.assert_allclose(
            (res.roc_auc[i], res.average_precision[i]), sk(y[m], s[m]), rtol=RTOL
        )
    sizes = res.n_pos + res.n_neg
    assert res.mean() == pytest.approx(np.average(res.roc_auc, weights=sizes))
    assert res.mean("average_precision", weight="uniform") == pytest.approx(
        res.average_precision.mean()
    )


def test_grouped_string_ids_and_degenerate_groups():
    y = np.array([1, 0, 1, 1, 0, 0])
    s = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3])
    g = np.array(["q2", "q2", "q1", "q1", "q3", "q2"])
    res = fastauc.roc_pr_auc_grouped(y, s, g)
    assert list(res.groups) == ["q1", "q2", "q3"]
    assert np.isnan(res.roc_auc[0])  # q1: positives only
    assert res.roc_auc[1] == 1.0
    assert np.isnan(res.roc_auc[2]) and np.isnan(res.average_precision[2])
    assert res.mean() == 1.0  # undefined groups are skipped


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def naive_bootstrap(y, s, b, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        idx = rng.integers(0, len(y), len(y))
        out.append(sk(y[idx], s[idx]))
    return np.array(out)


def test_bootstrap_shapes_and_reproducibility():
    y, s = make(2000, 0)
    a = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=200, seed=7)
    b = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=200, seed=7)
    c = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=200, seed=8)
    assert a.roc_auc.shape == a.average_precision.shape == (200,)
    np.testing.assert_array_equal(a.roc_auc, b.roc_auc)
    assert not np.array_equal(a.roc_auc, c.roc_auc)
    assert (a.roc_auc_estimate, a.average_precision_estimate) == fastauc.roc_pr_auc(y, s)
    # a longer run extends a shorter one
    d = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=50, seed=7)
    np.testing.assert_array_equal(a.roc_auc[:50], d.roc_auc)


def test_bootstrap_matches_naive_distribution():
    y, s = make(1500, 0, ties=True)
    fast = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=4000, seed=1)
    ref = naive_bootstrap(y, s, 1500, seed=2)
    for got, want in ((fast.roc_auc, ref[:, 0]), (fast.average_precision, ref[:, 1])):
        assert got.mean() == pytest.approx(want.mean(), abs=4 * want.std() / 30)
        assert got.std() == pytest.approx(want.std(), rel=0.1)


def test_bootstrap_replicates_are_independent():
    # Regression test: replicate b+1 must not be a shifted copy of replicate b.
    y, s = make(20_000, 0)
    r = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=2000, seed=0).roc_auc
    lag1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    assert abs(lag1) < 0.1


def test_bootstrap_ci_covers_estimate():
    y, s = make(5000, 0)
    r = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=1000)
    for metric in ("roc_auc", "average_precision"):
        lo, hi = r.ci(metric)
        assert lo < getattr(r, metric + "_estimate") < hi
        lo90, hi90 = r.ci(metric, level=0.90)
        assert lo <= lo90 and hi90 <= hi
        assert r.std(metric) > 0


def test_bootstrap_stratified():
    y, s = make(3000, 0, prevalence=0.003)
    assert 0 < y.sum() < 20
    plain = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=500)
    strat = fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=500, stratified=True)
    assert not np.isnan(strat.roc_auc).any()
    assert np.nanstd(strat.roc_auc) <= np.nanstd(plain.roc_auc) * 1.2
    lo, hi = strat.ci()
    assert lo < strat.roc_auc_estimate < hi


def test_bootstrap_multi_model_is_paired():
    rng = np.random.default_rng(0)
    y, s = make(4000, 0)
    good = s
    noisy = s + rng.normal(0, 0.05, 4000)  # nearly the same model
    bad = rng.random(4000)
    r = fastauc.bootstrap_roc_pr_auc(y, np.stack([good, noisy, bad]), n_bootstraps=1000, seed=3)
    assert r.roc_auc.shape == (3, 1000)
    assert r.roc_auc_estimate.shape == (3,)
    assert r.ci().shape == (3, 2)
    # Row 0 equals the single-model run with the same seed.
    solo = fastauc.bootstrap_roc_pr_auc(y, good, n_bootstraps=1000, seed=3)
    np.testing.assert_array_equal(r.roc_auc[0], solo.roc_auc)
    # Pairing: similar models move together across replicates.
    assert np.corrcoef(r.roc_auc[0], r.roc_auc[1])[0, 1] > 0.8

    diff, (lo, hi), p = r.compare(0, 2)
    assert diff == pytest.approx(r.roc_auc_estimate[0] - r.roc_auc_estimate[2])
    assert lo > 0 and p < 0.01
    _, (lo, hi), p = r.compare(0, 0)
    assert lo == hi == 0.0 and p == 1.0
    with pytest.raises(ValueError):
        solo.compare(0, 1)


def test_bootstrap_errors():
    y, s = make(100, 0)
    with pytest.raises(ValueError):
        fastauc.bootstrap_roc_pr_auc(y, s, n_bootstraps=0)
    with pytest.raises(ValueError, match="mismatch"):
        fastauc.bootstrap_roc_pr_auc(y, s[:-1])


def test_num_threads():
    assert fastauc.num_threads() >= 1
