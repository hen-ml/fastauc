"""Tests for roc_curve, precision_recall_curve, calibration_curve and DeLong."""
import inspect
import math

import numpy as np
import pytest
import sklearn.metrics as skm
from sklearn.calibration import calibration_curve as sk_calibration_curve

import fastauc


def make(n, seed, ties=False, prevalence=0.3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.uint8)
    p = np.clip(0.15 + 0.5 * y * rng.random(n) + 0.3 * rng.random(n), 0, 1)
    if ties:
        p = np.round(p, 1)
    return y, p


def weights(n, seed, zeros=True):
    w = np.random.default_rng(seed).random(n) * 2
    if zeros:
        w[::7] = 0.0
    return w


SK_PR_HAS_DROP = "drop_intermediate" in inspect.signature(skm.precision_recall_curve).parameters


# --------------------------------------------------------------------------
# ROC curve
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("drop", [True, False])
def test_roc_curve_matches_sklearn(ties, weighted, drop):
    y, s = make(3000, 0, ties)
    w = weights(len(y), 1) if weighted else None
    fpr, tpr, thr = fastauc.roc_curve(y, s, sample_weight=w, drop_intermediate=drop)
    sfpr, stpr, sthr = skm.roc_curve(y, s, sample_weight=w, drop_intermediate=drop)
    np.testing.assert_allclose(fpr, sfpr, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(tpr, stpr, rtol=1e-12, atol=1e-15)
    assert thr[0] == np.inf  # older scikit-learn used max(score) + 1 here
    np.testing.assert_array_equal(thr[1:], sthr[1:])
    # Trapezoidal area under the returned curve is the ROC-AUC.
    area = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    assert area == pytest.approx(fastauc.roc_auc(y, s, sample_weight=w), rel=1e-12)


def test_roc_curve_degenerate():
    fpr, tpr, thr = fastauc.roc_curve(np.ones(3), np.array([0.1, 0.5, 0.5]))
    assert np.isnan(fpr).all() and tpr[-1] == 1.0
    fpr, tpr, thr = fastauc.roc_curve(np.zeros(3), np.array([0.1, 0.5, 0.5]))
    assert np.isnan(tpr).all() and fpr[-1] == 1.0
    assert list(thr) == [np.inf, 0.5, 0.1]


# --------------------------------------------------------------------------
# Precision-recall curve
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("drop", [False, True])
def test_pr_curve_matches_sklearn(ties, weighted, drop):
    if drop and not SK_PR_HAS_DROP:
        pytest.skip("this scikit-learn has no drop_intermediate for PR curves")
    y, s = make(3000, 2, ties)
    w = weights(len(y), 3) if weighted else None
    kw = {"drop_intermediate": drop} if SK_PR_HAS_DROP else {}
    p, r, t = fastauc.precision_recall_curve(y, s, sample_weight=w, drop_intermediate=drop)
    sp, sr, st = skm.precision_recall_curve(y, s, sample_weight=w, **kw)
    np.testing.assert_allclose(p, sp, rtol=1e-12)
    np.testing.assert_allclose(r, sr, rtol=1e-12)
    np.testing.assert_array_equal(t, st)
    # Step-wise area is average precision.
    ap = -float(np.sum(np.diff(r) * p[:-1]))
    assert ap == pytest.approx(fastauc.average_precision(y, s, sample_weight=w), rel=1e-12)


def test_pr_curve_no_positives_is_nan():
    p, r, t = fastauc.precision_recall_curve(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.isnan(r[:-1]).all() and r[-1] == 0.0
    assert (p[:-1] == 0).all() and p[-1] == 1.0


def test_curves_are_order_invariant():
    y, s = make(2000, 4, ties=True)
    w = weights(len(y), 5)
    perm = np.random.default_rng(0).permutation(len(y))
    for fn in (fastauc.roc_curve, fastauc.precision_recall_curve):
        a = fn(y, s, sample_weight=w)
        b = fn(y[perm], s[perm], sample_weight=w[perm])
        for u, v in zip(a, b):
            np.testing.assert_allclose(u, v, rtol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------
# Calibration curve
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_bins", [1, 5, 10, 17])
@pytest.mark.parametrize("ties", [False, True])
def test_calibration_curve_uniform_matches_sklearn(n_bins, ties):
    y, p = make(5000, 6, ties)
    p[:5] = [0.0, 1.0, 0.2, 0.4, 0.6]  # values exactly on bin edges
    got = fastauc.calibration_curve(y, p, n_bins=n_bins)
    want = sk_calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    for g, s in zip(got, want):
        np.testing.assert_allclose(g, s, rtol=1e-12)


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
@pytest.mark.parametrize("weighted", [False, True])
def test_calibration_curve_consistent_with_ece(strategy, weighted):
    y, p = make(4000, 7, ties=True)
    w = weights(len(y), 8, zeros=False) if weighted else None
    pt, pp, bw = fastauc.calibration_curve(
        y, p, n_bins=8, strategy=strategy, sample_weight=w, return_weights=True
    )
    ece = fastauc.expected_calibration_error(y, p, n_bins=8, strategy=strategy, sample_weight=w)
    assert np.average(np.abs(pt - pp), weights=bw) == pytest.approx(ece, rel=1e-12)
    assert bw.sum() == pytest.approx(len(y) if w is None else w.sum())
    if strategy == "quantile":
        np.testing.assert_allclose(bw, bw[0], rtol=1e-12)  # equal-weight bins despite ties


def test_calibration_curve_quantile_without_ties_matches_sklearn():
    rng = np.random.default_rng(9)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    got = fastauc.calibration_curve(y, p, n_bins=10, strategy="quantile")
    want = sk_calibration_curve(y, p, n_bins=10, strategy="quantile")
    # Bin edges differ only by where each percentile cut lands between two samples.
    for g, s in zip(got, want):
        np.testing.assert_allclose(g, s, atol=0.02)


def test_calibration_curve_errors():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fastauc.calibration_curve([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError, match="n_bins"):
        fastauc.calibration_curve([0, 1], [0.5, 0.5], n_bins=0)
    with pytest.raises(ValueError, match="strategy"):
        fastauc.calibration_curve([0, 1], [0.5, 0.5], strategy="kmeans")


# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------


def delong_reference(y, scores):
    """Textbook DeLong (1988) with the O(m n) kernel."""
    y = np.asarray(y).astype(bool)
    scores = np.atleast_2d(scores)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    psi = (pos[:, :, None] > neg[:, None, :]) + 0.5 * (pos[:, :, None] == neg[:, None, :])
    auc = psi.mean(axis=(1, 2))
    v10 = psi.mean(axis=2)  # (k, m)
    v01 = psi.mean(axis=1)  # (k, n)
    s10 = np.atleast_2d(np.cov(v10))
    s01 = np.atleast_2d(np.cov(v01))
    return auc, s10 / m + s01 / n


@pytest.mark.parametrize("ties", [False, True])
def test_delong_matches_reference(ties):
    rng = np.random.default_rng(10)
    y = (rng.random(400) < 0.35).astype(int)
    base = rng.random(400) + 0.6 * y
    S = np.stack([base, base + rng.normal(0, 0.3, 400), rng.random(400) + 0.2 * y])
    if ties:
        S = np.round(S, 1)
    r = fastauc.delong_roc_auc(y, S)
    auc, cov = delong_reference(y, S)
    np.testing.assert_allclose(r.roc_auc, auc, rtol=1e-12)
    np.testing.assert_allclose(r.covariance, cov, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(r.roc_auc, fastauc.roc_auc_ap_batch(y, S)[0], rtol=1e-12)
    assert (r.n_pos, r.n_neg) == (int(y.sum()), int(len(y) - y.sum()))

    # The paired test against its own formula.
    diff, (lo, hi), p = r.compare(0, 2)
    se = math.sqrt(cov[0, 0] + cov[2, 2] - 2 * cov[0, 2])
    assert diff == pytest.approx(auc[0] - auc[2], rel=1e-12)
    assert p == pytest.approx(math.erfc(abs(diff) / se / math.sqrt(2)), rel=1e-9)
    assert lo == pytest.approx(diff - 1.959963984540054 * se, rel=1e-9)
    assert lo > 0 and p < 1e-3

    single = fastauc.delong_roc_auc(y, S[0])
    assert single.roc_auc == pytest.approx(auc[0], rel=1e-12)
    assert single.std() == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-10)
    lo, hi = single.ci()
    assert lo < single.roc_auc < hi
    assert single.ci(0.5)[0] > lo
    np.testing.assert_allclose(r.ci()[0], single.ci(), rtol=1e-10)
    via_axis = fastauc.delong_roc_auc(y, S.T, axis=0)
    np.testing.assert_array_equal(via_axis.covariance, r.covariance)


def test_delong_agrees_with_bootstrap():
    y, s = make(3000, 11)
    d = fastauc.delong_roc_auc(y, s)
    b = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=4000, seed=0)
    assert d.std() == pytest.approx(b.std(), rel=0.08)


def test_delong_coverage():
    rng = np.random.default_rng(12)
    # P(X > Y) for X ~ N(0.8, 1), Y ~ N(0, 1) is Phi(0.8 / sqrt(2)).
    true_auc = 0.5 * (1 + math.erf(0.8 / math.sqrt(2) / math.sqrt(2)))
    hits, sims = 0, 400
    for _ in range(sims):
        y = np.r_[np.ones(60), np.zeros(90)]
        s = np.r_[rng.normal(0.8, 1, 60), rng.normal(0, 1, 90)]
        lo, hi = fastauc.delong_roc_auc(y, s).ci(0.95)
        hits += lo <= true_auc <= hi
    assert 0.91 <= hits / sims <= 0.98


def test_delong_order_invariance_and_degenerate():
    y, s = make(1000, 13, ties=True)
    S = np.stack([s, np.round(s * 3, 0) / 3])
    perm = np.random.default_rng(1).permutation(len(y))
    a = fastauc.delong_roc_auc(y, S)
    b = fastauc.delong_roc_auc(y[perm], S[:, perm])
    np.testing.assert_allclose(a.covariance, b.covariance, rtol=1e-10)
    np.testing.assert_allclose(a.roc_auc, b.roc_auc, rtol=1e-12)

    same = fastauc.delong_roc_auc(y, np.stack([s, s]))
    assert same.compare(0, 1) == (0.0, (0.0, 0.0), 1.0)

    r = fastauc.delong_roc_auc(np.zeros(5), np.arange(5.0))
    assert math.isnan(r.roc_auc) and math.isnan(r.std())
    r = fastauc.delong_roc_auc([1, 0, 0], [0.9, 0.1, 0.2])  # one positive: AUC defined, variance not
    assert r.roc_auc == 1.0 and math.isnan(r.std())
    with pytest.raises(ValueError, match="2-D"):
        fastauc.delong_roc_auc(y, s).compare(0, 1)
    with pytest.raises(ValueError, match="NaN"):
        fastauc.delong_roc_auc([0, 1], [np.nan, 0.3])
    with pytest.raises(ValueError, match="mismatch"):
        fastauc.delong_roc_auc([0, 1, 1], [0.1, 0.3])


# --------------------------------------------------------------------------
# Average precision vs trapezoidal PR-AUC
# --------------------------------------------------------------------------


def test_ap_and_pr_auc_by_hand():
    # Curve points (R, P): start (0, 1); 0.9 -> (0.5, 1); 0.8 -> (0.5, 0.5); 0.7 -> (1, 2/3).
    y, s = np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])
    assert fastauc.average_precision(y, s) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    assert fastauc.pr_auc(y, s) == pytest.approx(0.5 * (1 + 1) / 2 + 0.5 * (0.5 + 2 / 3) / 2)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_pr_auc_matches_sklearn_trapezoid(ties, weighted):
    y, s = make(4000, 14, ties)
    w = weights(len(y), 15) if weighted else None  # includes zero weights
    p, r, _ = skm.precision_recall_curve(y, s, sample_weight=w)
    assert fastauc.pr_auc(y, s, sample_weight=w) == pytest.approx(skm.auc(r, p), rel=1e-12)
    # Same curve as precision_recall_curve: trapezoid of its points.
    fp, fr, _ = fastauc.precision_recall_curve(y, s, sample_weight=w)
    trap = float(np.sum(-np.diff(fr) * (fp[1:] + fp[:-1]) / 2))
    assert fastauc.pr_auc(y, s, sample_weight=w) == pytest.approx(trap, rel=1e-12)
    # ...and step-wise area of the same points is average precision.
    assert fastauc.average_precision(y, s, sample_weight=w) == pytest.approx(
        float(np.sum(-np.diff(fr) * fp[:-1])), rel=1e-12
    )


def test_pr_auc_in_binary_metrics_and_degenerate_cases():
    y, s = make(1000, 16, ties=True)
    m = fastauc.binary_metrics(y, s)
    assert m["pr_auc"] == fastauc.pr_auc(y, s)
    assert m["average_precision"] == fastauc.average_precision(y, s)
    assert m["pr_auc"] != m["average_precision"]
    assert math.isnan(fastauc.pr_auc(np.zeros(4), np.arange(4.0)))
    assert fastauc.pr_auc(np.ones(4), np.arange(4.0)) == 1.0
    assert fastauc.pr_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.75)  # (0,1) -> (1, 0.5)
    perm = np.random.default_rng(0).permutation(len(y))
    assert fastauc.pr_auc(y[perm], s[perm]) == pytest.approx(m["pr_auc"], rel=1e-12)
