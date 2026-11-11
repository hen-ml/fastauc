"""Tests for threshold, calibration and grouped ranking metrics."""
import itertools
import math

import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    ndcg_score,
    precision_recall_curve,
    roc_curve,
)

import fastauc

RTOL = 1e-9


def make(n, seed, ties=False, prevalence=0.3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.uint8)
    p = np.clip(0.15 + 0.5 * y * rng.random(n) + 0.3 * rng.random(n), 0, 1)
    if ties:
        p = np.round(p, 1)
    return y, p


# --------------------------------------------------------------------------
# Threshold metrics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ties", [False, True])
def test_ks(ties):
    y, s = make(5000, 0, ties)
    ks, thr = fastauc.ks_statistic(y, s)
    assert ks == pytest.approx(ks_2samp(s[y == 1], s[y == 0]).statistic, rel=RTOL)
    pred = s >= thr
    assert abs(pred[y == 1].mean() - pred[y == 0].mean()) == pytest.approx(ks, rel=RTOL)


@pytest.mark.parametrize("ties", [False, True])
def test_max_f1(ties):
    y, s = make(5000, 1, ties)
    prec, rec, thr = precision_recall_curve(y, s)
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-300)
    f1, t = fastauc.max_f1(y, s)
    assert f1 == pytest.approx(f.max(), rel=RTOL)
    pred = s >= t
    tp = (pred & (y == 1)).sum()
    assert 2 * tp / (pred.sum() + y.sum()) == pytest.approx(f1, rel=RTOL)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("target", [0.0, 0.2, 0.8, 0.95, 1.0])
def test_operating_points(ties, target):
    y, s = make(4000, 2, ties)
    prec, rec, _ = precision_recall_curve(y, s)
    prec, rec = prec[:-1], rec[:-1]  # drop sklearn's synthetic (1, 0) end point

    value, thr = fastauc.precision_at_recall(y, s, target)
    assert value == pytest.approx(prec[rec >= target].max(), rel=RTOL)
    pred = s >= thr
    assert (pred & (y == 1)).sum() / pred.sum() == pytest.approx(value, rel=RTOL)

    value, thr = fastauc.recall_at_precision(y, s, target)
    ok = prec >= target
    assert value == pytest.approx(rec[ok].max() if ok.any() else 0.0, rel=RTOL)

    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    value, thr = fastauc.tpr_at_fpr(y, s, target)
    assert value == pytest.approx(tpr[fpr <= target].max(), rel=RTOL)
    if math.isfinite(thr):
        pred = s >= thr
        assert pred[y == 1].mean() == pytest.approx(value, rel=RTOL)
        assert pred[y == 0].mean() <= target + 1e-12


def test_recall_at_unreachable_precision():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6])
    assert fastauc.recall_at_precision(y, s, 0.9) == (0.0, math.inf)
    assert fastauc.tpr_at_fpr(y, s, 0.0) == (0.0, math.inf)


def test_lift():
    y, s = make(10_000, 3)
    order = np.argsort(-s)
    for frac in (0.01, 0.1, 0.5, 1.0):
        k = int(frac * len(y))
        want = y[order[:k]].mean() / y.mean()
        assert fastauc.lift_at(y, s, frac) == pytest.approx(want, rel=RTOL)
    # Constant scores carry no information: lift is exactly 1 at any depth.
    assert fastauc.lift_at(y, np.zeros(len(y)), 0.1) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fastauc.lift_at(y, s, 0.0)


def test_threshold_metrics_weighted_equals_repetition():
    y, s = make(600, 4, ties=True)
    w = np.random.default_rng(0).integers(0, 4, 600)
    yr, sr = np.repeat(y, w), np.repeat(s, w)
    wf = w.astype(float)
    for fn, args in [
        (fastauc.ks_statistic, ()),
        (fastauc.max_f1, ()),
        (fastauc.precision_at_recall, (0.7,)),
        (fastauc.recall_at_precision, (0.6,)),
        (fastauc.tpr_at_fpr, (0.1,)),
        (fastauc.lift_at, (0.2,)),
    ]:
        np.testing.assert_allclose(
            fn(y, s, *args, sample_weight=wf), fn(yr, sr, *args), rtol=RTOL
        )


def test_threshold_metrics_degenerate():
    s = np.array([0.1, 0.5, 0.9])
    assert all(math.isnan(v) for v in fastauc.ks_statistic(np.zeros(3), s))
    assert math.isnan(fastauc.max_f1(np.zeros(3), s)[0])
    assert math.isnan(fastauc.ks_statistic(np.ones(3), s)[0])
    assert fastauc.max_f1(np.ones(3), s)[0] == 1.0
    assert math.isnan(fastauc.lift_at(np.zeros(3), s))
    with pytest.raises(ValueError):
        fastauc.precision_at_recall(np.ones(3), s, 1.5)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def ece_reference(y, p, w, edges_bins):
    total = 0.0
    for b in np.unique(edges_bins):
        m = edges_bins == b
        total += w[m].sum() * abs(
            np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m])
        )
    return total / w.sum()


@pytest.mark.parametrize("weighted", [False, True])
def test_calibration(weighted):
    y, p = make(20_000, 5)
    w = np.random.default_rng(1).random(len(y)) + 0.1 if weighted else None
    ww = np.ones(len(y)) if w is None else w
    m = fastauc.calibration_metrics(y, p, sample_weight=w, n_bins=15)

    ll = log_loss(y, p, sample_weight=w)
    assert m["log_loss"] == pytest.approx(ll, rel=1e-9)
    assert m["brier"] == pytest.approx(brier_score_loss(y, p, sample_weight=w), rel=1e-9)
    rate = np.average(y, weights=ww)
    base = log_loss(y, np.full(len(y), rate), sample_weight=w)
    assert m["normalized_entropy"] == pytest.approx(ll / base, rel=1e-9)
    assert m["calibration_ratio"] == pytest.approx((ww * p).sum() / (ww * y).sum(), rel=1e-9)
    bins = np.searchsorted(np.linspace(0, 1, 16)[1:-1], p)
    assert m["ece"] == pytest.approx(ece_reference(y, p, ww, bins), rel=1e-9)

    assert fastauc.log_loss(y, p, sample_weight=w) == m["log_loss"]
    assert fastauc.brier_score(y, p, sample_weight=w) == m["brier"]
    assert fastauc.normalized_entropy(y, p, sample_weight=w) == m["normalized_entropy"]
    assert fastauc.calibration_ratio(y, p, sample_weight=w) == m["calibration_ratio"]
    assert fastauc.expected_calibration_error(
        y, p, sample_weight=w, n_bins=15
    ) == pytest.approx(m["ece"], rel=1e-12)


def test_ece_quantile_and_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(np.uint8)  # perfectly calibrated
    for strategy in ("uniform", "quantile"):
        assert fastauc.expected_calibration_error(y, p, strategy=strategy) < 0.01
    assert fastauc.calibration_ratio(y, p) == pytest.approx(1.0, abs=0.01)
    assert fastauc.normalized_entropy(y, p) < 1.0
    order = np.argsort(p, kind="stable")
    bins = np.empty(len(p), dtype=int)
    bins[order] = np.arange(len(p)) * 10 // len(p)
    want = ece_reference(y, p, np.ones(len(p)), bins)
    assert fastauc.expected_calibration_error(y, p, strategy="quantile") == pytest.approx(want)
    # Shrinking every probability breaks calibration.
    assert fastauc.expected_calibration_error(y, p * 0.5) > 0.2
    assert fastauc.calibration_ratio(y, p * 0.5) == pytest.approx(0.5, abs=0.01)


def test_calibration_edge_cases():
    y = np.array([1, 0, 1, 0])
    assert math.isfinite(fastauc.log_loss(y, np.array([1.0, 0.0, 0.0, 1.0])))  # clipped
    assert fastauc.log_loss(y, np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fastauc.log_loss(y, np.array([0.1, 0.2, 1.3, 0.4]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fastauc.expected_calibration_error(y, np.array([0.1, 0.2, -0.3, 0.4]))
    m = fastauc.calibration_metrics(np.zeros(4), np.full(4, 0.2))
    assert math.isnan(m["normalized_entropy"]) and math.isnan(m["calibration_ratio"])
    assert m["brier"] == pytest.approx(0.04)


def test_binary_metrics():
    y, p = make(8000, 6)
    m = fastauc.binary_metrics(y, p)
    roc, ap = fastauc.roc_auc_ap(y, p)
    assert (m["roc_auc"], m["average_precision"]) == (roc, ap)
    assert (m["ks"], m["ks_threshold"]) == fastauc.ks_statistic(y, p)
    assert (m["f1_max"], m["f1_threshold"]) == fastauc.max_f1(y, p)
    assert m["prevalence"] == y.mean()
    for key, value in fastauc.calibration_metrics(y, p).items():
        assert m[key] == value

    # Raw logits: ranking metrics unchanged, calibration block is NaN.
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
    m2 = fastauc.binary_metrics(y, logits)
    assert m2["roc_auc"] == pytest.approx(roc) and m2["ks"] == pytest.approx(m["ks"])
    assert math.isnan(m2["log_loss"]) and math.isnan(m2["ece"])
    assert set(m2) == set(m)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fastauc.binary_metrics(y, logits, calibration=True)
    assert math.isnan(fastauc.binary_metrics(y, p, calibration=False)["brier"])


# --------------------------------------------------------------------------
# Ranking: brute-force reference
# --------------------------------------------------------------------------


def one_ordering(rel, gains, k):
    """Plain textbook metrics for a list already in ranked order."""
    n = len(rel)
    kk = n if k is None else min(k, n)
    pos = rel > 0
    total = pos.sum()
    disc = 1 / np.log2(np.arange(2, n + 2))
    hits = pos[:kk]
    first = np.flatnonzero(hits)
    prec_at = np.cumsum(pos) / np.arange(1, n + 1)
    return {
        "ndcg": (gains[:kk] * disc[:kk]).sum() / (np.sort(gains)[::-1][:kk] * disc[:kk]).sum(),
        "mrr": 1 / (first[0] + 1) if len(first) else 0.0,
        "map": (prec_at[:kk] * hits).sum() / min(total, kk),
        "recall": hits.sum() / total,
        "precision": hits.sum() / (n if k is None else k),
        "hit_rate": float(hits.any()),
    }


def brute_force(rel, score, k, gain):
    """Average the textbook metrics over every ordering consistent with the scores."""
    gains = np.exp2(rel) - 1 if gain == "exponential" else rel.astype(float)
    blocks = [np.flatnonzero(score == v) for v in np.unique(score)[::-1]]
    acc, count = {}, 0
    for perm in itertools.product(*(itertools.permutations(b) for b in blocks)):
        order = np.concatenate([np.array(p, dtype=int) for p in perm])
        for name, v in one_ordering(rel[order], gains[order], k).items():
            acc[name] = acc.get(name, 0.0) + v
        count += 1
    return {name: v / count for name, v in acc.items()}


def suffix(k):
    return "" if k is None else f"@{k}"


@pytest.mark.parametrize("gain", ["exponential", "linear"])
def test_ranking_matches_brute_force_with_ties(gain):
    rng = np.random.default_rng(0)
    ks = (1, 2, 3, 5, None)
    checked = 0
    for _ in range(120):
        n = int(rng.integers(1, 8))
        rel = rng.integers(0, 4, n) * (rng.random(n) < 0.6)
        if rel.max() == 0:
            continue
        score = rng.integers(0, int(rng.integers(1, 5)), n).astype(float)  # heavy ties
        got = fastauc.ranking_metrics(rel, score, group_sizes=[n], ks=ks, gain=gain)
        for k in ks:
            want = brute_force(rel, score, k, gain)
            for name, v in want.items():
                assert got[name + suffix(k)] == pytest.approx(v, rel=1e-9, abs=1e-12), (
                    name, k, rel, score,
                )
        checked += 1
    assert checked > 60


def test_ranking_constant_scores_do_not_leak_row_order():
    # Positives listed first, model outputs a constant: must not look perfect.
    rel = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    a = fastauc.ranking_metrics(rel, np.zeros(10), group_sizes=[10], ks=(1, 3))
    b = fastauc.ranking_metrics(rel[::-1], np.zeros(10), group_sizes=[10], ks=(1, 3))
    assert a == b
    assert a["precision@1"] == pytest.approx(0.2)
    assert a["hit_rate@3"] == pytest.approx(1 - (8 / 10) * (7 / 9) * (6 / 8))


def test_ranking_many_groups():
    rng = np.random.default_rng(1)
    n, n_groups = 30_000, 2_000
    g = rng.integers(0, n_groups, n)
    rel = rng.integers(0, 3, n) * (rng.random(n) < 0.3)
    s = rng.random(n) + 0.2 * rel  # continuous: no ties
    ks = (1, 5, 10, None)
    summary, detail = fastauc.ranking_metrics(rel, s, g, ks=ks, per_group=True)

    grouped = fastauc.roc_auc_ap_grouped(rel > 0, s, g)
    np.testing.assert_allclose(detail["roc_auc"], grouped.roc_auc, rtol=RTOL)
    np.testing.assert_array_equal(detail["group"], grouped.groups)
    assert summary["gauc"] == pytest.approx(grouped.mean("roc_auc"))
    assert summary["gauc_macro"] == pytest.approx(grouped.mean("roc_auc", weight="uniform"))

    gains = np.exp2(rel) - 1
    for i in rng.choice(len(detail["group"]), 150, replace=False):
        m = np.flatnonzero(g == detail["group"][i])
        assert detail["size"][i] == len(m)
        order = m[np.argsort(-s[m])]
        if rel[m].max() == 0:
            assert all(math.isnan(detail["ndcg" + suffix(k)][i]) for k in ks)
            continue
        for k in ks:
            for name, v in one_ordering(rel[order], gains[order], k).items():
                assert detail[name + suffix(k)][i] == pytest.approx(v, rel=1e-9), (name, k)

    for name in ("ndcg@5", "mrr@10", "map", "recall@1", "precision@5", "hit_rate@1"):
        assert summary[name] == pytest.approx(np.nanmean(detail[name]), rel=1e-12)
    ranked = int((~np.isnan(detail["mrr"])).sum())
    assert summary["n_groups"] == len(detail["group"])
    assert summary["n_groups_ranked"] == ranked

    zero = fastauc.ranking_metrics(rel, s, g, ks=ks, no_relevant="zero")
    assert zero["ndcg@5"] == pytest.approx(
        summary["ndcg@5"] * ranked / summary["n_groups"], rel=1e-12
    )
    assert zero["gauc"] == summary["gauc"]


def test_ranking_ndcg_matches_sklearn_including_ties():
    rng = np.random.default_rng(2)
    rel = rng.integers(0, 4, (200, 12))
    s = np.round(rng.random((200, 12)), 1)  # ties
    keep = rel.max(axis=1) > 0
    rel, s = rel[keep], s[keep]
    for k in (3, 10, None):
        got = fastauc.ranking_metrics(
            rel.ravel(), s.ravel(), group_sizes=[12] * len(rel), ks=[k], gain="linear"
        )
        assert got["ndcg" + suffix(k)] == pytest.approx(ndcg_score(rel, s, k=k), rel=1e-9)


def test_ranking_groups_and_group_sizes_agree():
    rng = np.random.default_rng(3)
    sizes = rng.integers(0, 9, 300)  # includes empty groups
    n = int(sizes.sum())
    rel = (rng.random(n) < 0.3).astype(int)
    s = np.round(rng.random(n), 1)
    ids = np.repeat(np.arange(300), sizes)
    a = fastauc.ranking_metrics(rel, s, group_sizes=sizes)
    perm = rng.permutation(n)
    b = fastauc.ranking_metrics(rel[perm], s[perm], ids[perm])
    assert a["n_groups"] == b["n_groups"] == int((sizes > 0).sum())
    for key in a:
        assert a[key] == pytest.approx(b[key], rel=1e-12, nan_ok=True), key


def test_ranking_deterministic():
    rng = np.random.default_rng(4)
    g = rng.integers(0, 3000, 50_000)
    rel = rng.random(50_000) < 0.2
    s = rng.random(50_000)
    assert fastauc.ranking_metrics(rel, s, g) == fastauc.ranking_metrics(rel, s, g)


def test_ranking_errors():
    y, s, g = np.array([1, 0, 1]), np.array([0.3, 0.2, 0.1]), np.array([0, 0, 1])
    with pytest.raises(ValueError, match="exactly one"):
        fastauc.ranking_metrics(y, s)
    with pytest.raises(ValueError, match="exactly one"):
        fastauc.ranking_metrics(y, s, g, group_sizes=[3])
    with pytest.raises(ValueError, match="sums to"):
        fastauc.ranking_metrics(y, s, group_sizes=[2])
    with pytest.raises(ValueError, match="cutoffs"):
        fastauc.ranking_metrics(y, s, g, ks=[0])
    with pytest.raises(ValueError, match="duplicate"):
        fastauc.ranking_metrics(y, s, g, ks=[5, 5])
    with pytest.raises(ValueError, match="non-negative"):
        fastauc.ranking_metrics(np.array([1, -1, 0]), s, g)
    with pytest.raises(ValueError, match="gain"):
        fastauc.ranking_metrics(y, s, g, gain="cubic")
    with pytest.raises(ValueError, match="NaN"):
        fastauc.ranking_metrics(y, np.array([0.1, np.nan, 0.2]), g)
    out = fastauc.ranking_metrics(np.zeros(3), s, g)  # nothing relevant anywhere
    assert math.isnan(out["ndcg@1"]) and math.isnan(out["gauc"])
    assert out["n_groups_ranked"] == 0


# --------------------------------------------------------------------------
# Row-order invariance: the guarantee that ties never leak input order
# --------------------------------------------------------------------------


def everything(y, p, g, w):
    """Every scalar the package can produce for one dataset."""
    out = dict(fastauc.binary_metrics(y, p, sample_weight=w))
    for strategy in ("uniform", "quantile"):
        out["ece_" + strategy] = fastauc.expected_calibration_error(
            y, p, sample_weight=w, n_bins=7, strategy=strategy
        )
    out["p@r"] = fastauc.precision_at_recall(y, p, 0.6, sample_weight=w)[0]
    out["r@p"] = fastauc.recall_at_precision(y, p, 0.4, sample_weight=w)[0]
    out["tpr@fpr"] = fastauc.tpr_at_fpr(y, p, 0.2, sample_weight=w)[0]
    out["lift"] = fastauc.lift_at(y, p, 0.13, sample_weight=w)
    out.update(fastauc.ranking_metrics(y, p, g, ks=(1, 3, 10, None)))
    grouped = fastauc.roc_auc_ap_grouped(y, p, g)
    out["grouped_roc"] = grouped.mean("roc_auc")
    out["grouped_ap"] = grouped.mean("average_precision")
    return out


@pytest.mark.parametrize("levels", [1, 2, 5, 1000])
def test_no_metric_depends_on_row_order(levels):
    rng = np.random.default_rng(levels)
    n = 6000
    y = (rng.random(n) < 0.3).astype(np.uint8)
    p = rng.integers(0, levels, n) / max(levels - 1, 1)  # levels=1: constant score
    g = rng.integers(0, 400, n)
    w = rng.integers(1, 4, n).astype(float)

    # Adversarial layout: within every group, positives first.
    best = np.lexsort((-y.astype(int), g))
    worst = np.lexsort((y.astype(int), g))
    shuffled = rng.permutation(n)
    ref = everything(y[best], p[best], g[best], w[best])
    for order in (worst, shuffled):
        other = everything(y[order], p[order], g[order], w[order])
        for key, value in ref.items():
            assert other[key] == pytest.approx(value, rel=1e-9, abs=1e-12, nan_ok=True), key


def test_constant_scores_give_chance_level():
    rng = np.random.default_rng(0)
    n, n_groups, k = 20_000, 1000, 5
    y = (rng.random(n) < 0.25).astype(np.uint8)
    g = rng.integers(0, n_groups, n)
    order = np.lexsort((-y.astype(int), g))  # positives on top of every group
    y, g = y[order], g[order]
    s = np.full(n, 0.25)

    m = fastauc.binary_metrics(y, s)
    assert m["roc_auc"] == 0.5 and m["ks"] == 0.0
    assert m["average_precision"] == pytest.approx(y.mean())
    assert fastauc.lift_at(y, s, 0.1) == pytest.approx(1.0)
    assert fastauc.expected_calibration_error(y, s, strategy="quantile") == pytest.approx(
        abs(0.25 - y.mean())
    )

    _, d = fastauc.ranking_metrics(y, s, g, ks=[k], per_group=True)
    ok = ~np.isnan(d["mrr@5"])
    n_g, pos = d["size"][ok], np.bincount(g, weights=y)[ok]
    # Expected precision@k of a random ordering is min(k, n) / k * (positives / n).
    np.testing.assert_allclose(d["precision@5"][ok], np.minimum(k, n_g) / k * pos / n_g)
    assert (d["roc_auc"][~np.isnan(d["roc_auc"])] == 0.5).all()
    assert d["hit_rate@5"][ok].mean() < 0.9  # not the 1.0 a stable sort would report
