"""Tests for axis handling, thread control, group ids, weights and grouped bootstraps."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import fastauc


def clustered(n_groups=400, seed=0, per_group=(5, 40)):
    """Rows within a group share a random effect: the case row-level bootstraps get wrong."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(*per_group, n_groups)
    g = np.repeat(np.arange(n_groups), sizes)
    effect = rng.normal(0, 1.0, n_groups)[g]
    quality = rng.normal(0, 0.8, n_groups)[g]  # how well the model does in this group
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-effect))).astype(np.uint8)
    s = effect * 0.5 + quality * (2 * y - 1) * 0.5 + rng.normal(0, 1, len(g))
    return y, s, g


# --------------------------------------------------------------------------
# axis
# --------------------------------------------------------------------------


def test_batch_axis():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    S = rng.random((500, 7))  # scikit-learn layout: (n_samples, n_models)
    Y = rng.integers(0, 2, (500, 7))
    for a, b in ((y, S), (Y, S[:, 0]), (Y, S)):
        got = fastauc.roc_auc_ap_batch(a, b, axis=0)
        want = fastauc.roc_auc_ap_batch(a.T if a.ndim == 2 else a, b.T if b.ndim == 2 else b)
        np.testing.assert_array_equal(got, want)
        assert got[0].shape == (7,)
    with pytest.raises(ValueError, match="axis=0"):
        fastauc.roc_auc_ap_batch(y, S)
    with pytest.raises(ValueError, match="axis=-1"):
        fastauc.roc_auc_ap_batch(y, S.T, axis=0)
    with pytest.raises(ValueError, match="axis must be"):
        fastauc.roc_auc_ap_batch(y, S, axis=2)


def test_bootstrap_axis():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    S = rng.random((800, 3)) + 0.2 * y[:, None]
    a = fastauc.bootstrap_roc_auc_ap(y, S, axis=0, n_bootstraps=50)
    b = fastauc.bootstrap_roc_auc_ap(y, S.T, n_bootstraps=50)
    np.testing.assert_array_equal(a.roc_auc, b.roc_auc)
    assert a.roc_auc.shape == (3, 50)


# --------------------------------------------------------------------------
# threads
# --------------------------------------------------------------------------


def test_set_num_threads_does_not_change_results():
    y, s, g = clustered(seed=1)
    S = np.stack([s, s + np.random.default_rng(0).normal(0, 1, len(s))])

    def run():
        return (
            fastauc.roc_auc_ap_batch(y, S),
            fastauc.ranking_metrics(y, s, g),
            fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=64, seed=3).roc_auc,
            fastauc.bootstrap_ranking_metrics(y, s, g, n_bootstraps=64, seed=3).samples["ndcg@5"],
            fastauc.calibration_metrics(y, 1 / (1 + np.exp(-s))),
        )

    default = fastauc.num_threads()
    try:
        ref = run()
        for n in (1, 2, 3):
            fastauc.set_num_threads(n)
            assert fastauc.num_threads() == n
            got = run()
            np.testing.assert_array_equal(got[0], ref[0])
            assert got[1] == ref[1]
            np.testing.assert_array_equal(got[2], ref[2])
            np.testing.assert_array_equal(got[3], ref[3])
            assert got[4] == ref[4]
    finally:
        fastauc.set_num_threads(None)
    assert fastauc.num_threads() == default
    with pytest.raises(ValueError):
        fastauc.set_num_threads(0)


# --------------------------------------------------------------------------
# group ids: fast integer path vs generic path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_ids",
    [
        lambda g: g,  # compact
        lambda g: g * 2 - 500,  # gaps and negatives, still compact enough for the fast path
        lambda g: g.astype(np.uint32) + 7,
        lambda g: g * 10_000_019,  # sparse: generic path
        lambda g: (g * 3).astype(float),  # non-integer dtype: generic path
    ],
)
def test_group_id_encodings_agree(make_ids):
    y, s, g = clustered(n_groups=300, seed=2)
    perm = np.random.default_rng(0).permutation(len(y))
    y, s, g = y[perm], s[perm], g[perm]
    ids = make_ids(g)
    names = np.array([f"q{v:05d}" for v in g])  # strings: always the generic path

    a_sum, a = fastauc.ranking_metrics(y, s, ids, per_group=True)
    b_sum, b = fastauc.ranking_metrics(y, s, names, per_group=True)
    assert len(a["group"]) == len(b["group"]) == len(np.unique(g))
    np.testing.assert_array_equal(a["group"], np.unique(ids))
    for key in a_sum:
        assert a_sum[key] == pytest.approx(b_sum[key], rel=1e-12, nan_ok=True)
    for key in ("size", "roc_auc", "ndcg@5", "mrr@1"):
        np.testing.assert_array_equal(a[key], b[key])

    ga = fastauc.roc_auc_ap_grouped(y, s, ids)
    gb = fastauc.roc_auc_ap_grouped(y, s, names)
    np.testing.assert_array_equal(ga.groups, np.unique(ids))
    np.testing.assert_array_equal(ga.roc_auc, gb.roc_auc)
    np.testing.assert_array_equal(ga.n_pos, gb.n_pos)


# --------------------------------------------------------------------------
# AUC bootstrap: groups and weights
# --------------------------------------------------------------------------


def test_grouped_bootstrap_singleton_groups_equal_row_bootstrap():
    y, s, _ = clustered(seed=3)
    a = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=100, seed=5)
    b = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=100, seed=5, groups=np.arange(len(y)))
    np.testing.assert_array_equal(a.roc_auc, b.roc_auc)
    np.testing.assert_array_equal(a.average_precision, b.average_precision)


def test_grouped_bootstrap_matches_naive_cluster_bootstrap():
    y, s, g = clustered(seed=4)
    row = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=2000, seed=0)
    grp = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=2000, seed=0, groups=g)
    assert grp.roc_auc_estimate == row.roc_auc_estimate

    rng = np.random.default_rng(1)
    members = [np.flatnonzero(g == k) for k in np.unique(g)]
    naive = []
    for _ in range(600):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        naive.append(roc_auc_score(y[idx], s[idx]))
    naive = np.array(naive)

    assert grp.std() == pytest.approx(naive.std(ddof=1), rel=0.12)
    assert grp.roc_auc.mean() == pytest.approx(naive.mean(), abs=4 * naive.std() / np.sqrt(600))
    # The point of the feature: ignoring the clustering understates uncertainty.
    assert grp.std() > 1.3 * row.std()

    again = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=2000, seed=0, groups=g)
    np.testing.assert_array_equal(grp.roc_auc, again.roc_auc)
    as_str = fastauc.bootstrap_roc_auc_ap(
        y, s, n_bootstraps=2000, seed=0, groups=np.array([f"{v:06d}" for v in g])
    )
    np.testing.assert_array_equal(grp.roc_auc, as_str.roc_auc)


def test_grouped_bootstrap_errors():
    y, s, g = clustered(seed=5)
    with pytest.raises(ValueError, match="stratified"):
        fastauc.bootstrap_roc_auc_ap(y, s, groups=g, stratified=True)
    with pytest.raises(ValueError, match="length"):
        fastauc.bootstrap_roc_auc_ap(y, s, groups=g[:-1])


def test_weighted_bootstrap():
    y, s, _ = clustered(n_groups=150, seed=6)
    w = np.random.default_rng(0).integers(1, 5, len(y)).astype(float)
    r = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=1500, sample_weight=w, seed=2)
    assert r.roc_auc_estimate == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    assert r.roc_auc_estimate == fastauc.roc_auc(y, s, sample_weight=w)

    rng = np.random.default_rng(3)
    naive = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y))
        naive.append(roc_auc_score(y[i], s[i], sample_weight=w[i]))
    assert r.std() == pytest.approx(np.std(naive, ddof=1), rel=0.12)
    lo, hi = r.ci()
    assert lo < r.roc_auc_estimate < hi
    # All-ones weights change nothing.
    a = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=50, sample_weight=np.ones(len(y)))
    b = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=50)
    np.testing.assert_allclose(a.roc_auc, b.roc_auc, rtol=1e-12)


# --------------------------------------------------------------------------
# ranking: sample_weight
# --------------------------------------------------------------------------


def test_ranking_sample_weight():
    y, s, g = clustered(n_groups=300, seed=7)
    rng = np.random.default_rng(0)
    qw = rng.random(300) * 3 + 0.1
    w = qw[g]  # every row carries its query's weight

    plain, d = fastauc.ranking_metrics(y, s, g, per_group=True)
    weighted = fastauc.ranking_metrics(y, s, g, sample_weight=w)

    for name in ("ndcg@5", "mrr@10", "map@1", "recall@5", "precision@10", "hit_rate@1"):
        ok = ~np.isnan(d[name])
        assert weighted[name] == pytest.approx(np.average(d[name][ok], weights=qw[ok]), rel=1e-12)
    ok = ~np.isnan(d["roc_auc"])
    assert weighted["gauc_macro"] == pytest.approx(
        np.average(d["roc_auc"][ok], weights=qw[ok]), rel=1e-12
    )
    assert weighted["gauc"] == pytest.approx(
        np.average(d["roc_auc"][ok], weights=(qw * d["size"])[ok]), rel=1e-12
    )
    assert weighted["n_groups_ranked"] == plain["n_groups_ranked"]

    # no_relevant="zero" with weights
    zero = fastauc.ranking_metrics(y, s, g, sample_weight=w, no_relevant="zero")
    assert zero["ndcg@5"] == pytest.approx(
        np.average(np.nan_to_num(d["ndcg@5"]), weights=qw), rel=1e-12
    )

    # A constant weight is the same as no weight.
    const = fastauc.ranking_metrics(y, s, g, sample_weight=np.full(len(y), 2.5))
    for key in plain:
        assert const[key] == pytest.approx(plain[key], rel=1e-12, nan_ok=True)
    with pytest.raises(ValueError, match="sample_weight"):
        fastauc.ranking_metrics(y, s, g, sample_weight=-w)
    with pytest.raises(ValueError, match="sample_weight"):
        fastauc.ranking_metrics(y, s, g, sample_weight=w[:-1])


# --------------------------------------------------------------------------
# ranking bootstrap
# --------------------------------------------------------------------------


def naive_group_bootstrap(values, weights, b, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        i = rng.integers(0, len(values), len(values))
        v, w = values[i], weights[i]
        ok = ~np.isnan(v)
        out.append(np.average(v[ok], weights=w[ok]))
    return np.array(out)


@pytest.mark.parametrize("no_relevant", ["skip", "zero"])
def test_bootstrap_ranking_metrics(no_relevant):
    y, s, g = clustered(n_groups=500, seed=8, per_group=(2, 15))
    ks = (1, 5, None)
    point, d = fastauc.ranking_metrics(y, s, g, ks=ks, no_relevant=no_relevant, per_group=True)
    r = fastauc.bootstrap_ranking_metrics(
        y, s, g, ks=ks, no_relevant=no_relevant, n_bootstraps=3000, seed=1
    )

    assert set(r.estimate) == set(r.samples)
    for key, value in r.estimate.items():
        assert value == pytest.approx(point[key], rel=1e-12)
        assert r.samples[key].shape == (3000,)
        lo, hi = r.ci(key)
        assert lo <= value <= hi  # recall and hit rate over the whole list are exactly 1
        assert r.ci(key, 0.5)[0] >= lo

    ones = np.ones(len(d["group"]))
    for key in ("ndcg@5", "mrr", "hit_rate@1", "precision@5"):
        v = d[key] if no_relevant == "skip" else np.nan_to_num(d[key])
        naive = naive_group_bootstrap(v, ones, 1500, 7)
        assert r.std(key) == pytest.approx(naive.std(ddof=1), rel=0.1), key
        assert r.samples[key].mean() == pytest.approx(naive.mean(), abs=5 * naive.std() / 38)
    naive = naive_group_bootstrap(d["roc_auc"], d["size"].astype(float), 1500, 7)
    assert r.std("gauc") == pytest.approx(naive.std(ddof=1), rel=0.1)
    naive = naive_group_bootstrap(d["roc_auc"], ones, 1500, 7)
    assert r.std("gauc_macro") == pytest.approx(naive.std(ddof=1), rel=0.1)

    again = fastauc.bootstrap_ranking_metrics(
        y, s, g, ks=ks, no_relevant=no_relevant, n_bootstraps=3000, seed=1
    )
    np.testing.assert_array_equal(r.samples["ndcg@5"], again.samples["ndcg@5"])
    with pytest.raises(ValueError, match="unknown metric"):
        r.ci("ndcg@7")


def test_bootstrap_ranking_group_sizes_and_weights():
    y, s, g = clustered(n_groups=200, seed=9, per_group=(2, 12))
    sizes = np.bincount(g)
    w = (np.random.default_rng(0).random(200) + 0.2)[g]
    a = fastauc.bootstrap_ranking_metrics(y, s, g, sample_weight=w, n_bootstraps=200, seed=4)
    b = fastauc.bootstrap_ranking_metrics(
        y, s, group_sizes=sizes, sample_weight=w, n_bootstraps=200, seed=4
    )
    for key in a.samples:
        np.testing.assert_array_equal(a.samples[key], b.samples[key])
    point = fastauc.ranking_metrics(y, s, g, sample_weight=w)
    assert a.estimate["ndcg@10"] == pytest.approx(point["ndcg@10"], rel=1e-12)
    assert a.estimate["gauc"] == pytest.approx(point["gauc"], rel=1e-12)


def test_bootstrap_ranking_compare_models():
    y, s, g = clustered(n_groups=600, seed=10, per_group=(4, 20))
    rng = np.random.default_rng(0)
    close = s + rng.normal(0, 0.1, len(s))
    worse = rng.normal(0, 1, len(s))
    r = fastauc.bootstrap_ranking_metrics(
        y, np.stack([s, close, worse]), g, ks=(10,), n_bootstraps=1500, seed=2
    )
    assert r.samples["ndcg@10"].shape == (3, 1500)
    assert r.estimate["ndcg@10"].shape == (3,)
    assert r.ci("ndcg@10").shape == (3, 2)
    solo = fastauc.bootstrap_ranking_metrics(y, s, g, ks=(10,), n_bootstraps=1500, seed=2)
    np.testing.assert_array_equal(r.samples["ndcg@10"][0], solo.samples["ndcg@10"])
    assert np.corrcoef(r.samples["ndcg@10"][0], r.samples["ndcg@10"][1])[0, 1] > 0.8

    diff, (lo, hi), p = r.compare(0, 2, "ndcg@10")
    assert diff > 0 and lo > 0 and p < 0.01
    _, (lo, hi), p = r.compare(0, 1, "ndcg@10")
    assert hi - lo < r.ci("ndcg@10")[0, 1] - r.ci("ndcg@10")[0, 0]  # pairing tightens it

    via_axis = fastauc.bootstrap_ranking_metrics(
        y, np.stack([s, close, worse]).T, g, ks=(10,), n_bootstraps=1500, seed=2, axis=0
    )
    np.testing.assert_array_equal(via_axis.samples["mrr@10"], r.samples["mrr@10"])
    with pytest.raises(ValueError):
        solo.compare(0, 1, "ndcg@10")
