"""Ranking metrics with complete relevance judgments (relevant items missing from the lists)."""
import itertools

import numpy as np
import pytest

import fastauc

DISC = 1.0 / np.log2(np.arange(2, 200))


def one_ordering(rel, gains, k, n_rel_total, ideal_gains, map_total):
    """Textbook metrics for one list in ranked order, with external judgments."""
    n = len(rel)
    kk = n if k is None else min(k, n)
    hits = rel[:kk] > 0
    cut = len(ideal_gains) if k is None else min(k, len(ideal_gains))
    idcg = float(np.sum(np.sort(ideal_gains)[::-1][:cut] * DISC[:cut]))
    prec_at = np.cumsum(rel > 0) / np.arange(1, n + 1)
    first = np.flatnonzero(hits)
    map_den = n_rel_total if (map_total or k is None) else min(n_rel_total, k)
    return {
        "ndcg": float(np.sum(gains[:kk] * DISC[:kk])) / idcg,
        "mrr": 1 / (first[0] + 1) if len(first) else 0.0,
        "map": float(np.sum(prec_at[:kk] * hits)) / map_den,
        "recall": hits.sum() / n_rel_total,
        "precision": hits.sum() / (n if k is None else k),
        "hit_rate": float(hits.any()),
    }


def brute_force(rel, score, k, n_rel_total, ideal_gains, map_total):
    gains = rel.astype(float)  # linear gain
    blocks = [np.flatnonzero(score == v) for v in np.unique(score)[::-1]]
    acc, count = {}, 0
    for perm in itertools.product(*(itertools.permutations(b) for b in blocks)):
        order = np.concatenate([np.array(p, dtype=int) for p in perm])
        for name, v in one_ordering(rel[order], gains[order], k, n_rel_total, ideal_gains,
                                    map_total).items():
            acc[name] = acc.get(name, 0.0) + v
        count += 1
    return {name: v / count for name, v in acc.items()}


def suffix(k):
    return "" if k is None else f"@{k}"


@pytest.mark.parametrize("map_total", [False, True])
def test_judgments_match_brute_force_with_ties(map_total):
    rng = np.random.default_rng(0)
    ks = (1, 2, 3, None)
    checked = 0
    for _ in range(80):
        n = int(rng.integers(1, 7))
        rel = rng.integers(0, 3, n) * (rng.random(n) < 0.6)
        missing = rng.integers(0, 3, int(rng.integers(0, 3)))  # judged relevant, not retrieved
        missing = missing[missing > 0]
        judged = np.concatenate([rel[rel > 0], missing]).astype(float)
        if judged.size == 0:
            continue
        score = rng.integers(0, 3, n).astype(float)  # heavy ties
        got = fastauc.ranking_metrics(
            rel, score, group_sizes=[n], ks=ks, gain="linear",
            judgments=(np.zeros(len(judged), dtype=int), judged),
            map_denominator="total" if map_total else "min",
        )
        for k in ks:
            want = brute_force(rel, score, k, len(judged), judged, map_total)
            for name, v in want.items():
                assert got[name + suffix(k)] == pytest.approx(v, rel=1e-9, abs=1e-12), (name, k)
        checked += 1
    assert checked > 40


def test_complete_judgments_change_nothing():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 300, 5000)
    rel = rng.integers(0, 4, 5000) * (rng.random(5000) < 0.3)
    s = np.round(rng.random(5000), 1)
    plain = fastauc.ranking_metrics(rel, s, g, ks=(1, 5, None))
    full = fastauc.ranking_metrics(rel, s, g, ks=(1, 5, None), judgments=(g, rel))
    for key in plain:
        assert full[key] == pytest.approx(plain[key], rel=1e-12, nan_ok=True), key


def test_missing_relevant_items_lower_recall_and_ndcg():
    rel = np.array([1, 0, 0])
    s = np.array([0.9, 0.5, 0.1])
    assert fastauc.ranking_metrics(rel, s, group_sizes=[3], ks=(3,))["recall@3"] == 1.0
    judged = fastauc.ranking_metrics(
        rel, s, group_sizes=[3], ks=(3,), judgments=([0, 0], [1, 1]), gain="linear"
    )
    assert judged["recall@3"] == 0.5
    assert judged["ndcg@3"] == pytest.approx(1.0 / (1.0 + 1.0 / np.log2(3)))
    assert judged["mrr@3"] == 1.0


def test_group_with_nothing_retrieved_counts_as_zero():
    rel = np.array([0, 0, 1, 0])
    s = np.array([0.4, 0.3, 0.9, 0.1])
    g = np.array(["a", "a", "b", "b"])
    out, d = fastauc.ranking_metrics(
        rel, s, g, ks=(2,), judgments=(["a", "b", "zzz"], [1, 1, 1]), per_group=True
    )
    assert list(d["group"]) == ["a", "b"]  # "zzz" is not in the lists: ignored
    assert d["ndcg@2"][0] == 0.0 and d["recall@2"][0] == 0.0  # "a" retrieved nothing relevant
    assert d["ndcg@2"][1] == 1.0
    assert out["n_groups_ranked"] == 2


def test_judgment_errors():
    rel, s = np.array([1, 1]), np.array([0.2, 0.1])
    with pytest.raises(ValueError, match="more relevant items"):
        fastauc.ranking_metrics(rel, s, group_sizes=[2], judgments=([0], [1]))
    with pytest.raises(ValueError, match="pair"):
        fastauc.ranking_metrics(rel, s, group_sizes=[2], judgments=[0, 1, 2])
    with pytest.raises(ValueError, match="differ in length"):
        fastauc.ranking_metrics(rel, s, group_sizes=[2], judgments=([0, 0], [1]))
    with pytest.raises(ValueError, match="map_denominator"):
        fastauc.ranking_metrics(rel, s, group_sizes=[2], map_denominator="R")


def test_bootstrap_accepts_judgments():
    rng = np.random.default_rng(2)
    g = rng.integers(0, 200, 3000)
    rel = (rng.random(3000) < 0.2).astype(int)
    s = rng.random(3000)
    extra_g = rng.integers(0, 200, 300)
    jg = np.concatenate([g[rel > 0], extra_g])
    jr = np.ones(len(jg))
    point = fastauc.ranking_metrics(rel, s, g, ks=(10,), judgments=(jg, jr))
    boot = fastauc.bootstrap_ranking_metrics(rel, s, g, ks=(10,), judgments=(jg, jr),
                                             n_bootstraps=100)
    assert boot.estimate["recall@10"] == pytest.approx(point["recall@10"], rel=1e-12)
    assert point["recall@10"] < fastauc.ranking_metrics(rel, s, g, ks=(10,))["recall@10"]


def test_matches_trec_eval():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = np.random.default_rng(3)
    n_q, n_docs = 60, 30
    qrels, run, rows = {}, {}, []
    for q in range(n_q):
        judged = {f"d{j}": int(rng.integers(1, 4)) for j in rng.choice(200, 12, replace=False)}
        retrieved = [f"d{j}" for j in rng.choice(200, n_docs, replace=False)]
        scores = rng.random(n_docs)  # no ties: trec_eval's tie order does not matter
        qrels[f"q{q}"] = judged
        run[f"q{q}"] = dict(zip(retrieved, scores.tolist()))
        rows += [(q, judged.get(d, 0), sc) for d, sc in zip(retrieved, scores)]
    q_arr, rel, s = (np.array(x) for x in zip(*rows))
    jg = np.concatenate([[q] * len(v) for q, v in enumerate(qrels.values())])
    jr = np.concatenate([list(v.values()) for v in qrels.values()])
    keep = np.array([any(r > 0 for qq, r, _ in rows if qq == q) for q in range(n_q)])

    measures = {"ndcg_cut.10", "recall.10", "P.10", "map_cut.10", "recip_rank"}
    res = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
    _, d = fastauc.ranking_metrics(rel, s, q_arr, ks=(10, None), gain="linear",
                                   judgments=(jg, jr), map_denominator="total", per_group=True)
    for ours, theirs in (("ndcg@10", "ndcg_cut_10"), ("recall@10", "recall_10"),
                         ("precision@10", "P_10"), ("map@10", "map_cut_10"), ("mrr", "recip_rank")):
        want = np.array([res[f"q{q}"][theirs] for q in range(n_q)])
        got = np.nan_to_num(d[ours])  # queries that retrieved no relevant item: 0 in both
        np.testing.assert_allclose(got[keep], want[keep], atol=1e-4, err_msg=ours)
