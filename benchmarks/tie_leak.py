"""Do other libraries' metrics depend on input order when scores are tied?

    pip install scikit-learn ranx pytrec-eval-terrier ir-measures lightgbm xgboost fastroc
    python benchmarks/tie_leak.py

1000 queries x 10 documents, 2 relevant per query, and a model that scores
every document 0. The two layouts contain the same data:

* "pos first": relevant documents come first in the input, with ids d0, d1
* "pos last":  relevant documents come last, with ids d8, d9

A tie-correct metric gives the same, chance-level value for both. Libraries
that are not installed are skipped.
"""
import warnings

import numpy as np

import fastauc

warnings.filterwarnings("ignore")
Q, D, R, K = 1000, 10, 2, 3
rows = []


def layout(first):
    rel = np.zeros((Q, D), dtype=int)
    if first:
        rel[:, :R] = 1
    else:
        rel[:, -R:] = 1
    return rel


rel_a, rel_b = layout(True), layout(False)
score = np.zeros((Q, D))
qid = np.repeat(np.arange(Q), D)
truth = fastauc.ranking_metrics(rel_a.ravel(), score.ravel(), qid, ks=(1, K))
truth["auc"] = 0.5


def check(lib, metric, fn):
    try:
        a, b = fn(rel_a), fn(rel_b)
    except ImportError:
        return
    rows.append((lib, metric, float(a), float(b), truth[metric]))


def fa(metric):
    return lambda rel: fastauc.ranking_metrics(rel.ravel(), score.ravel(), qid, ks=(1, K))[metric]


for m in ("ndcg@3", "map@3", "mrr@3", "precision@1"):
    check("fastauc", m, fa(m))


def sk_ndcg(rel):
    from sklearn.metrics import ndcg_score

    return ndcg_score(rel, score, k=K)


def sk_auc(rel):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(rel.ravel(), score.ravel())


check("scikit-learn ndcg_score", "ndcg@3", sk_ndcg)
check("scikit-learn roc_auc_score", "auc", sk_auc)


def ranx_metric(metric):
    def run(rel):
        from ranx import Qrels, Run, evaluate

        docs = [f"d{j}" for j in range(D)]
        qrels = {f"q{q}": {docs[j]: int(rel[q, j]) for j in range(D) if rel[q, j]} for q in range(Q)}
        ranking = {f"q{q}": {d: 0.0 for d in docs} for q in range(Q)}
        return evaluate(Qrels(qrels), Run(ranking), metric)

    return run


for m in ("ndcg@3", "map@3", "mrr@3", "precision@1"):
    check("ranx", m, ranx_metric(m))


def trec_metric(name):
    def run(rel):
        import pytrec_eval

        qrels = {f"q{q}": {f"d{j}": int(rel[q, j]) for j in range(D)} for q in range(Q)}
        ranking = {f"q{q}": {f"d{j}": 0.0 for j in range(D)} for q in range(Q)}
        key = {"ndcg_cut_3": "ndcg_cut.3", "P_1": "P.1"}[name]
        res = pytrec_eval.RelevanceEvaluator(qrels, {key}).evaluate(ranking)
        return np.mean([r[name] for r in res.values()])

    return run


check("pytrec_eval (trec_eval)", "ndcg@3", trec_metric("ndcg_cut_3"))
check("pytrec_eval (trec_eval)", "precision@1", trec_metric("P_1"))


def irm(which):
    def run(rel):
        import ir_measures
        from ir_measures import P, nDCG

        measure = nDCG @ 3 if which == "ndcg" else P @ 1
        qrels = [ir_measures.Qrel(f"q{q}", f"d{j}", int(rel[q, j])) for q in range(Q) for j in range(D)]
        ranking = [ir_measures.ScoredDoc(f"q{q}", f"d{j}", 0.0) for q in range(Q) for j in range(D)]
        return ir_measures.calc_aggregate([measure], qrels, ranking)[measure]

    return run


check("ir-measures", "ndcg@3", irm("ndcg"))
check("ir-measures", "precision@1", irm("p"))


def lgb_metric(metric):
    def run(rel):
        import lightgbm as lgb

        x = np.zeros((Q * D, 2))  # constant features: every prediction is equal
        train = lgb.Dataset(x, rel.ravel(), group=[D] * Q)
        valid = lgb.Dataset(x, rel.ravel(), group=[D] * Q, reference=train)
        hist = {}
        lgb.train(
            {"objective": "lambdarank", "metric": ["ndcg", "map"], "eval_at": [K], "verbose": -1},
            train, num_boost_round=1, valid_sets=[valid], valid_names=["v"],
            callbacks=[lgb.record_evaluation(hist)],
        )
        return hist["v"][metric][-1]

    return run


check("LightGBM (built-in eval)", "ndcg@3", lgb_metric("ndcg@3"))
check("LightGBM (built-in eval)", "map@3", lgb_metric("map@3"))


def xgb_metric(metric):
    def run(rel):
        import xgboost as xgb

        d = xgb.DMatrix(np.zeros((Q * D, 2)), rel.ravel())
        d.set_group([D] * Q)
        hist = {}
        xgb.train(
            {"objective": "rank:ndcg", "eval_metric": [metric]}, d, num_boost_round=1,
            evals=[(d, "v")], evals_result=hist, verbose_eval=False,
        )
        return hist["v"][metric][-1]

    return run


check("XGBoost (built-in eval)", "ndcg@3", xgb_metric("ndcg@3"))
check("XGBoost (built-in eval)", "map@3", xgb_metric("map@3"))


def fastroc_auc(rel):
    import fastroc

    return fastroc.calc_roc_auc(rel.ravel().astype(bool), score.ravel())


check("fastroc", "auc", fastroc_auc)

print(f"{'library':<28}{'metric':<13}{'pos first':>10}{'pos last':>10}{'correct':>10}  verdict")
for lib, m, a, b, t in rows:
    ok = abs(a - t) < 1e-6 and abs(b - t) < 1e-6
    verdict = "ok" if ok else ("DEPENDS ON ORDER" if abs(a - b) > 1e-6 else "wrong")
    print(f"{lib:<28}{m:<13}{a:>10.4f}{b:>10.4f}{t:>10.4f}  {verdict}")
