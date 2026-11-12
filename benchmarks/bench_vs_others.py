"""Benchmark fastauc against torchmetrics and ranx (whichever are installed).

    pip install torchmetrics ranx pandas
    python benchmarks/bench_vs_others.py

Each library is given data in its native format; conversion time is reported
separately where it is substantial. Numba-compiled code is warmed up first.
"""
import time

import numpy as np

import fastauc


def best(fn, repeat=3):
    out = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out


def line(task, lib, seconds, note=""):
    print(f"{task:<34} {lib:<26} {seconds * 1e3:>10.1f} ms  {note}")


def main():
    rng = np.random.default_rng(0)
    print(f"fastauc threads: {fastauc.num_threads()}\n")

    # ---------------- binary: ROC-AUC + average precision, N = 1M
    n = 1_000_000
    y = rng.integers(0, 2, n)
    s = rng.random(n) + 0.2 * y
    task = f"ROC-AUC + AP, N={n:,}"
    ref = fastauc.roc_auc_ap(y, s)
    line(task, "fastauc", best(lambda: fastauc.roc_auc_ap(y, s)))
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        line(task, "scikit-learn",
             best(lambda: (roc_auc_score(y, s), average_precision_score(y, s)), 1))
    except ImportError:
        pass
    try:
        import torch
        from torchmetrics.functional.classification import (
            binary_auroc,
            binary_average_precision,
        )

        ty, ts = torch.from_numpy(y), torch.from_numpy(s)
        got = (float(binary_auroc(ts, ty)), float(binary_average_precision(ts, ty)))
        assert np.allclose(got, ref, atol=1e-6), (got, ref)
        line(task, f"torchmetrics (CPU, {torch.get_num_threads()} thr)",
             best(lambda: (binary_auroc(ts, ty), binary_average_precision(ts, ty)), 2))
    except ImportError:
        print("torchmetrics not installed; skipping")

    # ---------------- ranking: N = 1M rows, 20k queries
    n_q = 20_000
    q = rng.integers(0, n_q, n)
    rel = (rng.random(n) < 0.2).astype(np.int64)
    s = rng.random(n) + 0.3 * rel
    order = np.argsort(q, kind="stable")
    q, rel, s = q[order], rel[order], s[order]
    sizes = np.bincount(q, minlength=n_q)
    task = f"NDCG@10 + MRR@10 + MAP@10, {n_q:,} q"
    ours = fastauc.ranking_metrics(rel, s, group_sizes=sizes, ks=(10,), gain="linear")
    line(task, "fastauc (group ids)",
         best(lambda: fastauc.ranking_metrics(rel, s, q, ks=(10,), gain="linear")))
    line(task, "fastauc (group_sizes)",
         best(lambda: fastauc.ranking_metrics(rel, s, group_sizes=sizes, ks=(10,), gain="linear")))

    try:
        import torch
        from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG

        ts, tr, tq = torch.from_numpy(s), torch.from_numpy(rel), torch.from_numpy(q)

        def tm():
            return [
                float(m(ts, tr, indexes=tq))
                for m in (
                    RetrievalNormalizedDCG(top_k=10, empty_target_action="skip"),
                    RetrievalMRR(top_k=10, empty_target_action="skip"),
                    RetrievalMAP(top_k=10, empty_target_action="skip"),
                )
            ]

        got = tm()
        print(f"    agreement  ndcg@10 ours {ours['ndcg@10']:.6f} torchmetrics {got[0]:.6f}"
              f" | mrr@10 ours {ours['mrr@10']:.6f} torchmetrics {got[1]:.6f}")
        line(task, "torchmetrics (CPU)", best(tm, 1))
    except ImportError:
        pass

    try:
        import pandas as pd
        from ranx import Qrels, Run, evaluate

        def build():
            # ranx wants string ids in object-dtype columns.
            doc = (np.arange(n) - np.repeat(np.cumsum(sizes) - sizes, sizes)).astype(str)
            doc, qs = doc.astype(object), q.astype(str).astype(object)
            pos = rel > 0

            def frame(mask, score):
                df = pd.DataFrame({"score": score[mask]})
                df["q_id"] = pd.Series(qs[mask], dtype=object)
                df["doc_id"] = pd.Series(doc[mask], dtype=object)
                return df

            qrels = Qrels.from_df(
                frame(pos, rel), q_id_col="q_id", doc_id_col="doc_id", score_col="score"
            )
            run = Run.from_df(
                frame(slice(None), s), q_id_col="q_id", doc_id_col="doc_id", score_col="score"
            )
            return qrels, run

        t0 = time.perf_counter()
        qrels, run = build()
        t_build = time.perf_counter() - t0
        metrics = ["ndcg@10", "mrr@10", "map@10"]
        got = evaluate(qrels, run, metrics, make_comparable=True)  # includes numba warm-up
        print(f"    agreement  ndcg@10 ours {ours['ndcg@10']:.6f} ranx {got['ndcg@10']:.6f}"
              f" | mrr@10 ours {ours['mrr@10']:.6f} ranx {got['mrr@10']:.6f}")
        line(task, "ranx (evaluate only)", best(lambda: evaluate(qrels, run, metrics, make_comparable=True), 3),
             f"+ {t_build * 1e3:.0f} ms to build Qrels/Run from arrays")
    except ImportError:
        print("ranx not installed; skipping")


if __name__ == "__main__":
    main()
