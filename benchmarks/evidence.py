"""Reproducible evidence that fastauc is correct and fast, on synthetic data.

    pip install pyfastauc scikit-learn        # numba optional, for one extra baseline
    python benchmarks/evidence.py             # writes evidence_report.txt and evidence.json

Two synthetic workloads mirror common offline ranking evaluations:

* "short lists": 2,110 queries x 10 documents, binary relevance, string query ids
* "ragged pools": 8,956 queries with 8-95 documents each (about 19 on
  average), graded gains (1, 0.1, 0.01, 0), integer query ids

Two models, "CE" and "DE", score every document; CE is slightly better. Scores
are probabilities rounded to bfloat16, as models served in bf16 produce, which
creates many tied scores. Within each query, relevant documents are listed
first, as they often are when evaluation sets are built from judgments.

The script checks, in order:
  1. nDCG@10 against a plain reference implementation, per query
  2. nDCG@10, ROC-AUC and average precision against scikit-learn
  3. how much position-based tie-breaking inflates nDCG@10, and that fastauc
     gives the same answer for any row order
  4. a paired, query-level bootstrap comparison of two models
  5. speed against scikit-learn, a Python per-query loop, and (if installed)
     a compiled Numba kernel

It exits with status 1 if any accuracy check fails. Timings are reported, not
checked.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

import fastauc

K = 10
TOL = 1e-12


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def bf16(x: np.ndarray) -> np.ndarray:
    """Round to bfloat16 (round to nearest even), returned as float64."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return bits.view(np.float32).astype(np.float64)


def make_workload(name: str, rng: np.random.Generator) -> dict:
    if name == "short lists":
        n_queries = 2110
        sizes = np.full(n_queries, 10)
        n = int(sizes.sum())
        gain = (rng.random(n) < 0.25).astype(np.float64)
        qid = np.repeat(np.array([f"q{i:05d}" for i in range(n_queries)]), sizes)
    else:
        n_queries = 8956
        sizes = np.clip(np.round(rng.lognormal(np.log(17), 0.5, n_queries)), 8, 95).astype(np.int64)
        n = int(sizes.sum())
        gain = rng.choice([1.0, 0.1, 0.01, 0.0], size=n, p=[0.35, 0.15, 0.1, 0.4])
        qid = np.repeat(np.arange(n_queries), sizes)

    # Shared document difficulty makes the two models' errors correlated, as
    # they are for real models scored on the same documents.
    hard = rng.normal(0.0, 1.0, n)
    relevant = (gain > 0).astype(np.float64) + gain

    def prob(signal, noise):
        logit = 0.5 * (signal * relevant + hard + noise * rng.normal(0.0, 1.0, n)) + 1.5
        return bf16(1.0 / (1.0 + np.exp(-logit)))

    ce, de = prob(0.40, 1.2), prob(0.34, 1.3)

    # Within each query, list relevant documents first.
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    order = np.lexsort((-gain, np.repeat(np.arange(n_queries), sizes)))
    gain, ce, de, qid = gain[order], ce[order], de[order], qid[order]
    return {"name": name, "gain": gain, "ce": ce, "de": de, "qid": qid, "sizes": sizes, "starts": starts}


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

DISC = 1.0 / np.log2(np.arange(2, 2 + 4096))


def ndcg10_reference(gains: np.ndarray, scores: np.ndarray):
    """Tie-averaged nDCG@10 for one query, written out step by step.

    Each block of equal scores starting at position p with length L contributes
    (sum of its discounts within the top 10) / L * (sum of its gains).
    Returns None when the query has no relevant document.
    """
    ideal = np.sort(gains)[::-1][:K]
    idcg = float(np.sum(ideal * DISC[: len(ideal)]))
    if idcg == 0.0:
        return None
    order = np.argsort(-scores, kind="stable")
    g, s = gains[order], scores[order]
    dcg, p, n = 0.0, 0, len(s)
    while p < n and p < K:
        q = p
        while q + 1 < n and s[q + 1] == s[p]:
            q += 1
        length = q - p + 1
        disc = float(np.sum(DISC[p: min(q, K - 1) + 1]))
        dcg += disc / length * float(np.sum(g[p: q + 1]))
        p = q + 1
    return dcg / idcg


def ndcg10_positional(gains: np.ndarray, scores: np.ndarray):
    """nDCG@10 with ties broken by input position (a stable sort)."""
    ideal = np.sort(gains)[::-1][:K]
    idcg = float(np.sum(ideal * DISC[: len(ideal)]))
    if idcg == 0.0:
        return None
    g = gains[np.argsort(-scores, kind="stable")][:K]
    return float(np.sum(g * DISC[: len(g)])) / idcg


def per_query(fn, w: dict, scores: np.ndarray) -> np.ndarray:
    out = np.full(len(w["sizes"]), np.nan)
    for i, (a, m) in enumerate(zip(w["starts"], w["sizes"])):
        v = fn(w["gain"][a: a + m], scores[a: a + m])
        if v is not None:
            out[i] = v
    return out


def padded(w: dict, scores: np.ndarray, keep: np.ndarray):
    """2-D arrays for sklearn.metrics.ndcg_score: one row per query.

    Padding has zero gain and a score below every real score, so it cannot
    change DCG@10 or IDCG@10.
    """
    sizes, starts = w["sizes"][keep], w["starts"][keep]
    width = int(sizes.max())
    floor = float(scores.min()) - 1.0
    G = np.zeros((len(sizes), width))
    S = np.full((len(sizes), width), floor)
    for r, (a, m) in enumerate(zip(starts, sizes)):
        G[r, :m] = w["gain"][a: a + m]
        S[r, :m] = scores[a: a + m]
    return G, S


# ---------------------------------------------------------------------------
# Optional Numba baseline
# ---------------------------------------------------------------------------


def numba_kernel():
    try:
        import numba
    except ImportError:
        return None, None

    @numba.njit(cache=False)
    def kernel(gain, score, starts, sizes):
        total, count = 0.0, 0
        disc = 1.0 / np.log2(np.arange(2, 2 + K).astype(np.float64))
        for qi in range(len(sizes)):
            a, m = starts[qi], sizes[qi]
            g, s = gain[a: a + m], score[a: a + m]
            ideal = np.sort(g)[::-1]
            idcg = 0.0
            for r in range(min(K, m)):
                idcg += ideal[r] * disc[r]
            if idcg == 0.0:
                continue
            order = np.argsort(-s, kind="mergesort")
            dcg, p = 0.0, 0
            while p < m and p < K:
                q = p
                while q + 1 < m and s[order[q + 1]] == s[order[p]]:
                    q += 1
                block = 0.0
                for t in range(p, q + 1):
                    block += g[order[t]]
                d = 0.0
                for r in range(p, min(q, K - 1) + 1):
                    d += disc[r]
                dcg += d / (q - p + 1) * block
                p = q + 1
            total += dcg / idcg
            count += 1
        return total / count

    return kernel, numba.__version__


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def timeit(fn, repeat: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3  # ms


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

LINES: list[str] = []


def add(s: str = "") -> None:
    LINES.append(s)


def table(headers, rows, right=()):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

    def fmt(row):
        return "| " + " | ".join(
            str(v).rjust(widths[i]) if i in right else str(v).ljust(widths[i]) for i, v in enumerate(row)
        ) + " |"

    add(fmt(headers))
    add("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        add(fmt(r))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=".", help="directory for evidence_report.txt and evidence.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="fewer timing repeats")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    reps = 3 if args.quick else 10

    import sklearn

    kernel, numba_version = numba_kernel()
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "fastauc": fastauc.__version__,
        "numba": numba_version,
        "cpu_count": os.cpu_count(),
        "fastauc_threads": fastauc.num_threads(),
        "github_runner": os.environ.get("RUNNER_NAME"),
    }
    results: dict = {"environment": env, "seed": args.seed, "workloads": {}}
    failures: list[str] = []

    def check(ok: bool, what: str) -> str:
        if not ok:
            failures.append(what)
        return "pass" if ok else "FAIL"

    for name in ("short lists", "ragged pools"):
        w = make_workload(name, rng)
        g, ce, de, qid, sizes = w["gain"], w["ce"], w["de"], w["qid"], w["sizes"]
        n_rows, n_q = len(g), len(sizes)
        R: dict = {"queries": n_q, "rows": n_rows}

        # ---- data description
        tied_blocks = 0
        for a, m in zip(w["starts"], sizes):
            s = ce[a: a + m]
            tied_blocks += int(len(np.unique(s)) < m)
        R["queries_with_tied_scores_ce"] = tied_blocks

        # ---- 1. reference, per query
        ref_ce = per_query(ndcg10_reference, w, ce)
        _, d_ce = fastauc.ranking_metrics(g, ce, qid, ks=(K,), gain="linear", per_group=True)
        fa_pq = d_ce[f"ndcg@{K}"]
        if name == "short lists":  # string ids come back sorted; map back to query order
            pos = {v: i for i, v in enumerate(d_ce["group"])}
            fa_pq = fa_pq[[pos[f"q{i:05d}"] for i in range(n_q)]]
        both = ~np.isnan(ref_ce)
        R["ref_max_abs_diff_per_query"] = float(np.max(np.abs(fa_pq[both] - ref_ce[both])))
        R["ref_same_skipped_queries"] = bool(np.array_equal(np.isnan(fa_pq), np.isnan(ref_ce)))
        fa_ce = fastauc.ranking_metrics(g, ce, qid, ks=(K,), gain="linear")[f"ndcg@{K}"]
        fa_de = fastauc.ranking_metrics(g, de, qid, ks=(K,), gain="linear")[f"ndcg@{K}"]
        R["ndcg_ce"], R["ndcg_de"] = fa_ce, fa_de
        R["ref_abs_diff_aggregate"] = abs(fa_ce - float(np.nanmean(ref_ce)))

        # ---- 2. scikit-learn
        G, S = padded(w, ce, both)
        sk = float(ndcg_score(G, S, k=K))
        R["sklearn_ndcg"] = sk
        R["sklearn_ndcg_abs_diff"] = abs(fa_ce - sk)
        y = (g > 0).astype(np.uint8)
        R["sklearn_auc_abs_diff"] = abs(fastauc.roc_auc(y, ce) - roc_auc_score(y, ce))
        R["sklearn_ap_abs_diff"] = abs(fastauc.average_precision(y, ce) - average_precision_score(y, ce))

        # ---- 3. ties
        pos_first = float(np.nanmean(per_query(ndcg10_positional, w, ce)))
        rev = np.concatenate([np.arange(a + m - 1, a - 1, -1) for a, m in zip(w["starts"], sizes)])
        w_rev = dict(w, gain=g[rev])
        pos_last = float(np.nanmean(per_query(ndcg10_positional, w_rev, ce[rev])))
        R["positional_relevant_first"], R["positional_relevant_last"] = pos_first, pos_last
        perm = rng.permutation(n_rows)
        shuffled = fastauc.ranking_metrics(g[perm], ce[perm], qid[perm], ks=(K,), gain="linear")[f"ndcg@{K}"]
        reversed_ = fastauc.ranking_metrics(g[rev], ce[rev], qid[rev], ks=(K,), gain="linear")[f"ndcg@{K}"]
        R["order_invariance_max_abs_diff"] = max(abs(shuffled - fa_ce), abs(reversed_ - fa_ce))

        # ---- 4. paired comparison
        boot = fastauc.bootstrap_ranking_metrics(g, np.stack([ce, de]), qid, ks=(K,), gain="linear",
                                                 n_bootstraps=2000, seed=args.seed)
        diff, (lo, hi), p = boot.compare(0, 1, f"ndcg@{K}")
        cis = boot.ci(f"ndcg@{K}")
        R["paired"] = {"delta": diff, "ci": [lo, hi], "p": p, "n_bootstraps": 2000,
                       "ci_ce": cis[0].tolist(), "ci_de": cis[1].tolist(),
                       "unpaired_overlap": bool(cis[0][0] <= cis[1][1])}

        # ---- 5. speed (fastauc pinned to one thread unless stated)
        t: dict = {}
        fastauc.set_num_threads(1)
        t["fastauc_1t_groups"] = timeit(lambda: fastauc.ranking_metrics(g, ce, qid, ks=(K,), gain="linear"), reps * 3)
        t["fastauc_1t_group_sizes"] = timeit(
            lambda: fastauc.ranking_metrics(g, ce, group_sizes=sizes, ks=(K,), gain="linear"), reps * 3)
        fastauc.set_num_threads(None)
        t["fastauc_all_group_sizes"] = timeit(
            lambda: fastauc.ranking_metrics(g, ce, group_sizes=sizes, ks=(K,), gain="linear"), reps * 3)
        t["sklearn_2d"] = timeit(lambda: ndcg_score(G, S, k=K), reps)
        t["python_loop"] = timeit(lambda: per_query(ndcg10_reference, w, ce), max(2, reps // 3), warmup=1)
        if kernel is not None:
            starts64, sizes64 = w["starts"].astype(np.int64), sizes.astype(np.int64)
            t0 = time.perf_counter()
            nb = kernel(g, ce, starts64, sizes64)
            if "numba_jit_ms" not in results:  # only the very first call compiles
                results["numba_jit_ms"] = (time.perf_counter() - t0) * 1e3
                t["numba_first_call_incl_jit"] = results["numba_jit_ms"]
            t["numba_warm"] = timeit(lambda: kernel(g, ce, starts64, sizes64), reps * 3)
            R["numba_abs_diff"] = abs(nb - fa_ce)
        R["timings_ms"] = t
        results["workloads"][name] = R

        # ---- report section
        add("=" * 78)
        add(f"WORKLOAD: {name} ({n_q:,} queries, {n_rows:,} rows; "
            f"{tied_blocks:,} queries have tied scores)")
        add("=" * 78)
        add("")
        add("Accuracy (nDCG@10 unless stated)")
        rows = [
            ["fastauc vs reference, per query", f"{R['ref_max_abs_diff_per_query']:.1e}",
             check(R["ref_max_abs_diff_per_query"] <= TOL and R["ref_same_skipped_queries"],
                   f"{name}: reference per query")],
            ["fastauc vs scikit-learn ndcg_score", f"{R['sklearn_ndcg_abs_diff']:.1e}",
             check(R["sklearn_ndcg_abs_diff"] <= TOL, f"{name}: sklearn nDCG")],
            ["fastauc vs scikit-learn roc_auc_score", f"{R['sklearn_auc_abs_diff']:.1e}",
             check(R["sklearn_auc_abs_diff"] <= TOL, f"{name}: sklearn AUC")],
            ["fastauc vs scikit-learn average_precision_score", f"{R['sklearn_ap_abs_diff']:.1e}",
             check(R["sklearn_ap_abs_diff"] <= TOL, f"{name}: sklearn AP")],
            ["same result after shuffling or reversing rows", f"{R['order_invariance_max_abs_diff']:.1e}",
             check(R["order_invariance_max_abs_diff"] <= TOL, f"{name}: order invariance")],
        ]
        if "numba_abs_diff" in R:
            rows.append(["fastauc vs Numba kernel", f"{R['numba_abs_diff']:.1e}",
                         check(R["numba_abs_diff"] <= 1e-9, f"{name}: numba")])
        table(["check", "max |difference|", "result"], rows, right=(1,))
        add("")
        add("Tie handling (model CE)")
        table(["evaluator", "nDCG@10"], [
            ["fastauc (ties averaged over orderings)", f"{fa_ce:.6f}"],
            ["stable sort, relevant documents listed first", f"{pos_first:.6f}"],
            ["stable sort, relevant documents listed last", f"{pos_last:.6f}"],
        ], right=(1,))
        add(f"Position-based tie-breaking moves nDCG@10 by {pos_first - fa_ce:+.4f} / {pos_last - fa_ce:+.4f}, "
            f"against a model gap of {fa_ce - fa_de:+.4f}.")
        add("")
        P = R["paired"]
        add(f"Paired comparison, CE minus DE (query-level bootstrap, {P['n_bootstraps']:,} replicates)")
        table(["", "nDCG@10", "95% CI"], [
            ["CE", f"{fa_ce:.4f}", f"[{P['ci_ce'][0]:.4f}, {P['ci_ce'][1]:.4f}]"],
            ["DE", f"{fa_de:.4f}", f"[{P['ci_de'][0]:.4f}, {P['ci_de'][1]:.4f}]"],
            ["CE - DE, paired", f"{P['delta']:+.4f}", f"[{P['ci'][0]:+.4f}, {P['ci'][1]:+.4f}]"],
        ], right=(1, 2))
        p_floor = 2 / (P["n_bootstraps"] + 1)
        p_txt = f"p < {p_floor:.0e} (bootstrap floor)" if P["p"] <= p_floor * 1.0001 else f"p = {P['p']:.4f}"
        add(f"{p_txt}. Separate model CIs overlap: {'yes' if P['unpaired_overlap'] else 'no'}.")
        add("")
        add("Speed (median of repeated runs; ms)")
        speed_rows = [
            ["fastauc, 1 thread, query ids", t["fastauc_1t_groups"]],
            ["fastauc, 1 thread, group_sizes", t["fastauc_1t_group_sizes"]],
            [f"fastauc, all {fastauc.num_threads()} threads, group_sizes", t["fastauc_all_group_sizes"]],
            ["scikit-learn ndcg_score, one 2-D call", t["sklearn_2d"]],
            ["Python loop over queries (reference)", t["python_loop"]],
        ]
        if "numba_warm" in t:
            speed_rows.append(["Numba kernel, warm (precompiled)", t["numba_warm"]])
        if "numba_first_call_incl_jit" in t:
            speed_rows.append(["Numba kernel, first call incl. JIT compile", t["numba_first_call_incl_jit"]])
        base = t["fastauc_1t_group_sizes"]
        table(["implementation", "time (ms)", "time / fastauc 1 thread"],
              [[k, f"{v:.2f}", f"{v / base:.1f}x"] for k, v in speed_rows], right=(1, 2))
        add("")

    head = [
        "FASTAUC EVIDENCE REPORT (synthetic data)",
        f"fastauc {env['fastauc']} | Python {env['python']} | NumPy {env['numpy']} | "
        f"scikit-learn {env['scikit-learn']}" + (f" | Numba {env['numba']}" if env["numba"] else ""),
        f"{env['platform']} | {env['machine']} | {env['cpu_count']} CPUs"
        + (f" | runner {env['github_runner']}" if env["github_runner"] else ""),
        f"seed {args.seed}",
        "",
        "Overall: " + ("ALL ACCURACY CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"),
        "",
    ]
    report = "\n".join(head + LINES) + "\n"
    results["failures"] = failures
    (out / "evidence_report.txt").write_text(report)
    (out / "evidence.json").write_text(json.dumps(results, indent=1, default=float))
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"## fastauc evidence on {env['platform']} ({env['machine']})\n\n```\n{report}```\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
