# Performance

All numbers on this page come from a **2-core** x86-64 Linux VM with
Python 3.13. The parallel paths scale with cores, so expect larger gaps on
real hardware, and run the scripts in `benchmarks/` before quoting numbers.

## Threads

fastauc releases the GIL and uses all cores by default. Inside data-loader
workers, joblib jobs or any code that is already parallel, call
`fastauc.set_num_threads(1)` to avoid oversubscription
(`set_num_threads(None)` restores the default; the `RAYON_NUM_THREADS`
environment variable also works). Results never depend on the thread count:
work is split in fixed chunks and reduced in a fixed order.

On free-threaded CPython 3.14t the extension declares itself safe
to run without the GIL, so calls from many Python threads run in parallel.
The test suite runs with `PYTHON_GIL=0` and checks that concurrent calls give
the single-threaded answers.

## Against scikit-learn

`python benchmarks/bench.py` (ROC-AUC + average precision unless noted):

| Task | scikit-learn | fastauc | Speedup |
|---|---:|---:|---:|
| Single pair, N = 1,000 | 2.1 ms | 0.03 ms | 83x |
| Single pair, N = 100,000 | 29 ms | 3.2 ms | 9x |
| Single pair, N = 10,000,000 | 5.9 s | 0.55 s | 11x |
| 1,000 models, 1 target, N = 10,000 | 4.46 s | 0.20 s | 22x |
| 1 model, 1,000 targets, N = 10,000 | 4.42 s | 0.039 s | 113x |
| 1,000 bootstrap replicates, N = 10,000 | 4.61 s | 0.049 s | 95x |
| Grouped AUC, N = 1,000,000, 20,000 groups | 37.3 s | 0.034 s | 1093x |
| NDCG@10, N = 1,000,000, 20,000 groups | 6.8 s | 0.051 s | 132x |
| `binary_metrics`, N = 1,000,000 | 0.79 s | 0.059 s | 13x |

The drop-in module, same calls on both sides
(`python benchmarks/bench_sklearn.py`):

| Call | scikit-learn | fastauc.sklearn | Speedup |
|---|---:|---:|---:|
| `roc_auc_score`, N = 1M | 163 ms | 36 ms | 5x |
| `roc_auc_score(max_fpr=0.1)`, N = 1M | 177 ms | 53 ms | 3x |
| `average_precision_score`, N = 1M | 176 ms | 33 ms | 5x |
| `log_loss`, N = 1M | 80 ms | 20 ms | 4x |
| `roc_auc_score(multi_class='ovr')`, 10 classes, N = 200k | 260 ms | 51 ms | 5x |
| `roc_auc_score(multi_class='ovo')`, 10 classes, N = 200k | 607 ms | 136 ms | 4x |
| `roc_auc_score`, multilabel, 20 labels, N = 200k | 677 ms | 159 ms | 4x |
| `ndcg_score(k=10)`, 10,000 queries x 100 docs | 249 ms | 23 ms | 11x |

## Against a hand-written Numba kernel

`benchmarks/evidence.py` contains a tie-aware NDCG@10 kernel in Numba, the
kind of code people write when scikit-learn is too slow. On one thread,
after JIT warm-up, against `ranking_metrics` computing all six ranking
metrics plus GAUC with exact ties:

| Workload | Numba (NDCG@10 only) | fastauc, `group_sizes` | fastauc, query ids |
|---|---:|---:|---:|
| 2,110 short lists of 10, string ids | 1.29 ms | 0.85 ms | 1.24 ms |
| 8,956 ragged pools of 8 to 95 items, integer ids | 9.3 ms | 7.8 ms | 8.1 ms |

Medians of 7 runs on a noisy VM (single runs vary by up to 30%). fastauc
needs no warm-up (the first Numba call spends about 3 s compiling) and runs
on all cores when allowed.

## Against ranx and fastroc

Against ranx 0.3.21 (`python benchmarks/bench_vs_others.py`; NDCG@10 +
MRR@10 + MAP@10 on 1,000,000 rows in 20,000 queries, no tied scores,
results identical to six decimals):

| | Time |
|---|---:|
| fastauc, straight from NumPy arrays | 0.036 s |
| ranx `evaluate`, after JIT warm-up | 0.93 s |
| ranx, building `Qrels` and `Run` from the same arrays | + 5.7 s |

Against fastroc 1.1.3, a C implementation of ROC-AUC, with 2 threads each:
10,000,000 samples take 2.5 s in fastroc and 0.53 s in fastauc (which also
computes average precision), and 1,000 models on 10,000 samples take 0.65 s
versus 0.18 s.

## On a Mac

The [Evidence (macOS)](https://github.com/hen-ml/fastauc/actions/workflows/evidence.yml)
workflow runs `benchmarks/evidence.py` on Apple silicon and Intel Macs and
shows the accuracy checks and timings in each run's summary.
