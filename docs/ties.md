# Exact tie handling

## The rule

Items with equal scores are interchangeable. No result may depend on the
order of the input rows, on item ids, or on a random seed.

- **Top-k ranking metrics** (NDCG, MRR, MAP, recall, precision, hit rate)
  return the expected value over all orderings of the tied items, computed
  in closed form. It is what averaging over infinitely many random
  tie-breaks would give, without the randomness.
- **Threshold metrics** (ROC-AUC, average precision, PR-AUC, KS, F1,
  operating points, curves) place thresholds only between distinct scores,
  so a tied block is one point on the curve. For ROC-AUC this gives a tied
  positive and negative pair the usual half credit.
- **Lift and quantile-binned calibration error** split a tied block that
  straddles a cut in proportion, instead of by position.

With no ties these are the textbook definitions.

## The closed form

Take a tied block that occupies ranks $a+1, \dots, a+t$ and holds $r$
relevant items with total gain $G$. Every item in the block is equally
likely to land at each of those ranks, so:

- **DCG**: the block contributes $\frac{G}{t} \sum_{i=a+1}^{a+t} d_i$, where
  $d_i$ is the discount at rank $i$ (truncated at the cutoff $k$).
- **Precision@k and recall@k**: when the cutoff falls inside the block,
  $k - a$ of its $t$ slots are kept, so the expected number of relevant items
  kept is $r (k - a) / t$.
- **Reciprocal rank**: only the first block holding a relevant item matters.
  Its first relevant item sits at rank $a + q$ with probability
  $\binom{t-q}{r-1} / \binom{t}{r}$, and the expectation of $1/(a+q)$ is
  summed exactly.
- **Average precision**: rank $a + q$ holds a relevant item with probability
  $r/t$; given that, the expected number of relevant items at or above it is
  $c + 1 + (q-1)(r-1)/(t-1)$, where $c$ counts the relevant items ranked
  above the block. Summing over $q$ leaves harmonic numbers, so each block
  costs $O(1)$ per cutoff.
- **Hit rate**: one minus the probability that none of the kept slots holds a
  relevant item, a product of hypergeometric terms.

The test suite checks every formula against brute-force enumeration of all
orderings, and checks that every metric is unchanged under row permutation,
including the adversarial "relevant items first" layout.

## Prior work

The tie-aware forms of precision, recall, F1, average precision, reciprocal
rank and NDCG were derived by McSherry and Najork (2008), who showed they
cost no more than the tie-oblivious measures. Cabanac et al. (2010) showed
that trec_eval breaks ties by document name and that this changes TREC
results. fastauc applies the same principle to every metric it computes,
including GAUC, bootstrap replicates and streaming.

- F. McSherry and M. Najork. *Computing Information Retrieval Performance
  Measures Efficiently in the Presence of Tied Scores.* ECIR 2008, LNCS 4956,
  pp. 414 to 421. [doi:10.1007/978-3-540-78646-7_38](https://doi.org/10.1007/978-3-540-78646-7_38)
- G. Cabanac, G. Hubert, M. Boughanem and C. Chrisment. *Tie-Breaking Bias:
  Effect of an Uncontrolled Parameter on Information Retrieval Evaluation.*
  CLEF 2010, LNCS 6360, pp. 112 to 123.
  [doi:10.1007/978-3-642-15998-5_13](https://doi.org/10.1007/978-3-642-15998-5_13)

For what this does to real training runs, see the
[case study](case-study.md): on Istella-S LETOR, LightGBM's own NDCG@10
after one round ranges from 0.465 to 0.694 depending on row order.

## How other libraries behave

`python benchmarks/tie_leak.py` runs other libraries on 1,000 queries of 10
documents (2 relevant each) scored by a model that gives every document the
same score. The two layouts hold identical data; only the input order and
document ids differ. A tie-correct metric returns the same chance-level
value for both.

| Library | Metrics checked | Relevant first | Relevant last | Correct |
|---|---|---:|---:|---:|
| **fastauc** | NDCG@3, MAP@3, MRR@3, P@1 | 0.26, 0.20, 0.34, 0.20 | same | same |
| scikit-learn 1.9.1 `ndcg_score`, `roc_auc_score` | NDCG@3, AUC | 0.26, 0.50 | same | same |
| ranx 0.3.21 | NDCG@3, MAP@3, MRR@3, P@1 | 1.0 | 0.0 | 0.26, 0.20, 0.34, 0.20 |
| pytrec_eval 0.5.11 / ir-measures 0.4.3 | NDCG@3, P@1 | 0.0 | 1.0 | 0.26, 0.20 |
| LightGBM 4.7.0 built-in eval | NDCG@3, MAP@3 | 1.0 | 0.0 | 0.26, 0.20 |
| XGBoost 3.4.1 built-in eval | NDCG@3, MAP@3 | 1.0 | 0.0 | 0.26, 0.20 |
| fastroc 1.1.3 | AUC | 0.5005 | 0.4995 | 0.5 |

From reading the source of libraries not run above: torchmetrics 1.9.0
averages ties for NDCG (a port of scikit-learn's method) but its retrieval
MRR, MAP, precision, recall and hit rate take `topk`/`argsort` with no tie
handling; TorchRec 1.9.2 computes AUC, GAUC and NDCG from an `argsort` with
no tie grouping. TF-Ranking breaks ties randomly, which is unbiased but noisy
and seed-dependent; fastauc returns the exact average of that randomization.
