# Confidence intervals and significance tests

## Bootstrap intervals

```python
b = fastauc.bootstrap_roc_auc_ap(y, s, n_bootstraps=2000, seed=0)
b.roc_auc_estimate            # point estimate
b.ci("roc_auc", level=0.95)   # (low, high) percentile interval
b.std("average_precision")    # bootstrap standard error
```

Scores are sorted once and each replicate is a linear scan, so thousands of
replicates cost about as much as a few dozen plain evaluations. Results
depend only on `seed`: each replicate has its own independently seeded
xoshiro256++ stream, so the thread count does not matter.

**Resample groups, not rows, when rows are correlated.** Impressions from
the same user, query or session are not independent, and a row-level
bootstrap then reports intervals that are too narrow:

```python
b = fastauc.bootstrap_roc_auc_ap(y, s, groups=user_id, n_bootstraps=2000)

r = fastauc.bootstrap_ranking_metrics(relevance, s, query_id, ks=(10,), n_bootstraps=2000)
r.estimate["ndcg@10"], r.ci("ndcg@10"), r.std("mrr@10")
```

Ranking metrics always resample whole queries. `stratified=True` keeps the
class balance fixed when positives are rare; `sample_weight` is supported.

## Is model A better than model B?

### DeLong, for ROC-AUC on independent rows

```python
d = fastauc.delong_roc_auc(y, np.stack([scores_a, scores_b]))
d.roc_auc, d.ci(), d.std()                       # per model
diff, (low, high), p_value = d.compare(0, 1)     # paired test of a vs b
```

The fast algorithm of Sun and Xu (2014): one sort per model, no
resampling. It assumes independent rows and no sample weights.

### Paired bootstrap, for any metric, clustered rows or weights

```python
b = fastauc.bootstrap_roc_auc_ap(y, np.stack([scores_a, scores_b]), groups=user_id)
diff, (low, high), p_value = b.compare(0, 1, metric="roc_auc")

r = fastauc.bootstrap_ranking_metrics(relevance, np.stack([scores_a, scores_b]), query_id)
diff, (low, high), p_value = r.compare(0, 1, metric="ndcg@10")
```

Every replicate uses the same resample for all models, so the comparison is
paired and far more sensitive than comparing two separate intervals.

### Per-query tests, as in IR evaluation

```python
_, a = fastauc.ranking_metrics(rel, scores_a, query_id, ks=(10,), per_group=True)
_, b = fastauc.ranking_metrics(rel, scores_b, query_id, ks=(10,), per_group=True)

fastauc.paired_test(a["ndcg@10"], b["ndcg@10"])                 # randomization test
fastauc.paired_test(a["ndcg@10"], b["ndcg@10"], method="t")     # paired t-test
```

`paired_test` takes any two aligned arrays of per-unit values (queries,
users, folds) and drops pairs where either value is `NaN`.

- `method="randomization"` (default) is the paired sign-flip test (Fisher's
  randomization test), the usual choice in IR evaluation. When
  $2^n \le$ `n_resamples` it enumerates every sign pattern and the p-value
  is exact; otherwise it draws `n_resamples` patterns with
  $p = (1 + \text{count}) / (1 + \text{n\_resamples})$, deterministic for a
  given `seed` and independent of the thread count.
- `method="t"` is the paired t-test, computed without scipy (it matches
  `scipy.stats.ttest_rel` to about 1e-12).
- `alternative` is `"two-sided"`, `"greater"` (A better) or `"less"`.

The result carries `mean_difference`, `p_value`, `n`, `statistic` and
`exact`.

### Many comparisons

Comparing several systems against a baseline, or one system on several
metrics, inflates false positives. `holm` adjusts p-values with the
Holm-Bonferroni procedure, valid under any dependence between the tests:

```python
p = [fastauc.paired_test(a, baseline).p_value for a in systems]
fastauc.holm(p)        # reject where the adjusted p-value <= alpha
```
