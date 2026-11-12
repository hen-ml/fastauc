# Ranking metrics

## Per-query metrics in one call

```python
fastauc.ranking_metrics(relevance, s, query_id, ks=(1, 5, 10, None))
# {'gauc', 'gauc_macro',
#  'ndcg@1', 'mrr@1', 'map@1', 'recall@1', 'precision@1', 'hit_rate@1', ... '@5', '@10',
#  'ndcg', 'mrr', 'map', 'recall', 'precision', 'hit_rate',      # k=None: whole list
#  'n_groups', 'n_groups_ranked', 'n_groups_auc'}
```

- `relevance` is binary or graded (non-negative). An item counts as relevant when it is `> 0`; grades only change NDCG (`gain="exponential"` or `"linear"`).
- Every cutoff comes from one sort per query, so four cutoffs cost about as much as one.
- `gauc` is ROC-AUC per query weighted by query size; `gauc_macro` is the plain mean.
- Queries with no relevant item are left out of the averages (`no_relevant="zero"` counts them as 0 instead). `n_groups_ranked` tells you how many were used.
- `sample_weight` weights the average over queries: a query's weight is the mean weight of its rows (the LightGBM convention).
- `per_group=True` also returns every per-query value, ready for [paired tests](significance.md).

## Query ids

Pass ids via `groups`, in any row order and any dtype, or `group_sizes=` for
data already laid out query by query (the LightGBM and XGBoost convention).

- Integer ids in a compact range take an O(N) path.
- String ids (`str`, `bytes`, or object arrays of strings) go through an exact hash table in Rust: the hash only picks a bucket and ids are compared byte for byte, so two different ids are never merged, whatever their hashes. Only the distinct ids are sorted, to keep the output order of `np.unique`. This is 4 to 6 times faster than `np.unique` on string ids, with identical output.
- Anything else falls back to `np.unique`.

!!! note "Why not hash ids to integers?"
    A common speed trick is to hash string ids to 64-bit integers and group by
    the hash. Two ids that collide are then silently merged into one query,
    and the metric changes. fastauc gets the speed without that risk.

## Incomplete lists: complete judgments

When the ranked lists can miss relevant items (first-stage retrieval, or
recommendations scored over a candidate subset), recall, MAP and the ideal
DCG must count every relevant item, not only the ones that were retrieved.
Pass the complete relevance judgments (qrels):

```python
fastauc.ranking_metrics(
    rel, s, query_id, ks=(10,),
    judgments=(qrels_query_id, qrels_relevance),   # one entry per judged item
    map_denominator="total",
)
```

With `map_denominator="total"` and `gain="linear"`, the results match
trec_eval's `ndcg_cut_10`, `recall_10`, `P_10`, `map_cut_10` and
`recip_rank` (tested against pytrec_eval on data without ties).

`map_denominator` chooses how AP@k is normalised:

| `map_denominator` | divides AP@k by | perfect ranking scores | used by |
|---|---|---|---|
| `"min"` (default) | $\min(R, k)$ | 1.0 | most recommender papers |
| `"total"` | $R$ | $\min(R, k)/R$ | trec_eval, ranx |

where $R$ is the number of relevant items for the query.

## Inside training loops

See [Integrations](integrations.md) for one-line LightGBM and XGBoost
metrics, PyTorch Lightning, and dataframe inputs.
