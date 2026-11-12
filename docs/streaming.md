# Streaming and distributed evaluation

`BinaryAccumulator` computes binary metrics over data that arrives in
batches, or is split across workers, with no approximation:

```python
acc = fastauc.BinaryAccumulator()
for y, s in batches:
    acc.update(y, s)            # sample_weight= is optional
acc.compute()                   # same keys as fastauc.binary_metrics
acc.roc_auc(), acc.average_precision(), acc.roc_curve()
```

## Why it is exact

Every threshold metric depends on the data only through the table of
distinct scores with the total positive and negative weight at each one:
samples with equal scores are interchangeable. `update` folds a batch into
that table (one parallel sort of the batch, then a linear merge), so
`compute` returns what `binary_metrics` returns on all rows at once.

- With unit weights the table holds integer counts, so ROC-AUC, average
  precision, PR-AUC, KS and max F1 are **bit-for-bit identical** to the
  one-shot result, whatever the batch sizes and merge order.
- Sums of non-integer weights, and the calibration losses, agree up to
  floating-point rounding (relative error around 1e-15).

Memory grows with the number of distinct scores, not rows: at most 65,536
for float16 or bfloat16 scores, at most the number of rows for float32 or
float64. Binned streaming AUC (Keras `tf.keras.metrics.AUC`, for example,
uses 200 thresholds by default) has fixed memory but is approximate; when
scores are low-precision anyway, the exact table is about as small.

## Across workers

Accumulators combine exactly, in any order:

```python
total = acc_worker_0 + acc_worker_1 + acc_worker_2
acc_worker_0.merge(acc_worker_1, acc_worker_2)   # in place
```

They pickle, and `state()` / `BinaryAccumulator.from_state()` expose plain
arrays for `torch.distributed.all_gather_object`, MPI or Spark:

```python
state = acc.state()      # {'score', 'pos', 'neg', 'n_samples'}
acc = fastauc.BinaryAccumulator.from_state(state)
```

## Scope

The accumulator covers single-model binary metrics. Per-query ranking
metrics need each query's rows together; for those, evaluate per batch of
whole queries and combine the per-query values (`per_group=True`).
