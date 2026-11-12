# Integrations

## LightGBM

```python
import lightgbm as lgb
from fastauc import lightgbm as fa_lgb

lgb.train(
    {"objective": "lambdarank", "metric": "None"}, train,
    valid_sets=[valid],
    feval=fa_lgb.ranking("ndcg@10", "mrr@10", "map@10", "gauc"),
    callbacks=[lgb.early_stopping(20)],
)

lgb.train(
    {"objective": "binary", "metric": "None"}, train,
    valid_sets=[valid],
    feval=fa_lgb.binary("roc_auc", "average_precision", "log_loss"),
)
```

- `ranking(*metrics)` takes names such as `"ndcg@10"`, `"mrr@5"`, `"map@10"`, `"recall@20"`, `"precision@1"`, `"hit_rate@3"`, the whole-list `"ndcg"`, `"gauc"` and `"gauc_macro"`. Query groups come from the evaluation `Dataset` (`group=`); row weights, if set, weight the average over queries. All metrics come from one sort per query.
- `binary(*metrics)` takes `"roc_auc"`, `"average_precision"`, `"pr_auc"`, `"ks"`, `"f1_max"`, and the losses `"log_loss"`, `"normalized_entropy"`, `"brier"`, `"ece"`. The losses need probabilities, which is what LightGBM passes for its built-in `binary` objective.
- Each metric reports whether higher is better, so early stopping works.

## XGBoost

```python
import xgboost as xgb
from fastauc import xgboost as fa_xgb

xgb.train(
    {"objective": "rank:ndcg", "disable_default_eval_metric": True}, dtrain,
    evals=[(dvalid, "valid")],
    custom_metric=fa_xgb.ranking("mrr@10", "ndcg@10"),
    maximize=True, early_stopping_rounds=20,
)
```

Same metric names as for LightGBM. Query groups come from the `DMatrix`
(`qid=` or `set_group`). XGBoost early-stops on the last metric listed, and
the direction is set by `maximize`.

## Why not the built-in metrics?

Early in training, many predictions are equal, and LightGBM's and XGBoost's
own NDCG and MAP then depend on the order of the rows within each query (see
[Exact tie handling](ties.md#how-other-libraries-behave)). That can move the
early-stopping round. fastauc's values depend only on the predictions.

## PyTorch and PyTorch Lightning

Tensors are accepted directly by every fastauc function, on any device and
with or without `requires_grad`: they are detached and copied to host
memory.

For evaluation spread over batches and GPUs, use `BinaryAccumulator`:
`update` per batch, then `all_gather()` merges the per-rank accumulators
through `torch.distributed` and gives every rank the exact total.

```python
import fastauc
import lightning as L
import torch


class Model(L.LightningModule):
    def on_validation_epoch_start(self):
        self.acc = fastauc.BinaryAccumulator()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.acc.update(y, torch.sigmoid(self(x)))

    def on_validation_epoch_end(self):
        m = self.acc.all_gather().compute()   # identical on every rank
        self.log_dict({"val_auc": m["roc_auc"], "val_ne": m["normalized_entropy"]})
```

Outside a process group `all_gather()` returns a copy, so the same code runs
on one device. For per-query ranking metrics, collect the predictions and
call `ranking_metrics` once per epoch; it takes well under a second for a
million rows.

## Polars, PyArrow and pandas

Columns are accepted as they are, including string query ids, which go
through the exact hash grouping:

```python
import polars as pl

df = pl.read_parquet("eval.parquet")
fastauc.ranking_metrics(df["clicked"], df["score"], df["query_id"], ks=(10,))
fastauc.binary_metrics(df["clicked"], df["score"])
```

PyArrow `Array` and `ChunkedArray`, pandas `Series` and NumPy arrays work the
same way.
