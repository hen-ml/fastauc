"""LightGBM / XGBoost helpers, dataframe and tensor inputs, distributed merge."""
import sys
import types

import numpy as np
import pytest

import fastauc
from fastauc import lightgbm as fa_lgb
from fastauc import xgboost as fa_xgb


@pytest.fixture(scope="module")
def ranking_data():
    rng = np.random.default_rng(0)
    sizes = np.full(300, 20)
    x = rng.normal(size=(sizes.sum(), 6))
    rel = np.clip(np.round(x[:, 0] + 0.5 * x[:, 1] + rng.normal(size=len(x))), 0, 3)
    return sizes, x, rel


@pytest.fixture(scope="module")
def binary_data():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6000, 5))
    y = (rng.random(6000) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    return x, y


# ------------------------------------------------------------- name parsing


def test_metric_names_are_validated():
    with pytest.raises(ValueError, match="unknown ranking metric"):
        fa_lgb.ranking("ndgc@10")
    with pytest.raises(ValueError, match="cutoff"):
        fa_lgb.ranking("ndcg@0")
    with pytest.raises(ValueError, match="unknown binary metric"):
        fa_xgb.binary("calibration_ratio")
    f = fa_lgb.ranking("ndcg@10", "mrr", "gauc")
    assert f.__name__ == "fastauc_ranking" and "mrr" in f.__doc__


class _FakeDataset:
    """The three methods LightGBM's feval uses."""

    def __init__(self, label, group=None, weight=None):
        self._label, self._group, self._weight = label, group, weight

    def get_label(self):
        return self._label

    def get_group(self):
        return self._group

    def get_weight(self):
        return self._weight


def test_ranking_feval_matches_ranking_metrics(ranking_data):
    sizes, x, rel = ranking_data
    preds = np.round(x[:, 0], 1)  # ties
    out = fa_lgb.ranking("ndcg@10", "mrr@5", "map", "gauc")(preds, _FakeDataset(rel, sizes))
    ref = fastauc.ranking_metrics(rel, preds, group_sizes=sizes, ks=(10, 5, None))
    assert out == [("ndcg@10", ref["ndcg@10"], True), ("mrr@5", ref["mrr@5"], True),
                   ("map", ref["map"], True), ("gauc", ref["gauc"], True)]
    with pytest.raises(ValueError, match="no query groups"):
        fa_lgb.ranking()(preds, _FakeDataset(rel, None))


def test_binary_feval_directions(binary_data):
    x, y = binary_data
    p = 1 / (1 + np.exp(-x[:, 0]))
    out = fa_lgb.binary("roc_auc", "log_loss", "ece")(p, _FakeDataset(y))
    ref = fastauc.binary_metrics(y, p)
    assert out == [("roc_auc", ref["roc_auc"], True), ("log_loss", ref["log_loss"], False),
                   ("ece", ref["ece"], False)]


# ------------------------------------------------------------ real libraries


def test_lightgbm_training(ranking_data, binary_data):
    lgb = pytest.importorskip("lightgbm")
    sizes, x, rel = ranking_data
    half = sizes[:150].sum()
    train = lgb.Dataset(x[:half], rel[:half], group=sizes[:150])
    valid = lgb.Dataset(x[half:], rel[half:], group=sizes[150:], reference=train)
    hist = {}
    booster = lgb.train(
        {"objective": "lambdarank", "metric": "None", "verbose": -1}, train, num_boost_round=15,
        valid_sets=[valid], valid_names=["valid"], feval=fa_lgb.ranking("ndcg@10", "mrr@10"),
        callbacks=[lgb.record_evaluation(hist), lgb.early_stopping(5, verbose=False)],
    )
    preds = booster.predict(x[half:], num_iteration=booster.best_iteration)
    ref = fastauc.ranking_metrics(rel[half:], preds, group_sizes=sizes[150:], ks=(10,))
    assert hist["valid"]["ndcg@10"][booster.best_iteration - 1] == pytest.approx(ref["ndcg@10"])

    xb, yb = binary_data
    train = lgb.Dataset(xb[:4000], yb[:4000])
    valid = lgb.Dataset(xb[4000:], yb[4000:], reference=train)
    hist = {}
    lgb.train({"objective": "binary", "metric": "None", "verbose": -1}, train, num_boost_round=10,
              valid_sets=[valid], valid_names=["valid"],
              feval=fa_lgb.binary("roc_auc", "log_loss"), callbacks=[lgb.record_evaluation(hist)])
    assert 0.6 < hist["valid"]["roc_auc"][-1] < 1.0
    assert hist["valid"]["log_loss"][-1] < hist["valid"]["log_loss"][0]


def test_xgboost_training(ranking_data, binary_data):
    xgb = pytest.importorskip("xgboost")
    sizes, x, rel = ranking_data
    half = sizes[:150].sum()
    qid = np.repeat(np.arange(len(sizes)), sizes)
    dtrain = xgb.DMatrix(x[:half], rel[:half], qid=qid[:half])
    dvalid = xgb.DMatrix(x[half:], rel[half:], qid=qid[half:])
    hist = {}
    booster = xgb.train({"objective": "rank:ndcg", "disable_default_eval_metric": True}, dtrain,
                        num_boost_round=10, evals=[(dvalid, "valid")],
                        custom_metric=fa_xgb.ranking("mrr@10", "ndcg@10"), evals_result=hist,
                        verbose_eval=False)
    ref = fastauc.ranking_metrics(rel[half:], booster.predict(dvalid), group_sizes=sizes[150:],
                                  ks=(10,))
    assert hist["valid"]["ndcg@10"][-1] == pytest.approx(ref["ndcg@10"], rel=1e-6)
    assert hist["valid"]["mrr@10"][-1] == pytest.approx(ref["mrr@10"], rel=1e-6)

    xb, yb = binary_data
    dtrain = xgb.DMatrix(xb[:4000], yb[:4000])
    dvalid = xgb.DMatrix(xb[4000:], yb[4000:])
    hist = {}
    xgb.train({"objective": "binary:logistic", "disable_default_eval_metric": True}, dtrain,
              num_boost_round=10, evals=[(dvalid, "valid")], evals_result=hist,
              custom_metric=fa_xgb.binary("log_loss", "roc_auc"), maximize=True,
              verbose_eval=False)
    assert 0.6 < hist["valid"]["roc_auc"][-1] < 1.0


# ------------------------------------------------------- dataframe inputs


def test_polars_and_arrow_inputs():
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.integers(0, 2, n)
    s = np.round(rng.random(n), 2)
    q = np.array([f"q{i}" for i in rng.integers(0, 80, n)])
    ref = fastauc.ranking_metrics(y, s, q, ks=(5,))
    ref_b = fastauc.binary_metrics(y, s)
    inputs = []
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"y": y, "s": s, "q": q})
    inputs += [(df["y"], df["s"], df["q"]), (df["y"], df["s"], df["q"].cast(pl.Categorical))]
    pa = pytest.importorskip("pyarrow")
    inputs += [(pa.array(y), pa.array(s), pa.array(q)),
               (pa.chunked_array([y[:900], y[900:]]), pa.chunked_array([s[:900], s[900:]]),
                pa.chunked_array([q[:900], q[900:]]))]
    for yy, ss, qq in inputs:
        got = fastauc.ranking_metrics(yy, ss, qq, ks=(5,))
        assert got.keys() == ref.keys()
        for k in ref:
            assert got[k] == pytest.approx(ref[k], nan_ok=True), k
        assert fastauc.binary_metrics(yy, ss)["roc_auc"] == ref_b["roc_auc"]


class _FakeTensor:
    """Duck-types the part of a torch tensor that fastauc relies on."""

    def __init__(self, a):
        self._a = np.asarray(a)
        self.calls = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        return self._a

    def __array__(self, *args, **kwargs):  # what a GPU tensor or one with grad cannot do
        raise TypeError("can't convert a device tensor with grad to numpy")


class _FakeBf16Tensor(_FakeTensor):
    """A bfloat16 tensor: .numpy() fails until it is widened with .float()."""

    def __init__(self, a):
        super().__init__(a)
        self.widened = False

    def float(self):
        self.calls.append("float")
        self.widened = True
        return self

    def numpy(self):
        if not self.widened:
            raise TypeError("Got unsupported ScalarType BFloat16")
        return self._a


def test_bfloat16_tensors_are_widened():
    s = _FakeBf16Tensor([0.125, 0.875, 0.75, 0.25])
    assert fastauc.roc_auc([0, 1, 1, 0], s) == 1.0
    assert s.calls == ["detach", "cpu", "float"]


def test_xgboost_per_query_weights(ranking_data):
    xgb = pytest.importorskip("xgboost")
    sizes, x, rel = ranking_data
    qid = np.repeat(np.arange(len(sizes)), sizes)
    w = np.linspace(0.5, 2.0, len(sizes))
    d = xgb.DMatrix(x, rel, qid=qid)
    d.set_weight(w)  # one weight per query, as XGBoost requires for ranking
    preds = np.round(x[:, 0], 1)
    got = fa_xgb.ranking("ndcg@10")(preds, d)
    # DMatrix stores weights as float32.
    ref = fastauc.ranking_metrics(rel, preds, group_sizes=sizes, ks=(10,),
                                  sample_weight=np.repeat(w.astype(np.float32), sizes))
    assert got[0][0] == "ndcg@10" and got[0][1] == pytest.approx(ref["ndcg@10"], rel=1e-12)
    assert ref["ndcg@10"] != fastauc.ranking_metrics(rel, preds, group_sizes=sizes, ks=(10,))["ndcg@10"]


def test_tensor_like_inputs_are_detached():
    y = _FakeTensor([0, 1, 1, 0])
    s = _FakeTensor([0.1, 0.9, 0.8, 0.3])
    assert fastauc.roc_auc(y, s) == 1.0
    assert s.calls == ["detach", "cpu"]
    acc = fastauc.BinaryAccumulator().update(_FakeTensor([0, 1]), _FakeTensor([0.2, 0.7]))
    assert acc.roc_auc() == 1.0


# ------------------------------------------------------- distributed merge


def test_all_gather_without_a_process_group_is_a_copy():
    acc = fastauc.BinaryAccumulator().update([0, 1, 1], [0.1, 0.8, 0.3])
    out = acc.all_gather()
    assert out is not acc and out.roc_auc() == acc.roc_auc()


def test_all_gather_merges_every_rank(monkeypatch):
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 3000)
    s = np.round(rng.random(3000), 3)
    shards = [fastauc.BinaryAccumulator().update(y[r::3], s[r::3]) for r in range(3)]

    dist = types.ModuleType("torch.distributed")
    dist.is_available = lambda: True
    dist.is_initialized = lambda: True
    dist.get_world_size = lambda group=None: 3

    def all_gather_object(out, obj, group=None):
        for r in range(3):  # what each rank would contribute
            out[r] = shards[r].state()

    dist.all_gather_object = all_gather_object
    torch = types.ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    total = shards[1].all_gather()
    assert total.n_samples == 3000
    assert total.compute()["roc_auc"] == fastauc.binary_metrics(y, s)["roc_auc"]
