"""fastauc.sklearn must return what sklearn.metrics returns, warnings and errors included."""
import warnings

import numpy as np
import pytest

from fastauc import sklearn as fm

sklearn = pytest.importorskip("sklearn")
from sklearn import metrics as sm  # noqa: E402

_V = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
# The drop-in follows scikit-learn 1.7+ behaviour (2-D Brier score, NaN with a
# warning for one-class ROC AUC, machine-epsilon clipping in log_loss, ...).
needs_recent = pytest.mark.skipif(_V < (1, 7), reason="parity target is scikit-learn >= 1.7")


def _same(a, b):
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
        return
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)


def _run(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    kinds = sorted(
        w.category.__name__ for w in caught
        if not issubclass(w.category, (FutureWarning, DeprecationWarning))
    )
    return out, kinds


def check(name, *args, **kw):
    ours, w1 = _run(getattr(fm, name), *args, **kw)
    theirs, w2 = _run(getattr(sm, name), *args, **kw)
    _same(ours, theirs)
    assert w1 == w2, (w1, w2)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    s = np.round(rng.normal(size=n) + y, 1)  # many ties
    w = rng.random(n)
    k = 4
    yc = rng.integers(0, k, n)
    P = np.round(rng.dirichlet(np.ones(k), n), 2)
    P /= P.sum(axis=1, keepdims=True)
    Y = rng.integers(0, 2, (n, 3))
    S = np.round(rng.normal(size=(n, 3)) + Y, 1)
    return dict(y=y, s=s, w=w, p=1 / (1 + np.exp(-s)), yc=yc, P=P, Y=Y, S=S)


# ----------------------------------------------------------------- binary


@needs_recent
@pytest.mark.parametrize("weighted", [False, True])
def test_binary_scores(data, weighted):
    y, s = data["y"], data["s"]
    kw = {"sample_weight": data["w"]} if weighted else {}
    check("roc_auc_score", y, s, **kw)
    check("average_precision_score", y, s, **kw)
    for max_fpr in (0.05, 0.3, 1.0):
        check("roc_auc_score", y, s, max_fpr=max_fpr, **kw)


@needs_recent
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_curves(data, drop, weighted):
    y, s = data["y"], data["s"]
    kw = {"sample_weight": data["w"]} if weighted else {}
    check("roc_curve", y, s, drop_intermediate=drop, **kw)
    check("precision_recall_curve", y, s, drop_intermediate=drop, **kw)


@needs_recent
def test_label_encodings(data):
    y, s, p = data["y"], data["s"], data["p"]
    words = np.where(y == 1, "spam", "ham")
    signed = np.where(y == 1, 1, -1)
    check("roc_auc_score", words, s)
    check("roc_auc_score", signed, s)
    check("average_precision_score", words, s, pos_label="spam")
    check("average_precision_score", signed, s)
    check("roc_curve", words, s, pos_label="spam")
    check("roc_curve", signed, s)
    check("precision_recall_curve", words, s, pos_label="spam")
    check("brier_score_loss", words, p, pos_label="spam")
    check("log_loss", words, p)
    check("roc_auc_score", data["Y"][:, :1], data["S"][:, :1])  # (n, 1) column


@needs_recent
def test_undefined_cases_warn_like_sklearn():
    x = np.arange(5.0)
    check("roc_auc_score", np.ones(5), x)
    check("average_precision_score", np.zeros(5), x)
    check("roc_curve", np.ones(5), x)
    check("roc_curve", np.zeros(5), x)
    check("precision_recall_curve", np.zeros(5), x)


# ----------------------------------------------------------- multiclass


@needs_recent
@pytest.mark.parametrize("multi_class", ["ovr", "ovo"])
@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_multiclass_roc_auc(data, multi_class, average):
    check("roc_auc_score", data["yc"], data["P"], multi_class=multi_class, average=average)


@needs_recent
def test_multiclass_ovr_options(data):
    yc, P = data["yc"], data["P"]
    check("roc_auc_score", yc, P, multi_class="ovr", average=None)
    check("roc_auc_score", yc, P, multi_class="ovr", average="micro")
    check("roc_auc_score", yc, P, multi_class="ovr", sample_weight=data["w"])
    check("roc_auc_score", yc, P, multi_class="ovr", labels=[0, 1, 2, 3])
    for average in ("macro", "micro", None):
        check("average_precision_score", yc, P, average=average)


# ----------------------------------------------------------- multilabel


@needs_recent
@pytest.mark.parametrize("average", ["macro", "weighted", "micro", "samples", None])
@pytest.mark.parametrize("weighted", [False, True])
def test_multilabel(data, average, weighted):
    kw = {"sample_weight": data["w"]} if weighted else {}
    check("roc_auc_score", data["Y"], data["S"], average=average, **kw)
    check("average_precision_score", data["Y"], data["S"], average=average, **kw)


# ---------------------------------------------------------------- losses


@needs_recent
def test_losses(data):
    y, p, w = data["y"], data["p"], data["w"]
    check("log_loss", y, p)
    check("log_loss", y, p, sample_weight=w, normalize=False)
    check("log_loss", data["yc"], data["P"])
    check("log_loss", [0, 1, 1], [0, 1, 0])  # hard 0/1 predictions are clipped
    # Rows that do not sum to one: scikit-learn >= 1.5 warns and does NOT
    # renormalise (older versions did), and neither do we.
    check("log_loss", [0, 1, 1], [[0.2, 0.3], [0.1, 0.1], [0.5, 0.5]])
    check("log_loss", [1, 1], [0.9, 0.8], labels=[0, 1])
    check("brier_score_loss", y, p, sample_weight=w)
    check("brier_score_loss", data["yc"], data["P"])
    check("brier_score_loss", [0, 1, 1], [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])


def test_old_argument_names_still_work(data):
    y, p = data["y"], data["p"]
    assert fm.log_loss(y, y_pred=p) == fm.log_loss(y, p)
    assert fm.brier_score_loss(y, y_prob=p) == fm.brier_score_loss(y, p)
    assert fm.log_loss([0, 1], [0.0, 1.0], eps=1e-15) == pytest.approx(1e-15)
    with pytest.raises(ValueError):
        fm.log_loss(y, p, y_pred=p)


def test_auc(data):
    fpr, tpr, _ = sm.roc_curve(data["y"], data["s"])
    check("auc", fpr, tpr)
    check("auc", fpr[::-1], tpr[::-1])
    assert fm.auc([0, 1], [1, 1]) == 1.0


# --------------------------------------------------------------- ranking


@needs_recent
@pytest.mark.parametrize("k", [None, 1, 3, 20])
def test_ndcg_dcg(k):
    rng = np.random.default_rng(0)
    G = rng.integers(0, 4, (60, 8))
    S = np.round(rng.normal(size=(60, 8)), 1)
    G[3] = 0  # no relevant item: NDCG 0 for that row, as in scikit-learn
    w = rng.random(60)
    check("ndcg_score", G, S, k=k)
    check("ndcg_score", G, S, k=k, sample_weight=w)
    for base in (2, 10):
        check("dcg_score", G, S, k=k, log_base=base)
        check("dcg_score", G, S, k=k, log_base=base, sample_weight=w)


def test_ignore_ties_is_accepted_but_ties_stay_exact():
    G = np.array([[1, 0, 0, 0]])
    S = np.zeros((1, 4))  # every item tied
    # Exact answer: expected NDCG@1 over all orderings = P(relevant item first) = 1/4.
    assert fm.ndcg_score(G, S, k=1, ignore_ties=True) == pytest.approx(0.25)
    assert fm.ndcg_score(G, S, k=1) == pytest.approx(0.25)
    # scikit-learn with ignore_ties=True keeps input order, so moving the
    # relevant item changes its answer; ours does not move.
    flipped = G[:, ::-1]
    assert fm.ndcg_score(flipped, S, k=1, ignore_ties=True) == pytest.approx(0.25)
    assert sm.ndcg_score(G, S, k=1, ignore_ties=True) != sm.ndcg_score(flipped, S, k=1, ignore_ties=True)


# ---------------------------------------------------------------- errors

P3 = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4], [0.5, 0.25, 0.25]])


@needs_recent
@pytest.mark.parametrize(
    "name, args, kw",
    [
        ("roc_auc_score", ([0, 1, 2, 1], P3), {}),
        ("roc_auc_score", ([0, 1, 2, 1], P3), {"multi_class": "ovr", "max_fpr": 0.5}),
        ("roc_auc_score", ([0, 1, 2, 1], P3 * 2), {"multi_class": "ovr"}),
        ("roc_auc_score", ([0, 1, 2, 1], P3[:, 0]), {"multi_class": "ovr"}),
        ("roc_auc_score", ([0, 1, 1], [0.1, 0.2, 0.3]), {"average": "bogus"}),
        ("average_precision_score", ([2, 3, 3, 2], [0.1, 0.2, 0.3, 0.4]), {}),
        ("average_precision_score", ([0, 1, 2, 1], P3), {"pos_label": 2}),
        ("ndcg_score", ([[1]], [[1]]), {}),
        ("ndcg_score", ([[1, -1]], [[1, 2]]), {}),
        ("ndcg_score", ([1, 0], [1, 2]), {}),
        ("log_loss", ([1, 1], [0.9, 0.8]), {}),
        ("roc_curve", (["a", "b", "a"], [0.1, 0.2, 0.3]), {}),
        ("auc", ([0, 1, 0.5], [0, 1, 1]), {}),
        ("auc", ([0], [0]), {}),
    ],
)
def test_errors_match(name, args, kw):
    with pytest.raises(ValueError) as theirs:
        getattr(sm, name)(*args, **kw)
    with pytest.raises(ValueError) as ours:
        getattr(fm, name)(*args, **kw)
    # Messages that print a Python set can differ only in element order.
    assert str(ours.value).split("{")[0] == str(theirs.value).split("{")[0]


@needs_recent
def test_ovo_average_none_not_implemented():
    for mod in (sm, fm):
        with pytest.raises(NotImplementedError):
            mod.roc_auc_score([0, 1, 2, 1], P3, multi_class="ovo", average=None)
