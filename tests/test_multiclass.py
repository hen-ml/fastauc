"""Multiclass and partial ROC-AUC against scikit-learn."""
import math

import numpy as np
import pytest
import sklearn
from sklearn.metrics import roc_auc_score

import fastauc

SKLEARN = tuple(int(p) for p in sklearn.__version__.split(".")[:2])


def probs(n=2000, C=4, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, n)
    z = rng.normal(size=(n, C))
    z[np.arange(n), y] += 1.0
    P = np.exp(z)
    P /= P.sum(1, keepdims=True)
    if ties:
        P = np.round(P, 2)
        P /= P.sum(1, keepdims=True)
    return y, P, rng.random(n)


@pytest.mark.parametrize("average", ["macro", "weighted", "micro", None])
@pytest.mark.parametrize("weighted", [False, True])
def test_ovr_matches_sklearn(average, weighted):
    if average == "micro" and SKLEARN < (1, 2):
        pytest.skip("scikit-learn < 1.2 has no micro-averaged multiclass ROC-AUC")
    y, P, w = probs()
    w = w if weighted else None
    got = fastauc.roc_auc_multiclass(y, P, multi_class="ovr", average=average, sample_weight=w)
    want = roc_auc_score(y, P, multi_class="ovr", average=average, sample_weight=w)
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_ovo_matches_sklearn(average):
    y, P, _ = probs(seed=1)
    got = fastauc.roc_auc_multiclass(y, P, multi_class="ovo", average=average)
    assert got == pytest.approx(roc_auc_score(y, P, multi_class="ovo", average=average), rel=1e-12)


def test_string_labels_and_label_order():
    y, P, _ = probs(seed=2, C=3)
    names = np.array(["cat", "dog", "emu"])[y]
    a = fastauc.roc_auc_multiclass(names, P, average=None)
    np.testing.assert_allclose(a, roc_auc_score(names, P, multi_class="ovr", average=None))
    # columns follow `labels`, in any order
    perm = [2, 0, 1]
    b = fastauc.roc_auc_multiclass(names, P[:, perm], labels=["emu", "cat", "dog"], average=None)
    np.testing.assert_allclose(b, a[perm], rtol=1e-12)


def test_scores_need_not_be_probabilities_and_absent_class_is_nan():
    y, P, _ = probs(seed=3)
    scaled = P * 8.0  # exact in floating point: same order and ties, rows no longer sum to 1
    np.testing.assert_allclose(fastauc.roc_auc_multiclass(y, scaled, average=None),
                               fastauc.roc_auc_multiclass(y, P, average=None), rtol=1e-12)
    out = fastauc.roc_auc_multiclass(y, np.c_[P, np.zeros(len(y))], labels=[0, 1, 2, 3, 9],
                                     average=None)
    assert math.isnan(out[-1]) and not np.isnan(out[:-1]).any()


def test_multiclass_errors():
    y, P, w = probs(seed=4)
    with pytest.raises(ValueError, match="columns"):
        fastauc.roc_auc_multiclass(y, P[:, :3])
    with pytest.raises(ValueError, match="sample_weight"):
        fastauc.roc_auc_multiclass(y, P, multi_class="ovo", sample_weight=w)
    with pytest.raises(ValueError, match="average"):
        fastauc.roc_auc_multiclass(y, P, multi_class="ovo", average="micro")
    with pytest.raises(ValueError, match="not in labels"):
        fastauc.roc_auc_multiclass(y, P, labels=[0, 1, 2, 5])


@pytest.mark.parametrize("max_fpr", [0.001, 0.05, 0.1, 0.33, 0.5, 1.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_partial_auc_matches_sklearn(max_fpr, weighted):
    rng = np.random.default_rng(5)
    y = (rng.random(4000) < 0.3).astype(int)
    s = np.round(rng.random(4000) + 0.4 * y, 2)
    w = rng.random(4000) if weighted else None
    got = fastauc.partial_roc_auc(y, s, max_fpr, sample_weight=w)
    assert got == pytest.approx(roc_auc_score(y, s, max_fpr=max_fpr, sample_weight=w), rel=1e-12)


def test_partial_auc_raw_area_and_edges():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert fastauc.partial_roc_auc(y, s, 1.0, standardized=False) == pytest.approx(fastauc.roc_auc(y, s))
    assert fastauc.partial_roc_auc(y, s, 1.0) == pytest.approx(fastauc.roc_auc(y, s))
    assert fastauc.partial_roc_auc([0, 1], [0.1, 0.9], 0.2) == 1.0  # perfect
    assert fastauc.partial_roc_auc([0, 1, 0, 1], [0.5] * 4, 0.3) == pytest.approx(0.5)  # chance
    assert math.isnan(fastauc.partial_roc_auc([1, 1], [0.1, 0.2], 0.5))
    with pytest.raises(ValueError):
        fastauc.partial_roc_auc(y, s, 0.0)
