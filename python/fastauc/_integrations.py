"""Shared logic for the LightGBM and XGBoost evaluation helpers."""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import binary_metrics, ranking_metrics

_RANKING = ("ndcg", "mrr", "map", "recall", "precision", "hit_rate")
_RANKING_NO_K = ("gauc", "gauc_macro")
_NAME = re.compile(r"^([a-z_]+?)(?:@(\d+))?$")

# Binary metrics usable for model selection, with their direction.
_BINARY: Dict[str, bool] = {
    "roc_auc": True,
    "average_precision": True,
    "pr_auc": True,
    "ks": True,
    "f1_max": True,
    "log_loss": False,
    "normalized_entropy": False,
    "brier": False,
    "ece": False,
}


def parse_ranking(metrics: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[Optional[int], ...]]:
    """Validate names like 'ndcg@10', 'mrr', 'gauc'; return (names, cutoffs)."""
    names = tuple(metrics) or ("ndcg@10",)
    ks: List[Optional[int]] = []
    for name in names:
        if name in _RANKING_NO_K:
            continue
        m = _NAME.match(name)
        if not m or m.group(1) not in _RANKING:
            raise ValueError(
                f"unknown ranking metric {name!r}; use one of {', '.join(_RANKING)} "
                "with an optional '@k' (e.g. 'ndcg@10'), or 'gauc' / 'gauc_macro'"
            )
        k = None if m.group(2) is None else int(m.group(2))
        if k is not None and k < 1:
            raise ValueError(f"cutoff must be at least 1 in {name!r}")
        if k not in ks:
            ks.append(k)
    return names, tuple(ks) if ks else (None,)


def parse_binary(metrics: Sequence[str]) -> Tuple[str, ...]:
    names = tuple(metrics) or ("roc_auc",)
    for name in names:
        if name not in _BINARY:
            raise ValueError(f"unknown binary metric {name!r}; use one of {', '.join(_BINARY)}")
    return names


def higher_is_better(name: str) -> bool:
    return _BINARY.get(name, True)


def weights_or_none(w) -> Optional[np.ndarray]:
    if w is None:
        return None
    w = np.asarray(w, dtype=np.float64)
    return w if w.size else None


def ranking_values(names, ks, labels, preds, sizes, weights, gain, no_relevant) -> List[Tuple[str, float]]:
    if sizes is None or len(sizes) == 0:
        raise ValueError(
            "the evaluation data has no query groups: pass group= (LightGBM) or "
            "qid= / set_group (XGBoost) when building it"
        )
    m = ranking_metrics(labels, preds, group_sizes=np.asarray(sizes, dtype=np.int64), ks=ks,
                        gain=gain, no_relevant=no_relevant, sample_weight=weights)
    return [(name, float(m[name])) for name in names]


def binary_values(names, labels, preds, weights) -> List[Tuple[str, float]]:
    p = np.asarray(preds, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"binary metrics need 1-D predictions, got shape {p.shape}")
    # Compute the calibration block only when asked for (it needs probabilities).
    m = binary_metrics(labels, p, sample_weight=weights,
                       calibration=any(not _BINARY[n] for n in names))
    return [(name, float(m[name])) for name in names]


def describe(fn: Callable, kind: str, names: Sequence[str]) -> Callable:
    fn.__name__ = f"fastauc_{kind}"
    fn.__qualname__ = fn.__name__
    fn.__doc__ = f"fastauc {kind} metrics: {', '.join(names)}"
    return fn
