"""Metrics + implementation note（§5.1 / §5.2）。

implementation note（implementation note）：
- Accuracy（implementation note）：argmax(p̂) == y implementation note
- Log Loss：-(1/N) Σ log p̂_{i,y_i}
- Brier（multiclass，implementation note K implementation note）：(1/N) Σ_i Σ_c (p̂_{i,c} - 1[y_i=c])²
- Macro-F1：implementation note F1 implementation note

implementation note（implementation note）：
1. p̂.sum(axis=1) ≈ 1、p̂ ∈ [0,1]、finite
2. class implementation note class_order() implementation note（A=0/D=1/H=2）
3. clip implementation note，implementation note
"""
from __future__ import annotations

import numpy as np

from .config import protocol as _protocol

EPS = 1e-12


# ------------------------------------------------------------------ #
# implementation note（A/D/H -> 0/1/2），implementation note frozen_protocol.json
# ------------------------------------------------------------------ #
def class_order() -> list[str]:
    return list(_protocol()["class_order"])


def encode_target(y: np.ndarray) -> np.ndarray:
    """implementation note -> implementation note（implementation note class_order）。"""
    labels = class_order()
    mp = {lab: i for i, lab in enumerate(labels)}
    # implementation note（implementation note）implementation note
    out = np.empty(len(y), dtype=int)
    for i, v in enumerate(y):
        if isinstance(v, (int, np.integer)):
            if not (0 <= int(v) < len(labels)):
                raise ValueError(f"implementation note {v!r} implementation note，implementation note 0..{len(labels)-1}")
            out[i] = int(v)
        else:
            if v not in mp:
                raise ValueError(f"implementation note {v!r}，implementation note {labels}")
            out[i] = mp[v]
    return out


def one_hot(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """implementation note -> one-hot (N, K)。"""
    y = np.asarray(y)
    oh = np.zeros((len(y), n_classes), dtype=float)
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh


# ------------------------------------------------------------------ #
# implementation note
# ------------------------------------------------------------------ #
def assert_valid_proba(p: np.ndarray, tol: float = 1e-8, context: str = ""):
    """implementation note：sum≈1、∈[0,1]、finite。implementation note。"""
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise AssertionError(f"{context}: p implementation note (N,K) implementation note，implementation note {p.shape}")
    if p.shape[1] != 3:
        raise AssertionError(f"{context}: p implementation note 3 implementation note（[A,D,H]），implementation note {p.shape}")
    if not np.all(np.isfinite(p)):
        raise AssertionError(f"{context}: implementation note NaN/Inf")
    if np.any(p < -tol) or np.any(p > 1 + tol):
        raise AssertionError(f"{context}: implementation note [{float(p.min()):.3g}, {float(p.max()):.3g}]")
    sums = p.sum(axis=1)
    if np.max(np.abs(sums - 1.0)) > 1e-6:
        raise AssertionError(f"{context}: implementation note≠1，implementation note {float(np.max(np.abs(sums - 1.0))):.3g}")


def clip_renormalize(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """implementation note：clip implementation note [eps,1-eps] implementation note（implementation note，implementation note）。"""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return p / p.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ #
# implementation note
# ------------------------------------------------------------------ #
def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y_int = encode_target(y)
    return float(np.mean(np.argmax(p, axis=1) == y_int))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = EPS) -> float:
    p = np.asarray(p, dtype=float)
    y_int = encode_target(y)
    p_clip = np.clip(p, eps, 1.0 - eps)
    nll = -np.log(p_clip[np.arange(len(y_int)), y_int])
    return float(np.mean(nll))


def brier_multiclass(y: np.ndarray, p: np.ndarray) -> float:
    """implementation note K implementation note： sum_c (p̂_{i,c} - 1[y_i=c])² implementation note。"""
    p = np.asarray(p, dtype=float)
    y_int = encode_target(y)
    return float(np.mean(np.sum((p - one_hot(y_int, p.shape[1])) ** 2, axis=1)))


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y_int = encode_target(y)
    pred = np.argmax(p, axis=1)
    k = p.shape[1]
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (y_int == c))
        fp = np.sum((pred == c) & (y_int != c))
        fn = np.sum((pred != c) & (y_int == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def compute_all_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(y) != p.shape[0]:
        raise AssertionError(f"metrics: len(y)={len(y)} implementation note p implementation note {p.shape[0]} implementation note")
    assert_valid_proba(p, context="metrics")
    return {
        "acc": accuracy(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_multiclass(y, p),
        "macro_f1": macro_f1(y, p),
    }