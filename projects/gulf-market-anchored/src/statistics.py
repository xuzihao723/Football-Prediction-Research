"""implementation note（§5.4）：paired cluster（league-season）bootstrap。

- paired implementation note：Residual−Market、Residual−Concat、+HT score−pre-match implementation note。
- cluster implementation note = league-season，implementation note cross-league implementation note CI。
- CI implementation note 0 implementation note comparable，implementation note significantly better。
"""
from __future__ import annotations

import numpy as np


def per_obs_metric(y: np.ndarray, p: np.ndarray, name: str) -> np.ndarray:
    """implementation note（implementation note headline implementation note）。

    acc: 1[argmax==y]; log_loss: -log p_y; brier: Σ_c (p_c-1[y=c])²;
    macro_f1 implementation note，implementation note None（implementation note bootstrap）。
    """
    from . import metrics
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if name == "acc":
        return (np.argmax(p, axis=1) == metrics.encode_target(y)).astype(float)
    if name == "log_loss":
        y_int = metrics.encode_target(y)
        pc = np.clip(p, metrics.EPS, 1.0 - metrics.EPS)
        return -np.log(pc[np.arange(len(y_int)), y_int])
    if name == "brier":
        y_int = metrics.encode_target(y)
        oh = metrics.one_hot(y_int, p.shape[1])
        return np.sum((p - oh) ** 2, axis=1)
    raise ValueError(f"implementation note {name} implementation note（macro_f1 implementation note bootstrap）")


def paired_cluster_bootstrap(obs_a: np.ndarray, obs_b: np.ndarray,
                             clusters: np.ndarray, n_boot: int = 10000,
                             seed: int = 42, alpha: float = 0.05) -> dict:
    """obs_a/obs_b implementation note fixture implementation note。

    implementation note cluster（league-season）implementation note，implementation note mean(obs_a-b) implementation note CI。
    implementation note {mean_diff, ci_low, ci_high, n_clusters}。
    """
    obs_a = np.asarray(obs_a, dtype=float)
    obs_b = np.asarray(obs_b, dtype=float)
    clusters = np.asarray(clusters)
    assert len(obs_a) == len(obs_b) == len(clusters), "obs/cluster implementation note"

    diff = obs_a - obs_b
    clist = np.unique(clusters)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample_clusters = rng.choice(clist, size=len(clist), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in sample_clusters])
        if len(idx) == 0:
            boot[b] = 0.0
        else:
            boot[b] = np.mean(diff[idx])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return {
        "mean_diff": float(np.mean(diff)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_clusters": int(len(clist)),
    }


def cluster_bootstrap_ci(obs: np.ndarray, clusters: np.ndarray,
                         n_boot: int = 10000, seed: int = 42,
                         alpha: float = 0.05) -> dict:
    """implementation note cluster bootstrap CI（§5.4 implementation note cross-league implementation note，implementation note feature-group ΔAcc）。

    cluster implementation note = league-season（15 clusters）；implementation note cluster implementation note，
    implementation note CI。implementation note {mean, ci_low, ci_high, n_clusters}。
    """
    obs = np.asarray(obs, dtype=float)
    clusters = np.asarray(clusters)
    assert len(obs) == len(clusters), "obs/cluster implementation note"

    clist = np.unique(clusters)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample_clusters = rng.choice(clist, size=len(clist), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in sample_clusters])
        boot[b] = np.mean(obs[idx]) if len(idx) > 0 else 0.0
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return {
        "mean": float(np.mean(obs)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_clusters": int(len(clist)),
    }