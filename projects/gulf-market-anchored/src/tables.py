from __future__ import annotations

import pandas as pd

from . import metrics
from .statistics import paired_cluster_bootstrap, per_obs_metric


def pooled_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, rep), g in raw.groupby(["method", "representation"], dropna=False):
        p = g[["p_A", "p_D", "p_H"]].to_numpy(float)
        rows.append({"method": method, "representation": rep, "N": len(g), **metrics.compute_all_metrics(g.actual_class.to_numpy(), p)})
    return pd.DataFrame(rows)


def paired_bootstrap(raw: pd.DataFrame, a: str, b: str, representation=None, n_boot=10000, seed=42) -> pd.DataFrame:
    """Paired cluster bootstrap for A-B, so negative LL/Brier means A is better."""
    x = raw[raw.method == a]
    y = raw[raw.method == b]
    if representation is not None:
        x = x[x.representation == representation]
        y = y[y.representation == representation]
    cols = ["fixture_key", "league", "season", "actual_class", "p_A", "p_D", "p_H"]
    x = x[cols].sort_values("fixture_key").reset_index(drop=True)
    y = y[cols].sort_values("fixture_key").reset_index(drop=True)
    if list(x.fixture_key) != list(y.fixture_key):
        raise AssertionError("paired bootstrap requires identical fixtures")
    out = []
    cl = (x.league.astype(str) + "|" + x.season.astype(str)).to_numpy()
    pa = x[["p_A", "p_D", "p_H"]].to_numpy(float)
    pb = y[["p_A", "p_D", "p_H"]].to_numpy(float)
    yy = x.actual_class.to_numpy()
    for name in ["acc", "log_loss", "brier"]:
        r = paired_cluster_bootstrap(per_obs_metric(yy, pa, name), per_obs_metric(yy, pb, name), cl, n_boot=n_boot, seed=seed)
        out.append({"metric": name, "A": a, "B": b, **r})
    return pd.DataFrame(out)


def build_table_a(raw: pd.DataFrame) -> pd.DataFrame:
    return pooled_metrics(raw).sort_values(["representation", "method"]).reset_index(drop=True)


def build_table_b(raw: pd.DataFrame, n_boot=10000, seed=42) -> dict[str, pd.DataFrame]:
    summary = pooled_metrics(raw).sort_values(["representation", "method"]).reset_index(drop=True)
    comps = []
    for rep in sorted(raw.representation.dropna().unique()):
        methods = set(raw.loc[raw.representation == rep, "method"])
        pairs = []
        for backend in ["LR", "XGB"]:
            r = f"Residual-{backend}"
            if r in methods:
                for ref in ["Bookmaker", "Recalibrated-Market", f"Concat-{backend}"]:
                    if ref in methods:
                        pairs.append((r, ref))
        for a, b in pairs:
            z = paired_bootstrap(raw, a, b, representation=rep, n_boot=n_boot, seed=seed)
            z.insert(0, "representation", rep)
            comps.append(z)
    return {"summary": summary, "paired": pd.concat(comps, ignore_index=True) if comps else pd.DataFrame()}


def build_table_c(raw: pd.DataFrame) -> pd.DataFrame:
    """Matched T_seq pooled metrics for all pre/HT architectures."""
    return pooled_metrics(raw).sort_values(["representation", "method"]).reset_index(drop=True)


def build_table_d(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = pooled_metrics(raw).sort_values(["method", "representation"]).reset_index(drop=True)
    rows = []
    for method, g in summary.groupby("method"):
        by = g.set_index("representation")
        if not {"Pre", "HT-F"} <= set(by.index):
            continue
        pre, ht = by.loc["Pre"], by.loc["HT-F"]
        rows.append({
            "method": method,
            "delta_accuracy_HTF_minus_Pre": float(ht["accuracy"] - pre["accuracy"]),
            "delta_log_loss_HTF_minus_Pre": float(ht["log_loss"] - pre["log_loss"]),
            "delta_brier_HTF_minus_Pre": float(ht["brier"] - pre["brier"]),
            "delta_macro_f1_HTF_minus_Pre": float(ht["macro_f1"] - pre["macro_f1"]),
        })
    return {"summary": summary, "delta": pd.DataFrame(rows)}


def build_table_e(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Full-vs-minus LOO and Residual-vs-Concat differential predictive utility."""
    summary = pooled_metrics(raw).sort_values(["method", "representation"]).reset_index(drop=True)
    loo = []
    for method, g in summary.groupby("method"):
        by = g.set_index("representation")
        if "Full" not in by.index:
            continue
        full = by.loc["Full"]
        for rep in by.index:
            if not str(rep).startswith("Minus-"):
                continue
            minus = by.loc[rep]
            group = str(rep)[len("Minus-"):]
            loo.append({
                "method": method, "group": group,
                # deletion effect: minus - full; positive LL/Brier means deletion hurts
                "delta_accuracy_minus_full": float(minus["accuracy"] - full["accuracy"]),
                "delta_log_loss_minus_full": float(minus["log_loss"] - full["log_loss"]),
                "delta_brier_minus_full": float(minus["brier"] - full["brier"]),
                "delta_macro_f1_minus_full": float(minus["macro_f1"] - full["macro_f1"]),
            })
    loo_df = pd.DataFrame(loo)
    diff = []
    if len(loo_df):
        for backend in ["LR", "XGB"]:
            rname, cname = f"Residual-{backend}", f"Concat-{backend}"
            r = loo_df[loo_df.method == rname].set_index("group")
            c = loo_df[loo_df.method == cname].set_index("group")
            for group in sorted(set(r.index) & set(c.index)):
                diff.append({
                    "backend": backend, "group": group,
                    "differential_delta_log_loss_Residual_minus_Concat": float(r.loc[group, "delta_log_loss_minus_full"] - c.loc[group, "delta_log_loss_minus_full"]),
                    "differential_delta_accuracy_Residual_minus_Concat": float(r.loc[group, "delta_accuracy_minus_full"] - c.loc[group, "delta_accuracy_minus_full"]),
                    "differential_delta_macro_f1_Residual_minus_Concat": float(r.loc[group, "delta_macro_f1_minus_full"] - c.loc[group, "delta_macro_f1_minus_full"]),
                })
    return {"summary": summary, "loo": loo_df, "differential": pd.DataFrame(diff)}
