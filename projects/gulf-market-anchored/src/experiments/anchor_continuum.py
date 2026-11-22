from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import metrics
from ..config import OUTPUT_DIR
from ..feature_registry import get_registry
from ..hpo import is_smoke
from ..models.anchoring import CanonicalAnchorLR
from ..models.market import bookmaker_prior
from ..pooled_split import pooled_inner_temporal_split
from ..preprocessing import Preprocessor
from ..outputs import prediction_frame
from .common import load_data, history_cols, odds_subset, split_fold


VARIANTS = ["fixed", "free", "soft"]
LAMBDA_H_GRID = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
LAMBDA_A_GRID = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def _fit_block(kind, train, valid, cols, lh, la):
    pp = Preprocessor(scale=True)
    Xtr = pp.fit_transform(train[cols])
    Xva = pp.transform(valid[cols])
    _, ztr, _ = bookmaker_prior(train)
    _, zva, _ = bookmaker_prior(valid)
    model = CanonicalAnchorLR(kind, lambda_h=lh, lambda_a=la).fit(
        Xtr, ztr, train.target_match_result.to_numpy())
    p = model.predict_proba(Xva, zva)
    return metrics.log_loss(valid.target_match_result.to_numpy(), p)


def tune_variant(train_df, cols, variant):
    work = train_df.copy()
    _, _, mask = bookmaker_prior(work)
    work = work.loc[mask].reset_index(drop=True)
    splits = pooled_inner_temporal_split(work, min_train_seasons=2)
    if not splits:
        raise RuntimeError(f"{variant}: no legal inner temporal HPO split")
    la_grid = [0.0] if variant in {"fixed", "free"} else LAMBDA_A_GRID
    candidates = [(lh, la) for lh in LAMBDA_H_GRID for la in la_grid]
    rows = []
    for lh, la in candidates:
        scores, weights = [], []
        for sp in splits:
            tr = work.iloc[sp["train_idx"]].reset_index(drop=True)
            va = work.iloc[sp["val_idx"]].reset_index(drop=True)
            score = _fit_block(variant, tr, va, cols, lh, la)
            scores.append(score); weights.append(len(va))
        rows.append({
            "lambda_h": float(lh), "lambda_a": float(la),
            "weighted_log_loss": float(np.average(scores, weights=weights)),
            "split_scores": [float(x) for x in scores],
            "split_weights": [int(x) for x in weights],
            "n_splits": len(splits),
        })
    best = min(rows, key=lambda x: x["weighted_log_loss"])
    return {"lambda_h": best["lambda_h"], "lambda_a": best["lambda_a"],
            "score": best["weighted_log_loss"], "n_splits": best["n_splits"],
            "candidates": rows}


def _fit_outer(train_df, cols, variant, tune):
    _, _, mask = bookmaker_prior(train_df)
    fitdf = train_df.loc[mask].reset_index(drop=True)
    pp = Preprocessor(scale=True)
    X = pp.fit_transform(fitdf[cols])
    _, z, _ = bookmaker_prior(fitdf)
    model = CanonicalAnchorLR(variant, tune["lambda_h"], tune["lambda_a"]).fit(
        X, z, fitdf.target_match_result.to_numpy())
    return pp, model


def run(data=None, folds=range(5)):
    df = load_data() if data is None else data
    all_rows, diagnostics = [], []
    for fold in folds:
        tr, te = split_fold(df, int(fold))
        cols = history_cols(int(fold), "selected")
        teo, _, _ = odds_subset(te)
        base_keys = teo.fixture_key.astype(str).tolist()
        for variant in VARIANTS:
            tune = tune_variant(tr, cols, variant)
            pp, model = _fit_outer(tr, cols, variant, tune)
            Xte = pp.transform(teo[cols])
            _, zte, valid = bookmaker_prior(teo)
            if not valid.all():
                raise AssertionError("outer B test block unexpectedly contains invalid odds")
            p = model.predict_proba(Xte, zte)
            frame = prediction_frame(teo, p, int(fold),
                                     {"fixed": "Canonical-Fixed-LR",
                                      "free": "FreeMarket-LR",
                                      "soft": "SoftAnchor-LR"}[variant],
                                     "selected")
            if frame.fixture_key.astype(str).tolist() != base_keys:
                raise AssertionError(f"anchor key order failure fold={fold}")
            all_rows.append(frame)
            diagnostics.append({
                "fold": int(fold), "method": frame.method.iloc[0],
                "features": cols, "tuning": tune,
                "fit": model.diagnostics(),
                "preprocessing": {"historical_scale": True,
                                   "market_logits_scaled": False},
                "test_n": len(frame),
            })
    return pd.concat(all_rows, ignore_index=True), diagnostics


def _metric_rows(df):
    rows = []
    for m, g in df.groupby("method"):
        p = g[["p_A", "p_D", "p_H"]].to_numpy(float)
        rows.append({"method": m, "N": int(len(g)),
                     **metrics.compute_all_metrics(g.actual_class.to_numpy(), p)})
    return rows


def _per_obs(y, p, name):
    yi = metrics.encode_target(y)
    if name == "acc": return (np.argmax(p, axis=1) == yi).astype(float)
    if name == "log_loss": return -np.log(np.clip(p[np.arange(len(yi)), yi], 1e-12, 1.0))
    if name == "brier":
        oh = np.zeros_like(p); oh[np.arange(len(yi)), yi] = 1.0
        return np.sum((p - oh) ** 2, axis=1)
    raise ValueError(name)


def _bootstrap(diff, clusters, n_boot=10000, seed=42):
    clist = np.unique(clusters); by = {c: np.where(clusters == c)[0] for c in clist}
    rng = np.random.default_rng(seed); vals = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(clist, len(clist), replace=True)
        idx = np.concatenate([by[c] for c in sampled])
        vals[i] = float(np.mean(diff[idx]))
    return {"mean_diff": float(np.mean(diff)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "n_clusters": int(len(clist))}


def paired_contrasts(new_df, legacy_df):
    legacy = legacy_df[legacy_df.representation == "selected"].copy()
    out = []
    for new_method in new_df.method.unique():
        x = new_df[new_df.method == new_method].sort_values("fixture_key").reset_index(drop=True)
        for ref in ["Bookmaker", "Recalibrated-Market", "Concat-LR", "Residual-LR"]:
            y = legacy[legacy.method == ref].sort_values("fixture_key").reset_index(drop=True)
            if x.fixture_key.astype(str).tolist() != y.fixture_key.astype(str).tolist():
                raise AssertionError(f"unmatched fixtures {new_method} vs {ref}")
            clusters = (x.league.astype(str) + "|" + x.season.astype(str)).to_numpy()
            px = x[["p_A", "p_D", "p_H"]].to_numpy(float)
            py = y[["p_A", "p_D", "p_H"]].to_numpy(float)
            row = {"new": new_method, "reference": ref, "N": int(len(x)),
                   "key_alignment": True, "class_order": ["A", "D", "H"]}
            for name in ["acc", "log_loss", "brier"]:
                row[name] = _bootstrap(_per_obs(x.actual_class, px, name) -
                                       _per_obs(y.actual_class, py, name), clusters)
            out.append(row)
    return out


def main():
    outdir = OUTPUT_DIR / "anchor_continuum"
    outdir.mkdir(parents=True, exist_ok=True)
    pred, diag = run()
    legacy = pd.read_csv(OUTPUT_DIR / "raw" / "experiment_b_oos.csv")
    selected = legacy[legacy.representation == "selected"]
    # Exact fold/method key check against frozen Experiment B selected rows.
    for fold, g in pred.groupby("fold"):
        expected = selected[selected.fold == fold].query("method == 'Bookmaker'").fixture_key.astype(str).tolist()
        got = g[g.method == "Canonical-Fixed-LR"].fixture_key.astype(str).tolist()
        if got != expected:
            raise AssertionError(f"frozen B key mismatch fold={fold}")
    pred.to_csv(outdir / "canonical_prematch_oos.csv", index=False)
    (outdir / "canonical_prematch_diagnostics.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"new_metrics": _metric_rows(pred),
              "paired_bootstrap": paired_contrasts(pred, legacy),
              "legacy_source": "results/raw/experiment_b_oos.csv",
              "n_boot": 10000, "seed": 42,
              "methods": ["Canonical-Fixed-LR", "FreeMarket-LR", "SoftAnchor-LR"],
              "representation": "selected"}
    (outdir / "canonical_prematch_paired_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": len(pred), "metrics": report["new_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
