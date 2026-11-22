from __future__ import annotations
import argparse
import importlib
import numpy as np
import pandas as pd

from .config import assert_definitive_allowed
from . import hpo
from .outputs import RAW_COLUMNS, write_raw, write_json

EXPERIMENTS = ["A", "B", "C", "D", "E"]


def _smoke_data() -> pd.DataFrame:
    """Load real data and keep a deterministic development-only pseudo-test slice.

    Smoke never uses definitive S6-S10 fixtures.  Formal experiment code receives the
    S1-S4 -> S5 split through experiments.common.split_fold while smoke mode is active.
    Only the first 3 S5 fixtures are retained to keep prequential structural refits fast.
    """
    from .experiments.common import load_data

    df = load_data()
    test_season = "2019-2020"  # S5 development-only pseudo test
    allowed_train = {"2015-2016", "2016-2017", "2017-2018", "2018-2019"}
    te = df[df["season"].astype(str) == test_season].sort_values("match_datetime")
    keep_test = set(te.head(3).index.tolist())
    keep = df["season"].astype(str).isin(allowed_train).to_numpy().copy()
    keep |= df.index.to_series().isin(keep_test).to_numpy()
    out = df.loc[keep].reset_index(drop=True)
    if set(out["season"].astype(str)) - (allowed_train | {test_season}):
        raise AssertionError("smoke data contains definitive S6-S10 seasons")
    return out


def _verify_raw(raw: pd.DataFrame, exp: str) -> None:
    if raw is None or len(raw) == 0:
        raise AssertionError(f"SMOKE {exp}: empty raw prediction frame")
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise AssertionError(f"SMOKE {exp}: raw columns missing {missing}")
    p = raw[["p_A", "p_D", "p_H"]].to_numpy(float)
    if not np.all(np.isfinite(p)):
        raise AssertionError(f"SMOKE {exp}: non-finite probabilities")
    sums = p.sum(axis=1)
    if np.max(np.abs(sums - 1.0)) > 1e-6:
        raise AssertionError(f"SMOKE {exp}: probabilities not normalized")
    # fixture_key is unique within each (fold, method, representation), not globally.
    for _, g in raw.groupby(["fold", "method", "representation"], dropna=False):
        if g["fixture_key"].duplicated().any():
            raise AssertionError(f"SMOKE {exp}: duplicated fixture_key within method")


def _module(e: str):
    return importlib.import_module(f"src.experiments.experiment_{e.lower()}")


def _smoke(exp: str) -> None:
    mods = EXPERIMENTS if exp == "all" else [exp.upper()]
    df = _smoke_data()
    hpo.set_smoke(True)
    try:
        for e in mods:
            m = _module(e)
            raw, diag = m.run(data=df, folds=[0])
            _verify_raw(raw, e)
            methods = sorted(raw["method"].astype(str).unique())
            print(f"SMOKE {e}: {len(raw)} rows; methods={methods}")
    finally:
        hpo.set_smoke(False)
    print("SMOKE OK: full data load/split/selection/HPO/fit/predict/match verified; "
          "development-only S1-S4->S5 smoke; no S6-S10 fixtures used.")


def _definitive(exp: str) -> None:
    mods = EXPERIMENTS if exp == "all" else [exp.upper()]
    for e in mods:
        m = _module(e)
        raw, diag = m.run()
        write_raw(raw, f"experiment_{e.lower()}_oos")
        write_json(diag, f"experiment_{e.lower()}_diagnostics.json")
        print(f"Experiment {e} predictions written; metrics are intentionally not auto-printed.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "C", "D", "E", "all"], required=True)
    ap.add_argument("--mode", choices=["smoke", "definitive"], default="smoke")
    ap.add_argument("--confirm-definitive", action="store_true")
    a = ap.parse_args()
    if a.mode == "smoke":
        _smoke(a.exp)
    else:
        assert_definitive_allowed(a.confirm_definitive)
        _definitive(a.exp)


if __name__ == "__main__":
    main()