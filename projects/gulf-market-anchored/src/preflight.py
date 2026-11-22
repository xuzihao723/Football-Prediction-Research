from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, CONFIG_DIR, DATA_DIR, assert_definitive_allowed, protocol
from .pooled_split import pooled_outer_split, legal_time, match_time, forecast_boundary

XG_REMOVED = {
    "home_xg_diff_5_proxy", "away_xg_diff_5_proxy", "xg_diff_mom_5_proxy_diff",
    "home_venue_xg_diff_5_proxy", "away_venue_xg_diff_5_proxy", "venue_xg_diff_5_proxy_diff",
}
EXPECTED_LEAGUE_COUNTS = {"Qatar": 1420, "Saudi": 2358, "UAE": 1721}
EXPECTED_CUTOFF_KEYS = {
    "Saudi|2019-2020|abha|alahli|2020-09-04",
    "Saudi|2019-2020|alfateh|altaawon|2020-09-04",
    "Saudi|2019-2020|alfeiha|aladalh|2020-09-04",
    "Saudi|2019-2020|alittifaq|dhamk|2020-09-04",
    "Saudi|2019-2020|alittihad|alnassr|2020-09-04",
    "Saudi|2019-2020|alraed|alfaisaly|2020-09-04",
    "Saudi|2019-2020|alshabab|alhazm|2020-09-04",
    "Saudi|2019-2020|alwahda|alhilal|2020-09-04",
    "Saudi|2019-2020|aladalh|alittihad|2020-09-09",
    "Saudi|2019-2020|alahli|alraed|2020-09-09",
    "Saudi|2019-2020|alfaisaly|abha|2020-09-09",
    "Saudi|2019-2020|alhazm|alwahda|2020-09-09",
    "Saudi|2019-2020|alhilal|alshabab|2020-09-09",
    "Saudi|2019-2020|alnassr|alittifaq|2020-09-09",
    "Saudi|2019-2020|altaawon|alfeiha|2020-09-09",
    "Saudi|2019-2020|dhamk|alfateh|2020-09-09",
}


def run_preflight() -> bool:
    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, detail: str = "") -> bool:
        checks.append((name, bool(ok), detail))
        return bool(ok)

    regp = CONFIG_DIR / "feature_registry_final.json"
    datap = DATA_DIR / "modeling" / "ModelingData_harmonized.csv"
    canonp = DATA_DIR / "canonical" / "CleanCanonicalPlayedFixtures.csv"
    metap = DATA_DIR / "canonical" / "canonical_metadata.json"
    selp = CONFIG_DIR / "sk_selection_frozen.json"
    hpop = CONFIG_DIR / "hpo_final_models.json"

    chk("authoritative feature registry", regp.exists(), str(regp))
    chk("harmonized modeling data", datap.exists(), str(datap))
    chk("canonical played data", canonp.exists(), str(canonp))
    chk("canonical metadata", metap.exists(), str(metap))
    chk("frozen fold selection", selp.exists(), str(selp))
    chk("final-model HPO config", hpop.exists(), str(hpop))

    registry = json.loads(regp.read_text(encoding="utf-8")) if regp.exists() else None
    model = pd.read_csv(datap, low_memory=False) if datap.exists() else None
    canon = pd.read_csv(canonp, low_memory=False) if canonp.exists() else None

    if registry is not None:
        hist = [c for g in registry["order"] for c in registry["groups"][g]]
        chk("7 groups", registry["order"] == ["Elo", "Form", "Standings", "Venue", "H2H", "Rest", "Draw"], str(registry["order"]))
        chk("92 historical features", len(hist) == 92 and len(set(hist)) == 92, f"n={len(hist)} unique={len(set(hist))}")
        chk("xG removed", not (set(hist) & XG_REMOVED), str(set(hist) & XG_REMOVED))
        meta = registry.get("features", {})
        chk("feature-level registry metadata", all(c in meta for c in hist + registry.get("league_context_columns", [])), f"metadata_n={len(meta)}")
        for c in registry.get("league_context_columns", []):
            m = meta.get(c, {})
            chk(f"league context fixed: {c}", m.get("selectable") is False and m.get("ablatable") is False and m.get("fixed_context") is True, str(m))

    if model is not None:
        chk("modeling N=5499", len(model) == 5499, f"N={len(model)}")
        got = model.groupby("league").size().to_dict()
        chk("modeling league counts", got == EXPECTED_LEAGUE_COUNTS, str(got))
        chk("fixture_key exists", "fixture_key" in model.columns)
        chk("fixture_key globally unique", "fixture_key" in model.columns and model["fixture_key"].is_unique)
        chk("target classes exactly A,D,H", set(model["target_match_result"].dropna().astype(str)) == {"A", "D", "H"})
        chk("no missing match_time", match_time(model).notna().all())
        chk("date_only count=115", int(model["timestamp_precision"].astype(str).eq("date_only").sum()) == 115)
        if registry is not None:
            hist = [c for g in registry["order"] for c in registry["groups"][g]]
            required = hist + registry.get("league_context_columns", []) + registry.get("market_prior_columns", []) + registry.get("ht_score_columns", []) + registry.get("ht_non_score_columns", [])
            miss = [c for c in required if c not in model.columns]
            chk("all registry runtime columns exist", not miss, str(miss))
            if not miss:
                arr = model[hist].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                chk("historical features finite", bool(np.isfinite(arr).all()), f"nonfinite={int((~np.isfinite(arr)).sum())}")

        # All formal outer folds must satisfy the actual-date rule.
        outer_ok = True
        details = []
        for fold in range(5):
            try:
                tr, te = pooled_outer_split(model, fold)
                good = legal_time(tr).max() < match_time(te).min()
                outer_ok &= bool(good)
                details.append(f"f{fold}:{len(tr)}/{len(te)}")
            except Exception as exc:
                outer_ok = False
                details.append(f"f{fold}:ERROR:{exc}")
        chk("five outer folds actual-date legal", outer_ok, ", ".join(details))

        cfg0 = protocol()["outer_folds"][0]
        te0 = model[model["season"].astype(str).isin(cfg0["outer_test"])]
        nom0 = model[model["season"].astype(str).isin(cfg0["outer_train"])]
        bound = forecast_boundary(te0)
        removed = set(nom0.loc[~(legal_time(nom0) < bound), "fixture_key"].astype(str))
        chk("exact 16 COVID cutoff fixture_keys", removed == EXPECTED_CUTOFF_KEYS, f"n={len(removed)} diff={sorted(removed ^ EXPECTED_CUTOFF_KEYS)[:4]}")

    if canon is not None:
        chk("canonical N=5499", len(canon) == 5499, f"N={len(canon)}")
        got = canon.groupby("league").size().to_dict()
        chk("canonical league counts", got == EXPECTED_LEAGUE_COUNTS, str(got))
        chk("canonical fixture_key unique", canon["fixture_key"].is_unique)
        chk("recovered=115", int((canon["source"].astype(str) == "RSSSF").sum()) == 115)
        chk("UAE canceled 49 excluded", len(canon[(canon["league"] == "UAE") & (canon["season"].astype(str) == "2019-2020")]) == 133)
        chk("canonical date_only=115", int(canon["timestamp_precision"].astype(str).eq("date_only").sum()) == 115)

    if selp.exists():
        sel = json.loads(selp.read_text(encoding="utf-8"))
        by = sel.get("selected_groups_by_fold", {})
        valid = set(by) == {"0", "1", "2", "3", "4"} and all(isinstance(by[k], list) and len(by[k]) > 0 for k in by)
        chk("S_k selection covers folds 0-4", valid, str(by))

    if hpop.exists():
        hc = json.loads(hpop.read_text(encoding="utf-8"))
        required_sections = {"Historical-LR", "Concat-LR", "Residual-LR", "XGB", "RF", "XGBClassifier", "LightGBM", "CatBoost", "FNN"}
        chk("HPO sections complete", required_sections <= set(hc), str(sorted(required_sections - set(hc))))
        chk("XGB early stopping configured", int(hc.get("XGB", {}).get("early_stopping_rounds", 0)) > 0)

    for mod in ["xgboost", "lightgbm", "catboost", "sklearn", "scipy", "pandas", "numpy"]:
        chk(f"dependency {mod}", importlib.util.find_spec(mod) is not None)

    # Static production-code audit. Migration utility may read legacy source configs;
    # runtime code may not use pooled match_id joins or local absolute paths.
    bad_abs, bad_markers, bad_match, bad_old_registry = [], [], [], []
    for p in (PROJECT_ROOT / "src").rglob("*.py"):
        s = p.read_text(encoding="utf-8", errors="ignore")
        rel = str(p.relative_to(PROJECT_ROOT))
        if re.search(r"[A-Za-z]:\\", s):
            bad_abs.append(rel)
        if p.name != "preflight.py" and re.search(r"\b(TODO|FIXME|NotImplemented)\b", s):
            bad_markers.append(rel)
        if p.name != "preflight.py" and "match_id" in s:
            bad_match.append(rel)
        if p.name not in {"build_feature_registry.py", "preflight.py"} and any(x in s for x in ["feature_groups_harmonized.json", "feature_availability.json", "odds_features.json"]):
            bad_old_registry.append(rel)
    chk("no absolute Windows paths", not bad_abs, str(bad_abs))
    chk("no TODO/FIXME/NotImplemented in production", not bad_markers, str(bad_markers))
    chk("no production match_id references", not bad_match, str(bad_match))
    chk("runtime uses only final feature registry", not bad_old_registry, str(bad_old_registry))

    # Safety-lock behavior: both confirmation and env variable are required.
    env_name = protocol().get("definitive_lock_env", "ALLOW_DEFINITIVE_OUTER_TEST")
    old = os.environ.pop(env_name, None)
    locked = False
    try:
        try:
            assert_definitive_allowed(False)
        except RuntimeError:
            locked = True
    finally:
        if old is not None:
            os.environ[env_name] = old
    chk("definitive safety lock blocks by default", locked)

    ok = all(x[1] for x in checks)
    for name, value, detail in checks:
        print(f"{'PASS' if value else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    print("READY_FOR_USER_AUDIT" if ok else "NOT_READY")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_preflight() else 2)
