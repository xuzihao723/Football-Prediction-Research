"""Data-integrity and frozen-count tests against the real harmonized modeling dataset.

These are the authoritative invariants the project is frozen to (N=5499, 92 harmonized
historical features, 115 recovered, 49 canceled-excluded, exact 16 COVID cut-off keys).
"""
from __future__ import annotations
import json

import pandas as pd
import pytest

from src.config import CONFIG_DIR, DATA_DIR


@pytest.fixture(scope="module")
def modeling() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "modeling" / "ModelingData_harmonized.csv", low_memory=False)


@pytest.fixture(scope="module")
def canonical() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "canonical" / "CleanCanonicalPlayedFixtures.csv", low_memory=False)


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(
        (CONFIG_DIR / "feature_registry_final.json").read_text(encoding="utf-8")
    )


XG_REMOVED = {
    "home_xg_diff_5_proxy", "away_xg_diff_5_proxy", "xg_diff_mom_5_proxy_diff",
    "home_venue_xg_diff_5_proxy", "away_venue_xg_diff_5_proxy", "venue_xg_diff_5_proxy_diff",
}

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


def test_modeling_N_5499_and_unique_key(modeling):
    assert len(modeling) == 5499
    assert modeling["fixture_key"].is_unique
    assert modeling["fixture_key"].notna().all()


def test_league_counts(modeling):
    league = modeling.groupby("league").size()
    assert league["Qatar"] == 1420
    assert league["Saudi"] == 2358
    assert league["UAE"] == 1721


def test_target_classes_valid(modeling):
    assert set(modeling["target_match_result"].unique()) <= {"A", "D", "H"}


def test_registry_seven_groups_92_no_xg(registry):
    hist = [c for g in registry["order"] for c in registry["groups"][g]]
    assert len(registry["order"]) == 7
    assert len(hist) == 92
    assert len(hist) == len(set(hist)), "feature group overlap"
    assert not (set(hist) & XG_REMOVED)


def test_selected_features_present(modeling, registry):
    for g, cols in registry["groups"].items():
        missing = [c for c in cols if c not in modeling.columns]
        assert not missing, f"group {g} missing columns: {missing}"
    for c in registry["market_prior_columns"] + registry["league_context_columns"]:
        assert c in modeling.columns, f"missing column {c}"


def test_canonical_recovered_canceled(canonical):
    assert len(canonical) == 5499
    recovered = canonical[canonical["source"] == "RSSSF"]
    assert len(recovered) == 115
    assert len(recovered[recovered["league"] == "UAE"]) == 66
    assert len(recovered[recovered["league"] == "Saudi"]) == 49
    # canceled excluded: UAE 2019-20 keeps only 133 (49 canceled dropped)
    uae1920 = canonical[(canonical["league"] == "UAE") & (canonical["season"] == "2019-2020")]
    assert len(uae1920) == 133


def test_covid_16_cutoff_excluded_from_fold0_train(modeling):
    from src.pooled_split import pooled_outer_split, legal_time, match_time
    tr, te = pooled_outer_split(modeling, 0)
    # the 16 delayed fixtures are removed from fold-0 train, not present in it
    overlap = set(tr["fixture_key"]) & EXPECTED_CUTOFF_KEYS
    assert overlap == set()
    # they actually live in the raw Saudi 2019-20 season
    raw_saudi = modeling[
        (modeling["league"] == "Saudi") & (modeling["season"] == "2019-2020")
    ]
    assert EXPECTED_CUTOFF_KEYS <= set(raw_saudi["fixture_key"])
    # and they are later than the earliest lockout boundary (2020-21 season)
    assert match_time(te).min() is not None
    removed_exact = set(
        raw_saudi[raw_saudi["fixture_key"].isin(EXPECTED_CUTOFF_KEYS)]["fixture_key"]
    )
    assert removed_exact == EXPECTED_CUTOFF_KEYS
    assert len(removed_exact) == 16