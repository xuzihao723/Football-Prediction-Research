"""Regression tests for the first-half feature builder."""

import numpy as np
import pandas as pd
import pytest

import build_momentum_features as momentum


def event_rows(count, match_id, period):
    columns = [*momentum.DIFF_COLS.values(), *momentum.HOME_COLS.values(),
               *momentum.AWAY_COLS.values()]
    frame = pd.DataFrame(0.0, index=range(count), columns=columns)
    return frame.assign(
        match_id=match_id, period=period, time_sec=np.arange(count) // 2,
        home_team="Home", away_team="Away", score_diff=0, red_card_diff=0,
    )


def test_explicit_io_preserves_period_order_and_missing_short_curves(tmp_path):
    events = pd.concat([
        event_rows(3, 3, 1),
        event_rows(momentum.MIN_ROWS, 2, 1),
        event_rows(5, 2, 2).assign(score_diff=50),
    ])
    source = tmp_path / "events.csv"
    output = tmp_path / "features.csv"
    events.sample(frac=1, random_state=42).to_csv(source, index=False)
    result = momentum.build_momentum_features(source, output)
    assert result["match_id"].tolist() == [2, 3]
    assert result["n_first_half_rows"].tolist() == [momentum.MIN_ROWS, 3]
    assert result.loc[0, "eq_mean"] == pytest.approx(momentum.W_INTERCEPT)
    assert result.loc[0, "int_area_signed"] == 0
    assert result.loc[1].filter(regex=r"^(eq|int)_").isna().all()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        momentum.build_momentum_features(source, output)
    assert output.read_bytes() == original


def test_missing_first_half_match_ids_fail_before_output(tmp_path):
    source = tmp_path / "events.csv"
    output = tmp_path / "features.csv"
    event_rows(2, None, 1).to_csv(source, index=False)
    with pytest.raises(ValueError, match="match_id"):
        momentum.build_momentum_features(source, output)
    assert not output.exists()
