import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import build_momentum_features as momentum
import run_momentum_prediction as prediction


PROJECT = Path(__file__).resolve().parents[1]


def test_momentum_cli_filters_second_half_and_writes_one_row(tmp_path):
    script = tmp_path / "build_momentum_features.py"
    shutil.copyfile(PROJECT / script.name, script)
    data = tmp_path / "real_time_data"
    data.mkdir()
    columns = set(momentum.DIFF_COLS.values())
    columns.update(momentum.HOME_COLS.values())
    columns.update(momentum.AWAY_COLS.values())
    first_half_rows = momentum.MIN_ROWS
    events = pd.DataFrame({col: [0.0] * (first_half_rows + 1) for col in columns})
    events["match_id"] = 1
    events["home_team"] = "Synthetic Home"
    events["away_team"] = "Synthetic Away"
    events["period"] = [1] * first_half_rows + [2]
    events["time_sec"] = list(range(first_half_rows + 1))
    events["red_card_diff"] = 0
    events["score_diff"] = [0] * first_half_rows + [20]
    events.to_csv(data / momentum.EVENT_STATE_CSV.name, index=False)

    process = subprocess.run(
        [sys.executable, "-X", "utf8", str(script)], cwd=tmp_path, capture_output=True, encoding='utf-8', timeout=60,
    )
    assert process.returncode == 0, process.stderr
    output = pd.read_csv(data / momentum.OUTPUT_CSV.name)
    assert output["match_id"].tolist() == [1]
    assert output.loc[0, "n_first_half_rows"] == first_half_rows
    assert output.loc[0, "eq_mean"] == pytest.approx(momentum.W_INTERCEPT)
    assert output.loc[0, "int_mean"] == 0.0


def test_event_prediction_entry_and_chronological_split():
    assert callable(prediction.main)
    fixtures = pd.DataFrame({"match_datetime": pd.date_range("2020-01-01", periods=10)[::-1]})
    train, valid, test = prediction.chronological_split(fixtures)
    assert (len(train), len(valid), len(test)) == (6, 2, 2)
    assert train.match_datetime.max() < valid.match_datetime.min() < test.match_datetime.min()
