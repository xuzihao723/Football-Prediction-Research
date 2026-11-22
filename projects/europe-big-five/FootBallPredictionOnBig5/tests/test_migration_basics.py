import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import tune_hyperparameters
from src.data.loader import clean_data, load_data
from src.data.splitter import time_series_split
from src.evaluation.metrics import evaluate_classification
from src.features.halftime_features import process_halftime_features
from src.features.prematch_features import process_prematch_features
from src.momentum.config import PARQUET_PATHS


PROJECT = Path(__file__).resolve().parents[1]


def test_cli_help_from_another_directory(tmp_path):
    process = subprocess.run(
        [sys.executable, "-X", "utf8", str(PROJECT / "scripts" / "run.py"), "--help"],
        cwd=tmp_path, capture_output=True, encoding='utf-8', timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert all(option in process.stdout for option in ("--task", "--input-csv", "--output-dir"))
    assert list(tmp_path.iterdir()) == []


def test_reference_inputs_resolve_from_project_layout():
    merged = PROJECT / "merged_data"
    assert tune_hyperparameters.DATA_DIR_DEFAULT == merged / "feature_engineer5"
    assert merged / "event_state_table_big5_2015_16.parquet" in PARQUET_PATHS


def test_clean_load_split_and_task_features(tmp_path):
    reference = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "season": ["2019-2020", "2021-2022", "2023-2024", "2023-2024"],
            "match_datetime": ["2020-02-01", "2021-10-01", "2024-02-01", "2024-02-02"],
            "status": ["complete", "complete", "complete", "incomplete"],
            "target_match_result": ["H", "D", "A", "A"],
            "home_team_goal_count": [3, 1, 0, 0],
            "home_team_goal_count_half_time": [1, 0, 0, 0],
            "home_elo": [1500.0, 1600.0, 1700.0, 1600.0],
        }
    )
    csv_path = tmp_path / "synthetic.csv"
    reference.to_csv(csv_path, index=False)
    train, valid, test = time_series_split(clean_data(load_data(csv_path)))
    assert [part["match_id"].tolist() for part in (train, valid, test)] == [[1], [2], [3]]
    assert train.match_datetime.max() < valid.match_datetime.min() < test.match_datetime.min()
    pre, y_pre = process_prematch_features(train)
    halftime, y_ht = process_halftime_features(train)
    assert pre.columns.tolist() == ["home_elo"]
    assert set(halftime.columns) == {"home_elo", "home_team_goal_count_half_time"}
    assert y_pre.tolist() == y_ht.tolist() == ["H"]


def test_three_way_evaluation_matches_probability_columns():
    truth = np.array(["A", "D", "H"])
    probabilities = np.full((3, 3), 0.1)
    np.fill_diagonal(probabilities, 0.8)
    result = evaluate_classification(truth, truth.copy(), probabilities)
    assert result["accuracy"] == result["f1_macro"] == 1.0
    assert result["log_loss"] == pytest.approx(-np.log(0.8))



@pytest.mark.parametrize("task", ["prematch", "halftime"])
def test_training_cli_smoke_writes_to_requested_output(tmp_path, task):
    records = []
    for season in ("2019-2020", "2021-2022", "2023-2024"):
        start = pd.Timestamp(int(season[:4]), 9, 1)
        for i in range(12):
            fixture_time = start + pd.Timedelta(days=i)
            records.append(
                {
                    "match_id": len(records) + 1,
                    "date_GMT": fixture_time.strftime("%b %d %Y - %I:%M%p"),
                    "season": season,
                    "status": "complete",
                    "target_match_result": "ADH"[i % 3],
                    "home_elo": 1500.0 + 10 * (i % 3),
                    "home_team_goal_count_half_time": i % 2,
                }
            )
    data_path = tmp_path / "fixtures.csv"
    pd.DataFrame(records).to_csv(data_path, index=False)
    output_path = tmp_path / task
    result = subprocess.run(
        [
            sys.executable, "-X", "utf8", str(PROJECT / "scripts" / "run.py"),
            "--task", task, "--input-csv", str(data_path),
            "--models", "logistic_regression", "--output-dir", str(output_path),
        ],
        cwd=tmp_path, capture_output=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_path / "models" / f"{task}_stageE_baseline_logistic_regression.joblib").is_file(), result.stdout
    assert list((output_path / "reports").glob("*/model_comparison_bar.png"))
