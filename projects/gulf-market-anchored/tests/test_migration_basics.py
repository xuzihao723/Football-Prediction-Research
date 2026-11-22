import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import protocol
from src.feature_registry import get_registry
from src.models.market import bookmaker_prior
from src.run_experiments import _verify_raw


PROJECT = Path(__file__).resolve().parents[1]


def test_runner_cli_and_definitive_lock():
    cli = [sys.executable, "-X", "utf8", "-m", "src.run_experiments"]
    help_result = subprocess.run(
        [*cli, "--help"], cwd=PROJECT, capture_output=True, encoding='utf-8', timeout=60,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--confirm-definitive" in help_result.stdout

    env = os.environ.copy()
    env.pop(protocol()["definitive_lock_env"], None)
    locked = subprocess.run(
        [*cli, "--exp", "A", "--mode", "definitive"],
        cwd=PROJECT, env=env, capture_output=True, encoding='utf-8', timeout=60,
    )
    assert locked.returncode != 0
    assert "Definitive outer-test run is locked" in locked.stderr


def test_synthetic_market_schema_and_prediction_smoke():
    registry = get_registry()
    odds_columns = registry.market_prior_columns
    odds = pd.DataFrame([[2.0, 4.0, 4.0], [2.0, 1.0, 4.0]], columns=odds_columns)
    registry.validate_frame(odds, odds_columns)
    p, _, valid = bookmaker_prior(odds)
    assert valid.tolist() == [True, False]
    np.testing.assert_allclose(p[0], [0.25, 0.25, 0.5])
    raw = pd.DataFrame(
        {
            "fixture_key": ["synthetic-001"],
            "league": ["Qatar"],
            "season": ["2019-2020"],
            "fold": [0],
            "method": ["Market"],
            "representation": ["synthetic"],
            "actual_class": ["H"],
            "p_A": p[valid, 0],
            "p_D": p[valid, 1],
            "p_H": p[valid, 2],
        }
    )
    _verify_raw(raw, "market fixture")
