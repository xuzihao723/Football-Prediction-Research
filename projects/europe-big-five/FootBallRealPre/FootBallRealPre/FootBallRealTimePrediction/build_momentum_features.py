# -*- coding: utf-8 -*-
"""
Real-time momentum feature construction (first half only).


Two curves per match:
  - M_eq(t)   directional curve (who is dominating)
  - M_int(t)  intensity curve  (how active the match is)

Each curve is restricted to the first half (period=1), resampled onto a
uniform grid of 91 points, and summarised by 10 curve features, giving
20 momentum features per match.

input：event_state_table_big5_2015_16.csv
Output: momentum_features_big5_2015_16.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
EVENT_STATE_CSV = SCRIPT_DIR / "real_time_data" / "event_state_table_big5_2015_16.csv"
OUTPUT_CSV = SCRIPT_DIR / "real_time_data" / "momentum_features_big5_2015_16.csv"

# ---------------------------------------------------------------------------
# Expert weights for M_eq
# ---------------------------------------------------------------------------
W_INTERCEPT = 0.07
W_XG = 0.07
W_SHOTS = 0.01
W_BOX = 0.03
W_CORNERS = 0.02
W_TURNOVERS = 0.00
W_RED = -0.23
W_SCORE = 0.09
BETA_SCORE = 0.5  # tanh saturation

# Feature list for M_int uses |w_i| from the rolling-window block
W_INT = {
    "xg": abs(W_XG),
    "shots": abs(W_SHOTS),
    "box": abs(W_BOX),
    "corners": abs(W_CORNERS),
    "turnovers": abs(W_TURNOVERS),
}

# Rolling-window diff column names in the event state table
DIFF_COLS = {
    "xg": "recent_xg_diff",
    "shots": "recent_shots_diff",
    "box": "recent_box_entries_diff_v3",
    "corners": "recent_corners_diff",
    "turnovers": "recent_high_turnovers_diff_v3",
}

# Rolling-window home/away column names for M_int
HOME_COLS = {
    "xg": "recent_xg_home",
    "shots": "recent_shots_home",
    "box": "recent_box_entries_home_v3",
    "corners": "recent_corners_home",
    "turnovers": "recent_high_turnovers_home_v3",
}
AWAY_COLS = {
    "xg": "recent_xg_away",
    "shots": "recent_shots_away",
    "box": "recent_box_entries_away_v3",
    "corners": "recent_corners_away",
    "turnovers": "recent_high_turnovers_away_v3",
}

# Grid size for resampling (paper: 91 points spanning the half)
N_GRID = 91
# Minimum number of first-half state rows required (paper: 50)
MIN_ROWS = 50
# Time constant for end-loaded decay feature (paper: 600 s)
TAU_DECAY = 600.0


# ---------------------------------------------------------------------------
# Curve computation
# ---------------------------------------------------------------------------
def compute_meq(df_match: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute directional momentum M_eq(t) for a single match (first half)."""
    meq = np.full(len(df_match), W_INTERCEPT, dtype=float)
    # rolling-window differentials
    for key, col in DIFF_COLS.items():
        w = {"xg": W_XG, "shots": W_SHOTS, "box": W_BOX,
             "corners": W_CORNERS, "turnovers": W_TURNOVERS}[key]
        if w != 0.0:
            meq = meq + w * df_match[col].to_numpy(dtype=float)
    # cumulative state
    meq = meq + W_RED * df_match["red_card_diff"].to_numpy(dtype=float)
    meq = meq + W_SCORE * np.tanh(BETA_SCORE * df_match["score_diff"].to_numpy(dtype=float))
    t = df_match["time_sec"].to_numpy(dtype=float)
    return t, meq


def compute_mint(df_match: pd.DataFrame) -> np.ndarray:
    """Compute intensity momentum M_int(t) for a single match (first half)."""
    mint = np.zeros(len(df_match), dtype=float)
    for key, w in W_INT.items():
        if w == 0.0:
            continue
        mint = mint + w * (
            df_match[HOME_COLS[key]].to_numpy(dtype=float)
            + df_match[AWAY_COLS[key]].to_numpy(dtype=float)
        )
    return mint


# ---------------------------------------------------------------------------
# Resampling onto uniform grid
# ---------------------------------------------------------------------------
def resample_curve(t: np.ndarray, m: np.ndarray, n_grid: int = N_GRID) -> np.ndarray:
    """Resample (t, m) onto a uniform grid of n_grid points spanning [t_min, t_max]."""
    if len(t) < 2:
        return np.full(n_grid, np.nan)
    t_min, t_max = float(t.min()), float(t.max())
    if t_max <= t_min:
        return np.full(n_grid, float(m[0]))
    grid = np.linspace(t_min, t_max, n_grid)
    return np.interp(grid, t, m)


# ---------------------------------------------------------------------------
# Curve feature extraction (10 features per curve)
# ---------------------------------------------------------------------------
def extract_curve_features(m_grid: np.ndarray, t_grid: np.ndarray | None = None) -> dict:
    """Extract 10 summary features from a resampled momentum curve."""
    feats: dict = {}
    if m_grid is None or np.all(np.isnan(m_grid)):
        return dict.fromkeys(CURVE_FEATURES, np.nan)

    m = np.asarray(m_grid, dtype=float)
    n = len(m)
    # Build a time axis in seconds for the grid (0 .. T)
    if t_grid is None:
        # default: assume grid spans 0..2700s (typical first half)
        t_grid = np.linspace(0.0, 2700.0, n)
    t = np.asarray(t_grid, dtype=float)
    T = float(t[-1])

    # --- distribution shape ---
    feats["mean"] = float(np.mean(m))
    feats["max"] = float(np.max(m))
    feats["min"] = float(np.min(m))
    feats["area_signed"] = float(np.trapezoid(m, t))
    # zero crossings: count sign changes (ignore zeros)
    signs = np.sign(m)
    # remove zeros to avoid spurious crossings
    nonzero = signs[signs != 0]
    if len(nonzero) > 1:
        feats["zero_crossings"] = int(np.sum(np.diff(nonzero) != 0))
    else:
        feats["zero_crossings"] = 0

    # --- trend ---
    if n > 1 and np.std(t) > 0:
        slope, _ = np.polyfit(t, m, 1)
        feats["slope_overall"] = float(slope)
    else:
        feats["slope_overall"] = 0.0

    # --- end-of-half emphasis ---
    feats["value_at_end"] = float(m[-1])
    # mean over last 600s
    mask_last = t >= (T - 600.0)
    if mask_last.any():
        feats["mean_last_600s"] = float(np.mean(m[mask_last]))
    else:
        feats["mean_last_600s"] = float(m[-1])
    # slope over last third
    third_idx = int(np.ceil(n * 2 / 3))
    if n - third_idx > 1 and np.std(t[third_idx:]) > 0:
        slope_last, _ = np.polyfit(t[third_idx:], m[third_idx:], 1)
        feats["slope_last_third"] = float(slope_last)
    else:
        feats["slope_last_third"] = 0.0
    # end-loaded decay integral: ∫ M(t) * exp((t-T)/tau) dt
    decay_w = np.exp((t - T) / TAU_DECAY)
    feats["area_decay_tau600s"] = float(np.trapezoid(m * decay_w, t))

    return feats

# Also defines the column order for matches too short to summarize.
CURVE_FEATURES = (
    "mean", "max", "min", "area_signed", "zero_crossings", "slope_overall",
    "value_at_end", "mean_last_600s", "slope_last_third", "area_decay_tau600s",
)
LOGGER = logging.getLogger(__name__)


def summarize_match(events: pd.DataFrame) -> dict[str, str | int | float]:
    """Summarize one match in chronological event order."""
    row = {
        "match_id": int(events["match_id"].iloc[0]),
        "home_team": events["home_team"].iloc[0],
        "away_team": events["away_team"].iloc[0],
        "n_first_half_rows": len(events),
    }
    if len(events) < MIN_ROWS:
        row.update({f"{prefix}_{name}": np.nan
                    for prefix in ("eq", "int") for name in CURVE_FEATURES})
        return row

    times, directional = compute_meq(events)
    intensity = compute_mint(events)
    time_grid = np.linspace(float(times.min()), float(times.max()), N_GRID)
    for prefix, values in (("eq", directional), ("int", intensity)):
        grid = resample_curve(times, values)
        features = extract_curve_features(grid, time_grid)
        row.update({f"{prefix}_{name}": value for name, value in features.items()})
    return row


def summarize_first_half(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-half events in the legacy match and timestamp order."""
    ordered = events.sort_values(["match_id", "time_sec"])
    match_ids = ordered["match_id"].to_numpy()
    if pd.isna(match_ids).any():
        raise ValueError("First-half events require a match_id")
    if not len(match_ids):
        return pd.DataFrame()

    # Sorting makes each match contiguous, so each slice is selected once.
    boundaries = np.flatnonzero(match_ids[1:] != match_ids[:-1]) + 1
    offsets = np.concatenate(([0], boundaries, [len(match_ids)]))
    return pd.DataFrame(
        summarize_match(ordered.iloc[start:end])
        for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    )


def build_momentum_features(
    input_csv: Path | None = None, output_csv: Path | None = None,
) -> pd.DataFrame:
    """Read event states and write a new match-level feature CSV."""
    input_csv = EVENT_STATE_CSV if input_csv is None else Path(input_csv)
    output_csv = OUTPUT_CSV if output_csv is None else Path(output_csv)
    if output_csv.exists():
        raise FileExistsError(f"Momentum output already exists: {output_csv}")

    columns = {
        "match_id", "period", "time_sec", "home_team", "away_team", "score_diff",
        "red_card_diff", *DIFF_COLS.values(), *HOME_COLS.values(), *AWAY_COLS.values(),
    }
    events = pd.read_csv(input_csv, usecols=sorted(columns))
    first_half = events.loc[events["period"] == 1]
    LOGGER.info("Loaded %d events from %s, including %d first-half rows",
                len(events), input_csv, len(first_half))
    del events
    out = summarize_first_half(first_half)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig", mode="x")
    LOGGER.info("Wrote %d matches and %d columns to %s", *out.shape, output_csv)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize first-half momentum features.")
    parser.add_argument("--input-csv", type=Path, default=EVENT_STATE_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_momentum_features(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
