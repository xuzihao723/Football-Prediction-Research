# Market Anchored Football Forecasting

This repository contains the reproducible implementation for three-way football outcome forecasting (Away, Draw, Home) with historical features, a pre-match market prior, and half-time score updating.

The project studies how historical football information should update an already strong external probabilistic forecast. Implementations include ordinary LR/XGBoost baselines, canonical market-anchored LR (Zero, Free, Soft, Fixed), Tied-Zero as a penalty-mechanism control, and structured XGBoost (Fixed, MC, KL).

## Repository layout

```text
src/                 Runtime models, feature construction, temporal splits, metrics
configs/             Frozen feature registry, protocol, and HPO grids
official_data/       Harmonized 5,499-match data and canonical source files
data_engineering/    Original feature engineering stages
tests/               Deterministic unit and protocol tests
tools/               Reproducible experiment and validation entry points
results/raw/         Frozen Experiment A-E OOS predictions
results/standard_prematch_baselines/  Ordinary historical and joint LR/XGB
results/xgb_v4/      Frozen structured XGBoost summaries and predictions
results/final_results/  Final canonical, HT, SmartVoting, contrasts, and metrics
docs/                Selected manuscript figures
```

The public release excludes audit workspaces, temporary logs, cached models, MCHR development code, raw SHAP fixture dumps, manuscript QA files, and generated archives. The original working directory remains the archival source.

## Data and forecast protocol

- 5,499 completed fixtures from Qatar, Saudi Arabia, and the UAE.
- Five expanding temporal folds with timestamp batching and date-only isolation.
- Historical representation selected inside training windows; the frozen headline representation is Elo plus Standings.
- Market probabilities use proportional de-vigging from valid decimal 1X2 odds.
- Market logits use Draw as the reference class: `log(p_H/p_D)` and `log(p_A/p_D)`.
- Half-time models add observed home, away, and total half-time goals while retaining the same pre-match market prior. No live half-time odds are used.
- All preprocessing and hyperparameter selection are training-only.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
.\.venv\Scripts\activate       # Windows
source .venv/bin/activate             # Linux/macOS
python -m pip install -r requirements.txt
```

## Verify the installation

```bash
python -m src.build_feature_registry
python -m src.preflight
python -m pytest -q tests
```

Smoke runs are safe for a fresh checkout:

```bash
python -m src.run_experiments --exp A --mode smoke
python -m src.run_experiments --exp B --mode smoke
python -m src.run_experiments --exp C --mode smoke
python -m src.run_experiments --exp D --mode smoke
python -m src.run_experiments --exp E --mode smoke
```

Definitive outer evaluation remains protected by the existing safety lock. Review the protocol and data before enabling it.

## Results

The final matched population contains 2,789 fixtures. Main saved tables include:

- `results/final_results/final_probability_diagnostics.csv`: final model metrics and descriptive ECE.
- `results/final_results/final_paired_contrasts.csv`: paired league-season bootstrap contrasts.
- `results/final_results/final_holm_family_registry.csv`: family-wise Holm records.
- `results/final_results/final_score_only_seven_learner_metrics.csv`: PRE and HT score-added robustness panel.
- `results/final_results/final_league_specific_results.csv`: descriptive league summaries.
- `results/raw/experiment_a_oos.csv` through `experiment_e_oos.csv`: frozen legacy prediction files.

Log Loss is the prespecified development and evaluation metric. Accuracy, Brier score, and Macro-F1 are complementary measures. Paired inference uses 15 league-season blocks, 10,000 resamples, and seed 42.

## Scope and data use

Prediction-level artifacts are included so metrics can be recomputed without retraining. The retrospective outer seasons are not an independent prospective validation set. Exact bookmaker identity and odds timing are not fully known. MCHR, lineup, injury, Power de-vig, Shin de-vig, and temperature-scaling experiments are outside the frozen main protocol and are not included in this public runtime package.

Verify that your intended GitHub repository has permission to redistribute the included data before making it public.
