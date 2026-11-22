# Football Prediction Research

**English** · [简体中文](README.zh-CN.md)

[![Research checks](https://github.com/xuzihao723/Football-Prediction-Research/actions/workflows/research-checks.yml/badge.svg?branch=main)](https://github.com/xuzihao723/Football-Prediction-Research/actions/workflows/research-checks.yml)

## About

How do historical data, market probabilities, and information observed during a match change a football forecast? This research monorepo evaluates three-way full-time outcomes in two settings, with separate Python environments and temporal evaluation protocols.

[Research projects](#research-projects) · [Publications](#publications) · [Quick start](#quick-start) · [Results](#results) · [Reproducibility](#reproducibility) · [Data and citation](#data-and-citation)

## Research projects

| Project | Research question | Implementation |
| --- | --- | --- |
| [European Big Five Prediction](projects/europe-big-five/README.md) | How do match histories and first-half information affect pre-match and half-time forecasts? | Match-level data preparation, training and evaluation. A separate StatsBomb event component builds first-half momentum features and runs a CatBoost pilot. |
| [Gulf Market-Anchored Forecasting](projects/gulf-market-anchored/README.md) | How should a pre-match market forecast be updated using historical features and the half-time score? | Bookmaker-derived probabilities, market-anchored logistic regression and structured XGBoost. Evaluation uses five expanding temporal folds across Qatar, Saudi Arabia, and the UAE. |

## Publications

| Study | Venue and status |
| --- | --- |
| Time-Aware Football Outcome Prediction across the Top-Five European Leagues: Quantifying Pre-Match, Half-Time, and Event-Level Momentum Signals | **IJOCTA** · Manuscript under revision |
| Market-Anchored Football Forecasting: Source-Specific Regularization and Score-Based Half-Time Updating | **ICPRAM 2027** · Accepted short paper |

[Paper records](papers/README.md) · [Paper-to-code mapping](docs/paper-code-mapping.md)

## Quick start

Clone the repository, then open one of the component directories below. Use CPython 3.11 and create a separate environment in each component you run.

~~~sh
git clone https://github.com/xuzihao723/Football-Prediction-Research.git
cd Football-Prediction-Research
~~~

| Component directory | First check after installation |
| --- | --- |
| `projects/europe-big-five/FootBallPredictionOnBig5` | `python scripts/run.py --help` |
| `projects/europe-big-five/FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction` | `python build_momentum_features.py --help` |
| `projects/gulf-market-anchored` | `python -m src.run_experiments --help` |

**Linux / macOS**

~~~sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-validation.txt
~~~

**Windows PowerShell**

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-validation.txt
~~~

The validation profile contains pinned CPU dependencies. The Europe validation suites use synthetic fixtures. The full Gulf test suite also uses local research files, so start with the data-independent Gulf checks below. See the reproducibility guide for platform requirements and full experiment setup.

~~~sh
# From either Europe component directory
python -m pytest -q tests

# From the Gulf component directory
python -m pytest -q tests/test_migration_basics.py tests/test_runner.py tests/test_temporal.py tests/test_temporal_regression.py
~~~

[Reproducibility guide](docs/reproducibility.md) · [CI test matrix](.github/workflows/research-checks.yml)

## Results

- **Europe** — The optimized feature builder reproduced the archived first-half feature table for **1,823 fixtures** byte for byte. Reproducing the manuscript's final forecast tables still requires the revision-specific fold runner and fixture-level outputs.
- **Gulf** — On **2,789 matched out-of-sample fixtures** from the 5,499-fixture study, Zero-LR improved accuracy by **1.47 percentage points** and reduced log loss by **0.01024** relative to Tied-Zero. These retrospective estimates were recalculated from frozen predictions.

[Optimization report](docs/OPTIMIZATION_REPORT.md) · [Frozen baseline](docs/provenance/frozen-baseline.md)

## Reproducibility

From the repository root, run `python scripts/verify_baseline.py` to check tracked files against the migration and optimization records. The run recorder captures input and output hashes, configuration, the Python environment, and the Git revision for new experiments.

`papers/` indexes the studies, `docs/` records the protocols and evidence, and `scripts/` contains the verification, benchmark, and run-recording tools.

[Reproduction instructions](docs/reproducibility.md) · [Research overview](docs/research-overview.md)

## Data and citation

Research datasets and per-fixture outputs required for full replication are kept in the local research copy. Consult the data guide for access, licensing, and attribution requirements. To cite a study, use the title and publication details in the paper records, and include the Git commit used for any code-based result.

[Data availability](docs/data-availability.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Citation details](papers/README.md)
