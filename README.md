# Football Prediction Research

## Project Overview

A research-oriented collection of reproducible football forecasting pipelines, covering time-aware prediction across Europe's top-five leagues and market-anchored probabilistic forecasting in Gulf leagues.

## Publications & Research

| Study | Venue | Status |
| --- | --- | --- |
| Time-Aware Football Outcome Prediction across the Top-Five European Leagues: Quantifying Pre-Match, Half-Time, and Event-Level Momentum Signals | IJOCTA | Manuscript under revision |
| Market-Anchored Football Forecasting: Source-Specific Regularization and Score-Based Half-Time Updating | ICPRAM 2027 | Accepted short paper |

See the [paper records](papers/README.md) and [paper-to-code mapping](docs/paper-code-mapping.md) for version details and reproduction coverage.

## Research Contributions

- Compare outcome forecasts at kick-off and half-time under chronological evaluation.
- Derive match-level historical features and first-half momentum descriptors from event states.
- Study source-specific regularization and structured boosted-tree updates to market probabilities.
- Evaluate classification and probability quality on matched fixtures, including calibration and paired comparisons.

## European Big Five Prediction

The [European project](projects/europe-big-five/README.md) retains match cleaning, historical feature stages, pre-match and half-time training, tuning, evaluation and plotting. The [event component](projects/europe-big-five/FootBallRealPre/README.md) retains event-state construction, a momentum feature CLI and a CatBoost/Optuna pilot.

## Gulf Market-Anchored Forecasting

The [Gulf project](projects/gulf-market-anchored/README.md) retains chronological historical models, bookmaker probabilities, canonical market-anchored logistic regression, structured XGBoost, half-time score updates and the original A–E experiment runner. The frozen result corpus includes earlier A–E and manuscript-related branches.

## Repository Architecture

~~~text
projects/
  europe-big-five/
    FootBallPredictionOnBig5/   Match-level modelling and evaluation
    FootBallRealPre/            Event processing and momentum pilot
    5-Top league Data…/         Historical ETL archive
  gulf-market-anchored/         Market priors, temporal experiments and tests
papers/                         Publication records
docs/                           Protocols, provenance and integration report
scripts/                        Baseline verification
.github/workflows/              Independent component checks
~~~

The two European components have separate entry points and environments. Run each component from its own directory so its original `src` imports resolve within that project.

## Installation

Use CPython 3.11 and an independent virtual environment in each runnable component. The original `requirements.txt` files are preserved. The integration profile pins CPU-compatible top-level dependencies and includes pytest. Full model and data-access dependencies are described in [Reproducibility](docs/reproducibility.md).

**Windows PowerShell**

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-validation.txt
~~~

**Linux**

~~~bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-validation.txt
~~~

## Quick Start

Use the Python interpreter from the current component's environment.

| Working directory under `projects/` | Command |
| --- | --- |
| `europe-big-five/FootBallPredictionOnBig5` | `python scripts/run.py --help` and `python -m pytest -q tests` |
| `europe-big-five/FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction` | `python -m pytest -q tests` exercises the feature-building entry point on synthetic events |
| `gulf-market-anchored` | `python -m src.run_experiments --help` and `python -m pytest -q tests/test_migration_basics.py tests/test_runner.py tests/test_temporal.py` |

For a local Gulf development smoke with authorized inputs, run `python -m src.run_experiments --exp A --mode smoke` from the Gulf directory. The event component accepts `--input-csv` and `--output-csv` for isolated first-half feature builds.

From the repository root, run `python scripts/verify_baseline.py` to verify tracked migration files and local-only index exclusions. See [Reproducibility](docs/reproducibility.md) before running experiments that write outputs.

## Experimental Protocols

The European source snapshot includes its fixed season split and a separate 2015/16 chronological event pilot. The IJOCTA revision describes a distinct three-fold match-level protocol and a final-season holdout. For Gulf, five expanding outer folds hold out seasons 2020/21 through 2024/25, with date-aware filtering and within-training model selection. Gulf half-time score experiments preserve the pre-match market prior.

## Experimental Results

The European optimization reproduced the archived first-half momentum features for 1,823 fixtures byte-for-byte. The manuscript's exact final forecast tables still require the revision-specific fold runner and fixture-level outputs.

The Gulf study covers 5,499 fixtures. In a matched out-of-sample comparison over 2,789 fixtures, Zero-LR improved accuracy by 1.47 percentage points and reduced Log Loss by 0.01024 relative to Tied-Zero. These retrospective estimates were recalculated from the frozen predictions. See the [optimization report](docs/OPTIMIZATION_REPORT.md) and [frozen baseline](docs/provenance/frozen-baseline.md) for the validation scope and artifact hashes.

## Reproducibility

Source commits, file paths and SHA-256 hashes are recorded in [provenance](docs/provenance/sources.json) and the [migration manifest](docs/provenance/migration-manifest.tsv). The integration tests exercise source imports, entry points, time splits, feature generation and probability evaluation. Dataset-backed checks use the local snapshots. Use [experiment receipts](docs/reproducibility.md#experiment-receipts) to record input hashes, configuration, the interpreter and output artifacts. The [optimization report](docs/OPTIMIZATION_REPORT.md) records measured changes and equivalence checks.

## Data Availability

The local research copy retains datasets and frozen results at their research paths. Git excludes these assets while redistribution permissions are reviewed. [Data Availability](docs/data-availability.md) describes expected paths, access and permissions.

## Citation

Use the paper records for manuscript titles, venue and status, and cite the relevant upstream commit for code. Complete publication citations will be added when the final metadata is available. [Paper records](papers/README.md)

## Acknowledgements

The implementations derive from [Corey30/FootBallPredictionOnBig5](https://github.com/Corey30/FootBallPredictionOnBig5) and [Corey30/Market-AnchoredFootballForecastingOnAsia](https://github.com/Corey30/Market-AnchoredFootballForecastingOnAsia). The [source manifest](docs/provenance/sources.json) pins the exact upstream versions. The studies use football statistics and odds and, for the event pilot, StatsBomb Open Data. See the [third-party notices](THIRD_PARTY_NOTICES.md) for source and license references.
