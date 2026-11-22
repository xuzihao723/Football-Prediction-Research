# Reproducibility

## Component roots

| Component | Run from |
| --- | --- |
| Europe match-level | `projects/europe-big-five/FootBallPredictionOnBig5` |
| Europe event-level | `projects/europe-big-five/FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction` |
| Gulf | `projects/gulf-market-anchored` |

Each uses its own environment and original project imports. The legacy Europe ETL archive also retains original one-off scripts with machine-specific paths. The active match-level data paths use the migrated project layout.

## Python setup

The reference environment is Windows x64 CPython 3.11.9. Use CPython 3.11 for each component. The Linux workflow targets x86-64 Ubuntu 24.04. The source requirements use minimum versions; Python 3.13 and other architectures are outside the verified matrix. Both European momentum implementations call `numpy.trapezoid`, so the integration profiles pin NumPy 2.x. The selected pandas, SciPy and scikit-learn versions are compatible with those profiles. Match-level and Gulf validation use separate CPU XGBoost 2.1.4 and 3.0.5 environments. CatBoost 1.2.8 is installed in the event and Gulf environments. TensorFlow is used only by the European match-level RNN branch and is excluded from routine validation.

From one component root, create a fresh environment and install its validation profile.

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-validation.txt
.\.venv\Scripts\Activate.ps1
~~~

~~~bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-validation.txt
source .venv/bin/activate
~~~

For the full original model stack, install `python -m pip install -c constraints-3.11.txt -r requirements.txt` in a separate fresh environment, especially when switching from `xgboost-cpu` to `xgboost`. The match profile constrains TensorFlow to 2.19.1 for optional use. This branch's TensorFlow runtime and GPU paths were outside validation. TensorFlow GPU use on Windows requires a supported alternative such as WSL2 for this generation. [TensorFlow platform support](https://www.tensorflow.org/install/pip)

## Run checks

- **Match-level** — `python -m compileall -q src scripts data merged_data tests`, `python scripts/run.py --help`, `python scripts/tune_hyperparameters.py --help` and `python -m pytest -q tests`.
- **Event-level** — `python -m compileall -q src build_momentum_features.py run_momentum_prediction.py tune_england_pre_momentum.py tests` and `python -m pytest -q tests`. The tests invoke the momentum builder in a temporary directory and exercise the chronological split.
- **Gulf local** — `python -m compileall -q src tools tests`, `python -m src.preflight` and `python -m pytest -q tests`. Some tests need the retained fixtures and archived diagnostics. Several fit small estimators to validate contracts.
- **Gulf CI** — The workflow runs a named, data-independent subset of the inherited tests and the new integration checks. See [workflow](../.github/workflows/research-checks.yml) for the exact list.
- **Root** — `python scripts/verify_baseline.py` on a Git checkout. Use `python scripts/verify_baseline.py --include-local --source-root ..` from the root of this workspace's research copy to verify both target and source bytes.

## Reproduction with authorized data

Run legacy commands that write outputs in a separate writable run copy of the component. Keep relative input paths from the manifest in that copy or pass the documented input flags. For example, from the match root run `python scripts/run.py --task prematch --league England --models logistic_regression --output-dir runs/example-pre` or replace `prematch` with `halftime`. For tuning, check the preserved criteria with `python scripts/tune_hyperparameters.py --help` and give each run its own `--output-dir`. Legacy runners without an output option use their component-local directories.

The event component's `python build_momentum_features.py` writes `real_time_data/momentum_features_big5_2015_16.csv`, so use a run copy with independent files. The existing `python run_momentum_prediction.py` requires the five local 2015/16 league files and the fixture-level momentum export to run the Optuna/CatBoost pilot. The StatsBomb retrieval scripts and large event material are optional for the synthetic tests.

Gulf `python -m src.run_experiments --exp A --mode smoke` uses the development seasons and requires the local harmonized table. Formal A–E runs use upstream confirmation and environment controls and may overwrite legacy-named outputs in a run copy. Use the [paper mapping](paper-code-mapping.md) to distinguish earlier runners and the final artifact chain. Use retained results to reproduce available tables before considering full model runs.
