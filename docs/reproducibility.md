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

The event component accepts explicit input and output files. Use `python build_momentum_features.py --input-csv real_time_data/event_state_table_big5_2015_16.csv --output-csv <new-file>`. The output path must be new. The existing `python run_momentum_prediction.py` requires the five local 2015/16 league files and the fixture-level momentum export to run the Optuna/CatBoost pilot. The StatsBomb retrieval scripts and large event material are optional for the synthetic tests.

Gulf `python -m src.run_experiments --exp A --mode smoke` uses the development seasons and requires the local harmonized table. Formal A–E runs use upstream confirmation and environment controls and may overwrite legacy-named outputs in a run copy. Use the [paper mapping](paper-code-mapping.md) to distinguish earlier runners and the final artifact chain. Use retained results to reproduce available tables before considering full model runs.


## Experiment receipts

Record a run with `scripts/record_run.py` before invoking a project entry point. Keep the spec and receipts under ignored `runs/`. The recorder captures the declared command, experiment ID, feature and model configuration, seed, training protocol, metric names, input SHA-256 hashes, Git state, interpreter and package versions. Finalization records the actual exit code, output hashes and optional metric values read from a run-local JSON file. It checks input contents and declared path targets again. Runs with changed inputs or Git state are invalidated, and runs that start on a dirty tree have unverified code provenance.

Create `runs/event-spec.json` with the following fields. Relative input paths resolve beside the spec.

~~~json
{
  "project": "europe-event",
  "experiment_id": "first-half-features",
  "inputs": {
    "events": "../projects/europe-big-five/FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction/real_time_data/event_state_table_big5_2015_16.csv"
  },
  "configuration": {"period": 1, "grid_size": 91, "minimum_events": 50},
  "random_seed": null,
  "model_parameters": {},
  "training_protocol": {"type": "feature-only"},
  "evaluation_metrics": [],
  "command": ["python", "build_momentum_features.py", "--input-csv", "<events>", "--output-csv", "<run>/outputs/momentum.csv"]
}
~~~

From the monorepo root, run the recorder and the chosen command with the same interpreter and environment. The command in the spec is descriptive. The recorder reserves an output directory and the existing research entry point performs the calculation.

~~~powershell
$py = '.\.venvs\event\Scripts\python.exe'
$event = 'projects/europe-big-five/FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction'
$run = & $py scripts/record_run.py start --python $py --spec runs/event-spec.json
if ($LASTEXITCODE -ne 0) { throw 'Run reservation failed' }
& $py "$event/build_momentum_features.py" --input-csv "$event/real_time_data/event_state_table_big5_2015_16.csv" --output-csv "$run/outputs/momentum.csv"
$exitCode = $LASTEXITCODE
& $py scripts/record_run.py finish $run --exit-code $exitCode
~~~

For a run that produces metrics, add `--metrics-json metrics.json` to `finish` and write the JSON object under the reserved `outputs/` directory. Put reports and predictions in that directory too. In Linux use the corresponding component `.venv/bin/python` and capture the experiment exit status with `$?`. For legacy entry points with implicit output paths, follow the isolated workspace procedure above.

## Benchmark the optimized stages

From the monorepo root, run these commands with the existing component environments and authorized local inputs. The benchmark uses the Git integration commit to load the baseline modules, so that commit must be present in the checkout. Event outputs are created in temporary files. The Gulf benchmark times grouping a loaded table. Both benchmarks check output hashes, report individual timings and omit the first warm-up run from each median.

~~~powershell
.\.venvs\event\Scripts\python.exe scripts/benchmark_refactors.py europe-event --repeats 4
.\.venvs\gulf\Scripts\python.exe scripts/benchmark_refactors.py gulf --repeats 8
~~~

Run `python scripts/verify_baseline.py --include-local --source-root ..` to check the original sources and frozen assets against the immutable migration manifest. Audited Python changes have separate SHA-256 entries in [optimization overrides](provenance/optimization-overrides.json). The optimized Git code fingerprints recorded by the benchmark identify the actual module bytes it executed.
