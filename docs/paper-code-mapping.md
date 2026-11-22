# Paper-to-code mapping

The [paper records](../papers/README.md) give the titles, versions and publication status. The [baseline audit](BASELINE_AUDIT.md) documents the supporting evidence and gaps.

## IJOCTA · Europe Big Five

| Study component | Implementation and entry points | Reproduction coverage |
| --- | --- | --- |
| Pre-match and half-time outcome prediction | `FootBallPredictionOnBig5/data/data_cleaning.py`, `src/features/`, `src/training/` and `scripts/run.py` | Legacy `all_results.json` and `search_spaces.json` retained locally at Europe project root. The exact revision-specific fit, refit and eligible fixture keys are pending. |
| Historical stats, odds and temporal evaluation | Match `merged_data/feature_engineer*`, `src/data/splitter.py`, `scripts/tune_hyperparameters.py` | The legacy fixed split and tuning are available. The revision describes a distinct three-fold development protocol and 2024/25 holdout. |
| Event ingestion and first-half momentum | Nested `FootBallRealTimePrediction/src/form_table.py`, `src/patch.py`, `build_momentum_features.py` and `run_momentum_prediction.py`. The match component also has `src/momentum/` and curve experiment scripts. | Local 2015/16 event tables, join bridge and feature CSV retained. The revision's 374 test keys and corresponding predictions are pending. |
| Metrics and figures | Match `src/evaluation/`, `scripts/plot_momentum_curves.py`, `scripts/plot_tuning_summary.py` and `scripts/analyze_feature_importance.py` | Exact figure input versions, grouped ablations and final figure assembly still need provenance checks. |

## ICPRAM 2027 · Gulf Market-Anchored

| Study component | Implementation and entry points | Reproduction coverage |
| --- | --- | --- |
| Historical and market baselines | `src/models/{structural,market}.py`, `src/experiments/`, `src/pooled_split.py` | `official_data/modeling/ModelingData_harmonized.csv`, A–E OOS and ordinary baselines. Seven original configuration files retained. |
| Linear and nonlinear market updates | `src/models/{anchoring,xgb_v4}.py`, `tools/run_xgb_v4.py` and `tools/final_design_execution.py` | `results/xgb_v4/` fold outputs and diagnostics, `results/final_results/` canonical PRE, Zero and Tied-Zero outputs. |
| Half-time score and probability evaluation | `tools/final_design_execution.py`, `src/{metrics,statistics,tables}.py` and `tools/analyze_xgb_v4.py` | Seven-learner PRE/HT predictions, probability diagnostics, paired contrasts and Holm family registry. Some exact replay inputs and the final figure workflow are pending. |

Europe reproduced the archived 1,823-row event feature CSV byte-for-byte. Gulf preserved all 4,049 chronological batches on the retained data and the outer and inner fold memberships. The manuscript reconstruction gaps above remain open. [Optimization validation](OPTIMIZATION_REPORT.md)

See [Reproducibility](reproducibility.md) for commands and output protection.
