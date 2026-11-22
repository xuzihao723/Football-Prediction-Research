# Frozen baseline

This record pins the inherited protocols and artifacts to the original source commits. Every inherited path, Git blob, byte size and SHA-256 is in [migration-manifest.tsv](migration-manifest.tsv). [Source references](sources.json)

| Study | Frozen protocol and code |
| --- | --- |
| Europe match-level | `src/config/config.py` sets H/D/A targets, seed 42 and model settings. `src/data/splitter.py` partitions the source runner into training 2015/16–2020/21, validation 2021/22–2022/23 and test 2023/24–2024/25. `src/features/`, `src/training/` and `src/evaluation/metrics.py` define the inherited feature, fitting and evaluation rules. `scripts/tune_hyperparameters.py` preserves candidate selection and accuracy scoring. The local search-space JSON includes its upstream archive provenance. |
| Europe event-level | `build_momentum_features.py` uses period 1, fixed equation and intensity weights, a 91-point curve and the original minimum event count. `run_momentum_prediction.py` preserves four conditions, a chronological 60/20/20 partition, seed 42, CatBoost/Optuna settings and evaluation. The event code's H/D/A encoding is preserved. |
| Gulf | `configs/protocol_final.json` defines A/D/H order, seed 42, outer seasons and the definitive lock. `configs/{feature_registry_final,feature_availability,feature_groups_harmonized,odds_features,sk_selection_frozen,hpo_final_models}.json` define the inherited feature, market and tuning rules. `src/pooled_split.py`, `src/hpo.py`, `src/metrics.py` and `src/statistics.py` preserve time cutoffs, train-only selection, metric formulas and paired evaluation. Preserve the distinction between training-time Log Loss and the manuscript's Accuracy-first final evaluation. |

## Key file hashes

All hashes below are for locally retained contents or original tracked configurations. Other source data, prediction outputs, model settings and analysis diagnostics are indexed by full path in the [migration manifest](migration-manifest.tsv).

| File relative to study root | Experiment | SHA-256 |
| --- | --- | --- |
| Europe `all_results.json` | Legacy match-level results | `e10586ac3fcba8a68b72545430d4e6fb2e6ebbd09bb7d86e279704c43b6b07e8` |
| Europe `search_spaces.json` | Archived model search configurations | `021a1dd678a2b7df83ae308791a3b4adcba66a0f15128e9d199564d9361faa55` |
| Europe event `real_time_data/event_state_table_big5_2015_16.csv` | Materialized first-half experiment input | `1e28a020f0b6eba0c9e1905de44cec44813b2c4bcf1147fde699a61a488a12e3` |
| Gulf `configs/protocol_final.json` | Five expanding folds | `2b76d1e4d625ed4e090eadba3a0f45403f46723d4bbd2327b9f0078a0f1d390e` |
| Gulf `official_data/modeling/ModelingData_harmonized.csv` | 5,499-match modeling table | `641171ab19cecc89a9afa241d5cde121369fdc2dcfa5bd52f46491e37f16db0a` |
| Gulf `results/final_results/canonical_prematch_oos.csv` | Canonical PRE comparisons | `81b90022863c0858dd56f314d66700e72c6798bdc2dce17e49bfa4086f2534a3` |
| Gulf `results/final_results/final_probability_diagnostics.csv` | Final evaluated probabilities | `e4e9a77b36967afb92d51fb590cf652bcd0c1b76bee4511e2a04a62247602eb5` |
| Gulf `results/final_results/final_score_only_seven_learner_predictions.csv` | Score-only PRE/HT robustness panel | `9e8ee9211dfdebac305f1c1d2afd5dfbac205255dcb0c747856da1d5fdd1c48c` |
| Gulf `results/final_results/final_paired_contrasts.csv` | Paired statistics and Holm-adjusted comparisons | `ebd98ba2aae48ba293d113bd3639324bdbe3faa2d396636dfaceb85c29c1818a` |
| Gulf `results/xgb_v4/reports/validated_oos_predictions.csv` | Structured XGBoost and comparison models | `7dc6228fa462ff186b60c274234c5db5341e199f7e1521f7b544ea5c88eb5319` |

The upstream fixed Europe split and the IJOCTA revision's reported three folds and final holdout are distinct records. Gulf A–E, XGB v4 and the manuscript-related dataset each have separate artifact paths. Existing results remain byte-identical to their source versions. Keep future output-writing runs in an isolated copy. [Reproduction instructions](../reproducibility.md)
