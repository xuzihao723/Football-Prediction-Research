# Data availability

The local monorepo retains the original match and event datasets and frozen research outputs at their upstream-relative paths. The Git checkout distributes provenance and the validation code. Run the data-dependent tests only after restoring authorized inputs described in the [migration manifest](provenance/migration-manifest.tsv).

| Component | Local inputs and results |
| --- | --- |
| Europe match-level | `projects/europe-big-five/FootBallPredictionOnBig5/merged_data/` contains the staged and final tables, bridge files and event Parquet. The historical Europe JSON summaries and original ETL archive have their original relative names. |
| Europe event-level | The nested `FootBallRealTimePrediction/` directory contains `15-16_official_data/`, `real_time_data/`, `sb_events/` and the event-state Parquet. The large CSV is the materialized LFS object from the source checkout. |
| Gulf | `projects/gulf-market-anchored/official_data/` contains the league, canonical, modeling and recovery tables. `results/` retains original A–E, ordinary baselines, structured XGBoost and final manuscript-related results. |

The root ignore rules keep local-only data, model and prediction artifacts out of Git. The manifest identifies their providers' source snapshots, Git objects and content hashes. Do not use forced Git additions for these files. Restore an authorized file from the pinned archive at its listed path and use the verification script to confirm its SHA-256. Synthetic tests and the GitHub Actions validation subset have no private dataset requirement.

The local benchmarks use the materialized Europe event CSV and Gulf harmonized CSV. Keep experiment receipts local because they contain input paths, configuration and environment details. [Experiment receipts](reproducibility.md#experiment-receipts)

Data redistribution depends on the applicable terms for FootyStats, Football-Data.co.uk, StatsBomb, recovery data and derived artifacts. Obtain release decisions before distribution, including for per-fixture predictions. Code redistribution decisions are also pending for the integration branch. See the [full audit](DATA_AND_LICENSE_AUDIT.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
