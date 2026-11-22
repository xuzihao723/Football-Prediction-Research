# Research papers

Publication status follows the research team's supplied project records. Citation metadata is tied to the inspected manuscript versions.

## IJOCTA

- **Title** — Time-Aware Football Outcome Prediction across the Top-Five European Leagues: Quantifying Pre-Match, Half-Time, and Event-Level Momentum Signals.
- **Authors in the inspected revision** — Gaoyuan Zhang, Youhao Zhang, Zihao Xu, Zixuan Feng, Ebenezer Fiifi Emire Atta Mills and Baha Ihnaini. The source lists the first three as equal contributors.
- **Venue and status** — An International Journal of Optimization and Control: Theories & Applications (IJOCTA), manuscript under revision.
- **Summary** — Compares pre-match and half-time forecasts in five European leagues and examines first-half momentum in a separate 2015/16 event pilot.
- **Project and implementation** — [Europe Big Five](../projects/europe-big-five/README.md). See `FootBallPredictionOnBig5/scripts/run.py`, `src/features/`, `src/data/splitter.py` and the outer `FootBallRealPre` component.
- **Reproduction** — Follow the Europe run instructions and verify the archived JSON and event data. The revision-specific fold runner, fixture-level outputs and 374 event-pilot test keys are pending. [Mapping](../docs/paper-code-mapping.md)
- **DOI, public URL and formal citation** — Pending verified publication metadata.

## ICPRAM 2027

- **Title** — Market-Anchored Football Forecasting: Source-Specific Regularization and Score-Based Half-Time Updating.
- **Authors** — Pending verification against the author-approved camera-ready record. The inspected review manuscript is anonymous.
- **Venue and status** — International Conference on Pattern Recognition Applications and Methods (ICPRAM 2027), accepted short paper.
- **Summary** — Evaluates source-specific regularization, structured XGBoost and score-only half-time updates with the same pre-match market prior under five expanding folds.
- **Project and implementation** — [Gulf Market-Anchored](../projects/gulf-market-anchored/README.md). See `src/models/anchoring.py`, `src/models/xgb_v4.py`, `src/pooled_split.py` and `tools/final_design_execution.py`.
- **Reproduction** — Verify data and frozen predictions in a local research copy, then use `python -m src.preflight` and the Gulf tests. Some manuscript-generation inputs are missing from the upstream snapshot. [Reproduction instructions](../docs/reproducibility.md)
- **DOI, public URL and formal citation** — Pending verified proceedings metadata and author list.
