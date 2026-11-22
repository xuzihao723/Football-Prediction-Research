# Research overview

| Component | Prediction task | Inputs and protocol |
| --- | --- | --- |
| Europe match-level | Three-way outcome before kick-off and at half-time | Ten-season structured match data, historical features and task-specific match state. Legacy runs and the revision's expanding-window evaluation have separate provenance. |
| Europe event-level | Outcome forecasts using first-half descriptors on the event pilot | 2015/16 event states, formula and intensity curves, match join and chronological trial. Separate next-goal analyses under the match component use an event-level target. |
| Gulf market-anchored | Three-way probabilities in Qatar, Saudi Arabia and the UAE | Historical predictors and de-vigged 1X2 probabilities, five expanding folds and score-only half-time updates in the manuscript analysis. |

Each component retains its original methods and entry points. Regression evidence covers the archived first-half feature CSV, exact Gulf time batches and fold memberships, and small before/after prediction probes. [Optimization report](OPTIMIZATION_REPORT.md) [Frozen baseline](provenance/frozen-baseline.md)
