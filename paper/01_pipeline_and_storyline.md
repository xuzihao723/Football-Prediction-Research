# Soccer Match Outcome Prediction — Paper Pipeline & Storyline

> Target journal: Big Data (Springer Nature) — same family as the reference
> paper your professor published. Template: `sn-jnl` (sn-mathphys-num style).

---

## 1. One-paragraph "elevator pitch"

We build a **two-stage match-outcome prediction system** for the top-5
European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1)
covering **9 seasons (2015/16 – 2023/24, ≈18,000 matches)**. Stage 1
(*pre-match*) predicts H/D/A using only information available **before kick-off**
— historical PPG/xG, market odds, head-to-head, form/momentum, rest, ranking.
Stage 2 (*half-time*) re-predicts the same target using stage-1 features **plus
half-time signals** (HT score, HT lead, HT cards). We benchmark six
classifiers (Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost,
Feed-forward NN) and two meta-learners (Voting, Stacking), under a strict
**season-based time-series split** with leakage controls, smart missing-value
handling, class-imbalance treatment for draws, and hyper-parameter tuning.
The contribution is **not a single new model** but **a reproducible pipeline,
a draw-aware evaluation protocol, and a quantification of the information
value of half-time signals** — i.e., how much accuracy/F1/log-loss improves
once 45 minutes of play are observed.

---

## 2. Research questions (drives the whole paper)

- **RQ1.** How accurately can match outcome (H/D/A) be predicted **before
  kick-off** from publicly available pre-match features (form, odds, xG,
  H2H, rest, momentum)?
- **RQ2.** How much **additional predictive value** do half-time signals add,
  measured in accuracy / F1-macro / log-loss / AUC, under an identical
  protocol?
- **RQ3.** Which **feature families** (market odds vs. on-pitch performance
  proxies vs. recent form) carry the most signal — and does that ranking
  change between pre-match and half-time?
- **RQ4.** How do classical models (LR, NB, SVM, RF) compare to gradient
  boosting (XGBoost) and neural models (FNN) on this task, and do
  ensembles yield meaningful gains?
- **RQ5.** Can class-imbalance treatment (re-weighting, threshold tuning)
  meaningfully improve **draw recall** without collapsing H/A precision?

---

## 3. Contributions (claim list to put in the introduction)

1. **End-to-end reproducible pipeline** for top-5-league outcome prediction,
   built around a six-stage feature engineering chain
   (`feature_engineer1 … feature_engineer6`) and a `season-based` time-aware
   split that strictly excludes leakage from the in-progress season.
2. **A draw-aware evaluation protocol** using F1-macro, per-class
   precision/recall, ROC-AUC OvR, and log-loss — addressing the well-known
   weakness of accuracy on imbalanced 3-class outcomes.
3. **A controlled comparison of pre-match vs. half-time prediction** using
   *the same model space, same feature pipeline, same split*, isolating the
   marginal information value of 45 minutes of play.
4. **A benchmark of six base learners + two ensembles** with documented
   hyper-parameters and class-weighting schedules tailored to draw
   imbalance.
5. **A feature-importance analysis** that quantifies the contribution of
   each engineered family (odds, PPG/xG, form/momentum, H2H, rest) to both
   stages.

---

## 4. Pipeline (mirrors `D:\program\SoccerPre\` exactly)

```
[Raw match logs + odds]
        │   (per league CSV)
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE A — Data acquisition & merging                        │
│   merged_data/feature_engineer1.py        (combine matches) │
│   merged_data/feature_enginner2(fetch_odd).py (odds)        │
│   merged_data/feature_engineer3（history).py (H2H + rates)  │
│   merged_data/feature_engineer4(momentum).py (form/momentum)│
│   merged_data/feature_engineer5(rest).py  (rest, fatigue)   │
│   merged_data/feature_engineer6.py        (H2H, streaks,    │
│                                            rankings)        │
│   → merged_data/feature_engineer6/{ENG-premier,             │
│     span, Italy, Germany, France}.csv                       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE B — Loading & cleaning                                │
│   src/data/loader.py    load_data → preprocess_data → clean │
│   - drop in-progress season (2025-2026)                     │
│   - drop leakage columns (full-time stats, 2nd-half stats)  │
│   - keep half-time columns gated by task                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE C — Season-based split (leakage-safe)                 │
│   src/data/splitter.py                                      │
│   TRAIN: 2015/16 – 2020/21   (6 seasons)                    │
│   VAL  : 2021/22 – 2022/23   (2 seasons)                    │
│   TEST : 2023/24             (1 season, hold-out)           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE D — Feature construction                              │
│   src/features/prematch_features.py                         │
│     • drop HALFTIME + IDENTIFIER + TARGET cols              │
│     • smart NA handling (5 type-aware strategies)           │
│     • encode H/D/A target                                   │
│   src/features/halftime_features.py                         │
│     • keep prematch cols + HT score/lead/cards              │
│     • encode ht_result                                      │
│     • smart NA handling                                     │
│   src/features/feature_selection.py                         │
│     • variance ≤ 0 → drop                                   │
│     • |corr| ≥ 0.97 → drop one of the pair                  │
│   scripts/analyze_feature_importance.py                     │
│     • Random Forest MDI per (league, task)                  │
│     • per-league top-30 + cross-league intersection         │
│     • outputs/reports/feature_importance/*                  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE E — Modelling                                         │
│   src/training/trainer.py     (LR, NB, SVM, RF, XGB, FNN)   │
│   src/training/ensemble.py    (Voting soft, Stacking, RNN)  │
│   src/training/hyperparameter_tuning.py (Grid / Random)     │
│   src/training/imbalance_handler.py                         │
│     • class_weight ∈ {balanced, sqrt, log, custom}          │
│     • optimal-threshold search on validation                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE F — Evaluation & reporting                            │
│   src/evaluation/metrics.py                                 │
│     accuracy, log_loss, F1 {macro, weighted, per-class},    │
│     precision/recall per class, ROC-AUC OvR (per/macro/wt), │
│     confusion matrix                                        │
│   src/evaluation/visualization.py                           │
│     comparison radar / bars / per-class / CM                │
│   → outputs/reports/*  &  outputs/models/*.joblib           │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Storyline / narrative arc

> **Hook → gap → method → result → implication.**

1. **Hook.** Match-outcome prediction underpins betting markets, scouting,
   tactical analysis, and fan engagement. Despite a decade of papers, three
   weaknesses persist: (i) most studies stop at pre-match prediction and
   ignore in-game updating, (ii) the **draw class** is consistently
   under-modelled (papers report >55% accuracy but ~25% draw recall), and
   (iii) **temporal leakage** through random splits inflates test scores.

2. **Gap.** No public benchmark of pre-match vs. half-time prediction over
   the top-5 European leagues with (a) season-aware splitting, (b)
   draw-aware metrics, and (c) a model space that spans logistic
   regression to gradient boosting and neural models — under one pipeline.

3. **Method.** We construct a six-stage feature pipeline yielding ~150
   numeric features per match (odds, PPG/xG, form, momentum, H2H, rest,
   ranking, half-time signals when applicable). We train six base learners
   and two ensembles, with class-weight schedules tuned for draws and
   per-class threshold optimisation on validation. Splits are strictly
   season-based.

4. **Result.** *(fill in once experiments are finalized; placeholders below
   for the abstract.)*
   - Pre-match best model: XGBoost — **acc ≈ 0.55 / F1-macro ≈ 0.43 / log-loss ≈ 0.95**.
   - Half-time best model: XGBoost — **acc ≈ 0.71 / F1-macro ≈ 0.62 / log-loss ≈ 0.65**.
   - Half-time signals add **+15 pp accuracy** and **roughly halve log-loss**.
   - Odds and ELO (elo_diff_pre, home_elo_pre) dominate pre-match — 11
     odds + 2 ELO features form the universally-important top set;
     HT-lead and HT-result dominate half-time, with MDI scores ~3× any
     odds feature.
   - Class-weighting + threshold tuning lifts **draw F1 by ~5 pp** with
     <1 pp loss in H/A precision.

5. **Implication.** A simple, reproducible pipeline already extracts most
   of the predictable signal; the remaining ceiling is intrinsic noise of
   football. Half-time updating is **practically valuable** for live betting
   and tactical decisions and is technically cheap. Future work: in-play
   sequential models (xG flow) and per-league transfer.

---

## 6. Suggested paper layout (Springer `sn-jnl`)

| §   | Title                       | Length      | Source files                            |
|-----|-----------------------------|-------------|------------------------------------------|
| 1   | Introduction                | ~1.5 pp     | RQs, contributions, paper map            |
| 2   | Related work                | ~1.5 pp     | `literature/literature_review_*.docx`    |
| 3   | Data                        | ~2 pp       | `merged_data/feature_engineer*.py`       |
| 4   | Feature engineering         | ~2 pp      | `src/features/`                          |
| 5   | Methodology                 | ~2.5 pp    | `src/training/`, `src/data/splitter.py`  |
| 6   | Experimental setup          | ~1 pp       | `scripts/run_*_enhanced.py`              |
| 7   | Results                     | ~3 pp       | `outputs/reports/`                       |
| 8   | Discussion                  | ~1.5 pp    | feature importance, errors, limitations  |
| 9   | Conclusion & future work    | ~0.5 pp    |                                           |
| App | Hyper-parameter grids etc.  | ~1 pp       | `src/config/config.py`                   |

---

## 7. Key tables/figures to plan (placeholders)

- **Table 1.** Dataset summary: leagues × seasons × matches × class
  distribution (H/D/A %).
- **Table 2.** Feature families with cardinality and example columns
  (odds, PPG/xG, form, H2H, rest, momentum, half-time).
- **Table 3.** Train/Val/Test season assignment.
- **Table 4.** Pre-match results: per-model accuracy, F1-macro,
  log-loss, AUC-macro, per-class F1.
- **Table 5.** Half-time results: same columns as Table 4.
- **Table 6.** Pre-match → half-time delta per model.
- **Tables 7–11.** Per-league top-30 MDI features (England, Spain,
  Germany, Italy, France) — pre-match vs. half-time side by side with
  importance scores. *(§4.7.1, done — data in
  `outputs/reports/feature_importance/<league>_*/feature_importance.csv`.)*
- **Table 12.** Universally important features — intersection of the
  five top-30 lists, pre-match vs. half-time. *(§4.7.1, done.)*
- **Table 13.** Hyper-parameter search space (Appendix).
- **Figure 1.** Pipeline schematic (Section §4 here).
- **Figure 2.** Class distribution per season (sanity / drift check).
- **Figure 3.** Confusion matrices — best pre-match vs. best half-time.
- **Figure 4.** ROC curves (OvR) — H, D, A — best model both stages.
- **Figure 5.** Top-20 feature importances — pre-match vs. half-time
  (side-by-side barh plots). *(Generated —
  `outputs/reports/feature_importance/cross_league_top30_*.png`.)*
- **Figure 6.** Reliability/calibration diagram for the best model.
- **Figure 7.** Effect of class-weight + threshold tuning on draw
  precision/recall trade-off.

---

## 8. Risks & how to mitigate (note for ourselves)

- **Leakage from full-time stats** → already handled by `LEAKAGE_COLUMNS`
  in `feature_engineer6.py`; double-check half-time pipeline does not
  ingest 2nd-half features.
- **Leakage from `target_*` columns** → enforced by
  `IDENTIFIER_COLUMNS + TARGET_COLUMNS` in `prematch_features.py`.
- **Season distribution shift** (COVID 2019/20, ref. behaviour change)
  → report per-season test breakdown, not just overall mean.
- **Bookmaker-shift bias** (odds get sharper over time) → keep odds-free
  ablation as a robustness check.
- **Class imbalance** for draws → quantified, treated, reported.

---

## 9. Concrete TODOs to finalize results before submission

1. Re-run `scripts/run_prematch_enhanced.py` and `run_halftime_enhanced.py`
   per league + pooled, with `--tune --handle-imbalance --optimize-threshold`.
2. Aggregate `outputs/reports/` into a single results CSV
   (`paper/results_pooled.csv`) — one row per (league, model, stage).
3. Generate the figures listed in §7 from that CSV.
4. Run an **odds-ablation** (drop all `odds_*` features) for both stages
   to isolate market-information value.
5. Compute **bootstrap 95% CIs** over 200 resamples of the test set for
   accuracy / F1-macro per model — needed to claim significance.
6. Per-season test breakdown (Premier League 2023/24 month-by-month).
7. Calibration plot + Brier score for the best model.

**Done:** feature-importance ranking (§4.7.1) — Random Forest MDI per
(league, task), per-league top-30 tables, cross-league intersection.
Outputs already in `outputs/reports/feature_importance/`.
