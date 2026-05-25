# Paper assets

Two companion documents for the *Big Data Journal* submission:

| File | Purpose |
|------|---------|
| [01_pipeline_and_storyline.md](01_pipeline_and_storyline.md) | The pitch, research questions, contribution list, full pipeline diagram (mirrors the repo structure one-to-one), narrative arc, paper layout, list of tables/figures, risks/mitigations, and the concrete TODOs needed to finalise the experimental numbers before submission. Read this first. |
| [02_main_draft.tex](02_main_draft.tex) | A full LaTeX manuscript draft in the Springer `sn-jnl` template style (same family as your professor's paper). Drop-in replacement for the body of `sn-article.tex`. Empty cells are marked `TBD` and must be filled from `outputs/reports/`. |

## How to compile

1. Copy `02_main_draft.tex` into `E:\soccerf\sn-article-template\` as
   `sn-article.tex` (back up the original first), **or** copy the
   template's `sn-jnl.cls`, `sn-mathphys-num.bst` and the two logo EPS
   files into `D:\program\SoccerPre\paper\` and compile here.
2. `pdflatex sn-article.tex && bibtex sn-article && pdflatex sn-article.tex && pdflatex sn-article.tex`

## How to fill the TBDs

Run the enhanced training scripts per league and stage, then aggregate:

```bash
python scripts/run_prematch_enhanced.py  --model xgboost --tune --handle-imbalance --optimize-threshold
python scripts/run_halftime_enhanced.py --model xgboost --tune --handle-imbalance --optimize-threshold
```

Do this for every model in {`logistic_regression`, `naive_bayes`, `svm`,
`random_forest`, `xgboost`, `fnn`, `voting`, `stacking`} and every stage.
Harvest the per-run JSON reports from `outputs/reports/` into a single
`paper/results_pooled.csv` with columns:

```
league, stage, model, accuracy, f1_macro, log_loss, auc_macro,
f1_H, f1_D, f1_A, precision_H, precision_D, precision_A,
recall_H, recall_D, recall_A
```

The `TBD` cells in Tables 4–6 and the `Δ` numbers in §7.3, the abstract
and §9 come directly from that CSV.

## What's still missing (do before submitting)

See §9 of [01_pipeline_and_storyline.md](01_pipeline_and_storyline.md):
odds-ablation run, bootstrap 95% CIs, per-season test breakdown, and a
calibration plot + Brier score for the best model.
