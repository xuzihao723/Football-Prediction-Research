# Football Match Prediction Research Repo

This repository is part of a collaborative research project on football
match outcome prediction across the top five European leagues.

The project studies a data-to-model pipeline for predicting full-time
football match outcomes in pre-match and half-time settings. It combines
match-level data construction, leakage-controlled preprocessing,
season-based validation, feature engineering, model training,
hyperparameter tuning, model comparison, and feature-importance analysis.

Original collaboration repository:
https://github.com/Corey30/FootBallPredictionOnBig5

## Project Highlights

- Builds a match-level modelling dataset from top-five European league
  matches across multiple seasons.
- Uses leakage-aware data cleaning and season-based train/validation/test
  splits to better approximate realistic forecasting conditions.
- Constructs feature groups for team strength, historical performance,
  home/away context, recent form, betting-market movements, rest,
  congestion, travel, and half-time match state.
- Compares classical machine-learning and neural-network models, including
  Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost, and FNN.
- Reports model performance with accuracy, macro-F1, AUC, log loss,
  per-class metrics, and feature-importance diagnostics.

## Repository Structure

```text
data/              Data loading, cleaning, and exploratory analysis scripts
merged_data/       Raw, intermediate, and engineered match-level datasets
src/               Core modelling package: data split, features, training, evaluation
scripts/           Training, tuning, aggregation, and feature-importance scripts
outputs/           Generated reports, tuning summaries, and figures
paper/             Manuscript drafts and research storyline notes
literature/        Literature review notes
```

## Main Workflow

1. Clean and validate match-level data.
2. Split matches chronologically by season.
3. Build pre-match and half-time feature spaces.
4. Run model training and hyperparameter tuning.
5. Aggregate evaluation results and analyse feature importance.

Example commands:

```bash
python scripts/run_all_tuning.py
python scripts/aggregate_tuning_results.py
python scripts/analyze_feature_importance.py --task prematch --league all
python scripts/analyze_feature_importance.py --task halftime --league all
```

## Notes on Data

This repository contains research data and derived outputs used during the
collaborative project. Public reuse of datasets should follow the terms of
the original data providers, including FootyStats, football-data.co.uk, and
StatsBomb where applicable.

## License and Use

See [LICENSE](LICENSE). This repository is provided for research portfolio
and academic collaboration reference. It is not released as an open-source
software package unless explicit permission is granted by the project
collaborators.
