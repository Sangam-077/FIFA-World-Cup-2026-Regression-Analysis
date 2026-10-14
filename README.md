# FIFA World Cup 2026 — Objective 2 Regression Project

This project builds and evaluates two linear regression models for Objective 2. It is designed to run from the project root and uses a single predictor configuration file (`model_config.py`) so that dataset audits, holdout validation and cross-validation stay aligned.

## Project structure

- `sources/base_matches.csv` — target match list and observed scores.
- `sources/results.csv` — historical international match results used to calculate pre-match team features.
- `sources/elo_ratings_wc2026.csv` — Elo rating snapshots.
- `build_datasets.py` — rebuilds both final model datasets directly from source files.
- `model_config.py` — authoritative predictor lists and model targets.
- `final_audit.py` — structural checks for row counts, predictors, missing values and overlap.
- `validate_models.py` — chronological holdout evaluation and training-sample diagnostics.
- `cross_validate_models.py` — five expanding-window chronological folds, grouped by date.
- `plot_diagnostics.py` — creates plots from cross-validation predictions.
- `report_statistics.py` — full-sample OLS tables, assumption diagnostics (VIF, Breusch-Pagan, Durbin-Watson, Shapiro-Wilk, RESET, Cook's D) and the report figure.
- `crosscheck_scores.py` — reconciles the 104 target scores with `sources/results.csv`.
- `validation/` — generated metrics, prediction tables and figures.

## Requirements

Python 3.10 or later is recommended. Install dependencies from this folder:

```bash
python -m pip install -r requirements.txt
```

## Reproduction order

Run these commands from the project root, in order:

```bash
python build_datasets.py
python final_audit.py
python validate_models.py
python cross_validate_models.py
python plot_diagnostics.py
python report_statistics.py
python crosscheck_scores.py
```

`final_audit.py` exits with a non-zero status if a structural requirement fails. Do not proceed to reporting if it fails. The final datasets are regenerated from the three source CSVs; there is no manual feature editing step.

## Model definitions

**Model 2.1 target:** `goal_difference` (home goals minus away goals), 104 match rows.

**Model 2.2 target:** `goals_scored`, 208 team-match rows (two rows per match).

Each model has exactly eight predictors. The only exact shared predictor is `is_knockout`, below the maximum of four allowed by the brief. Audit-only history counts and Elo snapshot dates are metadata, not predictors.

## Pre-match cutoff and limitations

Historical match features use source records with dates strictly earlier than the target match date. Same-day records are excluded because the source does not provide kickoff times. Elo features use the latest snapshot dated strictly before the target match date. The pipeline fails rather than silently filling missing model predictors.

These filters reduce temporal leakage, but do not independently prove source authenticity or that each source value was publicly available at the time. The source provenance and this limitation should be acknowledged in the report. The target match file is the source of the observed target scores; its values are not independently authenticated by this pipeline.

## Outputs

Canonical outputs are written under `validation/`:

- `holdout_results.json`
- `cross_validation_results.json`
- `model_2_1_holdout_predictions.csv`, `model_2_2_holdout_predictions.csv`
- `model_2_1_cv_predictions.csv`, `model_2_2_cv_predictions.csv`
- `full_sample_results.json` — full-sample OLS coefficients, robust p-values and diagnostics used in the report.
- `plots/` — actual-versus-predicted and residual plots from cross-validation, plus `report_diagnostics.png`.

Metrics from older configurations should not be mixed with these outputs. Regenerate all results using the run order above before writing the report.
