# Dataset dictionary

## Model 2.1: `linear_regression_2_1_dataset_enriched.csv`

One row per target match (104 rows). Target: `goal_difference` = home goals minus away goals.

Predictors (exactly eight):
- `home_elo_before`, `away_elo_before`: latest Elo rating snapshot dated before the target date.
- `home_avg_goals_for_before`, `home_avg_goals_against_before`: mean goals scored/conceded in all source-history matches dated before the target date.
- `away_avg_goals_for_before`, `away_avg_goals_against_before`: corresponding away-team historical means.
- `is_knockout`: 1 when the round is not `Group stage`; otherwise 0.
- `home_is_host`: 1 when the home team is USA, Canada or Mexico; otherwise 0.

Metadata columns such as team names, IDs, history counts and Elo snapshot dates are not predictors.

## Model 2.2: `linear_regression_2_2_dataset_enriched.csv`

Two rows per target match (208 rows). Target: `goals_scored` for the team in that row.

Predictors (exactly eight):
- `is_home`: 1 for the listed home team, otherwise 0.
- `is_knockout`: as defined above.
- `team_days_rest`, `opponent_days_rest`: calendar days between target date and each team's most recent earlier historical match.
- `team_recent_points_avg`, `opponent_recent_points_avg`: average 3/1/0 points over each team's latest five earlier historical matches.
- `team_recent_clean_sheet_rate`, `opponent_recent_clean_sheet_rate`: proportion of those latest five earlier matches with zero goals conceded.

The `*_history_matches_used` fields are audit metadata, not predictors. The build script fails if any selected predictor is missing instead of silently imputing a value.
