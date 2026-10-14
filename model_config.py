"""Single source of truth for Objective 2 model definitions."""
MODEL_21_FEATURES = [
    "home_elo_before",
    "away_elo_before",
    "home_avg_goals_for_before",
    "home_avg_goals_against_before",
    "away_avg_goals_for_before",
    "away_avg_goals_against_before",
    "is_knockout",
    "home_is_host",
]
MODEL_22_FEATURES = [
    "is_home",
    "is_knockout",
    "team_days_rest",
    "opponent_days_rest",
    "team_recent_points_avg",
    "opponent_recent_points_avg",
    "team_recent_clean_sheet_rate",
    "opponent_recent_clean_sheet_rate",
]
MODEL_SPECS = {
    "model_2_1": {
        "dataset": "linear_regression_2_1_dataset_enriched.csv",
        "target": "goal_difference",
        "features": MODEL_21_FEATURES,
        "expected_rows": 104,
    },
    "model_2_2": {
        "dataset": "linear_regression_2_2_dataset_enriched.csv",
        "target": "goals_scored",
        "features": MODEL_22_FEATURES,
        "expected_rows": 208,
    },
}
