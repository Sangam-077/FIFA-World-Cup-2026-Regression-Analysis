"""Rebuild both final Objective 2 datasets directly from source files.

Run: python build_datasets.py
Historical features use records strictly earlier than each target date.
Same-day historical records are excluded because kickoff times are unavailable.
"""
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources"

ALIASES = {
    "bosnia herz": "bosnia and herzegovina",
    "bosniaherz": "bosnia and herzegovina",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "south korea": "south korea",
    "ir iran": "iran",
    "iran": "iran",
    "czech republic": "czechia",
    "czechia": "czechia",
    "usa": "united states",
    "united states of america": "united states",
    "cote d ivoire": "ivory coast",
    "côte d ivoire": "ivory coast",
    "ivory coast": "ivory coast",
    "congo dr": "dr congo",
    "democratic republic of the congo": "dr congo",
    "dr congo": "dr congo",
    "turkiye": "turkey",
    "türkiye": "turkey",
    "turkey": "turkey",
    "cabo verde": "cape verde",
    "cape verde": "cape verde",
}


def norm_team(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    return ALIASES.get(text, text)


def load_sources():
    matches = pd.read_csv(SOURCE / "base_matches.csv")
    history = pd.read_csv(SOURCE / "results.csv")
    elo = pd.read_csv(SOURCE / "elo_ratings_wc2026.csv")

    matches["date"] = pd.to_datetime(matches["date"], errors="raise").dt.normalize()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    for col in ("home_score", "away_score"):
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history = history.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    elo["snapshot_date"] = pd.to_datetime(elo["snapshot_date"], errors="coerce").dt.normalize()
    elo["rating"] = pd.to_numeric(elo["rating"], errors="coerce")
    history["home_key"] = history["home_team"].map(norm_team)
    history["away_key"] = history["away_team"].map(norm_team)
    elo["team_key"] = elo["country"].map(norm_team)
    matches["home_key"] = matches["home_team"].map(norm_team)
    matches["away_key"] = matches["away_team"].map(norm_team)
    return matches, history, elo


def team_history_features(history: pd.DataFrame, team_key: str, cutoff: pd.Timestamp) -> dict:
    prior = history.loc[
        (history["date"] < cutoff)
        & ((history["home_key"] == team_key) | (history["away_key"] == team_key))
    ].copy()
    if prior.empty:
        return {"gf": np.nan, "ga": np.nan, "gd": np.nan, "win": np.nan, "points": np.nan, "n": 0}
    is_home = prior["home_key"].eq(team_key).to_numpy()
    gf = np.where(is_home, prior["home_score"], prior["away_score"]).astype(float)
    ga = np.where(is_home, prior["away_score"], prior["home_score"]).astype(float)
    wins = (gf > ga).astype(float)
    draws = (gf == ga).astype(float)
    return {
        "gf": float(gf.mean()), "ga": float(ga.mean()),
        "gd": float((gf - ga).mean()), "win": float(wins.mean()),
        "points": float((3 * wins + draws).mean()), "n": int(len(prior)),
    }


def elo_before(elo: pd.DataFrame, team_key: str, cutoff: pd.Timestamp) -> float:
    prior = elo.loc[(elo["team_key"] == team_key) & (elo["snapshot_date"] < cutoff)]
    if prior.empty:
        return np.nan
    latest_date = prior["snapshot_date"].max()
    latest = prior.loc[prior["snapshot_date"] == latest_date]
    if latest["rating"].nunique(dropna=True) > 1:
        raise ValueError(f"Conflicting Elo ratings for {team_key} on {latest_date.date()}")
    return float(latest["rating"].iloc[-1])


def recent_features(history: pd.DataFrame, team_key: str, cutoff: pd.Timestamp) -> dict:
    prior = history.loc[
        (history["date"] < cutoff)
        & ((history["home_key"] == team_key) | (history["away_key"] == team_key))
    ].copy()
    if prior.empty:
        return {"days_rest": np.nan, "points_avg": np.nan, "clean_sheet_rate": np.nan, "n": 0}
    is_home = prior["home_key"].eq(team_key).to_numpy()
    prior["goals_for"] = np.where(is_home, prior["home_score"], prior["away_score"])
    prior["goals_against"] = np.where(is_home, prior["away_score"], prior["home_score"])
    prior["points"] = np.select(
        [prior["goals_for"] > prior["goals_against"], prior["goals_for"] == prior["goals_against"]],
        [3, 1], default=0,
    )
    # Keep the latest five earlier matches; all matches on the target date are excluded.
    recent = prior.sort_values(["date"]).tail(5)
    last_date = recent["date"].max()
    return {
        "days_rest": int((cutoff - last_date).days),
        "points_avg": float(recent["points"].mean()),
        "clean_sheet_rate": float((recent["goals_against"] == 0).mean()),
        "n": int(len(recent)),
    }


def main():
    matches, history, elo = load_sources()
    if matches["match_id"].duplicated().any():
        raise ValueError("base_matches.csv contains duplicate match IDs")
    if len(matches) != 104:
        raise ValueError(f"Expected 104 target matches, found {len(matches)}")
    calculated_gd = matches["home_goals"] - matches["away_goals"]
    if not calculated_gd.equals(matches["goal_difference"]):
        raise ValueError("goal_difference does not match home_goals - away_goals")

    rows21, rows22 = [], []
    host_teams = {"united states", "canada", "mexico"}
    for m in matches.sort_values(["date", "match_id"]).itertuples(index=False):
        date = m.date
        home_key, away_key = m.home_key, m.away_key
        home_hist = team_history_features(history, home_key, date)
        away_hist = team_history_features(history, away_key, date)
        home_elo = elo_before(elo, home_key, date)
        away_elo = elo_before(elo, away_key, date)
        is_knockout = int(str(m.round).casefold() != "group stage")
        rows21.append({
            "match_id": m.match_id, "date": date.strftime("%Y-%m-%d"), "round": m.round,
            "home_team": m.home_team, "away_team": m.away_team,
            "home_elo_before": home_elo, "away_elo_before": away_elo,
            "home_avg_goals_for_before": home_hist["gf"],
            "home_avg_goals_against_before": home_hist["ga"],
            "away_avg_goals_for_before": away_hist["gf"],
            "away_avg_goals_against_before": away_hist["ga"],
            "is_knockout": is_knockout,
            "goal_difference": int(m.goal_difference),
            "home_is_host": int(home_key in host_teams),
            "home_history_matches_used": home_hist["n"],
            "away_history_matches_used": away_hist["n"],
            "home_elo_snapshot_date": (elo.loc[(elo.team_key == home_key) & (elo.snapshot_date < date), "snapshot_date"].max().strftime("%Y-%m-%d")),
            "away_elo_snapshot_date": (elo.loc[(elo.team_key == away_key) & (elo.snapshot_date < date), "snapshot_date"].max().strftime("%Y-%m-%d")),
        })
        for team, opponent, team_key, opponent_key, is_home, own_hist, opp_hist in [
            (m.home_team, m.away_team, home_key, away_key, 1, home_hist, away_hist),
            (m.away_team, m.home_team, away_key, home_key, 0, away_hist, home_hist),
        ]:
            tf = recent_features(history, team_key, date)
            of = recent_features(history, opponent_key, date)
            rows22.append({
                "match_id": m.match_id, "date": date.strftime("%Y-%m-%d"), "round": m.round,
                "team": team, "opponent": opponent, "is_home": is_home,
                "is_knockout": is_knockout,
                "goals_scored": int(m.home_goals if is_home else m.away_goals),
                "team_days_rest": tf["days_rest"], "opponent_days_rest": of["days_rest"],
                "team_recent_points_avg": tf["points_avg"],
                "opponent_recent_points_avg": of["points_avg"],
                "team_recent_clean_sheet_rate": tf["clean_sheet_rate"],
                "opponent_recent_clean_sheet_rate": of["clean_sheet_rate"],
                "team_history_matches_used": tf["n"],
                "opponent_history_matches_used": of["n"],
            })

    ds21, ds22 = pd.DataFrame(rows21), pd.DataFrame(rows22)
    # Fail loudly rather than silently imputing unavailable pre-match data.
    for name, df, features in [
        ("Model 2.1", ds21, ["home_elo_before", "away_elo_before", "home_avg_goals_for_before", "home_avg_goals_against_before", "away_avg_goals_for_before", "away_avg_goals_against_before", "is_knockout", "home_is_host"]),
        ("Model 2.2", ds22, ["is_home", "is_knockout", "team_days_rest", "opponent_days_rest", "team_recent_points_avg", "opponent_recent_points_avg", "team_recent_clean_sheet_rate", "opponent_recent_clean_sheet_rate"]),
    ]:
        missing = df[features].isna().sum()
        if missing.any():
            raise ValueError(f"{name} has missing predictor values; refusing to write final dataset:\n{missing[missing > 0]}")
    if len(ds21) != 104 or ds21.match_id.nunique() != 104:
        raise ValueError("Model 2.1 must contain 104 unique match rows")
    if len(ds22) != 208 or ds22.match_id.nunique() != 104 or not ds22.groupby("match_id").size().eq(2).all():
        raise ValueError("Model 2.2 must contain exactly two rows for each of 104 matches")

    # Atomic writes prevent a partial CSV if execution fails midway.
    for filename, df in [("linear_regression_2_1_dataset_enriched.csv", ds21), ("linear_regression_2_2_dataset_enriched.csv", ds22)]:
        temp_path = ROOT / (filename + ".tmp")
        df.to_csv(temp_path, index=False)
        temp_path.replace(ROOT / filename)
        print(f"Wrote {filename}: {len(df)} rows")
    print("Dataset build complete. Predictors are defined in model_config.py.")

if __name__ == "__main__":
    main()
