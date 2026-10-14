"""Reconcile the 104 target-match scores in sources/base_matches.csv with sources/results.csv.

Run: python crosscheck_scores.py
Reports exact matches, orientation-only differences (home/away listed the other way round)
and genuine score mismatches.
"""
from pathlib import Path
import pandas as pd
from build_datasets import norm_team

ROOT = Path(__file__).resolve().parent
base = pd.read_csv(ROOT / "sources" / "base_matches.csv")
hist = pd.read_csv(ROOT / "sources" / "results.csv")
wc = hist[(hist.tournament == "FIFA World Cup") & (hist.date >= base.date.min()) & (hist.date <= base.date.max())].copy()
wc["hk"], wc["ak"] = wc.home_team.map(norm_team), wc.away_team.map(norm_team)
base["hk"], base["ak"] = base.home_team.map(norm_team), base.away_team.map(norm_team)

exact = flipped = mismatch = missing = 0
for m in base.itertuples():
    same = wc[(wc.date == m.date) & (wc.hk == m.hk) & (wc.ak == m.ak)]
    swap = wc[(wc.date == m.date) & (wc.hk == m.ak) & (wc.ak == m.hk)]
    if len(same):
        ok = (same.home_score.iloc[0] == m.home_goals) and (same.away_score.iloc[0] == m.away_goals)
        exact += ok; mismatch += (not ok)
    elif len(swap):
        ok = (swap.home_score.iloc[0] == m.away_goals) and (swap.away_score.iloc[0] == m.home_goals)
        flipped += ok; mismatch += (not ok)
    else:
        missing += 1
print(f"Target matches: {len(base)} | exact: {exact} | same score, home/away order swapped: {flipped} | "
      f"score mismatches: {mismatch} | not found: {missing}")
