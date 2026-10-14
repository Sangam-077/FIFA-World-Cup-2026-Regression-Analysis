"""Structural audit for final Objective 2 datasets and model configurations."""
from pathlib import Path
import pandas as pd
from model_config import MODEL_SPECS, MODEL_21_FEATURES, MODEL_22_FEATURES

ROOT = Path(__file__).resolve().parent
all_ok = True
shared = set(MODEL_21_FEATURES) & set(MODEL_22_FEATURES)
print("Exact shared predictors:", sorted(shared))
print("Shared predictor count:", len(shared), "(maximum allowed: 4)")
if len(MODEL_21_FEATURES) != 8 or len(MODEL_22_FEATURES) != 8 or len(shared) > 4:
    all_ok = False
for name, spec in MODEL_SPECS.items():
    path = ROOT / spec["dataset"]
    print(f"\n--- {name} ---")
    if not path.exists():
        print("FAIL: dataset missing", path.name); all_ok = False; continue
    df = pd.read_csv(path)
    required = ["match_id", "date", spec["target"], *spec["features"]]
    missing_columns = sorted(set(required) - set(df.columns))
    checks = {
        "expected_rows": len(df) == spec["expected_rows"],
        "unique_matches": df.match_id.nunique() == 104,
        "no_duplicate_rows": not df.duplicated().any(),
        "required_columns_present": not missing_columns,
    }
    if not missing_columns:
        checks["no_missing_target_or_predictors"] = df[[spec["target"], *spec["features"]]].notna().all().all()
        checks["numeric_predictors"] = all(pd.api.types.is_numeric_dtype(df[c]) for c in spec["features"])
        checks["no_constant_predictors"] = all(df[c].nunique(dropna=True) > 1 for c in spec["features"])
    if name == "model_2_1":
        checks["one_row_per_match"] = df.match_id.is_unique
    else:
        checks["two_rows_per_match"] = len(df) == 208 and df.groupby("match_id").size().eq(2).all()
    for check, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {check}")
        all_ok &= bool(passed)
    if missing_columns:
        print("Missing columns:", missing_columns)
print("\nFINAL AUDIT:", "PASS" if all_ok else "FAIL")
print("This audit checks structure, not source authenticity or real-time availability.")
raise SystemExit(0 if all_ok else 1)
