"""Five-fold expanding-window cross-validation grouped by match date."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from model_config import MODEL_SPECS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation"
OUT.mkdir(exist_ok=True)
N_SPLITS = 5


def rmse(y, pred):
    return float(mean_squared_error(y, pred) ** 0.5)


def main():
    results = {}
    for name, spec in MODEL_SPECS.items():
        df = pd.read_csv(ROOT / spec["dataset"])
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
        dates = np.array(sorted(df["date"].unique()))
        splitter = TimeSeriesSplit(n_splits=N_SPLITS)
        folds, predictions = [], []
        for fold, (train_i, test_i) in enumerate(splitter.split(dates), start=1):
            train_dates, test_dates = dates[train_i], dates[test_i]
            train = df.loc[df.date.isin(train_dates)]
            test = df.loc[df.date.isin(test_dates)]
            features, target = spec["features"], spec["target"]
            model = LinearRegression().fit(train[features].astype(float), train[target].astype(float))
            actual = test[target].astype(float).to_numpy()
            predicted = model.predict(test[features].astype(float))
            baseline = np.full(len(test), train[target].mean())
            for j, (_, row) in enumerate(test.iterrows()):
                record = {
                    "model": name, "fold": fold, "match_id": row["match_id"],
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "actual": float(actual[j]), "predicted": float(predicted[j]),
                    "residual": float(actual[j] - predicted[j]),
                }
                for col in ("team", "opponent", "home_team", "away_team"):
                    if col in row.index:
                        record[col] = row[col]
                predictions.append(record)
            folds.append({
                "fold": fold, "train_rows": int(len(train)), "test_rows": int(len(test)),
                "train_end": str(train.date.max().date()),
                "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date()),
                "MAE": float(mean_absolute_error(actual, predicted)), "RMSE": rmse(actual, predicted),
                "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
                "baseline_MAE": float(mean_absolute_error(actual, baseline)),
                "baseline_RMSE": rmse(actual, baseline),
            })
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(OUT / f"{name}_cv_predictions.csv", index=False)
        y, p = pred_df.actual.to_numpy(), pred_df.predicted.to_numpy()
        baseline_values = []
        for fold in folds:
            # Pool fold-specific baseline MAE/RMSE with test-size weighting below.
            baseline_values.append(fold)
        pooled_baseline_mae = float(np.average([f["baseline_MAE"] for f in folds], weights=[f["test_rows"] for f in folds]))
        pooled_baseline_rmse = float(np.sqrt(np.average([f["baseline_RMSE"]**2 for f in folds], weights=[f["test_rows"] for f in folds])))
        results[name] = {
            "predictor_count": len(spec["features"]), "predictors": spec["features"],
            "folds": folds, "pooled_MAE": float(mean_absolute_error(y, p)),
            "pooled_RMSE": rmse(y, p), "pooled_R2": float(r2_score(y, p)),
            "pooled_baseline_MAE_weighted_fold_mean": pooled_baseline_mae,
            "pooled_baseline_RMSE_weighted_fold_root_mean_square": pooled_baseline_rmse,
            "evaluated_rows": int(len(pred_df)),
        }
    path = OUT / "cross_validation_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved canonical cross-validation results: {path}")

if __name__ == "__main__":
    main()
