"""Chronological holdout evaluation and training-set diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from model_config import MODEL_SPECS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation"
OUT.mkdir(exist_ok=True)


def rmse(y, pred):
    return float(mean_squared_error(y, pred) ** 0.5)


def main():
    results = {}
    for name, spec in MODEL_SPECS.items():
        df = pd.read_csv(ROOT / spec["dataset"])
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
        dates = sorted(df["date"].unique())
        split = int(np.floor(0.8 * len(dates)))
        test_dates = set(dates[split:])
        train = df.loc[~df["date"].isin(test_dates)].copy()
        test = df.loc[df["date"].isin(test_dates)].copy()
        features, target = spec["features"], spec["target"]
        X_train, X_test = train[features].astype(float), test[features].astype(float)
        y_train, y_test = train[target].astype(float), test[target].astype(float)
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict(X_test)
        baseline = np.full(len(test), y_train.mean())
        test_predictions = test[["match_id", "date"]].copy()
        for col in ("team", "opponent", "home_team", "away_team"):
            if col in test:
                test_predictions[col] = test[col]
        test_predictions["actual"] = y_test.to_numpy()
        test_predictions["predicted"] = pred
        test_predictions["residual"] = y_test.to_numpy() - pred
        test_predictions.to_csv(OUT / f"{name}_holdout_predictions.csv", index=False)

        ols = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit(cov_type="HC3")
        bp = het_breuschpagan(ols.resid, ols.model.exog)
        vif_design = sm.add_constant(X_train, has_constant="add").to_numpy()
        # Include an intercept so VIF reflects centered explanatory-variable relationships.
        vifs = [
            {"feature": col, "VIF": float(variance_inflation_factor(vif_design, i + 1))}
            for i, col in enumerate(features)
        ]
        results[name] = {
            "evaluation": "chronological holdout using final 20% of unique target dates",
            "n": int(len(df)), "unique_matches": int(df["match_id"].nunique()),
            "predictor_count": len(features), "predictors": features,
            "train_n": int(len(train)), "test_n": int(len(test)),
            "train_end_date": str(train["date"].max().date()),
            "test_start_date": str(test["date"].min().date()),
            "test_end_date": str(test["date"].max().date()),
            "test_MAE": float(mean_absolute_error(y_test, pred)),
            "test_RMSE": rmse(y_test, pred), "test_R2": float(r2_score(y_test, pred)),
            "baseline_MAE": float(mean_absolute_error(y_test, baseline)),
            "baseline_RMSE": rmse(y_test, baseline),
            "baseline_R2": float(r2_score(y_test, baseline)),
            "train_adjusted_R2_HC3_OLS": float(ols.rsquared_adj),
            "train_OLS_F_test_pvalue": float(ols.f_pvalue),
            "breusch_pagan_pvalue": float(bp[1]),
            "durbin_watson": float(durbin_watson(ols.resid)),
            "max_VIF": max(item["VIF"] for item in vifs), "vif": vifs,
            "coefficients": {"intercept": float(model.intercept_), **dict(zip(features, map(float, model.coef_)))},
        }
    path = OUT / "holdout_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved canonical holdout results: {path}")

if __name__ == "__main__":
    main()
