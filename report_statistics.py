"""Full-sample OLS statistics, assumption diagnostics and the report figure.

Run after build_datasets.py:  python report_statistics.py
Outputs: validation/full_sample_results.json and validation/plots/report_diagnostics.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from model_config import MODEL_SPECS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation"
(OUT / "plots").mkdir(parents=True, exist_ok=True)


def fit_model(name, spec):
    df = pd.read_csv(ROOT / spec["dataset"])
    X = sm.add_constant(df[spec["features"]].astype(float))
    y = df[spec["target"]].astype(float)
    ols = sm.OLS(y, X).fit()
    # Robust inference: HC3 for one-row-per-match data; match-clustered SEs for Model 2.2
    # because the two team-rows of a match share information (not independent).
    if name == "model_2_2":
        robust = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["match_id"].astype("category").cat.codes})
    else:
        robust = sm.OLS(y, X).fit(cov_type="HC3")
    ci = ols.conf_int()
    z = (df[spec["features"]] - df[spec["features"]].mean()) / df[spec["features"]].std()
    std_beta = sm.OLS((y - y.mean()) / y.std(), sm.add_constant(z)).fit().params
    cooks = ols.get_influence().cooks_distance[0]
    res = {
        "n": int(len(df)), "R2": ols.rsquared, "adj_R2": ols.rsquared_adj,
        "F": float(ols.fvalue), "F_p": float(ols.f_pvalue),
        "df_model": int(ols.df_model), "df_resid": int(ols.df_resid),
        "resid_sd": float(np.sqrt(ols.scale)),
        "target_mean": float(y.mean()), "target_sd": float(y.std()),
        "target_min": float(y.min()), "target_max": float(y.max()),
        "coefficients": {
            k: {"coef": float(ols.params[k]), "ci_lo": float(ci.loc[k, 0]), "ci_hi": float(ci.loc[k, 1]),
                "p": float(ols.pvalues[k]), "p_robust": float(robust.pvalues[k]),
                "std_beta": float(std_beta[k]) if k != "const" else None}
            for k in ols.params.index
        },
        "diagnostics": {
            "max_VIF": float(max(variance_inflation_factor(X.values, i) for i in range(1, X.shape[1]))),
            "breusch_pagan_p": float(het_breuschpagan(ols.resid, X)[1]),
            "durbin_watson": float(durbin_watson(ols.resid)),
            "shapiro_p": float(stats.shapiro(ols.resid)[1]),
            "resid_skew": float(stats.skew(ols.resid)),
            "ramsey_reset_p": float(linear_reset(ols, power=2, use_f=True).pvalue),
            "max_cooks_distance": float(cooks.max()),
            "n_cooks_above_4_over_n": int((cooks > 4 / len(df)).sum()),
        },
    }
    return df, y, ols, res


def main():
    results, fitted = {}, {}
    for name, spec in MODEL_SPECS.items():
        df, y, ols, res = fit_model(name, spec)
        results[name] = res
        fitted[name] = (ols.fittedvalues, ols.resid)
    # Context benchmark: how much does the single Elo-difference predictor explain in Model 2.1?
    d = pd.read_csv(ROOT / MODEL_SPECS["model_2_1"]["dataset"])
    elo = sm.OLS(d["goal_difference"], sm.add_constant(d["home_elo_before"] - d["away_elo_before"])).fit()
    results["model_2_1"]["elo_difference_only_R2"] = float(elo.rsquared)
    (OUT / "full_sample_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Figure: row = model; columns = CV actual vs predicted, residuals vs fitted, residual Q-Q
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))
    titles = {"model_2_1": "Model 2.1 (goal difference)", "model_2_2": "Model 2.2 (goals scored)"}
    for r, name in enumerate(["model_2_1", "model_2_2"]):
        cv = pd.read_csv(OUT / f"{name}_cv_predictions.csv")
        lo, hi = min(cv.actual.min(), cv.predicted.min()), max(cv.actual.max(), cv.predicted.max())
        ax = axes[r, 0]
        ax.scatter(cv.actual, cv.predicted, s=12, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], "--", lw=1, color="grey")
        ax.set_xlabel("Actual (CV)", fontsize=7); ax.set_ylabel(titles[name], fontsize=7, fontweight="bold")
        ax.set_title("Out-of-sample: actual vs predicted" if r == 0 else "", fontsize=7)
        fv, rs = fitted[name]
        ax = axes[r, 1]
        ax.scatter(fv, rs, s=12, alpha=0.7)
        ax.axhline(0, ls="--", lw=1, color="grey")
        ax.set_xlabel("Fitted value", fontsize=7); ax.set_ylabel("Residual", fontsize=7)
        ax.set_title("Residuals vs fitted" if r == 0 else "", fontsize=7)
        ax = axes[r, 2]
        (osm, osr), (slope, icpt, _) = stats.probplot(rs, dist="norm")
        ax.scatter(osm, osr, s=12, alpha=0.7)
        ax.plot(osm, slope * np.asarray(osm) + icpt, "--", lw=1, color="grey")
        ax.set_xlabel("Theoretical quantiles", fontsize=7); ax.set_ylabel("Sample quantiles", fontsize=7)
        ax.set_title("Residual Q-Q plot" if r == 0 else "", fontsize=7)
    for ax in axes.ravel():
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "report_diagnostics.png", dpi=250)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "coefficients"} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
