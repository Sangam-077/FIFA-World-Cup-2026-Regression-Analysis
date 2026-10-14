"""Create actual-vs-predicted and residual plots from CV predictions."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {
    "model_2_1": "Model 2.1 — Goal Difference",
    "model_2_2": "Model 2.2 — Goals Scored",
}
for name, title in MODELS.items():
    path = ROOT / "validation" / f"{name}_cv_predictions.csv"
    df = pd.read_csv(path)
    lo = min(df.actual.min(), df.predicted.min())
    hi = max(df.actual.max(), df.predicted.max())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.actual, df.predicted, alpha=0.8)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set(xlabel="Actual", ylabel="Predicted", title=f"{title}: Actual vs Predicted")
    fig.tight_layout(); fig.savefig(OUT / f"{name}_actual_vs_predicted.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.predicted, df.residual, alpha=0.8)
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.set(xlabel="Predicted", ylabel="Residual (actual − predicted)", title=f"{title}: Residuals vs Predicted")
    fig.tight_layout(); fig.savefig(OUT / f"{name}_residuals.png", dpi=220); plt.close(fig)
print(f"Saved diagnostic plots to {OUT}")
