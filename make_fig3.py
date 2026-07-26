import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import wide_returns

MODELS = ["RF", "AdaBoost", "XGBoost", "SVR", "KNN", "RNN"]

per_model = {}
union = set()
for m in MODELS:
    df = pd.read_csv(f"outputs/top10_detail_{m}.csv")
    companies = df["Company"].tolist()
    per_model[m] = companies
    union.update(companies)
union = sorted(union)
print(f"Union of top-10 stocks across all 6 models: {len(union)} unique companies")
print(union)

wide, date_map = wide_returns()

dates = date_map.loc[wide.index]
labels = [f"{int(y)}-{int(mo):02d}" for y, mo in zip(dates["Year"], dates["Month"])]
tick_idx = np.linspace(0, len(wide.index) - 1, 8, dtype=int)

# Shared y-limits (with a small margin) across all panels so that the vertical
# scale is directly comparable model-to-model, and so within-panel variation
# is not exaggerated by an auto-scaled axis.
all_vals = wide[union].values
ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
pad = 0.05 * (ymax - ymin)
ylim = (ymin - pad, ymax + pad)

tab10 = plt.get_cmap("tab10").colors
markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
for ax, model_name in zip(axes.flat, MODELS):
    companies = per_model[model_name]
    for i, c in enumerate(companies):
        ax.plot(
            wide.index, wide[c].values,
            color=tab10[i % 10], marker=markers[i % 10], markersize=3,
            markevery=6, linewidth=1.4, alpha=0.9, label=c,
        )
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_title(f"{model_name} top-10 pool", fontsize=11, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.set_xticks(wide.index[tick_idx])
    ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)
    ax.grid(alpha=0.25, linewidth=0.5)

for ax in axes[:, 0]:
    ax.set_ylabel("Monthly return", fontsize=9)
for ax in axes[-1, :]:
    ax.set_xlabel("Month", fontsize=9)

fig.suptitle(
    f"Monthly returns of each model's top-10 predicted-return candidate pool "
    f"(union across all six models: {len(union)} distinct companies)",
    fontsize=13,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/fig3_monthly_returns.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved outputs/fig3_monthly_returns.png")
