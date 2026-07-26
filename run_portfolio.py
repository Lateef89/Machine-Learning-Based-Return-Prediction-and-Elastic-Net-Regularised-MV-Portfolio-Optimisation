"""
run_portfolio.py
=================
Builds the six EN-MVP hybrid portfolios (one per prediction model) from the
top-10 candidate pools produced by run_prediction.py, and reproduces:

  - Figure 4 equivalent: mean return / std dev / Sharpe ratio vs. portfolio
    size i in {6,...,10}, at alpha=0.06, lambda=0.99
  - Tables 5/6 equivalents: SA-optimised weight allocation, objective F and
    mixing parameter r*, at lambda=0.01 and lambda=0.99, i=10, averaged over
    10 SA runs
  - Figures 5/6 equivalents: cumulative return of each EN-MVP portfolio vs.
    its equal-weight (1/N) benchmark, at lambda=0.01 and lambda=0.99

Historical covariance and the realised-return series used for the
cumulative-return backtest are computed from the FULL sample (Jan 2018-May
2023) of the candidate stocks' monthly returns; predicted expected returns
(mu) come from each model's forward prediction (run_prediction.py). This is
an in-sample backtest, not an out-of-sample rolling validation -- see
README.md, Limitations.

Run: python3 run_portfolio.py   (after run_prediction.py)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import wide_returns
from portfolio import simulated_annealing, run_sa_multi, sharpe_ratio, equal_weight

OUT = "outputs"
# (alpha_search.py): a train/validation split of the sample scores each
# candidate alpha on realised validation-period Sharpe ratio against the
# equal-weight (1/N) benchmark, and alpha=0.001 attains the highest win rate
# (9 of 12 model/regime combinations beat 1/N) while keeping r* away from
# its degenerate 0 (mean r*=0.262 across combinations at this alpha) and
# producing genuine sparsity (weights are driven below the 0.01 threshold in
# most combinations, unlike alpha=0.06). See alpha_search.py and
# outputs/alpha_search_summary.csv for the full grid and selection criteria.
ALPHA = 0.001
MODELS = ["RF", "AdaBoost", "XGBoost", "SVR", "KNN", "RNN"]
ASSET_COUNTS = [6, 7, 8, 9, 10]
LAMBDAS = [0.01, 0.99]
N_SA_RUNS = 10


def load_pool(model_name):
    df = pd.read_csv(f"{OUT}/top10_detail_{model_name}.csv")
    return df  # columns: Company, pred_return (ranked descending, top 10)


def sigma_for(companies, wide):
    sub = wide[companies]
    return sub.cov().values, sub


def portfolio_series(weights, returns_df):
    """Realised monthly portfolio return series given fixed weights and a
    (month x company) returns DataFrame (NaNs -> 0 contribution for that
    company that month, i.e. treated as not-yet-listed / no data)."""
    R = returns_df.fillna(0.0).values
    return R @ weights


def cumulative_return(monthly_returns):
    return np.cumprod(1 + monthly_returns) - 1


def asset_count_experiment(wide):
    records = []
    for model_name in MODELS:
        pool = load_pool(model_name)
        for i in ASSET_COUNTS:
            top_i = pool.head(i)
            companies = top_i["Company"].tolist()
            mu = top_i["pred_return"].values
            sigma, sub = sigma_for(companies, wide)

            x, r, f = simulated_annealing(mu, sigma, lam=0.99, alpha=ALPHA, seed=123)
            port_ret = portfolio_series(x, sub)
            mean_r, std_r = np.nanmean(port_ret), np.nanstd(port_ret)
            sr = sharpe_ratio(mean_r, std_r)

            records.append({"Model": model_name, "i": i, "mean_return": mean_r,
                             "std_return": std_r, "sharpe": sr, "r_star": r})
            print(f"[asset-count] {model_name:9s} i={i:2d}  mean={mean_r:+.5f} "
                  f"std={std_r:.5f} Sharpe={sr:.3f}")
    df = pd.DataFrame(records)
    df.to_csv(f"{OUT}/fig4_asset_count.csv", index=False)
    return df


def plot_asset_count(df):
    fig, axes = plt.subplots(3, 1, figsize=(8, 11))
    metrics = [("mean_return", "Mean Return"), ("std_return", "Standard Deviation"),
               ("sharpe", "Sharpe Ratio")]
    width = 0.13
    x_pos = np.arange(len(ASSET_COUNTS))
    for ax, (col, title) in zip(axes, metrics):
        for j, model_name in enumerate(MODELS):
            sub = df[df["Model"] == model_name].set_index("i").loc[ASSET_COUNTS, col]
            ax.bar(x_pos + j * width, sub.values, width=width, label=model_name)
        ax.set_xticks(x_pos + width * (len(MODELS) - 1) / 2)
        ax.set_xticklabels([f"i = {i}" for i in ASSET_COUNTS])
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_asset_count_performance.png", dpi=150)
    plt.close()


def risk_regime_tables(wide):
    all_tables = {}
    for lam in LAMBDAS:
        rows = {}
        f_row, r_row = {}, {}
        for model_name in MODELS:
            pool = load_pool(model_name).head(10)
            companies = pool["Company"].tolist()
            mu = pool["pred_return"].values
            sigma, sub = sigma_for(companies, wide)

            x_mean, r_mean, f_mean, xs, rs, fs = run_sa_multi(
                mu, sigma, lam=lam, alpha=ALPHA, n_runs=N_SA_RUNS, seed0=1000)

            rows[f"{model_name}+EN-MVP"] = x_mean
            f_row[f"{model_name}+EN-MVP"] = f_mean
            r_row[f"{model_name}+EN-MVP"] = r_mean
            print(f"[risk-regime lam={lam}] {model_name:9s} r*={r_mean:.4f} F={f_mean:.5f} "
                  f"nonzero={np.sum(x_mean > 0)}")

        table = pd.DataFrame(rows, index=[f"Stock {k+1}" for k in range(10)])
        table.loc["F"] = f_row
        table.loc["r*"] = r_row
        all_tables[lam] = table
        table.to_csv(f"{OUT}/{'table5' if lam == 0.01 else 'table6'}_allocation_lambda{lam}.csv")
    return all_tables


# One fixed, high-contrast colour per model, reused identically across both
# risk-regime figures and shared between a model's EN-MVP line and its 1/N
# benchmark line (distinguished instead by linestyle/marker). This replaces
# the previous approach of letting matplotlib's default 10-colour cycle run
# across all 12 lines, which put unrelated series next to near-duplicate
# hues and made the EN-MVP/1/N pairing for a given model hard to trace.
MODEL_COLORS = dict(zip(MODELS, plt.get_cmap("tab10").colors[:6]))
MODEL_MARKERS = dict(zip(MODELS, ["o", "s", "^", "D", "v", "P"]))


def cumulative_return_experiment(wide, weight_tables):
    figs = {}
    for lam in LAMBDAS:
        table = weight_tables[lam]
        fig, ax = plt.subplots(figsize=(11, 7))
        dates = None
        for model_name in MODELS:
            pool = load_pool(model_name).head(10)
            companies = pool["Company"].tolist()
            sub = wide[companies]
            weights = table[f"{model_name}+EN-MVP"].iloc[:10].values.astype(float)

            en_series = portfolio_series(weights, sub)
            ew_series = portfolio_series(equal_weight(len(companies)), sub)

            en_cum = cumulative_return(en_series)
            ew_cum = cumulative_return(ew_series)
            if dates is None:
                dates = sub.index.values

            color = MODEL_COLORS[model_name]
            marker = MODEL_MARKERS[model_name]
            ax.plot(dates, en_cum, color=color, linestyle="-", linewidth=2.0,
                     marker=marker, markersize=4, markevery=5,
                     label=f"{model_name}+EN-MVP")
            ax.plot(dates, ew_cum, color=color, linestyle="--", linewidth=1.3,
                     alpha=0.65, label=f"{model_name}+1/N")

        ax.set_xlabel("Month index (0 = Jan 2018)")
        ax.set_ylabel("Cumulative Return")
        ax.set_title(f"Cumulative return, lambda={lam} "
                     f"(solid+markers = EN-MVP, dashed = 1/N benchmark; "
                     f"colour identifies the prediction model)")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5, zorder=0)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
        plt.tight_layout()
        fname = f"{OUT}/fig{'5' if lam == 0.01 else '6'}_cumret_lambda{lam}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        figs[lam] = fname
        print(f"[cumret] saved {fname}")
    return figs


def main():
    wide, date_map = wide_returns()
    print(f"wide returns matrix: {wide.shape[0]} months x {wide.shape[1]} companies")

    df_ac = asset_count_experiment(wide)
    plot_asset_count(df_ac)

    weight_tables = risk_regime_tables(wide)

    cumulative_return_experiment(wide, weight_tables)

    print("\nDone. Outputs written to", OUT)


if __name__ == "__main__":
    main()
