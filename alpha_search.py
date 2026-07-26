"""
alpha_search.py
================
Resolves the "uncalibrated regularisation strength" limitation flagged in the
manuscript: alpha=0.06 was carried over from Yen & Yen (2014) for a different
dataset and, at that magnitude, dominates the risk-return trade-off terms of
Eq. (11) at this sample's return/covariance scale, driving the SA-optimised
mixing parameter r* to ~0 for every predictor/regime combination.

This script performs the small validation grid search the manuscript's
Limitations/Future work sections call for: a train/validation split of the
65-month sample is used to estimate the covariance matrix out of the data
window used to score candidate alpha values, so the selected alpha is chosen
by realised validation-period performance rather than in-sample fit.

  - Train window: months 0-47 (Jan 2018-Dec 2021, 48 months) -> Sigma_train
  - Validation window: months 48-64 (Jan 2022-May 2023, 17 months) -> scoring
  - mu (predicted expected return) is the same forward snapshot used
    throughout the manuscript (Section 4.2); it is not re-estimated per
    window, consistent with the rest of the pipeline (see Limitations).

For each candidate alpha, weights are solved by the same joint (x,r)
simulated-annealing algorithm (Algorithm 1) using Sigma_train and the
model's mu, for all 6 predictors x 2 risk regimes (lambda in {0.01, 0.99}),
and scored on the validation window's realised Sharpe ratio and cumulative
return against the equal-weight (1/N) benchmark on the same candidate pool.

Search-phase SA runs use reduced settings (n_runs=3, inner_iter=50 instead
of the manuscript's 10 runs / 100 inner iterations) purely for tractability
in a 2-core environment; this is a documented compute deviation, exactly
like the RF max_features / RNN epoch deviations in Table 1. The final
alpha's reported tables/figures (produced by run_portfolio.py after this
script) use the manuscript's full SA settings.

Run: python3 alpha_search.py   (~10-15 minutes on 2 cores; writes progress
incrementally to outputs/alpha_search_log.csv so it can be monitored/resumed)
"""
import time
import numpy as np
import pandas as pd

from data import wide_returns
from portfolio import simulated_annealing, sharpe_ratio, equal_weight

OUT = "outputs"
MODELS = ["RF", "AdaBoost", "XGBoost", "SVR", "KNN", "RNN"]
LAMBDAS = [0.01, 0.99]
ALPHA_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.06]
N_SEARCH_RUNS = 3
SEARCH_KWARGS = dict(inner_iter=50)

TRAIN_END = 48   # months 0..47 -> covariance estimation window
# validation window: months 48..64 (17 months) -> scoring window


def load_pool(model_name):
    return pd.read_csv(f"{OUT}/top10_detail_{model_name}.csv")


def portfolio_series(weights, returns_df):
    R = returns_df.fillna(0.0).values
    return R @ weights


def cumulative_return(monthly_returns):
    return np.cumprod(1 + monthly_returns) - 1


def main():
    wide, date_map = wide_returns()
    n_months = wide.shape[0]
    print(f"wide returns matrix: {n_months} months x {wide.shape[1]} companies")
    print(f"train window: months 0-{TRAIN_END - 1} ({TRAIN_END} months); "
          f"validation window: months {TRAIN_END}-{n_months - 1} ({n_months - TRAIN_END} months)")

    records = []
    t_start = time.time()
    for alpha in ALPHA_GRID:
        for model_name in MODELS:
            pool = load_pool(model_name).head(10)
            companies = pool["Company"].tolist()
            mu = pool["pred_return"].values
            sub_train = wide[companies].iloc[:TRAIN_END]
            sub_val = wide[companies].iloc[TRAIN_END:]
            sigma_train = sub_train.cov().values

            for lam in LAMBDAS:
                xs, rs = [], []
                for k in range(N_SEARCH_RUNS):
                    x, r, f = simulated_annealing(
                        mu, sigma_train, lam=lam, alpha=alpha,
                        seed=2000 + k, **SEARCH_KWARGS)
                    xs.append(x); rs.append(r)
                x_mean = np.mean(xs, axis=0)
                if x_mean.sum() > 0:
                    x_mean = x_mean / x_mean.sum()
                r_mean = float(np.mean(rs))
                n_zero = int(np.sum(x_mean < 0.01))

                en_val = portfolio_series(x_mean, sub_val)
                ew_val = portfolio_series(equal_weight(len(companies)), sub_val)
                en_mean, en_std = np.nanmean(en_val), np.nanstd(en_val)
                ew_mean, ew_std = np.nanmean(ew_val), np.nanstd(ew_val)
                en_sharpe = sharpe_ratio(en_mean, en_std)
                ew_sharpe = sharpe_ratio(ew_mean, ew_std)
                en_cum = cumulative_return(en_val)[-1]
                ew_cum = cumulative_return(ew_val)[-1]

                rec = dict(alpha=alpha, model=model_name, lam=lam,
                           r_star=r_mean, n_zero_weights=n_zero,
                           val_sharpe_en=en_sharpe, val_sharpe_ew=ew_sharpe,
                           val_cumret_en=en_cum, val_cumret_ew=ew_cum,
                           beats_1N_sharpe=int(en_sharpe > ew_sharpe),
                           beats_1N_cumret=int(en_cum > ew_cum))
                records.append(rec)
                elapsed = time.time() - t_start
                print(f"[{elapsed:6.0f}s] alpha={alpha:<8} {model_name:9s} lam={lam:<5} "
                      f"r*={r_mean:.4f} zero_w={n_zero} "
                      f"val_Sharpe(EN)={en_sharpe:+.3f} val_Sharpe(1/N)={ew_sharpe:+.3f} "
                      f"beats1N={rec['beats_1N_sharpe']}")

                pd.DataFrame(records).to_csv(f"{OUT}/alpha_search_log.csv", index=False)

    df = pd.DataFrame(records)
    summary = df.groupby("alpha").agg(
        mean_val_sharpe_en=("val_sharpe_en", "mean"),
        mean_val_sharpe_ew=("val_sharpe_ew", "mean"),
        mean_r_star=("r_star", "mean"),
        mean_zero_weights=("n_zero_weights", "mean"),
        win_rate_sharpe=("beats_1N_sharpe", "mean"),
        win_rate_cumret=("beats_1N_cumret", "mean"),
    ).reset_index()
    summary.to_csv(f"{OUT}/alpha_search_summary.csv", index=False)
    print("\n=== Summary by alpha (validation window, months 48-64) ===")
    print(summary.to_string(index=False))

    best = summary.loc[summary["mean_val_sharpe_en"].idxmax(), "alpha"]
    print(f"\nRecommended alpha (max mean validation Sharpe ratio across "
          f"12 model/regime combinations): {best}")


if __name__ == "__main__":
    main()
