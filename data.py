"""
data.py
=======
Loads the two datasets supplied by the user:

  - clean_sp500_stock_2018_2023.csv   (per-company monthly OHLCV + mo_return, Jan 2018-May 2023)
  - clean_sp500_index_2018_2023.csv   (S&P 500 index monthly OHLCV + mo_return, same period)

and builds a supervised-learning panel for next-month stock-return prediction.
"""
import numpy as np
import pandas as pd

STOCK_CSV = "clean_sp500_stock_2018_2023.csv"
INDEX_CSV = "clean_sp500_index_2018_2023.csv"

MIN_MONTHS = 60          # drop companies with less than this many months of history
MIN_HISTORY_FOR_FEATURES = 3  # need 3 trailing months to build lag/rolling features

FEATURE_COLS = [
    "lag1_return", "lag2_return", "lag3_return",
    "roll_mean_3", "roll_std_3",
    "volume_chg", "mkt_return",
]


def _month_index(df):
    """Integer month index (0 = Jan 2018), monotonically increasing, for sorting/lagging."""
    return (df["Year"] - df["Year"].min()) * 12 + df["Month"]


def load_raw(stock_path=STOCK_CSV, index_path=INDEX_CSV):
    stock = pd.read_csv(stock_path)
    index = pd.read_csv(index_path)
    stock["t"] = _month_index(stock)
    index["t"] = _month_index(index)
    return stock, index


def build_panel(stock_path=STOCK_CSV, index_path=INDEX_CSV, min_months=MIN_MONTHS):
    """Returns a tidy DataFrame with one row per (company, month) containing the
    feature columns in FEATURE_COLS and the prediction target `target_return`
    (next month's mo_return for that company)."""
    stock, index = load_raw(stock_path, index_path)

    # Robustness filter: drop companies with short listing history (spin-offs/IPOs
    # during the sample), mirroring the paper's stated robustness step.
    counts = stock.groupby("Company")["t"].nunique()
    keep = counts[counts >= min_months].index
    stock = stock[stock["Company"].isin(keep)].copy()

    stock = stock.sort_values(["Company", "t"]).reset_index(drop=True)

    mkt = index.set_index("t")["mo_return"].rename("mkt_return")

    frames = []
    for company, g in stock.groupby("Company", sort=False):
        g = g.sort_values("t").reset_index(drop=True)
        r = g["mo_return"]
        g["lag1_return"] = r.shift(1)
        g["lag2_return"] = r.shift(2)
        g["lag3_return"] = r.shift(3)
        g["roll_mean_3"] = r.shift(1).rolling(3).mean()
        g["roll_std_3"] = r.shift(1).rolling(3).std()
        g["volume_chg"] = g["Volume"].pct_change()
        g["target_return"] = r  # this month's return is the "next-month" target relative to lag1..lag3
        frames.append(g)

    panel = pd.concat(frames, ignore_index=True)
    panel["mkt_return"] = panel["t"].map(mkt)

    panel = panel.dropna(subset=FEATURE_COLS + ["target_return"]).reset_index(drop=True)
    return panel


def latest_feature_snapshot(panel):
    """For each company, the most recent row (used to predict *forward* returns
    for candidate-pool ranking, i.e. an out-of-sample prediction with no
    realised target available)."""
    idx = panel.groupby("Company")["t"].idxmax()
    return panel.loc[idx].reset_index(drop=True)


def wide_returns(stock_path=STOCK_CSV, index_path=INDEX_CSV, min_months=MIN_MONTHS):
    """Company x month matrix of realised mo_return, used for covariance
    estimation and for compounding cumulative portfolio returns."""
    stock, _ = load_raw(stock_path, index_path)
    counts = stock.groupby("Company")["t"].nunique()
    keep = counts[counts >= min_months].index
    stock = stock[stock["Company"].isin(keep)].copy()
    wide = stock.pivot_table(index="t", columns="Company", values="mo_return")
    date_map = stock.drop_duplicates("t").set_index("t")[["Year", "Month"]]
    return wide, date_map


if __name__ == "__main__":
    panel = build_panel()
    print("panel shape:", panel.shape)
    print("companies:", panel["Company"].nunique())
    print(panel[FEATURE_COLS + ["target_return"]].describe())
