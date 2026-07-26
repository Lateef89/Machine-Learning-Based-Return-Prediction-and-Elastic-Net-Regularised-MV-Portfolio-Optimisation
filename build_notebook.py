"""
Generates ml_portfolio_pipeline.ipynb: a single self-contained Jupyter
notebook combining data.py, models.py, portfolio.py, run_prediction.py and
run_portfolio.py into one linear, documented, executable pipeline.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
md(r"""
# Machine Learning Return Prediction + Elastic-Net Regularised Mean–Variance
# Portfolio Optimisation via Simulated Annealing — reference implementation

This notebook is a single-file, executable reimplementation of the methodology
described in *"Machine Learning-Based Return Prediction and Elastic-Net
Regularised Mean–Variance Portfolio Optimisation via Simulated Annealing"*,
run against the real dataset supplied by the user
(`clean_sp500_stock_2018_2023.csv`, `clean_sp500_index_2018_2023.csv`).

## Contents
1. Setup and configuration
2. Data loading and feature engineering
3. Six return-prediction models (RF, AdaBoost, XGBoost, SVR, KNN, RNN)
4. Prediction experiments — accuracy by split, best-split summary, top-10 pools
5. Elastic-net mean–variance portfolio model + simulated annealing solver
6. Portfolio experiments — asset-count sensitivity, risk-regime allocations, cumulative return
7. Results summary and comparison to the published paper

Run all cells top to bottom. Expect ~10-20 minutes end to end (the RNN and
the simulated-annealing risk-regime tables are the slow parts).
""")

# ---------------------------------------------------------------------------
md("## 1. Setup and configuration")

code(r"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn

import os
os.makedirs("outputs", exist_ok=True)

RANDOM_STATE = 42
SEED = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

STOCK_CSV = "clean_sp500_stock_2018_2023.csv"
INDEX_CSV = "clean_sp500_index_2018_2023.csv"
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Data loading and feature engineering

**Documented assumption:** the paper states only that "monthly returns of
stocks are used as an input variable". We build a standard, defensible
lag/rolling-statistics feature set: the three preceding months' returns,
a 3-month rolling mean/std of return, month-over-month volume change, and
the S&P 500 index return for that month. Target = that month's realised
return. Companies with fewer than 60 months of history (2020-2022
spin-offs/IPOs) are dropped, mirroring the paper's own stated robustness
filter.
""")

code(r"""
MIN_MONTHS = 60
FEATURE_COLS = [
    "lag1_return", "lag2_return", "lag3_return",
    "roll_mean_3", "roll_std_3",
    "volume_chg", "mkt_return",
]

def _month_index(df):
    return (df["Year"] - df["Year"].min()) * 12 + df["Month"]

def load_raw(stock_path=STOCK_CSV, index_path=INDEX_CSV):
    stock = pd.read_csv(stock_path)
    index = pd.read_csv(index_path)
    stock["t"] = _month_index(stock)
    index["t"] = _month_index(index)
    return stock, index

def build_panel(stock_path=STOCK_CSV, index_path=INDEX_CSV, min_months=MIN_MONTHS):
    stock, index = load_raw(stock_path, index_path)

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
        g["target_return"] = r
        frames.append(g)

    panel = pd.concat(frames, ignore_index=True)
    panel["mkt_return"] = panel["t"].map(mkt)
    panel = panel.dropna(subset=FEATURE_COLS + ["target_return"]).reset_index(drop=True)
    return panel

def latest_feature_snapshot(panel):
    idx = panel.groupby("Company")["t"].idxmax()
    return panel.loc[idx].reset_index(drop=True)

def wide_returns(stock_path=STOCK_CSV, index_path=INDEX_CSV, min_months=MIN_MONTHS):
    stock, _ = load_raw(stock_path, index_path)
    counts = stock.groupby("Company")["t"].nunique()
    keep = counts[counts >= min_months].index
    stock = stock[stock["Company"].isin(keep)].copy()
    wide = stock.pivot_table(index="t", columns="Company", values="mo_return")
    date_map = stock.drop_duplicates("t").set_index("t")[["Year", "Month"]]
    return wide, date_map

panel = build_panel()
print(f"panel: {panel.shape[0]} samples, {panel['Company'].nunique()} companies")
display(panel[FEATURE_COLS + ["target_return"]].describe())
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Six return-prediction models

Hyperparameters follow Table 1 of the revised manuscript as closely as
possible on this 7-feature set. Deviations (e.g. Random Forest's
`max_features: 40`, which exceeds the number of features available here; RNN
epochs reduced from 500 to 30 for tractability) are noted inline.
""")

code(r"""
def make_rf():
    # Table 1: n_estimators=500, max_depth=20, min_samples_split=10,
    # min_samples_leaf=10, max_features=40 (capped to 'sqrt' since 40 > 7 features here).
    return RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_split=10,
        min_samples_leaf=10, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1,
    )

def make_adaboost():
    # Table 1: n_estimators=50, learning_rate=1.
    base = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    try:
        return AdaBoostRegressor(estimator=base, n_estimators=50, learning_rate=1.0,
                                  random_state=RANDOM_STATE)
    except TypeError:
        return AdaBoostRegressor(base_estimator=base, n_estimators=50, learning_rate=1.0,
                                  random_state=RANDOM_STATE)

def make_xgboost():
    # Table 1: n_round=100, max_depth=7, learning_rate=0.01, gamma=2.
    return XGBRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.01, gamma=2,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )

def make_svr():
    # Table 1: C=10, gamma=0.1 (RBF kernel).
    return SVR(C=10, gamma=0.1, kernel="rbf")

def make_knn():
    # Table 1: n_neighbors=3.
    return KNeighborsRegressor(n_neighbors=3)
""")

code(r"""
# --- RNN (PyTorch) --------------------------------------------------------
# Table 1 specifies hidden layers=4, batch_size=128, epochs=500. The three lag
# returns are treated as a length-3 sequence fed to the recurrent unit, with
# the remaining engineered features concatenated to the final hidden state
# before the output layer. Epochs are capped at 30 (not 500) purely for
# tractability in this environment -- a real, documented deviation.
SEQ_FEATURES = ["lag3_return", "lag2_return", "lag1_return"]
AUX_FEATURES = ["roll_mean_3", "roll_std_3", "volume_chg", "mkt_return"]
RNN_EPOCHS = 30
RNN_BATCH = 128
RNN_HIDDEN = 16
RNN_LAYERS = 4

class ReturnRNN(nn.Module):
    def __init__(self, n_aux, hidden=RNN_HIDDEN, layers=RNN_LAYERS):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden, num_layers=layers,
                           batch_first=True, nonlinearity="tanh")
        self.head = nn.Sequential(
            nn.Linear(hidden + n_aux, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, seq, aux):
        out, hN = self.rnn(seq)
        last = out[:, -1, :]
        x = torch.cat([last, aux], dim=1)
        return self.head(x).squeeze(-1)

class RNNRegressor:
    def __init__(self, epochs=RNN_EPOCHS, batch_size=RNN_BATCH, lr=1e-3, seed=RANDOM_STATE):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        torch.manual_seed(seed)
        self.model = None
        self.x_mean = None
        self.x_std = None

    def _split_xy(self, X):
        X = np.asarray(X, dtype=np.float32)
        seq = X[:, :3][:, :, None]
        aux = X[:, 3:]
        return seq, aux

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - self.x_mean) / self.x_std
        seq, aux = self._split_xy(Xn)
        y = np.asarray(y, dtype=np.float32)

        self.model = ReturnRNN(n_aux=aux.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        seq_t = torch.tensor(seq)
        aux_t = torch.tensor(aux)
        y_t = torch.tensor(y)
        n = len(y_t)

        self.model.train()
        for _epoch in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                pred = self.model(seq_t[idx], aux_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.x_mean) / self.x_std
        seq, aux = self._split_xy(Xn)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(seq), torch.tensor(aux)).numpy()
        return pred

def make_rnn():
    return RNNRegressor()

MODEL_FEATURE_ORDER = SEQ_FEATURES + AUX_FEATURES
MODEL_FACTORY = {
    "RF": make_rf, "AdaBoost": make_adaboost, "XGBoost": make_xgboost,
    "SVR": make_svr, "KNN": make_knn, "RNN": make_rnn,
}
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Prediction experiments

Reproduces the paper's **Table 2** (accuracy by train/test split), **Table 3**
(mean/std/variance at each model's empirically best split, from 5 repeated
random splits), and **Table 4** (top-10 predicted stocks per model, ranked
from each model's forward prediction on the most recent available month).

**Documented assumption:** "best split" is chosen empirically here (lowest
test RMSE on this feature set), not hard-coded to the paper's reported
splits, since those were optimal for the original authors' unknown feature
set.
""")

code(r"""
SPLITS = [0.1, 0.2, 0.3, 0.4]
N_REPEATS = 5

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

X_all = panel[MODEL_FEATURE_ORDER].values
y_all = panel["target_return"].values

rows = []
for model_name, factory in MODEL_FACTORY.items():
    for frac in SPLITS:
        Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=frac, random_state=SEED)
        model = factory()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae, mse, rmse = metrics(yte, pred)
        rows.append({"Model": model_name, "Test set fraction": frac,
                     "MAE": mae, "MSE": mse, "RMSE": rmse})
        print(f"[split-table] {model_name:9s} frac={frac:.1f}  MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")

table2 = pd.DataFrame(rows)
table2.to_csv("outputs/table2_prediction_by_split.csv", index=False)
display(table2)
""")

code(r"""
idx = table2.groupby("Model")["RMSE"].idxmin()
best = table2.loc[idx, ["Model", "Test set fraction"]].reset_index(drop=True)
best_split = dict(zip(best["Model"], best["Test set fraction"]))
print("Best split per model (empirical, lowest test RMSE):", best_split)
with open("outputs/best_split_per_model.json", "w") as f:
    json.dump(best_split, f, indent=2)
""")

code(r"""
rows = []
for model_name, factory in MODEL_FACTORY.items():
    frac = best_split[model_name]
    maes, mses, rmses = [], [], []
    for rep in range(N_REPEATS):
        Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=frac, random_state=SEED + rep)
        model = factory()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae, mse, rmse = metrics(yte, pred)
        maes.append(mae); mses.append(mse); rmses.append(rmse)
    for stat_name, arr in [("mean", np.mean), ("sigma", np.std), ("sigma2", np.var)]:
        rows.append({
            "Model": model_name, "best_split_test_fraction": frac, "stat": stat_name,
            "MAE": arr(maes), "MSE": arr(mses), "RMSE": arr(rmses),
        })
    print(f"[summary] {model_name:9s} best_split={frac:.1f}  mean RMSE={np.mean(rmses):.6f}  sigma RMSE={np.std(rmses):.6f}")

table3 = pd.DataFrame(rows)
table3.to_csv("outputs/table3_prediction_summary.csv", index=False)
display(table3)
""")

code(r"""
top_n = 10
latest = latest_feature_snapshot(panel)
X_latest = latest[MODEL_FEATURE_ORDER].values

all_preds = {"Company": latest["Company"].values}
top_tables = {}
for model_name, factory in MODEL_FACTORY.items():
    frac = best_split[model_name]
    Xtr, _, ytr, _ = train_test_split(X_all, y_all, test_size=frac, random_state=SEED)
    model = factory()
    model.fit(Xtr, ytr)
    preds = model.predict(X_latest)
    all_preds[model_name] = preds

    ranked = latest.assign(pred_return=preds).sort_values("pred_return", ascending=False)
    top_tables[model_name] = ranked[["Company", "pred_return"]].head(top_n).reset_index(drop=True)
    print(f"[top10] {model_name}: {list(top_tables[model_name]['Company'])}")

preds_df = pd.DataFrame(all_preds)
preds_df.to_csv("outputs/predictions_latest.csv", index=False)

table4 = pd.DataFrame({name: tbl["Company"].tolist() for name, tbl in top_tables.items()})
table4.to_csv("outputs/table4_top10_stocks.csv", index=False)
for name, tbl in top_tables.items():
    tbl.to_csv(f"outputs/top10_detail_{name}.csv", index=False)

display(table4)
""")

code(r"""
# Sanity check flagged in the write-up: XGBoost's Table-1 hyperparameters
# (learning_rate=0.01, gamma=2) can make its forward predictions collapse to
# an almost-constant value on this feature set -- worth checking before
# trusting its candidate pool.
print(preds_df["XGBoost"].describe())
print("unique XGBoost predictions:", preds_df["XGBoost"].nunique())
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Elastic-net regularised mean–variance portfolio (EN-MVP) + simulated annealing

Implements Eq. (11) of the revised manuscript and the joint `(x, r)`
simulated-annealing solver of Algorithm 1:

$$
\min_{x,r}\ \lambda (x^\top \Sigma x) - (1-\lambda)(\mu^\top x)
+ \alpha\Big(r \sum_i |x_i| + \tfrac{1-r}{2}\sum_i x_i^2\Big)
\quad\text{s.t.}\quad \sum_i x_i = 1,\ 0\le x_i\le 1,\ 0\le r\le 1
$$
""")

code(r"""
def objective(x, r, mu, sigma, lam, alpha):
    var_term = x @ sigma @ x
    ret_term = mu @ x
    l1 = np.sum(np.abs(x))
    l2 = np.sum(x ** 2)
    penalty = alpha * (r * l1 + (1 - r) / 2 * l2)
    return lam * var_term - (1 - lam) * ret_term + penalty

def _project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    return w

def simulated_annealing(mu, sigma, lam, alpha, n_assets=None, seed=0,
                         T0=1000.0, Tf=0.01, beta=0.99, inner_iter=100,
                         step_x=0.05, step_r=0.05):
    rng = np.random.default_rng(seed)
    n = len(mu) if n_assets is None else n_assets

    x = _project_simplex(rng.random(n))
    r = rng.uniform(0, 1)
    f = objective(x, r, mu, sigma, lam, alpha)
    x_best, r_best, f_best = x.copy(), r, f

    T = T0
    while T > Tf:
        for _ in range(inner_iter):
            x_new = _project_simplex(x + rng.normal(0, step_x, size=n))
            r_new = np.clip(r + rng.normal(0, step_r), 0, 1)
            f_new = objective(x_new, r_new, mu, sigma, lam, alpha)
            df = f_new - f
            if df <= 0 or rng.random() < np.exp(-df / T):
                x, r, f = x_new, r_new, f_new
                if f < f_best:
                    x_best, r_best, f_best = x.copy(), r, f
        T *= beta

    x_report = x_best.copy()
    x_report[x_report < 0.01] = 0.0
    if x_report.sum() > 0:
        x_report = x_report / x_report.sum()

    return x_report, r_best, f_best

def sharpe_ratio(mean_return, std_return, rf=0.0):
    if std_return == 0:
        return np.nan
    return (mean_return - rf) / std_return

def equal_weight(n):
    return np.ones(n) / n

def run_sa_multi(mu, sigma, lam, alpha, n_runs=10, seed0=0, **kwargs):
    xs, rs, fs = [], [], []
    for k in range(n_runs):
        x, r, f = simulated_annealing(mu, sigma, lam, alpha, seed=seed0 + k, **kwargs)
        xs.append(x); rs.append(r); fs.append(f)
    x_mean = np.mean(xs, axis=0)
    if x_mean.sum() > 0:
        x_mean = x_mean / x_mean.sum()
    return x_mean, float(np.mean(rs)), float(np.mean(fs)), np.array(xs), np.array(rs), np.array(fs)
""")

code(r"""
# Quick sanity check on synthetic data before trusting it on real portfolios.
rng = np.random.default_rng(0)
n = 10
mu_syn = rng.uniform(-0.02, 0.05, n)
A = rng.normal(0, 0.05, (n, 20))
sigma_syn = A @ A.T / 20 + np.eye(n) * 1e-4

x, r, f = simulated_annealing(mu_syn, sigma_syn, lam=0.99, alpha=0.06, seed=1)
print("SA weights:", np.round(x, 4), "sum:", x.sum())
print("SA r*:", r, " SA objective:", f)
print("Equal-weight objective (for comparison):",
      objective(equal_weight(n), 0.5, mu_syn, sigma_syn, 0.99, 0.06))
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Portfolio experiments

Builds the six EN-MVP hybrid portfolios from the top-10 candidate pools
above and reproduces:
- **Figure 4** equivalent: mean return / std dev / Sharpe ratio vs. portfolio
  size *i* ∈ {6,...,10}, at α=0.06, λ=0.99
- **Tables 5/6** equivalents: SA-optimised weight allocation, objective *F*
  and mixing parameter *r\**, at λ=0.01 and λ=0.99, *i*=10, averaged over 10 SA runs
- **Figures 5/6** equivalents: cumulative return of each EN-MVP portfolio vs.
  its equal-weight (1/N) benchmark

**Documented assumption:** the covariance matrix and the realised-return
series used for the cumulative-return backtest are both computed from the
*full* Jan 2018-May 2023 sample of the candidate stocks — this is an
in-sample backtest, not an out-of-sample rolling validation (see Section 7.2,
Limitations, of the revised manuscript).
""")

code(r"""
ALPHA = 0.06   # regularisation strength, following Yen & Yen (2014)'s reported optimum
MODELS = ["RF", "AdaBoost", "XGBoost", "SVR", "KNN", "RNN"]
ASSET_COUNTS = [6, 7, 8, 9, 10]
LAMBDAS = [0.01, 0.99]
N_SA_RUNS = 10

wide, date_map = wide_returns()
print(f"wide returns matrix: {wide.shape[0]} months x {wide.shape[1]} companies")

def load_pool(model_name):
    return pd.read_csv(f"outputs/top10_detail_{model_name}.csv")

def sigma_for(companies, wide):
    sub = wide[companies]
    return sub.cov().values, sub

def portfolio_series(weights, returns_df):
    R = returns_df.fillna(0.0).values
    return R @ weights

def cumulative_return(monthly_returns):
    return np.cumprod(1 + monthly_returns) - 1
""")

code(r"""
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
        print(f"[asset-count] {model_name:9s} i={i:2d}  mean={mean_r:+.5f} std={std_r:.5f} Sharpe={sr:.3f}")

fig4_df = pd.DataFrame(records)
fig4_df.to_csv("outputs/fig4_asset_count.csv", index=False)
display(fig4_df.pivot(index="Model", columns="i", values="sharpe"))
""")

code(r"""
fig, axes = plt.subplots(3, 1, figsize=(8, 11))
metric_cols = [("mean_return", "Mean Return"), ("std_return", "Standard Deviation"),
               ("sharpe", "Sharpe Ratio")]
width = 0.13
x_pos = np.arange(len(ASSET_COUNTS))
for ax, (col, title) in zip(axes, metric_cols):
    for j, model_name in enumerate(MODELS):
        sub = fig4_df[fig4_df["Model"] == model_name].set_index("i").loc[ASSET_COUNTS, col]
        ax.bar(x_pos + j * width, sub.values, width=width, label=model_name)
    ax.set_xticks(x_pos + width * (len(MODELS) - 1) / 2)
    ax.set_xticklabels([f"i = {i}" for i in ASSET_COUNTS])
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig("outputs/fig4_asset_count_performance.png", dpi=150)
plt.show()
""")

code(r"""
weight_tables = {}
for lam in LAMBDAS:
    rows_w, f_row, r_row = {}, {}, {}
    for model_name in MODELS:
        pool = load_pool(model_name).head(10)
        companies = pool["Company"].tolist()
        mu = pool["pred_return"].values
        sigma, sub = sigma_for(companies, wide)

        x_mean, r_mean, f_mean, xs, rs, fs = run_sa_multi(
            mu, sigma, lam=lam, alpha=ALPHA, n_runs=N_SA_RUNS, seed0=1000)

        rows_w[f"{model_name}+EN-MVP"] = x_mean
        f_row[f"{model_name}+EN-MVP"] = f_mean
        r_row[f"{model_name}+EN-MVP"] = r_mean
        print(f"[risk-regime lam={lam}] {model_name:9s} r*={r_mean:.4f} F={f_mean:.5f} nonzero={np.sum(x_mean > 0)}")

    table = pd.DataFrame(rows_w, index=[f"Stock {k+1}" for k in range(10)])
    table.loc["F"] = f_row
    table.loc["r*"] = r_row
    weight_tables[lam] = table
    fname = f"outputs/{'table5' if lam == 0.01 else 'table6'}_allocation_lambda{lam}.csv"
    table.to_csv(fname)

display(weight_tables[0.01])
""")

code(r"""
display(weight_tables[0.99])
""")

code(r"""
for lam in LAMBDAS:
    table = weight_tables[lam]
    fig, ax = plt.subplots(figsize=(9, 6))
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

        ax.plot(dates, en_cum, marker="o", markersize=2, label=f"{model_name}+EN-MVP")
        ax.plot(dates, ew_cum, linestyle="--", linewidth=0.8, alpha=0.6, label=f"{model_name}+1/N")

    ax.set_xlabel("Month index (0 = Jan 2018)")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Cumulative return, lambda={lam}")
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    fname = f"outputs/fig{'5' if lam == 0.01 else '6'}_cumret_lambda{lam}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"[cumret] saved {fname}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Results summary and comparison to the published paper

""")

nb["cells"] = cells

# Ensure a sane kernelspec so the notebook opens cleanly in Jupyter/VS Code/etc.
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

with open("ml_portfolio_pipeline.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written: ml_portfolio_pipeline.ipynb  (", len(cells), "cells )")
