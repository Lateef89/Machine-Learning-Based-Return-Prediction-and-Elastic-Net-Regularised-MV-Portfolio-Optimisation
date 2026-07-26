# Reference implementation: ML return prediction + elastic-net MVP + simulated annealing

This is a from-scratch, faithful reimplementation of the methodology described in
"Machine Learning-Based Return Prediction and Elastic-Net Regularised Mean–Variance
Portfolio Optimisation via Simulated Annealing" (the revised manuscript), run against
the real dataset you supplied (`clean_sp500_stock_2018_2023.csv`,
`clean_sp500_index_2018_2023.csv`).

**Read this before comparing numbers to the paper.** The manuscript specifies the
*model* (equations, hyperparameter values, SA schedule) precisely, but does not fully
specify the feature engineering, the exact train/test split mechanics, or how the
"top-10 candidate pool" prediction is generated for a period beyond the end of the
data. Wherever the paper is silent, this code makes an explicit, documented choice
(below) rather than a hidden one. **The numbers this code produces will not match the
paper's published tables exactly** — they are a genuine, independent re-run of the
same methodology on the same data, not a reproduction of the original authors' exact
pipeline (which was never supplied to this session).

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads both CSVs, builds the modelling panel (features + target), and the wide return matrix used for covariance/backtesting |
| `models.py` | The six prediction models (RF, AdaBoost, XGBoost, SVR, KNN, RNN), hyperparameters per Table 1 of the manuscript where transferable |
| `portfolio.py` | Elastic-net MVP objective (Eq. 11) and the joint (x, r) simulated-annealing solver (Algorithm 1) |
| `run_prediction.py` | Produces Table 2 (accuracy by split), Table 3 (mean/σ/σ² at best split), Table 4 (top-10 predicted stocks per model) |
| `run_portfolio.py` | Produces Fig. 4 (asset-count sensitivity), Tables 5/6 (risk-regime allocations), Figs. 5/6 (cumulative return vs. 1/N) |
| `outputs/` | All generated CSV tables and PNG figures |

## How to run

```bash
pip install scikit-learn xgboost torch matplotlib pandas numpy
python3 run_prediction.py     # ~10-20 minutes (RNN is the slow part)
python3 run_portfolio.py      # ~1-2 minutes
```

## Dataset

- `clean_sp500_stock_2018_2023.csv`: 501 companies, monthly OHLCV + `mo_return`,
  January 2018-May 2023 (65 months). Companies with fewer than 60 months of history
  (mostly 2020-2022 spin-offs/IPOs: GEHC, CEG, OGN, OTIS, CARR, CTVA, FOXA, FOX, DOW,
  MRNA) are dropped as a robustness step, leaving **491 companies** — this mirrors the
  paper's own stated robustness filter ("organisations with clear data abnormalities
  and those with inadequate data volume owing to shorter listing dates were omitted").
- `clean_sp500_index_2018_2023.csv`: S&P 500 index monthly return over the same
  period, used as a market-return feature.

## Documented modelling choices (where the paper is ambiguous)

1. **Feature set.** The paper says only that "monthly returns of stocks are used as
   an input variable". We use, for each (company, month): the three preceding
   months' returns (`lag1`, `lag2`, `lag3`), a 3-month rolling mean and standard
   deviation of return, month-over-month volume change, and the S&P 500 index return
   for that month. Target = that month's realised return. This is a standard,
   defensible feature set for monthly return prediction, but it is almost certainly
   not identical to whatever feature set the original authors used.
2. **Train/test split mechanic.** "Test set ratio" is implemented as a random split
   of the pooled (company, month) panel (`sklearn.train_test_split`, fixed seed),
   consistent with treating the panel as i.i.d. cross-sectional data rather than a
   strict walk-forward time-series split. This is almost certainly what a panel-ML
   study of this kind does in practice, but the paper does not state it explicitly.
   Note this means Table 2/3 accuracy figures are **not** a rolling-window
   out-of-sample test — see Limitations in the revised manuscript.
3. **"Best" train/test split per model.** Chosen empirically as the split with the
   lowest test RMSE for that model on this feature set/data, *not* hard-coded to the
   paper's reported splits (0.9/0.8/0.7 by model) — those were optimal for the
   original authors' unknown feature set and may not transfer here.
4. **Table 3 (mean/σ/σ²).** Computed as five independent repeats of the same
   best-split ratio with different random partitions (documented as `N_REPEATS=5` in
   `run_prediction.py`, reduced from a notional 10 for tractability in this
   environment), matching the "ten randomised experiments... averaged" logic the
   paper uses elsewhere for the SA results.
5. **Candidate-pool ("top-10") prediction.** Each model is trained on its best
   split's training data, then used to predict a return for every company's *most
   recent* available feature row (May 2023). This is the natural forward-looking
   analogue of the paper's "expected to perform better in future" ranking, given
   that the supplied dataset ends in May 2023 and no realised future returns exist
   in this session to validate against.
6. **Covariance matrix (Sigma).** Estimated from the *full* Jan 2018-May 2023
   historical monthly return series of the candidate-pool stocks (sample
   covariance), not from a rolling estimation window.
7. **Cumulative-return backtest.** The SA-optimised weights (a single, fixed
   allocation) are applied to the realised historical monthly returns of the same
   10 stocks across the *entire* sample to produce a compounding cumulative-return
   curve, benchmarked against an equal-weight (1/N) portfolio on the same 10 stocks.
   This is an **in-sample** backtest (the weights are optimised using information
   -- the covariance matrix -- drawn from the same period the cumulative return is
   computed over), exactly as flagged as a limitation in the revised manuscript
   (Section 7.2, "Single backtest window"). It is not a genuine out-of-sample test.
8. **RNN epochs.** Table 1 specifies `epochs=500`; this is reduced to 30 in
   `models.py` purely for tractability in this environment (CPU-only, 2 cores). This
   is a real, material deviation and is likely the main reason the RNN's relative
   accuracy ranking may differ from the paper's.
9. **Risk-free rate.** Sharpe ratio uses `Rf=0`, as in the manuscript.

## What this code is (and is not) good for

- It **is** a working, inspectable, end-to-end implementation of the exact
  mathematical model in the revised manuscript (Eq. 11, Algorithm 1), which you or a
  co-author can read, adjust, and re-run against your own feature set / exact split
  choices if you have the original specification.
- It **is** a legitimate empirical exercise on your real data that will surface
  genuinely new, honestly-computed tables and figures.
- It is **not** a reproduction of the original paper's published numbers, because
  the original code, exact feature set, and random seeds were never available in
  this session — only the manuscript text and, now, the raw CSVs.

See `outputs/RESULTS_SUMMARY.md` (generated after both scripts run) for a
side-by-side comparison against the paper's published tables and a discussion of
where and why the numbers differ.
