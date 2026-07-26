"""
portfolio.py
============
Elastic-net regularised mean-variance portfolio (EN-MVP) objective, Eq. (11)
of the revised manuscript, and the simulated-annealing solver of Algorithm 1
that jointly optimises the asset weights x and the L1/L2 mixing parameter r.

    min_{x,r}  lambda * (x' Sigma x) - (1-lambda) * (mu' x)
               + alpha * ( r * sum|x_i| + (1-r)/2 * sum x_i^2 )
    s.t.       sum x_i = 1,  0 <= x_i <= 1,  0 <= r <= 1
"""
import numpy as np


def objective(x, r, mu, sigma, lam, alpha):
    var_term = x @ sigma @ x
    ret_term = mu @ x
    l1 = np.sum(np.abs(x))
    l2 = np.sum(x ** 2)
    penalty = alpha * (r * l1 + (1 - r) / 2 * l2)
    return lam * var_term - (1 - lam) * ret_term + penalty


def _project_simplex(v):
    """Euclidean projection of vector v onto the probability simplex
    {x : sum x_i = 1, x_i >= 0} (Wang & Carreira-Perpinan, 2013)."""
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
    """Algorithm 1 of the revised manuscript: joint (x, r) simulated
    annealing search for the EN-MVP problem. Returns (x_best, r_best, f_best)."""
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

    # Apply the same 0.01 convergence threshold used in the manuscript to
    # report "zero" weights (the L1 term is non-smooth, so SA does not
    # converge individual weights to exactly zero in finite iterations).
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
    """Average an SA solution over `n_runs` independent seeds, as the
    manuscript does ('results of ten randomised experiments ... averaged')."""
    xs, rs, fs = [], [], []
    for k in range(n_runs):
        x, r, f = simulated_annealing(mu, sigma, lam, alpha, seed=seed0 + k, **kwargs)
        xs.append(x); rs.append(r); fs.append(f)
    x_mean = np.mean(xs, axis=0)
    if x_mean.sum() > 0:
        x_mean = x_mean / x_mean.sum()
    return x_mean, float(np.mean(rs)), float(np.mean(fs)), np.array(xs), np.array(rs), np.array(fs)
