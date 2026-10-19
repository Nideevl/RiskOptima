from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize

from . import config as C

BENCHMARK_PATH = Path(__file__).resolve().parent / "data" / "benchmark_lapse.csv"


def logistic_hazard(t, a, b):
    """Annual lapse probability by policy year: logit(lapse) = a + b * ln(year).

    a is the log-odds of lapse in year 1; b (negative) is how fast the log-odds fall as policies age.
    """
    t = np.asarray(t, dtype=float)
    return 1.0 / (1.0 + np.exp(-(a + b * np.log(t))))


@dataclass(frozen=True)
class CurveFit:
    a: float
    b: float
    rmse: float

    def hazards(self, years):
        return logistic_hazard(years, self.a, self.b)


def load_benchmarks(path=BENCHMARK_PATH):
    return pd.read_csv(path)


def fit_hazard_curve(years, rates):
    years = np.asarray(years, dtype=float)
    rates = np.asarray(rates, dtype=float)
    popt, _ = curve_fit(logistic_hazard, years, rates, p0=[-2.0, -0.5], maxfev=20000)
    rmse = float(np.sqrt(np.mean((logistic_hazard(years, *popt) - rates) ** 2)))
    return CurveFit(float(popt[0]), float(popt[1]), rmse)


def fit_benchmark_curves(benchmarks=None):
    """One logistic lapse curve per risk category, fitted to the benchmark table."""
    benchmarks = load_benchmarks() if benchmarks is None else benchmarks
    return {risk: fit_hazard_curve(g.policy_year, g.lapse_rate)
            for risk, g in benchmarks.groupby("risk_name")}


@dataclass(frozen=True)
class Elasticity:
    beta: float
    beta_se: float
    year_intercepts: np.ndarray  # log-odds of lapse per policy year at price parity


_ELASTICITY_SQL = """
    SELECT py.policy_year, MIN(v.gap_to_best, ?) AS gap, COUNT(*) AS n, SUM(py.lapsed) AS lapses
    FROM policy_years py
    JOIN policies pol ON pol.policy_id = py.policy_id
    JOIN v_price_position v ON v.product_id = pol.product_id
                           AND v.cohort_id = pol.cohort_id
                           AND v.sa_band = pol.sa_band
    WHERE py.died = 0
    GROUP BY py.policy_year, gap
"""


def estimate_price_elasticity(conn):
    """Maximum-likelihood logistic fit: logit(lapse) = alpha[policy_year] + beta * price_gap.

    price_gap is how far a product's effective rate sits above the cheapest in its cell.
    Fitted on grouped counts with scipy.optimize (BFGS, analytic gradient).
    """
    import pandas as pd
    g = pd.read_sql_query(_ELASTICITY_SQL, conn, params=(C.MAX_PRICE_GAP,))
    year_idx = g.policy_year.values.astype(int) - 1
    n_years = int(year_idx.max()) + 1
    gap, n, lapses = g.gap.values, g.n.values.astype(float), g.lapses.values.astype(float)

    def nll_and_grad(theta):
        eta = theta[year_idx] + theta[-1] * gap
        p = 1.0 / (1.0 + np.exp(-eta))
        nll = np.sum(n * np.logaddexp(0.0, eta) - lapses * eta)
        resid = n * p - lapses
        grad = np.bincount(year_idx, weights=resid, minlength=n_years)
        return nll, np.append(grad, np.sum(resid * gap))

    start = np.append(np.full(n_years, -2.5), 0.0)
    res = minimize(nll_and_grad, start, jac=True, method="BFGS")
    return Elasticity(beta=float(res.x[-1]), beta_se=float(np.sqrt(res.hess_inv[-1, -1])),
                      year_intercepts=res.x[:-1])
