import numpy as np

from riskoptima import config as C
from riskoptima.churn import (estimate_price_elasticity, fit_benchmark_curves,
                              fit_hazard_curve, logistic_hazard)


def test_fit_recovers_known_curve():
    years = np.arange(1, 11)
    fit = fit_hazard_curve(years, logistic_hazard(years, -1.8, -0.6))
    assert abs(fit.a + 1.8) < 1e-6 and abs(fit.b + 0.6) < 1e-6 and fit.rmse < 1e-9


def test_benchmark_curves_fit_closely_and_decline():
    curves = fit_benchmark_curves()
    assert set(curves) == set(C.RISK_CATEGORIES)
    for c in curves.values():
        assert c.rmse < 0.01
        assert c.b < 0
        h = c.hazards(np.arange(1, 11))
        assert (np.diff(h) < 0).all()


def test_substandard_lapses_less_than_preferred():
    curves = fit_benchmark_curves()
    assert curves["Substandard"].hazards(1) < curves["Preferred"].hazards(1)


def test_price_elasticity_recovers_generator_value(big_conn):
    el = estimate_price_elasticity(big_conn)
    assert el.beta_se < 0.3
    assert abs(el.beta - C.LAPSE_PRICE_SENSITIVITY) < 3 * el.beta_se
    expected = np.log(np.array([C.BASE_LAPSE_BY_YEAR[t] for t in range(1, 11)])
                      / (1 - np.array([C.BASE_LAPSE_BY_YEAR[t] for t in range(1, 11)])))
    assert np.abs(el.year_intercepts - expected).max() < 0.35
