import numpy as np
import pytest

from riskoptima import profit
from riskoptima.profit import Assumptions


def _const(n, v):
    return np.full(n, v)


def test_closed_form_matches_simulator_for_constant_rates():
    a = Assumptions(horizon=10)
    lapse, q, prem, sa = 0.07, 0.004, 10_000.0, 5_000_000.0
    for d in (0.0, 0.05):
        sim = profit.break_even_cac(prem, sa, _const(10, q), _const(10, lapse), 0.0, d, a)
        cf = profit.break_even_cac_closed_form(prem, sa, lapse, q, d, a)
        assert sim == pytest.approx(cf, rel=1e-9)


def test_gap_after_discount_never_negative_and_zero_discount_is_identity():
    assert profit.gap_after_discount(0.12, 0.0) == pytest.approx(0.12)
    assert profit.gap_after_discount(0.02, 0.20) == 0.0
    assert profit.gap_after_discount(0.12, 0.05) < 0.12


def test_in_force_starts_at_one_and_never_increases():
    a = Assumptions()
    df = profit.project(10_000, 5e6, _const(10, 0.004), _const(10, 0.08), 0.1, 0.0, a)
    assert df.in_force.iloc[0] == 1.0
    assert (np.diff(df.in_force) <= 0).all()


def test_bigger_price_gap_means_more_lapse_and_less_value():
    a = Assumptions()
    args = (30_000, 5e6, _const(10, 0.004), _const(10, 0.08))  # profitable policy: margin > 0
    lo = profit.project(*args, 0.0, 0.0, a)
    hi = profit.project(*args, 0.3, 0.0, a)
    assert hi.lapse_prob.iloc[0] > lo.lapse_prob.iloc[0]
    assert lo.pv_profit.sum() > 0
    assert hi.pv_profit.sum() < lo.pv_profit.sum()


def test_value_falls_with_expenses_and_discount_rate():
    args = (30_000, 5e6, _const(10, 0.004), _const(10, 0.08), 0.0, 0.0)  # profitable policy
    base = profit.break_even_cac(*args, Assumptions())
    assert profit.break_even_cac(*args, Assumptions(expense_ratio=0.2)) < base
    assert profit.break_even_cac(*args, Assumptions(discount_rate=0.15)) < base


def test_scan_first_row_matches_direct_calculation(conn):
    a = Assumptions()
    seg = profit.primary_segments(conn, "Preferred").iloc[0]
    hz = _const(a.horizon, 0.07)
    scan = profit.scan_discounts(seg, hz, a, [0.0, 0.05])
    mort = profit.segment_mortality(seg["age"], seg["mortality_multiplier"], a.horizon)
    direct = profit.break_even_cac(seg["premium"], seg["sum_assured"], mort, hz, seg["gap"], 0.0, a)
    assert scan.break_even_cac.iloc[0] == pytest.approx(direct)
    assert scan.volume_multiplier.iloc[0] == 1.0
    assert scan.break_even_cac.iloc[1] < scan.break_even_cac.iloc[0]  # same retention curve, lower premium


def test_primary_segments_cover_four_age_bands(conn):
    seg = profit.primary_segments(conn, "Preferred")
    assert len(seg) == 4
    assert (seg.premium > 0).all() and (seg.gap >= 0).all()
    assert seg.premium.is_monotonic_increasing  # older cohorts hold larger cover and pay more


def test_discount_only_pays_when_there_is_a_gap_to_close():
    a = Assumptions()
    base = dict(age=45.0, mortality_multiplier=0.7, premium=10_000.0, sum_assured=2e6)  # healthy margin
    hz = _const(a.horizon, 0.07)
    grid = [0.0, 0.025, 0.05, 0.1]
    assert profit.optimal_discount(profit.scan_discounts({**base, "gap": 0.0}, hz, a, grid)) == 0.0
    assert profit.optimal_discount(profit.scan_discounts({**base, "gap": 0.15}, hz, a, grid)) > 0.0
