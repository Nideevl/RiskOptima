import dataclasses

import numpy as np
import pandas as pd
import pytest

from riskoptima import config as C
from riskoptima import profit
from riskoptima import sensitivity as S
from riskoptima.profit import Assumptions


def test_base_clv_matches_discount_scan(ctx):
    a = Assumptions(price_elasticity=ctx.beta)
    expected = 0.0
    for r in ctx.segments.itertuples():
        scan = profit.scan_discounts(pd.Series(r._asdict()), ctx.hazards, a, [0.0])
        expected += r.weight * scan.break_even_cac.iloc[0]
    assert S.evaluate(ctx, S.base_inputs(ctx))["clv"] == pytest.approx(expected)


def test_tornado_structure_and_order(ctx):
    t = S.tornado(ctx, outcome="clv")
    assert len(t) == 6 and set(t.input) == set(S.base_inputs(ctx))
    assert t.swing.is_monotonic_decreasing
    assert t.base_outcome.iloc[0] == pytest.approx(S.evaluate(ctx, S.base_inputs(ctx))["clv"])


def test_clv_falls_when_mortality_or_lapse_rise(ctx):
    t = S.tornado(ctx, outcome="clv").set_index("input")
    for name in ("mortality_multiplier", "lapse_multiplier", "expense_ratio", "discount_rate"):
        assert t.loc[name, "outcome_at_high"] < t.loc[name, "outcome_at_low"]


def test_conversion_elasticity_only_matters_for_discount_outcomes(ctx):
    clv = S.tornado(ctx, outcome="clv").set_index("input")
    tg = S.tornado(ctx, outcome="targeted_gain").set_index("input")
    assert clv.loc["conversion_elasticity", "swing"] == pytest.approx(0.0)
    assert tg.loc["conversion_elasticity", "outcome_at_high"] > tg.loc["conversion_elasticity", "outcome_at_low"]


def test_equal_swing_ranges(ctx):
    base = S.base_inputs(ctx)
    for name, (lo, hi) in S.equal_swing_ranges(ctx, 0.2).items():
        assert lo == pytest.approx(0.8 * base[name]) and hi == pytest.approx(1.2 * base[name])


def test_sample_inputs_reproducible_and_within_bounds(ctx):
    a, b = S.sample_inputs(ctx, 2000, seed=5), S.sample_inputs(ctx, 2000, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2000
    assert a.expense_ratio.between(0.07, 0.14).all()
    assert a.discount_rate.between(0.06, 0.10).all()
    assert a.conversion_elasticity.between(1.0, 5.0).all()
    assert (a.price_elasticity >= 0).all() and (a.lapse_multiplier > 0).all()


def test_monte_carlo_outputs(ctx):
    mc = S.monte_carlo(ctx, n=300, seed=1)
    assert len(mc) == 300 and not mc.isna().any().any()
    pd.testing.assert_frame_equal(mc, S.monte_carlo(ctx, n=300, seed=1))
    summary = S.summarise(mc)
    assert set(summary.outcome) == set(S.OUTCOMES)
    assert summary.prob_positive.between(0, 1).all()
    assert (summary.p5 <= summary.p50).all() and (summary.p50 <= summary.p95).all()


def test_driver_ranking_is_sorted_and_signed(ctx):
    mc = S.monte_carlo(ctx, n=600, seed=2)
    r = S.driver_ranking(mc, "clv")
    assert r.spearman.abs().is_monotonic_decreasing
    assert r.set_index("input").loc["mortality_multiplier", "spearman"] < 0


def test_discount_without_price_gap_only_costs_money(ctx):
    seg = ctx.segments.copy()
    seg["gap"] = 0.0
    flat = dataclasses.replace(ctx, segments=seg)
    out = S.evaluate(flat, S.base_inputs(flat))
    assert out["discount_gain"] < 0
    assert out["targeted_gain"] == 0.0


def test_targeted_gain_only_counts_cohorts_above_tolerance(ctx):
    out = S.evaluate(ctx, S.base_inputs(ctx))
    mask = ctx.segments.gap > C.COMPETITIVE_TOLERANCE
    assert mask.any() and not mask.all()
    assert out["targeted_gain"] > out["discount_gain"]
