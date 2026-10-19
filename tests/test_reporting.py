import numpy as np
import pandas as pd
import pytest

from riskoptima import reporting as R
from riskoptima import sensitivity as S


def test_discount_scan_table_shape_and_zero_discount_rows(ctx):
    t = R.discount_scan_table(ctx)
    assert len(t) == 4 * len(R.DISCOUNT_GRID)
    zero = t[t.discount == 0]
    assert len(zero) == 4 and np.allclose(zero.volume_multiplier, 1.0, rtol=0, atol=1e-12)


def test_scan_overrides_move_value_in_expected_direction(ctx):
    base = R.discount_scan_table(ctx)
    worse = R.discount_scan_table(ctx, mortality_multiplier=1.1, lapse_multiplier=1.2)
    assert (worse.break_even_cac < base.break_even_cac).all()


def test_scan_table_agrees_with_portfolio_clv(ctx):
    t = R.discount_scan_table(ctx)
    zero = t[t.discount == 0].merge(ctx.segments[["age_band", "weight"]], on="age_band")
    assert (zero.break_even_cac * zero.weight).sum() == pytest.approx(S.evaluate(ctx, S.base_inputs(ctx))["clv"])


def test_tornado_table_covers_every_outcome(ctx):
    t = R.tornado_table(ctx)
    assert len(t) == 6 * len(S.OUTCOMES) and set(t.outcome) == set(S.OUTCOMES)


def test_build_extracts_writes_clean_files(big_conn, ctx, tmp_path):
    mc = S.monte_carlo(ctx, n=100, seed=1)
    paths = R.build_extracts(big_conn, ctx, tmp_path / "ext", mc=mc)
    expected_rows = {"price_position": 144, "effective_premium": 48, "lapse_by_year": 40,
                     "discount_scan": 36, "tornado": 18, "monte_carlo_draws": 100}
    assert set(paths) == set(expected_rows)
    for name, path in paths.items():
        df = pd.read_csv(path)
        assert len(df) == expected_rows[name], name
        assert not df.drop(columns=["change_vs_prior_year"], errors="ignore").isna().any().any(), name
