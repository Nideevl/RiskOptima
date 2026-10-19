import numpy as np
import pandas as pd

from riskoptima import config as C
from riskoptima.datagen import generate


def test_twelve_cohorts(data):
    assert len(data["cohorts"]) == 12
    assert data["cohorts"].cohort_weight.sum() == np.float64(1.0) or abs(data["cohorts"].cohort_weight.sum() - 1.0) < 1e-9


def test_four_products_one_primary(data):
    p = data["products"]
    assert len(p) == 4
    assert p.is_primary.sum() == 1
    assert set(p.plan_type) == {"term", "whole_life", "endowment"}


def test_quotes_rise_with_risk_within_product(data):
    q = (data["premium_quotes"].merge(data["product_tiers"], on="tier_id")
         .merge(data["tier_risk_map"], on="tier_id")
         .merge(data["products"][["product_id", "protection_share"]], on="product_id"))
    q["eff"] = q.premium_per_1000 * q.protection_share
    g = q.groupby(["product_id", "age_band", "sa_band", "risk_category_id"]).eff.mean().unstack()
    assert (g[1] < g[2]).all() and (g[2] < g[3]).all()


def test_quotes_rise_with_age(data):
    q = data["premium_quotes"].merge(data["product_tiers"], on="tier_id")
    g = q.groupby(["tier_id", "sa_band", "age_band"]).premium_per_1000.mean().unstack()
    assert (g["25-34"] < g["35-44"]).all() and (g["35-44"] < g["45-54"]).all() and (g["45-54"] < g["55-64"]).all()


def test_product_mix_matches_market_share(data):
    mix = data["policies"].product_id.value_counts(normalize=True).sort_index()
    expected = data["products"].set_index("product_id").market_share
    assert (mix - expected).abs().max() < 0.03


def test_policy_years_are_contiguous_and_end_once(data):
    py = data["policy_years"].sort_values(["policy_id", "policy_year"])
    first = py.groupby("policy_id").policy_year.min()
    assert (first == 1).all()
    counts = py.groupby("policy_id").policy_year.agg(["count", "max"])
    assert (counts["count"] == counts["max"]).all()
    ended = (py.lapsed + py.died)
    assert ended.max() <= 1
    # no rows after a policy has ended
    py = py.assign(ended=ended)
    prior_ends = py.groupby("policy_id").ended.cumsum() - py.ended
    assert (prior_ends == 0).all()
    assert py.policy_year.max() <= C.MAX_POLICY_YEARS


def test_claims_match_deaths_and_sum_assured(data):
    py, cl, pol = data["policy_years"], data["claims"], data["policies"]
    assert len(cl) == py.died.sum()
    m = cl.merge(pol[["policy_id", "sum_assured"]], on="policy_id")
    assert (m.claim_amount == m.sum_assured).all()


def test_same_seed_is_reproducible():
    a, b = generate(500, seed=3), generate(500, seed=3)
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])
