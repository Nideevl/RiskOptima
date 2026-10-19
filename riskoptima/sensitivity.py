from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from . import profit
from .churn import estimate_price_elasticity, fit_benchmark_curves
from .profit import Assumptions

DISCOUNT_TESTED = 0.05
OUTCOMES = ("clv", "discount_gain", "targeted_gain")


@dataclass(frozen=True)
class Context:
    """Everything fixed while inputs are varied: primary Preferred cohorts, base lapse curve, fitted elasticity."""
    segments: pd.DataFrame      # one row per cohort with weight and baseline mortality path
    hazards: np.ndarray         # base lapse hazards by policy year (Preferred benchmark curve)
    beta: float
    beta_se: float
    horizon: int


def prepare(conn, risk_name="Preferred"):
    horizon = Assumptions().horizon
    seg = profit.primary_segments(conn, risk_name).copy()
    seg["weight"] = seg.policies / seg.policies.sum()
    seg["mort_path"] = [profit.segment_mortality(r.age, r.mortality_multiplier, horizon) for r in seg.itertuples()]
    curve = fit_benchmark_curves()[risk_name]
    el = estimate_price_elasticity(conn)
    return Context(seg, curve.hazards(np.arange(1, horizon + 1)), el.beta, el.beta_se, horizon)


def base_inputs(ctx):
    d = Assumptions()
    return {"lapse_multiplier": 1.0, "mortality_multiplier": 1.0, "expense_ratio": d.expense_ratio,
            "discount_rate": d.discount_rate, "price_elasticity": ctx.beta, "conversion_elasticity": d.conversion_elasticity}


def tornado_ranges(ctx):
    """Plausible low/high for each uncertain input (documented judgement, not estimated from data except beta)."""
    b = base_inputs(ctx)
    return {
        "lapse_multiplier": (0.7, 1.3),                       # persistency experience +/-30%
        "mortality_multiplier": (0.9, 1.1),                   # claims experience +/-10%
        "expense_ratio": (0.08, 0.12),
        "discount_rate": (0.06, 0.10),
        "price_elasticity": (max(b["price_elasticity"] - 2 * ctx.beta_se, 0.0), b["price_elasticity"] + 2 * ctx.beta_se),
        "conversion_elasticity": (1.5, 4.5),                  # ASSUMPTION, no data behind it
    }


def equal_swing_ranges(ctx, swing=0.2):
    return {k: (v * (1 - swing), v * (1 + swing)) for k, v in base_inputs(ctx).items()}


def evaluate(ctx, p, discount=DISCOUNT_TESTED):
    """Portfolio-weighted outcomes per policy sold.

    clv: expected discounted lifetime profit (= break-even CAC) with no discount.
    discount_gain: change in volume-adjusted value from giving `discount` to every Preferred cohort.
    targeted_gain: the same, but only for cohorts priced above the cheapest competitor by more than
    COMPETITIVE_TOLERANCE (others keep list price, so contribute zero).
    """
    a = Assumptions(expense_ratio=p["expense_ratio"], discount_rate=p["discount_rate"], horizon=ctx.horizon,
                    price_elasticity=p["price_elasticity"], conversion_elasticity=p["conversion_elasticity"])
    hazards = np.clip(ctx.hazards * p["lapse_multiplier"], 1e-6, 0.95)
    clv = blanket = targeted = 0.0
    for r in ctx.segments.itertuples():
        mort = np.clip(r.mort_path * p["mortality_multiplier"], 0.0, 1.0)
        c0 = profit.break_even_cac(r.premium, r.sum_assured, mort, hazards, r.gap, 0.0, a)
        c1 = profit.break_even_cac(r.premium, r.sum_assured, mort, hazards, r.gap, discount, a)
        volume = np.exp(a.conversion_elasticity * (r.gap - profit.gap_after_discount(r.gap, discount)))
        delta = r.weight * (volume * c1 - c0)
        clv += r.weight * c0
        blanket += delta
        if r.gap > C.COMPETITIVE_TOLERANCE:
            targeted += delta
    return {"clv": clv, "discount_gain": blanket, "targeted_gain": targeted}


def tornado(ctx, ranges=None, outcome="clv"):
    ranges = tornado_ranges(ctx) if ranges is None else ranges
    base = base_inputs(ctx)
    centre = evaluate(ctx, base)[outcome]
    rows = []
    for name, (lo, hi) in ranges.items():
        out_lo = evaluate(ctx, {**base, name: lo})[outcome]
        out_hi = evaluate(ctx, {**base, name: hi})[outcome]
        rows.append({"input": name, "low_value": lo, "high_value": hi, "outcome_at_low": out_lo,
                     "outcome_at_high": out_hi, "swing": abs(out_hi - out_lo), "base_outcome": centre})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def sample_inputs(ctx, n=5000, seed=C.SEED):
    """Independent draws: lognormal for the two experience multipliers, triangular for costs and the assumed
    conversion elasticity, normal (from the fitted standard error) for price elasticity."""
    rng = np.random.default_rng(seed)
    b = base_inputs(ctx)
    return pd.DataFrame({
        "lapse_multiplier": rng.lognormal(0.0, 0.20, n),
        "mortality_multiplier": rng.lognormal(0.0, 0.08, n),
        "expense_ratio": rng.triangular(0.07, b["expense_ratio"], 0.14, n),
        "discount_rate": rng.triangular(0.06, b["discount_rate"], 0.10, n),
        "price_elasticity": np.clip(rng.normal(b["price_elasticity"], ctx.beta_se, n), 0.0, None),
        "conversion_elasticity": rng.triangular(1.0, b["conversion_elasticity"], 5.0, n),
    })


def monte_carlo(ctx, n=5000, seed=C.SEED):
    draws = sample_inputs(ctx, n, seed)
    results = pd.DataFrame([evaluate(ctx, row) for row in draws.to_dict("records")])
    return pd.concat([draws, results], axis=1)


def summarise(mc):
    rows = []
    for o in OUTCOMES:
        q = mc[o].quantile([0.05, 0.5, 0.95])
        rows.append({"outcome": o, "mean": mc[o].mean(), "p5": q[0.05], "p50": q[0.5], "p95": q[0.95],
                     "prob_positive": float((mc[o] > 0).mean())})
    return pd.DataFrame(rows)


def driver_ranking(mc, outcome="clv"):
    """Spearman rank correlation of each input with the outcome across the draws, largest effect first."""
    inputs = [c for c in mc.columns if c not in OUTCOMES]
    rho = {c: spearmanr(mc[c], mc[outcome])[0] for c in inputs}
    out = pd.DataFrame({"input": list(rho), "spearman": list(rho.values())})
    return out.reindex(out.spearman.abs().sort_values(ascending=False).index).reset_index(drop=True)
