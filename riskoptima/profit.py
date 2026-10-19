from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass(frozen=True)
class Assumptions:
    expense_ratio: float = 0.10          # share of premium spent on servicing and renewal commission
    discount_rate: float = 0.08
    horizon: int = C.MAX_POLICY_YEARS
    price_elasticity: float = C.LAPSE_PRICE_SENSITIVITY  # beta; use the fitted value in real runs
    conversion_elasticity: float = 3.0   # ASSUMPTION: new-business volume multiplier = exp(this * gap closed)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def gap_after_discount(gap, discount):
    """Price gap to the cheapest competitor after cutting our premium by `discount` (competitors do not react)."""
    return max((1.0 + gap) * (1.0 - discount) - 1.0, 0.0)


def lapse_path(base_hazards, price_elasticity, gap):
    return _sigmoid(_logit(base_hazards) + price_elasticity * gap)


def _cash_flows(premium, sum_assured, mortality, base_hazards, gap, discount, a):
    mortality = np.asarray(mortality, dtype=float)[: a.horizon]
    lapse = lapse_path(np.asarray(base_hazards, dtype=float)[: a.horizon],
                       a.price_elasticity, gap_after_discount(gap, discount))
    in_force = np.concatenate(([1.0], np.cumprod((1.0 - lapse) * (1.0 - mortality))[:-1]))
    t = np.arange(1, len(mortality) + 1)
    premium_income = in_force * premium * (1.0 - discount)
    expenses = a.expense_ratio * premium_income
    claims = in_force * mortality * sum_assured
    profit = premium_income - expenses - claims
    pv_profit = profit * (1.0 + a.discount_rate) ** -(t - 1.0)
    return t, in_force, lapse, mortality, premium_income, expenses, claims, profit, pv_profit


def project(premium, sum_assured, mortality, base_hazards, gap, discount, a):
    """Year-by-year expected cash flows for one policy sold. Cash flows are discounted from the start of each year."""
    t, in_force, lapse, mort, premium_income, expenses, claims, profit, pv = _cash_flows(
        premium, sum_assured, mortality, base_hazards, gap, discount, a)
    return pd.DataFrame({
        "policy_year": t, "in_force": in_force, "lapse_prob": lapse, "mortality": mort,
        "premium_income": premium_income, "expenses": expenses, "claims": claims, "profit": profit,
        "pv_profit": pv,
    })


def break_even_cac(premium, sum_assured, mortality, base_hazards, gap, discount, a):
    """Most we can pay to acquire one policy: its expected discounted lifetime profit."""
    return float(_cash_flows(premium, sum_assured, mortality, base_hazards, gap, discount, a)[-1].sum())


def break_even_cac_closed_form(premium, sum_assured, lapse, mortality, discount, a):
    """Closed-form break-even CAC when lapse and mortality are constant per year (geometric series).

    CAC* = m * (1 - x^T) / (1 - x),  m = premium*(1-d)*(1-expense) - mortality*SA,
    x = (1 - lapse)(1 - mortality) / (1 + r).
    """
    m = premium * (1.0 - discount) * (1.0 - a.expense_ratio) - mortality * sum_assured
    x = (1.0 - lapse) * (1.0 - mortality) / (1.0 + a.discount_rate)
    return float(m * (1.0 - x ** a.horizon) / (1.0 - x))


_SEGMENT_SQL = """
    SELECT c.cohort_id, c.age_band, rc.risk_name, rc.mortality_multiplier,
           COUNT(*) AS policies, AVG(pol.age_at_issue) AS age, AVG(pol.annual_premium) AS premium,
           AVG(pol.sum_assured) AS sum_assured
    FROM policies pol
    JOIN products pr ON pr.product_id = pol.product_id AND pr.is_primary = 1
    JOIN cohorts c ON c.cohort_id = pol.cohort_id
    JOIN risk_categories rc ON rc.risk_category_id = c.risk_category_id
    WHERE rc.risk_name = ?
    GROUP BY c.cohort_id, c.age_band, rc.risk_name, rc.mortality_multiplier
    ORDER BY c.age_band
"""

_GAP_SQL = """
    SELECT v.cohort_id, AVG(MIN(v.gap_to_best, ?)) AS gap
    FROM v_price_position v
    JOIN products pr ON pr.product_id = v.product_id AND pr.is_primary = 1
    GROUP BY v.cohort_id
"""


def primary_segments(conn, risk_name="Preferred"):
    """Average policy of the primary insurer in each cohort of one risk category, with its price gap."""
    seg = pd.read_sql_query(_SEGMENT_SQL, conn, params=(risk_name,))
    gaps = pd.read_sql_query(_GAP_SQL, conn, params=(C.MAX_PRICE_GAP,))
    return seg.merge(gaps, on="cohort_id")


def segment_mortality(age, mortality_multiplier, horizon):
    return np.clip(C.qx(age + np.arange(horizon)) * mortality_multiplier, 0.0, 1.0)


def scan_discounts(segment, base_hazards, a, discounts):
    """Value of cutting the premium by each discount, for one segment (a row of primary_segments)."""
    mortality = segment_mortality(segment["age"], segment["mortality_multiplier"], a.horizon)
    rows = []
    for d in discounts:
        cac = break_even_cac(segment["premium"], segment["sum_assured"], mortality,
                             base_hazards, segment["gap"], d, a)
        gap_closed = segment["gap"] - gap_after_discount(segment["gap"], d)
        volume = float(np.exp(a.conversion_elasticity * gap_closed))
        rows.append({
            "discount": d, "price_gap_after": gap_after_discount(segment["gap"], d),
            "break_even_cac": cac, "be_cac_pct_list_premium": 100.0 * cac / segment["premium"],
            "volume_multiplier": volume, "value_index": volume * cac,
        })
    return pd.DataFrame(rows)


def optimal_discount(scan):
    return float(scan.loc[scan.value_index.idxmax(), "discount"])
