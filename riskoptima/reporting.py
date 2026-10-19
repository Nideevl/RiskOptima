from pathlib import Path

import numpy as np
import pandas as pd

from . import queries
from . import sensitivity as S
from . import profit
from .profit import Assumptions

DISCOUNT_GRID = tuple(np.round(np.arange(0.0, 0.2001, 0.025), 3))


def price_position(conn):
    return pd.read_sql_query("""
        SELECT v.*, p.insurer, p.product_name, p.plan_type, p.is_primary
        FROM v_price_position v JOIN products p ON p.product_id = v.product_id""", conn)


def effective_premium(conn):
    return pd.read_sql_query("SELECT * FROM v_effective_premium", conn)


def lapse_by_year(conn):
    lapse = queries.run(conn, "lapse_by_policy_year")
    products = pd.read_sql_query("SELECT product_id, insurer, product_name FROM products", conn)
    return lapse.merge(products, on="product_id")


def discount_scan_table(ctx, grid=DISCOUNT_GRID, assumptions=None, lapse_multiplier=1.0, mortality_multiplier=1.0):
    """Discount scan for every primary Preferred cohort, with optional experience overrides."""
    a = assumptions or Assumptions(price_elasticity=ctx.beta)
    hazards = np.clip(ctx.hazards * lapse_multiplier, 1e-6, 0.95)
    frames = []
    for r in ctx.segments.itertuples():
        seg = pd.Series({"age": r.age, "mortality_multiplier": r.mortality_multiplier * mortality_multiplier,
                         "premium": r.premium, "sum_assured": r.sum_assured, "gap": r.gap})
        scan = profit.scan_discounts(seg, hazards, a, grid)
        scan.insert(0, "age_band", r.age_band)
        scan.insert(1, "list_premium", r.premium)
        scan.insert(2, "price_gap", r.gap)
        scan.insert(3, "policies", r.policies)
        frames.append(scan)
    return pd.concat(frames, ignore_index=True)


def tornado_table(ctx, ranges=None):
    frames = []
    for outcome in S.OUTCOMES:
        t = S.tornado(ctx, ranges, outcome)
        t.insert(0, "outcome", outcome)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def build_extracts(conn, ctx, out_dir, mc=None):
    """Write flat CSV files that Tableau can connect to directly. Returns {name: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "price_position": price_position(conn),
        "effective_premium": effective_premium(conn),
        "lapse_by_year": lapse_by_year(conn),
        "discount_scan": discount_scan_table(ctx),
        "tornado": tornado_table(ctx),
        "monte_carlo_draws": S.monte_carlo(ctx) if mc is None else mc,
    }
    paths = {}
    for name, df in tables.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths
