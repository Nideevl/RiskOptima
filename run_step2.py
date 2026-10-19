import argparse
import numpy as np
import pandas as pd

from riskoptima import config as C
from riskoptima import db, profit
from riskoptima.churn import estimate_price_elasticity, fit_benchmark_curves

parser = argparse.ArgumentParser()
parser.add_argument("--db", default=None, help="existing riskoptima.db (default: build in memory)")
args = parser.parse_args()
conn = db.connect(args.db) if args.db else db.build()

curves = fit_benchmark_curves()
print("Fitted logistic lapse curves: logit(lapse) = a + b*ln(year)")
for risk, c in curves.items():
    print(f"  {risk:12s} a={c.a:.3f}  b={c.b:.3f}  year-1 lapse={c.hazards(1):.3f}  year-10 lapse={c.hazards(10):.3f}  rmse={c.rmse:.5f}")

el = estimate_price_elasticity(conn)
print(f"\nPrice elasticity beta = {el.beta:.3f} (approx SE {el.beta_se:.3f}); true value in generator = {C.LAPSE_PRICE_SENSITIVITY}")

a = profit.Assumptions(price_elasticity=el.beta)
hazards = curves["Preferred"].hazards(np.arange(1, a.horizon + 1))
grid = np.round(np.arange(0, 0.2001, 0.025), 3)
for _, seg in profit.primary_segments(conn, "Preferred").iterrows():
    scan = profit.scan_discounts(seg, hazards, a, grid)
    mort = profit.segment_mortality(seg["age"], seg["mortality_multiplier"], a.horizon)
    cf = profit.break_even_cac_closed_form(seg["premium"], seg["sum_assured"], hazards.mean(), mort.mean(), 0.0, a)
    print(f"\nPrimary / Preferred / {seg['age_band']}  premium={seg['premium']:,.0f}  price gap to cheapest={seg['gap']:.3f}")
    print(scan.round(3).to_string(index=False))
    sim0 = scan.break_even_cac.iloc[0]
    print(f"  closed-form BE CAC (constant-rate approx, no discount) = {cf:,.0f} vs simulated {sim0:,.0f} ({100 * (cf / sim0 - 1):+.1f}%)")
    print(f"  discount with the highest value index = {profit.optimal_discount(scan):.3f}")
