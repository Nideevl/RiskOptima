import argparse
import os

from riskoptima import db, sensitivity as S

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=5000, help="Monte Carlo draws")
parser.add_argument("--out", default="outputs")
args = parser.parse_args()

ctx = S.prepare(db.build())
os.makedirs(args.out, exist_ok=True)
pd_opts = dict(index=False, float_format="%.4f")

print(f"Beta (price elasticity) = {ctx.beta:.3f} +/- {ctx.beta_se:.3f}; discount tested = {S.DISCOUNT_TESTED:.0%}")
for outcome in S.OUTCOMES:
    t = S.tornado(ctx, outcome=outcome)
    t.to_csv(f"{args.out}/tornado_{outcome}.csv", **pd_opts)
    print(f"\nTORNADO ({outcome}) - plausible ranges, base = {t.base_outcome.iloc[0]:,.1f}")
    print(t.drop(columns="base_outcome").round(3).to_string(index=False))
    e = S.tornado(ctx, S.equal_swing_ranges(ctx), outcome)
    print(f"  ranking if every input swings +/-20%: {', '.join(e.input)}")

mc = S.monte_carlo(ctx, args.n)
mc.to_csv(f"{args.out}/monte_carlo_draws.csv", **pd_opts)
print(f"\nMONTE CARLO ({args.n:,} draws)")
print(S.summarise(mc).round(3).to_string(index=False))
for outcome in S.OUTCOMES:
    print(f"\nRank correlation with {outcome}")
    print(S.driver_ranking(mc, outcome).round(3).to_string(index=False))
