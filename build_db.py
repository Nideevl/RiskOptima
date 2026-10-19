import argparse
import os

from riskoptima import config as C
from riskoptima import db, queries

parser = argparse.ArgumentParser(description="Build the RiskOptima SQLite database from synthetic data.")
parser.add_argument("--out", default="riskoptima.db")
parser.add_argument("--n", type=int, default=C.N_POLICIES)
parser.add_argument("--seed", type=int, default=C.SEED)
args = parser.parse_args()

if os.path.exists(args.out):
    os.remove(args.out)
conn = db.build(args.out, args.n, args.seed)
for t in db.LOAD_ORDER:
    print(f"{t:18s} {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>9,d} rows")
print()
print(queries.run(conn, "competitive_share_by_segment").to_string(index=False))
