import argparse

from riskoptima import db, reporting, sensitivity

parser = argparse.ArgumentParser(description="Write Tableau-ready CSV extracts.")
parser.add_argument("--out", default="tableau_extracts")
args = parser.parse_args()

conn = db.build()
paths = reporting.build_extracts(conn, sensitivity.prepare(conn), args.out)
for name, path in paths.items():
    print(f"{path}")
