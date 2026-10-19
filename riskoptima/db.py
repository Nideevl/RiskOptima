import sqlite3
from pathlib import Path

from . import config as C
from .datagen import generate

SCHEMA_PATH = Path(__file__).resolve().parent / "sql" / "schema.sql"
LOAD_ORDER = ["risk_categories", "cohorts", "products", "product_tiers", "tier_risk_map",
              "premium_quotes", "policies", "policy_years", "claims"]


def connect(path=":memory:"):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn):
    conn.executescript(SCHEMA_PATH.read_text())


def load_tables(conn, data):
    for table in LOAD_ORDER:
        data[table].to_sql(table, conn, if_exists="append", index=False)
    conn.commit()


def build(path=":memory:", n_policies=C.N_POLICIES, seed=C.SEED):
    conn = connect(path)
    create_schema(conn)
    load_tables(conn, generate(n_policies, seed))
    return conn
