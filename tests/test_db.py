from riskoptima import config as C
from riskoptima import queries


def test_nine_tables_two_views(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    views = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")}
    assert len(tables) == 9
    assert views == {"v_effective_premium", "v_price_position"}


def test_foreign_keys_hold(conn):
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_tier_mapping_is_many_to_one(conn):
    rows = conn.execute("""
        SELECT p.insurer, COUNT(t.tier_id), COUNT(DISTINCT m.risk_category_id)
        FROM product_tiers t JOIN products p ON p.product_id = t.product_id
        JOIN tier_risk_map m ON m.tier_id = t.tier_id GROUP BY p.insurer""").fetchall()
    by = {r[0]: (r[1], r[2]) for r in rows}
    assert by["Competitor A"] == (4, 3)
    assert by["Primary Insurer"] == (3, 3)


def test_price_position_has_one_cheapest_per_cell(conn):
    n = conn.execute("""SELECT COUNT(*) FROM (SELECT cohort_id, sa_band FROM v_price_position
                        GROUP BY cohort_id, sa_band HAVING MIN(price_rank) = 1)""").fetchone()[0]
    assert n == 12 * len(C.SA_BANDS)
    assert conn.execute("SELECT COUNT(*) FROM v_price_position").fetchone()[0] == 4 * 12 * len(C.SA_BANDS)


def test_competitive_flag_matches_tolerance(conn):
    bad = conn.execute("""SELECT COUNT(*) FROM v_price_position
        WHERE is_competitive != (CASE WHEN gap_to_best <= ? + 1e-12 THEN 1 ELSE 0 END)""",
                       (C.COMPETITIVE_TOLERANCE,)).fetchone()[0]
    assert bad == 0


def test_effective_premium_view_totals(conn):
    assert conn.execute("SELECT SUM(policies) FROM v_effective_premium").fetchone()[0] == 6000
    # margin = (premium - claims) / premium, so it can never exceed 1; thin cells with one large claim can go below 0
    hi = conn.execute("SELECT MAX(effective_margin) FROM v_effective_premium").fetchone()[0]
    assert hi <= 1


def test_effective_margin_positive_overall(conn):
    paid, claims = conn.execute("SELECT SUM(effective_premium), SUM(claims_paid) FROM v_effective_premium").fetchone()
    assert paid > claims > 0


def test_competitive_share_query_bounds(conn):
    df = queries.run(conn, "competitive_share_by_segment")
    assert df.competitive_share.between(0, 1).all()
    assert len(df) == 4 * 3


def test_lapse_query_uses_lag(conn):
    df = queries.run(conn, "lapse_by_policy_year")
    first = df[df.policy_year == 1]
    assert first.change_vs_prior_year.isna().all()
    assert df[df.policy_year > 1].change_vs_prior_year.notna().all()
    assert df.lapse_rate.between(0, 1).all()


def test_cumulative_claims_never_decrease(conn):
    df = queries.run(conn, "cumulative_claims_by_year")
    for _, g in df.groupby("product_id"):
        assert g.cumulative_claims.is_monotonic_increasing
        assert abs(g.cumulative_claims.iloc[-1] - g.claims.sum()) < 1e-6
