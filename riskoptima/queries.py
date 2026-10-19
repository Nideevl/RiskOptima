import pandas as pd

QUERIES = {
    "competitive_share_by_segment": """
        SELECT p.insurer, p.product_name, v.risk_name,
               AVG(v.is_competitive) AS competitive_share,
               COUNT(*)              AS cells
        FROM v_price_position v
        JOIN products p ON p.product_id = v.product_id
        GROUP BY p.insurer, p.product_name, v.risk_name
        ORDER BY v.risk_name, competitive_share DESC
    """,
    "lapse_by_policy_year": """
        WITH yr AS (
            SELECT pol.product_id, py.policy_year,
                   COUNT(*) AS exposed, SUM(py.lapsed) AS lapses
            FROM policy_years py
            JOIN policies pol ON pol.policy_id = py.policy_id
            GROUP BY pol.product_id, py.policy_year
        )
        SELECT product_id, policy_year, exposed, lapses,
               lapses * 1.0 / exposed AS lapse_rate,
               lapses * 1.0 / exposed
                 - LAG(lapses * 1.0 / exposed) OVER (PARTITION BY product_id ORDER BY policy_year)
                 AS change_vs_prior_year
        FROM yr
        ORDER BY product_id, policy_year
    """,
    "cumulative_claims_by_year": """
        WITH yr AS (
            SELECT pol.product_id, cl.policy_year, SUM(cl.claim_amount) AS claims
            FROM claims cl
            JOIN policies pol ON pol.policy_id = cl.policy_id
            GROUP BY pol.product_id, cl.policy_year
        )
        SELECT product_id, policy_year, claims,
               SUM(claims) OVER (PARTITION BY product_id ORDER BY policy_year) AS cumulative_claims
        FROM yr
        ORDER BY product_id, policy_year
    """,
}


def run(conn, name):
    return pd.read_sql_query(QUERIES[name], conn)
