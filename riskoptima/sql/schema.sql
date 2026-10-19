PRAGMA foreign_keys = ON;

CREATE TABLE risk_categories (
    risk_category_id     INTEGER PRIMARY KEY,
    risk_name            TEXT NOT NULL UNIQUE,
    mortality_multiplier REAL NOT NULL CHECK (mortality_multiplier > 0),
    population_share     REAL NOT NULL
);

CREATE TABLE cohorts (
    cohort_id        INTEGER PRIMARY KEY,
    age_band         TEXT NOT NULL,
    risk_category_id INTEGER NOT NULL REFERENCES risk_categories (risk_category_id),
    cohort_weight    REAL NOT NULL,
    UNIQUE (age_band, risk_category_id)
);

CREATE TABLE products (
    product_id       INTEGER PRIMARY KEY,
    insurer          TEXT NOT NULL,
    product_name     TEXT NOT NULL,
    plan_type        TEXT NOT NULL CHECK (plan_type IN ('term', 'whole_life', 'endowment')),
    is_primary       INTEGER NOT NULL CHECK (is_primary IN (0, 1)),
    protection_share REAL NOT NULL CHECK (protection_share > 0 AND protection_share <= 1),
    market_share     REAL NOT NULL
);

-- Each insurer names and counts its underwriting tiers differently.
CREATE TABLE product_tiers (
    tier_id    INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products (product_id),
    tier_label TEXT NOT NULL,
    UNIQUE (product_id, tier_label)
);

-- Maps every insurer-specific tier to one common risk category.
CREATE TABLE tier_risk_map (
    tier_id          INTEGER PRIMARY KEY REFERENCES product_tiers (tier_id),
    risk_category_id INTEGER NOT NULL REFERENCES risk_categories (risk_category_id)
);

CREATE TABLE premium_quotes (
    tier_id          INTEGER NOT NULL REFERENCES product_tiers (tier_id),
    age_band         TEXT NOT NULL,
    sa_band          TEXT NOT NULL,
    premium_per_1000 REAL NOT NULL CHECK (premium_per_1000 > 0),
    PRIMARY KEY (tier_id, age_band, sa_band)
);

CREATE TABLE policies (
    policy_id      INTEGER PRIMARY KEY,
    product_id     INTEGER NOT NULL REFERENCES products (product_id),
    tier_id        INTEGER NOT NULL REFERENCES product_tiers (tier_id),
    cohort_id      INTEGER NOT NULL REFERENCES cohorts (cohort_id),
    age_at_issue   INTEGER NOT NULL,
    sa_band        TEXT NOT NULL,
    sum_assured    REAL NOT NULL,
    annual_premium REAL NOT NULL
);

-- One row per policy per year in force. lapsed = 1 means the policy ended at the end of that year.
CREATE TABLE policy_years (
    policy_id    INTEGER NOT NULL REFERENCES policies (policy_id),
    policy_year  INTEGER NOT NULL,
    premium_paid REAL NOT NULL,
    lapsed       INTEGER NOT NULL CHECK (lapsed IN (0, 1)),
    died         INTEGER NOT NULL CHECK (died IN (0, 1)),
    PRIMARY KEY (policy_id, policy_year)
);

CREATE TABLE claims (
    claim_id     INTEGER PRIMARY KEY,
    policy_id    INTEGER NOT NULL,
    policy_year  INTEGER NOT NULL,
    claim_amount REAL NOT NULL,
    FOREIGN KEY (policy_id, policy_year) REFERENCES policy_years (policy_id, policy_year)
);

CREATE INDEX idx_policies_product ON policies (product_id);
CREATE INDEX idx_policies_cohort  ON policies (cohort_id);
CREATE INDEX idx_claims_policy    ON claims (policy_id);

-- View 1: protection-adjusted premium vs. claims, by product and cohort.
-- Premium is scaled by protection_share so term, whole life and endowment are comparable.
CREATE VIEW v_effective_premium AS
WITH py AS (
    SELECT policy_id, SUM(premium_paid) AS premium, COUNT(*) AS exposure_years
    FROM policy_years
    GROUP BY policy_id
),
cl AS (
    SELECT policy_id, SUM(claim_amount) AS claims
    FROM claims
    GROUP BY policy_id
)
SELECT pol.product_id,
       pr.insurer,
       pr.product_name,
       pr.plan_type,
       pol.cohort_id,
       c.age_band,
       rc.risk_name,
       COUNT(*)                                        AS policies,
       SUM(py.exposure_years)                          AS exposure_years,
       SUM(py.premium * pr.protection_share)           AS effective_premium,
       COALESCE(SUM(cl.claims), 0)                     AS claims_paid,
       (SUM(py.premium * pr.protection_share) - COALESCE(SUM(cl.claims), 0)) * 1.0
           / SUM(py.premium * pr.protection_share)     AS effective_margin
FROM policies pol
JOIN py ON py.policy_id = pol.policy_id
JOIN products pr ON pr.product_id = pol.product_id
JOIN cohorts c ON c.cohort_id = pol.cohort_id
JOIN risk_categories rc ON rc.risk_category_id = c.risk_category_id
LEFT JOIN cl ON cl.policy_id = pol.policy_id
GROUP BY pol.product_id, pr.insurer, pr.product_name, pr.plan_type, pol.cohort_id, c.age_band, rc.risk_name;

-- View 2: price position of every product in every (cohort, sum assured band) cell.
-- Window functions rank products and find the cheapest effective rate in the cell.
-- A product is "competitive" when within 5% of the cheapest (see COMPETITIVE_TOLERANCE).
CREATE VIEW v_price_position AS
WITH cell AS (
    SELECT t.product_id,
           c.cohort_id,
           c.age_band,
           rc.risk_name,
           q.sa_band,
           AVG(q.premium_per_1000 * pr.protection_share) AS effective_rate
    FROM premium_quotes q
    JOIN product_tiers t ON t.tier_id = q.tier_id
    JOIN products pr ON pr.product_id = t.product_id
    JOIN tier_risk_map m ON m.tier_id = q.tier_id
    JOIN cohorts c ON c.risk_category_id = m.risk_category_id AND c.age_band = q.age_band
    JOIN risk_categories rc ON rc.risk_category_id = c.risk_category_id
    GROUP BY t.product_id, c.cohort_id, c.age_band, rc.risk_name, q.sa_band
),
ranked AS (
    SELECT cell.*,
           MIN(effective_rate) OVER (PARTITION BY cohort_id, sa_band) AS best_rate,
           RANK() OVER (PARTITION BY cohort_id, sa_band ORDER BY effective_rate) AS price_rank
    FROM cell
)
SELECT ranked.*,
       effective_rate / best_rate - 1.0 AS gap_to_best,
       CASE WHEN effective_rate <= best_rate * 1.05 THEN 1 ELSE 0 END AS is_competitive
FROM ranked;
