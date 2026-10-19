import math

SEED = 42
N_POLICIES = 100_000
MAX_POLICY_YEARS = 10
COMPETITIVE_TOLERANCE = 0.05  # must match the 1.05 factor in sql/schema.sql (v_price_position)

AGE_BANDS = {"25-34": (25, 34), "35-44": (35, 44), "45-54": (45, 54), "55-64": (55, 64)}
AGE_BAND_WEIGHTS = {"25-34": 0.30, "35-44": 0.30, "45-54": 0.25, "55-64": 0.15}

# name: (mortality multiplier vs. baseline, population share)
RISK_CATEGORIES = {
    "Preferred": (0.70, 0.30),
    "Standard": (1.00, 0.50),
    "Substandard": (1.80, 0.20),
}

# sum assured bands in INR
SA_BANDS = {"25L": 2_500_000, "50L": 5_000_000, "100L": 10_000_000}
SA_BAND_WEIGHTS = {"25L": 0.35, "50L": 0.40, "100L": 0.25}
SA_RATE_ADJ = {"25L": 1.04, "50L": 1.00, "100L": 0.96}  # volume discount, same for all insurers

# Base annual lapse probability by policy year (synthetic assumption, replaced by fitted curves in step 2)
BASE_LAPSE_BY_YEAR = {1: 0.14, 2: 0.10, 3: 0.08, 4: 0.06, 5: 0.05,
                      6: 0.04, 7: 0.04, 8: 0.04, 9: 0.04, 10: 0.04}
LAPSE_PRICE_SENSITIVITY = 1.2  # log-odds added per 100% price gap to the cheapest comparable quote
MAX_PRICE_GAP = 0.5
QUOTE_NOISE_SIGMA = 0.015


def qx(age):
    """Synthetic baseline annual mortality (Gompertz-style). Works on scalars and numpy arrays."""
    import numpy as np
    return 0.0004 * np.exp(0.085 * (np.asarray(age, dtype=float) - 30))


# protection_share: fraction of premium that funds the death cover (rest is savings).
# Used to compare term, whole life and endowment on a common "effective" basis.
# loading: expense/profit loading on expected claims, per risk category (float or per age band).
PRODUCTS = [
    dict(
        insurer="Primary Insurer", product_name="SecureLife Term", plan_type="term",
        is_primary=1, protection_share=1.00, market_share=0.25,
        tiers=[("Platinum", "Preferred", 1.00), ("Gold", "Standard", 1.00), ("Silver", "Substandard", 1.00)],
        loading={
            "Preferred": {"25-34": 0.55, "35-44": 0.60, "45-54": 0.42, "55-64": 0.32},
            "Standard": 0.45,
            "Substandard": 0.40,
        },
    ),
    dict(
        insurer="Competitor A", product_name="ShieldPro Term", plan_type="term",
        is_primary=0, protection_share=1.00, market_share=0.30,
        tiers=[("Class 1", "Preferred", 1.00), ("Class 2", "Standard", 0.97),
               ("Class 3", "Standard", 1.03), ("Class 4", "Substandard", 1.00)],
        loading={"Preferred": 0.40, "Standard": 0.42, "Substandard": 0.45},
    ),
    dict(
        insurer="Competitor B", product_name="LifeGuard Whole Life", plan_type="whole_life",
        is_primary=0, protection_share=0.40, market_share=0.25,
        tiers=[("Elite", "Preferred", 1.00), ("Select", "Standard", 1.00), ("Rated", "Substandard", 1.00)],
        loading={"Preferred": 0.45, "Standard": 0.50, "Substandard": 0.50},
    ),
    dict(
        insurer="Competitor C", product_name="FutureSure Endowment", plan_type="endowment",
        is_primary=0, protection_share=0.25, market_share=0.20,
        tiers=[("Tier A", "Preferred", 1.00), ("Tier B", "Standard", 1.00), ("Tier C", "Substandard", 1.00)],
        loading={"Preferred": 0.52, "Standard": 0.55, "Substandard": 0.52},
    ),
]


def loading_for(product, risk_name, age_band):
    value = product["loading"][risk_name]
    return value[age_band] if isinstance(value, dict) else value
