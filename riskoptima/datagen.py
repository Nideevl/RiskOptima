import numpy as np
import pandas as pd

from . import config as C


def _effective_rate(product, risk_name, age_band, tier_adj, sa_band):
    """Expected claim cost per 1000 of cover, grossed up by the insurer's loading."""
    lo, hi = C.AGE_BANDS[age_band]
    mid_age = (lo + hi) / 2
    mult = C.RISK_CATEGORIES[risk_name][0]
    # level premium: average expected mortality over the observed policy years, not just the issue-age year
    base = float(C.qx(mid_age + np.arange(C.MAX_POLICY_YEARS)).mean()) * mult * 1000
    return base * (1 + C.loading_for(product, risk_name, age_band)) * tier_adj * C.SA_RATE_ADJ[sa_band]


def build_reference():
    risk_names = list(C.RISK_CATEGORIES)
    risk_categories = pd.DataFrame({
        "risk_category_id": range(1, len(risk_names) + 1),
        "risk_name": risk_names,
        "mortality_multiplier": [C.RISK_CATEGORIES[r][0] for r in risk_names],
        "population_share": [C.RISK_CATEGORIES[r][1] for r in risk_names],
    })
    risk_id = dict(zip(risk_categories.risk_name, risk_categories.risk_category_id))

    cohort_rows = []
    for age_band in C.AGE_BANDS:
        for r in risk_names:
            cohort_rows.append((age_band, risk_id[r], C.AGE_BAND_WEIGHTS[age_band] * C.RISK_CATEGORIES[r][1]))
    cohorts = pd.DataFrame(cohort_rows, columns=["age_band", "risk_category_id", "cohort_weight"])
    cohorts.insert(0, "cohort_id", range(1, len(cohorts) + 1))

    product_rows, tier_rows, map_rows = [], [], []
    tier_id = 0
    for pid, p in enumerate(C.PRODUCTS, start=1):
        product_rows.append((pid, p["insurer"], p["product_name"], p["plan_type"],
                             p["is_primary"], p["protection_share"], p["market_share"]))
        for label, risk, adj in p["tiers"]:
            tier_id += 1
            tier_rows.append((tier_id, pid, label, adj))
            map_rows.append((tier_id, risk_id[risk]))
    products = pd.DataFrame(product_rows, columns=[
        "product_id", "insurer", "product_name", "plan_type", "is_primary", "protection_share", "market_share"])
    tiers_full = pd.DataFrame(tier_rows, columns=["tier_id", "product_id", "tier_label", "tier_adj"])
    tier_risk_map = pd.DataFrame(map_rows, columns=["tier_id", "risk_category_id"])
    return dict(risk_categories=risk_categories, cohorts=cohorts, products=products,
                product_tiers=tiers_full[["tier_id", "product_id", "tier_label"]],
                tier_risk_map=tier_risk_map, _tier_adj=tiers_full[["tier_id", "tier_adj"]])


def build_quotes(ref, rng):
    rows = []
    risk_by_tier = dict(zip(ref["tier_risk_map"].tier_id, ref["tier_risk_map"].risk_category_id))
    risk_name = dict(zip(ref["risk_categories"].risk_category_id, ref["risk_categories"].risk_name))
    adj = dict(zip(ref["_tier_adj"].tier_id, ref["_tier_adj"].tier_adj))
    prod_of_tier = dict(zip(ref["product_tiers"].tier_id, ref["product_tiers"].product_id))
    for tier_id in ref["product_tiers"].tier_id:
        product = C.PRODUCTS[prod_of_tier[tier_id] - 1]
        for age_band in C.AGE_BANDS:
            for sa_band in C.SA_BANDS:
                eff = _effective_rate(product, risk_name[risk_by_tier[tier_id]], age_band, adj[tier_id], sa_band)
                eff *= rng.lognormal(0.0, C.QUOTE_NOISE_SIGMA)
                rows.append((tier_id, age_band, sa_band, round(eff / product["protection_share"], 4)))
    return pd.DataFrame(rows, columns=["tier_id", "age_band", "sa_band", "premium_per_1000"])


def _price_gap_table(ref, quotes):
    """Per product x risk x age band x SA band: how far the average effective rate is above the cheapest product."""
    q = (quotes.merge(ref["product_tiers"], on="tier_id")
               .merge(ref["tier_risk_map"], on="tier_id")
               .merge(ref["products"][["product_id", "protection_share"]], on="product_id"))
    q["eff"] = q.premium_per_1000 * q.protection_share
    cell = q.groupby(["product_id", "risk_category_id", "age_band", "sa_band"], as_index=False).eff.mean()
    best = cell.groupby(["risk_category_id", "age_band", "sa_band"]).eff.transform("min")
    cell["price_gap"] = (cell.eff / best - 1.0).clip(0, C.MAX_PRICE_GAP)
    return cell[["product_id", "risk_category_id", "age_band", "sa_band", "price_gap"]]


def generate_policies(ref, quotes, n, rng):
    products = ref["products"]
    risks = ref["risk_categories"]
    product_id = rng.choice(products.product_id.values, size=n, p=products.market_share.values)
    age_band = rng.choice(list(C.AGE_BAND_WEIGHTS), size=n, p=list(C.AGE_BAND_WEIGHTS.values()))
    risk_category_id = rng.choice(risks.risk_category_id.values, size=n, p=risks.population_share.values)
    sa_band = rng.choice(list(C.SA_BAND_WEIGHTS), size=n, p=list(C.SA_BAND_WEIGHTS.values()))
    lo = np.array([C.AGE_BANDS[a][0] for a in age_band])
    hi = np.array([C.AGE_BANDS[a][1] for a in age_band])
    age_at_issue = lo + (rng.random(n) * (hi - lo + 1)).astype(int)

    # pick one of the insurer's tiers that maps to the customer's risk category
    tier_map = ref["product_tiers"].merge(ref["tier_risk_map"], on="tier_id")
    tier_id = np.zeros(n, dtype=int)
    for (pid, rid), grp in tier_map.groupby(["product_id", "risk_category_id"]):
        mask = (product_id == pid) & (risk_category_id == rid)
        tier_id[mask] = rng.choice(grp.tier_id.values, size=int(mask.sum()))

    pol = pd.DataFrame({
        "policy_id": np.arange(1, n + 1), "product_id": product_id, "tier_id": tier_id,
        "risk_category_id": risk_category_id, "age_band": age_band, "age_at_issue": age_at_issue,
        "sa_band": sa_band,
    })
    pol = pol.merge(ref["cohorts"][["cohort_id", "age_band", "risk_category_id"]],
                    on=["age_band", "risk_category_id"], how="left")
    pol = pol.merge(quotes, on=["tier_id", "age_band", "sa_band"], how="left")
    pol = pol.merge(_price_gap_table(ref, quotes),
                    on=["product_id", "risk_category_id", "age_band", "sa_band"], how="left")
    pol = pol.merge(risks[["risk_category_id", "mortality_multiplier"]], on="risk_category_id", how="left")
    pol["sum_assured"] = pol.sa_band.map(C.SA_BANDS).astype(float)
    pol["annual_premium"] = (pol.premium_per_1000 * pol.sum_assured / 1000).round(2)
    return pol.sort_values("policy_id").reset_index(drop=True)


def simulate_policy_years(pol, rng):
    n = len(pol)
    policy_id = pol.policy_id.values
    age0 = pol.age_at_issue.values
    mult = pol.mortality_multiplier.values
    gap = pol.price_gap.values
    premium = pol.annual_premium.values
    sum_assured = pol.sum_assured.values
    alive = np.ones(n, dtype=bool)
    year_frames, claim_frames = [], []
    for t in range(1, C.MAX_POLICY_YEARS + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        q = np.clip(C.qx(age0[idx] + t - 1) * mult[idx], 0, 1)
        died = rng.random(idx.size) < q
        base = C.BASE_LAPSE_BY_YEAR[t]
        logit = np.log(base / (1 - base)) + C.LAPSE_PRICE_SENSITIVITY * gap[idx]
        lapsed = (~died) & (rng.random(idx.size) < 1 / (1 + np.exp(-logit)))
        year_frames.append(pd.DataFrame({
            "policy_id": policy_id[idx], "policy_year": t, "premium_paid": premium[idx],
            "lapsed": lapsed.astype(int), "died": died.astype(int)}))
        if died.any():
            claim_frames.append(pd.DataFrame({
                "policy_id": policy_id[idx][died], "policy_year": t, "claim_amount": sum_assured[idx][died]}))
        alive[idx[died | lapsed]] = False
    policy_years = pd.concat(year_frames, ignore_index=True)
    claims = (pd.concat(claim_frames, ignore_index=True) if claim_frames
              else pd.DataFrame(columns=["policy_id", "policy_year", "claim_amount"]))
    claims.insert(0, "claim_id", np.arange(1, len(claims) + 1))
    return policy_years, claims


def generate(n_policies=C.N_POLICIES, seed=C.SEED):
    rng = np.random.default_rng(seed)
    ref = build_reference()
    quotes = build_quotes(ref, rng)
    pol = generate_policies(ref, quotes, n_policies, rng)
    policy_years, claims = simulate_policy_years(pol, rng)
    policies = pol[["policy_id", "product_id", "tier_id", "cohort_id", "age_at_issue",
                    "sa_band", "sum_assured", "annual_premium"]]
    return {
        "risk_categories": ref["risk_categories"], "cohorts": ref["cohorts"], "products": ref["products"],
        "product_tiers": ref["product_tiers"], "tier_risk_map": ref["tier_risk_map"],
        "premium_quotes": quotes, "policies": policies, "policy_years": policy_years, "claims": claims,
    }
