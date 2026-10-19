import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config as C

FEATURES = ["policy_year", "price_gap", "age_at_issue", "sum_assured", "rate_per_1000",
            "risk_multiplier", "plan_type", "is_primary"]
TARGET = "lapsed"

_DATASET_SQL = """
    SELECT py.policy_id, py.policy_year, py.lapsed AS lapsed, MIN(v.gap_to_best, ?) AS price_gap,
           pol.age_at_issue, pol.sum_assured,
           pol.annual_premium * 1000.0 / pol.sum_assured AS rate_per_1000,
           rc.mortality_multiplier AS risk_multiplier, pr.plan_type, pr.is_primary
    FROM policy_years py
    JOIN policies pol ON pol.policy_id = py.policy_id
    JOIN products pr ON pr.product_id = pol.product_id
    JOIN cohorts c ON c.cohort_id = pol.cohort_id
    JOIN risk_categories rc ON rc.risk_category_id = c.risk_category_id
    JOIN v_price_position v ON v.product_id = pol.product_id
                           AND v.cohort_id = pol.cohort_id AND v.sa_band = pol.sa_band
    WHERE py.died = 0
"""


def build_dataset(conn):
    """One row per policy-year at risk of lapsing (policies that died that year are excluded)."""
    return pd.read_sql_query(_DATASET_SQL, conn, params=(C.MAX_PRICE_GAP,))


def split_by_policy(df, test_size=0.25, seed=0):
    """Hold out whole policies so no policy appears in both train and test."""
    train_idx, test_idx = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                               .split(df, groups=df.policy_id))
    return df.iloc[train_idx], df.iloc[test_idx]


def make_pipeline(seed=0):
    prep = ColumnTransformer([("plan", OneHotEncoder(handle_unknown="ignore"), ["plan_type"])],
                             remainder="passthrough")
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=3,
                                         min_samples_leaf=200, random_state=seed)
    return Pipeline([("prep", prep), ("clf", clf)])


def oracle_probabilities(df):
    """True lapse probability under the synthetic generator: the best any model could possibly do."""
    base = df.policy_year.map(C.BASE_LAPSE_BY_YEAR).to_numpy(dtype=float)
    logit = np.log(base / (1 - base)) + C.LAPSE_PRICE_SENSITIVITY * df.price_gap.to_numpy()
    return 1.0 / (1.0 + np.exp(-logit))


def calibration_table(y, p, bins=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d.p, bins, duplicates="drop")
    out = d.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"),
                                              n=("y", "size")).reset_index(drop=True)
    return out


def evaluate(pipe, train, test):
    y = test[TARGET].to_numpy()
    p = pipe.predict_proba(test[FEATURES])[:, 1]
    year_rate = train.groupby("policy_year")[TARGET].mean()
    baseline = test.policy_year.map(year_rate).to_numpy(dtype=float)
    oracle = oracle_probabilities(test)
    return {
        "auc": roc_auc_score(y, p), "auc_year_baseline": roc_auc_score(y, baseline), "auc_oracle": roc_auc_score(y, oracle),
        "logloss": log_loss(y, p), "logloss_year_baseline": log_loss(y, baseline), "logloss_oracle": log_loss(y, oracle),
        "brier": brier_score_loss(y, p), "base_rate": float(y.mean()),
        "calibration": calibration_table(y, p),
    }


def importance(pipe, test, n_rows=60_000, seed=0):
    """Permutation importance: drop in held-out AUC when one feature is shuffled."""
    sample = test.sample(min(n_rows, len(test)), random_state=seed)
    res = permutation_importance(pipe, sample[FEATURES], sample[TARGET], scoring="roc_auc",
                                 n_repeats=5, random_state=seed)
    out = pd.DataFrame({"feature": FEATURES, "importance": res.importances_mean, "std": res.importances_std})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def run(conn, seed=0):
    df = build_dataset(conn)
    train, test = split_by_policy(df, seed=seed)
    pipe = make_pipeline(seed).fit(train[FEATURES], train[TARGET])
    metrics = evaluate(pipe, train, test)
    return {"model": pipe, "metrics": metrics, "importance": importance(pipe, test, seed=seed),
            "train_rows": len(train), "test_rows": len(test),
            "train_policies": train.policy_id.nunique(), "test_policies": test.policy_id.nunique()}
