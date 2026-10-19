import numpy as np
import pytest

from riskoptima import config as C
from riskoptima import lapse_model as L


@pytest.fixture(scope="module")
def dataset(big_conn):
    return L.build_dataset(big_conn)


@pytest.fixture(scope="module")
def result(big_conn):
    return L.run(big_conn, seed=0)


def test_dataset_only_has_policy_years_at_risk(big_conn, dataset):
    at_risk = big_conn.execute("SELECT COUNT(*) FROM policy_years WHERE died = 0").fetchone()[0]
    assert len(dataset) == at_risk
    assert set(dataset.lapsed.unique()) <= {0, 1}
    assert dataset.price_gap.between(0, C.MAX_PRICE_GAP).all()
    assert set(L.FEATURES) <= set(dataset.columns)


def test_split_never_shares_a_policy(dataset):
    train, test = L.split_by_policy(dataset, test_size=0.25, seed=1)
    assert set(train.policy_id).isdisjoint(set(test.policy_id))
    assert len(train) + len(test) == len(dataset)
    share = test.policy_id.nunique() / dataset.policy_id.nunique()
    assert share == pytest.approx(0.25, abs=0.01)


def test_oracle_probabilities_are_valid_and_follow_the_generator(dataset):
    p = L.oracle_probabilities(dataset.head(2000))
    assert ((p > 0) & (p < 1)).all()
    # at price parity the oracle equals the configured base hazard for that year
    parity = dataset[dataset.price_gap == 0].head(500)
    expected = parity.policy_year.map(C.BASE_LAPSE_BY_YEAR).to_numpy()
    assert np.allclose(L.oracle_probabilities(parity), expected)


def test_model_reaches_the_oracle_ceiling(result):
    m = result["metrics"]
    assert m["auc"] > 0.6
    assert m["auc_oracle"] - m["auc"] < 0.01
    assert m["logloss"] - m["logloss_oracle"] < 0.001
    assert m["auc"] >= m["auc_year_baseline"] - 0.002


def test_predictions_are_calibrated(result):
    cal = result["metrics"]["calibration"]
    gap = (cal.mean_predicted - cal.observed_rate).abs()
    assert gap.mean() < 0.006 and gap.max() < 0.02
    assert abs(np.average(cal.mean_predicted, weights=cal.n) - result["metrics"]["base_rate"]) < 0.002


def test_importance_recovers_the_true_drivers(result):
    imp = result["importance"].set_index("feature").importance
    assert imp.idxmax() == "policy_year"
    assert imp["policy_year"] > 20 * imp.drop("policy_year").abs().max()
    assert imp["price_gap"] > 0
    assert imp["price_gap"] > imp[["age_at_issue", "sum_assured", "rate_per_1000"]].max()


def test_pipeline_is_deterministic_for_a_seed(dataset):
    sample = dataset.sample(30_000, random_state=0)
    a = L.make_pipeline(seed=3).fit(sample[L.FEATURES], sample[L.TARGET])
    b = L.make_pipeline(seed=3).fit(sample[L.FEATURES], sample[L.TARGET])
    assert np.array_equal(a.predict_proba(sample[L.FEATURES]), b.predict_proba(sample[L.FEATURES]))
