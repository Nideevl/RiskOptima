import pytest

from riskoptima import db
from riskoptima.datagen import generate


@pytest.fixture(scope="session")
def data():
    return generate(n_policies=6000, seed=7)


@pytest.fixture(scope="session")
def conn():
    c = db.build(":memory:", n_policies=6000, seed=7)
    yield c
    c.close()


@pytest.fixture(scope="session")
def big_conn():
    c = db.build(":memory:", n_policies=40000, seed=11)
    yield c
    c.close()


@pytest.fixture(scope="session")
def ctx(big_conn):
    from riskoptima import sensitivity
    return sensitivity.prepare(big_conn)
