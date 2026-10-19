from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("RISKOPTIMA_POLICIES", "6000")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _metric(at, label):
    return next(m for m in at.metric if m.label == label).value


def test_app_loads_with_four_tabs(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Price position", "Retention", "Discount economics", "Sensitivity"]
    assert app.title[0].value.startswith("RiskOptima")


def test_conversion_slider_changes_targeted_gain(app):
    app.slider(key="conv").set_value(1.0).run()
    low = _metric(app, "Targeted discount gain")
    app.slider(key="conv").set_value(5.0).run()
    high = _metric(app, "Targeted discount gain")
    assert not app.exception
    assert low != high


def test_cohort_selection_changes_break_even(app):
    first = _metric(app, "Break-even CAC, no discount")
    app.selectbox(key="cohort").select("55-64").run()
    assert not app.exception
    assert _metric(app, "Break-even CAC, no discount") != first


def test_monte_carlo_button_renders_results(app):
    assert any("press the button" in i.value for i in app.info)
    app.slider(key="mc_n").set_value(500).run()
    app.button(key="run_mc").click().run()
    assert not app.exception
    assert len(app.dataframe) >= 3  # share table + summary + driver ranking


def test_train_button_shows_model_metrics(app):
    assert any("button above" in i.value for i in app.info)
    app.button(key="train_model").click().run()
    assert not app.exception
    auc = float(_metric(app, "Held-out AUC (model)"))
    oracle = float(_metric(app, "Oracle AUC (ceiling)"))
    assert 0.5 < auc <= 1.0 and oracle >= auc - 0.02
