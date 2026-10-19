import plotly.graph_objects as go
import pytest

from riskoptima import figures as F
from riskoptima import reporting as R
from riskoptima import sensitivity as S
from riskoptima.churn import fit_benchmark_curves, load_benchmarks


@pytest.fixture(scope="module")
def pos(big_conn):
    return R.price_position(big_conn)


def test_heatmap_has_every_product_and_cohort(pos):
    fig = F.price_gap_heatmap(pos)
    assert isinstance(fig, go.Figure)
    heat = fig.data[0]
    assert len(heat.y) == 4 and len(heat.x) == 12
    assert (heat.z >= 0).all()


def test_lapse_fit_chart_has_points_and_line_per_risk():
    fig = F.lapse_fit_chart(load_benchmarks(), fit_benchmark_curves())
    assert len(fig.data) == 6
    assert {t.mode for t in fig.data} == {"markers", "lines"}


def test_lapse_by_product_chart(big_conn):
    assert len(F.lapse_by_product_chart(R.lapse_by_year(big_conn)).data) == 4


def test_discount_chart_marks_best_discount(ctx):
    scan = R.discount_scan_table(ctx)
    one = scan[scan.age_band == "25-34"]
    fig = F.discount_scan_chart(one)
    assert len(fig.data) == 2
    best = one.loc[one.value_index.idxmax(), "discount"] * 100
    assert fig.layout.shapes[0].x0 == pytest.approx(best)


def test_tornado_chart_sorted_with_largest_swing_last(ctx):
    t = S.tornado(ctx, outcome="clv")
    fig = F.tornado_chart(t, "clv")
    assert list(fig.data[0].y)[-1] == t.input.iloc[0]


def test_mc_histogram(ctx):
    mc = S.monte_carlo(ctx, n=200, seed=3)
    fig = F.mc_histogram(mc, "targeted_gain", "targeted gain")
    assert len(fig.data[0].x) == 200


def test_calibration_chart_has_diagonal_and_points():
    import pandas as pd
    cal = pd.DataFrame({"mean_predicted": [0.04, 0.08, 0.15], "observed_rate": [0.045, 0.07, 0.16], "n": [10, 10, 10]})
    fig = F.calibration_chart(cal)
    assert len(fig.data) == 2 and list(fig.data[1].x) == pytest.approx([4.0, 8.0, 15.0])


def test_importance_chart_orders_features_by_importance():
    import pandas as pd
    imp = pd.DataFrame({"feature": ["a", "b", "c"], "importance": [0.1, 0.5, 0.0], "std": [0.01, 0.02, 0.0]})
    assert list(F.importance_chart(imp).data[0].y)[-1] == "b"
