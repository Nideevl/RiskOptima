import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C
from .churn import logistic_hazard

_RISK_ORDER = {name: i for i, name in enumerate(C.RISK_CATEGORIES)}


def price_gap_heatmap(price_position):
    """Average % above the cheapest effective rate, by product and cohort."""
    df = price_position.copy()
    df["cohort"] = df.risk_name + " | " + df.age_band
    df["order"] = df.risk_name.map(_RISK_ORDER)
    order = df.drop_duplicates("cohort").sort_values(["order", "age_band"]).cohort.tolist()
    pivot = df.pivot_table(index="product_name", columns="cohort", values="gap_to_best", aggfunc="mean")[order]
    fig = go.Figure(go.Heatmap(
        z=pivot.values * 100, x=order, y=pivot.index.tolist(), zmin=0, colorscale="RdYlGn_r",
        colorbar=dict(title="% above cheapest"), hovertemplate="%{y}<br>%{x}<br>%{z:.1f}% above cheapest<extra></extra>"))
    fig.update_layout(title="Price position by product and cohort", height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def lapse_fit_chart(benchmarks, curves):
    """Benchmark lapse points with the fitted logistic curve for each risk category."""
    fig = go.Figure()
    grid = np.linspace(1, 10, 100)
    for i, (risk, curve) in enumerate(curves.items()):
        colour = f"hsl({i * 110},65%,45%)"
        pts = benchmarks[benchmarks.risk_name == risk]
        fig.add_trace(go.Scatter(x=pts.policy_year, y=pts.lapse_rate * 100, mode="markers", name=f"{risk} benchmark",
                                 marker=dict(color=colour)))
        fig.add_trace(go.Scatter(x=grid, y=logistic_hazard(grid, curve.a, curve.b) * 100, mode="lines",
                                 name=f"{risk} fitted", line=dict(color=colour)))
    fig.update_layout(title="Annual lapse rate by policy year: benchmark vs fitted logistic curve",
                      xaxis_title="Policy year", yaxis_title="Lapse rate (%)", height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def lapse_by_product_chart(lapse):
    fig = go.Figure()
    for name, g in lapse.groupby("product_name"):
        fig.add_trace(go.Scatter(x=g.policy_year, y=g.lapse_rate * 100, mode="lines+markers", name=name))
    fig.update_layout(title="Observed lapse rate by policy year (synthetic book)", xaxis_title="Policy year",
                      yaxis_title="Lapse rate (%)", height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def discount_scan_chart(scan):
    """Break-even CAC and volume-adjusted value across discount levels for one cohort."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = scan.discount * 100
    fig.add_trace(go.Scatter(x=x, y=scan.value_index, mode="lines+markers", name="Volume-adjusted value (INR)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=scan.be_cac_pct_list_premium, mode="lines+markers", name="Break-even CAC (% of list premium)",
                             line=dict(dash="dot")), secondary_y=True)
    best = scan.loc[scan.value_index.idxmax(), "discount"] * 100
    fig.add_vline(x=best, line_dash="dash", line_color="grey", annotation_text=f"best: {best:.1f}%")
    fig.update_xaxes(title_text="Premium discount (%)")
    fig.update_yaxes(title_text="Value per policy sold (INR)", secondary_y=False)
    fig.update_yaxes(title_text="% of list premium", tickformat=".0f", secondary_y=True)
    fig.update_layout(title="Discount economics", height=400, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def tornado_chart(tornado_table, outcome_label):
    """Change in outcome from the base case when each input moves to its low and high value."""
    t = tornado_table.sort_values("swing")
    base = t.base_outcome.iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=t.input, x=t.outcome_at_low - base, orientation="h", name="input at low value"))
    fig.add_trace(go.Bar(y=t.input, x=t.outcome_at_high - base, orientation="h", name="input at high value"))
    fig.update_layout(barmode="overlay", title=f"Tornado: change in {outcome_label} from base ({base:,.0f})",
                      xaxis_title="Change in INR per policy sold", height=380, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def mc_histogram(mc, outcome, label):
    fig = go.Figure(go.Histogram(x=mc[outcome], nbinsx=60, name=label))
    fig.add_vline(x=float(mc[outcome].mean()), line_dash="dash", annotation_text="mean")
    if mc[outcome].min() < 0 < mc[outcome].max():
        fig.add_vline(x=0, line_color="red")
    fig.update_layout(title=f"Monte Carlo distribution: {label} ({len(mc):,} draws)", xaxis_title="INR per policy sold",
                      yaxis_title="Draws", height=380, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def calibration_chart(calibration):
    """Predicted vs observed lapse rate across deciles of predicted risk."""
    top = float(max(calibration.mean_predicted.max(), calibration.observed_rate.max())) * 100 * 1.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect calibration", line=dict(dash="dash", color="grey")))
    fig.add_trace(go.Scatter(x=calibration.mean_predicted * 100, y=calibration.observed_rate * 100, mode="markers+lines",
                             name="held-out deciles"))
    fig.update_layout(title="Calibration on held-out policies", xaxis_title="Predicted lapse rate (%)",
                      yaxis_title="Observed lapse rate (%)", height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def importance_chart(importance):
    t = importance.sort_values("importance")
    fig = go.Figure(go.Bar(y=t.feature, x=t.importance, orientation="h", error_x=dict(type="data", array=t["std"])))
    fig.update_layout(title="Permutation importance (drop in held-out AUC)", xaxis_title="AUC drop when shuffled",
                      height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig
