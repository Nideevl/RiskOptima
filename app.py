import os

import pandas as pd
import streamlit as st

from riskoptima import config as C
from riskoptima import db, queries
from riskoptima import lapse_model as L
from riskoptima import figures as F
from riskoptima import reporting as R
from riskoptima import sensitivity as S
from riskoptima.churn import fit_benchmark_curves, load_benchmarks
from riskoptima.profit import Assumptions

N_POLICIES = int(os.environ.get("RISKOPTIMA_POLICIES", C.N_POLICIES))

st.set_page_config(page_title="RiskOptima", layout="wide")


@st.cache_resource(show_spinner="Building the synthetic book and fitting models...")
def load(n):
    conn = db.build(n_policies=n)
    try:
        return {
            "ctx": S.prepare(conn),
            "position": R.price_position(conn),
            "share": queries.run(conn, "competitive_share_by_segment"),
            "lapse": R.lapse_by_year(conn),
        }
    finally:
        conn.close()


@st.cache_resource(show_spinner="Training the lapse classifier...")
def train_lapse(n):
    conn = db.build(n_policies=n)
    try:
        return L.run(conn)
    finally:
        conn.close()


@st.cache_data(show_spinner="Running Monte Carlo...")
def run_mc(n, _ctx):
    return S.monte_carlo(_ctx, n)


data = load(N_POLICIES)
ctx = data["ctx"]
defaults = Assumptions()

st.title("RiskOptima: underwriting and premium decisions")
st.caption("All data is synthetic. The benchmark lapse curve comes from riskoptima/data/benchmark_lapse.csv and is an "
           "assumption until replaced with published persistency data.")

tab_price, tab_retention, tab_discount, tab_sens = st.tabs(
    ["Price position", "Retention", "Discount economics", "Sensitivity"])

with tab_price:
    st.plotly_chart(F.price_gap_heatmap(data["position"]), width="stretch")
    share = data["share"].pivot_table(index="product_name", columns="risk_name", values="competitive_share")
    st.subheader("Share of price cells within 5% of the cheapest product")
    st.dataframe((share * 100).round(0).astype(int).astype(str) + "%", width="stretch")

with tab_retention:
    st.plotly_chart(F.lapse_fit_chart(load_benchmarks(), fit_benchmark_curves()), width="stretch")
    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(F.lapse_by_product_chart(data["lapse"]), width="stretch")
    c2.metric("Price elasticity of lapse (log-odds per 100% price gap)", f"{ctx.beta:.2f}")
    c2.caption(f"Approximate standard error {ctx.beta_se:.2f}, fitted by maximum likelihood on the synthetic book.")

    st.subheader("Lapse classifier")
    if st.button("Train gradient-boosting lapse classifier", key="train_model"):
        trained = train_lapse(N_POLICIES)
        m = trained["metrics"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Held-out AUC (model)", f"{m['auc']:.4f}")
        k2.metric("Policy-year lookup AUC", f"{m['auc_year_baseline']:.4f}")
        k3.metric("Oracle AUC (ceiling)", f"{m['auc_oracle']:.4f}")
        st.caption(f"Trained on {trained['train_policies']:,} policies and tested on {trained['test_policies']:,} held-out "
                   "policies. The oracle uses the true lapse probabilities of the synthetic generator, so it is the best "
                   "any model could do on this data.")
        g1, g2 = st.columns(2)
        g1.plotly_chart(F.calibration_chart(m["calibration"]), width="stretch")
        g2.plotly_chart(F.importance_chart(trained["importance"]), width="stretch")
    else:
        st.info("Train the classifier with the button above to see held-out accuracy, calibration and feature importance.")

with tab_discount:
    st.subheader("Primary insurer, Preferred risk tier")
    left, right = st.columns([1, 2])
    with left:
        cohort = st.selectbox("Cohort", list(ctx.segments.age_band), key="cohort")
        max_discount = st.slider("Largest discount to test", 0.05, 0.30, 0.20, 0.025, key="max_discount")
        tested = st.slider("Discount for the portfolio view", 0.0, 0.20, S.DISCOUNT_TESTED, 0.005, key="tested")
        conv = st.slider("Conversion elasticity (assumption)", 0.0, 6.0, defaults.conversion_elasticity, 0.25, key="conv")
        expense = st.slider("Expense ratio", 0.05, 0.20, defaults.expense_ratio, 0.01, key="expense")
        disc_rate = st.slider("Discount rate", 0.03, 0.15, defaults.discount_rate, 0.01, key="disc_rate")
        lapse_mult = st.slider("Lapse multiplier", 0.5, 1.5, 1.0, 0.05, key="lapse_mult")
        mort_mult = st.slider("Mortality multiplier", 0.8, 1.2, 1.0, 0.01, key="mort_mult")

    assumptions = Assumptions(expense_ratio=expense, discount_rate=disc_rate, price_elasticity=ctx.beta,
                              conversion_elasticity=conv)
    grid = tuple(round(x * 0.025, 3) for x in range(int(round(max_discount / 0.025)) + 1))
    scan = R.discount_scan_table(ctx, grid, assumptions, lapse_mult, mort_mult)
    one = scan[scan.age_band == cohort].reset_index(drop=True)
    params = {**S.base_inputs(ctx), "expense_ratio": expense, "discount_rate": disc_rate, "conversion_elasticity": conv,
              "lapse_multiplier": lapse_mult, "mortality_multiplier": mort_mult}
    portfolio = S.evaluate(ctx, params, discount=tested)

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Break-even CAC, no discount", f"{one.break_even_cac.iloc[0]:,.0f}",
                  f"{one.be_cac_pct_list_premium.iloc[0]:.0f}% of list premium", delta_color="off", delta_arrow="off")
        m2.metric("Best discount for this cohort", f"{one.loc[one.value_index.idxmax(), 'discount'] * 100:.1f}%",
                  f"price gap {one.price_gap.iloc[0] * 100:.1f}%", delta_color="off", delta_arrow="off")
        m3.metric("Policies in cohort", f"{int(one.policies.iloc[0]):,}")
        st.plotly_chart(F.discount_scan_chart(one), width="stretch")
        p1, p2, p3 = st.columns(3)
        p1.metric("Portfolio profit per policy (no discount)", f"{portfolio['clv']:,.0f}")
        p2.metric("Blanket discount gain", f"{portfolio['discount_gain']:,.0f}")
        p3.metric("Targeted discount gain", f"{portfolio['targeted_gain']:,.0f}")
        st.caption("Targeted means only cohorts priced more than 5% above the cheapest competitor receive the discount.")

with tab_sens:
    choice = st.radio("Input ranges", ["Plausible ranges", "Equal +/-20% swings"], horizontal=True, key="ranges")
    outcome = st.selectbox("Outcome", list(S.OUTCOMES), key="outcome")
    ranges = S.tornado_ranges(ctx) if choice == "Plausible ranges" else S.equal_swing_ranges(ctx)
    st.plotly_chart(F.tornado_chart(S.tornado(ctx, ranges, outcome), outcome), width="stretch")

    st.subheader("Monte Carlo")
    n = st.slider("Draws", 500, 5000, 2000, 500, key="mc_n")
    if st.button("Run Monte Carlo", key="run_mc"):
        mc = run_mc(n, ctx)
        st.plotly_chart(F.mc_histogram(mc, outcome, outcome), width="stretch")
        st.dataframe(S.summarise(mc).round(3), width="stretch")
        st.dataframe(S.driver_ranking(mc, outcome).round(3), width="stretch")
    else:
        st.info("Choose the number of draws and press the button.")
