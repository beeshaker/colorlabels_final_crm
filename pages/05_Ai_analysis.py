# pages/AI_Insights.py
import os
from datetime import datetime
import pandas as pd
import streamlit as st
from openai import OpenAI
from menu import menu

from helpers import (
    load_all_data,                 # cached data loader
    get_months, get_latest_month,  # month utilities
    format_currency,               # KSH formatting
)

# ---------------- Page Config ----------------
st.set_page_config(page_title="🤖 AI Insights", layout="wide")
st.header("🤖 AI-Powered Insights & Recommendations")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.switch_page("app.py")
    st.stop()
else:
    menu()

username = st.session_state["username"]
role = st.session_state["role"]

st.info(
    "**AI Analysis**: This section uses pattern recognition to identify opportunities and risks in your sales data. "
    "You can also generate a concise executive summary if an OpenAI API key is configured in `st.secrets`."
)

# ---------------- Load data ----------------
sales_df, bookings_df = load_all_data()
if (sales_df is None or sales_df.empty) and (bookings_df is None or bookings_df.empty):
    st.error("No data available. Please check your data connection.")
    st.stop()

# ---------------- Sidebar Filters ----------------
with st.sidebar:
    st.header("🔍 Filters")

    salespeople = ["All"] + sorted(
        list(set(sales_df["Salesperson"].dropna().unique()) |
             set(bookings_df["Salesperson"].dropna().unique()))
    )
    selected_sp = st.selectbox("Salesperson", salespeople, index=0)

    months = get_months(sales_df, bookings_df)
    default_month = get_latest_month(sales_df, bookings_df)
    if not months:
        st.error("No months found in data.")
        st.stop()

    selected_month = st.selectbox(
        "Analysis Month",
        months,
        index=months.index(default_month) if default_month in months else len(months) - 1,
        format_func=lambda x: x.strftime("%Y-%b"),
        help="Insights will use this month as the reference point."
    )

    month_only = st.checkbox(
        "Use this month only",
        value=True,
        help="If unchecked, numbers include all months up to the selected month (cumulative)."
    )

# ---------------- Scope the data ----------------
if selected_sp != "All":
    s_df = sales_df[sales_df["Salesperson"] == selected_sp].copy()
    b_df = bookings_df[bookings_df["Salesperson"] == selected_sp].copy()
else:
    s_df = sales_df.copy()
    b_df = bookings_df.copy()

if month_only:
    ai_sales = s_df[s_df["Month"] == selected_month].copy()
    ai_bookings = b_df[b_df["Month"] == selected_month].copy()
else:
    ai_sales = s_df[s_df["Month"] <= selected_month].copy()
    ai_bookings = b_df[b_df["Month"] <= selected_month].copy()

# A few handy totals
total_sales = float(ai_sales["Sales"].sum())
total_bookings = float(ai_bookings["Bookings"].sum())
overall_conversion = (total_sales / total_bookings * 100) if total_bookings > 0 else 0.0

# Recent vs previous month (always month-on-month comparison)
recent_month = selected_month
prev_month = (months[months.index(selected_month) - 1]
              if months.index(selected_month) > 0 else selected_month)

# ---------------- Quick patterns ----------------
st.subheader("📊 Key Patterns Detected")

patterns = []

# Conversion analysis
if overall_conversion < 60:
    patterns.append({
        "type": "🔴 Critical",
        "insight": (
            f"Low overall conversion rate (**{overall_conversion:.1f}%**). "
            f"Potentially stuck revenue: **{format_currency(total_bookings - total_sales)}**."
        ),
        "action": "1) Triage fulfillment delays\n2) Call top 5 customers with pending bookings\n3) Unblock dispatch approvals",
        "impact": "High"
    })
elif overall_conversion < 80:
    gap_to_target = (total_bookings * 0.8) - total_sales
    patterns.append({
        "type": "🟡 Warning",
        "insight": (
            f"Moderate conversion (**{overall_conversion:.1f}%**). "
            f"Gap to 80% target: **{format_currency(max(gap_to_target, 0))}**."
        ),
        "action": "1) Weekly pipeline review\n2) SLA on order processing\n3) Automated follow-ups on pending items",
        "impact": "Medium"
    })
else:
    patterns.append({
        "type": "✅ Good",
        "insight": (
            f"Strong conversion (**{overall_conversion:.1f}%**). "
            f"Exceeds 80% target by **{(overall_conversion - 80):.1f}pp**."
        ),
        "action": "1) Document best practices\n2) Share with team\n3) Maintain quality controls",
        "impact": "Positive"
    })

# Month-on-month customer loss
recent_customers = set(s_df[s_df["Month"] == recent_month]["CustomerName"].unique())
prev_customers = set(s_df[s_df["Month"] == prev_month]["CustomerName"].unique())
lost_customers = prev_customers - recent_customers

if len(lost_customers) > 0:
    lost_revenue = float(
        s_df[(s_df["CustomerName"].isin(lost_customers)) & (s_df["Month"] == prev_month)]["Sales"].sum()
    )
    patterns.append({
        "type": "⚠️ Alert",
        "insight": f"**{len(lost_customers)}** customers inactive this month. Potential loss: **{format_currency(lost_revenue)}**.",
        "action": "Win-back playbook: 1) call, 2) small incentive, 3) delivery assurance",
        "impact": "High"
    })

# Month-on-month booking trend
recent_bookings = float(b_df[b_df["Month"] == recent_month]["Bookings"].sum())
prev_bookings = float(b_df[b_df["Month"] == prev_month]["Bookings"].sum())
if prev_bookings > 0:
    booking_trend = ((recent_bookings - prev_bookings) / prev_bookings) * 100.0
    if booking_trend <= -20:
        patterns.append({
            "type": "📉 Trend Alert",
            "insight": f"Bookings down **{abs(booking_trend):.1f}%** vs last month — forward revenue risk.",
            "action": "Outreach blitz + promo test; audit pricing vs competitors",
            "impact": "High"
        })

# Render patterns
if not patterns:
    st.success("No warnings — metrics look steady. 🎉")

for pattern in patterns:
    impact_color = "🔴" if pattern["impact"] == "High" else "🟡" if pattern["impact"] == "Medium" else "🟢"
    with st.expander(f"{pattern['type']} - {impact_color} Impact"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**📋 Insight Details**")
            st.write(pattern["insight"])
        with c2:
            st.write("**🎯 Recommended Actions**")
            st.write(pattern["action"])

st.caption(
    f"Scope: **{selected_month.strftime('%Y-%b')}** · "
    f"Salesperson: **{selected_sp if selected_sp != 'All' else 'All'}** · "
    f"View: **{'Month only' if month_only else 'Cumulative'}**"
)

st.divider()

# ---------------- Data-Driven Recommendations ----------------
st.subheader("🎯 Data-Driven Recommendations")

tab1, tab2, tab3 = st.tabs(["💰 Revenue Opportunities", "⚠️ Risk Mitigation", "📈 Growth Strategy"])

with tab1:
    st.write("### High-Value Opportunities")
    opportunity_list = []
    for customer in ai_bookings["CustomerName"].unique():
        cust_b = float(ai_bookings[ai_bookings["CustomerName"] == customer]["Bookings"].sum())
        cust_s = float(ai_sales[ai_sales["CustomerName"] == customer]["Sales"].sum())
        if cust_b > 0:
            conv_rate = (cust_s / cust_b) * 100.0
            if conv_rate < 70 and cust_b > 50_000:
                opportunity_list.append({
                    "Customer": customer,
                    "Bookings": cust_b,
                    "Sales": cust_s,
                    "Potential": cust_b - cust_s,
                    "Conversion": conv_rate,
                    "Action": "Expedite fulfillment" if conv_rate > 50 else "Investigate blockers"
                })
    opportunity_df = pd.DataFrame(opportunity_list)
    if not opportunity_df.empty:
        opportunity_df = opportunity_df.sort_values("Potential", ascending=False).head(10)
        for c in ["Bookings", "Sales", "Potential"]:
            opportunity_df[c] = opportunity_df[c].apply(format_currency)
        opportunity_df["Conversion"] = opportunity_df["Conversion"].round(1).astype(str) + "%"
        st.dataframe(opportunity_df, use_container_width=True, hide_index=True)
        total_opportunity = sum([row["Potential"] for row in opportunity_list])
        st.success(f"**Total Opportunity**: {format_currency(total_opportunity)} in untapped revenue")
    else:
        st.info("All customers are converting well. Focus on new customer acquisition.")

with tab2:
    st.write("### Risk Mitigation Priorities")
    risk_list = []
    # compare this month vs previous month for each customer
    for customer in s_df["CustomerName"].unique():
        cust_recent = float(s_df[(s_df["CustomerName"] == customer) & (s_df["Month"] == recent_month)]["Sales"].sum())
        cust_prev = float(s_df[(s_df["CustomerName"] == customer) & (s_df["Month"] == prev_month)]["Sales"].sum())
        if cust_prev > 0:
            decline_pct = ((cust_prev - cust_recent) / cust_prev) * 100.0
            if decline_pct > 30:  # more than 30% decline
                risk_list.append({
                    "Customer": customer,
                    "Previous": cust_prev,
                    "Current": cust_recent,
                    "Decline": cust_prev - cust_recent,
                    "Decline %": decline_pct,
                    "Risk Level": "🔴 High" if decline_pct > 50 else "🟡 Medium"
                })
    risk_df = pd.DataFrame(risk_list)
    if not risk_df.empty:
        risk_df = risk_df.sort_values("Decline", ascending=False).head(10)
        for c in ["Previous", "Current", "Decline"]:
            risk_df[c] = risk_df[c].apply(format_currency)
        risk_df["Decline %"] = risk_df["Decline %"].round(1).astype(str) + "%"
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        total_at_risk = sum([row["Decline"] for row in risk_list])
        st.warning(f"**Revenue at Risk**: {format_currency(total_at_risk)} from declining customers")
    else:
        st.success("No significant customer declines detected. Sales momentum is strong.")

with tab3:
    st.write("### Growth Strategy Recommendations")
    # rollups based on scoped (month or cumulative) AI data
    avg_customer_value = ai_sales.groupby("CustomerName")["Sales"].sum().mean() if not ai_sales.empty else 0.0
    top20_count = max(int(len(ai_sales["CustomerName"].unique()) * 0.2), 1)
    top_20_pct_revenue = (
        ai_sales.groupby("CustomerName")["Sales"].sum().nlargest(top20_count).sum() / total_sales * 100
        if total_sales > 0 and not ai_sales.empty else 0.0
    )
    conversion_gap = (80 - overall_conversion) if overall_conversion < 80 else 0.0
    customer_count = len(ai_sales["CustomerName"].unique())

    colA, colB = st.columns(2)
    with colA:
        st.metric("Avg Customer Value", format_currency(avg_customer_value))
        st.metric("Customer Concentration", f"{top_20_pct_revenue:.1f}%",
                  help="Share of revenue from the top 20% of customers")
    with colB:
        st.metric("Active Customers", customer_count)
        if conversion_gap > 0:
            st.metric("Conversion Gap", f"{conversion_gap:.1f}%",
                      help="Gap to the 80% conversion target")

    st.write("**🚀 Strategic Recommendations:**")
    recs = []
    if top_20_pct_revenue > 70:
        recs.append("• **Diversify customer base** — Top-20% concentration is high (>70%).")
    if overall_conversion < 70:
        recs.append("• **Fix fulfillment gaps** — Raise conversion to 80%+ with SLAs & root-cause fixes.")
    if avg_customer_value < 100_000:
        recs.append("• **Increase order values** — Upsell/cross-sell bundles and minimum order incentives.")
    if len(lost_customers) > 5:
        recs.append(f"• **Win-back campaign** — {len(lost_customers)} recently inactive customers; quick wins possible.")
    if recs:
        for r in recs:
            st.write(r)
    else:
        st.success("Portfolio looks balanced — keep optimizing execution. 🎯")

# ---------------- AI Executive Summary ----------------
st.divider()
st.subheader("💡 AI-Generated Executive Summary")

# ---- Read OpenAI config from st.secrets (no env vars) ----
_openai = st.secrets.get("openai", {})
OPENAI_MODEL = _openai.get("model", "gpt-4-mini")
OPENAI_API_KEY = _openai.get("api_key")
OPENAI_BASE_URL = _openai.get("base_url", None)  # optional

if OPENAI_API_KEY:
    if st.button("Generate AI Summary", key="ai_summary_btn", type="primary"):
        with st.spinner("Generating personalized insights..."):
            # Init client using secrets
            client_kwargs = {"api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                client_kwargs["base_url"] = OPENAI_BASE_URL
            client = OpenAI(**client_kwargs)

            # Build context string
            all_customers = sorted(set(ai_sales["CustomerName"].unique()) | set(ai_bookings["CustomerName"].unique()))
            # Recompute quick opp count
            opp_count = 0
            if not ai_bookings.empty:
                _tmp = []
                for customer in ai_bookings["CustomerName"].unique():
                    cb = float(ai_bookings[ai_bookings["CustomerName"] == customer]["Bookings"].sum())
                    cs = float(ai_sales[ai_sales["CustomerName"] == customer]["Sales"].sum())
                    if cb > 0 and (cs / cb * 100.0) < 70 and cb > 50_000:
                        _tmp.append(1)
                opp_count = sum(_tmp)

            context = f"""
            Sales Insights for {selected_sp if selected_sp != 'All' else 'All Salespeople'} — {selected_month.strftime('%Y-%b')} ({'Monthly' if month_only else 'Cumulative'}):

            Totals:
            - Sales: {format_currency(total_sales)}
            - Bookings: {format_currency(total_bookings)}
            - Conversion: {overall_conversion:.1f}% (Target: 80%)
            - Active Customers: {len(all_customers)}
            - Inactive MoM: {len(lost_customers)}
            - High-Value Opportunities: {opp_count}

            Please provide:
            1) One key insight
            2) The single most important action this week
            3) Expected impact in numbers (≤150 words, specific & actionable)
            """

            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system",
                         "content": "You are a sales analytics expert. Be concise, numeric, and action-oriented."},
                        {"role": "user", "content": context}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                ai_summary = resp.choices[0].message.content
                st.success("**AI Executive Summary:**")
                st.markdown(ai_summary)

                st.download_button(
                    "📥 Download AI Summary",
                    ai_summary,
                    file_name=f"ai_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"AI generation failed: {e}")
else:
    st.info("Add your OpenAI credentials to **st.secrets**:\n\n"
            "```toml\n[openai]\napi_key = \"sk-...\"\nmodel = \"gpt-4-mini\"\n# base_url = \"https://api.openai.com/v1\"  # optional\n```")

# ---------------- Footer / Export ----------------
st.divider()

st.info("**Tip**: Review these insights weekly in team meetings for strategic planning.")

st.caption("""
**Metric Notes**
- **Conversion Rate** = Sales ÷ Bookings × 100 (target ≥ 80%).
- **Opportunities** highlight high bookings with low conversion.
- **Risk** flags customers with >30% month-on-month decline.
""")

# Quick export snapshot (text)
if st.sidebar.button("📊 Export Insights Snapshot", type="primary"):
    snapshot = []
    snapshot.append("AI INSIGHTS SNAPSHOT")
    snapshot.append("=" * 50)
    snapshot.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    snapshot.append(f"Salesperson: {selected_sp}")
    snapshot.append(f"Month: {selected_month.strftime('%Y-%b')} ({'Monthly' if month_only else 'Cumulative'})")
    snapshot.append(f"Total Sales: {format_currency(total_sales)}")
    snapshot.append(f"Total Bookings: {format_currency(total_bookings)}")
    snapshot.append(f"Overall Conversion: {overall_conversion:.1f}%")
    snapshot.append(f"Inactive Customers (MoM): {len(lost_customers)}")
    snapshot.append("")

    if patterns:
        snapshot.append("Key Patterns:")
        for p in patterns:
            snapshot.append(f"- {p['type']}: {p['insight']}")
        snapshot.append("")

    st.download_button(
        label="💾 Download Snapshot",
        data="\n".join(snapshot),
        file_name=f"insights_{(selected_sp if selected_sp!='All' else 'ALL')}_{selected_month.strftime('%Y%m')}.txt",
        mime="text/plain"
    )
