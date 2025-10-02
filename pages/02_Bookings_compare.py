# pages/Pipeline.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from menu import menu

from helpers import (
    load_all_data,                 # cached data loader
    get_months, get_latest_month,  # month utilities
    format_currency,               # KSH formatting
)

# ---------------- Page Config ----------------
# ---------- Auth ----------
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.switch_page("app.py")
    st.stop()
else:
    menu()
username = st.session_state["username"]
role = st.session_state["role"]

st.set_page_config(page_title="📈 Pipeline Analysis", layout="wide")
st.header("📈 Bookings → Sales Pipeline Analysis")

st.info(
    "**Understanding the Pipeline**: This shows how efficiently your bookings convert to sales. "
    "The funnel visualizes where revenue might be getting stuck."
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
        help="Compute the pipeline for this month."
    )

    month_only = st.checkbox(
        "Show this month only",
        value=True,
        help="If unchecked, numbers include all months up to the selected month (cumulative)."
    )

    show_problems_only = st.checkbox(
        "Show only customers with conversion issues (<80%)",
        value=True
    )

# ---------------- Filter working sets ----------------
if selected_sp != "All":
    s_df = sales_df[sales_df["Salesperson"] == selected_sp].copy()
    b_df = bookings_df[bookings_df["Salesperson"] == selected_sp].copy()
else:
    s_df = sales_df.copy()
    b_df = bookings_df.copy()

if month_only:
    sales_scope = s_df[s_df["Month"] == selected_month].copy()
    bookings_scope = b_df[b_df["Month"] == selected_month].copy()
else:
    sales_scope = s_df[s_df["Month"] <= selected_month].copy()
    bookings_scope = b_df[b_df["Month"] <= selected_month].copy()

# ---------------- Top KPIs & Funnel ----------------
total_bookings = float(bookings_scope["Bookings"].sum())
converted_sales = float(sales_scope["Sales"].sum())
conversion_rate = (converted_sales / total_bookings * 100) if total_bookings > 0 else 0.0
target_sales = total_bookings * 0.80  # 80% target
gap_to_target = max(target_sales - converted_sales, 0.0)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Bookings", format_currency(total_bookings))
with c2:
    st.metric("Converted to Sales", format_currency(converted_sales))
with c3:
    st.metric("Conversion Rate", f"{conversion_rate:.1f}%", help="Target ≥ 80%")
with c4:
    st.metric("Gap to 80% Target", format_currency(gap_to_target))

fig = go.Figure(go.Funnel(
    y=["Total Bookings", "Converted to Sales", "Target (80% of Bookings)"],
    x=[total_bookings, converted_sales, target_sales],
    textinfo="value+percent initial",
    texttemplate='%{value:,.0f}<br>(%{percentInitial})',
    marker=dict(color=[
        "blue",
        ("green" if conversion_rate >= 80 else "orange" if conversion_rate >= 60 else "red"),
        "gold"
    ]),
    connector={"line": {"color": "gray", "dash": "dot", "width": 2}}
))
fig.update_layout(
    title=f"Bookings → Sales Conversion Funnel — {selected_month.strftime('%Y-%b')} ({'Monthly' if month_only else 'Cumulative'})",
    height=420
)
st.plotly_chart(fig, use_container_width=True)

# Insight callouts
if conversion_rate < 60:
    st.error(
        f"⚠️ **Critical**: Only {conversion_rate:.1f}% of bookings convert to sales. "
        f"Potential revenue stuck: {format_currency(total_bookings - converted_sales)}."
    )
elif conversion_rate < 80:
    st.warning(
        f"📊 **Attention**: {conversion_rate:.1f}% conversion rate. "
        f"Gap to target: {format_currency(gap_to_target)}."
    )
else:
    st.success(f"✅ **Excellent**: {conversion_rate:.1f}% conversion rate meets/exceeds the 80% target!")

st.caption(
    f"Scope: **{selected_month.strftime('%Y-%b')}** · "
    f"Salesperson: **{selected_sp if selected_sp != 'All' else 'All'}** · "
    f"View: **{'Month only' if month_only else 'Cumulative'}**"
)

st.divider()
st.subheader("🔍 Customer Pipeline Details")

# ---------------- Customer breakdown ----------------
customer_pipeline = []
customers = sorted(set(bookings_scope["CustomerName"].unique()) |
                   set(sales_scope["CustomerName"].unique()))

for customer in customers:
    cust_b = bookings_scope[bookings_scope["CustomerName"] == customer]
    cust_s = sales_scope[sales_scope["CustomerName"] == customer]

    total_booked = float(cust_b["Bookings"].sum())
    total_sold = float(cust_s["Sales"].sum())
    if total_booked <= 0:
        continue

    conv_pct = (total_sold / total_booked * 100) if total_booked > 0 else 0.0
    status = "✅ Good" if conv_pct >= 80 else "⚠️ Needs Attention" if conv_pct >= 60 else "🔴 Critical"

    customer_pipeline.append({
        "Customer": customer,
        "Total Bookings": total_booked,
        "Total Sales": total_sold,
        "Conversion %": conv_pct,
        "Gap": total_booked - total_sold,
        "Status": status
    })

pipeline_df = pd.DataFrame(customer_pipeline)

if pipeline_df.empty:
    st.info("No customer pipeline rows found for the selected scope.")
    st.stop()

# Optional problem-only filter
if show_problems_only:
    pipeline_df = pipeline_df[pipeline_df["Conversion %"] < 80]

# Sort by biggest gap
pipeline_df = pipeline_df.sort_values("Gap", ascending=False).reset_index(drop=True)

# Summary row KPIs
col1, col2, col3 = st.columns(3)
with col1:
    problem_customers = int((pipeline_df["Conversion %"] < 80).sum())
    st.metric("Problem Customers", problem_customers, help="Customers with conversion below 80%")
with col2:
    total_gap = float(pipeline_df["Gap"].sum())
    st.metric("Total Revenue Gap", format_currency(total_gap), help="Potential revenue stuck in pipeline")
with col3:
    avg_conv = float(pipeline_df["Conversion %"].mean()) if not pipeline_df.empty else 0.0
    st.metric("Avg Customer Conversion", f"{avg_conv:.1f}%", help="Across the listed customers")

st.divider()

# Pretty table
display_df = pipeline_df.copy()
for c in ["Total Bookings", "Total Sales", "Gap"]:
    display_df[c] = display_df[c].apply(format_currency)
display_df["Conversion %"] = display_df["Conversion %"].round(1)

st.dataframe(
    display_df,
    use_container_width=True,
    column_config={
        "Conversion %": st.column_config.ProgressColumn(
            "Conversion %",
            min_value=0, max_value=100,
            help="Percentage of bookings converted to sales"
        ),
        "Status": st.column_config.TextColumn(
            "Status",
            help="✅ Good (≥80%), ⚠️ 60–79%, 🔴 <60%"
        ),
        "Gap": st.column_config.TextColumn(
            "Revenue Gap",
            help="Amount of bookings not yet converted to sales"
        ),
    },
    hide_index=True,
)

# ---------------- Focus list (actionable) ----------------
st.subheader("🧭 Focus List (Top 5 biggest gaps)")
top5 = pipeline_df.head(5).copy()
if not top5.empty:
    for _, r in top5.iterrows():
        badge = "🚨" if r["Conversion %"] < 60 else "⚠️"
        st.write(
            f"{badge} **{r['Customer']}** — Gap: {format_currency(r['Gap'])} "
            f"· Conv: {r['Conversion %']:.1f}% "
            f"→ _Action_: confirm fulfillment dates, unblock dispatch, and follow-up on approvals."
        )
else:
    st.success("No major issues — great job!")

# ---------------- Export ----------------
csv = display_df.to_csv(index=False)
st.download_button(
    "📥 Download Customer Pipeline",
    data=csv,
    file_name=f"pipeline_{(selected_sp if selected_sp!='All' else 'ALL')}_{selected_month.strftime('%Y%m')}{'_M' if month_only else '_CUM'}.csv",
    mime="text/csv",
)
