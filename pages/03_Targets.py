# pages/Targets.py
import pandas as pd
import streamlit as st
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from urllib.parse import unquote  # NEW
from menu import menu

from helpers import (
    load_all_data,                 # cached data loader
    get_months, get_latest_month,  # month utilities
    calculate_conversion_metrics,
    calculate_target_score,
    generate_action_plan,
    format_currency,
)

# ---------------- Page Config ----------------
st.set_page_config(page_title="🎯 Targets", layout="wide")
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.switch_page("app.py")
    st.stop()
else:
    menu()

username = st.session_state["username"]
role = st.session_state["role"]

st.header("🎯 Customer Targets & Actions")

st.info(
    "**How to use this list**: Focus on High Priority (🔴) customers first. "
    "The Score column shows overall opportunity — higher scores mean bigger impact potential. "
    "Actions column tells you exactly what to do for each customer."
)

# ---------------- Load data ----------------
sales_df, bookings_df = load_all_data()
if (sales_df is None or sales_df.empty) and (bookings_df is None or bookings_df.empty):
    st.error("No data available. Please check your data connection.")
    st.stop()

# -------- Handle inbound query params from Pipeline --------
params = st.query_params  # e.g., {'customer': 'Acme', 'month': '2025-09'}
prefill_customer = params.get("customer", None)
prefill_month = params.get("month", None)

# ---------------- Sidebar Filters ----------------
with st.sidebar:
    st.header("🔍 Filters & Settings")

    salespeople = ["All"] + sorted(
        list(set(sales_df["Salesperson"].dropna().unique()) |
             set(bookings_df["Salesperson"].dropna().unique()))
    )
    selected_sp = st.selectbox("Salesperson", salespeople, index=0)

    # Month filter (applies to 'current month' context for the targets logic)
    months = get_months(sales_df, bookings_df)
    default_month = get_latest_month(sales_df, bookings_df)
    if not months:
        st.error("No months found in data.")
        st.stop()

    # If month query param is present, try to use it as default
    if prefill_month:
        try:
            _pm = pd.to_datetime(prefill_month + "-01")
            _pm = pd.Timestamp(year=_pm.year, month=_pm.month, day=1)
            if _pm in months:
                default_month = _pm
        except Exception:
            pass

    current_month = st.selectbox(
        "Analysis Month",
        months,
        index=months.index(default_month) if default_month in months else len(months) - 1,
        format_func=lambda x: x.strftime("%Y-%b"),
        help="Targets and gaps are computed relative to this month."
    )

    st.divider()
    st.subheader("Target Filters")
    min_score = st.slider("Min Target Score", 0.0, 1.0, 0.30, 0.05)
    min_gap = st.number_input("Min Gap vs LY (KSH)", value=10_000.0, step=5_000.0)
    show_only_declining = st.checkbox("Show only declining customers (MoM)", value=False)
    top_n = st.number_input("Top N Customers", 10, 200, 30, 5)

# Prefill banner (if navigated from Pipeline)
if prefill_customer:
    prefill_customer = unquote(prefill_customer)
    st.info(f"🔗 Filtered from Pipeline: **{prefill_customer}** · **{current_month.strftime('%Y-%b')}**")

# ---------------- Prep ----------------
prev_month = current_month - DateOffset(months=1)
last_year_month = current_month - DateOffset(years=1)

# Restrict working sets to salesperson (we’ll still compute LY/TY averages from full range)
if selected_sp != "All":
    sales_work = sales_df[sales_df["Salesperson"] == selected_sp].copy()
    bookings_work = bookings_df[bookings_df["Salesperson"] == selected_sp].copy()
else:
    sales_work = sales_df.copy()
    bookings_work = bookings_df.copy()

# ---------------- Cached aggregates for speed ----------------
@st.cache_data(show_spinner=False)
def build_customer_aggregates(sales_work: pd.DataFrame, bookings_work: pd.DataFrame):
    # Sum sales by (Customer, Month)
    s_by_cm = (
        sales_work.groupby(["CustomerName", "Month"], as_index=False)["Sales"].sum()
                  .set_index(["CustomerName", "Month"])["Sales"]
                  .to_dict()
    )
    # Sum bookings by (Customer, Month)
    b_by_cm = (
        bookings_work.groupby(["CustomerName", "Month"], as_index=False)["Bookings"].sum()
                     .set_index(["CustomerName", "Month"])["Bookings"]
                     .to_dict()
    )
    # Yearly average sales per customer
    tmp = sales_work.copy()
    tmp["Year"] = tmp["Month"].dt.year
    avg_by_cy = tmp.groupby(["CustomerName", "Year"])["Sales"].mean().to_dict()

    customers_s = set(sales_work["CustomerName"].unique())
    customers_b = set(bookings_work["CustomerName"].unique())
    return s_by_cm, b_by_cm, avg_by_cy, customers_s, customers_b

s_by_cm, b_by_cm, avg_by_cy, customers_s, customers_b = build_customer_aggregates(sales_work, bookings_work)

# ---------------- Build targets ----------------
targets = []

# Customer scope (respect pre-filter if coming from Pipeline)
if prefill_customer:
    all_customers = {prefill_customer}
else:
    all_customers = (customers_s | customers_b)

if not all_customers:
    st.info("No customers found for the selected filters.")
    st.stop()

progress_bar = st.progress(0)
status_text = st.empty()

for idx, customer in enumerate(sorted(all_customers)):
    progress_bar.progress((idx + 1) / max(len(all_customers), 1))
    status_text.text(f"Analyzing {customer} ({idx + 1}/{len(all_customers)})…")

    # Fast lookups from cached dicts
    cur_sales = float(s_by_cm.get((customer, current_month), 0.0))
    prev_sales = float(s_by_cm.get((customer, prev_month), 0.0))
    ly_sales   = float(s_by_cm.get((customer, last_year_month), 0.0))

    # LY & TY averages from cache
    cy = int(pd.Timestamp(current_month).year)
    ly_avg = avg_by_cy.get((customer, cy - 1), None)
    ty_avg = avg_by_cy.get((customer, cy), None)

    # Bookings snapshots
    cur_bookings = float(b_by_cm.get((customer, current_month), 0.0))
    # Sum FUTURE bookings > current_month
    future_bookings = sum(
        v for (cust, m), v in b_by_cm.items()
        if cust == customer and m > current_month
    )

    # Conversion metrics (historical across months, filtered to salesperson if selected)
    conv_metrics = calculate_conversion_metrics(sales_work, bookings_work, customer, selected_sp)

    # Current-month conversion for badge (reuse Pipeline logic)
    conv_pct_current = (cur_sales / cur_bookings * 100.0) if cur_bookings > 0 else 0.0
    has_conversion_issue = (cur_bookings > 0) and (conv_pct_current < 80.0)

    # Build target row
    row = {
        "Customer": customer,
        "Current Sales": cur_sales,
        "Prev Sales": prev_sales,
        "MoM_Δ": cur_sales - prev_sales,
        "MoM_%": ((cur_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0.0,
        "LY Sales": ly_sales,
        "YoY_Δ": cur_sales - ly_sales,
        "Gap_vs_LY": max((ly_avg or 0) - cur_sales, 0) if ly_avg is not None else 0.0,
        "Gap_vs_TY": max((ty_avg or 0) - cur_sales, 0) if ty_avg is not None else 0.0,
        "potential": (ly_avg if pd.notnull(ly_avg) else ty_avg) or 0.0,
        "Current Bookings": cur_bookings,
        "Future Bookings": future_bookings,
        "booking_pipeline": cur_bookings + future_bookings,
        "conversion_rate": conv_metrics["avg_conversion_rate"],  # historical avg (%)
        "Conversion Trend": conv_metrics["conversion_trend"],
        "Recency_m": 0,  # (Optional) plug in if you compute recency elsewhere
        "current_sales": cur_sales,
        # NEW fields
        "Conversion % (Current)": conv_pct_current,
        "Has Conversion Issue": "❗ Has conversion issue" if has_conversion_issue else "—",
    }

    # Score & actions
    row["Score"] = calculate_target_score(row)
    action_plan = generate_action_plan(row, conv_metrics)
    row["Priority"] = action_plan["priority"]
    row["Recommended Target"] = action_plan["recommended_target"]
    row["Actions"] = ", ".join(action_plan["actions"][:2]) if action_plan["actions"] else "—"

    targets.append(row)

progress_bar.empty()
status_text.empty()

targets_df = pd.DataFrame(targets)

# ---------------- Apply Filters & Display ----------------
if targets_df.empty:
    st.info("No customers match the current criteria.")
    st.stop()

# Numeric safety
for col in ["Score", "MoM_%", "conversion_rate", "Gap_vs_LY", "Gap_vs_TY",
            "Current Sales", "Current Bookings", "Future Bookings", "Recommended Target",
            "Conversion % (Current)"]:
    if col in targets_df.columns:
        targets_df[col] = pd.to_numeric(targets_df[col], errors="coerce").fillna(0)

# Filters
targets_df = targets_df[targets_df["Score"] >= float(min_score)]
if show_only_declining and "MoM_Δ" in targets_df.columns:
    targets_df = targets_df[targets_df["MoM_Δ"] < 0]
if min_gap > 0 and "Gap_vs_LY" in targets_df.columns:
    targets_df = targets_df[targets_df["Gap_vs_LY"] >= float(min_gap)]

# Sort & limit
targets_df = targets_df.sort_values("Score", ascending=False).head(int(top_n))

# Top metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    high_priority = int((targets_df["Priority"] == "High").sum())
    st.metric("🔴 High Priority", high_priority, help="Customers needing immediate action")
with c2:
    med_priority = int((targets_df["Priority"] == "Medium").sum())
    st.metric("🟡 Medium Priority", med_priority, help="Monitor closely this week")
with c3:
    low_priority = int((targets_df["Priority"] == "Low").sum()) if "Low" in targets_df["Priority"].unique() else 0
    st.metric("🟢 Low Priority", low_priority, help="Healthy — keep warm")
with c4:
    total_target_value = float(targets_df["Recommended Target"].sum())
    st.metric("💰 Total Target Value", format_currency(total_target_value),
              help="Combined sales potential if all targets are met")

st.caption(
    f"Scope: **{current_month.strftime('%Y-%b')}** · "
    f"Salesperson: **{selected_sp if selected_sp != 'All' else 'All'}** · "
    f"Min Score: **{min_score:.2f}**"
)

st.divider()

# Display table (pretty)
display_cols = [
    "Customer", "Priority", "Score",
    "Has Conversion Issue",                    # NEW badge column
    "Current Sales", "MoM_%",
    "Gap_vs_LY", "Current Bookings", "Future Bookings",
    "conversion_rate", "Conversion Trend", "Recommended Target", "Actions"
]
display_df = targets_df[display_cols].copy()

display_df["Score"] = display_df["Score"].round(3)
display_df["MoM_%"] = display_df["MoM_%"].round(1)
display_df["conversion_rate"] = display_df["conversion_rate"].round(1)

for c in ["Current Sales", "Gap_vs_LY", "Current Bookings", "Future Bookings", "Recommended Target"]:
    display_df[c] = display_df[c].apply(format_currency)

st.dataframe(
    display_df,
    use_container_width=True,
    height=520,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=1,
            help="Overall opportunity score (0–1). Higher = more important"
        ),
        "Has Conversion Issue": st.column_config.TextColumn(
            "⚠️ Conversion Flag",
            help="Flags customers with current-month conversion <80% (and bookings > 0)"
        ),
        "conversion_rate": st.column_config.ProgressColumn(
            "Conversion % (Hist Avg)", min_value=0, max_value=100,
            help="Historical average conversion (all months)"
        ),
        "Priority": st.column_config.TextColumn(
            "Priority",
            help="🔴 High = Act today, 🟡 Medium = This week, 🟢 Low = Monitor"
        ),
        "MoM_%": st.column_config.NumberColumn(
            "MoM Change %",
            help="Month-over-month sales change. Negative = declining"
        ),
        "Actions": st.column_config.TextColumn(
            "Recommended Actions",
            help="Specific steps to take for this customer"
        ),
    },
    hide_index=True,
)

# ---------------- Export ----------------
csv = display_df.to_csv(index=False)
st.download_button(
    label="📥 Download Target List",
    data=csv,
    file_name=f"targets_{selected_sp}_{current_month.strftime('%Y%m%d')}.csv",
    mime="text/csv",
)
