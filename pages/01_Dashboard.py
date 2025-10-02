# pages/01_Executive_Dashboard.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from menu import menu
from helpers.data import load_all_data, get_latest_month, get_months, filter_by_salesperson_month

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")
st.header("📊 Executive Dashboard")

# ---------- Auth ----------
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.switch_page("app.py")
    st.stop()
else:
    menu()

# ===================== Load & Prep =====================
sales_df, bookings_df = load_all_data()

if (sales_df is None or sales_df.empty) and (bookings_df is None or bookings_df.empty):
    st.info("No data available yet. Upload or connect your sources.")
    st.stop()

# Months (as Timestamp month-start) & Latest
all_months = get_months(sales_df, bookings_df)
latest_month = get_latest_month(sales_df, bookings_df)

# Salespeople list
sp_list = ["All"]
if sales_df is not None and not sales_df.empty:
    sp_list.extend(sorted(sales_df["Salesperson"].dropna().unique()))
elif bookings_df is not None and not bookings_df.empty:
    sp_list.extend(sorted(bookings_df["Salesperson"].dropna().unique()))

# ===================== Sidebar Filters =====================
with st.sidebar:
    st.subheader("Filters")
    selected_sp = st.selectbox("Salesperson", sp_list, index=0)
    month_options = ["Latest"] + [m for m in all_months]
    picked = st.selectbox(
        "Month",
        month_options,
        index=0,
        format_func=lambda x: x if x == "Latest" else x.strftime("%b %Y")
    )

# Resolve month selection
current_month = latest_month if picked == "Latest" else picked

# Previous month
prev_month = (current_month - pd.DateOffset(months=1)) if isinstance(current_month, pd.Timestamp) else None

# Filter by salesperson for all-month views (charts/tables)
sales_filtered_all, bookings_filtered_all = filter_by_salesperson_month(
    sales_df, bookings_df, selected_sp, None
)

# Helpers
def safe_month_slice(df: pd.DataFrame, month: pd.Timestamp, value_col: str) -> float:
    if df is None or df.empty or month is None or not isinstance(month, pd.Timestamp):
        return 0.0
    return float(df.loc[df["Month"] == month, value_col].sum())

def pct_change(curr: float, prev: float) -> str:
    if prev and np.isfinite(prev) and prev != 0:
        return f"{((curr - prev) / prev * 100):+.1f}%"
    return "—"

def fmt_kes(x: float) -> str:
    return f"KSH {x:,.0f}"

# KPIs
current_sales = safe_month_slice(sales_filtered_all, current_month, "Sales")
current_bookings = safe_month_slice(bookings_filtered_all, current_month, "Bookings")
prev_sales = safe_month_slice(sales_filtered_all, prev_month, "Sales") if prev_month else 0.0
prev_bookings = safe_month_slice(bookings_filtered_all, prev_month, "Bookings") if prev_month else 0.0
conversion_rate = (current_sales / current_bookings * 100.0) if current_bookings > 0 else 0.0

pipeline_value = float(
    bookings_filtered_all.loc[bookings_filtered_all["Month"] >= current_month, "Bookings"].sum()
) if isinstance(current_month, pd.Timestamp) else float(
    bookings_filtered_all["Bookings"].sum() if not bookings_filtered_all.empty else 0.0
)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Current Month Sales", fmt_kes(current_sales), pct_change(current_sales, prev_sales))
with c2: st.metric("Current Month Bookings", fmt_kes(current_bookings), pct_change(current_bookings, prev_bookings))
with c3: st.metric("Conversion Rate", f"{conversion_rate:.1f}%", "Target: 80%")
with c4: st.metric("Pipeline Value", fmt_kes(pipeline_value), "Open bookings")

st.divider()

# ===================== Trend Data (last 12 months) =====================
def build_monthly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Month", value_col])
    return df.groupby("Month", as_index=False)[value_col].sum()

sales_monthly = build_monthly(sales_filtered_all, "Sales")
bookings_monthly = build_monthly(bookings_filtered_all, "Bookings")

trend = pd.merge(sales_monthly, bookings_monthly, on="Month", how="outer").fillna(0.0).sort_values("Month")
if isinstance(latest_month, pd.Timestamp) and not trend.empty:
    cutoff = latest_month - pd.DateOffset(months=11)
    trend = trend[trend["Month"].between(cutoff, latest_month, inclusive="both")]
trend["Conversion %"] = np.where(trend["Bookings"] > 0, (trend["Sales"] / trend["Bookings"]) * 100.0, 0.0)

# --------------------- Mini Trend Renderer (inline, outside expanders) ---------------------
def render_client_mini_trend(client_name: str, anchor_month: pd.Timestamp, salesperson: str):
    """Renders 5 client KPIs (single + 2 + 2), 3-year line trend, and MoM change bar."""
    dd_client = client_name
    dd_sp = salesperson

    # ----- Local month_value fallback -----
    def month_value_local(dd_sp_: str, dd_client_: str, month_: pd.Timestamp) -> float:
        if month_ is None or sales_df is None or sales_df.empty:
            return 0.0
        df = sales_df
        mask = (df["CustomerName"] == dd_client_) & (df["Month"] == month_)
        if dd_sp_ != "All":
            mask &= (df["Salesperson"] == dd_sp_)
        return float(df.loc[mask, "Sales"].sum())

    # ----- Client KPIs for selected month -----
    v_now = month_value_local(dd_sp, dd_client, anchor_month)

    # Previous month for MoM
    prev_anchor_month = anchor_month - pd.DateOffset(months=1)
    v_prev = month_value_local(dd_sp, dd_client, prev_anchor_month)

    this_year = int(anchor_month.year)
    base_mask = (sales_df["CustomerName"] == dd_client)
    if dd_sp != "All":
        base_mask &= (sales_df["Salesperson"] == dd_sp)

    this_year_slice = sales_df.loc[
        base_mask
        & (sales_df["Month"].dt.year == this_year)
        & (sales_df["Month"] <= anchor_month),
        ["Month", "Sales"]
    ].copy()
    if this_year_slice.empty:
        avg_this_year = 0.0
        months_this_year = 0
    else:
        months_this_year = this_year_slice["Month"].dt.month.nunique()
        avg_this_year = float(this_year_slice["Sales"].sum() / max(months_this_year, 1))

    last_year = this_year - 1
    last_year_slice = sales_df.loc[
        base_mask & (sales_df["Month"].dt.year == last_year),
        ["Month", "Sales"]
    ].copy()
    if last_year_slice.empty:
        avg_last_year = 0.0
        months_last_year = 0
    else:
        months_last_year = last_year_slice["Month"].dt.month.nunique()
        avg_last_year = float(last_year_slice["Sales"].sum() / max(months_last_year, 1))

    v_last_year_same_month = month_value_local(dd_sp, dd_client, anchor_month - pd.DateOffset(years=1))
    anchor_label = anchor_month.strftime('%b')

    # ---- KPIs layout: 1 on top, then 2, then 2 ----
    # Row 1: Selected month sales (single)
    (r1c1,) = st.columns(1)
    with r1c1:
        st.metric(f"{anchor_label} {this_year} Sales", fmt_kes(v_now))

    # Row 2: Vs previous month + Vs this-year avg
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        mom_abs = v_now - v_prev
        mom_pct = f"{(mom_abs / v_prev * 100):+.2f}%" if v_prev else "—"
        st.metric(
            f"Vs previous month ({prev_anchor_month.strftime('%b %Y')})",
            fmt_kes(mom_abs),
            mom_pct
        )
    with r2c2:
        st.metric(
            f"Vs this-year monthly avg ({months_this_year} mo)",
            fmt_kes(v_now - avg_this_year),
            f"{((v_now - avg_this_year)/avg_this_year*100):+.2f}%" if avg_this_year else "—"
        )

    # Row 3: Vs last-year avg + Same-month YoY
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.metric(
            f"Vs last-year avg ({months_last_year} mo)",
            fmt_kes(v_now - avg_last_year),
            f"{((v_now - avg_last_year)/avg_last_year*100):+.2f}%" if avg_last_year else "—"
        )
    with r3c2:
        st.metric(
            "Same month YoY",
            fmt_kes(v_now - v_last_year_same_month),
            f"{((v_now - v_last_year_same_month)/v_last_year_same_month*100):+.2f}%" if v_last_year_same_month else "—"
        )

    # ---- Line trend (3 yrs) ----
    st.caption(f"Filters → Month: {anchor_month.strftime('%b %Y')} • Salesperson: {salesperson}")
    growth_pct = st.number_input(
        "Growth % to apply on last year's average",
        value=0.0, step=1.0, format="%.2f", key="growth_pct_avgline"
    )

    monthly_sc = sales_df.copy()  # expects: Salesperson, CustomerName, Month, Sales
    mask_line = (monthly_sc["CustomerName"] == dd_client)
    if dd_sp != "All":
        mask_line &= (monthly_sc["Salesperson"] == dd_sp)
    trend_all = monthly_sc.loc[mask_line, ["Month", "Sales"]].copy()

    if trend_all.empty:
        st.info("No sales history for this client under the current filter.")
        return

    trend_all["Year"] = trend_all["Month"].dt.year
    trend_all["MonthNum"] = trend_all["Month"].dt.month
    trend_all["MonthLabel"] = trend_all["Month"].dt.strftime("%b")

    years_sorted = sorted(trend_all["Year"].unique())
    years_keep = years_sorted[-3:] if len(years_sorted) > 3 else years_sorted
    trend_all = trend_all[trend_all["Year"].isin(years_keep)]

    trend_all = (
        trend_all.groupby(["Year", "MonthNum", "MonthLabel"], as_index=False)["Sales"]
        .sum()
        .sort_values(["Year", "MonthNum"])
    )

    fig_line_years = px.line(
        trend_all,
        x="MonthNum",
        y="Sales",
        color="Year",
        markers=True,
        line_group="Year",
        hover_data={"Sales": ":,.2f", "MonthLabel": True},
        title=f"Monthly Sales by Year — {dd_client}{'' if dd_sp=='All' else f' (handled by {dd_sp})'}"
    )
    fig_line_years.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        ),
        yaxis_title="Sales Amount",
        xaxis_title="Month",
        legend_title_text="Year",
        height=500
    )

    current_year_local = int(anchor_month.year)
    last_year_local = current_year_local - 1

    def year_avg(y: int) -> float:
        dfy = trend_all[trend_all["Year"] == y].copy()
        if y == current_year_local:
            dfy = dfy[dfy["MonthNum"] <= anchor_month.month]
        months = dfy["MonthNum"].nunique()
        return float(dfy["Sales"].sum() / months) if months else 0.0

    avg_curr = year_avg(current_year_local)
    avg_last = year_avg(last_year_local)

    x_vals = list(range(1, 13))
    if avg_curr > 0:
        fig_line_years.add_scatter(
            x=x_vals, y=[avg_curr]*12, mode="lines",
            name=f"{current_year_local} avg (to-date)", line=dict(dash="dash"),
            hovertemplate="%{y:,.2f}<extra></extra>"
        )

    avg_last_proj = avg_last * (1.0 + growth_pct / 100.0)
    if avg_last_proj > 0:
        fig_line_years.add_scatter(
            x=x_vals, y=[avg_last_proj]*12, mode="lines",
            name=f"{last_year_local} avg × (1+{growth_pct:.2f}%)", line=dict(dash="dot"),
            hovertemplate="%{y:,.2f}<extra></extra>"
        )

    st.plotly_chart(fig_line_years, use_container_width=True)

    

# ===================== Losses & Gaps (each in its own expander) =====================
st.caption(f"Month: {current_month.strftime('%b %Y')}  •  Salesperson: {selected_sp}")

def monthly_by_customer(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Month", "CustomerName", value_col])
    return df.groupby(["Month", "CustomerName"], as_index=False)[value_col].sum()

sales_cust = monthly_by_customer(sales_filtered_all, "Sales")
book_cust  = monthly_by_customer(bookings_filtered_all, "Bookings")

# ---------- Top 25 Losses (MoM drops) ----------
prev_mon = prev_month if isinstance(prev_month, pd.Timestamp) else None

cur_sales = sales_cust[sales_cust["Month"] == current_month][["CustomerName", "Sales"]].copy()
cur_sales.rename(columns={"Sales": "Sales_Current"}, inplace=True)

if prev_mon is not None:
    prev_sales = sales_cust[sales_cust["Month"] == prev_mon][["CustomerName", "Sales"]].copy()
else:
    prev_sales = pd.DataFrame(columns=["CustomerName", "Sales"])
prev_sales.rename(columns={"Sales": "Sales_Previous"}, inplace=True)

dropped = pd.merge(cur_sales, prev_sales, on="CustomerName", how="left").fillna({"Sales_Previous": 0.0})
dropped["MoM Δ Sales"] = dropped["Sales_Current"] - dropped["Sales_Previous"]
dropped = dropped[dropped["MoM Δ Sales"] < 0].copy()
dropped = dropped.sort_values("MoM Δ Sales").head(25)

# ---------- Current Month: Booked vs Sold Gaps ----------
cur_book = book_cust[book_cust["Month"] == current_month][["CustomerName", "Bookings"]].copy()
cur_book.rename(columns={"Bookings": "Bookings_Current"}, inplace=True)
cur_sales_for_gap = cur_sales.rename(columns={"Sales_Current": "Sales_Current"}).copy()

gap = pd.merge(cur_book, cur_sales_for_gap, on="CustomerName", how="outer").fillna(0.0)
gap["Gap (Bookings - Sales)"] = gap["Bookings_Current"] - gap["Sales_Current"]
gap_pos = gap.sort_values("Gap (Bookings - Sales)", ascending=False).head(25).copy()

# ---------- Clickable list/table helper ----------
def render_clicklist_table(df: pd.DataFrame, title: str, cols_order: list[str], key_prefix: str):
    st.markdown(f"### {title}")

    if df.empty:
        st.info(f"No records for {title.lower()}.")
        return

    # Header row
    header_cols = st.columns([3, 2, 2, 2])
    header_cols[0].markdown("**CustomerName**")
    for i, colname in enumerate(cols_order[1:], start=1):
        header_cols[i].markdown(f"**{colname}**")

    # Rows (name is a button)
    for i, row in df.iterrows():
        c = st.columns([3, 2, 2, 2])
        if c[0].button(str(row["CustomerName"]), key=f"{key_prefix}_sel_{i}", help="Click to view mini trend", type="secondary"):
            st.session_state["client_for_trend"] = str(row["CustomerName"])
            st.rerun()
        for j, colname in enumerate(cols_order[1:], start=1):
            val = row.get(colname, 0.0)
            if isinstance(val, (int, float, np.floating)):
                c[j].write(f"KSH {val:,.0f}")
            else:
                c[j].write(val)

# ----- Expander: Losses -----
with st.expander("📉 Losses (MoM drops)"):
    dropped_show = dropped[["CustomerName", "Sales_Previous", "Sales_Current", "MoM Δ Sales"]].copy()
    render_clicklist_table(
        dropped_show,
        "🔻 Top 25 Losses",
        ["CustomerName", "Sales_Previous", "Sales_Current", "MoM Δ Sales"],
        key_prefix="losses"
    )
    if not dropped_show.empty:
        st.download_button(
            "⬇️ Download Losses (CSV)",
            data=dropped_show.to_csv(index=False),
            file_name="losses_top25.csv",
            mime="text/csv",
            key="dl_losses_csv"
        )

# ----- Expander: Gaps -----
with st.expander("📐 Booked vs Sold Gaps"):
    gap_show = gap_pos[["CustomerName", "Bookings_Current", "Sales_Current", "Gap (Bookings - Sales)"]].copy()
    render_clicklist_table(
        gap_show,
        "🧮 Largest Booked vs Sold Gaps",
        ["CustomerName", "Bookings_Current", "Sales_Current", "Gap (Bookings - Sales)"],
        key_prefix="gaps"
    )
    if not gap_show.empty:
        st.download_button(
            "⬇️ Download Booked vs Sold Gaps (CSV)",
            data=gap_show.to_csv(index=False),
            file_name="booked_vs_sold_gaps_top25.csv",
            mime="text/csv",
            key="dl_gaps_csv"
        )

# ===================== Fixed Chart Area (OUTSIDE the expanders) =====================
st.markdown("---")
client_for_trend = st.session_state.get("client_for_trend")
if client_for_trend:
    st.subheader(f"📈 Client Mini Trend — {client_for_trend}")
    render_client_mini_trend(client_for_trend, current_month, selected_sp)
    cols = st.columns([1, 6])
    if cols[0].button("Clear selection", key="clear_inline"):
        st.session_state.pop("client_for_trend", None)
        st.rerun()
else:
    st.info("Click any customer name in the Losses or Gaps tables above to view their KPIs and mini trend here.")

# ===================== Summary Charts (separate expander) =====================
with st.expander("Summary Charts"):
    left, right = st.columns(2)

    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["Month"], y=trend["Sales"], name="Sales", mode="lines+markers", line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=trend["Month"], y=trend["Bookings"], name="Bookings", mode="lines+markers", line=dict(width=2)
        ))
        fig.update_layout(
            title="Sales vs Bookings — Last 12 Months",
            xaxis_title="Month",
            yaxis_title="Amount (KSH)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=trend["Month"], y=trend["Conversion %"], name="Conversion %"))
        fig2.add_hline(y=80, line_dash="dash", annotation_text="Target 80%", annotation_position="top left")
        fig2.update_layout(
            title="Monthly Conversion Rate %",
            xaxis_title="Month",
            yaxis_title="Conversion %",
            showlegend=False,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    if picked != "Latest":
        st.caption(f"Showing KPIs for {current_month.strftime('%b %Y')} (trends always show the latest 12 months for context).")
