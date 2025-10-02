# helpers/data.py
import os
import pandas as pd
import streamlit as st
from conn import Conn

# central cache for data load (shared by all pages)
@st.cache_data(ttl=300, show_spinner=False)
def load_all_data():
    """
    Returns:
        sales_df, bookings_df
        Both have columns: Salesperson, CustomerName, Month (ts), and either Sales or Bookings
    """
    c = Conn()

    # sales
    sales_data = c.load_sales_data()
    sales_df = sales_data["monthly_sc"].copy()
    sales_df["Month"] = pd.to_datetime(sales_df["Month"]).dt.to_period("M").dt.to_timestamp()

    # bookings
    bookings_data = c.load_bookings_data()
    bookings_df = bookings_data["monthly_sc"].copy()
    bookings_df["Month"] = pd.to_datetime(bookings_df["Month"]).dt.to_period("M").dt.to_timestamp()

    return sales_df, bookings_df


def get_latest_month(sales_df: pd.DataFrame, bookings_df: pd.DataFrame) -> pd.Timestamp:
    if (sales_df is None or sales_df.empty) and (bookings_df is None or bookings_df.empty):
        return pd.Timestamp.min
    smax = sales_df["Month"].max() if not sales_df.empty else pd.Timestamp.min
    bmax = bookings_df["Month"].max() if not bookings_df.empty else pd.Timestamp.min
    return max(smax, bmax)


def get_months(sales_df: pd.DataFrame, bookings_df: pd.DataFrame) -> list[pd.Timestamp]:
    """Unique sorted months from both sources (as Timestamp month-start)."""
    months = pd.Series(dtype="datetime64[ns]")
    if sales_df is not None and not sales_df.empty:
        months = pd.concat([months, sales_df["Month"]])
    if bookings_df is not None and not bookings_df.empty:
        months = pd.concat([months, bookings_df["Month"]])
    if months.empty:
        return []
    months = months.dropna().drop_duplicates().sort_values()
    return list(months)


def filter_by_salesperson_month(
    sales_df: pd.DataFrame,
    bookings_df: pd.DataFrame,
    salesperson: str | None,
    month: pd.Timestamp | None
):
    """Returns filtered (sales, bookings) by salesperson (or 'All') and exact month (or None to skip)."""
    s = sales_df.copy()
    b = bookings_df.copy()

    if salesperson and salesperson != "All":
        s = s[s["Salesperson"] == salesperson]
        b = b[b["Salesperson"] == salesperson]

    if month is not None:
        s = s[s["Month"] == month]
        b = b[b["Month"] == month]

    return s, b
