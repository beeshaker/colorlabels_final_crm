# helpers/metrics.py
import pandas as pd
import numpy as np

def calculate_conversion_metrics(sales_df, bookings_df, customer, salesperson=None):
    if salesperson and salesperson != "All":
        sales_cust = sales_df[(sales_df["CustomerName"] == customer) & (sales_df["Salesperson"] == salesperson)]
        bookings_cust = bookings_df[(bookings_df["CustomerName"] == customer) & (bookings_df["Salesperson"] == salesperson)]
    else:
        sales_cust = sales_df[sales_df["CustomerName"] == customer]
        bookings_cust = bookings_df[bookings_df["CustomerName"] == customer]

    merged = pd.merge(
        bookings_cust.groupby("Month")["Bookings"].sum().reset_index(),
        sales_cust.groupby("Month")["Sales"].sum().reset_index(),
        on="Month", how="outer"
    ).fillna(0)

    metrics = {
        "avg_conversion_rate": 0.0,
        "conversion_trend": "stable",
        "avg_lag_days": 0,
        "booking_fulfillment": 0.0,
    }
    if not merged.empty and merged["Bookings"].sum() > 0:
        metrics["avg_conversion_rate"] = (merged["Sales"].sum() / merged["Bookings"].sum()) * 100
        recent = merged.tail(3)
        if len(recent) >= 2:
            recent_rate = recent["Sales"].sum() / max(recent["Bookings"].sum(), 1)
            older = merged.head(len(merged) - 3)
            older_rate = older["Sales"].sum() / max(older["Bookings"].sum(), 1) if len(older) else recent_rate
            if recent_rate > older_rate * 1.1:
                metrics["conversion_trend"] = "improving"
            elif recent_rate < older_rate * 0.9:
                metrics["conversion_trend"] = "declining"
        metrics["booking_fulfillment"] = min(metrics["avg_conversion_rate"], 100)
    return metrics


def calculate_target_score(row, weights=None):
    if weights is None:
        weights = {
            "booking_pipeline": 0.25,
            "conversion_rate": 0.15,
            "gap_vs_ly": 0.20,
            "gap_vs_ty": 0.15,
            "potential": 0.15,
            "mom_decline": 0.10,
        }
    def normalize(value, min_val=0, max_val=100):
        if max_val == min_val:
            return 0
        return min(max((value - min_val) / (max_val - min_val), 0), 1)

    score = 0.0
    if "booking_pipeline" in row:
        score += weights["booking_pipeline"] * normalize(row["booking_pipeline"], 0, row.get("potential", 1) or 1)
    if "conversion_rate" in row:
        score += weights["conversion_rate"] * normalize(row["conversion_rate"], 0, 100)
    if "Gap_vs_LY" in row:
        score += weights["gap_vs_ly"] * normalize(row["Gap_vs_LY"], 0, row.get("potential", 100000))
    if "Gap_vs_TY" in row:
        score += weights["gap_vs_ty"] * normalize(row["Gap_vs_TY"], 0, row.get("potential", 100000))
    if "potential" in row:
        score += weights["potential"] * normalize(row["potential"], 0, 1_000_000)
    if "MoM_Δ" in row and row["MoM_Δ"] < 0:
        score += weights["mom_decline"] * normalize(abs(row["MoM_Δ"]), 0, row.get("potential", 100000))
    return float(score)


def calculate_recommended_target(customer_data, conversion_metrics):
    base = customer_data.get("potential", 0) or 0
    booking_adj = customer_data.get("booking_pipeline", 0) * (conversion_metrics.get("avg_conversion_rate", 0) / 100.0)
    trend = conversion_metrics.get("conversion_trend", "stable")
    mult = 1.1 if trend == "improving" else (0.9 if trend == "declining" else 1.0)
    recommended = max(base * mult, booking_adj)
    recommended = min(recommended, base * 1.5)
    recommended = max(recommended, base * 0.5)
    return float(recommended)


def generate_action_plan(customer_data, conversion_metrics):
    actions = []
    priority = "Medium"

    if conversion_metrics["avg_conversion_rate"] < 50:
        actions.append("🔴 Low conversion rate — investigate fulfilment issues")
        priority = "High"
    elif conversion_metrics["avg_conversion_rate"] < 75:
        actions.append("🟡 Moderate conversion — optimize order processing")

    trend = conversion_metrics["conversion_trend"]
    if trend == "declining":
        actions.append("📉 Conversion declining — urgent review with customer")
        priority = "High"
    elif trend == "improving":
        actions.append("📈 Conversion improving — keep momentum")

    if customer_data.get("Gap_vs_LY", 0) > (customer_data.get("potential", 0) or 0) * 0.3:
        actions.append("💰 Big gap vs LY — recovery opportunity")
        priority = "High"

    if customer_data.get("booking_pipeline", 0) > (customer_data.get("current_sales", 0) or 0) * 1.5:
        actions.append("📊 Strong pipeline — ensure fulfilment capacity")
    elif customer_data.get("booking_pipeline", 0) < (customer_data.get("current_sales", 0) or 0) * 0.5:
        actions.append("⚠️ Weak pipeline — needs sales push")
        priority = "High"

    if customer_data.get("Recency_m", 0) > 2:
        actions.append("🕐 No recent activity — re-engage")

    return {
        "priority": priority,
        "actions": actions,
        "recommended_target": calculate_recommended_target(customer_data, conversion_metrics),
    }
