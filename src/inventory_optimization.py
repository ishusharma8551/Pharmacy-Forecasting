"""Module 3: inventory optimization (reorder logic, overstock, FEFO).

Formulas and ASSUMPTIONS (all documented, all in config.py):
  daily_forecast      = Forecast_30d / 30                      (ML demand forecast)
  Lead-Time Demand    = daily_forecast x supplier_lead_time
  Safety Stock        = Z x sigma_daily x sqrt(lead_time)      Z = 1.65 (~95% service level)
                        sigma_daily = std-dev of daily sales over the last 90 days
  Reorder Point (ROP) = Lead-Time Demand + Safety Stock
  Target Stock        = daily_forecast x (lead_time + review_period) + Safety Stock
                        capped at Maximum_Stock_Level (shelf/capital limit), never below ROP
  Order Quantity      = max(Target Stock - Usable Stock, 0)    only when Usable Stock <= ROP
  Usable Stock        = stock in batches that have NOT expired
  Overstock Quantity  = max(Usable Stock - Maximum_Stock_Level, 0)
  Potential Expiry Loss = sum(expected unsold units x purchase price) over non-expired batches
Assumptions: demand is stationary over the lead time, no on-order stock, single supplier,
lead time is deterministic, sigma_daily is estimated from history (not from forecast error).
"""
import numpy as np
import pandas as pd

import config

ACTION_PRIORITY = ["Reorder Now", "High Expiry Risk", "Overstock Warning", "Near Expiry", "Stock Sufficient"]


def lead_time_demand(daily_forecast, lead_time):
    return np.asarray(daily_forecast, float) * np.asarray(lead_time, float)


def safety_stock(sigma_daily, lead_time, z=None):
    z = config.SERVICE_LEVEL_Z if z is None else z
    return z * np.asarray(sigma_daily, float) * np.sqrt(np.asarray(lead_time, float))


def reorder_point(ltd, ss):
    return np.asarray(ltd, float) + np.asarray(ss, float)


def target_stock(daily_forecast, lead_time, ss, max_level=None, review_days=None):
    review_days = config.REVIEW_PERIOD_DAYS if review_days is None else review_days
    target = np.asarray(daily_forecast, float) * (np.asarray(lead_time, float) + review_days) + np.asarray(ss, float)
    if max_level is not None:
        rop = lead_time_demand(daily_forecast, lead_time) + np.asarray(ss, float)
        target = np.minimum(target, np.maximum(np.asarray(max_level, float), rop))
    return target


def order_quantity(target, usable_stock):
    """Negative requirements become zero; whole units, rounded up."""
    return np.ceil(np.maximum(np.asarray(target, float) - np.asarray(usable_stock, float), 0.0))


def primary_action(flags):
    for a in ACTION_PRIORITY:
        if a in flags:
            return a
    return "Stock Sufficient"


def medicine_recommendations(medicines, scored_batches, stats):
    """One row per medicine with all inventory metrics and the recommended action."""
    sb = scored_batches
    live = sb[sb["Days_Until_Expiry"] > 0]
    agg = pd.DataFrame({
        "Usable_Stock": live.groupby("Medicine_ID")["Current_Stock"].sum(),
        "Expired_Stock": sb[sb["Days_Until_Expiry"] <= 0].groupby("Medicine_ID")["Current_Stock"].sum(),
        "Near_Expiry_Units": live[live["Days_Until_Expiry"] <= config.NEAR_EXPIRY_DAYS].groupby("Medicine_ID")["Current_Stock"].sum(),
        "Potential_Expiry_Loss": live.groupby("Medicine_ID")["Expected_Unsold_Value"].sum(),
        "Expected_Unsold_Units": live.groupby("Medicine_ID")["Expected_Unsold_Stock"].sum(),
        "High_Risk_Batches": (live["Risk_Class"] == "High").groupby(live["Medicine_ID"]).sum(),
    }).reset_index()
    r = medicines.merge(agg, on="Medicine_ID", how="left").merge(stats, on="Medicine_ID", how="left")
    for c in ["Usable_Stock", "Expired_Stock", "Near_Expiry_Units", "Potential_Expiry_Loss",
              "Expected_Unsold_Units", "High_Risk_Batches"]:
        r[c] = r[c].fillna(0)
    r["Forecast_Daily"] = (r["Forecast_30d"].fillna(r["Avg_Daily_28"] * 30) / 30.0).round(2)
    lt = r["Supplier_Lead_Time"]
    r["Lead_Time_Demand"] = lead_time_demand(r["Forecast_Daily"], lt).round(1)
    r["Safety_Stock"] = safety_stock(r["Sigma_Daily"], lt).round(1)
    r["Reorder_Point"] = reorder_point(r["Lead_Time_Demand"], r["Safety_Stock"]).round(1)
    r["Target_Stock"] = target_stock(r["Forecast_Daily"], lt, r["Safety_Stock"], r["Maximum_Stock_Level"]).round(1)
    r["Flag_Reorder"] = r["Usable_Stock"] <= r["Reorder_Point"]
    r["Recommended_Order_Qty"] = np.where(r["Flag_Reorder"], order_quantity(r["Target_Stock"], r["Usable_Stock"]), 0).astype(int)
    r["Overstock_Qty"] = np.maximum(r["Usable_Stock"] - r["Maximum_Stock_Level"], 0).astype(int)
    r["Flag_Overstock"] = r["Overstock_Qty"] > 0
    r["Flag_NearExpiry"] = r["Near_Expiry_Units"] > 0
    r["Flag_HighRisk"] = r["High_Risk_Batches"] > 0
    r["Expired_Loss"] = (r["Expired_Stock"] * r["Purchase_Price"]).round(2)
    r["Overstock_Value"] = (r["Overstock_Qty"] * r["Purchase_Price"]).round(2)
    r["Inventory_Value"] = (r["Usable_Stock"] * r["Purchase_Price"]).round(2)
    r["Days_of_Cover"] = np.where(r["Forecast_Daily"] > 0, r["Usable_Stock"] / r["Forecast_Daily"], np.nan).round(1)
    r["Order_Cost"] = (r["Recommended_Order_Qty"] * r["Purchase_Price"]).round(2)
    r["Potential_Expiry_Loss"] = r["Potential_Expiry_Loss"].round(2)

    def flags(row):
        f = []
        if row["Flag_Reorder"]: f.append("Reorder Now")
        if row["Flag_HighRisk"]: f.append("High Expiry Risk")
        if row["Flag_Overstock"]: f.append("Overstock Warning")
        if row["Flag_NearExpiry"]: f.append("Near Expiry")
        return f

    fl = r.apply(flags, axis=1)
    r["All_Flags"] = fl.apply(lambda x: "; ".join(x) if x else "Stock Sufficient")
    r["Recommendation"] = fl.apply(primary_action)
    cols = ["Medicine_ID", "Medicine_Name", "Category", "Purchase_Price", "Selling_Price", "Supplier_Lead_Time",
            "Minimum_Stock_Level", "Maximum_Stock_Level", "Usable_Stock", "Expired_Stock", "Inventory_Value",
            "Forecast_7d", "Forecast_15d", "Forecast_30d", "Forecast_Daily", "Days_of_Cover",
            "Lead_Time_Demand", "Safety_Stock", "Reorder_Point", "Target_Stock", "Recommended_Order_Qty",
            "Order_Cost", "Overstock_Qty", "Overstock_Value", "Near_Expiry_Units", "High_Risk_Batches",
            "Expected_Unsold_Units", "Potential_Expiry_Loss", "Expired_Loss", "Flag_Reorder", "Flag_Overstock",
            "Flag_NearExpiry", "Flag_HighRisk", "All_Flags", "Recommendation"]
    return r[cols]


def fefo_table(scored_batches):
    """First-Expiry-First-Out dispensing order and suggested handling for every batch."""
    t = scored_batches.sort_values(["Medicine_ID", "Expiry_Date"]).copy()

    def action(row):
        if row["Days_Until_Expiry"] <= 0:
            return "Quarantine and remove - expired (follow SOP / return policy)"
        if row["Risk_Class"] == "High" or row["Days_Until_Expiry"] <= 30:
            return "Dispense first; consider discount or supplier return"
        if row["FEFO_Rank"] == 1:
            return "Dispense first (earliest expiry)"
        return f"Dispense after {row['FEFO_Rank'] - 1} earlier batch(es)"

    t["FEFO_Action"] = t.apply(action, axis=1)
    return t[["Medicine_ID", "Medicine_Name", "Batch_Number", "FEFO_Rank", "Expiry_Date", "Days_Until_Expiry",
              "Current_Stock", "Risk_Class", "Expected_Unsold_Stock", "FEFO_Action"]]


def estimate_savings(recs):
    """Summary of avoidable waste. RECOVERY RATE IS AN ASSUMPTION (config), not a measured value."""
    potential = float(recs["Potential_Expiry_Loss"].sum())
    overstock = float(recs["Overstock_Value"].sum())
    expired = float(recs["Expired_Loss"].sum())
    return {"potential_expiry_loss": potential, "overstock_value": overstock, "expired_value": expired,
            "assumed_recovery_rate": config.ASSUMED_RECOVERY_RATE,
            "estimated_recoverable": potential * config.ASSUMED_RECOVERY_RATE}
