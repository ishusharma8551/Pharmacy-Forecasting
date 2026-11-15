"""Page 4 of 6: Inventory Optimization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from src import inventory_optimization as inv  # noqa: E402
from src import ui  # noqa: E402

ui.setup("Inventory Optimization", "📦")
d = ui.get_data()
recs, fefo = d["recommendations"], d["fefo"]

sv = inv.estimate_savings(recs)
c = st.columns(4)
c[0].metric("Potential expiry loss", ui.money(sv["potential_expiry_loss"]))
c[1].metric("Capital in overstock", ui.money(sv["overstock_value"]))
c[2].metric("Expired stock (realised loss)", ui.money(sv["expired_value"]))
c[3].metric(f"Est. recoverable @ {sv['assumed_recovery_rate']:.0%} (assumption)", ui.money(sv["estimated_recoverable"]),
            help="ASSUMPTION: share of at-risk value recovered via discounts, returns or redistribution. Adjust in config.py.")

tab1, tab2, tab3, tab4 = st.tabs(["Reorder recommendations", "Overstock alerts", "FEFO", "Formulas and assumptions"])

with tab1:
    r = recs[recs["Flag_Reorder"]].sort_values("Order_Cost", ascending=False)
    st.write(f"**{len(r)} medicine(s)** are at or below their reorder point.")
    if len(r):
        fig = px.bar(r.head(15), x="Medicine_Name", y="Recommended_Order_Qty", color="Category", title="Suggested order quantities")
        ui.chart(fig)
    ui.table(r[["Medicine_Name", "Category", "Usable_Stock", "Lead_Time_Demand", "Safety_Stock", "Reorder_Point",
                "Target_Stock", "Recommended_Order_Qty", "Order_Cost"]])

with tab2:
    o = recs[recs["Flag_Overstock"]].sort_values("Overstock_Value", ascending=False)
    st.write(f"**{len(o)} medicine(s)** hold more than their maximum stock level.")
    if len(o):
        ui.chart(px.bar(o.head(15), x="Medicine_Name", y="Overstock_Value", color="Category", title=f"Overstock value ({config.CURRENCY})"))
    ui.table(o[["Medicine_Name", "Usable_Stock", "Maximum_Stock_Level", "Overstock_Qty", "Overstock_Value", "Days_of_Cover",
                "Potential_Expiry_Loss"]])

with tab3:
    st.write("First-Expiry-First-Out: dispense the earliest-expiring batch first.")
    only = st.checkbox("Only show batches needing action (High risk, <= 30 days, or expired)", value=True)
    f = fefo
    if only:
        f = f[(f["Risk_Class"].isin(["High", "Expired"])) | (f["Days_Until_Expiry"] <= 30)]
    pick = st.selectbox("Medicine", ["All"] + sorted(f["Medicine_Name"].unique()))
    if pick != "All":
        f = f[f["Medicine_Name"] == pick]
    ui.table(f.sort_values(["Days_Until_Expiry"]))

with tab4:
    st.markdown(f"""
| Quantity | Formula |
|---|---|
| Daily forecast | 30-day forecast / 30 |
| Lead-time demand | daily forecast x supplier lead time |
| Safety stock | Z x sigma_daily x sqrt(lead time), Z = {config.SERVICE_LEVEL_Z} (about 95% service level) |
| Reorder point | lead-time demand + safety stock |
| Target stock | daily forecast x (lead time + {config.REVIEW_PERIOD_DAYS} review days) + safety stock, capped at maximum stock level |
| Order quantity | max(target stock - usable stock, 0), only when usable stock <= reorder point |
| Overstock | max(usable stock - maximum stock level, 0) |
| Potential expiry loss | expected unsold units x purchase price (live batches) |

**Assumptions:** demand is stationary over the lead time; lead time is fixed; no stock already on order; sigma_daily is the standard
deviation of the last 90 days of sales; near expiry means <= {config.NEAR_EXPIRY_DAYS} days.
""")
    ui.table(recs[["Medicine_Name", "Usable_Stock", "Forecast_Daily", "Supplier_Lead_Time", "Lead_Time_Demand", "Safety_Stock",
                   "Reorder_Point", "Target_Stock", "Recommended_Order_Qty", "Recommendation"]])
st.info(ui.EXPIRY_DISCLAIMER)
ui.footer()
