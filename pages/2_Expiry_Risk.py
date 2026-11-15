"""Page 3 of 6: Expiry Risk Analysis."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui  # noqa: E402

ui.setup("Expiry Risk Analysis", "⏳")
d = ui.get_data()
st.info(ui.EXPIRY_DISCLAIMER)
sb = d["expiry_predictions"].copy()

f1, f2 = st.columns(2)
cats = f1.multiselect("Category", sorted(sb["Category"].unique()))
risks = f2.multiselect("Risk class", ["High", "Medium", "Low", "Expired"])
if cats:
    sb = sb[sb["Category"].isin(cats)]
if risks:
    sb = sb[sb["Risk_Class"].isin(risks)]

live = sb[sb["Days_Until_Expiry"] > 0]
m = st.columns(4)
m[0].metric("Batches", len(sb))
m[1].metric("High-risk batches", int((sb["Risk_Class"] == "High").sum()))
m[2].metric("Expected unsold units (live)", f"{live['Expected_Unsold_Stock'].sum():,.0f}")
m[3].metric("Expected unsold value", ui.money(live["Expected_Unsold_Value"].sum()))

a, b = st.columns(2)
dist = sb["Risk_Class"].value_counts().reset_index()
dist.columns = ["Risk_Class", "Batches"]
with a:
    fig = px.pie(dist, names="Risk_Class", values="Batches", color="Risk_Class", color_discrete_map=ui.RISK_COLORS,
                 hole=0.45, title="Expiry risk distribution (batches)")
    ui.chart(fig)
with b:
    fig = px.scatter(live, x="Days_Until_Expiry", y="Expected_Unsold_Stock", color="Risk_Class", size="Current_Stock",
                     color_discrete_map=ui.RISK_COLORS, hover_data=["Medicine_Name", "Batch_Number"],
                     title="Days to expiry vs expected unsold units")
    ui.chart(fig)

st.subheader("Medicine-wise expiry risk")
order = {"Expired": 0, "High": 1, "Medium": 2, "Low": 3}
worst = (sb.assign(_o=sb["Risk_Class"].map(order)).sort_values(["Medicine_ID", "_o"]).groupby("Medicine_ID").first().reset_index())
agg = sb.groupby("Medicine_ID").agg(Batches=("Batch_Number", "count"), Units=("Current_Stock", "sum"),
                                    Expected_Unsold=("Expected_Unsold_Stock", "sum"),
                                    Expected_Unsold_Value=("Expected_Unsold_Value", "sum")).reset_index()
mw = worst[["Medicine_ID", "Medicine_Name", "Category", "Risk_Class"]].rename(columns={"Risk_Class": "Worst_Batch_Risk"}).merge(agg, on="Medicine_ID")
ui.table(mw.sort_values("Expected_Unsold_Value", ascending=False))

st.subheader("High-risk and expired inventory")
hr = sb[sb["Risk_Class"].isin(["High", "Expired"])].sort_values("Expected_Unsold_Value", ascending=False)
ui.table(hr[["Medicine_Name", "Batch_Number", "Expiry_Date", "Days_Until_Expiry", "Current_Stock", "Expected_Units_Sold",
             "Expected_Unsold_Stock", "Expected_Unsold_Value", "Risk_Class", "Risk_Prob_High"]])

st.subheader("Batch-wise expiry information")
pick = st.selectbox("Medicine", ["All"] + sorted(sb["Medicine_Name"].unique()))
bt = sb if pick == "All" else sb[sb["Medicine_Name"] == pick]
ui.table(bt[["Medicine_Name", "Batch_Number", "Manufacturing_Date", "Expiry_Date", "Days_Until_Expiry", "Status",
             "Current_Stock", "Expected_Units_Sold", "Expected_Unsold_Stock", "Risk_Class", "FEFO_Rank"]].sort_values(["Medicine_Name", "FEFO_Rank"]))
st.caption("Expected Unsold Stock = Current Stock - Expected Units Sold before expiry. Demand is shared between batches "
           "first-expiry-first-out, so later batches only get demand left over by earlier ones.")
ui.footer()
