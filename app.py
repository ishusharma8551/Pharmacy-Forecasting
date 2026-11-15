"""Page 1 of 6: Executive Dashboard.  Run with:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from src import ui  # noqa: E402

ui.setup("Executive Dashboard")
d = ui.get_data()
recs, sales, sb = d["recommendations"], d["sales_daily"], d["expiry_predictions"]

potential = recs["Potential_Expiry_Loss"].sum()
expired = recs["Expired_Loss"].sum()
c = st.columns(6)
c[0].metric("Total medicines", f"{len(recs)}")
c[1].metric("Inventory value (usable, at cost)", ui.money(recs["Inventory_Value"].sum()))
c[2].metric("Medicines with high expiry risk", int(recs["Flag_HighRisk"].sum()))
c[3].metric("Low-stock (reorder now)", int(recs["Flag_Reorder"].sum()))
c[4].metric("Potential expiry loss", ui.money(potential), help="Expected unsold units x purchase price, live batches")
c[5].metric("Expired stock on shelf", ui.money(expired), help="Already expired - realised loss")

st.subheader("Demand overview")
left, right = st.columns([2, 1])
daily = sales.groupby("Date")["Units_Sold"].sum().reset_index()
daily["7-day average"] = daily["Units_Sold"].rolling(7).mean()
recent = daily.tail(180)
fig = go.Figure()
fig.add_bar(x=recent["Date"], y=recent["Units_Sold"], name="Daily units", marker_color="#a8dadc")
fig.add_scatter(x=recent["Date"], y=recent["7-day average"], name="7-day average", line=dict(color="#264653", width=3))
fig.update_layout(title="Total daily demand - last 180 days", height=380, margin=dict(t=50, b=10),
                  legend=dict(orientation="h", y=-0.15))
with left:
    ui.chart(fig)
with right:
    rows = []
    for h in (7, 15, 30):
        rows.append({"Window": f"{h} days", "Last actual": sales[sales["Date"] > d["as_of"] - pd.Timedelta(days=h)]["Units_Sold"].sum(),
                     "Next forecast": d["forecasts"][f"Forecast_{h}d"].sum()})
    cmp = pd.DataFrame(rows).melt("Window", var_name="Type", value_name="Units")
    fig2 = px.bar(cmp, x="Window", y="Units", color="Type", barmode="group", title="Last actual vs next forecast (all medicines)",
                  color_discrete_map={"Last actual": "#a8dadc", "Next forecast": "#e76f51"})
    fig2.update_layout(height=380, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.15))
    ui.chart(fig2)

a, b = st.columns(2)
with a:
    cnt = recs["Recommendation"].value_counts().reset_index()
    cnt.columns = ["Recommendation", "Medicines"]
    fig3 = px.bar(cnt, x="Recommendation", y="Medicines", color="Recommendation", title="Recommended actions")
    fig3.update_layout(showlegend=False, height=340)
    ui.chart(fig3)
with b:
    by_cat = recs.groupby("Category")["Inventory_Value"].sum().sort_values().reset_index()
    fig4 = px.bar(by_cat, y="Category", x="Inventory_Value", orientation="h", title=f"Inventory value by category ({config.CURRENCY})")
    fig4.update_layout(height=340)
    ui.chart(fig4)

st.subheader("Needs attention first")
urgent = recs[recs["Recommendation"] != "Stock Sufficient"].copy()
order = {"Reorder Now": 0, "High Expiry Risk": 1, "Overstock Warning": 2, "Near Expiry": 3}
urgent["_o"] = urgent["Recommendation"].map(order)
urgent = urgent.sort_values(["_o", "Potential_Expiry_Loss"], ascending=[True, False])
ui.table(urgent[["Medicine_Name", "Category", "Recommendation", "All_Flags", "Usable_Stock", "Reorder_Point",
                 "Recommended_Order_Qty", "Potential_Expiry_Loss"]].head(15))
st.info(ui.EXPIRY_DISCLAIMER)
ui.footer()
