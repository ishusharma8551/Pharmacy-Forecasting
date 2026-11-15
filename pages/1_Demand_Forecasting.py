"""Page 2 of 6: Demand Forecasting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui  # noqa: E402

ui.setup("Demand Forecasting", "📈")
d = ui.get_data()
meds = d["medicines"].sort_values("Medicine_Name")
name = st.selectbox("Medicine", meds["Medicine_Name"])
mid = meds.loc[meds["Medicine_Name"] == name, "Medicine_ID"].iloc[0]
info = meds[meds["Medicine_ID"] == mid].iloc[0]
st.caption(f"{info['Category']} | seasonal pattern: {info['Seasonal_Demand']} | supplier lead time: {info['Supplier_Lead_Time']} days")

hist = d["sales_daily"][d["sales_daily"]["Medicine_ID"] == mid].sort_values("Date")
fc = d["forecasts"][d["forecasts"]["Medicine_ID"] == mid].iloc[0]
as_of = d["as_of"]

c = st.columns(3)
for col, h in zip(c, (7, 15, 30)):
    last = hist[hist["Date"] > as_of - pd.Timedelta(days=h)]["Units_Sold"].sum()
    col.metric(f"Next {h} days (forecast)", f"{fc[f'Forecast_{h}d']:,.0f} units",
               delta=f"{fc[f'Forecast_{h}d'] - last:+,.0f} vs last {h} days", delta_color="off")

st.subheader("Historical sales and forecast")
days = st.slider("History to show (days)", 60, 730, 180, step=30)
h = hist.tail(days)
f7, f15, f30 = fc["Forecast_7d"], fc["Forecast_15d"], fc["Forecast_30d"]
daily_fc = np.clip(np.r_[np.repeat(f7 / 7, 7), np.repeat((f15 - f7) / 8, 8), np.repeat((f30 - f15) / 15, 15)], 0, None)
fdates = pd.date_range(as_of + pd.Timedelta(days=1), periods=30)
fig = go.Figure()
fig.add_scatter(x=h["Date"], y=h["Units_Sold"], name="Daily sales", line=dict(color="#a8dadc"))
fig.add_scatter(x=h["Date"], y=h["Units_Sold"].rolling(7).mean(), name="7-day average", line=dict(color="#264653", width=3))
fig.add_scatter(x=fdates, y=daily_fc, name="Forecast (avg per day)", line=dict(color="#e76f51", width=3, dash="dash"))
for hh, color in ((7, "#f4a261"), (15, "#e9c46a"), (30, "#2a9d8f")):
    fig.add_vline(x=as_of + pd.Timedelta(days=hh), line_dash="dot", line_color=color, annotation_text=f"{hh}d")
fig.update_layout(height=420, yaxis_title="Units per day", legend=dict(orientation="h", y=-0.15))
ui.chart(fig)
st.caption("Each horizon has its own model and predicts a TOTAL. The dashed line spreads those totals into an average per-day "
           "level (days 1-7, 8-15, 16-30) for plotting.")

st.subheader("Actual vs predicted (held-out test period)")
ft = d["forecast_test"]
hsel = st.radio("Forecast horizon", [7, 15, 30], horizontal=True)
t = ft[(ft["Medicine_ID"] == mid) & (ft["Horizon"] == hsel)].sort_values("Origin_Date")
if t.empty:
    st.info("No test-period predictions for this medicine.")
else:
    fig2 = go.Figure()
    fig2.add_scatter(x=t["Origin_Date"], y=t["Actual"], name=f"Actual units in next {hsel} days", line=dict(color="#264653"))
    fig2.add_scatter(x=t["Origin_Date"], y=t["Predicted"], name="Predicted", line=dict(color="#e76f51", dash="dash"))
    fig2.update_layout(height=380, xaxis_title="Forecast origin date",
                       yaxis_title=f"Units over next {hsel} days", legend=dict(orientation="h", y=-0.2))
    ui.chart(fig2)
    err = (t["Actual"] - t["Predicted"]).abs()
    nz = t["Actual"] != 0
    m = st.columns(3)
    m[0].metric("MAE (this medicine)", f"{err.mean():.1f} units")
    m[1].metric("RMSE (this medicine)", f"{np.sqrt(((t['Actual'] - t['Predicted']) ** 2).mean()):.1f} units")
    m[2].metric("MAPE (this medicine)", f"{(err[nz] / t['Actual'][nz]).mean() * 100:.1f}%" if nz.any() else "n/a")
    st.caption(f"Model used: {t['Model'].iloc[0]} (selected on a validation period, evaluated on the last 90 days).")
ui.footer()
