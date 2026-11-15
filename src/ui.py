"""Shared Streamlit helpers (page setup, cached data loading, version-safe widgets)."""
import json

import pandas as pd
import streamlit as st

import config
from src import database as db

RISK_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e63946", "Expired": "#6d6875"}
EXPIRY_DISCLAIMER = ("Expiry-risk outputs are **inventory decision-support estimates** based on forecast demand. "
                     "They are not medical, clinical or regulatory determinations. Always follow your pharmacy's SOPs "
                     "and applicable drug regulations for expired or near-expiry stock.")


def setup(title, icon="💊"):
    st.set_page_config(page_title=f"{title} | Pharmacy Forecasting", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


@st.cache_data(show_spinner=False)
def _load_all(db_mtime):
    names = ["medicines", "batches", "sales_daily", "forecasts", "forecast_test",
             "expiry_predictions", "recommendations", "fefo"]
    data = {n: db.read_table(n) for n in names}
    meta = db.read_table("meta").set_index("key")["value"].to_dict()
    data["as_of"] = pd.Timestamp(meta["as_of"])
    data["meta"] = meta
    return data


def get_data():
    """Load everything from SQLite (cached; refreshes when the DB file changes)."""
    if not config.DB_PATH.exists():
        st.error("Database not found. Open a terminal in the project folder and run:  `python train.py`")
        st.stop()
    data = _load_all(config.DB_PATH.stat().st_mtime)
    with st.sidebar:
        st.markdown("### Pharmacy Forecasting")
        st.caption(f"Data as of **{data['as_of']:%d %b %Y}**")
        st.warning(config.DATA_DISCLAIMER)
    return data


def load_report(name):
    p = config.REPORTS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if name.endswith(".json") else pd.read_csv(p)


def chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                                   # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def money(x):
    return f"{config.CURRENCY} {x:,.0f}"


def footer():
    st.divider()
    st.caption(config.DATA_DISCLAIMER + " | Decision-support tool, not medical or regulatory advice.")
