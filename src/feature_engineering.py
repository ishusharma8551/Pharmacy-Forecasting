"""Feature engineering for demand forecasting.

Design (direct multi-horizon forecasting):
  A row = (medicine, forecast origin date t).  Features use ONLY sales up to and
  including day t.  The targets are y_h = units sold on days t+1 ... t+h for
  h in {7, 15, 30}.  One model is trained per horizon.

  lag_k       = units sold k days before the origin (lag_0 = the origin day itself)
  roll_mean_w = mean units over the w days ending at the origin
  Nothing in a row looks past day t, so there is no feature leakage.
"""
import numpy as np
import pandas as pd

import config

# India-style seasons: 0 winter (Dec-Feb), 1 summer (Mar-Jun), 2 monsoon (Jul-Sep), 3 post-monsoon (Oct-Nov)
SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3}

CALENDAR_FEATURES = ["day_of_week", "day_of_month", "week_of_year", "month", "season", "is_weekend"]
LAG_FEATURES = [f"lag_{k}" for k in config.LAG_DAYS]
ROLL_FEATURES = ([f"roll_mean_{w}" for w in config.ROLL_WINDOWS] + ["roll_std_7", "roll_std_28"])
STATIC_FEATURES = ["medicine_code", "category_code", "seasonal_code"]
FEATURE_COLUMNS = CALENDAR_FEATURES + LAG_FEATURES + ROLL_FEATURES + STATIC_FEATURES


def add_calendar_features(df, date_col="Date"):
    d = pd.to_datetime(df[date_col])
    out = df.copy()
    out["day_of_week"] = d.dt.dayofweek
    out["day_of_month"] = d.dt.day
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    out["season"] = d.dt.month.map(SEASON_MAP)
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    return out


def make_encoders(medicines):
    """Stable integer codes for categorical columns (saved with the model)."""
    return {
        "medicine": {m: i for i, m in enumerate(sorted(medicines["Medicine_ID"].unique()))},
        "category": {c: i for i, c in enumerate(sorted(medicines["Category"].unique()))},
        "seasonal": {s: i for i, s in enumerate(sorted(medicines["Seasonal_Demand"].unique()))},
    }


def build_demand_features(sales, medicines, encoders=None, horizons=None):
    """Return (features_df, encoders). Target columns y_7, y_15, y_30 are NaN where the future is unknown."""
    horizons = horizons or config.FORECAST_HORIZONS
    encoders = encoders or make_encoders(medicines)
    df = sales.sort_values(["Medicine_ID", "Date"]).reset_index(drop=True).copy()
    g = df.groupby("Medicine_ID")["Units_Sold"]

    for k in config.LAG_DAYS:
        df[f"lag_{k}"] = g.shift(k)
    for w in config.ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = g.rolling(w).mean().reset_index(level=0, drop=True)
    for w in (7, 28):
        df[f"roll_std_{w}"] = g.rolling(w).std().reset_index(level=0, drop=True)

    cs = g.cumsum()
    for h in horizons:                       # units sold on days t+1 .. t+h
        df[f"y_{h}"] = cs.groupby(df["Medicine_ID"]).shift(-h) - cs

    df = add_calendar_features(df)
    meta = medicines.set_index("Medicine_ID")
    df["medicine_code"] = df["Medicine_ID"].map(encoders["medicine"]).fillna(-1).astype(int)
    df["category_code"] = df["Medicine_ID"].map(meta["Category"]).map(encoders["category"]).fillna(-1).astype(int)
    df["seasonal_code"] = df["Medicine_ID"].map(meta["Seasonal_Demand"]).map(encoders["seasonal"]).fillna(-1).astype(int)
    return df, encoders
