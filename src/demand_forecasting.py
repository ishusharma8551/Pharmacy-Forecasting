"""Module 1: demand forecasting (7 / 15 / 30 days).

Evaluation protocol (no leakage):
  * Chronological split on forecast ORIGIN dates.
  * test       : origins in the last TEST_DAYS days
  * train      : origins whose target window ends BEFORE the test period starts
  * validation : the last VAL_DAYS days of the training origins (model selection)
  * The best ML model is chosen on VALIDATION error, then its TEST error is reported.
Baselines (no learning) must be beaten by the ML models to justify them.
"""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.feature_engineering import FEATURE_COLUMNS, build_demand_features

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


# ------------------------------------------------------------------ metrics
def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))


def rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(p)) ** 2)))


def mape(a, p):
    """MAPE in %, computed only where actual != 0 (division by zero is undefined)."""
    a, p = np.asarray(a, float), np.asarray(p, float)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def wape(a, p):
    """Weighted APE in % = sum|error| / sum(actual). Robust companion to MAPE."""
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.abs(a - p).sum() / a.sum() * 100) if a.sum() != 0 else float("nan")


def metric_row(name, kind, split, horizon, a, p):
    return {"Horizon": horizon, "Model": name, "Type": kind, "Split": split,
            "MAE": mae(a, p), "RMSE": rmse(a, p), "MAPE_%": mape(a, p), "WAPE_%": wape(a, p)}


# ---------------------------------------------------------------- baselines
BASELINES = {
    "Naive (last day x h)": lambda df, h: df["lag_0"] * h,
    "Moving Avg 7d x h": lambda df, h: df["roll_mean_7"] * h,
    "Moving Avg 28d x h": lambda df, h: df["roll_mean_28"] * h,
}


def make_models(n_estimators=100):
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=n_estimators, max_depth=10, min_samples_leaf=20,
                                               n_jobs=-1, random_state=config.RANDOM_SEED),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=n_estimators * 2, max_depth=4, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                         random_state=config.RANDOM_SEED, verbosity=0)
    return models


# ---------------------------------------------------------------- splitting
def split_by_origin(feat, h, test_days=None, val_days=None):
    """Chronological, purged split. Returns SimpleNamespace(subtrain, val, train, test)."""
    test_days = test_days or config.TEST_DAYS
    val_days = val_days or config.VAL_DAYS
    target = f"y_{h}"
    d = feat.dropna(subset=FEATURE_COLUMNS + [target])
    last = feat["Date"].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    train_end = test_start - pd.Timedelta(days=h + 1)       # target window ends before test_start
    val_start = train_end - pd.Timedelta(days=val_days - 1)
    subtrain_end = val_start - pd.Timedelta(days=h + 1)     # target window ends before val_start
    return SimpleNamespace(
        subtrain=d[d["Date"] <= subtrain_end],
        val=d[(d["Date"] >= val_start) & (d["Date"] <= train_end)],
        train=d[d["Date"] <= train_end],
        test=d[d["Date"] >= test_start],
        test_start=test_start,
    )


# --------------------------------------------------------------- evaluation
def evaluate_horizon(feat, h, n_estimators=100):
    """Train/evaluate all models for one horizon. Returns (metrics_df, best_name, test_pred_df)."""
    target = f"y_{h}"
    sp = split_by_origin(feat, h)
    rows = []
    for split_name, part in (("validation", sp.val), ("test", sp.test)):
        for name, fn in BASELINES.items():
            rows.append(metric_row(name, "Baseline", split_name, h, part[target], fn(part, h)))

    fitted_val, fitted_test = {}, {}
    for name, model in make_models(n_estimators).items():
        fitted_val[name] = clone(model).fit(sp.subtrain[FEATURE_COLUMNS], sp.subtrain[target])
        fitted_test[name] = clone(model).fit(sp.train[FEATURE_COLUMNS], sp.train[target])
        pv = np.clip(fitted_val[name].predict(sp.val[FEATURE_COLUMNS]), 0, None)
        pt = np.clip(fitted_test[name].predict(sp.test[FEATURE_COLUMNS]), 0, None)
        rows.append(metric_row(name, "ML", "validation", h, sp.val[target], pv))
        rows.append(metric_row(name, "ML", "test", h, sp.test[target], pt))

    metrics = pd.DataFrame(rows)
    val_ml = metrics[(metrics.Split == "validation") & (metrics.Type == "ML")]
    best = val_ml.sort_values("MAE").iloc[0]["Model"]          # selected on VALIDATION error

    pred = np.clip(fitted_test[best].predict(sp.test[FEATURE_COLUMNS]), 0, None)
    test_pred = pd.DataFrame({"Medicine_ID": sp.test["Medicine_ID"].values, "Origin_Date": sp.test["Date"].values,
                              "Horizon": h, "Actual": sp.test[target].values, "Predicted": pred, "Model": best})
    metrics["Selected"] = (metrics["Model"] == best) & (metrics["Type"] == "ML")
    return metrics, best, test_pred


def feature_importance(model):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    else:
        vals = np.abs(est.coef_)
        vals = vals / vals.sum()
    return pd.Series(vals, index=FEATURE_COLUMNS)


def train_demand_models(sales, medicines, n_estimators=100, save=True, verbose=True):
    feat, enc = build_demand_features(sales, medicines)
    all_metrics, all_preds, importances = [], [], []
    bundle = {"models": {}, "best_models": {}, "features": FEATURE_COLUMNS, "encoders": enc,
              "trained_through": str(feat["Date"].max().date())}
    for h in config.FORECAST_HORIZONS:
        metrics, best, test_pred = evaluate_horizon(feat, h, n_estimators)
        all_metrics.append(metrics)
        all_preds.append(test_pred)
        d = feat.dropna(subset=FEATURE_COLUMNS + [f"y_{h}"])           # production refit on all known targets
        final = clone(make_models(n_estimators)[best]).fit(d[FEATURE_COLUMNS], d[f"y_{h}"])
        bundle["models"][h] = final
        bundle["best_models"][h] = best
        imp = feature_importance(final).rename("Importance").reset_index()
        imp.columns = ["Feature", "Importance"]
        imp.insert(0, "Horizon", h)
        imp["Model"] = best
        importances.append(imp)
        if verbose:
            t = metrics[(metrics.Split == "test")].sort_values("MAE")
            print(f"  [{h:>2}d] best on validation: {best}")
            print(t[["Model", "Type", "MAE", "RMSE", "MAPE_%", "WAPE_%"]].round(2).to_string(index=False))
    metrics = pd.concat(all_metrics, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    imp = pd.concat(importances, ignore_index=True)
    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        import sklearn
        bundle["sklearn_version"] = sklearn.__version__
        joblib.dump(bundle, config.MODELS_DIR / "demand_model.pkl", compress=3)
        metrics.to_csv(config.REPORTS_DIR / "demand_model_comparison.csv", index=False)
        preds.to_csv(config.REPORTS_DIR / "forecast_test_predictions.csv", index=False)
        imp.to_csv(config.REPORTS_DIR / "demand_feature_importance.csv", index=False)
    return {"bundle": bundle, "metrics": metrics, "test_predictions": preds, "importance": imp, "features": feat}


# --------------------------------------------------------------- forecasting
def forecast_latest(bundle, sales, medicines):
    """Forecast 7/15/30-day demand for every medicine from the latest available date."""
    feat, _ = build_demand_features(sales, medicines, encoders=bundle["encoders"])
    last = feat["Date"].max()
    latest = feat[feat["Date"] == last].dropna(subset=bundle["features"])
    out = pd.DataFrame({"Medicine_ID": latest["Medicine_ID"].values, "Forecast_Origin": last})
    for h, model in bundle["models"].items():
        out[f"Forecast_{h}d"] = np.clip(model.predict(latest[bundle["features"]]), 0, None).round(1)
    return out.reset_index(drop=True)


def rolling_origin_forecast(feat, snaps, horizon=30, step_days=60, n_estimators=60):
    """Honest out-of-time `horizon`-day demand forecast for historical inventory snapshots.

    The expiry-risk model needs 'predicted future demand' for snapshots in the PAST. Using the
    final model would be in-sample. Instead, for each block of `step_days` days, a Random Forest is
    trained ONLY on origins whose targets finished before the block starts.
    """
    target = f"y_{horizon}"
    s = snaps[["Medicine_ID", "Snapshot_Date"]].reset_index(drop=True)
    out = pd.Series(np.nan, index=s.index)
    step = pd.Timedelta(days=step_days)
    cols = ["Medicine_ID", "Date"] + FEATURE_COLUMNS
    lookup = feat[cols].rename(columns={"Date": "Snapshot_Date"})
    b0 = s["Snapshot_Date"].min()
    while b0 <= s["Snapshot_Date"].max():
        mask = (s["Snapshot_Date"] >= b0) & (s["Snapshot_Date"] < b0 + step)
        if mask.any():
            train = feat[feat["Date"] <= b0 - pd.Timedelta(days=horizon + 1)].dropna(subset=FEATURE_COLUMNS + [target])
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=10, min_samples_leaf=20,
                                          n_jobs=-1, random_state=config.RANDOM_SEED)
            model.fit(train[FEATURE_COLUMNS], train[target])
            block = s[mask].reset_index().merge(lookup, on=["Medicine_ID", "Snapshot_Date"], how="left")
            out.loc[block["index"].values] = np.clip(model.predict(block[FEATURE_COLUMNS]), 0, None)
        b0 += step
    return out.values
