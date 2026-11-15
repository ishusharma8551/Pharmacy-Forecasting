"""Module 2: expiry-risk classification (Low / Medium / High).

IMPORTANT: outputs are INVENTORY DECISION-SUPPORT ESTIMATES. They are not medical,
clinical or regulatory determinations. Actual expiry handling must follow the
pharmacy's SOPs and applicable drug regulations.

How the labels are built (avoids circular logic)
------------------------------------------------
We create historical inventory SNAPSHOTS: "on date t a batch with S units had E days left".
  * Features are computed from information available AT t only
    (trailing sales, an out-of-time demand forecast, stock, days to expiry).
  * The LABEL comes from what actually happened afterwards:
        unsold_ratio = (S - units actually sold between t and expiry) / S
        Low    : unsold_ratio <  RISK_LOW_MAX   (0.15)
        Medium : 0.15 <= unsold_ratio < RISK_HIGH_MIN (0.50)
        High   : unsold_ratio >= 0.50
  * The classifier therefore predicts a future outcome; it does not just re-apply the rule.
  * A rule baseline (same thresholds applied to the FORECAST unsold ratio) is reported so you can
    see whether machine learning adds anything over the simple rule.
Because the sales data is synthetic, the scores show the pipeline works, not real-world accuracy.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from src.demand_forecasting import rolling_origin_forecast
from src.feature_engineering import build_demand_features

RISK_FEATURES = ["days_until_expiry", "current_stock", "avg_daily_sales_28", "predicted_daily_demand",
                 "stock_turnover", "batch_quantity", "expected_unsold_ratio", "category_code"]


def label_risk(unsold_ratio):
    """Map an unsold ratio (0..1) to 'Low' / 'Medium' / 'High'."""
    r = np.asarray(unsold_ratio, dtype=float)
    return np.where(r < config.RISK_LOW_MAX, "Low", np.where(r >= config.RISK_HIGH_MIN, "High", "Medium"))


# --------------------------------------------------------------- features
def make_risk_features(df, category_map):
    """Input columns: days_until_expiry, current_stock, avg_daily_sales_28, predicted_daily_demand,
    batch_quantity, category.  Adds the derived features used by the classifier."""
    d = df.copy()
    stock = np.maximum(d["current_stock"].astype(float), 1.0)
    d["expected_units_sold"] = np.minimum(d["current_stock"], d["predicted_daily_demand"] * d["days_until_expiry"])
    d["expected_unsold"] = d["current_stock"] - d["expected_units_sold"]       # = Stock - expected sold before expiry
    d["expected_unsold_ratio"] = d["expected_unsold"] / stock
    d["stock_turnover"] = d["avg_daily_sales_28"] * 30.0 / stock               # months of sales / stock
    d["category_code"] = d["category"].map(category_map).fillna(-1).astype(int)
    return d


# -------------------------------------------------------------- snapshots
def build_snapshots(sales, medicines, feat, n_per_med=None, seed=None, n_estimators=60):
    """Historical inventory snapshots with features (as of t) and true outcome labels."""
    n_per_med = n_per_med or config.SNAPSHOTS_PER_MEDICINE
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    wide = sales.pivot(index="Date", columns="Medicine_ID", values="Units_Sold").sort_index()
    dates = wide.index
    P = np.vstack([np.zeros(wide.shape[1]), wide.values.cumsum(axis=0)])        # P[k] = sum of first k days
    col = {m: j for j, m in enumerate(wide.columns)}
    n = len(dates)
    min_i = 150 if n > 400 else 60
    max_shelf = min(180, max((n - min_i) // 2, 20))
    rows = []
    for _, m in medicines.iterrows():
        j = col.get(m["Medicine_ID"])
        if j is None:
            continue
        for _ in range(n_per_med):
            shelf = int(rng.integers(15, max_shelf + 1))
            i = int(rng.integers(min_i, n - shelf))
            avg28 = (P[i + 1, j] - P[i - 27, j]) / 28.0
            cover = rng.uniform(0.3, 3.0)                                       # stock relative to expected need
            stock = max(int(round(avg28 * shelf * cover)), 5)
            future = P[i + shelf + 1, j] - P[i + 1, j]                          # ACTUAL units sold t+1..expiry
            sold = min(stock, future)
            rows.append({
                "Medicine_ID": m["Medicine_ID"], "category": m["Category"],
                "Snapshot_Date": dates[i], "Expiry_Date": dates[i] + pd.Timedelta(days=shelf),
                "days_until_expiry": shelf, "current_stock": stock,
                "batch_quantity": int(np.ceil(stock / rng.uniform(0.6, 1.0))),
                "avg_daily_sales_28": avg28,
                "actual_units_sold": sold, "actual_unsold": stock - sold,
                "actual_unsold_ratio": (stock - sold) / stock,
            })
    snaps = pd.DataFrame(rows)
    snaps["predicted_daily_demand"] = rolling_origin_forecast(feat, snaps, 30, 60, n_estimators) / 30.0
    snaps["Risk_Label"] = label_risk(snaps["actual_unsold_ratio"])
    cat_map = {c: i for i, c in enumerate(sorted(medicines["Category"].unique()))}
    snaps = make_risk_features(snaps, cat_map)
    return snaps.sort_values("Snapshot_Date").reset_index(drop=True), cat_map


# ----------------------------------------------------------------- models
def make_classifiers(n_estimators=150):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced",
                                                random_state=config.RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=config.RANDOM_SEED),
    }


def class_metrics(name, split, y_true, y_pred):
    labels = config.RISK_LABELS
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    hi = labels.index("High")
    return {"Model": name, "Split": split, "Accuracy": accuracy_score(y_true, y_pred),
            "Precision_macro": p.mean(), "Recall_macro": r.mean(), "F1_macro": f.mean(),
            "Precision_High": p[hi], "Recall_High": r[hi], "F1_High": f[hi], "N": len(y_true)}


def purged_split(df, cutoff):
    """train = snapshots whose OUTCOME (expiry) is known before `cutoff`; test = snapshots at/after cutoff."""
    train = df[(df["Snapshot_Date"] < cutoff) & (df["Expiry_Date"] < cutoff)]
    test = df[df["Snapshot_Date"] >= cutoff]
    return train, test


def feature_importance(model):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    else:
        vals = np.abs(est.coef_).mean(axis=0)
        vals = vals / vals.sum()
    return pd.DataFrame({"Feature": RISK_FEATURES, "Importance": vals}).sort_values("Importance", ascending=False)


def train_expiry_models(snaps, category_map, n_estimators=150, save=True, verbose=True):
    cutoff = snaps["Snapshot_Date"].quantile(0.75)                 # last 25% of snapshot dates = test
    train, test = purged_split(snaps, cutoff)
    val_cut = train["Snapshot_Date"].quantile(0.80)                # last 20% of training dates = validation
    subtrain, val = purged_split(train, val_cut)
    X = lambda d: d[RISK_FEATURES]  # noqa: E731
    rows, cms, fitted_test = [], {}, {}

    rule_val = label_risk(val["expected_unsold_ratio"])
    rule_test = label_risk(test["expected_unsold_ratio"])
    rows.append(class_metrics("Rule baseline (forecast unsold ratio)", "validation", val["Risk_Label"], rule_val))
    rows.append(class_metrics("Rule baseline (forecast unsold ratio)", "test", test["Risk_Label"], rule_test))
    cms["Rule baseline (forecast unsold ratio)"] = confusion_matrix(test["Risk_Label"], rule_test, labels=config.RISK_LABELS).tolist()

    for name, model in make_classifiers(n_estimators).items():
        mv = clone(model).fit(X(subtrain), subtrain["Risk_Label"])
        rows.append(class_metrics(name, "validation", val["Risk_Label"], mv.predict(X(val))))
        mt = clone(model).fit(X(train), train["Risk_Label"])
        fitted_test[name] = mt
        pred = mt.predict(X(test))
        rows.append(class_metrics(name, "test", test["Risk_Label"], pred))
        cms[name] = confusion_matrix(test["Risk_Label"], pred, labels=config.RISK_LABELS).tolist()

    metrics = pd.DataFrame(rows)
    v = metrics[(metrics.Split == "validation") & ~metrics.Model.str.startswith("Rule")]
    best = v.sort_values(["Recall_High", "F1_macro"], ascending=False).iloc[0]["Model"]   # High-risk recall first
    metrics["Selected"] = metrics["Model"] == best

    final = clone(make_classifiers(n_estimators)[best]).fit(X(snaps), snaps["Risk_Label"])  # refit on all snapshots
    imp = feature_importance(final)
    imp["Model"] = best
    bundle = {"model": final, "name": best, "features": RISK_FEATURES, "category_map": category_map,
              "classes": list(final.classes_), "cutoff_date": str(cutoff.date())}
    dist = {"train": train["Risk_Label"].value_counts().to_dict(), "test": test["Risk_Label"].value_counts().to_dict()}
    if verbose:
        print(f"  snapshots: {len(snaps)} | train {len(train)} | validation {len(val)} | test {len(test)}")
        print(f"  test label distribution: {dist['test']}")
        print(metrics[metrics.Split == "test"][["Model", "Accuracy", "Precision_macro", "Recall_macro",
                                                 "F1_macro", "Recall_High"]].round(3).to_string(index=False))
        print(f"  selected on validation (High-risk recall, then macro-F1): {best}")
    if save:
        import sklearn
        bundle["sklearn_version"] = sklearn.__version__
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, config.MODELS_DIR / "expiry_model.pkl", compress=3)
        metrics.to_csv(config.REPORTS_DIR / "expiry_model_comparison.csv", index=False)
        imp.to_csv(config.REPORTS_DIR / "expiry_feature_importance.csv", index=False)
        (config.REPORTS_DIR / "expiry_confusion_matrices.json").write_text(
            json.dumps({"labels": config.RISK_LABELS, "matrices": cms, "label_distribution": dist}, indent=2))
        snaps.drop(columns=["category"]).to_csv(config.PROCESSED_DIR / "expiry_snapshots.csv", index=False)
    return {"bundle": bundle, "metrics": metrics, "confusion": cms, "importance": imp, "best": best}


# ------------------------------------------------- scoring CURRENT batches
def score_batches(batches, medicines, stats, bundle, as_of):
    """Score the batches currently in stock.

    FEFO logic: within a medicine, demand is consumed by the earliest-expiring batch first.
    For batch j, units it can sell before ITS expiry = forecast demand until its expiry minus the
    units earlier batches already take. Each batch is then classified by the trained model using the
    demand that is really available to it.  stats needs: Medicine_ID, Forecast_30d, Avg_Daily_28.
    """
    as_of = pd.Timestamp(as_of)
    b = (batches.merge(medicines[["Medicine_ID", "Medicine_Name", "Category", "Purchase_Price", "Selling_Price"]],
                       on="Medicine_ID", how="left")
         .merge(stats[["Medicine_ID", "Forecast_30d", "Avg_Daily_28"]], on="Medicine_ID", how="left"))
    b["Avg_Daily_28"] = b["Avg_Daily_28"].fillna(0.0)
    b["Forecast_Daily"] = (b["Forecast_30d"] / 30.0).fillna(b["Avg_Daily_28"])
    b["Days_Until_Expiry"] = (pd.to_datetime(b["Expiry_Date"]) - as_of).dt.days
    b = b.sort_values(["Medicine_ID", "Expiry_Date"]).reset_index(drop=True)

    avail_scale = np.zeros(len(b))
    exp_sold = np.zeros(len(b))
    for _, idx in b.groupby("Medicine_ID").groups.items():
        consumed = 0.0
        for i in idx:                                            # idx already ordered by expiry (FEFO)
            d, stock, rate = b.at[i, "Days_Until_Expiry"], b.at[i, "Current_Stock"], b.at[i, "Forecast_Daily"]
            if d <= 0:
                continue                                         # expired: nothing can be sold
            total = rate * d
            avail = max(total - consumed, 0.0)
            sold = min(float(stock), avail)
            consumed += sold
            exp_sold[i] = sold
            avail_scale[i] = avail / total if total > 0 else 0.0

    b["Expected_Units_Sold"] = np.round(exp_sold, 1)
    b["Expected_Unsold_Stock"] = np.where(b["Days_Until_Expiry"] <= 0, b["Current_Stock"],
                                          np.round(b["Current_Stock"] - exp_sold, 1))
    b["Expected_Unsold_Value"] = (b["Expected_Unsold_Stock"] * b["Purchase_Price"]).round(2)

    feats = pd.DataFrame({
        "days_until_expiry": b["Days_Until_Expiry"].clip(lower=1), "current_stock": b["Current_Stock"],
        "batch_quantity": b["Quantity_Received"], "category": b["Category"],
        "avg_daily_sales_28": b["Avg_Daily_28"] * avail_scale,
        "predicted_daily_demand": b["Forecast_Daily"] * avail_scale})
    feats = make_risk_features(feats, bundle["category_map"])
    model = bundle["model"]
    b["Risk_Class"] = model.predict(feats[bundle["features"]])
    hi = list(model.classes_).index("High")
    b["Risk_Prob_High"] = model.predict_proba(feats[bundle["features"]])[:, hi].round(3)
    expired = b["Days_Until_Expiry"] <= 0
    b.loc[expired, "Risk_Class"] = "Expired"
    b.loc[expired, "Risk_Prob_High"] = 1.0
    b["Status"] = np.where(expired, "Expired", np.where(b["Days_Until_Expiry"] <= config.NEAR_EXPIRY_DAYS,
                                                       f"Near expiry (<= {config.NEAR_EXPIRY_DAYS}d)", "OK"))
    b["FEFO_Rank"] = 0                                          # expired batches are not dispensed (rank 0)
    live = ~expired
    b.loc[live, "FEFO_Rank"] = b[live].groupby("Medicine_ID")["Days_Until_Expiry"].rank(method="first").astype(int)
    cols = ["Batch_Number", "Medicine_ID", "Medicine_Name", "Category", "Purchase_Date", "Manufacturing_Date",
            "Expiry_Date", "Days_Until_Expiry", "Status", "Quantity_Received", "Current_Stock",
            "Expected_Units_Sold", "Expected_Unsold_Stock", "Expected_Unsold_Value", "Purchase_Price",
            "Risk_Class", "Risk_Prob_High", "FEFO_Rank"]
    return b[cols]
