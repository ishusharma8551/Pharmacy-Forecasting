import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

import config
from src import expiry_prediction as ep


def test_label_risk_boundaries():
    got = ep.label_risk([0.0, 0.1499, 0.15, 0.4999, 0.5, 1.0])
    assert list(got) == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_expected_unsold_formula():
    df = pd.DataFrame({"days_until_expiry": [10, 10], "current_stock": [100, 100], "avg_daily_sales_28": [5.0, 20.0],
                       "predicted_daily_demand": [5.0, 20.0], "batch_quantity": [120, 120], "category": ["A", "A"]})
    out = ep.make_risk_features(df, {"A": 0})
    assert list(out["expected_units_sold"]) == [50.0, 100.0]          # capped at stock
    assert list(out["expected_unsold"]) == [50.0, 0.0]                # Stock - expected sold
    assert list(out["expected_unsold_ratio"]) == [0.5, 0.0]


def test_purged_split_excludes_unknown_outcomes():
    d = pd.DataFrame({"Snapshot_Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                      "Expiry_Date": pd.to_datetime(["2024-01-20", "2024-04-01", "2024-05-01"])})
    train, test = ep.purged_split(d, pd.Timestamp("2024-03-01"))
    assert list(train.index) == [0]            # row 1 expires AFTER cutoff -> outcome not yet known -> excluded
    assert list(test.index) == [2]


def test_snapshots_have_valid_labels(small_data):
    from src.feature_engineering import build_demand_features
    feat, _ = build_demand_features(small_data["sales"], small_data["medicines"])
    snaps, cat = ep.build_snapshots(small_data["sales"], small_data["medicines"], feat, n_per_med=10, n_estimators=8)
    assert set(snaps["Risk_Label"]) <= set(config.RISK_LABELS)
    assert snaps["actual_unsold_ratio"].between(0, 1).all()
    assert (snaps["actual_unsold"] <= snaps["current_stock"]).all()
    assert snaps["predicted_daily_demand"].notna().all()
    assert (snaps["Expiry_Date"] > snaps["Snapshot_Date"]).all()
    assert set(ep.RISK_FEATURES) <= set(snaps.columns)


@pytest.fixture
def dummy_bundle():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((60, len(ep.RISK_FEATURES))), columns=ep.RISK_FEATURES)
    y = np.array(["Low", "Medium", "High"] * 20)
    return {"model": DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y), "features": ep.RISK_FEATURES,
            "category_map": {"Cat": 0}}


def test_score_batches_fefo_allocation_and_expired(dummy_bundle):
    as_of = pd.Timestamp("2025-01-01")
    meds = pd.DataFrame({"Medicine_ID": ["M1"], "Medicine_Name": ["X"], "Category": ["Cat"],
                         "Purchase_Price": [2.0], "Selling_Price": [3.0]})
    batches = pd.DataFrame({
        "Batch_Number": ["A", "B", "C"], "Medicine_ID": "M1",
        "Purchase_Date": as_of, "Manufacturing_Date": as_of - pd.Timedelta(days=300),
        "Expiry_Date": [as_of + pd.Timedelta(days=5), as_of + pd.Timedelta(days=20), as_of - pd.Timedelta(days=3)],
        "Quantity_Received": [100, 100, 40], "Current_Stock": [100, 100, 40]})
    stats = pd.DataFrame({"Medicine_ID": ["M1"], "Forecast_30d": [300.0], "Avg_Daily_28": [10.0]})   # 10 units/day
    out = ep.score_batches(batches, meds, stats, dummy_bundle, as_of).set_index("Batch_Number")
    assert out.loc["A", "Expected_Units_Sold"] == 50 and out.loc["A", "Expected_Unsold_Stock"] == 50   # 10/day x 5 days
    assert out.loc["B", "Expected_Units_Sold"] == 100 and out.loc["B", "Expected_Unsold_Stock"] == 0   # 200 demand - 50 used by A = 150 >= 100
    assert out.loc["C", "Risk_Class"] == "Expired" and out.loc["C", "Expected_Unsold_Stock"] == 40
    assert out.loc["A", "FEFO_Rank"] == 1 and out.loc["B", "FEFO_Rank"] == 2 and out.loc["C", "FEFO_Rank"] == 0
    assert (out["Expected_Unsold_Stock"] >= 0).all()
    assert set(out["Risk_Class"]) <= {"Low", "Medium", "High", "Expired"}
