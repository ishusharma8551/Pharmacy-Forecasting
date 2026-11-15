import numpy as np
import pandas as pd

import config
from src import inventory_optimization as inv


def test_basic_formulas():
    assert inv.lead_time_demand(10, 5) == 50
    ss = inv.safety_stock(4, 9, z=1.65)
    assert np.isclose(ss, 1.65 * 4 * 3)
    assert np.isclose(inv.reorder_point(50, ss), 50 + ss)


def test_order_quantity_never_negative_and_rounds_up():
    assert inv.order_quantity(100, 150) == 0
    assert inv.order_quantity(100.2, 50) == 51
    assert list(inv.order_quantity([100, 10], [30, 40])) == [70, 0]


def test_target_stock_capped_by_max_level_but_not_below_rop():
    t = inv.target_stock(10, 5, 20, max_level=1000, review_days=30)       # 10*35+20 = 370
    assert t == 370
    t = inv.target_stock(10, 5, 20, max_level=200, review_days=30)        # capped at 200 (>= ROP 70)
    assert t == 200
    t = inv.target_stock(10, 5, 20, max_level=10, review_days=30)         # cap below ROP -> use ROP (50+20)
    assert t == 70


def test_primary_action_priority():
    assert inv.primary_action(["Near Expiry", "Reorder Now"]) == "Reorder Now"
    assert inv.primary_action(["Near Expiry", "High Expiry Risk"]) == "High Expiry Risk"
    assert inv.primary_action([]) == "Stock Sufficient"


def _fixture(stock_a, stock_b):
    meds = pd.DataFrame({"Medicine_ID": ["M1", "M2"], "Medicine_Name": ["Low", "Over"], "Category": "C",
                         "Purchase_Price": 2.0, "Selling_Price": 3.0, "Supplier_Lead_Time": 5,
                         "Minimum_Stock_Level": 10, "Maximum_Stock_Level": 500})
    sb = pd.DataFrame({"Medicine_ID": ["M1", "M2"], "Current_Stock": [stock_a, stock_b], "Days_Until_Expiry": [200, 200],
                       "Expected_Unsold_Value": [0.0, 100.0], "Expected_Unsold_Stock": [0.0, 50.0],
                       "Risk_Class": ["Low", "Low"]})
    stats = pd.DataFrame({"Medicine_ID": ["M1", "M2"], "Forecast_7d": 70.0, "Forecast_15d": 150.0, "Forecast_30d": 300.0,
                          "Avg_Daily_28": 10.0, "Sigma_Daily": 3.0})
    return meds, sb, stats


def test_recommendations_reorder_and_overstock():
    meds, sb, stats = _fixture(20, 900)
    r = inv.medicine_recommendations(meds, sb, stats).set_index("Medicine_ID")
    assert r.loc["M1", "Recommendation"] == "Reorder Now"
    assert r.loc["M1", "Recommended_Order_Qty"] > 0
    assert r.loc["M2", "Recommendation"] == "Overstock Warning"
    assert r.loc["M2", "Overstock_Qty"] == 400
    assert r.loc["M2", "Recommended_Order_Qty"] == 0
    assert np.isclose(r.loc["M1", "Reorder_Point"], 10 * 5 + config.SERVICE_LEVEL_Z * 3 * np.sqrt(5), atol=0.1)


def test_stock_sufficient():
    meds, sb, stats = _fixture(200, 200)
    sb["Expected_Unsold_Value"] = 0.0
    r = inv.medicine_recommendations(meds, sb, stats)
    assert (r["Recommendation"] == "Stock Sufficient").all()


def test_expired_stock_is_not_usable():
    meds, sb, stats = _fixture(300, 300)
    sb.loc[0, "Days_Until_Expiry"] = -5
    r = inv.medicine_recommendations(meds, sb, stats).set_index("Medicine_ID")
    assert r.loc["M1", "Usable_Stock"] == 0 and r.loc["M1", "Expired_Stock"] == 300
    assert r.loc["M1", "Recommendation"] == "Reorder Now"


def test_fefo_orders_by_expiry():
    sb = pd.DataFrame({"Medicine_ID": "M1", "Medicine_Name": "X", "Batch_Number": ["B2", "B1"],
                       "FEFO_Rank": [2, 1], "Expiry_Date": pd.to_datetime(["2025-06-01", "2025-03-01"]),
                       "Days_Until_Expiry": [150, 60], "Current_Stock": 10, "Risk_Class": "Low", "Expected_Unsold_Stock": 0.0})
    f = inv.fefo_table(sb)
    assert list(f["Batch_Number"]) == ["B1", "B2"]
    assert "first" in f.iloc[0]["FEFO_Action"].lower()
