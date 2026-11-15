import numpy as np
import pandas as pd

import config
from src import demand_forecasting as dfc
from src.feature_engineering import FEATURE_COLUMNS, build_demand_features


def test_mape_ignores_zero_actuals_and_handles_all_zero():
    assert dfc.mape([0, 10, 20], [5, 11, 18]) == np.mean([0.1, 0.1]) * 100
    assert np.isnan(dfc.mape([0, 0], [1, 2]))


def test_mae_rmse_wape():
    a, p = [10, 20], [12, 17]
    assert dfc.mae(a, p) == 2.5
    assert np.isclose(dfc.rmse(a, p), np.sqrt((4 + 9) / 2))
    assert np.isclose(dfc.wape(a, p), 5 / 30 * 100)


def test_target_is_sum_of_next_h_days(small_data):
    feat, _ = build_demand_features(small_data["sales"], small_data["medicines"])
    one = feat[feat["Medicine_ID"] == "MED001"].reset_index(drop=True)
    i = 100
    assert one.loc[i, "y_7"] == one.loc[i + 1:i + 7, "Units_Sold"].sum()
    assert one.loc[i, "y_30"] == one.loc[i + 1:i + 30, "Units_Sold"].sum()
    assert one["y_30"].tail(30).isna().all()               # unknown future stays NaN


def test_features_do_not_leak_future(small_data):
    """Changing sales AFTER day T must not change any feature on or before day T."""
    sales, meds = small_data["sales"], small_data["medicines"]
    T = sales["Date"].min() + pd.Timedelta(days=150)
    a, _ = build_demand_features(sales, meds)
    altered = sales.copy()
    altered.loc[altered["Date"] > T, "Units_Sold"] += 999
    b, _ = build_demand_features(altered, meds)
    ca, cb = a[a["Date"] <= T], b[b["Date"] <= T]
    pd.testing.assert_frame_equal(ca[FEATURE_COLUMNS], cb[FEATURE_COLUMNS])


def test_chronological_split_has_no_overlap(small_data):
    feat, _ = build_demand_features(small_data["sales"], small_data["medicines"])
    for h in config.FORECAST_HORIZONS:
        sp = dfc.split_by_origin(feat, h, test_days=60, val_days=40)
        assert sp.train["Date"].max() + pd.Timedelta(days=h) < sp.test["Date"].min()    # train targets end before test starts
        assert sp.subtrain["Date"].max() + pd.Timedelta(days=h) < sp.val["Date"].min()
        assert sp.test["Date"].min() > sp.train["Date"].max()
        assert len(sp.subtrain) and len(sp.val) and len(sp.test)


def test_evaluate_horizon_returns_models_and_baselines(small_data):
    feat, _ = build_demand_features(small_data["sales"], small_data["medicines"])
    metrics, best, preds = dfc.evaluate_horizon(feat, 7, n_estimators=10)
    assert {"Linear Regression", "Random Forest"} <= set(metrics["Model"])
    assert "Moving Avg 28d x h" in set(metrics["Model"])
    assert best in set(metrics["Model"])
    assert (preds["Predicted"] >= 0).all()
    assert metrics[["MAE", "RMSE"]].notna().all().all()


def test_forecast_latest_shape(small_data):
    feat, enc = build_demand_features(small_data["sales"], small_data["medicines"])
    res = dfc.train_demand_models(small_data["sales"], small_data["medicines"], n_estimators=8, save=False, verbose=False)
    fc = dfc.forecast_latest(res["bundle"], small_data["sales"], small_data["medicines"])
    assert len(fc) == 8
    assert {"Forecast_7d", "Forecast_15d", "Forecast_30d"} <= set(fc.columns)
    assert (fc["Forecast_30d"] >= 0).all()
