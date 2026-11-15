import numpy as np
import pandas as pd

from src import preprocessing as pp


def test_generated_data_shape_and_columns(small_data):
    d = small_data
    assert len(d["medicines"]) == 8
    assert len(d["sales"]) == 8 * 300
    assert set(pp.FLAT_REQUIRED_COLUMNS) <= set(d["flat"].columns)
    assert (d["sales"]["Units_Sold"] >= 0).all()
    assert (d["medicines"]["Data_Source"] == "SYNTHETIC").all()


def test_generated_data_passes_validation(small_data):
    d = small_data
    errors = [m for sev, m in pp.validate_tables(d["medicines"], d["batches"], d["sales"]) if sev == "error"]
    assert errors == []


def test_batches_dates_are_consistent(small_data):
    b = small_data["batches"]
    assert (b["Manufacturing_Date"] < b["Expiry_Date"]).all()
    assert (b["Current_Stock"] > 0).all()


def test_clean_sales_fixes_duplicates_missing_and_outliers(small_data):
    s = small_data["sales"].copy()
    s["Units_Sold"] = s["Units_Sold"].astype(float)
    s.loc[10, "Units_Sold"] = np.nan                       # missing
    s.loc[50, "Units_Sold"] = 10_000                       # absurd outlier
    s = pd.concat([s, s.iloc[[3, 4]]], ignore_index=True)  # 2 duplicates
    clean, rep = pp.clean_sales(s)
    assert rep["duplicates_removed"] == 2
    assert rep["missing_filled"] == 1
    assert rep["outliers_capped"] >= 1
    assert clean["Units_Sold"].isna().sum() == 0
    assert clean["Units_Sold"].max() < 10_000
    assert not clean.duplicated(["Medicine_ID", "Date"]).any()
    assert len(clean) == 8 * 300                           # complete grid


def test_validate_flat_inventory(small_data):
    flat = small_data["flat"].copy()
    errors, _ = pp.validate_flat_inventory(flat)
    assert errors == []
    errors, _ = pp.validate_flat_inventory(flat.drop(columns=["Expiry_Date"]))
    assert any("Missing required columns" in e for e in errors)
    bad = flat.copy()
    bad.loc[0, "Current_Stock"] = -5
    assert any("negative" in e.lower() for e in pp.validate_flat_inventory(bad)[0])
    bad = flat.copy()
    bad.loc[0, "Manufacturing_Date"] = bad.loc[0, "Expiry_Date"]
    assert any("before Expiry" in e for e in pp.validate_flat_inventory(bad)[0])


def test_split_flat_inventory_roundtrip(small_data):
    meds, batches = pp.split_flat_inventory(small_data["flat"])
    assert len(meds) == 8
    assert len(batches) == len(small_data["batches"])
