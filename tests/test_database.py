import sqlite3

import pandas as pd
import pytest

from src import database as db


@pytest.fixture
def dbpath(tmp_path, small_data):
    p = tmp_path / "t.db"
    db.init_db(reset=True, path=p)
    db.save_table("medicines", small_data["medicines"], path=p)
    db.save_table("batches", small_data["batches"], path=p)
    return p


def test_read_back_and_dates(dbpath):
    b = db.read_table("batches", path=dbpath)
    assert pd.api.types.is_datetime64_any_dtype(b["Expiry_Date"])
    assert len(db.read_table("medicines", path=dbpath)) == 8


def test_medicine_crud(dbpath):
    rec = {"Medicine_ID": "MED900", "Medicine_Name": "Test", "Category": "C", "Manufacturer": "M", "Purchase_Price": 1.0,
           "Selling_Price": 2.0, "Minimum_Stock_Level": 1, "Maximum_Stock_Level": 9, "Supplier_Lead_Time": 3,
           "Seasonal_Demand": "Non-seasonal", "Data_Source": "TEST"}
    db.create_medicine(rec, path=dbpath)
    assert db.get_medicine("MED900", path=dbpath)["Medicine_Name"] == "Test"
    assert db.update_medicine("MED900", path=dbpath, Selling_Price=5.5) == 1
    assert db.get_medicine("MED900", path=dbpath)["Selling_Price"] == 5.5
    assert db.delete_medicine("MED900", path=dbpath) == 1
    assert db.get_medicine("MED900", path=dbpath) is None


def test_batch_crud_and_constraints(dbpath):
    first = db.get_batches(path=dbpath).iloc[0]
    assert db.update_batch(first["Batch_Number"], path=dbpath, Current_Stock=7) == 1
    assert db.get_batches(first["Medicine_ID"], path=dbpath).query("Batch_Number == @first['Batch_Number']")["Current_Stock"].iloc[0] == 7
    with pytest.raises(sqlite3.IntegrityError):                              # negative stock rejected
        db.update_batch(first["Batch_Number"], path=dbpath, Current_Stock=-1)
    with pytest.raises(sqlite3.IntegrityError):                              # unknown medicine (foreign key)
        db.create_batch({"Batch_Number": "X1", "Medicine_ID": "NOPE", "Expiry_Date": pd.Timestamp("2026-01-01"),
                         "Current_Stock": 1}, path=dbpath)
    assert db.delete_batch(first["Batch_Number"], path=dbpath) == 1


def test_update_rejects_unknown_columns(dbpath):
    with pytest.raises(ValueError):
        db.update_batch("whatever", path=dbpath, Evil="1; DROP TABLE batches")
