"""SQLite storage layer with basic CRUD.

Core tables (explicit schema + foreign keys): medicines, batches, sales_daily.
Derived tables (rewritten whenever predictions are refreshed): forecasts, forecast_test,
expiry_predictions, recommendations, fefo, meta.
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS medicines (
    Medicine_ID TEXT PRIMARY KEY, Medicine_Name TEXT NOT NULL, Category TEXT, Manufacturer TEXT,
    Purchase_Price REAL NOT NULL, Selling_Price REAL NOT NULL, Minimum_Stock_Level INTEGER,
    Maximum_Stock_Level INTEGER, Supplier_Lead_Time INTEGER NOT NULL, Seasonal_Demand TEXT, Data_Source TEXT);
CREATE TABLE IF NOT EXISTS batches (
    Batch_Number TEXT PRIMARY KEY,
    Medicine_ID TEXT NOT NULL REFERENCES medicines(Medicine_ID) ON DELETE CASCADE,
    Purchase_Date TEXT, Manufacturing_Date TEXT, Expiry_Date TEXT NOT NULL,
    Quantity_Received INTEGER, Current_Stock INTEGER NOT NULL CHECK (Current_Stock >= 0));
CREATE TABLE IF NOT EXISTS sales_daily (
    Date TEXT NOT NULL, Medicine_ID TEXT NOT NULL REFERENCES medicines(Medicine_ID) ON DELETE CASCADE,
    Units_Sold REAL NOT NULL, PRIMARY KEY (Date, Medicine_ID));
CREATE INDEX IF NOT EXISTS idx_batches_med ON batches(Medicine_ID);
CREATE INDEX IF NOT EXISTS idx_sales_med ON sales_daily(Medicine_ID);
"""
CORE_TABLES = {"medicines", "batches", "sales_daily"}
DATE_COLUMNS = ["Date", "Purchase_Date", "Manufacturing_Date", "Expiry_Date", "Forecast_Origin", "Origin_Date"]
MEDICINE_FIELDS = ["Medicine_Name", "Category", "Manufacturer", "Purchase_Price", "Selling_Price",
                   "Minimum_Stock_Level", "Maximum_Stock_Level", "Supplier_Lead_Time", "Seasonal_Demand", "Data_Source"]
BATCH_FIELDS = ["Medicine_ID", "Purchase_Date", "Manufacturing_Date", "Expiry_Date", "Quantity_Received", "Current_Stock"]


@contextmanager
def get_connection(path=None):
    con = sqlite3.connect(str(path or config.DB_PATH))
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(reset=False, path=None):
    p = path or config.DB_PATH
    if reset and p.exists():
        p.unlink()
    with get_connection(p) as con:
        con.executescript(SCHEMA)


def _prep(df):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
    for c in d.columns:                                   # bool -> int for SQLite
        if d[c].dtype == bool:
            d[c] = d[c].astype(int)
    return d


def save_table(name, df, path=None):
    """Core tables are appended (schema kept); derived tables are replaced."""
    with get_connection(path) as con:
        if name in CORE_TABLES:
            _prep(df).to_sql(name, con, if_exists="append", index=False)
        else:
            _prep(df).to_sql(name, con, if_exists="replace", index=False)


def read_table(name, path=None):
    with get_connection(path) as con:
        df = pd.read_sql_query(f"SELECT * FROM {name}", con)   # name is never user-supplied
    for c in DATE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    for c in df.columns:                                  # SQLite stores booleans as 0/1
        if c.startswith("Flag_"):
            df[c] = df[c].astype(bool)
    return df


def table_exists(name, path=None):
    with get_connection(path) as con:
        return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _fmt(v):
    return v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else v


# ------------------------------------------------------------------- CRUD
def create_medicine(rec, path=None):
    cols = ["Medicine_ID"] + MEDICINE_FIELDS
    vals = [_fmt(rec.get(c)) for c in cols]
    with get_connection(path) as con:
        con.execute(f"INSERT INTO medicines ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)


def get_medicine(medicine_id, path=None):
    with get_connection(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM medicines WHERE Medicine_ID = ?", (medicine_id,)).fetchone()
    return dict(row) if row else None


def update_medicine(medicine_id, path=None, **fields):
    bad = set(fields) - set(MEDICINE_FIELDS)
    if bad or not fields:
        raise ValueError(f"Invalid or empty fields: {sorted(bad)}")
    sets = ", ".join(f"{k} = ?" for k in fields)
    with get_connection(path) as con:
        return con.execute(f"UPDATE medicines SET {sets} WHERE Medicine_ID = ?",
                           [*map(_fmt, fields.values()), medicine_id]).rowcount


def delete_medicine(medicine_id, path=None):
    """Also removes the medicine's batches and sales (ON DELETE CASCADE)."""
    with get_connection(path) as con:
        return con.execute("DELETE FROM medicines WHERE Medicine_ID = ?", (medicine_id,)).rowcount


def create_batch(rec, path=None):
    cols = ["Batch_Number"] + BATCH_FIELDS
    vals = [_fmt(rec.get(c)) for c in cols]
    with get_connection(path) as con:
        con.execute(f"INSERT INTO batches ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)


def get_batches(medicine_id=None, path=None):
    q, args = "SELECT * FROM batches", ()
    if medicine_id:
        q, args = q + " WHERE Medicine_ID = ?", (medicine_id,)
    with get_connection(path) as con:
        return pd.read_sql_query(q + " ORDER BY Expiry_Date", con, params=args)


def update_batch(batch_number, path=None, **fields):
    bad = set(fields) - set(BATCH_FIELDS)
    if bad or not fields:
        raise ValueError(f"Invalid or empty fields: {sorted(bad)}")
    sets = ", ".join(f"{k} = ?" for k in fields)
    with get_connection(path) as con:
        return con.execute(f"UPDATE batches SET {sets} WHERE Batch_Number = ?",
                           [*map(_fmt, fields.values()), batch_number]).rowcount


def delete_batch(batch_number, path=None):
    with get_connection(path) as con:
        return con.execute("DELETE FROM batches WHERE Batch_Number = ?", (batch_number,)).rowcount
