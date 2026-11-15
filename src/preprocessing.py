"""Loading, validation and cleaning of pharmacy data.

Cleaning policy (documented so it can be defended):
  * duplicates     -> drop repeated (Medicine_ID, Date) rows, keep the first
  * missing / <0   -> mark as missing, then linearly interpolate per medicine
  * outliers       -> a value above  median + 10 * robust_sd  is treated as a
                      data-entry error and replaced by the 7-day rolling median.
                      robust_sd = max(1.4826 * MAD, sqrt(median)).
                      Genuine promotional spikes (~2-3x) stay below this limit.
"""
import numpy as np
import pandas as pd

import config

FLAT_REQUIRED_COLUMNS = [
    "Medicine_ID", "Medicine_Name", "Category", "Manufacturer", "Purchase_Price", "Selling_Price",
    "Current_Stock", "Minimum_Stock_Level", "Maximum_Stock_Level", "Daily_Sales", "Monthly_Sales",
    "Supplier_Lead_Time", "Purchase_Date", "Manufacturing_Date", "Expiry_Date", "Batch_Number",
    "Seasonal_Demand", "Historical_Sales",
]
MEDICINE_COLUMNS = ["Medicine_ID", "Medicine_Name", "Category", "Manufacturer", "Purchase_Price",
                    "Selling_Price", "Minimum_Stock_Level", "Maximum_Stock_Level",
                    "Supplier_Lead_Time", "Seasonal_Demand", "Data_Source"]
BATCH_COLUMNS = ["Batch_Number", "Medicine_ID", "Purchase_Date", "Manufacturing_Date",
                 "Expiry_Date", "Quantity_Received", "Current_Stock"]


# ------------------------------------------------------------------ loading
def load_raw():
    meds = pd.read_csv(config.RAW_DIR / "medicines.csv")
    batches = pd.read_csv(config.RAW_DIR / "batches.csv",
                          parse_dates=["Purchase_Date", "Manufacturing_Date", "Expiry_Date"])
    sales = pd.read_csv(config.RAW_DIR / "sales_daily.csv", parse_dates=["Date"])
    return meds, batches, sales


# --------------------------------------------------------------- validation
def validate_tables(medicines, batches, sales=None):
    """Return a list of (severity, message). severity is 'error' or 'warning'."""
    issues = []

    def add(sev, msg):
        issues.append((sev, msg))

    if medicines["Medicine_ID"].duplicated().any():
        add("error", "Duplicate Medicine_ID values in medicines table")
    if (medicines["Selling_Price"] < medicines["Purchase_Price"]).any():
        add("warning", "Some medicines have Selling_Price below Purchase_Price")
    if (medicines["Minimum_Stock_Level"] > medicines["Maximum_Stock_Level"]).any():
        add("error", "Minimum_Stock_Level exceeds Maximum_Stock_Level for some medicines")
    if (medicines["Supplier_Lead_Time"] <= 0).any():
        add("error", "Supplier_Lead_Time must be positive")
    for col in ["Medicine_Name", "Category", "Purchase_Price", "Selling_Price"]:
        if medicines[col].isna().any():
            add("error", f"Missing values in medicines.{col}")
    if medicines["Manufacturer"].isna().any():
        add("warning", "Missing Manufacturer values (will be filled with 'Unknown')")

    if batches["Batch_Number"].duplicated().any():
        add("error", "Duplicate Batch_Number values")
    if (batches["Current_Stock"] < 0).any():
        add("error", "Negative Current_Stock")
    if (batches["Manufacturing_Date"] >= batches["Expiry_Date"]).any():
        add("error", "Manufacturing_Date must be before Expiry_Date")
    if (batches["Purchase_Date"] < batches["Manufacturing_Date"]).any():
        add("warning", "Some batches were purchased before their manufacturing date")
    unknown = set(batches["Medicine_ID"]) - set(medicines["Medicine_ID"])
    if unknown:
        add("error", f"Batches reference unknown Medicine_IDs: {sorted(unknown)[:5]}")

    if sales is not None:
        if sales.duplicated(["Medicine_ID", "Date"]).any():
            add("warning", f"{int(sales.duplicated(['Medicine_ID', 'Date']).sum())} duplicate sales rows (will be removed)")
        if sales["Units_Sold"].isna().any():
            add("warning", f"{int(sales['Units_Sold'].isna().sum())} missing Units_Sold (will be interpolated)")
        if (sales["Units_Sold"] < 0).any():
            add("error", "Negative Units_Sold values")
    return issues


def validate_flat_inventory(df):
    """Validate an uploaded flat inventory CSV. Returns (errors, warnings)."""
    errors, warnings = [], []
    missing = [c for c in FLAT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required columns: {missing}"], warnings
    if df.empty:
        return ["File contains no rows"], warnings
    for col in ["Purchase_Date", "Manufacturing_Date", "Expiry_Date"]:
        if pd.to_datetime(df[col], errors="coerce").isna().any():
            errors.append(f"{col} has missing or unparseable dates")
    for col in ["Purchase_Price", "Selling_Price", "Current_Stock", "Daily_Sales",
                "Supplier_Lead_Time", "Minimum_Stock_Level", "Maximum_Stock_Level"]:
        num = pd.to_numeric(df[col], errors="coerce")
        if num.isna().any():
            errors.append(f"{col} has missing or non-numeric values")
        elif (num < 0).any():
            errors.append(f"{col} contains negative values")
    if df["Batch_Number"].duplicated().any():
        errors.append("Duplicate Batch_Number values")
    if not errors:
        mf = pd.to_datetime(df["Manufacturing_Date"], errors="coerce")
        ex = pd.to_datetime(df["Expiry_Date"], errors="coerce")
        if (mf >= ex).any():
            errors.append("Manufacturing_Date must be before Expiry_Date")
        if (pd.to_numeric(df["Selling_Price"]) < pd.to_numeric(df["Purchase_Price"])).any():
            warnings.append("Some rows have Selling_Price below Purchase_Price")
        if (pd.to_numeric(df["Supplier_Lead_Time"]) <= 0).any():
            errors.append("Supplier_Lead_Time must be positive")
    return errors, warnings


# ----------------------------------------------------------------- cleaning
def clean_medicines(medicines):
    m = medicines.copy()
    m["Manufacturer"] = m["Manufacturer"].fillna("Unknown")
    if "Data_Source" not in m:
        m["Data_Source"] = "SYNTHETIC"
    m = m.drop_duplicates("Medicine_ID", keep="first")
    return m[MEDICINE_COLUMNS].reset_index(drop=True)


def clean_batches(batches):
    b = batches.copy()
    if "Quantity_Received" not in b:
        b["Quantity_Received"] = b["Current_Stock"]
    for c in ["Purchase_Date", "Manufacturing_Date", "Expiry_Date"]:
        b[c] = pd.to_datetime(b[c])
    b = b.drop_duplicates("Batch_Number", keep="first")
    b["Current_Stock"] = b["Current_Stock"].clip(lower=0).astype(int)
    return b[BATCH_COLUMNS].reset_index(drop=True)


def detect_outliers(sales, k=10.0):
    """Boolean Series: True where Units_Sold exceeds median + k * robust_sd (per medicine)."""
    g = sales.groupby("Medicine_ID")["Units_Sold"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    robust_sd = np.maximum(1.4826 * mad, np.sqrt(np.maximum(med, 1.0)))
    return sales["Units_Sold"] > med + k * robust_sd


def clean_sales(sales):
    """Return (clean_sales, report_dict). Output has exactly one row per medicine per day."""
    s = sales[["Date", "Medicine_ID", "Units_Sold"]].copy()
    s["Date"] = pd.to_datetime(s["Date"])
    s["Units_Sold"] = pd.to_numeric(s["Units_Sold"], errors="coerce")
    report = {"rows_in": len(s)}

    dup = s.duplicated(["Medicine_ID", "Date"], keep="first")
    report["duplicates_removed"] = int(dup.sum())
    s = s[~dup]

    s.loc[s["Units_Sold"] < 0, "Units_Sold"] = np.nan
    full = pd.MultiIndex.from_product(
        [sorted(s["Medicine_ID"].unique()), pd.date_range(s["Date"].min(), s["Date"].max())],
        names=["Medicine_ID", "Date"])
    s = s.set_index(["Medicine_ID", "Date"]).reindex(full).reset_index()
    report["missing_filled"] = int(s["Units_Sold"].isna().sum())
    s["Units_Sold"] = (s.groupby("Medicine_ID")["Units_Sold"]
                       .transform(lambda x: x.interpolate(limit_direction="both")))

    out = detect_outliers(s)
    report["outliers_capped"] = int(out.sum())
    roll_med = (s.groupby("Medicine_ID")["Units_Sold"]
                .transform(lambda x: x.rolling(7, center=True, min_periods=1).median()))
    s.loc[out, "Units_Sold"] = roll_med[out]

    s["Units_Sold"] = s["Units_Sold"].round().clip(lower=0).astype(int)
    s = s.sort_values(["Medicine_ID", "Date"]).reset_index(drop=True)[["Date", "Medicine_ID", "Units_Sold"]]
    report["rows_out"] = len(s)
    return s, report


def outlier_summary(sales):
    """Per-medicine count of outliers (used in EDA / documentation)."""
    flag = detect_outliers(sales)
    return sales.assign(is_outlier=flag).groupby("Medicine_ID")["is_outlier"].sum().sort_values(ascending=False)


def split_flat_inventory(flat):
    """Split an uploaded flat file into (medicines, batches) tables."""
    f = flat.copy()
    for c in ["Purchase_Date", "Manufacturing_Date", "Expiry_Date"]:
        f[c] = pd.to_datetime(f[c])
    if "Data_Source" not in f:
        f["Data_Source"] = "UPLOADED"
    if "Batch_Quantity" in f:
        f["Quantity_Received"] = f["Batch_Quantity"]
    meds = clean_medicines(f.drop_duplicates("Medicine_ID"))
    batches = clean_batches(f)
    return meds, batches


def run_cleaning():
    """Load raw files, validate, clean and save to data/processed. Returns dict."""
    meds_raw, batches_raw, sales_raw = load_raw()
    issues = validate_tables(meds_raw, batches_raw, sales_raw)
    meds = clean_medicines(meds_raw)
    batches = clean_batches(batches_raw)
    sales, report = clean_sales(sales_raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meds.to_csv(config.PROCESSED_DIR / "medicines_clean.csv", index=False)
    batches.to_csv(config.PROCESSED_DIR / "batches_clean.csv", index=False)
    sales.to_csv(config.PROCESSED_DIR / "sales_daily_clean.csv", index=False)
    return {"medicines": meds, "batches": batches, "sales": sales, "issues": issues, "report": report}
