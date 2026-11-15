"""Synthetic pharmacy data generator.

EVERYTHING produced here is SYNTHETIC. Medicine names are common generic names
used only for realism; manufacturers are fictional; quantities, prices and
sales are random. No patient or real pharmacy data is involved.

Outputs (data/raw/):
  medicines.csv                 one row per medicine
  batches.csv                   one row per batch currently in stock (as-of date)
  sales_daily.csv               one row per medicine per day (>= 43,000 rows)
  pharmacy_inventory_flat.csv   flat file with all requested columns (upload format)

The raw sales file deliberately contains a few data-quality faults
(duplicates, missing values, extreme outliers) so that the cleaning step in
src/preprocessing.py has something real to fix.

Run:  python data/generate_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# (name, category, seasonal pattern)
CATALOG = [
    ("Paracetamol 500mg", "Pain & Fever", "Monsoon"), ("Ibuprofen 400mg", "Pain & Fever", "Non-seasonal"),
    ("Diclofenac 50mg", "Pain & Fever", "Non-seasonal"), ("Aceclofenac 100mg", "Pain & Fever", "Non-seasonal"),
    ("Naproxen 250mg", "Pain & Fever", "Non-seasonal"), ("Mefenamic Acid 500mg", "Pain & Fever", "Non-seasonal"),
    ("Amoxicillin 500mg", "Antibiotics", "Winter"), ("Azithromycin 500mg", "Antibiotics", "Winter"),
    ("Ciprofloxacin 500mg", "Antibiotics", "Monsoon"), ("Cefixime 200mg", "Antibiotics", "Monsoon"),
    ("Doxycycline 100mg", "Antibiotics", "Non-seasonal"), ("Metronidazole 400mg", "Antibiotics", "Monsoon"),
    ("Cefuroxime 250mg", "Antibiotics", "Winter"), ("Cefpodoxime 200mg", "Antibiotics", "Non-seasonal"),
    ("Metformin 500mg", "Diabetes", "Non-seasonal"), ("Glimepiride 2mg", "Diabetes", "Non-seasonal"),
    ("Sitagliptin 100mg", "Diabetes", "Non-seasonal"), ("Gliclazide 80mg", "Diabetes", "Non-seasonal"),
    ("Vildagliptin 50mg", "Diabetes", "Non-seasonal"),
    ("Amlodipine 5mg", "Cardiac", "Non-seasonal"), ("Atorvastatin 10mg", "Cardiac", "Non-seasonal"),
    ("Losartan 50mg", "Cardiac", "Non-seasonal"), ("Telmisartan 40mg", "Cardiac", "Non-seasonal"),
    ("Metoprolol 50mg", "Cardiac", "Non-seasonal"), ("Rosuvastatin 10mg", "Cardiac", "Non-seasonal"),
    ("Clopidogrel 75mg", "Cardiac", "Non-seasonal"), ("Furosemide 40mg", "Cardiac", "Non-seasonal"),
    ("Pantoprazole 40mg", "Gastro", "Non-seasonal"), ("Omeprazole 20mg", "Gastro", "Non-seasonal"),
    ("Rabeprazole 20mg", "Gastro", "Non-seasonal"), ("Domperidone 10mg", "Gastro", "Monsoon"),
    ("Ondansetron 4mg", "Gastro", "Monsoon"), ("ORS Sachet", "Gastro", "Summer"),
    ("Lactulose Syrup", "Gastro", "Non-seasonal"),
    ("Cetirizine 10mg", "Respiratory & Allergy", "Winter"), ("Levocetirizine 5mg", "Respiratory & Allergy", "Winter"),
    ("Montelukast 10mg", "Respiratory & Allergy", "Winter"), ("Salbutamol Inhaler", "Respiratory & Allergy", "Winter"),
    ("Dextromethorphan Cough Syrup", "Respiratory & Allergy", "Winter"),
    ("Loratadine 10mg", "Respiratory & Allergy", "Winter"), ("Fexofenadine 120mg", "Respiratory & Allergy", "Winter"),
    ("Vitamin C 500mg", "Vitamins & Supplements", "Winter"), ("Vitamin D3 60000IU", "Vitamins & Supplements", "Non-seasonal"),
    ("Vitamin B12 1500mcg", "Vitamins & Supplements", "Non-seasonal"), ("Calcium + D3", "Vitamins & Supplements", "Non-seasonal"),
    ("Iron + Folic Acid", "Vitamins & Supplements", "Non-seasonal"), ("Multivitamin", "Vitamins & Supplements", "Non-seasonal"),
    ("Clotrimazole Cream", "Dermatology", "Monsoon"), ("Betamethasone Cream", "Dermatology", "Non-seasonal"),
    ("Mupirocin Ointment", "Dermatology", "Non-seasonal"), ("Calamine Lotion", "Dermatology", "Summer"),
    ("Ketoconazole Shampoo", "Dermatology", "Non-seasonal"), ("Adapalene Gel", "Dermatology", "Non-seasonal"),
    ("Levothyroxine 25mcg", "Thyroid", "Non-seasonal"), ("Levothyroxine 50mcg", "Thyroid", "Non-seasonal"),
    ("Levothyroxine 100mcg", "Thyroid", "Non-seasonal"),
    ("Povidone Iodine 5%", "Antiseptic & First Aid", "Non-seasonal"), ("Chlorhexidine Mouthwash", "Antiseptic & First Aid", "Non-seasonal"),
    ("Lubricant Eye Drops", "Antiseptic & First Aid", "Summer"), ("Zinc Sulphate 20mg", "Antiseptic & First Aid", "Monsoon"),
]

MANUFACTURERS = ["Medora Pharma", "Healwell Labs", "Zenith Remedies", "CureNova Pharma",
                 "Vitalis Pharmaceuticals", "BioCare Formulations", "Sanora Lifesciences",
                 "Aarogya Generics"]  # fictional

# category -> (typical units/day, overdispersion shape (higher = steadier), (min, max) cost per unit INR)
CATEGORY_PARAMS = {
    "Pain & Fever": (22, 8, (2, 15)), "Antibiotics": (10, 6, (8, 60)),
    "Diabetes": (14, 25, (3, 30)), "Cardiac": (13, 25, (3, 40)),
    "Gastro": (16, 8, (2, 30)), "Respiratory & Allergy": (11, 6, (3, 120)),
    "Vitamins & Supplements": (9, 10, (3, 80)), "Dermatology": (5, 6, (40, 200)),
    "Thyroid": (10, 30, (2, 6)), "Antiseptic & First Aid": (4, 8, (30, 150)),
}
SEASON_PEAK_DOY = {"Winter": 15, "Summer": 140, "Monsoon": 225}
DOW_FACTOR = np.array([1.10, 1.05, 1.00, 1.00, 1.05, 0.95, 0.80])  # Mon..Sun


def generate_medicines(rng, n):
    rows = []
    for i, (name, cat, season) in enumerate(CATALOG[:n], start=1):
        base_rate, _, (p_lo, p_hi) = CATEGORY_PARAMS[cat]
        purchase = round(float(rng.uniform(p_lo, p_hi)), 2)
        rows.append({
            "Medicine_ID": f"MED{i:03d}", "Medicine_Name": name, "Category": cat,
            "Manufacturer": rng.choice(MANUFACTURERS),
            "Purchase_Price": purchase,
            "Selling_Price": round(purchase * float(rng.uniform(1.15, 1.45)), 2),
            "Supplier_Lead_Time": int(rng.integers(2, 11)),
            "Seasonal_Demand": season,
            "Base_Rate": float(np.clip(rng.lognormal(np.log(base_rate), 0.45), 1.5, 120)),
        })
    return pd.DataFrame(rows)


def generate_sales(rng, medicines, start, days):
    dates = pd.date_range(start, periods=days, freq="D")
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values
    t = np.arange(days)
    frames = []
    for _, m in medicines.iterrows():
        shape = CATEGORY_PARAMS[m.Category][1]
        season = m.Seasonal_Demand
        amp = rng.uniform(0.35, 0.60) if season != "Non-seasonal" else 0.03
        peak = SEASON_PEAK_DOY.get(season, 0)
        seas = 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
        trend = 1 + rng.uniform(-0.15, 0.25) * t / 365.0
        weekly = DOW_FACTOR[dow] if shape < 20 else 1 + (DOW_FACTOR[dow] - 1) * 0.3
        spike = np.where(rng.random(days) < 0.01, rng.uniform(1.8, 2.8, days), 1.0)
        lam = m.Base_Rate * seas * trend * weekly * spike
        lam = lam * rng.gamma(shape, 1.0 / shape, days)       # overdispersion
        units = rng.poisson(lam)
        frames.append(pd.DataFrame({"Date": dates, "Medicine_ID": m.Medicine_ID, "Units_Sold": units}))
    return pd.concat(frames, ignore_index=True)


def finalize_medicines(medicines, sales):
    """Stock limits are derived from realised demand so they are consistent."""
    mean_daily = sales.groupby("Medicine_ID")["Units_Sold"].mean()
    m = medicines.copy()
    md = m["Medicine_ID"].map(mean_daily)
    m["Minimum_Stock_Level"] = np.ceil(md * m["Supplier_Lead_Time"]).astype(int)
    m["Maximum_Stock_Level"] = np.ceil(md * (m["Supplier_Lead_Time"] + config.REVIEW_PERIOD_DAYS + 30)).astype(int)
    m["Data_Source"] = "SYNTHETIC"
    return m.drop(columns="Base_Rate")


def generate_batches(rng, medicines, sales, as_of):
    recent = sales[sales.Date > as_of - pd.Timedelta(days=28)].groupby("Medicine_ID")["Units_Sold"].mean()
    rows = []
    for _, m in medicines.iterrows():
        rate = max(float(recent.get(m.Medicine_ID, 1.0)), 0.5)
        n_b = int(rng.integers(2, 5))
        # ~25% of medicines are over-bought (slow movers): realistic source of expiry risk
        cover_days = rng.uniform(100, 260) if rng.random() < 0.25 else rng.uniform(3, 80)
        total_stock = rate * cover_days
        shares = rng.dirichlet(np.ones(n_b))
        expiries = np.sort(rng.integers(15, 420, n_b)).astype(int)
        if rng.random() < 0.05:                                # a few already-expired batches
            expiries[0] = -int(rng.integers(1, 60))
        for j in range(n_b):
            expiry = as_of + pd.Timedelta(days=int(expiries[j]))
            shelf = int(rng.choice([730, 1095]))
            mfg = expiry - pd.Timedelta(days=shelf)
            purchase = min(mfg + pd.Timedelta(days=int(rng.integers(7, 46))), as_of - pd.Timedelta(days=1))
            stock = max(int(round(total_stock * shares[j])), 1)
            rows.append({
                "Batch_Number": f"BN-{m.Medicine_ID[3:]}-{expiry:%y%m}-{j + 1}",
                "Medicine_ID": m.Medicine_ID,
                "Purchase_Date": purchase, "Manufacturing_Date": mfg, "Expiry_Date": expiry,
                "Quantity_Received": int(np.ceil(stock / rng.uniform(0.45, 1.0))),
                "Current_Stock": stock,
            })
    return pd.DataFrame(rows)


def build_flat_inventory(medicines, batches, sales, as_of):
    """Flat file with all columns requested in the project brief (one row per batch)."""
    last30 = sales[sales.Date > as_of - pd.Timedelta(days=30)].groupby("Medicine_ID")["Units_Sold"].sum()
    last365 = sales[sales.Date > as_of - pd.Timedelta(days=365)].groupby("Medicine_ID")["Units_Sold"].sum()
    flat = batches.merge(medicines, on="Medicine_ID", how="left")
    flat["Monthly_Sales"] = flat["Medicine_ID"].map(last30).fillna(0).astype(int)
    flat["Daily_Sales"] = (flat["Monthly_Sales"] / 30).round(2)
    flat["Historical_Sales"] = flat["Medicine_ID"].map(last365).fillna(0).astype(int)  # units, last 365 days
    flat["Batch_Quantity"] = flat["Quantity_Received"]
    cols = ["Medicine_ID", "Medicine_Name", "Category", "Manufacturer", "Purchase_Price", "Selling_Price",
            "Current_Stock", "Minimum_Stock_Level", "Maximum_Stock_Level", "Daily_Sales", "Monthly_Sales",
            "Supplier_Lead_Time", "Purchase_Date", "Manufacturing_Date", "Expiry_Date", "Batch_Number",
            "Seasonal_Demand", "Historical_Sales", "Batch_Quantity", "Data_Source"]
    return flat[cols]


def inject_faults(rng, sales, medicines):
    """Add realistic data-entry faults to the RAW files (cleaned later)."""
    s = sales.copy()
    s["Units_Sold"] = s["Units_Sold"].astype(float)
    n = len(s)
    s.loc[rng.choice(n, 60, replace=False), "Units_Sold"] = np.nan                  # missing
    idx = rng.choice(n, 15, replace=False)
    s.loc[idx, "Units_Sold"] = (s.loc[idx, "Units_Sold"].fillna(5) + 3) * rng.integers(12, 21, 15)  # outliers
    s = pd.concat([s, s.sample(40, random_state=int(rng.integers(1e6)))], ignore_index=True)       # duplicates
    s = s.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)
    m = medicines.copy()
    m.loc[m.sample(2, random_state=1).index, "Manufacturer"] = np.nan                # missing text
    return s, m


def generate_all(n_medicines=None, history_days=None, seed=None, start=None):
    """Return dict of CLEAN generated tables plus the as-of date."""
    n_medicines = n_medicines or config.N_MEDICINES
    history_days = history_days or config.HISTORY_DAYS
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    start = start or config.HISTORY_START
    meds = generate_medicines(rng, n_medicines)
    sales = generate_sales(rng, meds, start, history_days)
    meds = finalize_medicines(meds, sales)
    as_of = sales["Date"].max()
    batches = generate_batches(rng, meds, sales, as_of)
    flat = build_flat_inventory(meds, batches, sales, as_of)
    return {"medicines": meds, "batches": batches, "sales": sales, "flat": flat, "as_of": as_of, "rng": rng}


def write_raw_files(n_medicines=None, history_days=None, seed=None, faults=True):
    out = generate_all(n_medicines, history_days, seed)
    rng = out.pop("rng")
    raw_sales, raw_meds = (inject_faults(rng, out["sales"], out["medicines"])
                           if faults else (out["sales"], out["medicines"]))
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_meds.to_csv(config.RAW_DIR / "medicines.csv", index=False)
    out["batches"].to_csv(config.RAW_DIR / "batches.csv", index=False)
    raw_sales.to_csv(config.RAW_DIR / "sales_daily.csv", index=False)
    out["flat"].to_csv(config.RAW_DIR / "pharmacy_inventory_flat.csv", index=False)
    (config.RAW_DIR / "DATA_NOTICE.txt").write_text(
        config.DATA_DISCLAIMER + "\nMedicine names are generic names used only for realism; manufacturers are "
        "fictional; all quantities, prices, dates and sales are randomly generated.\n"
        f"As-of date of the inventory snapshot: {out['as_of']:%Y-%m-%d}\n", encoding="utf-8")
    return out, raw_sales


if __name__ == "__main__":
    out, raw_sales = write_raw_files()
    print(config.DATA_DISCLAIMER)
    print(f"medicines: {len(out['medicines'])} | batches: {len(out['batches'])} | "
          f"daily sales rows (raw, incl. faults): {len(raw_sales):,} | as-of: {out['as_of']:%Y-%m-%d}")
    print(f"Files written to {config.RAW_DIR}")
