"""Page 6 of 6: Data Management (browse, edit, upload, download)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from src import database as db  # noqa: E402
from src import pipeline, preprocessing as pp, ui  # noqa: E402

ui.setup("Data Management", "🗂️")
d = ui.get_data()
tab1, tab2, tab3, tab4 = st.tabs(["Browse data", "Edit inventory (CRUD)", "Upload and score CSV", "Download reports"])

with tab1:
    tables = ["medicines", "batches", "sales_daily", "forecasts", "expiry_predictions", "recommendations", "fefo"]
    t = st.selectbox("Table", tables)
    df = d[t]
    q = st.text_input("Filter (text contained in any column)")
    if q:
        df = df[df.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)]
    st.caption(f"{len(df):,} rows")
    ui.table(df.head(2000))

with tab2:
    st.caption("Changes are written to the SQLite database. Press **Recompute predictions** afterwards to refresh "
               "risk classes and recommendations.")
    meds = d["medicines"].sort_values("Medicine_Name")
    st.subheader("Update or delete a batch")
    mname = st.selectbox("Medicine", meds["Medicine_Name"], key="crud_med")
    mid = meds.loc[meds["Medicine_Name"] == mname, "Medicine_ID"].iloc[0]
    bt = db.get_batches(mid)
    st.dataframe(bt, hide_index=True)
    if len(bt):
        bn = st.selectbox("Batch", bt["Batch_Number"])
        cur = int(bt.loc[bt["Batch_Number"] == bn, "Current_Stock"].iloc[0])
        new_stock = st.number_input("New current stock", min_value=0, value=cur, step=1)
        c1, c2 = st.columns(2)
        if c1.button("Update stock"):
            db.update_batch(bn, Current_Stock=int(new_stock))
            st.cache_data.clear()
            st.success(f"Updated {bn}: stock {cur} -> {int(new_stock)}")
        if c2.button("Delete batch"):
            db.delete_batch(bn)
            st.cache_data.clear()
            st.success(f"Deleted {bn}")
    st.subheader("Add a batch")
    with st.form("add_batch"):
        bno = st.text_input("Batch number", value=f"BN-{mid[3:]}-NEW")
        a, b, c = st.columns(3)
        mfg = a.date_input("Manufacturing date", d["as_of"] - pd.Timedelta(days=60))
        exp = b.date_input("Expiry date", d["as_of"] + pd.Timedelta(days=365))
        qty = c.number_input("Quantity received", min_value=1, value=500)
        if st.form_submit_button("Add batch"):
            try:
                if pd.Timestamp(mfg) >= pd.Timestamp(exp):
                    raise ValueError("Manufacturing date must be before expiry date")
                db.create_batch({"Batch_Number": bno, "Medicine_ID": mid, "Purchase_Date": d["as_of"], "Manufacturing_Date": mfg,
                                 "Expiry_Date": exp, "Quantity_Received": int(qty), "Current_Stock": int(qty)})
                st.cache_data.clear()
                st.success(f"Added {bno}")
            except Exception as e:  # duplicate key, bad dates, ...
                st.error(f"Could not add batch: {e}")
    st.subheader("Add a medicine")
    with st.form("add_med"):
        a, b = st.columns(2)
        new_id = a.text_input("Medicine ID", value="MED999")
        new_name = b.text_input("Medicine name")
        cat = a.selectbox("Category", sorted(meds["Category"].unique()))
        pp_, sp_ = b.number_input("Purchase price", min_value=0.01, value=10.0), b.number_input("Selling price", min_value=0.01, value=12.0)
        lead = a.number_input("Supplier lead time (days)", min_value=1, value=5)
        if st.form_submit_button("Add medicine"):
            try:
                if not new_name:
                    raise ValueError("Name is required")
                db.create_medicine({"Medicine_ID": new_id, "Medicine_Name": new_name, "Category": cat, "Manufacturer": "Unknown",
                                    "Purchase_Price": pp_, "Selling_Price": sp_, "Minimum_Stock_Level": 0,
                                    "Maximum_Stock_Level": 0, "Supplier_Lead_Time": int(lead), "Seasonal_Demand": "Non-seasonal",
                                    "Data_Source": "USER_ENTERED"})
                st.cache_data.clear()
                st.success(f"Added {new_id}. It has no sales history, so it has no forecast until the model is retrained.")
            except Exception as e:
                st.error(f"Could not add medicine: {e}")
    st.divider()
    if st.button("Recompute predictions and recommendations"):
        with st.spinner("Scoring..."):
            pipeline.refresh_recommendations()
            st.cache_data.clear()
        st.success("Recommendations refreshed. Open the other pages to see the effect.")

with tab3:
    st.markdown("Upload a flat inventory CSV (one row per batch). Required columns:")
    st.code(", ".join(pp.FLAT_REQUIRED_COLUMNS))
    sample = config.RAW_DIR / "pharmacy_inventory_flat.csv"
    if sample.exists():
        st.download_button("Download sample CSV (synthetic)", sample.read_bytes(), "sample_inventory.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up, keep_default_na=False, na_values=[""])
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()
        errors, warns = pp.validate_flat_inventory(df)
        for w in warns:
            st.warning(w)
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.success(f"Validation passed: {len(df)} rows, {df['Medicine_ID'].nunique()} medicines.")
            ui.table(df.head(100))
            if st.button("Score uploaded inventory"):
                scored, recs, fefo = pipeline.analyze_uploaded(df)
                st.subheader("Recommendations")
                ui.table(recs[["Medicine_Name", "Usable_Stock", "Reorder_Point", "Recommended_Order_Qty", "Overstock_Qty",
                               "Potential_Expiry_Loss", "Recommendation"]])
                st.subheader("Batch expiry risk")
                ui.table(scored[["Medicine_Name", "Batch_Number", "Days_Until_Expiry", "Current_Stock", "Expected_Unsold_Stock",
                                 "Risk_Class"]])
                st.download_button("Download recommendations", recs.to_csv(index=False).encode(), "uploaded_recommendations.csv", "text/csv")
                st.download_button("Download batch risk", scored.to_csv(index=False).encode(), "uploaded_batch_risk.csv", "text/csv")
                st.caption("Uploaded data has no sales history: demand = the Daily_Sales column (variability assumed 35%), unless the "
                           "medicine already exists in the database.")

with tab4:
    for label, key, fname in [("Inventory recommendations", "recommendations", "inventory_recommendations.csv"),
                              ("Batch expiry-risk predictions", "expiry_predictions", "expiry_predictions.csv"),
                              ("FEFO dispensing plan", "fefo", "fefo_recommendations.csv"),
                              ("Demand forecasts (7/15/30 days)", "forecasts", "demand_forecasts.csv")]:
        st.download_button(f"Download: {label}", d[key].to_csv(index=False).encode(), fname, "text/csv", key=fname)
ui.footer()
