"""End-to-end test on a small dataset in a temporary folder."""
import pandas as pd
import pytest

import config


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    for name, sub in [("RAW_DIR", "raw"), ("PROCESSED_DIR", "processed"), ("MODELS_DIR", "models"), ("REPORTS_DIR", "reports")]:
        monkeypatch.setattr(config, name, tmp_path / sub)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    return tmp_path


def test_full_pipeline_small(tmp_project):
    from src import database as db
    from src import pipeline
    pipeline.run_pipeline(quick=True, verbose=False, n_medicines=12, history_days=420)
    assert (config.MODELS_DIR / "demand_model.pkl").exists()
    assert (config.MODELS_DIR / "expiry_model.pkl").exists()
    assert (config.REPORTS_DIR / "demand_model_comparison.csv").exists()
    assert len(list((config.REPORTS_DIR / "figures").glob("*.png"))) == 9
    recs = db.read_table("recommendations")
    assert len(recs) == 12
    assert set(recs["Recommendation"]) <= {"Reorder Now", "High Expiry Risk", "Overstock Warning", "Near Expiry", "Stock Sufficient"}
    assert (recs["Recommended_Order_Qty"] >= 0).all()
    # CRUD edit -> refresh recomputes derived tables
    b = db.get_batches().iloc[0]
    db.update_batch(b["Batch_Number"], Current_Stock=1)
    scored, recs2, fefo = pipeline.refresh_recommendations()
    assert scored.set_index("Batch_Number").loc[b["Batch_Number"], "Current_Stock"] == 1
    # uploaded-file scoring path
    flat = pd.read_csv(config.RAW_DIR / "pharmacy_inventory_flat.csv")
    scored_u, recs_u, fefo_u = pipeline.analyze_uploaded(flat)
    assert len(recs_u) == 12 and len(scored_u) == len(flat)
