"""Run the full pipeline: generate data -> clean -> EDA -> train models -> optimise -> SQLite.

    python train.py            # full run (a few minutes)
    python train.py --quick    # fast smoke-test run
    python train.py --keep-data  # re-use existing data/raw files instead of regenerating
"""
import argparse

from src.pipeline import run_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller models / fewer snapshots (smoke test)")
    ap.add_argument("--keep-data", action="store_true", help="do not regenerate raw synthetic data")
    a = ap.parse_args()
    run_pipeline(quick=a.quick, regenerate=not a.keep_data)
