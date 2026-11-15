"""Central configuration: paths, random seed and business assumptions.

All paths are read as `config.X` at call time (not copied at import time),
so tests can redirect them to temporary folders.
"""
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = BASE_DIR / "pharmacy.db"

# ---------- Reproducibility ----------
RANDOM_SEED = 42

# ---------- Synthetic data settings ----------
N_MEDICINES = 60
HISTORY_START = "2024-01-01"
HISTORY_DAYS = 730            # 2 years of daily sales per medicine
SNAPSHOTS_PER_MEDICINE = 40   # historical inventory snapshots for expiry-risk training
DATA_DISCLAIMER = ("SYNTHETIC DATA - generated for demonstration only. "
                   "No real patient or pharmacy data is used.")
CURRENCY = "INR"

# ---------- Forecasting ----------
FORECAST_HORIZONS = [7, 15, 30]
LAG_DAYS = [0, 1, 6, 13, 27]  # lag_k = units sold k days before the forecast origin
ROLL_WINDOWS = [7, 14, 28]
TEST_DAYS = 90                # last 90 days of origins = chronological test set
VAL_DAYS = 60                 # 60 days before the test set = validation (model selection)

# ---------- Inventory assumptions (documented in README) ----------
SERVICE_LEVEL_Z = 1.65        # ~95% cycle service level
NEAR_EXPIRY_DAYS = 90         # batches expiring within 90 days are "near expiry"
REVIEW_PERIOD_DAYS = 30       # target stock covers lead time + review period
ASSUMED_RECOVERY_RATE = 0.40  # share of at-risk value assumed recoverable (discount/return). ASSUMPTION.

# ---------- Expiry-risk labelling ----------
# Label is defined from the ACTUAL unsold share of a batch at its expiry date:
#   unsold_ratio = (stock - units actually sold before expiry) / stock
RISK_LABELS = ["Low", "Medium", "High"]
RISK_LOW_MAX = 0.15           # unsold_ratio <  0.15  -> Low
RISK_HIGH_MIN = 0.50          # unsold_ratio >= 0.50  -> High, otherwise Medium
