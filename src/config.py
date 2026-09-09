"""
config.py
Single source of truth for paths, constants, and choices made about
this project. Nothing here should be hardcoded anywhere else.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PATHS
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "scaler.pkl"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"
BEST_PARAMS_PATH = PROJECT_ROOT / "artifacts" / "best_params.json"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"

# DATA
TARGET_COLUMN = 'Class'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# IMBALANCE HANDLING
BALANCE_METHOD = 'class_weight'  # OR SMOTE

# METRICS
PRIMARY_METRIC = 'average_precision'
DECISION_THRESHOLD = 0.2  # PLACEHOLDER

# CROSS-VALIDATION
CV = 5