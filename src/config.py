"""
config.py
Single source of truth for paths, constants, and choices made about
this project. Nothing here should be hardcoded anywhere else.
"""

# PATHS
RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed/"
SCALER_PATH = "artifacts/scaler.pkl"
MODEL_PATH = "artifacts/model.pkl"
BEST_PARAMS_PATH = "artifacts/best_params.json"
METRICS_PATH = "results/metrics.json"

#DATA
TARGET_COLUMN = 'Class'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# IMBALANCE HANDLING
BALANCE_METHOD = 'class_weight' # OR SMOTE

# METRICS
PRIMARY_METRIC = 'average_precision'
DECISION_THRESHOLD = 0.2 # PLACE HOLDER

# CROSS-VALIDATION
CV = 5


