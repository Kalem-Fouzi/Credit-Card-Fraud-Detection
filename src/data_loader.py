import pandas as pd
from src import config
from sklearn.model_selection import train_test_split



def load_raw_data() -> pd.DataFrame:
    """LOADS The raw CSV into a DataFrame, no transformations applied yet """
    df = pd.read_csv(config.RAW_DATA_PATH)
    return df

def X_y(df):
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return X, y

def split_data(df: pd.DataFrame):
    """
    Split into train/test, stratified on the target column
    to preserve the fraud/non-fraud ratio in both sets.
    """
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

def load_and_split():
    """
    LOADING RAW DATA AND SPLITING IT in one CALL
    """
    df = load_raw_data()
    return split_data(df)

def save_processed_data(X_train, X_test, y_train, y_test):
    """Save the scaled/processed splits so they don't need recomputing."""
    X_train.to_parquet(config.PROCESSED_DIR /"X_train.parquet")
    X_test.to_parquet(config.PROCESSED_DIR /"X_test.parquet")
    y_train.to_frame().to_parquet(config.PROCESSED_DIR / "y_train.parquet")
    y_test.to_frame().to_parquet(config.PROCESSED_DIR / "y_test.parquet")


def load_processed_data():
    """Load already-processed train/test splits from disk."""
    X_train = pd.read_parquet(f"{config.PROCESSED_DIR}/X_train.parquet")
    X_test = pd.read_parquet(f"{config.PROCESSED_DIR}/X_test.parquet")
    y_train = pd.read_parquet(f"{config.PROCESSED_DIR}/y_train.parquet").squeeze()
    y_test = pd.read_parquet(f"{config.PROCESSED_DIR}/y_test.parquet").squeeze()
    return X_train, X_test, y_train, y_test
