import os
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Config ---
INPUT_PATH = os.path.join("outputs", "cleaned_churn_dataset_for_modeling.csv")
OUTPUT_PATH = os.path.join("outputs", "prepared_churn_dataset.csv")
LOG_PATH = os.path.join("logs", "phase2_data_preparation.log")
RANDOM_STATE = 42

# --- Setup Logging ---
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler()
    ]
)

# --- Function: Load Data ---
def load_dataset(filepath):
    try:
        logging.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath)
        logging.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

# --- Function: Clean & Transform ---
def preprocess_data(df):
    logging.info("Starting data preprocessing...")

    # Drop identifier columns
    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)
        logging.info("Dropped 'CustomerID' column.")

    # Convert date columns if any
    if "LastLoginDate" in df.columns:
        try:
            df["LastLoginDate"] = pd.to_datetime(df["LastLoginDate"], errors='coerce')
            df["DaysSinceLastLogin"] = (pd.Timestamp.now() - df["LastLoginDate"]).dt.days
            df.drop("LastLoginDate", axis=1, inplace=True)
            logging.info("Processed 'LastLoginDate' into numeric days.")
        except Exception as e:
            logging.warning(f"Failed to parse LastLoginDate: {e}")

    # Separate target
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # Detect categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    logging.info(f"Categorical columns: {categorical_cols}")
    logging.info(f"Numeric columns: {numeric_cols}")

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Create column transformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    feature_names = (
        numeric_cols +
        list(preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols))
    )

    X_final = pd.DataFrame(X_processed, columns=feature_names)
    df_final = pd.concat([X_final, y.reset_index(drop=True)], axis=1)

    logging.info(f"Preprocessed dataset shape: {df_final.shape}")
    return df_final

# --- Function: Save Processed Data ---
def save_dataset(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Saved processed dataset to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save processed data: {e}")
        raise

# --- Main ---
def main():
    try:
        df = load_dataset(INPUT_PATH)
        df_prepared = preprocess_data(df)
        save_dataset(df_prepared, OUTPUT_PATH)
        logging.info("Data preparation completed successfully.")
    except Exception as e:
        logging.fatal(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
