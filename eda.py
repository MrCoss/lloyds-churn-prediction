import os
import logging
import pandas as pd
import numpy as np

# --- Configuration ---
DATA_FILE = os.path.join("datasets", "Customer_Churn_Data_Large.xlsx")
LOG_FILE = os.path.join("logs", "phase1_data_preparation.log")
OUTPUT_DIR = "outputs"
FINAL_DATASET_PATH = os.path.join(OUTPUT_DIR, "cleaned_churn_dataset_for_modeling.csv")

# --- Setup Environment ---
def setup_environment():
    for folder in ["logs", OUTPUT_DIR, "datasets"]:
        os.makedirs(folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()]
    )
    logging.info("Environment setup complete.")

# --- Load Excel File with All Sheets ---
def load_all_sheets(file_path):
    try:
        logging.info(f"Reading dataset: {file_path}")
        xl = pd.read_excel(file_path, sheet_name=None)
        expected_sheets = ["Customer_Demographics", "Transaction_History", "Customer_Service", "Online_Activity", "Churn_Status"]
        for sheet in expected_sheets:
            if sheet not in xl:
                raise ValueError(f"Missing expected sheet: {sheet}")
        return xl
    except Exception as e:
        logging.exception(f"Error loading Excel file: {e}")
        raise

# --- Feature Engineering ---
def engineer_features(sheets):
    try:
        demo_df = sheets["Customer_Demographics"]
        trans_df = sheets["Transaction_History"]
        service_df = sheets["Customer_Service"]
        online_df = sheets["Online_Activity"]
        churn_df = sheets["Churn_Status"]

        # Parse dates
        trans_df["TransactionDate"] = pd.to_datetime(trans_df["TransactionDate"], errors="coerce")
        service_df["InteractionDate"] = pd.to_datetime(service_df["InteractionDate"], errors="coerce")
        online_df["LastLoginDate"] = pd.to_datetime(online_df["LastLoginDate"], errors="coerce")

        # Snapshot date: one day after the latest date in all sheets
        snapshot_date = max(
            trans_df["TransactionDate"].max(),
            service_df["InteractionDate"].max(),
            online_df["LastLoginDate"].max()
        ) + pd.Timedelta(days=1)

        # Transaction Aggregation
        trans_agg = trans_df.groupby("CustomerID").agg(
            TotalTransactions=("TransactionID", "count"),
            TotalAmountSpent=("AmountSpent", "sum"),
            AvgTransaction=("AmountSpent", "mean"),
            SpendingVolatility=("AmountSpent", "std"),
            DaysSinceLastTransaction=("TransactionDate", lambda x: (snapshot_date - x.max()).days),
            CustomerTenure=("TransactionDate", lambda x: (snapshot_date - x.min()).days)
        ).reset_index()

        # Customer Service Aggregation
        service_agg = service_df.groupby("CustomerID").agg(
            TotalInteractions=("InteractionID", "count"),
            Complaints=("InteractionType", lambda x: (x == "Complaint").sum())
        ).reset_index()
        service_agg["ComplaintRatio"] = (service_agg["Complaints"] / service_agg["TotalInteractions"]).fillna(0)

        # Merge all
        df = demo_df \
            .merge(trans_agg, on="CustomerID", how="left") \
            .merge(service_agg, on="CustomerID", how="left") \
            .merge(online_df, on="CustomerID", how="left") \
            .merge(churn_df.rename(columns={"ChurnStatus": "Churn"}), on="CustomerID", how="left")

        # New Features
        df["DaysSinceLastLogin"] = (snapshot_date - df["LastLoginDate"]).dt.days
        df["TransactionsPerMonth"] = (df["TotalTransactions"] * 30.4) / (df["CustomerTenure"] + 1e-6)
        df["LoginsPerMonth"] = (df["LoginFrequency"] * 30.4) / (df["CustomerTenure"] + 1e-6)
        df["RelativeVolatility"] = df["SpendingVolatility"] / (df["AvgTransaction"] + 1e-6)
        df["HasComplained"] = (df["Complaints"] > 0).astype(int)
        df["HighSpender"] = (df["TotalAmountSpent"] > df["TotalAmountSpent"].quantile(0.75)).astype(int)

        # Age group
        df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 30, 50, 70], labels=["Young Adult", "Adult", "Senior"])

        # Fill NaNs
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        df["AgeGroup"] = df["AgeGroup"].astype(str).fillna("Missing")

        logging.info("Feature engineering complete.")
        return df

    except Exception as e:
        logging.exception(f"Error during feature engineering: {e}")
        raise

# --- Save Final Dataset ---
def save_final_dataset(df):
    try:
        df.to_csv(FINAL_DATASET_PATH, index=False)
        logging.info(f"Final cleaned dataset saved to {FINAL_DATASET_PATH}")
    except Exception as e:
        logging.exception(f"Error saving final dataset: {e}")
        raise

# --- Main Execution ---
def main():
    setup_environment()
    try:
        sheets = load_all_sheets(DATA_FILE)
        df = engineer_features(sheets)
        save_final_dataset(df)
        logging.info("--- Phase 1 completed successfully. Dataset ready for modeling. ---")
    except Exception as e:
        logging.fatal("Phase 1 failed.", exc_info=True)

if __name__ == "__main__":
    main()
