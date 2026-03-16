"""
Feature Engineering Module for Salon No-Show Prediction
Generates all model features from a cleaned salon booking DataFrame.

Features created
----------------
Temporal:
  Lead_Time_Days          - Days between booking and appointment
  Appt_Hour               - Hour of appointment (0-23)
  Is_Edge_Hour            - 1 if hour <= 9 or hour >= 18 (high no-show risk times)

Customer behaviour:
  Customer_No_Show_Rate   - Past no-shows / past visits (0.0 for new customers)
  Cancellation_Ratio      - Past cancellations / past visits (0.0 for new customers)
  Successful_Visit_Rate   - Completed visits / past visits, clipped to [0, 1]
  Is_New_Customer         - 1 if customer has no prior visits

Encoded:
  One-hot columns for Service_Type, Branch, Payment_Method, Day_of_Week
  No_Show                 - Binary target (1 = no-show, 0 = attended)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

# Columns dropped before building X (identifiers, raw timestamps, superseded counts)
DROP_COLS = [
    'Booking_ID', 'Customer_ID',
    'Booking_Time', 'Appointment_Time',
    'Booking_Lead_Time_Days',
    'Outcome', 'No_Show',
    'Past_Visit_Count', 'Past_No_Show_Count', 'Past_Cancellation_Count',
]

CATEGORICAL_COLS = ['Service_Type', 'Branch', 'Payment_Method', 'Day_of_Week']


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def add_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lead time in whole days between booking and appointment.

    Args:
        df: DataFrame with parsed Booking_Time and Appointment_Time columns.

    Returns:
        DataFrame with new Lead_Time_Days column.
    """
    df = df.copy()
    df['Lead_Time_Days'] = (df['Appointment_Time'] - df['Booking_Time']).dt.days
    return df


def add_appointment_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the hour of the appointment (0-23).

    Args:
        df: DataFrame with parsed Appointment_Time column.

    Returns:
        DataFrame with new Appt_Hour column.
    """
    df = df.copy()
    df['Appt_Hour'] = df['Appointment_Time'].dt.hour
    return df


def add_edge_hour_flag(df: pd.DataFrame,
                       early_cutoff: int = 9,
                       late_cutoff: int = 18) -> pd.DataFrame:
    """
    Flag appointments at edge hours (early morning or late evening).

    Early morning (<= early_cutoff) and late-evening (>= late_cutoff) slots
    carry higher no-show risk — easy to forget or bumped by end-of-day plans.

    Args:
        df:            DataFrame with parsed Appointment_Time column.
        early_cutoff:  Hour at or before which an appointment is "early". Default 9.
        late_cutoff:   Hour at or after which an appointment is "late". Default 18.

    Returns:
        DataFrame with new Is_Edge_Hour column (0/1 int).
    """
    df = df.copy()
    hour = df['Appointment_Time'].dt.hour
    df['Is_Edge_Hour'] = ((hour <= early_cutoff) | (hour >= late_cutoff)).astype(int)
    return df


def add_no_show_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each customer's historical no-show rate.

    Safe for new customers (Past_Visit_Count == 0) — returns 0.0.

    Args:
        df: DataFrame with Past_No_Show_Count and Past_Visit_Count columns.

    Returns:
        DataFrame with new Customer_No_Show_Rate column.
    """
    df = df.copy()
    df['Customer_No_Show_Rate'] = np.where(
        df['Past_Visit_Count'] == 0,
        0.0,
        df['Past_No_Show_Count'] / df['Past_Visit_Count']
    ).round(4)
    return df


def add_cancellation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each customer's historical cancellation ratio.

    Safe for new customers (Past_Visit_Count == 0) — returns 0.0.

    Args:
        df: DataFrame with Past_Cancellation_Count and Past_Visit_Count columns.

    Returns:
        DataFrame with new Cancellation_Ratio column.
    """
    df = df.copy()
    df['Cancellation_Ratio'] = np.where(
        df['Past_Visit_Count'] == 0,
        0.0,
        df['Past_Cancellation_Count'] / df['Past_Visit_Count']
    ).round(4)
    return df


def add_new_customer_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag first-time customers who have no prior visit history.

    New customers have no track record, making them harder to predict and
    generally showing higher no-show rates than returning customers.

    Args:
        df: DataFrame with Past_Visit_Count column.

    Returns:
        DataFrame with new Is_New_Customer column (0/1 int).
    """
    df = df.copy()
    df['Is_New_Customer'] = (df['Past_Visit_Count'] == 0).astype(int)
    return df


def add_successful_visit_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the fraction of past visits that were completed successfully.

    Successful = not a no-show and not a cancellation.
    Safe for new customers (Past_Visit_Count == 0) — returns 0.0.
    Result is clipped to [0, 1] to guard against data inconsistencies.

    Args:
        df: DataFrame with Past_Visit_Count, Past_No_Show_Count,
            and Past_Cancellation_Count columns.

    Returns:
        DataFrame with new Successful_Visit_Rate column.
    """
    df = df.copy()
    good_visits = (
        df['Past_Visit_Count']
        - df['Past_No_Show_Count']
        - df['Past_Cancellation_Count']
    )
    df['Successful_Visit_Rate'] = np.where(
        df['Past_Visit_Count'] == 0,
        0.0,
        good_visits / df['Past_Visit_Count']
    ).round(4)
    df['Successful_Visit_Rate'] = df['Successful_Visit_Rate'].clip(0, 1)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable: 1 = No-Show, 0 = attended.

    Args:
        df: DataFrame with Outcome column.

    Returns:
        DataFrame with new No_Show column (0/1 int).
    """
    df = df.copy()
    df['No_Show'] = (df['Outcome'] == 'No-Show').astype(int)
    return df


def encode_categoricals(df: pd.DataFrame,
                        cols: List[str] = None) -> pd.DataFrame:
    """
    One-hot encode categorical columns, dropping the first level to avoid
    multicollinearity.

    Args:
        df:   DataFrame containing the categorical columns.
        cols: List of column names to encode. Defaults to CATEGORICAL_COLS.

    Returns:
        DataFrame with original categorical columns replaced by OHE columns.
    """
    if cols is None:
        cols = CATEGORICAL_COLS
    present = [c for c in cols if c in df.columns]
    return pd.get_dummies(df, columns=present, drop_first=True, dtype=int)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a cleaned DataFrame.

    Expects the output of SalonDataProcessor.process() (datetime columns
    already parsed, no missing values).

    Steps
    -----
    1. Temporal features       (Lead_Time_Days, Appt_Hour, Is_Edge_Hour)
    2. Customer behaviour      (No_Show_Rate, Cancellation_Ratio,
                                Is_New_Customer, Successful_Visit_Rate)
    3. Target variable         (No_Show)
    4. One-hot encode          (Service_Type, Branch, Payment_Method, Day_of_Week)

    Args:
        df: Cleaned DataFrame from the data-processing pipeline.

    Returns:
        Feature-engineered DataFrame (full, including ID/datetime/target cols).
    """
    df = add_lead_time(df)
    df = add_appointment_hour(df)
    df = add_edge_hour_flag(df)
    df = add_no_show_rate(df)
    df = add_cancellation_ratio(df)
    df = add_new_customer_flag(df)
    df = add_successful_visit_rate(df)
    df = add_target(df)
    df = encode_categoricals(df)
    return df


def get_X_y(df_featured: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the feature-engineered DataFrame into feature matrix X and target y.

    Drops identifier columns, raw timestamps, the Outcome string column,
    and raw count columns that have been superseded by ratio features.

    Args:
        df_featured: Output of build_features().

    Returns:
        X: Feature matrix ready for model training.
        y: Binary target series (No_Show).
    """
    y = df_featured['No_Show']
    drop = [c for c in DROP_COLS if c in df_featured.columns]
    X = df_featured.drop(columns=drop)
    return X, y


def run_feature_engineering(df: pd.DataFrame,
                            save_dir: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full feature engineering pipeline: build features, split X/y, optionally save.

    Args:
        df:       Cleaned DataFrame from the data-processing pipeline.
        save_dir: If provided, saves X.csv, y.csv, and
                  salon_bookings_featured.csv to this directory.

    Returns:
        X: Feature matrix.
        y: Target series.
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60 + "\n")

    df_feat = build_features(df)
    print(f"[OK] Features built: {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns")

    X, y = get_X_y(df_feat)
    print(f"[OK] Feature matrix X : {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"[OK] Target vector  y : {y.shape[0]:,} rows  |  No-show rate: {y.mean():.2%}")

    engineered = ['Lead_Time_Days', 'Appt_Hour', 'Customer_No_Show_Rate',
                  'Cancellation_Ratio', 'Is_New_Customer',
                  'Successful_Visit_Rate', 'Is_Edge_Hour']
    print(f"\nEngineered features: {engineered}")
    print(f"All features ({len(X.columns)}): {X.columns.tolist()}")

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(out / 'salon_bookings_featured.csv', index=False)
        X.to_csv(out / 'X.csv', index=False)
        y.to_csv(out / 'y.csv', index=False, header=True)
        print(f"\n[OK] Saved to {out}")

    print("\n" + "="*60 + "\n")
    return X, y


if __name__ == "__main__":
    from pathlib import Path
    from data_processing import load_and_clean_data

    script_dir = Path(__file__).parent.parent
    data_path  = script_dir / "data" / "raw" / "salon_bookings.csv"
    save_dir   = script_dir / "data" / "processed"

    df_clean = load_and_clean_data(str(data_path))
    X, y = run_feature_engineering(df_clean, save_dir=str(save_dir))

    print("Feature matrix sample:")
    print(X.head(3).to_string())
