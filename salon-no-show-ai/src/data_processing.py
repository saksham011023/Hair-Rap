"""
Data Processing Module for Salon Booking Dataset
Loads, cleans, and prepares the salon booking dataset for feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class SalonDataProcessor:
    """Process and clean the salon booking dataset."""

    DATETIME_COLUMNS = ['Booking_Time', 'Appointment_Time']
    CATEGORICAL_COLUMNS = ['Service_Type', 'Branch', 'Payment_Method',
                          'Day_of_Week', 'Outcome']
    NUMERIC_COLUMNS = ['Booking_Lead_Time_Days', 'Past_Visit_Count',
                      'Past_Cancellation_Count', 'Past_No_Show_Count']
    ID_COLUMNS = ['Booking_ID', 'Customer_ID']

    def __init__(self, data_path: str):
        """
        Initialize the data processor.

        Args:
            data_path: Path to the salon_bookings.csv file
        """
        self.data_path = Path(data_path)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV.

        Returns:
            DataFrame with raw data
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        print(f"[OK] Dataset loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        return self.df

    def parse_datetime_columns(self) -> pd.DataFrame:
        """
        Convert datetime columns from string to pandas datetime format.

        Returns:
            DataFrame with converted datetime columns
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        for col in self.DATETIME_COLUMNS:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"[OK] Parsed datetime column: {col}")

        return self.df

    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            strategy: 'drop' to remove rows with NaN, 'forward_fill' for forward fill,
                     'mean' for numeric columns, 'mode' for categorical

        Returns:
            DataFrame with missing values handled
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        missing_before = self.df.isnull().sum().sum()

        if missing_before == 0:
            print("[OK] No missing values detected")
            return self.df

        print(f"Found {missing_before} missing value(s). Applying '{strategy}' strategy...")

        if strategy == 'drop':
            self.df = self.df.dropna()

        elif strategy == 'forward_fill':
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')

        elif strategy == 'mean':
            for col in self.NUMERIC_COLUMNS:
                if col in self.df.columns and self.df[col].isnull().any():
                    self.df[col].fillna(self.df[col].mean(), inplace=True)

            for col in self.CATEGORICAL_COLUMNS:
                if col in self.df.columns and self.df[col].isnull().any():
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        missing_after = self.df.isnull().sum().sum()
        print(f"[OK] Missing values handled: {missing_before} → {missing_after}")

        return self.df

    def validate_data_types(self) -> dict:
        """
        Validate that all columns have expected data types.

        Returns:
            Dictionary with data type validation results
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        validation_results = {
            'datetime_columns': {},
            'numeric_columns': {},
            'categorical_columns': {},
            'id_columns': {}
        }

        for col in self.DATETIME_COLUMNS:
            if col in self.df.columns:
                is_datetime = pd.api.types.is_datetime64_any_dtype(self.df[col])
                validation_results['datetime_columns'][col] = is_datetime

        for col in self.NUMERIC_COLUMNS:
            if col in self.df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
                validation_results['numeric_columns'][col] = is_numeric

        for col in self.CATEGORICAL_COLUMNS:
            if col in self.df.columns:
                is_object = self.df[col].dtype == 'object'
                validation_results['categorical_columns'][col] = is_object

        for col in self.ID_COLUMNS:
            if col in self.df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
                validation_results['id_columns'][col] = is_numeric

        return validation_results

    def get_data_summary(self) -> dict:
        """
        Generate a summary of the cleaned dataset.

        Returns:
            Dictionary with dataset summary statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        summary = {
            'shape': self.df.shape,
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        return summary

    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Return the cleaned dataset.

        Returns:
            Cleaned pandas DataFrame ready for feature engineering
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        return self.df.copy()

    def process(self, missing_strategy: str = 'drop') -> pd.DataFrame:
        """
        Complete data processing pipeline: load → parse datetime → handle missing.

        Args:
            missing_strategy: Strategy for handling missing values

        Returns:
            Cleaned DataFrame ready for feature engineering
        """
        print("\n" + "="*60)
        print("SALON BOOKING DATA PROCESSING PIPELINE")
        print("="*60 + "\n")

        # Step 1: Load data
        self.load_data()
        print()

        # Step 2: Parse datetime columns
        self.parse_datetime_columns()
        print()

        # Step 3: Handle missing values
        self.handle_missing_values(strategy=missing_strategy)
        print()

        # Step 4: Validate data types
        validation = self.validate_data_types()
        all_valid = all(
            all(v for v in vals.values())
            for vals in validation.values()
        )
        print(f"[OK] Data type validation: {'PASSED' if all_valid else 'FAILED'}")
        print()

        # Step 5: Print summary
        summary = self.get_data_summary()
        print("DATASET SUMMARY:")
        print(f"  - Shape: {summary['shape'][0]:,} rows x {summary['shape'][1]} columns")
        print(f"  - Missing values: {summary['missing_values']}")
        print(f"  - Duplicates: {summary['duplicates']}")
        print(f"  - Memory usage: {summary['memory_usage_mb']:.2f} MB")
        print()
        print("="*60 + "\n")

        return self.get_cleaned_data()


def load_and_clean_data(data_path: str, missing_strategy: str = 'drop') -> pd.DataFrame:
    """
    Convenience function to load and clean the salon booking dataset in one call.

    Args:
        data_path: Path to the salon_bookings.csv file
        missing_strategy: Strategy for handling missing values ('drop', 'forward_fill', 'mean')

    Returns:
        Cleaned pandas DataFrame ready for feature engineering

    Example:
        >>> df = load_and_clean_data('../data/raw/salon_bookings.csv')
        >>> print(df.head())
    """
    processor = SalonDataProcessor(data_path)
    return processor.process(missing_strategy=missing_strategy)


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    # Resolve path relative to script location
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / "data" / "raw" / "salon_bookings.csv"

    # Method 1: Using the convenience function
    df_clean = load_and_clean_data(str(data_path))

    # Method 2: Using the class directly for more control
    # processor = SalonDataProcessor(str(data_path))
    # df_clean = processor.process(missing_strategy='drop')

    # Display cleaned data info
    print("\nCLEANED DATA INFO:")
    print(df_clean.info())
    print("\nFirst 5 rows:")
    print(df_clean.head())
