"""
Prediction Module – Salon No-Show Risk Scoring
Loads trained model artifacts and scores new booking records.

Usage
-----
Single booking (dict):
    from predict import NoShowPredictor
    predictor = NoShowPredictor()
    result = predictor.predict_one({
        "Booking_ID": 1023,
        "Customer_ID": 42,
        "Service_Type": "Haircut",
        "Branch": "Mumbai_Bandra",
        "Booking_Time": "2024-06-01 10:00:00",
        "Appointment_Time": "2024-06-08 09:00:00",
        "Booking_Lead_Time_Days": 7,
        "Past_Visit_Count": 5,
        "Past_Cancellation_Count": 1,
        "Past_No_Show_Count": 2,
        "Payment_Method": "Cash",
        "Day_of_Week": "Saturday",
        "Customer_Latent_Risk": 0.3,
    })
    # {"booking_id": 1023, "no_show_probability": 0.67, "risk_level": "High"}

Batch (DataFrame):
    results = predictor.predict_batch(df_new_bookings)
    # returns list of dicts

Convenience function:
    result  = predict_booking(booking_dict)     # single
    results = predict_bookings(df)              # batch
"""

import sys
import warnings
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import (
    add_appointment_hour,
    add_cancellation_ratio,
    add_edge_hour_flag,
    add_lead_time,
    add_new_customer_flag,
    add_no_show_rate,
    add_successful_visit_rate,
    encode_categoricals,
    DROP_COLS,
    CATEGORICAL_COLS,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

# Risk thresholds (probability boundaries)
RISK_BANDS = [
    (0.60, "High"),
    (0.40, "Medium"),
    (0.00, "Low"),
]


def _risk_label(probability: float) -> str:
    """Map a probability to a human-readable risk level."""
    for cutoff, label in RISK_BANDS:
        if probability >= cutoff:
            return label
    return "Low"


def _build_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to raw booking records for inference.

    Identical to the training pipeline with two differences:
    - No add_target() call  (no Outcome column in new records)
    - 'Outcome' stub is not needed; DROP_COLS handles it during get_X_y at
      training time; here we simply never add it.

    Args:
        df: Raw booking DataFrame with at minimum the columns the model was
            trained on (Booking_Time, Appointment_Time, Past_* counts, etc.)

    Returns:
        DataFrame with engineered feature columns (no target, no IDs).
    """
    df = df.copy()

    # Parse datetimes if still strings
    for col in ("Booking_Time", "Appointment_Time"):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = add_lead_time(df)
    df = add_appointment_hour(df)
    df = add_edge_hour_flag(df)
    df = add_no_show_rate(df)
    df = add_cancellation_ratio(df)
    df = add_new_customer_flag(df)
    df = add_successful_visit_rate(df)
    df = encode_categoricals(df, cols=CATEGORICAL_COLS)

    # Drop columns that were excluded from X at training time
    # (IDs, raw timestamps, raw counts, target – keep whatever exists)
    inference_drop = [c for c in DROP_COLS if c in df.columns and c != "No_Show"]
    df = df.drop(columns=inference_drop, errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class NoShowPredictor:
    """
    Load trained model artifacts once and score booking records on demand.

    Attributes
    ----------
    model      : fitted sklearn estimator
    scaler     : fitted StandardScaler
    threshold  : float – optimal decision threshold from training
    feature_cols: list[str] – ordered feature columns the model expects
    """

    def __init__(self, models_dir: str = None):
        """
        Load model, scaler, threshold, and expected feature columns from disk.

        Args:
            models_dir: Override path to the models directory.
                        Defaults to PROJECT_ROOT/models.
        """
        mdir = Path(models_dir) if models_dir else MODELS_DIR

        self.model     = joblib.load(mdir / "best_model.pkl")
        self.scaler    = joblib.load(mdir / "scaler.pkl")
        self.threshold = float(joblib.load(mdir / "best_threshold.pkl"))

        # Recover exact column order from the scaler (fit on X_train DataFrame)
        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_cols = list(self.scaler.feature_names_in_)
        else:
            # Fallback: read from saved X.csv column order
            x_csv = PROJECT_ROOT / "data" / "processed" / "X.csv"
            self.feature_cols = pd.read_csv(x_csv, nrows=0).columns.tolist()

        print(f"[OK] Model loaded       : {type(self.model).__name__}")
        print(f"[OK] Decision threshold : {self.threshold:.3f}")
        print(f"[OK] Features expected  : {len(self.feature_cols)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build features, align columns to training order, scale, return array.

        Missing OHE columns (unseen categories) are filled with 0.
        Extra columns (not seen at training) are dropped.
        """
        feat_df = _build_inference_features(df)

        # Align to training column order; fill unseen OHE cols with 0
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0)

        return self.scaler.transform(feat_df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_one(self, booking: dict) -> dict:
        """
        Score a single booking record.

        Args:
            booking: Dictionary with raw booking fields.

        Returns:
            dict with keys:
                booking_id          – Booking_ID from input (or None)
                no_show_probability – float [0, 1] rounded to 4 dp
                risk_level          – "High" | "Medium" | "Low"

        Example:
            >>> predictor.predict_one({"Booking_ID": 1023, ...})
            {"booking_id": 1023, "no_show_probability": 0.67, "risk_level": "High"}
        """
        df = pd.DataFrame([booking])
        X_scaled = self._prepare(df)
        probability = float(self.model.predict_proba(X_scaled)[0, 1])

        return {
            "booking_id"          : booking.get("Booking_ID"),
            "no_show_probability" : round(probability, 4),
            "risk_level"          : _risk_label(probability),
        }

    def predict_batch(self, df: pd.DataFrame) -> list[dict]:
        """
        Score a DataFrame of booking records.

        Args:
            df: Raw bookings DataFrame (same schema as training data).

        Returns:
            List of result dicts, one per row, in input order.

        Example:
            >>> results = predictor.predict_batch(df_new_bookings)
            >>> pd.DataFrame(results).head()
        """
        booking_ids = df.get("Booking_ID", pd.Series([None] * len(df)))

        X_scaled     = self._prepare(df)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return [
            {
                "booking_id"          : int(bid) if bid is not None else None,
                "no_show_probability" : round(float(p), 4),
                "risk_level"          : _risk_label(float(p)),
            }
            for bid, p in zip(booking_ids, probabilities)
        ]

    def predict_with_flag(self, booking: dict) -> dict:
        """
        Like predict_one but also includes the binary flag using the tuned threshold.

        Returns:
            dict with booking_id, no_show_probability, risk_level, and
            predicted_no_show (0 or 1).
        """
        result = self.predict_one(booking)
        result["predicted_no_show"] = int(result["no_show_probability"] >= self.threshold)
        return result


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def predict_booking(booking: dict, models_dir: str = None) -> dict:
    """
    One-call function to score a single booking dict.

    Loads model artifacts from disk on every call — use NoShowPredictor
    directly when scoring many bookings to avoid repeated disk I/O.
    """
    return NoShowPredictor(models_dir).predict_one(booking)


def predict_bookings(df: pd.DataFrame, models_dir: str = None) -> list[dict]:
    """
    One-call function to score a DataFrame of bookings.

    Loads model artifacts from disk on every call — use NoShowPredictor
    directly when scoring many batches to avoid repeated disk I/O.
    """
    return NoShowPredictor(models_dir).predict_batch(df)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    predictor = NoShowPredictor()
    print()

    # --- single booking: high-risk profile ---
    high_risk = {
        "Booking_ID"              : 1023,
        "Customer_ID"             : 77,
        "Service_Type"            : "Haircut",
        "Branch"                  : "Mumbai_Bandra",
        "Booking_Time"            : "2024-06-01 10:00:00",
        "Appointment_Time"        : "2024-06-08 09:00:00",   # early-morning edge hour
        "Booking_Lead_Time_Days"  : 7,
        "Past_Visit_Count"        : 5,
        "Past_Cancellation_Count" : 1,
        "Past_No_Show_Count"      : 3,                       # high no-show history
        "Payment_Method"          : "Cash",
        "Day_of_Week"             : "Saturday",
        "Customer_Latent_Risk"    : 0.6,
    }

    # --- single booking: low-risk profile ---
    low_risk = {
        "Booking_ID"              : 2048,
        "Customer_ID"             : 15,
        "Service_Type"            : "Facial",
        "Branch"                  : "Delhi_CP",
        "Booking_Time"            : "2024-06-10 09:00:00",
        "Appointment_Time"        : "2024-06-11 14:00:00",   # same-day, midday
        "Booking_Lead_Time_Days"  : 1,
        "Past_Visit_Count"        : 20,
        "Past_Cancellation_Count" : 0,
        "Past_No_Show_Count"      : 0,                       # perfect track record
        "Payment_Method"          : "UPI",
        "Day_of_Week"             : "Tuesday",
        "Customer_Latent_Risk"    : 0.05,
    }

    print("=== Single booking predictions ===")
    for booking in (high_risk, low_risk):
        result = predictor.predict_one(booking)
        print(f"  Booking {result['booking_id']:>5}  |  "
              f"P(no-show)={result['no_show_probability']:.4f}  |  "
              f"Risk: {result['risk_level']}")

    print()
    print("=== Batch prediction (same two records) ===")
    df_batch = pd.DataFrame([high_risk, low_risk])
    results  = predictor.predict_batch(df_batch)
    print(pd.DataFrame(results).to_string(index=False))

    print()
    print("=== With binary flag (tuned threshold) ===")
    print(predictor.predict_with_flag(high_risk))
