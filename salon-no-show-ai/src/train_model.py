"""
Model Training Script – Salon No-Show Prediction
Pipeline: data -> features -> train -> evaluate -> save

Steps
-----
1. Load raw CSV and clean it via SalonDataProcessor
2. Build feature matrix X and target y via build_features / get_X_y
3. Stratified 80/20 train-test split
4. Scale features for Logistic Regression (StandardScaler)
5. Train four classifiers with class-weighting:
     Logistic Regression | Decision Tree | Random Forest | XGBoost
6. Evaluate all models at threshold = 0.50
7. Select best model by ROC-AUC; tune decision threshold to maximise
   Recall while keeping Precision >= 40 %
8. Print final comparison table
9. Save best_model.pkl, scaler.pkl, best_threshold.pkl  ->  models/
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Make sibling imports work when running directly from /src
sys.path.insert(0, str(Path(__file__).parent))
from data_processing import load_and_clean_data
from feature_engineering import build_features, get_X_y

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "raw"  / "salon_bookings.csv"
MODELS_DIR   = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_metrics(clf, X_eval: np.ndarray, y_eval: pd.Series,
                threshold: float = 0.50) -> dict:
    """Return Accuracy / Precision / Recall / F1 / ROC-AUC at a given threshold."""
    y_proba = clf.predict_proba(X_eval)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return {
        "Accuracy" : round(accuracy_score(y_eval, y_pred), 4),
        "Precision": round(precision_score(y_eval, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_eval, y_pred), 4),
        "F1"       : round(f1_score(y_eval, y_pred), 4),
        "ROC-AUC"  : round(roc_auc_score(y_eval, y_proba), 4),
    }


def tune_threshold(clf, X_eval: np.ndarray, y_eval: pd.Series,
                   min_precision: float = 0.40) -> float:
    """
    Find the lowest threshold at which Recall is maximised
    while Precision stays >= min_precision.

    Returns the optimal threshold (falls back to 0.50 if no viable point found).
    """
    y_proba                    = clf.predict_proba(X_eval)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_eval, y_proba)

    thresh_df = pd.DataFrame({
        "Threshold": thresholds,
        "Precision": precisions[:-1],
        "Recall"   : recalls[:-1],
    })

    viable = thresh_df[thresh_df["Precision"] >= min_precision]
    if viable.empty:
        print(f"  [WARN] No threshold found with Precision >= {min_precision:.0%}; using 0.50")
        return 0.50

    best_row = viable.loc[viable["Recall"].idxmax()]
    return float(best_row["Threshold"])


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Stage 1-2: load, clean, and engineer features."""
    print("\n" + "="*60)
    print("STAGE 1/5  –  DATA LOADING & FEATURE ENGINEERING")
    print("="*60)

    df_clean = load_and_clean_data(str(DATA_PATH))
    df_feat  = build_features(df_clean)
    X, y     = get_X_y(df_feat)

    print(f"[OK] Feature matrix : {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"[OK] No-show rate   : {y.mean():.2%}")
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    """Stage 3-4: stratified split + StandardScaler for LR."""
    print("\n" + "="*60)
    print("STAGE 2/5  –  TRAIN / TEST SPLIT & SCALING")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)
    X_te_sc   = scaler.transform(X_test)

    print(f"[OK] Train : {X_train.shape[0]:,} rows  |  Test : {X_test.shape[0]:,} rows")
    print(f"[OK] No-show  ->  train: {y_train.mean():.2%}  |  test: {y_test.mean():.2%}")

    return X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test, scaler


def train_models(X_train, X_tr_sc, y_train) -> dict:
    """Stage 5: train all four classifiers."""
    print("\n" + "="*60)
    print("STAGE 3/5  –  MODEL TRAINING")
    print("="*60)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            X_tr_sc,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42),
            X_train,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   max_depth=10, random_state=42, n_jobs=-1),
            X_train,
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                          random_state=42, n_jobs=-1),
            X_train,
        ),
    }

    trained = {}
    for name, (clf, X_fit) in models.items():
        clf.fit(X_fit, y_train)
        trained[name] = (clf, X_fit)          # keep training X for threshold tuning
        print(f"[OK] {name} trained")

    return trained


def evaluate_models(trained: dict, X_test, X_te_sc, y_test) -> pd.DataFrame:
    """Stage 6: evaluate all models at threshold = 0.50."""
    print("\n" + "="*60)
    print("STAGE 4/5  –  BASELINE EVALUATION (threshold = 0.50)")
    print("="*60)

    # Resolve eval split per model (LR uses scaled data)
    eval_X = {
        "Logistic Regression": X_te_sc,
        "Decision Tree"      : X_test,
        "Random Forest"      : X_test,
        "XGBoost"            : X_test,
    }

    results = {}
    for name, (clf, _) in trained.items():
        results[name] = get_metrics(clf, eval_X[name], y_test)

    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    return df_results


def tune_and_select(trained: dict, baseline: pd.DataFrame,
                    X_test, X_te_sc, y_test) -> tuple:
    """Stage 7: pick best model by AUC, tune threshold, print final comparison."""
    print("\n" + "="*60)
    print("STAGE 5/5  –  THRESHOLD TUNING & FINAL COMPARISON")
    print("="*60)

    best_name = baseline["ROC-AUC"].idxmax()
    best_clf, _ = trained[best_name]
    best_X_eval = X_te_sc if "Logistic" in best_name else X_test

    print(f"[OK] Best model by ROC-AUC: {best_name}  ({baseline.loc[best_name, 'ROC-AUC']})")

    best_threshold = tune_threshold(best_clf, best_X_eval, y_test)

    tuned_metrics = get_metrics(best_clf, best_X_eval, y_test, threshold=best_threshold)
    tuned_metrics["ROC-AUC"] = baseline.loc[best_name, "ROC-AUC"]   # AUC is threshold-independent

    comparison = pd.DataFrame({
        f"{best_name}  (thr=0.50)"                    : baseline.loc[best_name],
        f"{best_name}  (thr={best_threshold:.2f})"    : tuned_metrics,
    }).T

    print(f"\nOptimal threshold : {best_threshold:.3f}")
    print(comparison.to_string())

    recall_default = baseline.loc[best_name, "Recall"]
    recall_tuned   = tuned_metrics["Recall"]
    print(f"\nRecall  {recall_default:.3f} -> {recall_tuned:.3f}  "
          f"(+{recall_tuned - recall_default:.3f})")

    return best_name, best_clf, best_threshold


def save_artifacts(best_name: str, best_clf, scaler: StandardScaler,
                   best_threshold: float) -> None:
    """Save model, scaler, and threshold to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_clf,       MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler,         MODELS_DIR / "scaler.pkl")
    joblib.dump(best_threshold, MODELS_DIR / "best_threshold.pkl")

    print(f"\n[OK] Saved  best_model.pkl      ({best_name})")
    print(f"[OK] Saved  scaler.pkl")
    print(f"[OK] Saved  best_threshold.pkl  ({best_threshold:.3f})")
    print(f"\nArtifacts written to: {MODELS_DIR}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "#"*60)
    print("#   SALON NO-SHOW  –  MODEL TRAINING PIPELINE")
    print("#"*60)

    X, y = load_data()
    X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test, scaler = split_and_scale(X, y)
    trained  = train_models(X_train, X_tr_sc, y_train)
    baseline = evaluate_models(trained, X_test, X_te_sc, y_test)
    best_name, best_clf, best_threshold = tune_and_select(
        trained, baseline, X_test, X_te_sc, y_test
    )
    save_artifacts(best_name, best_clf, scaler, best_threshold)

    print("\n" + "#"*60)
    print("#   PIPELINE COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
