# Salon No-Show AI — Production Deployment Guide

Predicts the probability that a booked salon appointment will result in a no-show,
then automatically routes each booking to the right intervention (from a single SMS
to full prepayment) before any slot is lost.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Results](#2-model-results)
3. [Model Integration with Booking API](#3-model-integration-with-booking-api)
4. [Real-Time Inference Flow](#4-real-time-inference-flow)
5. [Retraining Strategy](#5-retraining-strategy)
6. [Data Drift Detection](#6-data-drift-detection)
7. [Monitoring Metrics](#7-monitoring-metrics)
8. [Scalability for 200K+ Users](#8-scalability-for-200k-users)
9. [Repository Structure](#9-repository-structure)
10. [Quick Start](#10-quick-start)

---

## 1. Project Overview

| Item | Detail |
|---|---|
| Dataset | 50,000 bookings across 7 branches, 7 service types, 4 payment methods |
| No-show rate | **24.1%** (37,958 shows / 12,042 no-shows) |
| Class handling | `class_weight='balanced'` (LR / DT / RF); `scale_pos_weight` ≈ 3.15 (XGBoost) |
| Train / Val / Test | 60% / 20% / 20%, stratified |
| Decision threshold | Tuned on the **validation set** to maximise F1 — never on test (no data leakage) |
| Business output | 7-tier action ladder: NO_ACTION → SMS → MULTI_REMINDER → CALL → SOFT_DEPOSIT (25%) → HARD_DEPOSIT (50%) → PREPAYMENT (100%) |

**Key dataset findings that drive business rules:**

| Signal | No-show rate | Action implication |
|---|---|---|
| Cash payment | **39.4%** | 2.6× higher than Card (14.9%) — primary deposit trigger |
| Weekend slot | ~25.2% | Modest uplift vs Tuesday (23.0%) |
| Facial / Pedicure / Manicure | ~26% | Highest no-show services |
| Bridal Makeup | ~19.8% | Lowest no-show — revenue protection priority |
| Hair Coloring | ~20.6% | Long slot — time-loss risk |
| Mean lead time | 3.6 days (max 13) | Bookings ≥ 7 days ahead trigger multi-touch reminders |

---

## 2. Model Results

### 2.1 Engineered Features (12 inputs after pruning)

| Category | Feature | Rationale |
|---|---|---|
| Temporal | `Lead_Time_Days` | Longer lead → higher forget risk |
| Temporal | `Appt_Hour` | Hour-of-day no-show pattern |
| Temporal | `Is_Edge_Hour` | Flag for ≤ 9h or ≥ 18h slots (38.7% of bookings) |
| Behaviour | `Customer_No_Show_Rate` | Mean 22.0%, std 32.2% — strongest individual signal |
| Behaviour | `Is_New_Customer` | 10% of bookings; no history = high uncertainty |
| Behaviour | `Successful_Visit_Rate` | Loyalty metric; mean 64.6%, captures combined no-show + cancellation |
| Risk score | `Customer_Latent_Risk` | Generator-embedded risk score; high correlation with target |
| Encoded | `Branch_Bangalore_Indiranagar` | Only branch with measurable effect after pruning |
| Encoded | `Service_Type_Bridal Makeup` | Significantly lowest no-show tier |
| Encoded | `Service_Type_Hair Coloring` | Premium time-slot service |
| Encoded | `Payment_Method_Card` | Reference: 14.9% no-show (Cash base at 39.4%) |
| Encoded | `Payment_Method_UPI` | Digital payment = lower risk than Cash |

Dropped features (near-zero importance after cross-validation):
`Cancellation_Ratio`, `Log_Past_Visit_Count`, `Smoothed_NoShow_Rate`,
`Capped_Good_Visits`, interaction terms, 5 branches, 6 days of week, 3 services,
`Past_Cancellation_Count`.

### 2.2 Baseline Models (threshold = 0.50, test set)

Four models were trained with class balancing. XGBoost configuration:

```python
XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=10, gamma=0.2,
    reg_alpha=0.1, reg_lambda=2.0,
    scale_pos_weight=3.15          # majority / minority ratio
)
```

Decision Tree was capped at `max_depth=8, min_samples_leaf=50` to control overfitting.
Random Forest used `n_estimators=500, max_depth=12, min_samples_leaf=20`.

### 2.3 Threshold Tuning and Ensemble

- Each model's decision threshold was tuned by maximising F1 on the **val set**,
  then reported on the unseen **test set**.
- A soft-voting ensemble averages probabilities from LR + RF + XGB — smoothing
  individual model errors and improving calibration.
- Final action engine uses four probability bands derived from the tuned threshold:

| Band | Probability | Minimum action |
|---|---|---|
| Caution | 0.25 – 0.40 | SMS reminder (or Soft Deposit if Cash) |
| Elevated | 0.40 – 0.60 | Multi-channel reminder or Hard Deposit |
| High | 0.60 – 0.75 | Hard Deposit (50%) |
| Critical | ≥ 0.75 | Prepayment (100%) for chronic + premium; Hard Deposit otherwise |

Saved artifacts: `models/best_model.pkl`, `models/scaler.pkl`, `models/best_threshold.pkl`

---

## 3. Model Integration with Booking API

### Architecture

```
Booking System (REST API)
        │
        │  POST /bookings  (on every new booking)
        ▼
┌─────────────────────────────┐
│     API Gateway / BFF       │  Rate limiting, auth, logging
└────────────┬────────────────┘
             │  Async event: booking_created
             ▼
┌─────────────────────────────┐
│   No-Show Risk Service      │  Python FastAPI / Flask
│   ─────────────────────     │
│   NoShowPredictor.predict() │  Loaded once at startup
│   ActionEngine.evaluate()   │  Rule-based, stateless
└────────────┬────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
Booking DB       Notification Service
(store score)    (SMS / Email / Call jobs)
```

### API Contract

**Request** — every new booking fires this payload:

```json
{
  "Booking_ID": 10045,
  "Customer_ID": 728,
  "Service_Type": "Hair Coloring",
  "Branch": "Mumbai_Bandra",
  "Booking_Time": "2026-03-16T10:00:00",
  "Appointment_Time": "2026-03-23T09:00:00",
  "Past_Visit_Count": 3,
  "Past_No_Show_Count": 2,
  "Past_Cancellation_Count": 0,
  "Payment_Method": "Cash",
  "Day_of_Week": "Monday",
  "Customer_Latent_Risk": 0.55
}
```

**Response** — synchronous, < 10 ms:

```json
{
  "booking_id": 10045,
  "no_show_probability": 0.7312,
  "risk_level": "High",
  "action_type": "HARD_DEPOSIT",
  "channels": ["SMS", "Email", "Call"],
  "timing_days_before": [3, 1],
  "deposit_pct": 50,
  "reason": "50% refundable deposit. Cash no-show rate 39.4%; chronic history (67% rate).",
  "risk_drivers": [
    "chronic no-show history (67% rate)",
    "cash payment (39.4% no-show rate vs 14.9% for card)",
    "booked 7d in advance (mean=3.6d)"
  ],
  "priority": 4,
  "escalation_note": "If deposit not paid within 12h, release slot to waitlist."
}
```

### Integration Code

```python
from src.predict import NoShowPredictor
from src.business_actions import ActionEngine

# Loaded once per process — disk I/O happens here, not per request
predictor = NoShowPredictor()
engine    = ActionEngine()

def handle_new_booking(booking: dict) -> dict:
    prediction = predictor.predict_one(booking)
    action     = engine.evaluate(booking, prediction)
    return action.to_dict()
```

---

## 4. Real-Time Inference Flow

```
New booking received
        │
        ▼
1. PARSE DATETIMES
   Booking_Time, Appointment_Time → datetime64

        │
        ▼
2. FEATURE ENGINEERING  (src/feature_engineering.py)
   Lead_Time_Days     = (Appointment_Time - Booking_Time).dt.days
   Appt_Hour          = Appointment_Time.dt.hour
   Is_Edge_Hour       = hour ≤ 9 OR hour ≥ 18
   Customer_No_Show_Rate = Past_No_Show_Count / Past_Visit_Count  (0.0 if new)
   Is_New_Customer    = 1 if Past_Visit_Count == 0
   Successful_Visit_Rate = clipped to [0, 1]
   One-hot encode: Service_Type, Branch, Payment_Method, Day_of_Week
   Align to training column order; fill unseen OHE columns with 0

        │
        ▼
3. SCALE
   StandardScaler.transform(X)  — scaler fitted on 30,000-row training set

        │
        ▼
4. SCORE
   best_model.predict_proba(X_scaled)[:, 1]
   → no_show_probability ∈ [0, 1]

        │
        ▼
5. CLASSIFY
   risk_level = "High"   if p ≥ 0.60
              = "Medium"  if p ≥ 0.40
              = "Low"     otherwise

        │
        ▼
6. BUSINESS ACTION  (src/business_actions.py)
   RiskDiagnosis: extract active risk drivers
   ActionEngine._select_action(): 5-band rule tree
   → BusinessAction (action_type, channels, deposit_pct, timing, reason)

        │
        ▼
7. DISPATCH
   Store score in booking DB
   Enqueue intervention jobs at computed timing_days_before
```

**Latency budget (single booking):**

| Step | Typical time |
|---|---|
| Feature engineering | < 1 ms |
| StandardScaler transform | < 0.5 ms |
| Model inference (XGBoost / ensemble) | 2 – 5 ms |
| Action engine rule evaluation | < 0.5 ms |
| **Total (P99)** | **< 10 ms** |

---

## 5. Retraining Strategy

### Trigger-Based Retraining

Do **not** retrain on a fixed calendar. Retrain when observed performance degrades:

| Trigger | Threshold | Action |
|---|---|---|
| Rolling 4-week Recall drops | below 0.55 | Initiate retraining pipeline |
| Rolling 4-week Precision drops | below 0.35 | Initiate retraining pipeline |
| PSI on `Customer_No_Show_Rate` | PSI > 0.20 | Retraining + feature investigation |
| Monthly booking volume increase | > 20% of training size | Schedule proactive retrain |

### Retraining Pipeline

```
1. Append last 90 days of confirmed outcomes (Show / No-Show)
   to the training corpus

2. Re-run feature engineering (src/feature_engineering.py)
   — global no-show rate prior (24.1% → recalculate from corpus)

3. Split: 60 / 20 / 20  (stratified, same random_state)

4. Retrain all 4 models with same hyperparameters
   (scale_pos_weight recalculated from new imbalance ratio)

5. Tune threshold on new validation set
   (maximise F1 with Precision ≥ 0.40 floor)

6. Run champion vs challenger:
   If new model F1 improvement > 0.02  → promote to production
   Otherwise                           → keep current model

7. Atomic swap of models/best_model.pkl, models/scaler.pkl,
   models/best_threshold.pkl
   — zero-downtime: load new artifacts before retiring old ones

8. Tag git commit with model version and val/test metrics
```

### Minimum Training Data Requirements

| Condition | Minimum records |
|---|---|
| Full retrain (all 4 models) | 20,000 bookings with confirmed outcomes |
| Threshold-only retune | 5,000 bookings (val set only) |
| Emergency patch (concept drift) | 2,000 most recent bookings |

---

## 6. Data Drift Detection

### Features to Monitor

The model uses 12 features. Priority drift checks, ranked by impact on predictions:

| Feature | Drift metric | Alert threshold | Why it matters |
|---|---|---|---|
| `Customer_No_Show_Rate` | PSI | > 0.20 | Strongest predictor; seasonal shifts break the model fast |
| `Customer_Latent_Risk` | PSI | > 0.20 | Generator-embedded signal; if source changes, model fails silently |
| `Lead_Time_Days` | KS test | p < 0.01 | Booking-window changes (e.g., new "book same-day" feature) |
| `Is_New_Customer` | Chi-squared | p < 0.01 | Marketing campaigns shift new-vs-returning ratio |
| `Payment_Method` (distribution) | Chi-squared | p < 0.01 | Cash → UPI migration changes base rates dramatically |
| `Is_Edge_Hour` | Chi-squared | p < 0.01 | New operating hours alter slot distribution |
| No-show rate (ground truth) | Z-test on rolling 4-week | > 2σ from 24.1% | Direct signal that world has changed |

### PSI Formula

```
PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)

Interpretation:
  PSI < 0.10  → no significant drift
  0.10 – 0.20 → moderate drift; monitor closely
  PSI > 0.20  → significant drift; retrain required
```

### Implementation Sketch

```python
import numpy as np

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between reference and current distribution."""
    expected_pct, breakpoints = np.histogram(expected, bins=bins, density=False)
    actual_pct, _             = np.histogram(actual, bins=breakpoints, density=False)

    expected_pct = expected_pct / expected_pct.sum()
    actual_pct   = actual_pct   / actual_pct.sum()

    # Clip to avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct   = np.clip(actual_pct,   1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

# Run weekly against training distribution snapshot
drift_score = psi(training_no_show_rates, last_30_days_no_show_rates)
if drift_score > 0.20:
    trigger_retraining_pipeline()
```

### Concept Drift vs Covariate Drift

| Type | What changes | Detection | Response |
|---|---|---|---|
| **Covariate drift** | Input distribution shifts (e.g., more Cash bookings) | PSI / KS on features | Retrain on recent data |
| **Concept drift** | Relationship between features and no-show changes (e.g., Cash customers start showing up) | Model calibration check; precision/recall drop on recent actuals | Full retrain required |
| **Label drift** | Actual no-show rate shifts (e.g., post-COVID recovery) | Rolling Z-test on observed rate vs 24.1% baseline | Adjust class weights + retrain |

---

## 7. Monitoring Metrics

### 7.1 Model Performance Metrics (computed weekly on confirmed outcomes)

| Metric | Target | Alert threshold | Why |
|---|---|---|---|
| **Recall** | ≥ 0.65 | < 0.55 | Missing real no-shows costs empty chair time |
| **Precision** | ≥ 0.45 | < 0.35 | False alarms annoy reliable customers with unnecessary deposits |
| **F1 Score** | ≥ 0.55 | < 0.45 | Balanced measure for imbalanced class (24.1% minority) |
| **ROC-AUC** | ≥ 0.82 | < 0.75 | Ranking quality independent of threshold |
| **Calibration (Brier Score)** | ≤ 0.15 | > 0.22 | Ensures probabilities are meaningful, not just ordinal |

> Precision matters here: Cash customers have a **2.6× higher base no-show rate** (39.4% vs 14.9% for Card).
> A poorly calibrated model that deposits everyone with Cash would hurt loyal customers.

### 7.2 Business Outcome Metrics (computed monthly)

| Metric | Baseline | Target |
|---|---|---|
| No-show rate (overall) | 24.1% | < 18% after interventions |
| No-show rate on HIGH-risk bookings | ~65%+ | < 40% post-action |
| Deposit collection rate (when requested) | — | > 75% |
| Customer churn from deposit friction | — | < 2% of deposit-requested bookings |
| Revenue recovered per month (avoided empty slots) | — | Track via slot × avg service price |

### 7.3 Operational Metrics (real-time)

| Metric | SLO |
|---|---|
| Inference latency P50 | < 5 ms |
| Inference latency P99 | < 10 ms |
| Service availability | 99.9% |
| Failed predictions (model errors) | < 0.1% of requests |
| Missing features at inference (null rate) | < 1% of requests |

### 7.4 Drift Metrics (weekly automated report)

| Feature | Metric | Reference value | Alert if |
|---|---|---|---|
| `Customer_No_Show_Rate` | PSI | Training distribution | PSI > 0.20 |
| `Lead_Time_Days` | Mean | 3.08 days | Mean shifts by > 1.0 day |
| `Is_New_Customer` | Proportion | 10.0% | Proportion shifts by > 5 pp |
| `Payment_Method_Cash` | Proportion | From training | Shifts by > 8 pp |
| Observed no-show rate | Rolling 4-week | 24.1% | Outside ±4 pp for 3 consecutive weeks |

---

## 8. Scalability for 200K+ Users

### 8.1 Current Capacity

The trained model scores a single booking in **< 10 ms** on one CPU core.
A single `NoShowPredictor` instance (model loaded once at startup) can handle:
- **~100 predictions/second** per single-core worker
- **~1,000 predictions/second** per 8-core machine

50,000 bookings/day = ~0.58 bookings/second average — well within single-instance capacity.

### 8.2 Path to 200K+ Users

**Projected load:** 200,000 active users × ~0.5 bookings/user/month ≈ **58 bookings/minute peak.**

| Scale milestone | Architecture |
|---|---|
| 0 – 50K users | Single FastAPI process, 4 workers, single machine |
| 50K – 200K users | Horizontally scaled workers behind a load balancer (e.g., Nginx + Gunicorn) |
| 200K+ users | Kubernetes deployment; auto-scaling on CPU > 60%; model served via ONNX for 3–5× inference speedup |

### 8.3 Stateless Design (already in place)

`NoShowPredictor` and `ActionEngine` are **stateless** — all state lives in the
model artifacts (`.pkl` files) loaded at startup. This means:

- Any worker can handle any request — no session affinity required.
- Horizontal scaling requires only adding replicas; no shared mutable state.
- Rolling restarts during model updates are safe (old workers finish in-flight requests
  before shutdown).

### 8.4 Batch Inference for Non-Real-Time Paths

For next-day reminder scheduling (not latency-sensitive):

```python
# Score all tomorrow's bookings in one batch — O(n) not O(n × 1)
predictor = NoShowPredictor()          # model loaded once
results   = predictor.predict_batch(df_tomorrows_bookings)
# → list of dicts; < 1s for 10,000 bookings
```

At 200K users, batch scoring 5,000 next-day bookings takes ≈ 500 ms —
well within any overnight scheduling window.

### 8.5 Database Design for 200K Users

```
bookings table:
  booking_id         PK
  customer_id        FK → customers
  appointment_time
  ...raw fields...

risk_scores table:
  booking_id         FK → bookings
  no_show_probability FLOAT
  risk_level         VARCHAR(10)
  action_type        VARCHAR(20)
  scored_at          TIMESTAMP
  model_version      VARCHAR(20)   ← essential for A/B and rollback

intervention_log:
  booking_id         FK
  channel            VARCHAR(10)
  sent_at            TIMESTAMP
  outcome            VARCHAR(10)   ← delivered / bounced / confirmed
```

Indexes: `(risk_level, appointment_time)` for the daily intervention queue;
`(customer_id, scored_at)` for customer-level drift analysis.

### 8.6 Model Serving at Scale — ONNX Export Path

```python
# Export to ONNX for 3-5x faster CPU inference (no sklearn overhead)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

onnx_model = convert_sklearn(
    best_model,
    initial_types=[("X", FloatTensorType([None, len(feature_cols)]))]
)
with open("models/best_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# At inference (onnxruntime):
import onnxruntime as rt
sess = rt.InferenceSession("models/best_model.onnx")
proba = sess.run(["probabilities"], {"X": X_scaled.astype("float32")})[0][:, 1]
```

ONNX inference runs at ~30–50K predictions/second on a single core —
sufficient for **>1M daily bookings** without GPU.

---

## 9. Repository Structure

```
salon-no-show-ai/
├── data/
│   ├── raw/
│   │   └── salon_bookings.csv          # 50,000 raw bookings, 13 columns
│   └── processed/
│       ├── salon_bookings_featured.csv # All 37 columns post-engineering
│       ├── X.csv                       # Feature matrix (12 columns)
│       └── y.csv                       # Target (No_Show, 24.1% positive rate)
│
├── models/
│   ├── best_model.pkl                  # Champion model (LR / RF / XGB / Ensemble)
│   ├── scaler.pkl                      # StandardScaler fitted on 30K training rows
│   ├── best_threshold.pkl              # F1-optimal threshold (val-tuned, not test)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Distribution, correlation, hourly/daily patterns
│   ├── 02_feature_engineering.ipynb   # 12-feature pipeline with rationale
│   └── 03_model_training.ipynb        # 4 models + ensemble + threshold tuning
│
├── src/
│   ├── data_processing.py              # Load, clean, parse datetimes
│   ├── feature_engineering.py          # build_features(), get_X_y(), modular functions
│   ├── train_model.py                  # Training orchestrator
│   ├── predict.py                      # NoShowPredictor (predict_one / predict_batch)
│   └── business_actions.py            # ActionEngine — 7-tier intervention ladder
│
├── dashboard/
│   └── streamlit_app.py               # Interactive risk dashboard
│
├── reports/figures/                    # ROC curves, confusion matrices, PR curves
├── requirements.txt
└── README.md
```

---

## 10. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Score a single booking
python -c "
from src.predict import NoShowPredictor
from src.business_actions import ActionEngine

predictor = NoShowPredictor()
engine    = ActionEngine()

booking = {
    'Booking_ID': 1023, 'Customer_ID': 77,
    'Service_Type': 'Haircut', 'Branch': 'Mumbai_Bandra',
    'Booking_Time': '2026-03-16 10:00:00',
    'Appointment_Time': '2026-03-23 09:00:00',
    'Past_Visit_Count': 5, 'Past_Cancellation_Count': 1,
    'Past_No_Show_Count': 3, 'Payment_Method': 'Cash',
    'Day_of_Week': 'Monday', 'Customer_Latent_Risk': 0.6,
}
pred   = predictor.predict_one(booking)
action = engine.evaluate(booking, pred)
print(action.summary())
"

# Run full demo pipeline
python demo_full_pipeline.py

# Launch Streamlit dashboard
streamlit run dashboard/streamlit_app.py
```
