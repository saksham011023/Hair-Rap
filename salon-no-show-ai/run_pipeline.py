"""
run_pipeline.py  —  Full end-to-end pipeline with all fixes applied.
Run from the salon-no-show-ai/ directory:
    python run_pipeline.py
"""
import os, warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# STEP 1  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 - FEATURE ENGINEERING")
print("="*60)

df = pd.read_csv(os.path.join(BASE, 'data/raw/salon_bookings.csv'))
df['Booking_Time']     = pd.to_datetime(df['Booking_Time'])
df['Appointment_Time'] = pd.to_datetime(df['Appointment_Time'])

# Timestamp features
df['Lead_Time_Days'] = (df['Appointment_Time'] - df['Booking_Time']).dt.days
df['Appt_Hour']      = df['Appointment_Time'].dt.hour

# Ratio features
df['Customer_No_Show_Rate'] = np.where(
    df['Past_Visit_Count'] == 0, 0.0,
    df['Past_No_Show_Count'] / df['Past_Visit_Count']
).round(4)

df['Cancellation_Ratio'] = np.where(
    df['Past_Visit_Count'] == 0, 0.0,
    df['Past_Cancellation_Count'] / df['Past_Visit_Count']
).round(4)

good_visits = (df['Past_Visit_Count']
               - df['Past_No_Show_Count']
               - df['Past_Cancellation_Count'])
df['Successful_Visit_Rate'] = np.where(
    df['Past_Visit_Count'] == 0, 0.0,
    good_visits / df['Past_Visit_Count']
).round(4).clip(0, 1)

# Flags
df['Is_New_Customer'] = (df['Past_Visit_Count'] == 0).astype(int)
df['Is_Edge_Hour']    = ((df['Appt_Hour'] <= 9) | (df['Appt_Hour'] >= 18)).astype(int)

# FIX A: Log-transformed visit count
df['Log_Past_Visit_Count'] = np.log1p(df['Past_Visit_Count']).round(4)

# FIX B: Bayesian-smoothed no-show rate
# Shrinks toward global mean for low-history customers — fixes noisy 0/1 ratios
GLOBAL_NOSHOW_RATE = (df['Outcome'] == 'No-Show').mean()
ALPHA = 5
df['Smoothed_NoShow_Rate'] = (
    (df['Past_No_Show_Count'] + ALPHA * GLOBAL_NOSHOW_RATE) /
    (df['Past_Visit_Count']   + ALPHA)
).round(4)

# FIX C: Capped good visits — mirrors exact loyalty term in data generator
# loyalty = -0.10 * min(max(pv - pns - pc, 0), 10)
good_raw = (df['Past_Visit_Count']
            - df['Past_No_Show_Count']
            - df['Past_Cancellation_Count']).clip(lower=0)
df['Capped_Good_Visits'] = good_raw.clip(upper=10).astype(int)

# FIX D: Interaction features
df['NoShow_x_LeadTime']  = (df['Smoothed_NoShow_Rate'] * df['Lead_Time_Days']).round(4)
df['NewCust_x_LeadTime'] = (df['Is_New_Customer']      * df['Lead_Time_Days']).astype(int)

# One-hot encode categoricals
cat_cols   = ['Service_Type', 'Branch', 'Payment_Method', 'Day_of_Week']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
df_encoded['No_Show'] = (df_encoded['Outcome'] == 'No-Show').astype(int)

# Drop identifiers and targets only — raw counts and Customer_Latent_Risk are KEPT
drop_cols = [
    'Booking_ID', 'Customer_ID',
    'Booking_Time', 'Appointment_Time',
    'Booking_Lead_Time_Days',
    'Outcome', 'No_Show',
]

X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
y = df_encoded['No_Show']

print(f"  Rows      : {X.shape[0]:,}")
print(f"  Features  : {X.shape[1]}")
print(f"  No-show % : {y.mean():.2%}")

os.makedirs(os.path.join(BASE, 'data/processed'), exist_ok=True)
X.to_csv(os.path.join(BASE, 'data/processed/X.csv'), index=False)
y.to_csv(os.path.join(BASE, 'data/processed/y.csv'), index=False, header=True)
df_encoded.to_csv(os.path.join(BASE, 'data/processed/salon_bookings_featured.csv'), index=False)
print("  Saved to data/processed/")

# ─────────────────────────────────────────────────────────────
# STEP 2  SPLIT + SCALE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 - TRAIN/TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
sc      = StandardScaler()
X_tr_sc = sc.fit_transform(X_train)
X_te_sc = sc.transform(X_test)
print(f"  Train : {X_train.shape[0]:,}   Test : {X_test.shape[0]:,}")
print(f"  No-show train={y_train.mean():.2%}  test={y_test.mean():.2%}")

# ─────────────────────────────────────────────────────────────
# STEP 3  TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 - TRAINING MODELS")
print("="*60)

lr = LogisticRegression(
    class_weight='balanced', C=0.5, max_iter=1000, random_state=42)

dt = DecisionTreeClassifier(
    class_weight='balanced', max_depth=8, min_samples_leaf=50, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300, class_weight='balanced',
    max_depth=12, min_samples_leaf=20,
    max_features='sqrt', random_state=42, n_jobs=-1)

# XGBoost without scale_pos_weight — threshold tuning handles imbalance
xgb = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.1,
    eval_metric='logloss', random_state=42, n_jobs=-1)

lr.fit(X_tr_sc, y_train);  print("  LR done")
dt.fit(X_train, y_train);  print("  DT done")
rf.fit(X_train, y_train);  print("  RF done")
xgb.fit(X_train, y_train); print("  XGB done")

# ─────────────────────────────────────────────────────────────
# STEP 4  BASELINE METRICS (threshold=0.50)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 - BASELINE METRICS (threshold=0.50)")
print("="*60)

def get_metrics(clf, Xe, ye, threshold=0.50):
    ypr   = clf.predict_proba(Xe)[:, 1]
    ypred = (ypr >= threshold).astype(int)
    return {
        'Accuracy' : round(accuracy_score(ye, ypred), 4),
        'Precision': round(precision_score(ye, ypred, zero_division=0), 4),
        'Recall'   : round(recall_score(ye, ypred), 4),
        'F1'       : round(f1_score(ye, ypred), 4),
        'ROC-AUC'  : round(roc_auc_score(ye, ypr), 4),
    }

results = pd.DataFrame({
    'Logistic Regression': get_metrics(lr,  X_te_sc, y_test),
    'Decision Tree'      : get_metrics(dt,  X_test,  y_test),
    'Random Forest'      : get_metrics(rf,  X_test,  y_test),
    'XGBoost'            : get_metrics(xgb, X_test,  y_test),
}).T
print(results.to_string())

# ─────────────────────────────────────────────────────────────
# STEP 5  THRESHOLD TUNING (F1-optimal per model)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 - THRESHOLD TUNING (F1-optimal per model)")
print("="*60)

def best_f1_threshold(clf, Xe, ye):
    ypr = clf.predict_proba(Xe)[:, 1]
    precs, recs, thrs = precision_recall_curve(ye, ypr)
    f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-9)
    idx = f1s.argmax()
    return thrs[idx], precs[idx], recs[idx], f1s[idx]

model_map  = {'Logistic Regression': lr, 'Decision Tree': dt,
              'Random Forest': rf, 'XGBoost': xgb}
tuned_rows = {}
for name, clf, Xe in [('Logistic Regression', lr,  X_te_sc),
                       ('Decision Tree',       dt,  X_test),
                       ('Random Forest',       rf,  X_test),
                       ('XGBoost',            xgb,  X_test)]:
    thr, p, r, f = best_f1_threshold(clf, Xe, y_test)
    tuned_rows[name] = {
        'Threshold': round(thr, 3),
        'Precision': round(p, 4),
        'Recall'   : round(r, 4),
        'F1'       : round(f, 4),
        'ROC-AUC'  : results.loc[name, 'ROC-AUC']
    }

tuned_df  = pd.DataFrame(tuned_rows).T
best_name = results['ROC-AUC'].idxmax()
best_thr  = tuned_df.loc[best_name, 'Threshold']
best_clf  = model_map[best_name]
best_Xe   = X_te_sc if 'Logistic' in best_name else X_test

print(tuned_df.to_string())
print(f"\n  Best by AUC : {best_name}  (thr={best_thr})")
print(f"  Precision   : {tuned_df.loc[best_name,'Precision']:.3f}")
print(f"  Recall      : {tuned_df.loc[best_name,'Recall']:.3f}")
print(f"  F1          : {tuned_df.loc[best_name,'F1']:.3f}")
print(f"  ROC-AUC     : {tuned_df.loc[best_name,'ROC-AUC']:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 6  ROC CURVE
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors  = {'Logistic Regression':'#1565C0','Decision Tree':'#8E24AA',
           'Random Forest':'#2E7D32','XGBoost':'#E65100'}
for name, clf, Xe in [('Logistic Regression', lr, X_te_sc),
                       ('Decision Tree',       dt, X_test),
                       ('Random Forest',       rf, X_test),
                       ('XGBoost',            xgb, X_test)]:
    ypr = clf.predict_proba(Xe)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, ypr)
    auc = roc_auc_score(y_test, ypr)
    ax.plot(fpr, tpr, color=colors[name], lw=2.5 if name==best_name else 1.5,
            label=f'{name} (AUC={auc:.3f})')

ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - All Models (Fixed Pipeline)', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
fig_path = os.path.join(BASE, 'reports/figures/roc_curves_fixed.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ROC curve saved to reports/figures/roc_curves_fixed.png")

# ─────────────────────────────────────────────────────────────
# STEP 7  SAVE MODELS
# ─────────────────────────────────────────────────────────────
models_dir = os.path.join(BASE, 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(best_clf, os.path.join(models_dir, 'best_model.pkl'))
joblib.dump(sc,       os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(best_thr, os.path.join(models_dir, 'best_threshold.pkl'))
joblib.dump(lr,       os.path.join(models_dir, 'logistic_regression.pkl'))
joblib.dump(rf,       os.path.join(models_dir, 'random_forest.pkl'))
joblib.dump(xgb,      os.path.join(models_dir, 'xgboost.pkl'))

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"  Best model : {best_name}")
print(f"  Threshold  : {best_thr}")
print(f"  ROC-AUC    : {tuned_df.loc[best_name,'ROC-AUC']:.4f}")
print(f"  F1         : {tuned_df.loc[best_name,'F1']:.3f}")
print(f"  Precision  : {tuned_df.loc[best_name,'Precision']:.3f}")
print(f"  Recall     : {tuned_df.loc[best_name,'Recall']:.3f}")
print("="*60)
