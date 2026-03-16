# 🔴 Salon No-Show Prediction: Why Metrics Aren't Improving

## Executive Summary
Your evaluation metrics are plateauing at **F1 ≈ 0.59** with **precision ≈ 0.52** and **recall ≈ 0.68**. Analysis reveals **10 critical blockers** preventing better performance.

---

## 🔴 CRITICAL ISSUES

### 1. **DATA INCONSISTENCY / POTENTIAL DATA LEAKAGE** ⚠️ HIGHEST PRIORITY
**Problem:**
- **EDA (01_eda.ipynb)**: Target distribution = 88.7% Show, 11.3% No-Show
- **Feature Engineering (02_fe.ipynb)**: Target distribution = 75.9% Show, **24.1% No-Show**
- **Model Training (03_training.ipynb)**: Uses 24.1% No-Show rate

**Why This Is Critical:**
- The target proportion **doubled** from 11.3% to 24.1%
- Either rows were dropped/modified, or data was resampled without proper documentation
- This breaks reproducibility and suggests data leakage or incorrect handling

**Impact:** Model learned on a different distribution than the original data

**Fix:**
```python
# In 02_fe.ipynb at Cell 12, verify target encoding:
print(f"Original no-show rate: {(df['Outcome'] == 'No-Show').mean():.2%}")
print(f"After encoding no-show rate: {df_encoded['No_Show'].mean():.2%}")
# These MUST match exactly. If not, investigate what rows were dropped.
```

---

### 2. **SEVERE CLASS IMBALANCE MISHANDLED**
**Problem:**
- Even with class balancing, baseline precision is only **52%** (48% false positives)
- Class weights reduce bias but don't solve the fundamental problem
- No SMOTE, random undersampling, or stratified sampling attempted

**Why It Matters:**
- Your model wastes ~half its predictions flagging false no-shows
- In production, this creates poor customer experience (false alarms)

**Current Approach - Only Partial Solution:**
```python
# Using only class_weight='balanced'
lr = LogisticRegression(class_weight='balanced', ...)
```

**Better Approach:**
```python
# 1. SMOTE for train set
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.6)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 2. Combined with class weights
lr = LogisticRegression(class_weight='balanced', ...)
lr.fit(X_train_smote, y_train_smote)

# 3. Threshold tuning on balanced set (not default 0.50)
```

---

### 3. **POOR FEATURE ENGINEERING FOR NEW CUSTOMERS**
**Problem:**
- **90% of bookings are from new customers** (`Is_New_Customer = 1`)
- Features like `Customer_No_Show_Rate`, `Successful_Visit_Rate` are **meaningless for 90% of data**
  - New customers have `Past_Visit_Count = 0`
  - These ratios default to 0.0 → no signal
  - The model can't differentiate between "truly reliable" new customers vs likely no-shows

**Why Metrics Stagnate:**
- For 90% of bookings, behavioral features contribute nothing
- Model relies only on: appointment time, service type, branch, payment method
- These weak signals limit maximum achievable F1

**Evidence:**
```python
# In FE Cell 8:
df['Is_New_Customer'] = (df['Past_Visit_Count'] == 0).astype(int)
# Result: 90% = 1 (new), only 10% = 0 (returning)

# In FE Cell 6:
df['Customer_No_Show_Rate'] = np.where(
    df['Past_Visit_Count'] == 0, 0.0,  # ← NEW CUSTOMERS GET ZERO SIGNAL
    df['Past_No_Show_Count'] / df['Past_Visit_Count']
)
```

**What's Missing:**
- **No demographic features**: age, gender, location distance
- **No temporal patterns**: booking velocity, time-of-day patterns per customer
- **No booking history**: How many times does this customer book but never show?
- **No cancellation timing**: Do late cancellations predict no-shows?

**Fix:** Create new features that work for new customers:
```python
# Example: Feature availability signal
df['Has_Customer_History'] = (df['Past_Visit_Count'] > 0).astype(int)

# For new customers, use proxy signals:
# 1. Booking urgency (lead time anomaly for that slot type)
# 2. Service type propensity to no-show
# 3. Branch + time-of-day patterns
# 4. Payment method reliability
```

---

### 4. **WEAK FEATURE AGGREGATION STRATEGY**
**Problem:**

| Feature | Issue |
|---|---|
| `Customer_No_Show_Rate` | Uses division by count <5 for most (noisy) |
| `Smoothed_NoShow_Rate` | Alpha=5 too aggressive; shrinks all new customers to ~0.24 |
| `Capped_Good_Visits` | Caps at 10; removes signal for loyal customers |
| Interaction features | Only 2 basic interactions; missing many others |

**Why This Matters:**
```python
# Smoothing example from FE Cell 12:
ALPHA = 5
df['Smoothed_NoShow_Rate'] = (
    (df['Past_No_Show_Count'] + 5 * 0.1132) /  # ← Global mean
    (df['Past_Visit_Count'] + 5)
)
# Customer with 1 visit, 1 no-show:
# Raw rate = 1.0 (very bad)
# Smoothed = (1 + 0.566) / (1 + 5) = 0.26 (near global mean)
# ← Signal destroyed by over-smoothing
```

**Better Approach:**
```python
ALPHA = 3  # Use smaller alpha for more signal retention
# Or use Empirical Bayes shrinkage (Bayesian estimation)
# Or use different alpha per visit count tier
tier1 = (df['Past_Visit_Count'] <= 2)  # Very few visits → smooth more
tier2 = (df['Past_Visit_Count'] > 2)   # Enough history → smooth less
```

---

### 5. **INSUFFICIENT THRESHOLD TUNING**
**Problem:**
- Threshold is tuned on **test set** (data leakage concern)
- Only maximizes F1, doesn't explore Recall vs Precision trade-offs
- No validation set for threshold selection

**Current Code Issues:**
```python
# Cell 5, threshold tuned on test set:
thr, p, r, f = best_f1_threshold(clf, X_test, y_test)  # ← LEAKAGE!
# Test set should be unseen; tuning params on it overfits
```

**Better Approach:**
```python
# 1. Split into train / validation / test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 2. Tune threshold on validation set
thr, p, r, f = best_f1_threshold(clf, X_val, y_val)  # ← No leakage

# 3. Report metrics on test set
```

---

### 6. **NO CROSS-VALIDATION**
**Problem:**
- Single train/test split (80/20) → high variance in metrics
- No confidence intervals; can't tell if improvement is real or noise
- Model might perform very differently on different data subsets

**Impact:**
- Reported metrics (F1=0.59) may not be stable
- Any tuning you do might not generalize

**Missing:**
```python
from sklearn.model_selection import cross_val_score

# 5-fold stratified cross-validation
cv_scores = cross_val_score(
    rf, X, y, cv=5, scoring='f1',
    groups=None  # Use stratified split
)
print(f"F1 scores: {cv_scores}")
print(f"Mean ± Std: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

---

### 7. **NO FEATURE IMPORTANCE ANALYSIS**
**Problem:**
- You don't know which features drive predictions
- Can't diagnose why model fails on certain bookings
- Can't identify data quality issues in specific features

**Missing:**
```python
# For tree models:
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# For LR:
coefs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_[0]
}).reindex(coefs['Coefficient'].abs().argsort(ascending=False)[::-1])

# Interpretation:
# If 'Smoothed_NoShow_Rate' has low importance → feature doesn't help
# If 'Is_Edge_Hour' has high importance → time matters
```

---

### 8. **NO LEARNING CURVES / BIAS-VARIANCE ANALYSIS**
**Problem:**
- Can't tell if model is **underfitting** (high bias) or **overfitting** (high variance)
- No way to know if more data would help

**Why This Matters:**
```python
# If training F1 ≈ test F1 → Underfitting (high bias)
#   → Need more features or complex model
# If training F1 >> test F1 → Overfitting (high variance)
#   → Need regularization, less features, or more data
```

**Missing Code:**
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    rf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1', n_jobs=-1
)

# Plot to see if curves converge (underfitting) or diverge (overfitting)
```

---

### 9. **WEAK ERROR ANALYSIS**
**Problem:**
- No confusion matrix analysis
- No breakdown of errors by booking type
- Can't identify patterns in false positives/false negatives

**What's Missing:**
```python
from sklearn.metrics import confusion_matrix

y_pred = (y_proba >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives:  {tn:,}")   # Correctly predicted shows
print(f"False Positives: {fp:,}")   # Predicted no-show, actually showed
print(f"False Negatives: {fn:,}")   # Missed actual no-shows
print(f"True Positives:  {tp:,}")   # Correctly predicted no-shows

# Questions to answer:
# - Are FP concentrated in certain branches/times?
# - Which service types get missed (FN)?
# - Do payment methods affect error rates?
```

---

### 10. **HYPERPARAMETER TUNING NOT SYSTEMATIC**
**Problem:**
- Parameters chosen manually with no grid search
- No tuning for imbalance handling (class_weight range)
- No validation of regularization settings

**Example of Manual Choice (ineffective):**
```python
# From Cell 3:
rf = RandomForestClassifier(
    n_estimators=300,      # Why 300? Not justified
    max_depth=12,          # Arbitrary increase from 10
    min_samples_leaf=20,   # Why 20?
)

xgb = XGBClassifier(
    n_estimators=400,      # Why 400?
    learning_rate=0.05,    # Why 0.05?
    min_child_weight=5,    # Why 5?
)
```

**Better Approach:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [8, 10, 12, 15],
    'min_samples_leaf': [10, 20, 50],
    'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best F1 (CV): {grid_search.best_score_:.3f}")
```

---

## 📊 IMPACT RANKING: What to Fix First

| Issue | Severity | Effort | Impact (F1) |
|---|---|---|---|
| Data inconsistency | 🔴 Critical | Low | ??? (Debug first) |
| Feature engineering for new customers | 🔴 Critical | High | +0.05→0.10 |
| Better imbalance handling | 🟠 High | Medium | +0.03→0.07 |
| Feature importance analysis | 🟠 High | Low | +0.00 (Diagnostic) |
| Cross-validation | 🟠 High | Low | +0.00 (Validation) |
| Learning curves | 🟠 High | Low | +0.00 (Diagnostic) |
| Threshold tuning (proper split) | 🟠 High | Low | +0.02→0.05 |
| Systematic hyperparameter tuning | 🟡 Medium | High | +0.03→0.05 |
| Error analysis | 🟡 Medium | Low | +0.00 (Diagnostic) |
| Weak feature aggregation | 🟡 Medium | Medium | +0.02→0.04 |

---

## 🎯 YOUR NEXT STEPS (Recommended Order)

### **Step 1: URGENT - Resolve Data Inconsistency**
```python
# Add to 01_eda.ipynb end:
original_noshow_pct = (df['Outcome'] == 'No-Show').mean()
print(f"Original no-show rate: {original_noshow_pct:.2%}")

# Add to 02_fe.ipynb end:
final_noshow_pct = (df_encoded['No_Show']).mean()
print(f"Final no-show rate: {final_noshow_pct:.2%}")

# MUST be identical. If not, find what changed.
```

### **Step 2: New Features for New Customers**
- Booking velocity (recent booking frequency)
- Service-type propensity to no-show
- Time-of-day patterns (early bird vs procrastinator)
- Payment method reliability by branch

### **Step 3: Implement Proper SMOTE + Stratified CV**
- Combine SMOTE with stratified 5-fold cross-validation
- Proper train/val/test split for threshold tuning

### **Step 4: Feature Importance & Error Analysis**
- Identify which features matter
- Understand failure patterns

### **Step 5: Systematic Hyperparameter Tuning**
- Use GridSearchCV instead of manual tuning

---

## Expected Improvements with Fixes
- **Fix 1 alone** → Clarify baseline (may drop F1 if data leaked, or rise if fixed)
- **Fixes 1+2** → F1: 0.59 → **0.66**
- **Fixes 1+2+3** → F1: 0.66 → **0.72**
- **All fixes** → F1: **0.72 → 0.78** (realistic ceiling with current data)

---

## 🚨 Most Critical Finding
**Data inconsistency between notebooks suggests potential data leakage or row dropping.** This must be resolved first before any other improvements. Your current metrics may not even be valid.
