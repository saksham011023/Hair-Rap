#!/usr/bin/env python
# coding: utf-8

# # 🤖 Salon No-Show – Model Comparison & Tuning
# Pipeline:
# 1. Train 4 baseline models (Logistic Regression, Decision Tree, Random Forest, XGBoost) with class balancing
# 2. Compare baseline performance metrics
# 3. **Threshold tuning** on the best model to maximise Recall while keeping Precision ≥ 40%
# 4. Final Side-by-side comparison

# ## Cell 1 – Import Libraries

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    precision_recall_curve
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
print('Libraries imported ✅')


# ## Cell 2 – Load Data and Split

# In[12]:


X = pd.read_csv('../data/processed/X.csv')
y = pd.read_csv('../data/processed/y.csv').squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_train)
X_te_sc = sc.transform(X_test)

print(f'Train : {X_train.shape[0]:,} rows  |  Test : {X_test.shape[0]:,} rows')
print(f'No-show rate  →  train: {y_train.mean():.2%}  |  test: {y_test.mean():.2%}')


# ## Cell 3 – Train Models with Class Weighting

# In[13]:


def get_metrics(clf, X_eval, y_eval, threshold=0.50):
    y_proba = clf.predict_proba(X_eval)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return {
        'Accuracy' : round(accuracy_score(y_eval, y_pred), 4),
        'Precision': round(precision_score(y_eval, y_pred, zero_division=0), 4),
        'Recall'   : round(recall_score(y_eval, y_pred), 4),
        'F1'       : round(f1_score(y_eval, y_pred), 4),
        'ROC-AUC'  : round(roc_auc_score(y_eval, y_proba), 4),
    }

lr  = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
dt  = DecisionTreeClassifier(class_weight='balanced', max_depth=8, random_state=42)
rf  = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             max_depth=10, random_state=42, n_jobs=-1)
spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=spw, eval_metric='logloss',
                    random_state=42, n_jobs=-1)

lr.fit(X_tr_sc, y_train); print('✅ Logistic Regression trained')
dt.fit(X_train, y_train); print('✅ Decision Tree trained')
rf.fit(X_train, y_train); print('✅ Random Forest trained')
xgb.fit(X_train, y_train); print('✅ XGBoost trained')


# ## Cell 4 – Base Model Metrics (threshold = 0.50)

# In[14]:


baseline_results = pd.DataFrame({
    'Logistic Regression': get_metrics(lr,  X_te_sc, y_test),
    'Decision Tree'      : get_metrics(dt,  X_test,  y_test),
    'Random Forest'      : get_metrics(rf,  X_test,  y_test),
    'XGBoost'            : get_metrics(xgb, X_test,  y_test),
}).T

print('BASELINE METRICS (threshold = 0.50)')
print('=' * 60)
print(baseline_results.to_string())


# ## Cell 5 – Threshold Tuning on Best Model
# We use our best model (highest AUC) and adjust the probability threshold to safely target the no-shows while maintaining a baseline acceptable precision (at least 40%).

# In[15]:


best_model_name = baseline_results['ROC-AUC'].idxmax()
best_clf        = {'Logistic Regression': lr, 'Decision Tree': dt, 'Random Forest': rf, 'XGBoost': xgb}[best_model_name]
best_X_eval     = X_te_sc if 'LR' in best_model_name or 'Logistic' in best_model_name else X_test
print(f'Best model by AUC: {best_model_name}')

y_proba_best = best_clf.predict_proba(best_X_eval)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_best)

thresh_df = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precisions[:-1],
    'Recall'   : recalls[:-1],
    'F1'       : 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
})

viable   = thresh_df[thresh_df['Precision'] >= 0.40]
best_row = viable.loc[viable['Recall'].idxmax()]
best_threshold = best_row['Threshold']

print(f'\nOptimal threshold (Recall-max, Precision >= 40%): {best_threshold:.3f}')
print(f'  Precision : {best_row["Precision"]:.3f}')
print(f'  Recall    : {best_row["Recall"]:.3f}')
print(f'  F1        : {best_row["F1"]:.3f}')


# ## Cell 6 – Precision-Recall Curve

# In[16]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(thresholds, precisions[:-1], '#1565C0', lw=2, label='Precision')
ax.plot(thresholds, recalls[:-1],    '#D32F2F', lw=2, label='Recall')
ax.axvline(best_threshold, color='green', linestyle='--', lw=1.5,
           label=f'Optimal = {best_threshold:.2f}')
ax.axhline(0.40, color='gray', linestyle=':', lw=1, label='Precision floor (0.40)')
ax.set_xlabel('Decision Threshold', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Precision & Recall vs Threshold', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)

ax = axes[1]
ax.plot(recalls[:-1], precisions[:-1], '#1565C0', lw=2)
ax.scatter(best_row['Recall'], best_row['Precision'],
           color='green', s=100, zorder=5,
           label=f'Optimal (Rec={best_row["Recall"]:.2f}, Pre={best_row["Precision"]:.2f})')
ax.axhline(0.40, color='gray', linestyle=':', lw=1, label='Precision floor')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)

plt.suptitle(f'{best_model_name} – Threshold Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ## Cell 7 – Final Metrics: Default vs Tuned Threshold

# In[17]:


tuned_metrics = get_metrics(best_clf, best_X_eval, y_test, threshold=best_threshold)
tuned_metrics['ROC-AUC'] = baseline_results.loc[best_model_name, 'ROC-AUC']

summary = pd.DataFrame({
    f'{best_model_name} (thr=0.50)'                  : baseline_results.loc[best_model_name],
    f'{best_model_name} (thr={best_threshold:.2f})': tuned_metrics,
}).T

print('FINAL SUMMARY')
print('=' * 70)
print(summary.to_string())
print(f'\nRecall improvement: {baseline_results.loc[best_model_name,"Recall"]:.3f} → {tuned_metrics["Recall"]:.3f} (+{tuned_metrics["Recall"]-baseline_results.loc[best_model_name,"Recall"]:.3f})')


# ## Cell 8 – ROC Curves: All Models

# In[18]:


fig, ax = plt.subplots(figsize=(8, 6))
plot_configs = [
    ('Logistic Regression',  lr,  X_te_sc,  '#1565C0'),
    ('Decision Tree',        dt,  X_test,   '#8E24AA'),
    ('Random Forest',        rf,  X_test,   '#2E7D32'),
    ('XGBoost',             xgb, X_test,   '#E65100'),
]
for name, clf, Xe, color in plot_configs:
    ypr = clf.predict_proba(Xe)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, ypr)
    auc = roc_auc_score(y_test, ypr)
    lw  = 2.5 if name == best_model_name else 1.5
    ax.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC={auc:.3f})')

ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves – All Models', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.spines[['top','right']].set_visible(False); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Cell 9 – Save Best Model

# In[19]:


joblib.dump(best_clf,       '../models/best_model.pkl')
joblib.dump(sc,             '../models/scaler.pkl')
joblib.dump(best_threshold, '../models/best_threshold.pkl')

print(f'Best model  : {best_model_name}')
print(f'Threshold   : {best_threshold:.3f}')
print(f'Final Recall: {tuned_metrics["Recall"]:.3f}')
print('\nSaved to ../models/')

