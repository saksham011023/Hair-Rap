import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

X = pd.read_csv('data/processed/X.csv')
y = pd.read_csv('data/processed/y.csv').squeeze()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
Xtr_sc = sc.fit_transform(X_train)
Xte_sc = sc.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
Xtr_sm_sc, ytr_sm = smote.fit_resample(Xtr_sc, y_train)
Xtr_sm_raw, _     = smote.fit_resample(X_train, y_train)

print(f'Before SMOTE: {len(y_train):,}  No-show: {y_train.mean():.2%}')
print(f'After  SMOTE: {len(ytr_sm):,}   No-show: {ytr_sm.mean():.2%}')

def show(n, clf, Xe, ye, thr=0.50):
    ypr = clf.predict_proba(Xe)[:,1]
    yp  = (ypr >= thr).astype(int)
    print(f'{n:28s}  Acc={accuracy_score(ye,yp):.3f}  Pre={precision_score(ye,yp,zero_division=0):.3f}  '
          f'Rec={recall_score(ye,yp):.3f}  F1={f1_score(ye,yp):.3f}  AUC={roc_auc_score(ye,ypr):.3f}')

print('\nBASELINE:')
lr  = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
rf  = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, random_state=42, n_jobs=-1)
spw = (y_train==0).sum() / (y_train==1).sum()
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=spw, eval_metric='logloss', random_state=42, n_jobs=-1)
lr.fit(Xtr_sc, y_train);   show('LR  baseline', lr,  Xte_sc, y_test)
rf.fit(X_train, y_train);  show('RF  baseline', rf,  X_test, y_test)
xgb.fit(X_train, y_train); show('XGB baseline', xgb, X_test, y_test)

print('\nSMOTE:')
lr_s  = LogisticRegression(max_iter=1000, random_state=42)
rf_s  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
xgb_s = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       scale_pos_weight=1, eval_metric='logloss', random_state=42, n_jobs=-1)
lr_s.fit(Xtr_sm_sc, ytr_sm);   show('LR  + SMOTE', lr_s,  Xte_sc, y_test)
rf_s.fit(Xtr_sm_raw, ytr_sm);  show('RF  + SMOTE', rf_s,  X_test, y_test)
xgb_s.fit(Xtr_sm_raw, ytr_sm); show('XGB + SMOTE', xgb_s, X_test, y_test)

# Threshold tuning on LR+SMOTE
proba = lr_s.predict_proba(Xte_sc)[:,1]
precs, recs, thrs = precision_recall_curve(y_test, proba)
viable = [(t, p, r) for t, p, r in zip(thrs, precs, recs) if p >= 0.40]
best_t, best_p, best_r = max(viable, key=lambda x: x[2])
f1 = 2*best_p*best_r/(best_p+best_r)
auc = roc_auc_score(y_test, proba)
print(f'\nTuned LR+SMOTE (thr={best_t:.2f}):')
print(f'  Precision={best_p:.3f}  Recall={best_r:.3f}  F1={f1:.3f}  AUC={auc:.3f}')
