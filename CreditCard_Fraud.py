# =========================
# CREDIT CARD FRAUD DETECTION – DEBUGGED FULL CODE
# =========================

# Install required library (run once if needed)
# !pip install imbalanced-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# -------------------------
# LOAD DATASET
# -------------------------
df = pd.read_csv("creditcard.csv")  # change path if needed

assert 'Class' in df.columns, "❌ Class column missing"
assert not df.isnull().any().any(), "❌ Dataset contains NaN values"

# -------------------------
# BASIC EDA
# -------------------------
print(df['Class'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (Highly Imbalanced)")
plt.show()

# -------------------------
# FEATURE SCALING
# -------------------------
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------
# HANDLE IMBALANCE (MEMORY SAFE)
# -------------------------
smote_tomek = SMOTETomek(random_state=42)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)

print("After balancing:")
print(pd.Series(y_train_res).value_counts())

# -------------------------
# LOGISTIC REGRESSION
# -------------------------
lr = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    n_jobs=-1
)

lr.fit(X_train_res, y_train_res)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

if len(np.unique(y_test)) > 1:
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

# -------------------------
# RANDOM FOREST MODEL
# -------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_res, y_train_res)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest Report:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_test, y_pred_rf, labels=[0,1])

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Random Forest")
plt.show()

# -------------------------
# ROC CURVE
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# -------------------------
# PRECISION–RECALL CURVE
# -------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob_rf)

plt.figure(figsize=(6,4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.show()

# -------------------------
# HYPERPARAMETER TUNING (SAFE & FAST)
# -------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid,
    scoring='recall',
    cv=2,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_res, y_train_res)

print("\nBest Parameters from GridSearch:")
print(grid.best_params_)
#The code has been run and debugged several times to minimise errors