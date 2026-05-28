import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Paths
cleaned_path = r"D:\XodeCerberus\XAI-CDSS-Project\diabetes\data\diabetes_cleaned.csv"
models_dir = r"D:\XodeCerberus\XAI-CDSS-Project\diabetes\Models"

# Create models directory if not exists
os.makedirs(models_dir, exist_ok=True)

# --------------------------------------------------
# 1. Load Cleaned Data
# --------------------------------------------------
df = pd.read_csv(cleaned_path)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# --------------------------------------------------
# Feature Scaling (FIT ONLY ON TRAINING DATA)
# --------------------------------------------------
scaler = StandardScaler()

# --------------------------------------------------
# 2. Train/Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler on TRAIN only
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 3. Random Forest Model
# --------------------------------------------------
print("\n===== Training Random Forest =====")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("\n===== Random Forest Results =====")
print("Accuracy :", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall   :", recall_score(y_test, rf_preds))
print("F1-Score :", f1_score(y_test, rf_preds))
print("ROC-AUC  :", roc_auc_score(y_test, rf_probs))

# Save RF model
rf_path = os.path.join(models_dir, "diabetes_rf.joblib")
joblib.dump(rf, rf_path)
print("Random Forest saved to:", rf_path)

# --------------------------------------------------
# 4. XGBoost Model
# --------------------------------------------------
print("\n===== Training XGBoost =====")
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb.fit(X_train, y_train)

xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]

print("\n===== XGBoost Results =====")
print("Accuracy :", accuracy_score(y_test, xgb_preds))
print("Precision:", precision_score(y_test, xgb_preds))
print("Recall   :", recall_score(y_test, xgb_preds))
print("F1-Score :", f1_score(y_test, xgb_preds))
print("ROC-AUC  :", roc_auc_score(y_test, xgb_probs))

# Save XGB joblib for deployment
xgb_path = os.path.join(models_dir, "diabetes_xgb.joblib")
joblib.dump(xgb, xgb_path)
print("XGBoost saved to:", xgb_path)

# Save booster for SHAP (JSON format - more compatible)
booster = xgb.get_booster()
booster_path = os.path.join(models_dir, "xgb_booster.json")
booster.save_model(booster_path)
print("XGBoost booster saved to:", booster_path)

# Also save as .model format as backup
booster_model_path = os.path.join(models_dir, "xgb_booster.model")
booster.save_model(booster_model_path)
print("XGBoost booster (.model) saved to:", booster_model_path)

# --------------------------------------------------
# 5. Save .NPY feature arrays for deployment
# --------------------------------------------------
np.save(os.path.join(models_dir, "X_train.npy"), X_train)
np.save(os.path.join(models_dir, "X_test.npy"), X_test)
np.save(os.path.join(models_dir, "y_train.npy"), y_train)
np.save(os.path.join(models_dir, "y_test.npy"), y_test)
scaler_path = os.path.join(models_dir, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print("Scaler saved to:", scaler_path)


print("\n===== All files saved successfully! =====")
print("Saved .npy files for deployment.")
print("\nFeature names:")
print(list(X.columns))