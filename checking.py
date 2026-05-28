import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

rf = joblib.load("Models/diabetes_rf.joblib")
scaler = joblib.load("Models/scaler.joblib")

df = pd.read_csv("data/diabetes_cleaned.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_scaled = scaler.transform(X)
preds = rf.predict(X_scaled)
probs = rf.predict_proba(X_scaled)[:, 1]
cm = confusion_matrix(y, preds)
print(cm)
