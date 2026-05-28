import pandas as pd
import shap
import xgboost as xgb

# Load cleaned data
data_path = r"D:/XodeCerberus/XAI-CDSS-Project/diabetes/data/diabetes_cleaned.csv"
model_path = r"D:/XodeCerberus/XAI-CDSS-Project/diabetes/Models/xgb_booster.json"

df = pd.read_csv(data_path)
X = df.drop("Outcome", axis=1)

# Load booster correctly
booster = xgb.Booster()
booster.load_model(model_path)

# SHAP TreeExplainer
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X)

# Visuals
shap.summary_plot(shap_values, X)
shap.dependence_plot("Glucose", shap_values, X)
