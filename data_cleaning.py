import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Load Data ---
file_path = r"D:\XodeCerberus\XAI-CDSS-Project\diabetes\data\diabetes_uncleaned.csv"
df = pd.read_csv(file_path)

print("Original Data Info:")
print(df.info())
print(df.describe())

# --- 2. Columns where zero is invalid ---
zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zeros with NaN for these columns
df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)

# --- 3. Fill missing values ---
# Option 1: Fill with median (robust to outliers)
for col in zero_invalid_cols:
    df[col].fillna(df[col].median(), inplace=True)

# --- 4. Optional: Feature Scaling ---
scaler = StandardScaler()
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# --- 5. Save cleaned data ---
clean_file_path = r"D:\XodeCerberus\XAI-CDSS-Project\diabetes\data\diabetes_cleaned.csv"
df.to_csv(clean_file_path, index=False)

print(f"Cleaned data saved to {clean_file_path}")
print("First 5 rows of cleaned data:")
print(df.head())
