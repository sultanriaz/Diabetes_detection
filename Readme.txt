# 🩺 Diabetes Detection System with XAI
**Explainable Clinical Decision Support System (CDSS)**

This project is a machine learning-based tool designed to predict diabetes risk while providing transparent, interpretable explanations for its predictions using **Explainable AI (XAI)** techniques.

## 🚀 Features
* **Predictive Modeling:** Utilizes Random Forest and XGBoost classifiers for high-accuracy detection.
* **XAI Integration:** Provides feature importance and local explanations (SHAP/LIME) to help clinicians understand *why* a prediction was made.
* **Symptom Search:** Interactive tool for mapping patient symptoms to potential risk factors.
* **Treatment Recommendations:** Rule-based suggestions based on identified risk levels.

## 📁 Project Structure
* `app.py`: Main Flask application.
* `xai_explain.py`: Logic for generating model interpretations.
* `Models/`: Contains trained serialized models (`.joblib`, `.json`).
* `data_cleaning.py`: Preprocessing pipeline for medical datasets.

## 🛠️ Tech Stack
* **Backend:** Python (Flask)
* **ML Libraries:** Scikit-Learn, XGBoost
* **Explainability:** SHAP / LIME / Custom XAI logic
* **Frontend:** HTML5, CSS3 (Jinja2 Templates)

## 📋 How to Run
1. Clone the repo: `git clone https://github.com/sultanriazkhan/diabetes-detection-xai.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`
