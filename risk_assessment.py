# risk_assessment.py

def get_risk_level(probability):
    """
    Determine risk level and lifestyle recommendations based on prediction probability.

    Args:
        probability (float): Probability output from the XGBoost model (0 to 1)

    Returns:
        dict: {
            "risk_level": "Low" | "Medium" | "High",
            "recommendations": list of strings
        }
    """
    if probability < 0.3:
        risk = "Low"
        recs = [
            "Maintain a balanced diet",
            "Exercise regularly (at least 30 min/day)",
            "Continue regular health checkups",
            "Monitor weight and BMI"
        ]
    elif probability < 0.7:
        risk = "Medium"
        recs = [
            "Adopt a low-sugar, high-fiber diet",
            "Increase physical activity (walking, jogging, or gym)",
            "Monitor blood glucose regularly",
            "Consult a healthcare professional if symptoms persist"
        ]
    else:
        risk = "High"
        recs = [
            "Seek medical advice immediately",
            "Follow a strict diabetes-friendly diet",
            "Monitor blood glucose multiple times a day",
            "Consider medication or insulin as prescribed",
            "Avoid sugary drinks and processed foods"
        ]

    return {"risk_level": risk, "recommendations": recs}
