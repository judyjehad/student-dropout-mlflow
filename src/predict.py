"""
Student Dropout — Early Warning Model
predict.py

Loads the best registered model from MLflow Model Registry
and predicts dropout risk for a new student.
"""

import pickle
import numpy as np
import pandas as pd
import mlflow.sklearn

# ── 1. Setup ──────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ── 2. Load preprocessor and label encoder ───────────────────────────────────
with open("data/processed/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ── 3. Load best model from MLflow Model Registry ────────────────────────────
model = mlflow.sklearn.load_model("models:/student-dropout-classifier@Production")
print("✅ Model loaded from MLflow Model Registry\n")

# ── 4. Define predict function ────────────────────────────────────────────────
def predict(student: dict) -> dict:
    """
    Takes a raw student record (dict) and returns prediction + confidence.

    Parameters:
        student: dict with raw feature values (before preprocessing)

    Returns:
        dict with keys: prediction, confidence, probabilities
    """
    # Convert to DataFrame (preprocessor expects a DataFrame)
    df = pd.DataFrame([student])

    # Apply same preprocessing as training
    X = preprocessor.transform(df)

    # Predict class and probabilities
    prediction_encoded = model.predict(X)[0]

    # Get probabilities if model supports it
    try:
        proba = model.predict_proba(X)[0]
        confidence = round(float(proba.max()) * 100, 1)
        probabilities = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }
    except AttributeError:
        # pyfunc wrapper may not expose predict_proba directly
        confidence = None
        probabilities = None

    # Decode label back to string
    label = le.inverse_transform([int(prediction_encoded)])[0]

    return {
        "prediction":    label,
        "confidence":    confidence,
        "probabilities": probabilities,
    }


# ── 5. Example student ────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Example: at-risk student
    # (fees not paid, low sem 1 approvals, older age)
    at_risk_student = {
        "Marital status":                               1,
        "Application mode":                             17,
        "Application order":                            5,
        "Course":                                       171,
        "Daytime/evening attendance":                   1,
        "Previous qualification":                       1,
        "Previous qualification (grade)":               122.0,
        "Nacionality":                                  1,
        "Mother's qualification":                       19,
        "Father's qualification":                       12,
        "Mother's occupation":                          5,
        "Father's occupation":                          9,
        "Admission grade":                              127.3,
        "Displaced":                                    1,
        "Educational special needs":                    0,
        "Debtor":                                       1,   # has debt
        "Tuition fees up to date":                      0,   # fees NOT paid
        "Gender":                                       1,
        "Scholarship holder":                           0,   # no scholarship
        "Age at enrollment":                            30,  # older student
        "International":                                0,
        "Curricular units 1st sem (credited)":          0,
        "Curricular units 1st sem (enrolled)":          6,
        "Curricular units 1st sem (evaluations)":       6,
        "Curricular units 1st sem (approved)":          2,   # only 2/6 passed
        "Curricular units 1st sem (grade)":             8.5, # low grade
        "Curricular units 1st sem (without evaluations)": 0,
        "Unemployment rate":                            10.8,
        "Inflation rate":                               1.4,
        "GDP":                                          1.74,
    }

    # Example: low-risk student
    # (fees paid, high sem 1 approvals, scholarship)
    low_risk_student = {
        "Marital status":                               1,
        "Application mode":                             1,
        "Application order":                            1,
        "Course":                                       9254,
        "Daytime/evening attendance":                   1,
        "Previous qualification":                       1,
        "Previous qualification (grade)":               160.0,
        "Nacionality":                                  1,
        "Mother's qualification":                       1,
        "Father's qualification":                       3,
        "Mother's occupation":                          3,
        "Father's occupation":                          3,
        "Admission grade":                              142.5,
        "Displaced":                                    1,
        "Educational special needs":                    0,
        "Debtor":                                       0,   # no debt
        "Tuition fees up to date":                      1,   # fees paid
        "Gender":                                       1,
        "Scholarship holder":                           1,   # has scholarship
        "Age at enrollment":                            19,  # young student
        "International":                                0,
        "Curricular units 1st sem (credited)":          0,
        "Curricular units 1st sem (enrolled)":          6,
        "Curricular units 1st sem (evaluations)":       6,
        "Curricular units 1st sem (approved)":          6,   # all 6 passed
        "Curricular units 1st sem (grade)":             14.0, # high grade
        "Curricular units 1st sem (without evaluations)": 0,
        "Unemployment rate":                            13.9,
        "Inflation rate":                               -0.3,
        "GDP":                                          0.79,
    }

    print("=" * 50)
    print("Student 1 — At risk profile")
    print("=" * 50)
    result1 = predict(at_risk_student)
    print(f"  Prediction   : {result1['prediction']}")
    if result1['probabilities']:
        for cls, prob in result1['probabilities'].items():
            bar = "█" * int(prob / 5)
            print(f"  {cls:<12}: {prob:>5.1f}%  {bar}")

    print()
    print("=" * 50)
    print("Student 2 — Low risk profile")
    print("=" * 50)
    result2 = predict(low_risk_student)
    print(f"  Prediction   : {result2['prediction']}")
    if result2['probabilities']:
        for cls, prob in result2['probabilities'].items():
            bar = "█" * int(prob / 5)
            print(f"  {cls:<12}: {prob:>5.1f}%  {bar}")