"""
Student Dropout — Early Warning Model
serve_model.py

FastAPI server that exposes the trained model as a REST API.

Endpoints:
  GET  /            → health check
  GET  /model       → model info
  POST /predict     → predict dropout risk for a student
  GET  /monitoring  → live prediction stats + drift alert
  GET  /alerts      → dedicated alerts endpoint
  POST /feedback    → submit actual outcome for a prediction
"""

import pickle
import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.monitoring import log_prediction, log_feedback, get_stats, get_alerts

# ── 1. Setup ──────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Early warning system — predicts dropout risk after semester 1",
    version="1.0.0",
)

# ── 2. Load model and preprocessor at startup ─────────────────────────────────
print("Loading model from MLflow Model Registry...")

with open("data/processed/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

model = mlflow.sklearn.load_model("models:/student-dropout-classifier@Production")
print("✅ Model ready\n")

# ── 3. Request schemas ────────────────────────────────────────────────────────
class StudentFeatures(BaseModel):
    marital_status:                            int
    application_mode:                          int
    application_order:                         int
    course:                                    int
    daytime_evening_attendance:                int
    previous_qualification:                    int
    previous_qualification_grade:              float
    nacionality:                               int
    mothers_qualification:                     int
    fathers_qualification:                     int
    mothers_occupation:                        int
    fathers_occupation:                        int
    admission_grade:                           float
    displaced:                                 int
    educational_special_needs:                 int
    debtor:                                    int
    tuition_fees_up_to_date:                   int
    gender:                                    int
    scholarship_holder:                        int
    age_at_enrollment:                         int
    international:                             int
    curricular_units_1st_sem_credited:         int
    curricular_units_1st_sem_enrolled:         int
    curricular_units_1st_sem_evaluations:      int
    curricular_units_1st_sem_approved:         int
    curricular_units_1st_sem_grade:            float
    curricular_units_1st_sem_without_eval:     int
    unemployment_rate:                         float
    inflation_rate:                            float
    gdp:                                       float

class FeedbackRequest(BaseModel):
    prediction_id:  int
    actual_outcome: str  # "Dropout", "Enrolled", or "Graduate"

# ── 4. Helper: map API fields → original dataset column names ─────────────────
def to_dataframe(s: StudentFeatures) -> pd.DataFrame:
    return pd.DataFrame([{
        "Marital status":                                  s.marital_status,
        "Application mode":                                s.application_mode,
        "Application order":                               s.application_order,
        "Course":                                          s.course,
        "Daytime/evening attendance":                      s.daytime_evening_attendance,
        "Previous qualification":                          s.previous_qualification,
        "Previous qualification (grade)":                  s.previous_qualification_grade,
        "Nacionality":                                     s.nacionality,
        "Mother's qualification":                          s.mothers_qualification,
        "Father's qualification":                          s.fathers_qualification,
        "Mother's occupation":                             s.mothers_occupation,
        "Father's occupation":                             s.fathers_occupation,
        "Admission grade":                                 s.admission_grade,
        "Displaced":                                       s.displaced,
        "Educational special needs":                       s.educational_special_needs,
        "Debtor":                                          s.debtor,
        "Tuition fees up to date":                         s.tuition_fees_up_to_date,
        "Gender":                                          s.gender,
        "Scholarship holder":                              s.scholarship_holder,
        "Age at enrollment":                               s.age_at_enrollment,
        "International":                                   s.international,
        "Curricular units 1st sem (credited)":             s.curricular_units_1st_sem_credited,
        "Curricular units 1st sem (enrolled)":             s.curricular_units_1st_sem_enrolled,
        "Curricular units 1st sem (evaluations)":          s.curricular_units_1st_sem_evaluations,
        "Curricular units 1st sem (approved)":             s.curricular_units_1st_sem_approved,
        "Curricular units 1st sem (grade)":                s.curricular_units_1st_sem_grade,
        "Curricular units 1st sem (without evaluations)":  s.curricular_units_1st_sem_without_eval,
        "Unemployment rate":                               s.unemployment_rate,
        "Inflation rate":                                  s.inflation_rate,
        "GDP":                                             s.gdp,
    }])

# ── 5. Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Student Dropout Prediction API is running"}


@app.get("/model")
def model_info():
    return {
        "model_name":    "student-dropout-classifier",
        "alias":         "Production",
        "classes":       list(le.classes_),
        "features_used": "semester 1 only (early-warning)",
    }


@app.post("/predict")
def predict(student: StudentFeatures):
    try:
        df = to_dataframe(student)
        X  = preprocessor.transform(df)

        prediction_encoded = model.predict(X)[0]
        proba              = model.predict_proba(X)[0]

        label         = le.inverse_transform([int(prediction_encoded)])[0]
        confidence    = round(float(proba.max()) * 100, 1)
        probabilities = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }

        result = {
            "prediction":    label,
            "confidence":    confidence,
            "probabilities": probabilities,
        }

        # Log prediction and get its ID
        prediction_id = log_prediction(student.model_dump(), result)
        result["prediction_id"] = prediction_id

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring")
def monitoring():
    return get_stats()


@app.get("/alerts")
def alerts(limit: int = 20):
    return get_alerts(limit=limit)


@app.post("/feedback")
def feedback(body: FeedbackRequest):
    valid_classes = ["Dropout", "Enrolled", "Graduate"]
    if body.actual_outcome not in valid_classes:
        raise HTTPException(
            status_code=400,
            detail=f"actual_outcome must be one of {valid_classes}"
        )
    result = log_feedback(body.prediction_id, body.actual_outcome)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── 6. Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serve_model:app", host="0.0.0.0", port=8000, reload=True)