# Student Dropout — Early Warning Model

A machine learning pipeline that predicts whether a student will **drop out**, remain **enrolled**, or **graduate**, using only **semester 1** data. The project uses [MLflow](https://mlflow.org/) for experiment tracking, model registry, and deployment, plus a **FastAPI** REST API with live prediction monitoring.

## Problem

Universities need early signals of dropout risk. This model acts as an early-warning system: after a student completes their first semester, you can score them before second-semester outcomes are known.

**Target classes:** `Dropout`, `Enrolled`, `Graduate`

## Dataset

- **Source:** [UCI Student Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (4,424 students)
- **File:** `data/students.csv` (semicolon-separated)
- **Design choice:** All **2nd-semester curricular unit** columns are dropped so the model only uses information available after semester 1.

## Architecture

```
data/students.csv
       │
       ▼
  preprocess.py  ──►  data/processed/  (arrays + preprocessor.pkl)
       │
       ├──► train.py      ──► MLflow experiment: student-dropout-prediction
       │                         (Logistic Regression, Random Forest, XGBoost)
       │
       └──► tune.py       ──► MLflow experiment: student-dropout-tuning
                                 (grid search on RF + XGBoost)
       │
       ▼
  Model Registry: student-dropout-classifier
       │
       ├── set_stage.py   (Production / Archived aliases)
       ├── predict.py     (CLI batch examples)
       └── serve_model.py (FastAPI on :8000)
                │
                └── monitoring.py ──► data/monitoring.db
```

## Project structure

```
student-dropout-mlflow/
├── data/
│   ├── students.csv              # Raw dataset
│   ├── processed/                # Preprocessed train/test + pickles
│   └── monitoring.db             # SQLite log of API predictions
├── src/
│   ├── preprocess.py             # Feature engineering & train/test split
│   ├── train.py                  # Train 3 baselines, register best
│   ├── tune.py                   # Hyperparameter grid search, register best
│   ├── set_stage.py              # Set Production / Archived model aliases
│   ├── predict.py                # Load Production model, example predictions
│   ├── serve_model.py            # FastAPI REST API
│   └── monitoring.py             # SQLite logging + drift alerts
├── docs/screenshots/             # README screenshots (confusion matrices + UI captures)
├── mlruns/                       # MLflow tracking & model registry (local)
├── outputs/                      # Confusion matrix PNGs from training
└── requirements.txt
```

## Setup

```bash
# Clone or cd into the project
cd student-dropout-mlflow

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn   # API server (used by serve_model.py)
```

Run all commands from the **project root** so relative paths (`data/`, `mlruns/`) resolve correctly.

## End-to-end workflow

### 1. Preprocess

```bash
python src/preprocess.py
```

- Fixes a hidden tab character in the `Daytime/evening attendance` column name
- Drops semester 2 features (early-warning constraint)
- Encodes target with `LabelEncoder` (alphabetical: Dropout=0, Enrolled=1, Graduate=2)
- Applies `ColumnTransformer`:
  - **Categorical** → `OneHotEncoder` (drop first)
  - **Numerical** → `StandardScaler`
  - **Binary** (0/1) → passthrough
- 80/20 stratified train/test split
- Saves `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `preprocessor.pkl`, `label_encoder.pkl`

### 2. Train baseline models

```bash
python src/train.py
```

Trains three models, each logged as a separate MLflow run:

| Model | Notes |
|-------|--------|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest | 200 trees, `max_depth=10`, balanced |
| XGBoost | No class balancing (noted for future fairness comparison) |

For each run, MLflow logs parameters, accuracy, F1 (macro & weighted), confusion matrix image, classification report, and the model artifact.

The **best model by F1 macro** on the test set is registered as `student-dropout-classifier` **version 1**.

### 3. Hyperparameter tuning

```bash
python src/tune.py
```

Grid search logged to experiment `student-dropout-tuning`:

- **Random Forest:** `n_estimators` × `max_depth` × `min_samples_split` (18 runs)
- **XGBoost:** `n_estimators` × `max_depth` × `learning_rate` (27 runs), with balanced `sample_weight`

Each run also logs **5-fold CV F1 macro**. The overall best model (by test F1 macro) is registered as **version 2**.

### 4. Promote models in the registry

```bash
python src/set_stage.py
```

Sets MLflow model aliases:

- **Version 2** → `Production` (tuned model)
- **Version 1** → `Archived` (baseline from `train.py`)

`predict.py` and `serve_model.py` load: `models:/student-dropout-classifier@Production`

### 5. Explore experiments in the UI

```bash
mlflow ui
```

Open http://127.0.0.1:5000 to compare runs, metrics, and artifacts.

### 6. CLI predictions

```bash
python src/predict.py
```

Runs two example students (at-risk vs low-risk) and prints class probabilities.

### 7. Serve the REST API

```bash
uvicorn src.serve_model:app --reload
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/model` | Model name, version, classes |
| `POST` | `/predict` | Predict dropout risk for one student |
| `GET` | `/monitoring` | Live stats + drift alert |
| — | `/docs` | Swagger UI (interactive testing) |

**Example `POST /predict` body** (snake_case field names):

```json
{
  "marital_status": 1,
  "application_mode": 17,
  "application_order": 5,
  "course": 171,
  "daytime_evening_attendance": 1,
  "previous_qualification": 1,
  "previous_qualification_grade": 122.0,
  "nacionality": 1,
  "mothers_qualification": 19,
  "fathers_qualification": 12,
  "mothers_occupation": 5,
  "fathers_occupation": 9,
  "admission_grade": 127.3,
  "displaced": 1,
  "educational_special_needs": 0,
  "debtor": 1,
  "tuition_fees_up_to_date": 0,
  "gender": 1,
  "scholarship_holder": 0,
  "age_at_enrollment": 30,
  "international": 0,
  "curricular_units_1st_sem_credited": 0,
  "curricular_units_1st_sem_enrolled": 6,
  "curricular_units_1st_sem_evaluations": 6,
  "curricular_units_1st_sem_approved": 2,
  "curricular_units_1st_sem_grade": 8.5,
  "curricular_units_1st_sem_without_eval": 0,
  "unemployment_rate": 10.8,
  "inflation_rate": 1.4,
  "gdp": 1.74
}
```

**Example response:**

```json
{
  "prediction": "Dropout",
  "confidence": 72.3,
  "probabilities": {
    "Dropout": 72.3,
    "Enrolled": 18.1,
    "Graduate": 9.6
  }
}
```

## Final results

All metrics are on the **held-out test set** (20% stratified split, 885 students). Models are ranked by **F1 macro** (primary metric for imbalanced classes).

### Baseline models (`train.py`)

| Model | Accuracy | F1 Macro | F1 Weighted | Registry |
|-------|----------|----------|-------------|----------|
| Random Forest | 72.09% | **66.67%** | 72.50% | v1 (Archived) |
| Logistic Regression | 69.60% | 65.45% | 70.89% | — |
| XGBoost | 72.43% | 64.71% | 71.50% | — |

### Tuned models (`tune.py`) — top 3 of 45 runs

| Model | Hyperparameters | Accuracy | F1 Macro | CV F1 Macro | Registry |
|-------|-----------------|----------|----------|-------------|----------|
| **Random Forest** | `n_estimators=300`, `max_depth=10`, `min_samples_split=2` | **72.88%** | **67.96%** | 68.07% | **v2 (Production)** |
| XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.05` | 71.98% | 67.83% | 67.63% | — |
| XGBoost | `n_estimators=100`, `max_depth=6`, `learning_rate=0.2` | 72.09% | 67.46% | 67.46% | — |

### Production model summary

| | |
|---|---|
| **Deployed model** | Tuned Random Forest |
| **Registry** | `student-dropout-classifier` v2 → `@Production` |
| **Test F1 macro** | 67.96% (+1.29 pp vs baseline RF) |
| **Test accuracy** | 72.88% (+0.79 pp vs baseline RF) |
| **5-fold CV F1 macro** | 68.07% |

## Screenshots

### Confusion matrices (baseline training)

Generated by `train.py` and saved to `outputs/` and `docs/screenshots/`.

| Logistic Regression | Random Forest | XGBoost |
|---------------------|---------------|---------|
| ![LR confusion matrix](docs/screenshots/confusion_matrix_logistic_regression.png) | ![RF confusion matrix](docs/screenshots/confusion_matrix_random_forest.png) | ![XGB confusion matrix](docs/screenshots/confusion_matrix_xgboost.png) |

### MLflow experiment tracking

Run `mlflow ui` and open http://127.0.0.1:5000.

| Experiments comparison | Model Registry |
|------------------------|----------------|
| ![MLflow experiments](docs/screenshots/mlflow-experiments.png) | ![MLflow registry](docs/screenshots/mlflow-registry.png) |

### FastAPI prediction API

Run `uvicorn src.serve_model:app --reload` and open http://127.0.0.1:8000/docs.

| Swagger UI | Sample prediction |
|------------|-------------------|
| ![FastAPI docs](docs/screenshots/fastapi-docs.png) | ![API predict response](docs/screenshots/api-predict.png) |

### Live monitoring

Call `GET /monitoring` after a few `POST /predict` requests.

![Monitoring dashboard](docs/screenshots/monitoring.png)

## Monitoring

Every `POST /predict` call is logged to `data/monitoring.db` via `monitoring.py`.

`GET /monitoring` returns:

- Total prediction count
- Average confidence (all time and last 20 predictions)
- Prediction class distribution
- Recent prediction history
- **Drift alert** if the rolling average confidence drops below **60%**

## MLflow details

| Item | Value |
|------|--------|
| Tracking URI | `file:./mlruns` |
| Experiments | `student-dropout-prediction`, `student-dropout-tuning` |
| Registered model | `student-dropout-classifier` |
| Production alias | Version 2 (tuned) |
| Archived alias | Version 1 (baseline) |
| Selection metric | F1 macro (test set) |

## Dependencies

Core packages (`requirements.txt`):

- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `mlflow`, `matplotlib`, `seaborn`, `jupyter`, `openpyxl`

Additional packages for the API:

- `fastapi`, `uvicorn`, `pydantic` (installed with FastAPI)

## Notes

- **Early warning only:** Do not include semester 2 features at inference time; the preprocessor and training pipeline exclude them by design.
- **Class imbalance:** Logistic Regression and Random Forest use balanced weights; XGBoost tuning uses `sample_weight`. Compare fairness vs performance before production use.
- **MLflow file store:** The local `mlruns/` backend works for development; MLflow may warn about migrating to a database backend for long-term use.

## License

Dataset: see [UCI ML Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) for citation and terms.
