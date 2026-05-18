"""
Student Dropout — Early Warning Model
train.py

Trains 3 models and logs every run to MLflow:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost

Then registers the best model to the MLflow Model Registry.
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay, confusion_matrix
)
from xgboost import XGBClassifier

# ── 1. Load preprocessed data ─────────────────────────────────────────────────
X_train = np.load("data/processed/X_train.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"Classes: {list(le.classes_)}\n")

# ── 2. MLflow experiment setup ────────────────────────────────────────────────
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("student-dropout-prediction")

# ── 3. Define models to train ─────────────────────────────────────────────────
models = {
    "logistic_regression": {
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "params": {
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "log_fn": mlflow.sklearn.log_model,
    },
    "random_forest": {
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "class_weight": "balanced",
        },
        "log_fn": mlflow.sklearn.log_model,
    },
    "xgboost": {
        # Note: no class balancing applied here intentionally.
        # Future option: compute sample_weight via compute_sample_weight
        # and pass to model.fit() — then compare fairness vs performance.
        "model": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
        ),
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
        "log_fn": mlflow.xgboost.log_model,
    },
}

# ── 4. Train, evaluate, and log each model ────────────────────────────────────
results = {}

for model_name, config in models.items():
    print(f"Training {model_name}...")

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(config["params"])
        mlflow.set_tag("model_type", model_name)

        # Train
        model = config["model"]
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        # Log metrics
        mlflow.log_metric("accuracy",    accuracy)
        mlflow.log_metric("f1_macro",    f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # Save and log confusion matrix as image artifact
        os.makedirs("outputs", exist_ok=True)
        cm_path = f"outputs/confusion_matrix_{model_name}.png"

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=le.classes_
        ).plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {model_name}")
        fig.tight_layout()
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)

        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # Log full classification report as text artifact
        mlflow.log_text(
            classification_report(y_test, y_pred, target_names=le.classes_),
            f"classification_report_{model_name}.txt"
        )

        # Log model artifact
        config["log_fn"](model, artifact_path=model_name)

        # Store for comparison
        run_id = mlflow.active_run().info.run_id
        results[model_name] = {
            "run_id":      run_id,
            "accuracy":    accuracy,
            "f1_macro":    f1_macro,
            "f1_weighted": f1_weighted,
            "model":       model,
        }

        print(f"  Accuracy    : {accuracy:.4f}")
        print(f"  F1 macro    : {f1_macro:.4f}")
        print(f"  F1 weighted : {f1_weighted:.4f}")
        print(f"  Run ID      : {run_id}\n")

    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("-" * 60)

# ── 5. Pick best model (by F1 macro) ─────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1_macro"])
best      = results[best_name]

print(f"\n🏆 Best model: {best_name}")
print(f"   F1 macro : {best['f1_macro']:.4f}")
print(f"   Accuracy : {best['accuracy']:.4f}")
print(f"   Run ID   : {best['run_id']}")

# ── 6. Register best model to MLflow Model Registry ──────────────────────────
model_uri = f"runs:/{best['run_id']}/{best_name}"

registered = mlflow.register_model(
    model_uri=model_uri,
    name="student-dropout-classifier",
)

print(f"\n✅ Registered to Model Registry")
print(f"   Name    : {registered.name}")
print(f"   Version : {registered.version}")
print(f"\nRun  →  mlflow ui  to explore all runs in your browser.")