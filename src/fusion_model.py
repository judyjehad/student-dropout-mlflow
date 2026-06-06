"""
Student Dropout — Early Warning Model
fusion_model.py

Advanced feature: model fusion / ensemble learning.

Builds two fusion models on top of the three base learners already used in
the project (Logistic Regression, Random Forest, XGBoost):

  1. Soft VotingClassifier   — averages predicted probabilities
  2. StackingClassifier      — Logistic Regression meta-learner

Each fusion model is logged to MLflow under the 'student-dropout-fusion'
experiment (params, metrics, confusion matrix image, model artifact).

Both are then compared against the current best model:
  Random Forest Grid Search — Accuracy 72.88%, F1 macro 0.6796.

If the best fusion model beats that F1 macro, it is registered as a new
version of 'student-dropout-classifier'.
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay, confusion_matrix
)
from xgboost import XGBClassifier

# Current best model to beat (Random Forest Grid Search).
BEST_REFERENCE = {
    "name":     "random_forest_grid_search",
    "accuracy": 0.7288,
    "f1_macro": 0.6796,
}

MODEL_NAME = "student-dropout-classifier"

# ── 1. Load preprocessed data ─────────────────────────────────────────────────
X_train = np.load("data/processed/X_train.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

n_classes = len(le.classes_)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"Classes: {list(le.classes_)}\n")

# ── 2. MLflow experiment setup ────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("student-dropout-fusion")

# ── 3. Define base learners ───────────────────────────────────────────────────
# Logistic Regression and Random Forest use class_weight="balanced".
# XGBoost has no class_weight argument, so it uses multiclass softprob settings
# (objective + num_class + mlogloss), consistent with train.py.
def make_base_estimators():
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    # Tuned Grid Search values from the 'student-dropout-tuning' experiment.
    rand_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    # Tuned Grid Search values from the 'student-dropout-tuning' experiment.
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=42,
    )
    return [
        ("logistic_regression", log_reg),
        ("random_forest",       rand_forest),
        ("xgboost",             xgb),
    ]

# ── 4. Define the two fusion models ───────────────────────────────────────────
fusion_models = {
    "voting_soft": {
        "model": VotingClassifier(
            estimators=make_base_estimators(),
            voting="soft",
            n_jobs=-1,
        ),
        "params": {
            "fusion_type":  "soft_voting",
            "base_models":  "logreg + rf + xgboost",
            "voting":       "soft",
        },
    },
    "stacking_logreg": {
        "model": StackingClassifier(
            estimators=make_base_estimators(),
            final_estimator=LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            stack_method="predict_proba",
            cv=5,
            n_jobs=-1,
        ),
        "params": {
            "fusion_type":   "stacking",
            "base_models":   "logreg + rf + xgboost",
            "meta_learner":  "logistic_regression",
            "cv":            5,
        },
    },
}

# ── 5. Train, evaluate, and log each fusion model ─────────────────────────────
results = {}

for model_name, config in fusion_models.items():
    print(f"Training {model_name}...")

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(config["params"])
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("category", "fusion")

        # Train
        model = config["model"]
        model.fit(X_train, y_train)

        # Evaluate
        y_pred      = model.predict(X_test)
        accuracy    = accuracy_score(y_test, y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro")
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
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        # Store for comparison
        run_id = mlflow.active_run().info.run_id
        results[model_name] = {
            "run_id":      run_id,
            "accuracy":    accuracy,
            "f1_macro":    f1_macro,
            "f1_weighted": f1_weighted,
        }

        print(f"  Accuracy    : {accuracy:.4f}")
        print(f"  F1 macro    : {f1_macro:.4f}")
        print(f"  F1 weighted : {f1_weighted:.4f}")
        print(f"  Run ID      : {run_id}\n")

    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("-" * 60)

# ── 6. Pick best fusion model (by F1 macro) ───────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1_macro"])
best      = results[best_name]

print()
print("=" * 60)
print(f"🏆 Best fusion model: {best_name}")
print(f"   F1 macro : {best['f1_macro']:.4f}")
print(f"   Accuracy : {best['accuracy']:.4f}")
print(f"   Run ID   : {best['run_id']}")
print("=" * 60)

# ── 7. Compare against the current best model ─────────────────────────────────
print("\nComparison vs current best model "
      f"({BEST_REFERENCE['name']}):")
print(f"{'model':<22}{'accuracy':>12}{'f1_macro':>12}")
print("-" * 46)
print(f"{BEST_REFERENCE['name']:<22}"
      f"{BEST_REFERENCE['accuracy']:>12.4f}{BEST_REFERENCE['f1_macro']:>12.4f}")
for name, r in results.items():
    print(f"{name:<22}{r['accuracy']:>12.4f}{r['f1_macro']:>12.4f}")

improvement = best["f1_macro"] - BEST_REFERENCE["f1_macro"]
print(f"\nBest fusion F1 macro vs reference: "
      f"{best['f1_macro']:.4f} - {BEST_REFERENCE['f1_macro']:.4f} "
      f"= {improvement:+.4f}")

# ── 8. Register if it improves on the current best F1 macro ───────────────────
if best["f1_macro"] > BEST_REFERENCE["f1_macro"]:
    print(f"\n✅ Fusion model improves F1 macro — registering new version.")

    model_uri = f"runs:/{best['run_id']}/{best_name}"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print(f"   Name    : {registered.name}")
    print(f"   Version : {registered.version}")
else:
    print(f"\nℹ️  No improvement over the current best F1 macro "
          f"({BEST_REFERENCE['f1_macro']:.4f}) — not registering.")
    print(f"   The current best model remains the production candidate.")

print(f"\nOpen MLflow UI to inspect the fusion runs:")
print(f"   mlflow ui")
