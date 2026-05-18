"""
Student Dropout — Early Warning Model
tune.py

Hyperparameter tuning for Random Forest and XGBoost.
Every parameter combination is logged as a separate MLflow run.
Best model is registered to the Model Registry.
"""

import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from itertools import product

# ── 1. Load preprocessed data ─────────────────────────────────────────────────
X_train = np.load("data/processed/X_train.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"Classes: {list(le.classes_)}\n")

# ── 2. MLflow setup ───────────────────────────────────────────────────────────
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("student-dropout-tuning")

# ── 3. Parameter grids ────────────────────────────────────────────────────────
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth":    [8, 10, 15],
    "min_samples_split": [2, 5],
}

xgb_param_grid = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
}

# ── 4. Helper: train, evaluate, log one run ───────────────────────────────────
def run_experiment(model, params, model_name, log_fn):
    with mlflow.start_run(run_name=f"{model_name}"):
        mlflow.set_tag("model_type", model_name)
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred      = model.predict(X_test)
        accuracy    = accuracy_score(y_test, y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        # Cross-validation score (more reliable than single split)
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1   = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring="f1_macro").mean()

        mlflow.log_metric("accuracy",    accuracy)
        mlflow.log_metric("f1_macro",    f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("cv_f1_macro", cv_f1)

        log_fn(model, artifact_path=model_name)

        run_id = mlflow.active_run().info.run_id

    return {
        "run_id":      run_id,
        "model_name":  model_name,
        "params":      params,
        "f1_macro":    f1_macro,
        "cv_f1_macro": cv_f1,
        "accuracy":    accuracy,
        "model":       model,
    }

# ── 5. Tune Random Forest ─────────────────────────────────────────────────────
print("=" * 60)
print("Tuning Random Forest...")
print("=" * 60)

rf_results = []
rf_keys    = list(rf_param_grid.keys())
rf_combos  = list(product(*rf_param_grid.values()))
total_rf   = len(rf_combos)

for i, combo in enumerate(rf_combos, 1):
    params = dict(zip(rf_keys, combo))
    print(f"  RF run {i}/{total_rf}: {params}")

    model = RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    result = run_experiment(model, params, "random_forest", mlflow.sklearn.log_model)
    rf_results.append(result)
    print(f"    F1 macro: {result['f1_macro']:.4f}  CV F1: {result['cv_f1_macro']:.4f}")

# ── 6. Tune XGBoost ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Tuning XGBoost...")
print("=" * 60)

xgb_results = []
xgb_keys    = list(xgb_param_grid.keys())
xgb_combos  = list(product(*xgb_param_grid.values()))
total_xgb   = len(xgb_combos)

# Compute sample weights for class balancing
sample_weights = compute_sample_weight("balanced", y_train)

for i, combo in enumerate(xgb_combos, 1):
    params = dict(zip(xgb_keys, combo))
    print(f"  XGB run {i}/{total_xgb}: {params}")

    model = XGBClassifier(
        **params,
        eval_metric="mlogloss",
        random_state=42,
    )

    # Override fit to include sample weights
    with mlflow.start_run(run_name="xgboost"):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.log_params(params)

        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred      = model.predict(X_test)
        accuracy    = accuracy_score(y_test, y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1   = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring="f1_macro").mean()

        mlflow.log_metric("accuracy",    accuracy)
        mlflow.log_metric("f1_macro",    f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("cv_f1_macro", cv_f1)

        mlflow.xgboost.log_model(model, artifact_path="xgboost")
        run_id = mlflow.active_run().info.run_id

    xgb_results.append({
        "run_id":      run_id,
        "model_name":  "xgboost",
        "params":      params,
        "f1_macro":    f1_macro,
        "cv_f1_macro": cv_f1,
        "accuracy":    accuracy,
        "model":       model,
    })
    print(f"    F1 macro: {f1_macro:.4f}  CV F1: {cv_f1:.4f}")

# ── 7. Pick overall best model ────────────────────────────────────────────────
all_results = rf_results + xgb_results
best        = max(all_results, key=lambda r: r["f1_macro"])

print()
print("=" * 60)
print(f"🏆 Best model: {best['model_name']}")
print(f"   Params   : {best['params']}")
print(f"   F1 macro : {best['f1_macro']:.4f}")
print(f"   CV F1    : {best['cv_f1_macro']:.4f}")
print(f"   Accuracy : {best['accuracy']:.4f}")
print(f"   Run ID   : {best['run_id']}")
print("=" * 60)

# ── 8. Register best tuned model ──────────────────────────────────────────────
artifact_path = best["model_name"]
model_uri     = f"runs:/{best['run_id']}/{artifact_path}"

registered = mlflow.register_model(
    model_uri=model_uri,
    name="student-dropout-classifier",
)

print(f"\n✅ Registered to Model Registry")
print(f"   Name    : {registered.name}")
print(f"   Version : {registered.version}")
print(f"\nOpen MLflow UI to compare all runs:")
print(f"   mlflow ui")