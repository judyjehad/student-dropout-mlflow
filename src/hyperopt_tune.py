"""
Student Dropout — Early Warning Model
hyperopt_tune.py

Hyperparameter optimization using Hyperopt (Bayesian search).
Every evaluation is logged as a separate MLflow run.
Best model is registered to the Model Registry.
"""

import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

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
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("student-dropout-hyperopt")

# ── 3. Track best result across all runs ──────────────────────────────────────
best_result = {"f1_macro": 0, "run_id": None, "params": None, "model": None, "model_name": None}

# ── 4. Objective function — Random Forest ────────────────────────────────────
def rf_objective(params):
    with mlflow.start_run(run_name="random_forest"):
        mlflow.set_tag("model_type", "random_forest")
        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred      = model.predict(X_test)
        accuracy    = accuracy_score(y_test, y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1  = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1_macro").mean()

        mlflow.log_metric("accuracy",    accuracy)
        mlflow.log_metric("f1_macro",    f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("cv_f1_macro", cv_f1)
        mlflow.sklearn.log_model(model, artifact_path="random_forest")

        run_id = mlflow.active_run().info.run_id

    if f1_macro > best_result["f1_macro"]:
        best_result.update({
            "f1_macro":   f1_macro,
            "run_id":     run_id,
            "params":     params,
            "model":      model,
            "model_name": "random_forest",
        })

    print(f"  RF  n_est={int(params['n_estimators'])} depth={int(params['max_depth'])} "
          f"split={int(params['min_samples_split'])} → F1={f1_macro:.4f}")

    return {"loss": -f1_macro, "status": STATUS_OK}


# ── 5. Objective function — XGBoost ──────────────────────────────────────────
def xgb_objective(params):
    with mlflow.start_run(run_name="xgboost"):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.log_params(params)

        sample_weights = compute_sample_weight("balanced", y_train)

        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred      = model.predict(X_test)
        accuracy    = accuracy_score(y_test, y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1  = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1_macro").mean()

        mlflow.log_metric("accuracy",    accuracy)
        mlflow.log_metric("f1_macro",    f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("cv_f1_macro", cv_f1)
        mlflow.xgboost.log_model(model, artifact_path="xgboost")

        run_id = mlflow.active_run().info.run_id

    if f1_macro > best_result["f1_macro"]:
        best_result.update({
            "f1_macro":   f1_macro,
            "run_id":     run_id,
            "params":     params,
            "model":      model,
            "model_name": "xgboost",
        })

    print(f"  XGB n_est={int(params['n_estimators'])} depth={int(params['max_depth'])} "
          f"lr={params['learning_rate']:.3f} → F1={f1_macro:.4f}")

    return {"loss": -f1_macro, "status": STATUS_OK}


# ── 6. Define search spaces ───────────────────────────────────────────────────
rf_space = {
    "n_estimators":      scope.int(hp.quniform("n_estimators",      50, 400, 50)),
    "max_depth":         scope.int(hp.quniform("max_depth",          4,  20,  2)),
    "min_samples_split": scope.int(hp.quniform("min_samples_split",  2,  10,  1)),
}

xgb_space = {
    "n_estimators":  scope.int(hp.quniform("xgb_n_estimators", 50, 400, 50)),
    "max_depth":     scope.int(hp.quniform("xgb_max_depth",     3,  10,  1)),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
}

# ── 7. Run Hyperopt ───────────────────────────────────────────────────────────
MAX_EVALS = 40  # 20 per model

print("=" * 60)
print(f"Hyperopt — Random Forest ({MAX_EVALS // 2} evaluations)")
print("=" * 60)
rf_trials = Trials()
fmin(fn=rf_objective, space=rf_space, algo=tpe.suggest,
     max_evals=MAX_EVALS // 2, trials=rf_trials, verbose=False)

print()
print("=" * 60)
print(f"Hyperopt — XGBoost ({MAX_EVALS // 2} evaluations)")
print("=" * 60)
xgb_trials = Trials()
fmin(fn=xgb_objective, space=xgb_space, algo=tpe.suggest,
     max_evals=MAX_EVALS // 2, trials=xgb_trials, verbose=False)

# ── 8. Register best model ────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"🏆 Best model: {best_result['model_name']}")
print(f"   F1 macro : {best_result['f1_macro']:.4f}")
print(f"   Params   : {best_result['params']}")
print(f"   Run ID   : {best_result['run_id']}")
print("=" * 60)

model_uri  = f"runs:/{best_result['run_id']}/{best_result['model_name']}"
registered = mlflow.register_model(
    model_uri=model_uri,
    name="student-dropout-classifier",
)

print(f"\n✅ Registered to Model Registry")
print(f"   Name    : {registered.name}")
print(f"   Version : {registered.version}")
print(f"\nRun  →  mlflow ui  to compare Hyperopt runs with grid search runs.")