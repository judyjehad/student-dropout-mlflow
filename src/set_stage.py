"""
Student Dropout — Early Warning Model
set_stage.py

Assigns registry aliases by deriving versions from the registry instead of
hardcoding version numbers. Each registered version is mapped back to the
experiment that produced it, and the best version (by test f1_macro) from each
experiment is promoted:

  student-dropout-tuning      (grid search)   → Production
  student-dropout-hyperopt    (Bayesian/TPE)  → Staging
  student-dropout-prediction  (baseline)      → Archived

This is robust to re-runs and shifting version numbers.
"""

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "student-dropout-classifier"

# experiment that produced a version  →  alias to assign
EXPERIMENT_TO_ALIAS = {
    "student-dropout-tuning":     "Production",
    "student-dropout-hyperopt":   "Staging",
    "student-dropout-prediction": "Archived",
}


def best_version_per_experiment():
    """Return {experiment_name: ModelVersion} picking the highest f1_macro."""
    best = {}  # experiment_name -> (f1_macro, ModelVersion)

    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if not mv.run_id:
            continue
        try:
            run = client.get_run(mv.run_id)
        except Exception:
            # source run was deleted — skip this version
            continue

        exp_name = client.get_experiment(run.info.experiment_id).name
        if exp_name not in EXPERIMENT_TO_ALIAS:
            continue

        f1 = run.data.metrics.get("f1_macro", float("-inf"))
        current = best.get(exp_name)
        if current is None or f1 > current[0]:
            best[exp_name] = (f1, mv)

    return {name: mv for name, (_f1, mv) in best.items()}


def main():
    selected = best_version_per_experiment()

    if not selected:
        raise SystemExit(
            f"No registered versions of '{MODEL_NAME}' found for the known "
            f"experiments. Run train.py, tune.py and hyperopt_tune.py first."
        )

    for exp_name, alias in EXPERIMENT_TO_ALIAS.items():
        mv = selected.get(exp_name)
        if mv is None:
            print(f"⚠️  No version found from '{exp_name}' — skipping {alias} alias")
            continue

        client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        f1 = client.get_run(mv.run_id).data.metrics.get("f1_macro")
        f1_str = f"F1={f1:.4f}" if f1 is not None else "F1=n/a"
        print(f"✅ v{mv.version} → {alias:<11} (from {exp_name}, {f1_str})")


if __name__ == "__main__":
    main()
