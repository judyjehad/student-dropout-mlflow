import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# Version 2 → Production (best grid search model, F1=0.6796)
client.set_registered_model_alias(
    name="student-dropout-classifier",
    alias="Production",
    version="2"
)

# Version 3 → Staging (best Hyperopt model, F1=0.6781)
client.set_registered_model_alias(
    name="student-dropout-classifier",
    alias="Staging",
    version="3"
)

# Version 1 → Archived (baseline model)
client.set_registered_model_alias(
    name="student-dropout-classifier",
    alias="Archived",
    version="1"
)

print("✅ Version 1 → Archived")
print("✅ Version 2 → Production (grid search, F1=0.6796)")
print("✅ Version 3 → Staging  (Hyperopt,     F1=0.6781)")