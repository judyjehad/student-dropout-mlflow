import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# Set tuned model as Production alias
client.set_registered_model_alias(
    name="student-dropout-classifier",
    alias="Production",
    version="2"
)

# Set baseline model as Archived alias
client.set_registered_model_alias(
    name="student-dropout-classifier",
    alias="Archived",
    version="1"
)

print("✅ Version 2 → Production")
print("✅ Version 1 → Archived")