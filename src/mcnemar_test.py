"""
Student Dropout — Early Warning Model
mcnemar_test.py

Statistical significance test for model fusion.

Compares the soft VotingClassifier (from fusion_model.py) against the
production grid search Random Forest using McNemar's test. The goal is to
decide whether the small F1 macro gap between them (0.6760 vs 0.6796) is a
real difference or just noise.

Both models are fit on the exact same saved train/test split (no re-splitting,
no re-preprocessing), predictions are taken on the test set, and McNemar's
exact test is run on their agreement/disagreement contingency table.

The result is logged to the existing 'student-dropout-fusion' experiment as a
run named 'mcnemar_test' (p-value as a metric, contingency counts as params).
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from statsmodels.stats.contingency_tables import mcnemar

ALPHA = 0.05

# ── 1. Load preprocessed data (use saved splits as-is) ────────────────────────
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

# ── 3. Base learners ──────────────────────────────────────────────────────────
# Mirrors fusion_model.py's make_base_estimators() exactly (same tuned Grid
# Search params). Duplicated rather than imported because importing
# fusion_model.py would execute its full training/registration run on import.
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

# ── 4. Build and fit both models on the same split ────────────────────────────
# Production baseline: tuned Grid Search Random Forest.
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Soft VotingClassifier identical to fusion_model.py.
voting = VotingClassifier(
    estimators=make_base_estimators(),
    voting="soft",
    n_jobs=-1,
)

print("Fitting Random Forest (production baseline)...")
rf.fit(X_train, y_train)

print("Fitting soft VotingClassifier (fusion)...")
voting.fit(X_train, y_train)

# ── 5. Predictions on the test set ────────────────────────────────────────────
rf_pred     = rf.predict(X_test)
voting_pred = voting.predict(X_test)

rf_correct     = (rf_pred == y_test)
voting_correct = (voting_pred == y_test)

both_correct = int(np.sum(rf_correct & voting_correct))
rf_only      = int(np.sum(rf_correct & ~voting_correct))   # RF right, voting wrong
voting_only  = int(np.sum(~rf_correct & voting_correct))   # voting right, RF wrong
both_wrong   = int(np.sum(~rf_correct & ~voting_correct))

# ── 6. McNemar's test on the 2x2 contingency table ────────────────────────────
# Rows = Random Forest correct/wrong, Cols = Voting correct/wrong.
# The discordant (off-diagonal) cells rf_only and voting_only drive the test.
table = [
    [both_correct, rf_only],
    [voting_only,  both_wrong],
]

result  = mcnemar(table, exact=True)
p_value = result.pvalue

rf_acc     = accuracy_score(y_test, rf_pred)
voting_acc = accuracy_score(y_test, voting_pred)

# ── 7. Report ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("McNemar's test — soft Voting vs production Random Forest")
print("=" * 60)
print("Contingency table (test set):")
print(f"  both correct            : {both_correct}")
print(f"  RF-only correct         : {rf_only}")
print(f"  Voting-only correct     : {voting_only}")
print(f"  both wrong              : {both_wrong}")
print()
print(f"Random Forest accuracy    : {rf_acc:.4f}")
print(f"Soft Voting accuracy      : {voting_acc:.4f}")
print()
print(f"Discordant pairs          : RF-only={rf_only}  Voting-only={voting_only}")
print(f"McNemar statistic         : {result.statistic}")
print(f"McNemar p-value           : {p_value:.4f}")
print()

if p_value < ALPHA:
    verdict = "statistically significant difference"
    print(f"✅ p = {p_value:.4f} < {ALPHA} → {verdict}.")
    print("   The two models make meaningfully different errors.")
else:
    verdict = ("no statistically significant difference — "
               "models perform equivalently")
    print(f"ℹ️  p = {p_value:.4f} ≥ {ALPHA} → {verdict}.")
    print("   The small F1 macro gap is consistent with noise.")
print("=" * 60)

# ── 8. Visualize the contingency table ────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
fig_path = "outputs/mcnemar_test.png"

cells = np.array([[both_correct, rf_only],
                  [voting_only,  both_wrong]])

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.imshow(cells, cmap="Blues")

ax.set_xticks([0, 1], ["Voting correct", "Voting wrong"])
ax.set_yticks([0, 1], ["RF correct", "RF wrong"])

# Annotate each cell with its count; use dark text on light cells.
threshold = cells.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cells[i, j]}", ha="center", va="center",
                fontsize=18,
                color="white" if cells[i, j] > threshold else "black")

ax.set_title("McNemar's test — soft Voting vs Random Forest")
sig_txt = "significant" if p_value < ALPHA else "not significant"
ax.set_xlabel(
    f"Discordant: RF-only={rf_only}  Voting-only={voting_only}\n"
    f"McNemar exact p = {p_value:.4f}  →  {sig_txt} (α={ALPHA})"
)
fig.tight_layout()
fig.savefig(fig_path, dpi=120)
plt.close(fig)

print(f"\nSaved contingency-table figure to {fig_path}")

# ── 9. Log to MLflow ──────────────────────────────────────────────────────────
with mlflow.start_run(run_name="mcnemar_test"):
    mlflow.set_tag("test_type", "mcnemar")
    mlflow.set_tag("category", "fusion")
    mlflow.set_tag("comparison", "voting_soft_vs_random_forest_grid_search")

    mlflow.log_params({
        "model_a":         "random_forest_grid_search",
        "model_b":         "voting_soft",
        "exact":           True,
        "alpha":           ALPHA,
        "both_correct":    both_correct,
        "rf_only_correct": rf_only,
        "voting_only_correct": voting_only,
        "both_wrong":      both_wrong,
    })

    mlflow.log_metric("mcnemar_pvalue", p_value)
    mlflow.log_metric("rf_accuracy",     rf_acc)
    mlflow.log_metric("voting_accuracy", voting_acc)
    mlflow.log_metric("significant", int(p_value < ALPHA))

    mlflow.log_artifact(fig_path, artifact_path="mcnemar_test")

    mlflow.log_text(
        f"McNemar's test (exact)\n"
        f"model_a = random_forest_grid_search\n"
        f"model_b = voting_soft\n\n"
        f"both_correct        = {both_correct}\n"
        f"rf_only_correct     = {rf_only}\n"
        f"voting_only_correct = {voting_only}\n"
        f"both_wrong          = {both_wrong}\n\n"
        f"statistic = {result.statistic}\n"
        f"p_value   = {p_value:.6f}\n"
        f"verdict   = {verdict}\n",
        "mcnemar_result.txt",
    )

    run_id = mlflow.active_run().info.run_id
    print(f"\nLogged to MLflow run 'mcnemar_test' (run_id: {run_id})")

print(f"\nOpen MLflow UI to inspect the result:")
print(f"   mlflow ui")
