"""
Student Dropout — Early Warning Model
preprocess.py

Column strategy:
  Numerical columns         → StandardScaler
  Categorical coded columns → OneHotEncoder
  Binary columns            → passthrough
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/students.csv", sep=";")
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Fix column name (hidden tab character) ─────────────────────────────────
df.rename(columns={"Daytime/evening attendance\t": "Daytime/evening attendance"},
          inplace=True)

# ── 3. Drop semester 2 features (early-warning model) ────────────────────────
SEMESTER_2_COLS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]
df.drop(columns=SEMESTER_2_COLS, inplace=True)
print(f"After dropping sem 2: {df.shape[1]} columns remaining")

# ── 4. Encode target (alphabetical: Dropout=0, Enrolled=1, Graduate=2) ────────
le = LabelEncoder()
df["Target"] = le.fit_transform(df["Target"])
print(f"\nTarget classes: {list(le.classes_)}")
print(pd.Series(df["Target"]).value_counts().sort_index().to_string())

# ── 5. Separate X and y ───────────────────────────────────────────────────────
y = df["Target"]
X = df.drop(columns=["Target"])

# ── 6. Define column groups ───────────────────────────────────────────────────

# True categories — codes with no numeric meaning → OneHotEncoder
CATEGORICAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Continuous numeric → StandardScaler
NUMERICAL_COLS = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Already 0/1 → passthrough
BINARY_COLS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

print(f"\nColumn groups:")
print(f"  Categorical (OHE)   : {len(CATEGORICAL_COLS)} columns")
print(f"  Numerical (scaled)  : {len(NUMERICAL_COLS)} columns")
print(f"  Binary (passthrough): {len(BINARY_COLS)} columns")

# ── 7. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print(f"\nTrain: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

# ── 8. Build preprocessor ─────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe",    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_COLS),
        ("scaler", StandardScaler(),                                                           NUMERICAL_COLS),
    ],
    remainder="passthrough",  # binary cols pass through unchanged
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

# ── 9. Recover feature names ──────────────────────────────────────────────────
ohe_feature_names = (
    preprocessor.named_transformers_["ohe"]
    .get_feature_names_out(CATEGORICAL_COLS)
    .tolist()
)
all_feature_names = ohe_feature_names + NUMERICAL_COLS + BINARY_COLS

print(f"\nFinal feature matrix: {X_train_processed.shape[1]} features")
print(f"  OHE dummies : {len(ohe_feature_names)}")
print(f"  Numerical   : {len(NUMERICAL_COLS)}")
print(f"  Binary      : {len(BINARY_COLS)}")

# ── 10. Save everything ───────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X_train.npy", X_train_processed)
np.save("data/processed/X_test.npy",  X_test_processed)
np.save("data/processed/y_train.npy", y_train.values)
np.save("data/processed/y_test.npy",  y_test.values)

with open("data/processed/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("data/processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("data/processed/feature_names.pkl", "wb") as f:
    pickle.dump(all_feature_names, f)

print(f"\n✅ Preprocessing complete.")
print(f"   X_train : {X_train_processed.shape}   y_train : {y_train.shape}")
print(f"   X_test  : {X_test_processed.shape}    y_test  : {y_test.shape}")
print(f"\nSaved to data/processed/")