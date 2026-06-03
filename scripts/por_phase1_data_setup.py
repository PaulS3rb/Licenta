# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Portuguese Dataset — Phase 1: Data Setup & Preprocessing
# =============================================================================
# Run this script from inside the scripts/ folder.
# Reads from:  ../data/student-por.csv
# Writes to:   ../data/student_por_preprocessed.csv
# =============================================================================

import os
import pandas as pd
import numpy as np

# ── ENSURE OUTPUT FOLDERS EXIST ───────────────────────────────────────────────
os.makedirs("../data",    exist_ok=True)
os.makedirs("../plots",   exist_ok=True)
os.makedirs("../results", exist_ok=True)

# ── 1. LOAD THE DATA ──────────────────────────────────────────────────────────
# Portuguese language dataset — 649 students, same 33-variable structure as Math.
# The separator in these CSV files is a semicolon (;), not a comma.

df = pd.read_csv("../data/student-por.csv", sep=";")

print("=" * 60)
print("PORTUGUESE — PHASE 1: Data Setup")
print("=" * 60)
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print()

# ── 2. FIRST LOOK ─────────────────────────────────────────────────────────────
print("First 5 rows:")
print(df.head())
print()

print("Dataset info:")
df.info()
print()

print("Summary statistics:")
print(df.describe())
print()

# ── 3. CHECK FOR MISSING VALUES ───────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Checking for missing values")
print("=" * 60)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
print()

# ── 4. TARGET VARIABLE ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Target variable — G3 (final grade)")
print("=" * 60)
print(df["G3"].describe())
print(f"\nGrade distribution:\n{df['G3'].value_counts().sort_index()}")
print()

# ── 5. CREATE BINARY PASS/FAIL COLUMN ────────────────────────────────────────
# Same passing threshold as Math: 10/20 in the Portuguese school system.

df["pass_fail"] = (df["G3"] >= 10).astype(int)

pass_count = df["pass_fail"].value_counts()
print("=" * 60)
print("STEP 4: Binary pass/fail column created")
print("=" * 60)
print(f"Pass (1): {pass_count.get(1, 0)} students")
print(f"Fail (0): {pass_count.get(0, 0)} students")
print(f"Pass rate: {pass_count.get(1, 0) / len(df) * 100:.1f}%")
print()

# ── 6. ENCODE CATEGORICAL VARIABLES ──────────────────────────────────────────
categorical_cols = df.select_dtypes(include="object").columns.tolist()
print("=" * 60)
print("STEP 5: Encoding categorical variables")
print("=" * 60)
print(f"Categorical columns found: {categorical_cols}")
print()

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"Shape before encoding: {df.shape}")
print(f"Shape after encoding:  {df_encoded.shape}")
print()

# ── 7. DEFINE FEATURE SETS ────────────────────────────────────────────────────
# Exclude G1 and G2 (data leakage) and the two target columns.

exclude_cols = ["G3", "pass_fail", "G1", "G2"]
feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]

X                = df_encoded[feature_cols]
y_regression     = df_encoded["G3"]
y_classification = df_encoded["pass_fail"]

print("=" * 60)
print("STEP 6: Feature sets defined")
print("=" * 60)
print(f"Number of features: {X.shape[1]}")
print(f"Feature columns:\n{feature_cols}")
print()

# ── 8. SAVE PREPROCESSED DATA ─────────────────────────────────────────────────
# Saved as student_por_preprocessed.csv to avoid overwriting the Math dataset.

df_encoded.to_csv("../data/student_por_preprocessed.csv", index=False)

print("=" * 60)
print("STEP 7: Preprocessed data saved")
print("=" * 60)
print("File saved: ../data/student_por_preprocessed.csv")
print()

# ── 9. SUMMARY ────────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 COMPLETE — Summary (Portuguese)")
print("=" * 60)
print(f"  Total students:      {len(df)}")
print(f"  Features available:  {X.shape[1]}")
print(f"  Pass rate:           {df['pass_fail'].mean() * 100:.1f}%")
print(f"  Mean final grade:    {df['G3'].mean():.2f} / 20")
print(f"  Missing values:      {df.isnull().sum().sum()}")
print()
