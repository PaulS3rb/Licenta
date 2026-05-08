# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Phase 1: Data Setup & Preprocessing
# =============================================================================
# Dataset: UCI Student Performance Dataset (Cortez & Silva, 2008)
# Download from: https://archive.ics.uci.edu/dataset/320/student+performance
# After downloading, unzip and place student-mat.csv and student-por.csv
# in the same folder as this script.
# =============================================================================

import pandas as pd
import numpy as np

# ── 1. LOAD THE DATA ──────────────────────────────────────────────────────────
# The dataset comes in two files:
#   student-mat.csv  → Math course
#   student-por.csv  → Portuguese language course
# We will work with the Math dataset for this thesis.
# The separator in these CSV files is a semicolon (;), not a comma.

df = pd.read_csv("student-mat.csv", sep=";")

print("=" * 60)
print("STEP 1: Dataset loaded successfully")
print("=" * 60)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print()

# ── 2. FIRST LOOK ─────────────────────────────────────────────────────────────
# Always inspect the first few rows to understand the structure.

print("First 5 rows:")
print(df.head())
print()

# .info() shows column names, data types, and whether there are missing values.
print("Dataset info:")
df.info()
print()

# Basic statistics for all numeric columns.
print("Summary statistics:")
print(df.describe())
print()

# ── 3. CHECK FOR MISSING VALUES ───────────────────────────────────────────────
# The UCI student dataset is well-curated and typically has no missing values,
# but we always check — this is good practice and important for the thesis.

print("=" * 60)
print("STEP 2: Checking for missing values")
print("=" * 60)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
print()

# ── 4. UNDERSTAND THE TARGET VARIABLE ────────────────────────────────────────
# G3 = final grade (0–20 scale) — this is what we want to predict.
# G1 = first period grade, G2 = second period grade (also 0–20).
# Note: G1, G2, and G3 are highly correlated — we will discuss this in EDA.

print("=" * 60)
print("STEP 3: Target variable — G3 (final grade)")
print("=" * 60)
print(df["G3"].describe())
print(f"\nGrade distribution:\n{df['G3'].value_counts().sort_index()}")
print()

# ── 5. CREATE BINARY PASS/FAIL COLUMN ────────────────────────────────────────
# For classification models (logistic regression, random forest), we need
# a binary target. In the Portuguese school system, 10/20 is the pass mark.

df["pass_fail"] = (df["G3"] >= 10).astype(int)
# 1 = pass (G3 >= 10), 0 = fail (G3 < 10)

pass_count = df["pass_fail"].value_counts()
print("=" * 60)
print("STEP 4: Binary pass/fail column created")
print("=" * 60)
print(f"Pass (1): {pass_count.get(1, 0)} students")
print(f"Fail (0): {pass_count.get(0, 0)} students")
print(f"Pass rate: {pass_count.get(1, 0) / len(df) * 100:.1f}%")
print()

# ── 6. ENCODE CATEGORICAL VARIABLES ──────────────────────────────────────────
# Machine learning models need numbers, not text.
# We use pandas get_dummies() for one-hot encoding of categorical columns.
# drop_first=True avoids the "dummy variable trap" (multicollinearity).

# First, let's identify which columns are categorical (object type).
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
# We define two feature sets:
#   X_regression  → for linear regression (predicting G3 as a number)
#   X_classification → for logistic regression & random forest (predicting pass/fail)
#
# IMPORTANT: We EXCLUDE G1 and G2 from the main feature sets.
# Using G1/G2 to predict G3 would give artificially high accuracy,
# because they are interim grades from the same school year.
# Your thesis should mention this choice and its justification.

# Columns to always exclude from features
exclude_cols = ["G3", "pass_fail", "G1", "G2"]

feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]

X = df_encoded[feature_cols]        # Features (input variables)
y_regression = df_encoded["G3"]     # Target for regression (numeric grade)
y_classification = df_encoded["pass_fail"]  # Target for classification (0/1)

print("=" * 60)
print("STEP 6: Feature sets defined")
print("=" * 60)
print(f"Number of features: {X.shape[1]}")
print(f"Feature columns:\n{feature_cols}")
print()

# ── 8. SAVE PREPROCESSED DATA ─────────────────────────────────────────────────
# Save the encoded dataframe so we can reuse it in later phases
# without repeating all the preprocessing steps.

df_encoded.to_csv("student_preprocessed.csv", index=False)

print("=" * 60)
print("STEP 7: Preprocessed data saved")
print("=" * 60)
print("File saved: student_preprocessed.csv")
print()

# ── 9. SUMMARY ────────────────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 1 COMPLETE — Summary")
print("=" * 60)
print(f"  Total students:      {len(df)}")
print(f"  Features available:  {X.shape[1]}")
print(f"  Pass rate:           {df['pass_fail'].mean() * 100:.1f}%")
print(f"  Mean final grade:    {df['G3'].mean():.2f} / 20")
print(f"  Missing values:      {df.isnull().sum().sum()}")
