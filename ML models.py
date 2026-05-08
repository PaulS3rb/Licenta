# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Phase 4: Machine Learning Models
# =============================================================================
# Run this AFTER phase1_data_setup.py.
# Make sure student_preprocessed.csv is in the same folder.
#
# This script trains and evaluates three models:
#   1. Linear Regression  — predicts G3 as a numeric score (0–20)
#   2. Logistic Regression — classifies pass (1) vs fail (0)
#   3. Random Forest       — classifies pass (1) vs fail (0)
#
# Outputs:
#   - Printed metrics for each model
#   - plot7_linear_regression.png
#   - plot8_confusion_matrices.png
#   - plot9_feature_importances.png
#   - plot10_model_comparison.png
#   - results_model_metrics.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# ── SETUP ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("student_preprocessed.csv")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150

# ── FEATURE LABEL MAP ─────────────────────────────────────────────────────────
# Used for readable axis labels in feature importance plot.
LABELS = {
    "age":        "Age",
    "Medu":       "Mother's Education",
    "Fedu":       "Father's Education",
    "traveltime": "Travel Time to School",
    "studytime":  "Weekly Study Time",
    "failures":   "Past Class Failures",
    "famrel":     "Family Relationship Quality",
    "freetime":   "Free Time After School",
    "goout":      "Going Out with Friends",
    "Dalc":       "Workday Alcohol Consumption",
    "Walc":       "Weekend Alcohol Consumption",
    "health":     "Current Health Status",
    "absences":   "Number of Absences",
}

# ── DEFINE FEATURES AND TARGETS ───────────────────────────────────────────────
# Exclude G1, G2 (data leakage) and the two target columns.
exclude_cols = ["G3", "pass_fail", "G1", "G2"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y_reg   = df["G3"]        # Continuous target for Linear Regression
y_clf   = df["pass_fail"] # Binary target for Logistic Regression & Random Forest

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
# 80% training, 20% testing. random_state=42 ensures reproducibility —
# running the script multiple times gives the same split every time.
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42  # same split, same random_state
)

print("=" * 65)
print("PHASE 4: MACHINE LEARNING MODELS")
print("=" * 65)
print(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
print()

# ── FEATURE SCALING ───────────────────────────────────────────────────────────
# Logistic Regression is sensitive to feature scale — a variable ranging
# 0–75 (absences) would dominate one ranging 1–4 (studytime) without scaling.
# We scale AFTER the train/test split to avoid data leakage from the test set.
# Random Forest does NOT need scaling (tree-based models are scale-invariant).

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled  = scaler.transform(X_test)        # apply same scale to test

# =============================================================================
# MODEL 1 — LINEAR REGRESSION
# =============================================================================
# Predicts G3 as a continuous score (0–20).
# Metrics:
#   R²  → proportion of variance explained (1.0 = perfect, 0 = no better than mean)
#   MAE → mean absolute error in grade points (e.g. 2.1 means off by ~2 points)
#   RMSE → root mean squared error (penalises large errors more than MAE)
# =============================================================================

print("─" * 65)
print("MODEL 1: Linear Regression  (predicts G3 score)")
print("─" * 65)

lr = LinearRegression()
lr.fit(X_train, y_reg_train)          # no scaling needed for linear regression
y_pred_lr = lr.predict(X_test)

r2   = r2_score(y_reg_test, y_pred_lr)
mae  = mean_absolute_error(y_reg_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_lr))

# Cross-validation R² (5-fold) — more reliable estimate than a single split
cv_r2 = cross_val_score(lr, X, y_reg, cv=5, scoring="r2")

print(f"  R²:             {r2:.4f}")
print(f"  MAE:            {mae:.4f}  (avg error in grade points)")
print(f"  RMSE:           {rmse:.4f}")
print(f"  CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print()

# Plot: Actual vs Predicted grades
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_reg_test, y_pred_lr, alpha=0.5, color="#4C72B0", s=50, label="Students")
ax.plot([0, 20], [0, 20], color="#C44E52", linewidth=1.5,
        linestyle="--", label="Perfect prediction")
ax.set_title("Linear Regression — Actual vs Predicted Final Grade",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Actual Final Grade (G3)", fontsize=12)
ax.set_ylabel("Predicted Final Grade (G3)", fontsize=12)
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 21)
ax.legend()
# Add R² annotation
ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
        fontsize=11, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))
plt.tight_layout()
plt.savefig("plot7_linear_regression.png")
plt.show()
print("Plot 7 saved: plot7_linear_regression.png")
print()

# =============================================================================
# MODEL 2 — LOGISTIC REGRESSION
# =============================================================================
# Predicts pass (1) or fail (0).
# Uses scaled features. max_iter=1000 ensures the solver converges.
# Metrics:
#   Accuracy  → % of students correctly classified
#   Precision → of predicted passes, how many actually passed?
#   Recall    → of actual passes, how many did we catch?
#   F1        → harmonic mean of precision and recall
# =============================================================================

print("─" * 65)
print("MODEL 2: Logistic Regression  (pass / fail classification)")
print("─" * 65)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_clf_train)
y_pred_log = log_reg.predict(X_test_scaled)

acc_log  = accuracy_score(y_clf_test, y_pred_log)
prec_log = precision_score(y_clf_test, y_pred_log)
rec_log  = recall_score(y_clf_test, y_pred_log)
f1_log   = f1_score(y_clf_test, y_pred_log)
cv_acc_log = cross_val_score(log_reg, X_train_scaled, y_clf_train, cv=5, scoring="accuracy")

print(f"  Accuracy:        {acc_log:.4f}")
print(f"  Precision:       {prec_log:.4f}")
print(f"  Recall:          {rec_log:.4f}")
print(f"  F1 Score:        {f1_log:.4f}")
print(f"  CV Accuracy:     {cv_acc_log.mean():.4f} ± {cv_acc_log.std():.4f}")
print()
print("  Classification Report:")
print(classification_report(y_clf_test, y_pred_log,
                             target_names=["Fail", "Pass"]))

# =============================================================================
# MODEL 3 — RANDOM FOREST
# =============================================================================
# Ensemble of 100 decision trees. Votes on the majority class (pass/fail).
# Does NOT need feature scaling.
# Extra benefit: gives feature importances — which variables matter most?
# n_estimators=100 is a solid default; more trees = more stable but slower.
# =============================================================================

print("─" * 65)
print("MODEL 3: Random Forest  (pass / fail classification)")
print("─" * 65)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_clf_train)           # raw (unscaled) features
y_pred_rf = rf.predict(X_test)

acc_rf  = accuracy_score(y_clf_test, y_pred_rf)
prec_rf = precision_score(y_clf_test, y_pred_rf)
rec_rf  = recall_score(y_clf_test, y_pred_rf)
f1_rf   = f1_score(y_clf_test, y_pred_rf)
cv_acc_rf = cross_val_score(rf, X, y_clf, cv=5, scoring="accuracy")

print(f"  Accuracy:        {acc_rf:.4f}")
print(f"  Precision:       {prec_rf:.4f}")
print(f"  Recall:          {rec_rf:.4f}")
print(f"  F1 Score:        {f1_rf:.4f}")
print(f"  CV Accuracy:     {cv_acc_rf.mean():.4f} ± {cv_acc_rf.std():.4f}")
print()
print("  Classification Report:")
print(classification_report(y_clf_test, y_pred_rf,
                             target_names=["Fail", "Pass"]))

# =============================================================================
# PLOT 8 — CONFUSION MATRICES (Logistic Regression & Random Forest)
# =============================================================================
# A confusion matrix shows:
#   True Positives  (TP): predicted Pass, actually Pass  ✓
#   True Negatives  (TN): predicted Fail, actually Fail  ✓
#   False Positives (FP): predicted Pass, actually Fail  ✗ (Type I error)
#   False Negatives (FN): predicted Fail, actually Pass  ✗ (Type II error)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, y_pred, title in zip(
    axes,
    [y_pred_log, y_pred_rf],
    ["Logistic Regression", "Random Forest"]
):
    cm = confusion_matrix(y_clf_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"],
                yticklabels=["Fail", "Pass"],
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix\n{title}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)

plt.tight_layout()
plt.savefig("plot8_confusion_matrices.png")
plt.show()
print("Plot 8 saved: plot8_confusion_matrices.png")
print()

# =============================================================================
# PLOT 9 — FEATURE IMPORTANCES (Random Forest)
# =============================================================================
# Random Forest calculates how much each feature reduces impurity across all
# trees. Higher importance = more useful for predicting pass/fail.
# This is one of the most valuable outputs for your thesis discussion.

importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)

# Apply readable labels where available, keep original name otherwise
importances.index = [LABELS.get(col, col) for col in importances.index]

# Show top 15 features to keep the plot readable
top_importances = importances.tail(15)

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#4C72B0" if v >= top_importances.quantile(0.6) else "#9DB8D9"
          for v in top_importances.values]
top_importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_title("Random Forest — Top 15 Feature Importances",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_ylabel("")
ax.axvline(x=top_importances.mean(), color="#C44E52", linestyle="--",
           linewidth=1.2, label=f"Mean importance ({top_importances.mean():.3f})")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plot9_feature_importances.png")
plt.show()
print("Plot 9 saved: plot9_feature_importances.png")
print()

# =============================================================================
# PLOT 10 — MODEL COMPARISON BAR CHART
# =============================================================================
# Side-by-side comparison of all classification metrics for both models.
# Makes it easy to see at a glance which model performs better and where.

metrics = {
    "Accuracy":  [acc_log,  acc_rf],
    "Precision": [prec_log, prec_rf],
    "Recall":    [rec_log,  rec_rf],
    "F1 Score":  [f1_log,   f1_rf],
}

x      = np.arange(len(metrics))
width  = 0.30
labels = list(metrics.keys())
log_vals = [metrics[m][0] for m in labels]
rf_vals  = [metrics[m][1] for m in labels]

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, log_vals, width, label="Logistic Regression",
               color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x + width / 2, rf_vals,  width, label="Random Forest",
               color="#55A868", edgecolor="white")

# Add value labels on top of each bar
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

ax.set_title("Model Comparison — Classification Metrics",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.axhline(y=0.671, color="#C44E52", linestyle="--", linewidth=1,
           label="Baseline (always predict pass = 67.1%)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plot10_model_comparison.png")
plt.show()
print("Plot 10 saved: plot10_model_comparison.png")
print()

# =============================================================================
# SAVE METRICS TO CSV
# =============================================================================

metrics_data = {
    "Model": [
        "Linear Regression",
        "Logistic Regression",
        "Random Forest"
    ],
    "R²":        [round(r2, 4),   "-",            "-"],
    "MAE":       [round(mae, 4),  "-",            "-"],
    "RMSE":      [round(rmse, 4), "-",            "-"],
    "Accuracy":  ["-", round(acc_log, 4),  round(acc_rf, 4)],
    "Precision": ["-", round(prec_log, 4), round(prec_rf, 4)],
    "Recall":    ["-", round(rec_log, 4),  round(rec_rf, 4)],
    "F1 Score":  ["-", round(f1_log, 4),   round(f1_rf, 4)],
    "CV Score":  [
        f"{cv_r2.mean():.4f} ± {cv_r2.std():.4f}",
        f"{cv_acc_log.mean():.4f} ± {cv_acc_log.std():.4f}",
        f"{cv_acc_rf.mean():.4f} ± {cv_acc_rf.std():.4f}"
    ],
}

pd.DataFrame(metrics_data).to_csv("results_model_metrics.csv", index=False)

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 65)
print("PHASE 4 COMPLETE — Summary")
print("=" * 65)
print()
print("  Linear Regression:")
print(f"    R² = {r2:.4f}  |  MAE = {mae:.4f}  |  RMSE = {rmse:.4f}")
print()
print("  Logistic Regression:")
print(f"    Accuracy = {acc_log:.4f}  |  Precision = {prec_log:.4f}  "
      f"|  Recall = {rec_log:.4f}  |  F1 = {f1_log:.4f}")
print()
print("  Random Forest:")
print(f"    Accuracy = {acc_rf:.4f}  |  Precision = {prec_rf:.4f}  "
      f"|  Recall = {rec_rf:.4f}  |  F1 = {f1_rf:.4f}")
print()

best_clf = "Random Forest" if f1_rf > f1_log else "Logistic Regression"
print(f"  Best classifier (by F1): {best_clf}")
print()
print("Files saved:")
print("  plot7_linear_regression.png")
print("  plot8_confusion_matrices.png")
print("  plot9_feature_importances.png")
print("  plot10_model_comparison.png")
print("  results_model_metrics.csv")
print()
print("Ready for Phase 5: Evaluation & Thesis Conclusion")