# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Portuguese Dataset — Phase 4: Machine Learning Models
# =============================================================================
# Run this AFTER por_phase1_data_setup.py.
# Reads from:  ../data/student_por_preprocessed.csv
# Writes to:   ../plots/por_plot7_*.png … ../plots/por_plot10_*.png
#              ../results/por_results_model_metrics.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# ── SETUP ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/student_por_preprocessed.csv")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150

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

# ── FEATURES & TARGETS ────────────────────────────────────────────────────────
exclude_cols = ["G3", "pass_fail", "G1", "G2"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X     = df[feature_cols]
y_reg = df["G3"]
y_clf = df["pass_fail"]

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# ── FEATURE SCALING ───────────────────────────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("=" * 65)
print("PORTUGUESE — PHASE 4: MACHINE LEARNING MODELS")
print("=" * 65)
print(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
print()

# =============================================================================
# MODEL 1 — LINEAR REGRESSION
# =============================================================================

print("─" * 65)
print("MODEL 1: Linear Regression  (predicts G3 score)")
print("─" * 65)

lr = LinearRegression()
lr.fit(X_train, y_reg_train)
y_pred_lr = lr.predict(X_test)

r2   = r2_score(y_reg_test, y_pred_lr)
mae  = mean_absolute_error(y_reg_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_lr))
cv_r2 = cross_val_score(lr, X, y_reg, cv=5, scoring="r2")

print(f"  R²:             {r2:.4f}")
print(f"  MAE:            {mae:.4f}  (avg error in grade points)")
print(f"  RMSE:           {rmse:.4f}")
print(f"  CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print()

# Plot 7: Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_reg_test, y_pred_lr, alpha=0.5, color="#4C72B0", s=50)
ax.plot([0, 20], [0, 20], color="#C44E52", linewidth=1.5,
        linestyle="--", label="Perfect prediction")
ax.set_title("Linear Regression — Actual vs Predicted (Portuguese)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Actual Final Grade (G3)", fontsize=12)
ax.set_ylabel("Predicted Final Grade (G3)", fontsize=12)
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 21)
ax.legend()
ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
        fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))
plt.tight_layout()
plt.savefig("../plots/por_plot7_linear_regression.png")
plt.show()
print("Plot 7 saved: ../plots/por_plot7_linear_regression.png")
print()

# =============================================================================
# MODEL 2 — LOGISTIC REGRESSION
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
print(classification_report(y_clf_test, y_pred_log, target_names=["Fail", "Pass"]))

# =============================================================================
# MODEL 3 — RANDOM FOREST
# =============================================================================

print("─" * 65)
print("MODEL 3: Random Forest  (pass / fail classification)")
print("─" * 65)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_clf_train)
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
print(classification_report(y_clf_test, y_pred_rf, target_names=["Fail", "Pass"]))

# =============================================================================
# MODEL 4 — NEURAL NETWORK
# =============================================================================

print("─" * 65)
print("MODEL 4: Neural Network  (pass / fail classification)")
print("─" * 65)

nn = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
nn.fit(X_train_scaled, y_clf_train)
y_pred_nn = nn.predict(X_test_scaled)

acc_nn  = accuracy_score(y_clf_test, y_pred_nn)
prec_nn = precision_score(y_clf_test, y_pred_nn, zero_division=0)
rec_nn  = recall_score(y_clf_test, y_pred_nn, zero_division=0)
f1_nn   = f1_score(y_clf_test, y_pred_nn, zero_division=0)
cv_acc_nn = cross_val_score(nn, X_train_scaled, y_clf_train, cv=5, scoring="accuracy")

print(f"  Architecture:    Input → 32 → Output")
print(f"  Activation:      ReLU")
print(f"  Early stopping:  Yes (validation fraction = 10%)")
print(f"  Iterations run:  {nn.n_iter_}")
print()
print(f"  Accuracy:        {acc_nn:.4f}")
print(f"  Precision:       {prec_nn:.4f}")
print(f"  Recall:          {rec_nn:.4f}")
print(f"  F1 Score:        {f1_nn:.4f}")
print(f"  CV Accuracy:     {cv_acc_nn.mean():.4f} ± {cv_acc_nn.std():.4f}")
print()
print("  Classification Report:")
print(classification_report(y_clf_test, y_pred_nn,
                             target_names=["Fail", "Pass"], zero_division=0))

# =============================================================================
# PLOT 8 — CONFUSION MATRICES (all three classifiers)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, y_pred, title in zip(
    axes,
    [y_pred_log, y_pred_rf, y_pred_nn],
    ["Logistic Regression", "Random Forest", "Neural Network"]
):
    cm = confusion_matrix(y_clf_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"],
                yticklabels=["Fail", "Pass"],
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix\n{title} (Portuguese)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Actual Label", fontsize=10)
plt.tight_layout()
plt.savefig("../plots/por_plot8_confusion_matrices.png")
plt.show()
print("Plot 8 saved: ../plots/por_plot8_confusion_matrices.png")

# =============================================================================
# PLOT 9 — FEATURE IMPORTANCES (Random Forest)
# =============================================================================

importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)
importances.index = [LABELS.get(col, col) for col in importances.index]
top_importances = importances.tail(15)

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#4C72B0" if v >= top_importances.quantile(0.6) else "#9DB8D9"
          for v in top_importances.values]
top_importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_title("Random Forest — Top 15 Feature Importances (Portuguese)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_ylabel("")
ax.axvline(x=top_importances.mean(), color="#C44E52", linestyle="--",
           linewidth=1.2, label=f"Mean importance ({top_importances.mean():.3f})")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("../plots/por_plot9_feature_importances.png")
plt.show()
print("Plot 9 saved: ../plots/por_plot9_feature_importances.png")

# =============================================================================
# PLOT 10 — MODEL COMPARISON
# =============================================================================

metrics  = ["Accuracy", "Precision", "Recall", "F1 Score"]
log_vals = [acc_log, prec_log, rec_log, f1_log]
rf_vals  = [acc_rf,  prec_rf,  rec_rf,  f1_rf]
nn_vals  = [acc_nn,  prec_nn,  rec_nn,  f1_nn]

x     = np.arange(len(metrics))
width = 0.22

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width, log_vals, width, label="Logistic Regression",
               color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x,          rf_vals,  width, label="Random Forest",
               color="#55A868", edgecolor="white")
bars3 = ax.bar(x + width,  nn_vals,  width, label="Neural Network",
               color="#C44E52", edgecolor="white")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

baseline = y_clf.mean()
ax.set_title("Model Comparison — Classification Metrics (Portuguese)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.15)
ax.axhline(y=baseline, color="#888888", linestyle="--", linewidth=1,
           label=f"Baseline (always predict pass = {baseline*100:.1f}%)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../plots/por_plot10_model_comparison.png")
plt.show()
print("Plot 10 saved: ../plots/por_plot10_model_comparison.png")

# =============================================================================
# SAVE METRICS CSV
# =============================================================================

metrics_data = {
    "Model":     ["Linear Regression", "Logistic Regression", "Random Forest", "Neural Network"],
    "R²":        [round(r2, 4),   "-",             "-",            "-"],
    "MAE":       [round(mae, 4),  "-",             "-",            "-"],
    "RMSE":      [round(rmse, 4), "-",             "-",            "-"],
    "Accuracy":  ["-", round(acc_log, 4),  round(acc_rf, 4),  round(acc_nn, 4)],
    "Precision": ["-", round(prec_log, 4), round(prec_rf, 4), round(prec_nn, 4)],
    "Recall":    ["-", round(rec_log, 4),  round(rec_rf, 4),  round(rec_nn, 4)],
    "F1 Score":  ["-", round(f1_log, 4),   round(f1_rf, 4),   round(f1_nn, 4)],
    "CV Score":  [
        f"{cv_r2.mean():.4f} ± {cv_r2.std():.4f}",
        f"{cv_acc_log.mean():.4f} ± {cv_acc_log.std():.4f}",
        f"{cv_acc_rf.mean():.4f} ± {cv_acc_rf.std():.4f}",
        f"{cv_acc_nn.mean():.4f} ± {cv_acc_nn.std():.4f}",
    ],
}

pd.DataFrame(metrics_data).to_csv("../results/por_results_model_metrics.csv", index=False)

# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 65)
print("PORTUGUESE PHASE 4 COMPLETE — Summary")
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
print("  Neural Network:")
print(f"    Accuracy = {acc_nn:.4f}  |  Precision = {prec_nn:.4f}  "
      f"|  Recall = {rec_nn:.4f}  |  F1 = {f1_nn:.4f}")
print()

best_f1  = max(f1_log, f1_rf, f1_nn)
best_clf = {f1_log: "Logistic Regression",
            f1_rf:  "Random Forest",
            f1_nn:  "Neural Network"}[best_f1]
print(f"  Best classifier (by F1): {best_clf}  (F1 = {best_f1:.4f})")
print()
print("Files saved:")
print("  ../plots/por_plot7_linear_regression.png")
print("  ../plots/por_plot8_confusion_matrices.png")
print("  ../plots/por_plot9_feature_importances.png")
print("  ../plots/por_plot10_model_comparison.png")
print("  ../results/por_results_model_metrics.csv")
print()
print("Ready for Portuguese Phase 5: Evaluation & Thesis Conclusion")