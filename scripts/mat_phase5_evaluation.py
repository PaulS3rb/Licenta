# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Phase 5: Evaluation & Thesis Conclusion
# =============================================================================
# Run this AFTER all previous phases.
# Make sure student_preprocessed.csv and student-mat.csv are in the same folder.
#
# This script:
#   1. Re-trains all three models (self-contained — no need to re-run Phase 4)
#   2. Produces a final comprehensive evaluation
#   3. Generates a ROC curve comparison plot
#   4. Generates a Precision-Recall curve plot
#   5. Prints a structured thesis conclusion summary
#   6. Saves results_final_report.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# ── SETUP ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/student_mat_preprocessed.csv")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150

# ── FEATURES & TARGETS ────────────────────────────────────────────────────────
exclude_cols = ["G3", "pass_fail", "G1", "G2"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X     = df[feature_cols]
y_reg = df["G3"]
y_clf = df["pass_fail"]

# ── TRAIN/TEST SPLIT (same seed as Phase 4 — identical split) ─────────────────
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# ── SCALING ───────────────────────────────────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── TRAIN MODELS ──────────────────────────────────────────────────────────────
lr      = LinearRegression()
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf      = RandomForestClassifier(n_estimators=100, random_state=42)
nn      = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

lr.fit(X_train, y_reg_train)
log_reg.fit(X_train_scaled, y_clf_train)
rf.fit(X_train, y_clf_train)
nn.fit(X_train_scaled, y_clf_train)

y_pred_lr  = lr.predict(X_test)
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_rf  = rf.predict(X_test)
y_pred_nn  = nn.predict(X_test_scaled)

# Probability scores for ROC and PR curves
y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]
y_prob_rf  = rf.predict_proba(X_test)[:, 1]
y_prob_nn  = nn.predict_proba(X_test_scaled)[:, 1]

print("=" * 65)
print("PHASE 5: EVALUATION & THESIS CONCLUSION")
print("=" * 65)
print()

# =============================================================================
# PART 1 — FULL METRICS TABLE
# =============================================================================

print("─" * 65)
print("PART 1: Complete metrics — all three models")
print("─" * 65)

# Linear Regression metrics
r2   = r2_score(y_reg_test, y_pred_lr)
mae  = mean_absolute_error(y_reg_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_lr))

# Classification metrics
def clf_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
    }

m_log = clf_metrics(y_clf_test, y_pred_log)
m_rf  = clf_metrics(y_clf_test, y_pred_rf)
m_nn  = clf_metrics(y_clf_test, y_pred_nn)

# Cross-validation using StratifiedKFold — preserves class balance in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_r2  = cross_val_score(lr, X, y_reg, cv=5, scoring="r2")
cv_log = cross_val_score(log_reg, X_train_scaled, y_clf_train, cv=skf, scoring="f1")
cv_rf  = cross_val_score(rf, X, y_clf, cv=skf, scoring="f1")
cv_nn  = cross_val_score(nn, X_train_scaled, y_clf_train, cv=skf, scoring="f1")

print(f"  {'Metric':<28} {'Linear Reg':>12} {'Logistic Reg':>14} {'Random Forest':>14} {'Neural Net':>11}")
print(f"  {'-'*28} {'-'*12} {'-'*14} {'-'*14} {'-'*11}")
print(f"  {'R²':<28} {r2:>12.4f} {'—':>14} {'—':>14} {'—':>11}")
print(f"  {'MAE':<28} {mae:>12.4f} {'—':>14} {'—':>14} {'—':>11}")
print(f"  {'RMSE':<28} {rmse:>12.4f} {'—':>14} {'—':>14} {'—':>11}")
print(f"  {'Accuracy':<28} {'—':>12} {m_log['accuracy']:>14.4f} {m_rf['accuracy']:>14.4f} {m_nn['accuracy']:>11.4f}")
print(f"  {'Precision (Pass)':<28} {'—':>12} {m_log['precision']:>14.4f} {m_rf['precision']:>14.4f} {m_nn['precision']:>11.4f}")
print(f"  {'Recall (Pass)':<28} {'—':>12} {m_log['recall']:>14.4f} {m_rf['recall']:>14.4f} {m_nn['recall']:>11.4f}")
print(f"  {'F1 Score (Pass)':<28} {'—':>12} {m_log['f1']:>14.4f} {m_rf['f1']:>14.4f} {m_nn['f1']:>11.4f}")
print(f"  {'CV Score (5-fold)':<28} {cv_r2.mean():>12.4f} {cv_log.mean():>14.4f} {cv_rf.mean():>14.4f} {cv_nn.mean():>11.4f}")
print(f"  {'CV Std Dev':<28} {cv_r2.std():>12.4f} {cv_log.std():>14.4f} {cv_rf.std():>14.4f} {cv_nn.std():>11.4f}")
print()

# =============================================================================
# PART 2 — FAIL CLASS DEEP DIVE
# =============================================================================
# The Fail class is the most educationally important group — and the hardest
# to predict. We report separate metrics for it here.

print("─" * 65)
print("PART 2: Fail class performance (the at-risk students)")
print("─" * 65)

def fail_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fail_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    fail_recall    = tn / (tn + fp) if (tn + fp) > 0 else 0
    fail_f1        = (2 * fail_precision * fail_recall /
                      (fail_precision + fail_recall)
                      if (fail_precision + fail_recall) > 0 else 0)
    return fail_precision, fail_recall, fail_f1, tn, fp, fn, tp

fp_log, fr_log, ff_log, tn_l, fp_l, fn_l, tp_l = fail_metrics(y_clf_test, y_pred_log)
fp_rf,  fr_rf,  ff_rf,  tn_r, fp_r, fn_r, tp_r = fail_metrics(y_clf_test, y_pred_rf)
fp_nn,  fr_nn,  ff_nn,  tn_n, fp_n, fn_n, tp_n = fail_metrics(y_clf_test, y_pred_nn)

total_fail = (y_clf_test == 0).sum()
total_pass = (y_clf_test == 1).sum()

print(f"  Total failing students in test set: {total_fail}")
print(f"  Total passing students in test set: {total_pass}")
print()
print(f"  {'Metric':<35} {'Logistic Reg':>14} {'Random Forest':>14} {'Neural Net':>11}")
print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*11}")
print(f"  {'Fail Precision':<35} {fp_log:>14.4f} {fp_rf:>14.4f} {fp_nn:>11.4f}")
print(f"  {'Fail Recall (at-risk detection rate)':<35} {fr_log:>14.4f} {fr_rf:>14.4f} {fr_nn:>11.4f}")
print(f"  {'Fail F1':<35} {ff_log:>14.4f} {ff_rf:>14.4f} {ff_nn:>11.4f}")
print(f"  {'Failing students correctly caught':<35} {tn_l:>14} {tn_r:>14} {tn_n:>11}")
print(f"  {'Failing students missed':<35} {fn_l:>14} {fn_r:>14} {fn_n:>11}")
print()
print("  Note: 'missed' failing students are the most concerning outcome")
print("  in an educational context — these are at-risk students overlooked.")
print()

# =============================================================================
# PLOT 11 — ROC CURVES
# =============================================================================
# ROC curve plots True Positive Rate vs False Positive Rate at every threshold.
# AUC (Area Under Curve): 1.0 = perfect, 0.5 = random guessing.
# Higher AUC = better model regardless of the classification threshold.

fpr_log, tpr_log, _ = roc_curve(y_clf_test, y_prob_log)
fpr_rf,  tpr_rf,  _ = roc_curve(y_clf_test, y_prob_rf)
fpr_nn,  tpr_nn,  _ = roc_curve(y_clf_test, y_prob_nn)
auc_log = auc(fpr_log, tpr_log)
auc_rf  = auc(fpr_rf,  tpr_rf)
auc_nn  = auc(fpr_nn,  tpr_nn)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr_log, tpr_log, color="#4C72B0", linewidth=2,
        label=f"Logistic Regression (AUC = {auc_log:.3f})")
ax.plot(fpr_rf,  tpr_rf,  color="#55A868", linewidth=2,
        label=f"Random Forest       (AUC = {auc_rf:.3f})")
ax.plot(fpr_nn,  tpr_nn,  color="#C44E52", linewidth=2,
        label=f"Neural Network      (AUC = {auc_nn:.3f})")
ax.plot([0, 1], [0, 1], color="#888888", linestyle="--",
        linewidth=1.2, label="Random baseline (AUC = 0.500)")
ax.fill_between(fpr_log, tpr_log, alpha=0.07, color="#4C72B0")
ax.fill_between(fpr_rf,  tpr_rf,  alpha=0.07, color="#55A868")
ax.fill_between(fpr_nn,  tpr_nn,  alpha=0.07, color="#C44E52")
ax.set_title("ROC Curves — All Classifiers (Mathematics)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
plt.tight_layout()
plt.savefig("../plots/mat_plot11_roc_curves.png")
plt.show()
print("Plot 11 saved: ../plots/mat_plot11_roc_curves.png")

# =============================================================================
# PLOT 12 — PRECISION-RECALL CURVES
# =============================================================================
# More informative than ROC when classes are imbalanced (as here — 67% pass).
# Shows the trade-off between precision and recall at different thresholds.
# AP (Average Precision) summarises the curve as a single number.

prec_log_curve, rec_log_curve, _ = precision_recall_curve(y_clf_test, y_prob_log)
prec_rf_curve,  rec_rf_curve,  _ = precision_recall_curve(y_clf_test, y_prob_rf)
prec_nn_curve,  rec_nn_curve,  _ = precision_recall_curve(y_clf_test, y_prob_nn)
ap_log = average_precision_score(y_clf_test, y_prob_log)
ap_rf  = average_precision_score(y_clf_test, y_prob_rf)
ap_nn  = average_precision_score(y_clf_test, y_prob_nn)
baseline_pr = y_clf_test.mean()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec_log_curve, prec_log_curve, color="#4C72B0", linewidth=2,
        label=f"Logistic Regression (AP = {ap_log:.3f})")
ax.plot(rec_rf_curve,  prec_rf_curve,  color="#55A868", linewidth=2,
        label=f"Random Forest       (AP = {ap_rf:.3f})")
ax.plot(rec_nn_curve,  prec_nn_curve,  color="#C44E52", linewidth=2,
        label=f"Neural Network      (AP = {ap_nn:.3f})")
ax.axhline(y=baseline_pr, color="#888888", linestyle="--", linewidth=1.2,
           label=f"Baseline (AP = {baseline_pr:.3f})")
ax.set_title("Precision-Recall Curves — All Classifiers",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.11)
plt.tight_layout()
plt.savefig("../plots/mat_plot12_precision_recall.png")
plt.show()
print("Plot 12 saved: ../plots/mat_plot12_precision_recall.png")
print()

# =============================================================================
# PART 3 — FEATURE IMPORTANCE SUMMARY (top 5 for thesis)
# =============================================================================

LABELS = {
    "age": "Age", "Medu": "Mother's Education", "Fedu": "Father's Education",
    "traveltime": "Travel Time to School", "studytime": "Weekly Study Time",
    "failures": "Past Class Failures", "famrel": "Family Relationship Quality",
    "freetime": "Free Time After School", "goout": "Going Out with Friends",
    "Dalc": "Workday Alcohol Consumption", "Walc": "Weekend Alcohol Consumption",
    "health": "Current Health Status", "absences": "Number of Absences",
}

importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances.index = [LABELS.get(c, c) for c in importances.index]
top5 = importances.sort_values(ascending=False).head(5)

print("─" * 65)
print("PART 3: Top 5 most important features (Random Forest)")
print("─" * 65)
for i, (feat, score) in enumerate(top5.items(), 1):
    print(f"  {i}. {feat:<35} {score:.4f}")
print()

# =============================================================================
# PART 4 — STRUCTURED THESIS CONCLUSION
# =============================================================================

print("=" * 65)
print("PART 4: Thesis Conclusion Summary")
print("=" * 65)
print()
print("RESEARCH QUESTION:")
print("  Can student academic performance be predicted using background")
print("  factors (demographics, study habits, family situation)?")
print()
print("KEY FINDINGS:")
print()
print("  1. Linear relationships are weak (R² = 0.14).")
print("     Background variables alone cannot reliably predict the exact")
print("     final grade. This is expected: without prior grades (G1, G2),")
print("     the task is genuinely difficult.")
print()
print("  2. Classification (pass/fail) is more feasible than regression.")
print(f"     Logistic Regression achieved F1 = {m_log['f1']:.2f} and accuracy")
print(f"     = {m_log['accuracy']:.2f}, beating the naive baseline of 0.671.")
print()
print("  3. Logistic Regression outperformed Random Forest and Neural Network")
print("     on this dataset. With only 395 students and weak non-linear signals,")
print("     a simpler model generalised better. The Neural Network requires more")
print("     data to realise its potential — a known limitation on small datasets.")
print(f"     CV F1 scores: LR={cv_log.mean():.2f}, RF={cv_rf.mean():.2f}, NN={cv_nn.mean():.2f}")
print()
print("  4. Failing students are the hardest to detect.")
print(f"     Logistic Regression identified {tn_l}/{total_fail} failing students")
print(f"     (recall = {fr_log:.2f}). This is the main limitation of the models")
print("     and the most educationally significant finding.")
print()
print("  5. Past class failures is the strongest predictor of final grade,")
print("     followed by parental education and alcohol consumption.")
print("     Study time and absences showed surprisingly weak linear effects.")
print()
print("LIMITATIONS:")
print("  - Small dataset (395 students, single school, Math subject only)")
print("  - Class imbalance (67% pass) biases models toward predicting pass")
print("  - No prior grade data used (intentional — avoids data leakage)")
print("  - Neural Networks require larger datasets to reach full potential")
print("  - Results may not generalise to other schools or subjects")
print()
print("FUTURE WORK:")
print("  - Apply SMOTE or class weighting to improve Fail class detection")
print("  - Include Portuguese language dataset for cross-subject comparison")
print("  - Test deeper Neural Network architectures with more data")
print("  - Test additional models: SVM, Gradient Boosting")
print("  - Adjust classification threshold to prioritise Fail recall")
print()

# =============================================================================
# SAVE FINAL REPORT CSV
# =============================================================================

report = pd.DataFrame([
    {
        "Model":     "Linear Regression",
        "Task":      "Regression (predict G3)",
        "R²":        round(r2, 4),
        "MAE":       round(mae, 4),
        "RMSE":      round(rmse, 4),
        "Accuracy":  "—",
        "Precision": "—",
        "Recall":    "—",
        "F1":        "—",
        "CV Mean":   round(cv_r2.mean(), 4),
        "CV Std":    round(cv_r2.std(), 4),
        "AUC-ROC":   "—",
    },
    {
        "Model":     "Logistic Regression",
        "Task":      "Classification (pass/fail)",
        "R²":        "—",
        "MAE":       "—",
        "RMSE":      "—",
        "Accuracy":  round(m_log["accuracy"], 4),
        "Precision": round(m_log["precision"], 4),
        "Recall":    round(m_log["recall"], 4),
        "F1":        round(m_log["f1"], 4),
        "CV Mean":   round(cv_log.mean(), 4),
        "CV Std":    round(cv_log.std(), 4),
        "AUC-ROC":   round(auc_log, 4),
    },
    {
        "Model":     "Random Forest",
        "Task":      "Classification (pass/fail)",
        "R²":        "—",
        "MAE":       "—",
        "RMSE":      "—",
        "Accuracy":  round(m_rf["accuracy"], 4),
        "Precision": round(m_rf["precision"], 4),
        "Recall":    round(m_rf["recall"], 4),
        "F1":        round(m_rf["f1"], 4),
        "CV Mean":   round(cv_rf.mean(), 4),
        "CV Std":    round(cv_rf.std(), 4),
        "AUC-ROC":   round(auc_rf, 4),
    },
    {
        "Model":     "Neural Network",
        "Task":      "Classification (pass/fail)",
        "R²":        "—",
        "MAE":       "—",
        "RMSE":      "—",
        "Accuracy":  round(m_nn["accuracy"], 4),
        "Precision": round(m_nn["precision"], 4),
        "Recall":    round(m_nn["recall"], 4),
        "F1":        round(m_nn["f1"], 4),
        "CV Mean":   round(cv_nn.mean(), 4),
        "CV Std":    round(cv_nn.std(), 4),
        "AUC-ROC":   round(auc_nn, 4),
    },
])

report.to_csv("../results/mat_results_final_report.csv", index=False)

print("=" * 65)
print("PHASE 5 COMPLETE")
print("=" * 65)
print("Files saved:")
print("  ../plots/mat_plot11_roc_curves.png")
print("  ../plots/mat_plot12_precision_recall.png")
print("  ../results/mat_results_final_report.csv")
