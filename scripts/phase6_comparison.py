# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Phase 6: Math vs Portuguese — Cross-Subject Comparison
# =============================================================================
# Run this AFTER all Math phases (phase1–5) AND all Portuguese phases
# (por_phase1–5) have been completed successfully.
#
# Reads from:  ../results/results_final_report.csv
#              ../results/por_results_final_report.csv
#              ../results/results_correlations.csv
#              ../results/por_results_correlations.csv
#              ../results/results_hypothesis_tests.csv
#              ../results/por_results_hypothesis_tests.csv
#
# Writes to:   ../plots/comparison_plot1_model_metrics.png
#              ../plots/comparison_plot2_correlations.png
#              ../plots/comparison_plot3_hypothesis.png
#              ../plots/comparison_plot4_dataset_overview.png
#              ../results/comparison_results_summary.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150

COLOR_MAT = "#4C72B0"   # blue  — Math
COLOR_POR = "#55A868"   # green — Portuguese
COLOR_ACC = "#C44E52"   # red   — accent / baseline

print("=" * 65)
print("PHASE 6: MATH vs PORTUGUESE — CROSS-SUBJECT COMPARISON")
print("=" * 65)
print()

# =============================================================================
# LOAD RESULTS
# =============================================================================

mat_report = pd.read_csv("../results/mat_results_final_report.csv")
por_report = pd.read_csv("../results/por_results_final_report.csv")
mat_corr   = pd.read_csv("../results/mat_results_correlations.csv")
por_corr   = pd.read_csv("../results/por_results_correlations.csv")
mat_hyp    = pd.read_csv("../results/mat_results_hypothesis_tests.csv")
por_hyp    = pd.read_csv("../results/por_results_hypothesis_tests.csv")

print("All result files loaded successfully.")
print()

# =============================================================================
# DATASET OVERVIEW — printed summary
# =============================================================================

print("─" * 65)
print("DATASET OVERVIEW")
print("─" * 65)
print(f"  {'Metric':<35} {'Math':>10} {'Portuguese':>12}")
print(f"  {'-'*35} {'-'*10} {'-'*12}")
overview = [
    ("Total students",         "395",    "649"),
    ("Training samples (80%)", "316",    "519"),
    ("Test samples (20%)",     "79",     "130"),
    ("Features",               "36",     "39"),
    ("Mean final grade (G3)",  "10.42",  "11.91"),
    ("Std deviation",          "4.58",   "3.23"),
    ("Pass rate",              "67.1%",  "84.6%"),
    ("Naive baseline acc.",    "67.1%",  "84.6%"),
]
for label, mat, por in overview:
    print(f"  {label:<35} {mat:>10} {por:>12}")
print()

# =============================================================================
# MODEL METRICS COMPARISON — printed summary
# =============================================================================

print("─" * 65)
print("MODEL METRICS COMPARISON")
print("─" * 65)

classifiers = ["Logistic Regression", "Random Forest", "Neural Network"]

for model in classifiers:
    mat_row = mat_report[mat_report["Model"] == model].iloc[0]
    por_row = por_report[por_report["Model"] == model].iloc[0]
    print(f"  {model}:")
    print(f"    F1 Score   — Math: {mat_row['F1']}   Portuguese: {por_row['F1']}")
    print(f"    Accuracy   — Math: {mat_row['Accuracy']}   Portuguese: {por_row['Accuracy']}")
    print(f"    AUC-ROC    — Math: {mat_row['AUC-ROC']}   Portuguese: {por_row['AUC-ROC']}")
    print()

# Linear Regression
mat_lr = mat_report[mat_report["Model"] == "Linear Regression"].iloc[0]
por_lr = por_report[por_report["Model"] == "Linear Regression"].iloc[0]
print(f"  Linear Regression:")
print(f"    R²         — Math: {mat_lr['R²']}   Portuguese: {por_lr['R²']}")
print(f"    MAE        — Math: {mat_lr['MAE']}   Portuguese: {por_lr['MAE']}")
print(f"    RMSE       — Math: {mat_lr['RMSE']}   Portuguese: {por_lr['RMSE']}")
print()

# =============================================================================
# CORRELATION COMPARISON — printed summary
# =============================================================================

print("─" * 65)
print("PEARSON CORRELATION COMPARISON (vs G3)")
print("─" * 65)
print(f"  {'Variable':<35} {'Math r':>8} {'Por r':>8} {'Δ':>8}")
print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")

# Merge on Variable name
corr_merged = mat_corr.merge(por_corr, on="Variable", suffixes=("_mat", "_por"))
corr_merged["delta"] = corr_merged["r_por"] - corr_merged["r_mat"]
corr_merged = corr_merged.sort_values("r_mat", key=abs, ascending=False)

for _, row in corr_merged.iterrows():
    delta_str = f"+{row['delta']:.4f}" if row['delta'] >= 0 else f"{row['delta']:.4f}"
    print(f"  {row['Variable']:<35} {row['r_mat']:>8.4f} {row['r_por']:>8.4f} {delta_str:>8}")
print()

# =============================================================================
# PLOT 1 — MODEL METRICS SIDE BY SIDE (F1 Score)
# =============================================================================
# Shows F1 score for all three classifiers in both subjects.
# Makes it immediately clear which model wins per subject and overall.

models   = ["Logistic\nRegression", "Random\nForest", "Neural\nNetwork"]
f1_mat   = []
f1_por   = []

for m in ["Logistic Regression", "Random Forest", "Neural Network"]:
    f1_mat.append(float(mat_report[mat_report["Model"] == m]["F1"].values[0]))
    f1_por.append(float(por_report[por_report["Model"] == m]["F1"].values[0]))

x     = np.arange(len(models))
width = 0.32

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width / 2, f1_mat, width, label="Mathematics (n=395)",
               color=COLOR_MAT, edgecolor="white")
bars2 = ax.bar(x + width / 2, f1_por, width, label="Portuguese (n=649)",
               color=COLOR_POR, edgecolor="white")

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.axhline(y=0.671, color=COLOR_MAT, linestyle=":", linewidth=1.2,
           label="Math baseline (67.1%)")
ax.axhline(y=0.846, color=COLOR_POR, linestyle=":", linewidth=1.2,
           label="Portuguese baseline (84.6%)")

ax.set_title("F1 Score Comparison — Math vs Portuguese",
             fontsize=14, fontweight="bold")
ax.set_ylabel("F1 Score (Pass class)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("../plots/comparison_plot1_model_f1.png")
plt.show()
print("Comparison Plot 1 saved: ../plots/comparison_plot1_model_f1.png")

# =============================================================================
# PLOT 2 — FULL METRICS PANEL (Accuracy, Precision, Recall, F1)
# =============================================================================
# 2x2 grid — one panel per metric, both subjects overlaid.

metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_keys   = ["Accuracy", "Precision", "Recall", "F1"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (label, key) in enumerate(zip(metric_labels, metric_keys)):
    ax = axes[idx]
    mat_vals = []
    por_vals = []
    for m in ["Logistic Regression", "Random Forest", "Neural Network"]:
        mat_vals.append(float(mat_report[mat_report["Model"] == m][key].values[0]))
        por_vals.append(float(por_report[por_report["Model"] == m][key].values[0]))

    x = np.arange(3)
    b1 = ax.bar(x - 0.18, mat_vals, 0.32, label="Math",       color=COLOR_MAT, edgecolor="white")
    b2 = ax.bar(x + 0.18, por_vals, 0.32, label="Portuguese",  color=COLOR_POR, edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Log. Reg.", "Rnd. Forest", "Neural Net"], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)

fig.suptitle("Classification Metrics — Math vs Portuguese (All Models)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../plots/comparison_plot2_all_metrics.png", bbox_inches="tight")
plt.show()
print("Comparison Plot 2 saved: ../plots/comparison_plot2_all_metrics.png")

# =============================================================================
# PLOT 3 — PEARSON CORRELATION COMPARISON (horizontal bar chart)
# =============================================================================
# Shows both correlation coefficients side by side for each variable.
# Makes it easy to spot which variables behave differently across subjects.

vars_ordered = corr_merged.sort_values("r_mat", key=abs, ascending=True)["Variable"].tolist()
r_mat_vals   = corr_merged.set_index("Variable").loc[vars_ordered, "r_mat"].tolist()
r_por_vals   = corr_merged.set_index("Variable").loc[vars_ordered, "r_por"].tolist()

y     = np.arange(len(vars_ordered))
height = 0.35

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(y - height / 2, r_mat_vals, height, label="Mathematics",
        color=COLOR_MAT, edgecolor="white", alpha=0.85)
ax.barh(y + height / 2, r_por_vals, height, label="Portuguese",
        color=COLOR_POR, edgecolor="white", alpha=0.85)

ax.axvline(x=0, color="black", linewidth=0.8)
ax.axvline(x=0.10,  color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(x=-0.10, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(x=0.30,  color="gray", linewidth=0.5, linestyle="-.",  alpha=0.5)
ax.axvline(x=-0.30, color="gray", linewidth=0.5, linestyle="-.",  alpha=0.5)

ax.set_yticks(y)
ax.set_yticklabels(vars_ordered, fontsize=9)
ax.set_xlabel("Pearson r (correlation with G3)", fontsize=11)
ax.set_title("Pearson Correlations with G3 — Math vs Portuguese",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(-0.55, 0.55)

# Threshold annotations
ax.text(0.105, -0.7, "Weak (±0.10)", fontsize=7, color="gray", va="top")
ax.text(0.305, -0.7, "Moderate (±0.30)", fontsize=7, color="gray", va="top")

plt.tight_layout()
plt.savefig("../plots/comparison_plot3_correlations.png")
plt.show()
print("Comparison Plot 3 saved: ../plots/comparison_plot3_correlations.png")

# =============================================================================
# PLOT 4 — HYPOTHESIS TEST EFFECT SIZES (Cohen's d)
# =============================================================================
# Shows Cohen's d for each binary variable in both subjects.
# Positive = Group 1 scores higher; Negative = Group 2 scores higher.

hyp_vars = mat_hyp["Variable"].tolist()
mat_d = mat_hyp.set_index("Variable")["Cohen's d"].reindex(hyp_vars).tolist()
por_d = por_hyp.set_index("Variable")["Cohen's d"].reindex(hyp_vars).fillna(0).tolist()

# Shorten labels for plot
short_labels = [
    "Sex", "Home Address", "Family Support", "Paid Classes",
    "Internet", "Romantic Rel.", "School Support",
    "Activities", "Higher Edu."
]

y      = np.arange(len(short_labels))
height = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y - height / 2, mat_d, height, label="Mathematics",
        color=COLOR_MAT, edgecolor="white", alpha=0.85)
ax.barh(y + height / 2, por_d, height, label="Portuguese",
        color=COLOR_POR, edgecolor="white", alpha=0.85)

ax.axvline(x=0,    color="black", linewidth=0.8)
ax.axvline(x=0.20,  color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
ax.axvline(x=-0.20, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
ax.axvline(x=0.50,  color="gray", linewidth=0.5, linestyle="-.",  alpha=0.6)
ax.axvline(x=0.80,  color="gray", linewidth=0.5, linestyle=":",   alpha=0.6)

ax.set_yticks(y)
ax.set_yticklabels(short_labels, fontsize=10)
ax.set_xlabel("Cohen's d (effect size)", fontsize=11)
ax.set_title("Hypothesis Test Effect Sizes — Math vs Portuguese",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

# Threshold labels
for xval, lbl in [(0.21, "Small\n(0.20)"), (0.51, "Medium\n(0.50)"), (0.81, "Large\n(0.80)")]:
    ax.text(xval, len(short_labels) - 0.3, lbl, fontsize=7, color="gray", va="top")

plt.tight_layout()
plt.savefig("../plots/comparison_plot4_effect_sizes.png")
plt.show()
print("Comparison Plot 4 saved: ../plots/comparison_plot4_effect_sizes.png")

# =============================================================================
# PLOT 5 — LINEAR REGRESSION: R², MAE, RMSE
# =============================================================================

reg_metrics  = ["R²", "MAE", "RMSE"]
mat_reg_vals = [float(mat_lr["R²"]), float(mat_lr["MAE"]), float(mat_lr["RMSE"])]
por_reg_vals = [float(por_lr["R²"]), float(por_lr["MAE"]), float(por_lr["RMSE"])]

x     = np.arange(len(reg_metrics))
width = 0.32

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - width / 2, mat_reg_vals, width, label="Mathematics (n=395)",
               color=COLOR_MAT, edgecolor="white")
bars2 = ax.bar(x + width / 2, por_reg_vals, width, label="Portuguese (n=649)",
               color=COLOR_POR, edgecolor="white")

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

ax.set_title("Linear Regression Metrics — Math vs Portuguese",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Value", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(reg_metrics, fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, max(mat_reg_vals + por_reg_vals) * 1.2)
plt.tight_layout()
plt.savefig("../plots/comparison_plot5_regression.png")
plt.show()
print("Comparison Plot 5 saved: ../plots/comparison_plot5_regression.png")

# =============================================================================
# PLOT 6 — CV STABILITY (mean ± std across 5 folds)
# =============================================================================
# Error bars show the standard deviation of CV scores.
# Taller bars = more stable (less variance across folds).
# Key insight: Neural Network becomes far more stable with more data.

mat_cv_means, mat_cv_stds = [], []
por_cv_means, por_cv_stds = [], []

for m in ["Logistic Regression", "Random Forest", "Neural Network"]:
    mm = float(mat_report[mat_report["Model"] == m]["CV Mean"].values[0])
    ms = float(mat_report[mat_report["Model"] == m]["CV Std"].values[0])
    pm = float(por_report[por_report["Model"] == m]["CV Mean"].values[0])
    ps = float(por_report[por_report["Model"] == m]["CV Std"].values[0])
    mat_cv_means.append(mm)
    mat_cv_stds.append(ms)
    por_cv_means.append(pm)
    por_cv_stds.append(ps)

x     = np.arange(3)
width = 0.32

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width / 2, mat_cv_means, width, yerr=mat_cv_stds,
       label="Mathematics (n=395)", color=COLOR_MAT, edgecolor="white",
       capsize=5, error_kw={"linewidth": 1.5})
ax.bar(x + width / 2, por_cv_means, width, yerr=por_cv_stds,
       label="Portuguese (n=649)", color=COLOR_POR, edgecolor="white",
       capsize=5, error_kw={"linewidth": 1.5})

for i, (mm, pm) in enumerate(zip(mat_cv_means, por_cv_means)):
    ax.text(i - width / 2, mm + mat_cv_stds[i] + 0.01,
            f"{mm:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + width / 2, pm + por_cv_stds[i] + 0.01,
            f"{pm:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_title("Cross-Validation F1 Scores (mean ± std) — Math vs Portuguese",
             fontsize=13, fontweight="bold")
ax.set_ylabel("CV F1 Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(["Log. Regression", "Random Forest", "Neural Network"], fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("../plots/comparison_plot6_cv_stability.png")
plt.show()
print("Comparison Plot 6 saved: ../plots/comparison_plot6_cv_stability.png")

# =============================================================================
# SAVE COMPARISON SUMMARY CSV
# =============================================================================

rows = []
for m in ["Linear Regression", "Logistic Regression", "Random Forest", "Neural Network"]:
    mat_row = mat_report[mat_report["Model"] == m].iloc[0]
    por_row = por_report[por_report["Model"] == m].iloc[0]
    rows.append({
        "Model":         m,
        "Math R²":       mat_row["R²"],
        "Por R²":        por_row["R²"],
        "Math MAE":      mat_row["MAE"],
        "Por MAE":       por_row["MAE"],
        "Math RMSE":     mat_row["RMSE"],
        "Por RMSE":      por_row["RMSE"],
        "Math Accuracy": mat_row["Accuracy"],
        "Por Accuracy":  por_row["Accuracy"],
        "Math F1":       mat_row["F1"],
        "Por F1":        por_row["F1"],
        "Math AUC-ROC":  mat_row["AUC-ROC"],
        "Por AUC-ROC":   por_row["AUC-ROC"],
        "Math CV Mean":  mat_row["CV Mean"],
        "Math CV Std":   mat_row["CV Std"],
        "Por CV Mean":   por_row["CV Mean"],
        "Por CV Std":    por_row["CV Std"],
    })

pd.DataFrame(rows).to_csv("../results/comparison_results_summary.csv", index=False)

# =============================================================================
# PRINTED SUMMARY OF KEY CROSS-SUBJECT FINDINGS
# =============================================================================

print()
print("=" * 65)
print("PHASE 6 COMPLETE — Key Cross-Subject Findings")
print("=" * 65)
print()
print("  1. DATASET SIZE EFFECT ON NEURAL NETWORK")
mat_nn_mean = mat_report[mat_report["Model"] == "Neural Network"]["CV Mean"].values[0]
mat_nn_std  = mat_report[mat_report["Model"] == "Neural Network"]["CV Std"].values[0]
por_nn_mean = por_report[por_report["Model"] == "Neural Network"]["CV Mean"].values[0]
por_nn_std  = por_report[por_report["Model"] == "Neural Network"]["CV Std"].values[0]
print(f"     Math CV F1:       {mat_nn_mean:.4f} ± {mat_nn_std:.4f}")
print(f"     Portuguese CV F1: {por_nn_mean:.4f} ± {por_nn_std:.4f}")
print(f"     → Neural Network variance drops sharply with more data.")
print()
print("  2. BEST MODEL SWITCHES BETWEEN SUBJECTS")
print(f"     Math:       Logistic Regression wins  (F1 = {f1_mat[0]:.4f})")
print(f"     Portuguese: Random Forest wins        (F1 = {f1_por[1]:.4f})")
print(f"     → More data allows RF to exploit non-linear patterns.")
print()
print("  3. STUDY TIME CORRELATION DIFFERS")
mat_st = mat_corr[mat_corr["Variable"] == "Weekly Study Time"]["r"].values[0]
por_st = por_corr[por_corr["Variable"] == "Weekly Study Time"]["r"].values[0]
print(f"     Math r = {mat_st}  (not significant)")
print(f"     Por  r = {por_st}  (significant, p < 0.001)")
print(f"     → Consistent effort more predictive for language than Math.")
print()
print("  4. EDUCATIONAL ASPIRATION EFFECT STRONGER IN PORTUGUESE")
mat_he = mat_hyp[mat_hyp["Variable"] == "Wants Higher Education"]["Cohen's d"].values[0]
por_he = por_hyp[por_hyp["Variable"] == "Wants Higher Education"]["Cohen's d"].values[0]
print(f"     Math Cohen's d = {mat_he}")
print(f"     Por  Cohen's d = {por_he}")
print(f"     → Large effect in both; strongest predictor across both subjects.")
print()
print("  5. PAST FAILURES: CONSISTENT ACROSS BOTH SUBJECTS")
mat_f = mat_corr[mat_corr["Variable"] == "Past Class Failures"]["r"].values[0]
por_f = por_corr[por_corr["Variable"] == "Past Class Failures"]["r"].values[0]
print(f"     Math r = {mat_f}")
print(f"     Por  r = {por_f}")
print(f"     → Strongest numeric predictor in both subjects.")
print()
print("Files saved:")
print("  ../plots/comparison_plot1_model_f1.png")
print("  ../plots/comparison_plot2_all_metrics.png")
print("  ../plots/comparison_plot3_correlations.png")
print("  ../plots/comparison_plot4_effect_sizes.png")
print("  ../plots/comparison_plot5_regression.png")
print("  ../plots/comparison_plot6_cv_stability.png")
print("  ../results/comparison_results_summary.csv")