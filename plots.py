# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Phase 2: Exploratory Data Analysis (EDA)
# =============================================================================
# Run this AFTER phase1_data_setup.py.
# Make sure student_preprocessed.csv is in the same folder.
# This script produces 6 plots saved as PNG files.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── SETUP ─────────────────────────────────────────────────────────────────────
# Load the preprocessed dataset from Phase 1
df = pd.read_csv("student_preprocessed.csv")

# Also reload the raw dataset for plots that use original column names
df_raw = pd.read_csv("student-mat.csv", sep=";")
df_raw["pass_fail"] = (df_raw["G3"] >= 10).astype(int)

# Set a consistent visual style for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150

print("EDA started — generating 6 plots...")
print()

# ── PLOT 1: Distribution of Final Grades (G3) ─────────────────────────────────
# This is the most important plot in your EDA section.
# It shows whether grades are normally distributed, skewed, or bimodal.
# Note the spike at 0 — students who dropped out or were absent for the exam.

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(df_raw["G3"], bins=21, range=(-0.5, 20.5),
        color="#4C72B0", edgecolor="white", linewidth=0.6)
ax.axvline(x=10, color="#C44E52", linestyle="--", linewidth=1.5,
           label="Pass threshold (10)")
ax.axvline(x=df_raw["G3"].mean(), color="#55A868", linestyle="--", linewidth=1.5,
           label=f"Mean grade ({df_raw['G3'].mean():.1f})")

ax.set_title("Distribution of Final Grades (G3)", fontsize=14, fontweight="bold")
ax.set_xlabel("Final Grade (G3)", fontsize=12)
ax.set_ylabel("Number of Students", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("plot1_grade_distribution.png")
plt.show()
print("Plot 1 saved: plot1_grade_distribution.png")

# ── PLOT 2: Pass vs Fail Count ────────────────────────────────────────────────
# Simple bar chart showing class balance.
# If one class is much larger than the other, it affects model evaluation.
# Mention this in your thesis — class imbalance matters for precision/recall.

fig, ax = plt.subplots(figsize=(5, 5))

counts = df_raw["pass_fail"].value_counts().sort_index()
bars = ax.bar(["Fail (0)", "Pass (1)"], counts.values,
              color=["#C44E52", "#55A868"], edgecolor="white", width=0.5)

# Add count labels on top of each bar
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(count), ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_title("Pass vs Fail Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Students", fontsize=12)
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig("plot2_pass_fail.png")
plt.show()
print("Plot 2 saved: plot2_pass_fail.png")

# ── PLOT 3: Study Time vs Final Grade ─────────────────────────────────────────
# studytime is coded 1–4 (1=<2h, 2=2–5h, 3=5–10h, 4=>10h per week).
# A boxplot shows both the median and spread per group.
# This directly supports your hypothesis that study time affects grades.

studytime_labels = {1: "<2h", 2: "2–5h", 3: "5–10h", 4: ">10h"}
df_raw["studytime_label"] = df_raw["studytime"].map(studytime_labels)
order = ["<2h", "2–5h", "5–10h", ">10h"]

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df_raw, x="studytime_label", y="G3",
            order=order, palette="Blues", ax=ax)
ax.axhline(y=10, color="#C44E52", linestyle="--", linewidth=1.2,
           label="Pass threshold")
ax.set_title("Study Time vs Final Grade (G3)", fontsize=14, fontweight="bold")
ax.set_xlabel("Weekly Study Time", fontsize=12)
ax.set_ylabel("Final Grade (G3)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("plot3_studytime_grade.png")
plt.show()
print("Plot 3 saved: plot3_studytime_grade.png")

# ── PLOT 4: Absences vs Final Grade ───────────────────────────────────────────
# Scatterplot with a regression line to show the trend.
# More absences typically correlates with lower grades.
# The spread shows it is not a perfect predictor — interesting to discuss.

fig, ax = plt.subplots(figsize=(7, 5))
sns.regplot(data=df_raw, x="absences", y="G3",
            scatter_kws={"alpha": 0.4, "s": 40, "color": "#4C72B0"},
            line_kws={"color": "#C44E52", "linewidth": 2},
            ax=ax)
ax.set_title("Absences vs Final Grade (G3)", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Absences", fontsize=12)
ax.set_ylabel("Final Grade (G3)", fontsize=12)
plt.tight_layout()
plt.savefig("plot4_absences_grade.png")
plt.show()
print("Plot 4 saved: plot4_absences_grade.png")

# ── PLOT 5: Correlation Heatmap ────────────────────────────────────────────────
# Shows correlations between all numeric variables.
# Key insight: G1 and G2 will be very highly correlated with G3.
# This justifies your decision (from Phase 1) to exclude them from features.
# Other notable correlations to discuss: studytime, failures, Medu (mother's education).

numeric_cols = ["G1", "G2", "G3", "age", "absences", "studytime",
                "failures", "Medu", "Fedu", "famrel", "freetime",
                "goout", "Dalc", "Walc", "health"]

corr_matrix = df_raw[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # hide upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            annot_kws={"size": 8}, ax=ax)
ax.set_title("Correlation Heatmap — Numeric Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot5_correlation_heatmap.png")
plt.show()
print("Plot 5 saved: plot5_correlation_heatmap.png")

# ── PLOT 6: Parent Education vs Final Grade ────────────────────────────────────
# Medu = mother's education level (0=none, 1=primary, 2=5th–9th grade,
#                                   3=secondary, 4=higher education)
# Research consistently shows parental education influences student outcomes.
# This is a good variable to highlight in your demographic analysis section.

fig, ax = plt.subplots(figsize=(7, 5))
medu_labels = {0: "None", 1: "Primary", 2: "Middle", 3: "Secondary", 4: "Higher"}
df_raw["Medu_label"] = df_raw["Medu"].map(medu_labels)
medu_order = ["None", "Primary", "Middle", "Secondary", "Higher"]

sns.boxplot(data=df_raw, x="Medu_label", y="G3",
            order=medu_order, palette="Greens", ax=ax)
ax.axhline(y=10, color="#C44E52", linestyle="--", linewidth=1.2,
           label="Pass threshold")
ax.set_title("Mother's Education Level vs Final Grade (G3)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Mother's Education Level", fontsize=12)
ax.set_ylabel("Final Grade (G3)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("plot6_medu_grade.png")
plt.show()
print("Plot 6 saved: plot6_medu_grade.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PHASE 2 COMPLETE — EDA Summary")
print("=" * 60)
print(f"  Mean final grade (G3):   {df_raw['G3'].mean():.2f} / 20")
print(f"  Std deviation:           {df_raw['G3'].std():.2f}")
print(f"  Pass rate:               {df_raw['pass_fail'].mean() * 100:.1f}%")
print(f"  Highest corr. with G3:   G2 ({corr_matrix['G3']['G2']:.2f}), "
      f"G1 ({corr_matrix['G3']['G1']:.2f})")
print(f"  Absences–G3 correlation: {corr_matrix['G3']['absences']:.2f}")
print(f"  Studytime–G3 correlation:{corr_matrix['G3']['studytime']:.2f}")
print()
print("All 6 plots saved as PNG files in your project folder.")
print("Ready for Phase 3: Statistical Analysis")