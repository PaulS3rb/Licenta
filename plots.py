# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Phase 2: Exploratory Data Analysis (EDA)
# =============================================================================
# Run this AFTER phase1_data_setup.py.
# Make sure student-mat.csv and student_preprocessed.csv are in the same folder.
# This script produces 6 plots saved as PNG files.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── SETUP ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("student_preprocessed.csv")

df_raw = pd.read_csv("student-mat.csv", sep=";")
df_raw["pass_fail"] = (df_raw["G3"] >= 10).astype(int)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150

# ── FULL LABEL MAP ─────────────────────────────────────────────────────────────
# Single source of truth for all axis labels, titles, and legends.
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
    "G1":         "First Period Grade",
    "G2":         "Second Period Grade",
    "G3":         "Final Grade (G3)",
}

print("EDA started — generating 6 plots...")
print()

# ── PLOT 1: Distribution of Final Grades (G3) ─────────────────────────────────
# Shows whether grades are normally distributed, skewed, or bimodal.
# Note the spike at 0 — students who dropped out or were absent for the exam.

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(df_raw["G3"], bins=21, range=(-0.5, 20.5),
        color="#4C72B0", edgecolor="white", linewidth=0.6)
ax.axvline(x=10, color="#C44E52", linestyle="--", linewidth=1.5,
           label="Pass threshold (10)")
ax.axvline(x=df_raw["G3"].mean(), color="#55A868", linestyle="--", linewidth=1.5,
           label=f"Mean grade ({df_raw['G3'].mean():.1f})")

ax.set_title("Distribution of Final Grades (G3)", fontsize=14, fontweight="bold")
ax.set_xlabel(LABELS["G3"], fontsize=12)
ax.set_ylabel("Number of Students", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("plot1_grade_distribution.png")
plt.show()
print("Plot 1 saved: plot1_grade_distribution.png")

# ── PLOT 2: Pass vs Fail Count ────────────────────────────────────────────────
# Simple bar chart showing class balance.
# Class imbalance affects model evaluation — discuss in thesis.

fig, ax = plt.subplots(figsize=(5, 5))

counts = df_raw["pass_fail"].value_counts().sort_index()
bars = ax.bar(["Fail (Grade < 10)", "Pass (Grade ≥ 10)"], counts.values,
              color=["#C44E52", "#55A868"], edgecolor="white", width=0.5)

for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(count), ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_title("Pass vs Fail Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Outcome", fontsize=12)
ax.set_ylabel("Number of Students", fontsize=12)
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig("plot2_pass_fail.png")
plt.show()
print("Plot 2 saved: plot2_pass_fail.png")

# ── PLOT 3: Study Time vs Final Grade ─────────────────────────────────────────
# studytime is coded 1–4 (1=<2h, 2=2–5h, 3=5–10h, 4=>10h per week).
# A boxplot shows both the median and spread per group.

studytime_labels = {1: "Less than 2h", 2: "2 to 5h", 3: "5 to 10h", 4: "More than 10h"}
df_raw["studytime_label"] = df_raw["studytime"].map(studytime_labels)
order = ["Less than 2h", "2 to 5h", "5 to 10h", "More than 10h"]

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_raw, x="studytime_label", y="G3",
            order=order, palette="Blues", ax=ax)
ax.axhline(y=10, color="#C44E52", linestyle="--", linewidth=1.2,
           label="Pass threshold (10)")
ax.set_title(f"{LABELS['studytime']} vs {LABELS['G3']}", fontsize=14, fontweight="bold")
ax.set_xlabel(LABELS["studytime"], fontsize=12)
ax.set_ylabel(LABELS["G3"], fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("plot3_studytime_grade.png")
plt.show()
print("Plot 3 saved: plot3_studytime_grade.png")

# ── PLOT 4: Absences vs Final Grade ───────────────────────────────────────────
# Scatterplot with a regression line to show the trend.
# Near-zero correlation (r=0.03) — discuss non-linear effects in thesis.

fig, ax = plt.subplots(figsize=(7, 5))
sns.regplot(data=df_raw, x="absences", y="G3",
            scatter_kws={"alpha": 0.4, "s": 40, "color": "#4C72B0"},
            line_kws={"color": "#C44E52", "linewidth": 2},
            ax=ax)
ax.set_title(f"{LABELS['absences']} vs {LABELS['G3']}", fontsize=14, fontweight="bold")
ax.set_xlabel(LABELS["absences"], fontsize=12)
ax.set_ylabel(LABELS["G3"], fontsize=12)
plt.tight_layout()
plt.savefig("plot4_absences_grade.png")
plt.show()
print("Plot 4 saved: plot4_absences_grade.png")

# ── PLOT 5: Correlation Heatmap ────────────────────────────────────────────────
# Shows correlations between all numeric variables.
# Key insight: G1 and G2 are very highly correlated with G3 — justifies exclusion.

numeric_cols = ["G1", "G2", "G3", "age", "absences", "studytime",
                "failures", "Medu", "Fedu", "famrel", "freetime",
                "goout", "Dalc", "Walc", "health"]

# Build a renamed copy for the heatmap so axis labels are readable
df_heatmap = df_raw[numeric_cols].copy()
df_heatmap.rename(columns=LABELS, inplace=True)

corr_matrix = df_heatmap.corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation Heatmap — Numeric Variables", fontsize=14, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("plot5_correlation_heatmap.png")
plt.show()
print("Plot 5 saved: plot5_correlation_heatmap.png")

# ── PLOT 6: Mother's Education vs Final Grade ──────────────────────────────────
# Medu: 0=none, 1=primary, 2=5th–9th grade, 3=secondary, 4=higher education.
# Parental education is a well-documented predictor of student outcomes.

medu_labels = {
    0: "None",
    1: "Primary school",
    2: "Middle school",
    3: "Secondary school",
    4: "Higher education"
}
df_raw["Medu_label"] = df_raw["Medu"].map(medu_labels)
medu_order = ["None", "Primary school", "Middle school", "Secondary school", "Higher education"]

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_raw, x="Medu_label", y="G3",
            order=medu_order, palette="Greens", ax=ax)
ax.axhline(y=10, color="#C44E52", linestyle="--", linewidth=1.2,
           label="Pass threshold (10)")
ax.set_title(f"{LABELS['Medu']} vs {LABELS['G3']}", fontsize=14, fontweight="bold")
ax.set_xlabel(LABELS["Medu"], fontsize=12)
ax.set_ylabel(LABELS["G3"], fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("plot6_medu_grade.png")
plt.show()
print("Plot 6 saved: plot6_medu_grade.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
corr_matrix_raw = df_raw[numeric_cols].corr()
print()
print("=" * 60)
print("PHASE 2 COMPLETE — EDA Summary")
print("=" * 60)
print(f"  Mean final grade (G3):        {df_raw['G3'].mean():.2f} / 20")
print(f"  Std deviation:                {df_raw['G3'].std():.2f}")
print(f"  Pass rate:                    {df_raw['pass_fail'].mean() * 100:.1f}%")
print(f"  Highest corr. with G3:        Second Period Grade ({corr_matrix_raw['G3']['G2']:.2f}), "
      f"First Period Grade ({corr_matrix_raw['G3']['G1']:.2f})")
print(f"  Number of Absences — G3 corr: {corr_matrix_raw['G3']['absences']:.2f}")
print(f"  Weekly Study Time  — G3 corr: {corr_matrix_raw['G3']['studytime']:.2f}")
print()
print("All 6 plots saved as PNG files in your project folder.")
print("Ready for Phase 3: Statistical Analysis")