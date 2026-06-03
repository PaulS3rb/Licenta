# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Portuguese Dataset — Phase 3: Statistical Analysis
# =============================================================================
# Run this AFTER por_phase1_data_setup.py and por_phase2_eda.py.
# Reads from:  ../data/student-por.csv
# Writes to:   ../results/por_results_correlations.csv
#              ../results/por_results_hypothesis_tests.csv
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats

# ── SETUP ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/student-por.csv", sep=";")
df["pass_fail"] = (df["G3"] >= 10).astype(int)

# ── FULL LABEL MAP ─────────────────────────────────────────────────────────────
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
    "sex":        "Sex",
    "address":    "Home Address Type",
    "famsup":     "Family Educational Support",
    "paid":       "Paid Extra Classes",
    "internet":   "Internet Access at Home",
    "romantic":   "In a Romantic Relationship",
    "schoolsup":  "Extra School Support",
    "activities": "Extracurricular Activities",
    "higher":     "Wants Higher Education",
}

print("=" * 65)
print("PORTUGUESE — PHASE 3: STATISTICAL ANALYSIS")
print("=" * 65)
print()

# =============================================================================
# PART 1 — PEARSON CORRELATION ANALYSIS
# =============================================================================

print("─" * 65)
print("PART 1: Pearson Correlation — numeric variables vs G3")
print("─" * 65)
print(f"{'Variable':<30} {'r':>8} {'p-value':>12} {'Significant':>14} {'Strength':>12}")
print("-" * 65)

numeric_vars = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences"
]

correlation_results = []

for var in numeric_vars:
    r, p = stats.pearsonr(df[var], df["G3"])

    significant = "Yes ***" if p < 0.001 else ("Yes **" if p < 0.01 else ("Yes *" if p < 0.05 else "No"))

    abs_r = abs(r)
    if abs_r >= 0.50:       strength = "Strong"
    elif abs_r >= 0.30:     strength = "Moderate"
    elif abs_r >= 0.10:     strength = "Weak"
    else:                   strength = "Negligible"

    direction = "positive" if r > 0 else "negative"
    label     = LABELS.get(var, var)

    print(f"{label:<30} {r:>8.4f} {p:>12.4f} {significant:>14} {strength:>12}")

    correlation_results.append({
        "Variable":    label,
        "r":           round(r, 4),
        "p-value":     round(p, 4),
        "Significant": "Yes" if p < 0.05 else "No",
        "Strength":    strength,
        "Direction":   direction
    })

print()
print("Significance levels: * p<0.05  ** p<0.01  *** p<0.001")
print()

corr_df = pd.DataFrame(correlation_results)
top3 = corr_df.reindex(corr_df["r"].abs().sort_values(ascending=False).index).head(3)
print("Top 3 predictors (by |r|):")
for _, row in top3.iterrows():
    print(f"  {row['Variable']}: r = {row['r']} ({row['Strength']}, {row['Direction']})")
print()

# =============================================================================
# PART 2 — HYPOTHESIS TESTS: BINARY GROUPS vs G3
# =============================================================================

print("=" * 65)
print("PART 2: Hypothesis Tests — group differences in G3")
print("=" * 65)
print()

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std()**2 + (n2 - 1) * group2.std()**2) / (n1 + n2 - 2)
    )
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std != 0 else 0

def interpret_d(d):
    abs_d = abs(d)
    if abs_d >= 0.80:   return "Large"
    elif abs_d >= 0.50: return "Medium"
    elif abs_d >= 0.20: return "Small"
    else:               return "Negligible"

binary_tests = [
    ("sex",        "F",   "M",   "Female",                "Male"),
    ("address",    "U",   "R",   "Urban",                 "Rural"),
    ("famsup",     "yes", "no",  "Family support",        "No family support"),
    ("paid",       "yes", "no",  "Paid tutoring",         "No tutoring"),
    ("internet",   "yes", "no",  "Internet at home",      "No internet at home"),
    ("romantic",   "yes", "no",  "In a relationship",     "Not in a relationship"),
    ("schoolsup",  "yes", "no",  "Extra school support",  "No school support"),
    ("activities", "yes", "no",  "Extracurricular act.",  "No activities"),
    ("higher",     "yes", "no",  "Wants higher education","Does not want higher edu."),
]

hypothesis_results = []

for var, val1, val2, label1, label2 in binary_tests:
    g1 = df[df[var] == val1]["G3"]
    g2 = df[df[var] == val2]["G3"]

    t_stat, t_p = stats.ttest_ind(g1, g2)
    u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    d           = cohens_d(g1, g2)
    d_label     = interpret_d(d)
    var_label   = LABELS.get(var, var)

    print(f"Variable: {var_label}")
    print(f"  Groups:        {label1}  vs  {label2}")
    print(f"  Mean G3:       {label1}: {g1.mean():.2f}  |  {label2}: {g2.mean():.2f}")
    print(f"  t-test:        t = {t_stat:.3f},  p = {t_p:.4f}")
    print(f"  Mann-Whitney:  U = {u_stat:.1f},  p = {u_p:.4f}")
    print(f"  Cohen's d:     {d:.3f}  ({d_label} effect)")
    print(f"  Significant:   {'Yes' if t_p < 0.05 else 'No'}")
    print()

    hypothesis_results.append({
        "Variable":            var_label,
        "Group 1":             label1,
        "Group 2":             label2,
        "Mean G3 (Group 1)":   round(g1.mean(), 2),
        "Mean G3 (Group 2)":   round(g2.mean(), 2),
        "t-statistic":         round(t_stat, 3),
        "t p-value":           round(t_p, 4),
        "Mann-Whitney p":      round(u_p, 4),
        "Cohen's d":           round(d, 3),
        "Effect Size":         d_label,
        "Significant":         "Yes" if t_p < 0.05 else "No"
    })

# =============================================================================
# PART 3 — NORMALITY CHECK
# =============================================================================

print("=" * 65)
print("PART 3: Normality check — Shapiro-Wilk test on G3")
print("=" * 65)
w_stat, p_norm = stats.shapiro(df["G3"])
print(f"  Shapiro-Wilk: W = {w_stat:.4f},  p = {p_norm:.6f}")
if p_norm < 0.05:
    print("  Result: G3 is NOT normally distributed (p < 0.05).")
    print("  → This justifies using Mann-Whitney U (non-parametric) in your thesis.")
else:
    print("  Result: G3 appears normally distributed (p >= 0.05).")
print()

# =============================================================================
# PART 4 — SAVE RESULTS AS CSV
# =============================================================================

corr_df.to_csv("../results/por_results_correlations.csv", index=False)
hyp_df = pd.DataFrame(hypothesis_results)
hyp_df.to_csv("../results/por_results_hypothesis_tests.csv", index=False)

print("=" * 65)
print("PORTUGUESE PHASE 3 COMPLETE")
print("=" * 65)
print("Files saved:")
print("  ../results/por_results_correlations.csv")
print("  ../results/por_results_hypothesis_tests.csv")
print()
