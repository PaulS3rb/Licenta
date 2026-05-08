# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Phase 3: Statistical Analysis
# =============================================================================
# Run this AFTER phase1_data_setup.py and phase2_eda.py.
# Make sure student-mat.csv and student_preprocessed.csv are in the same folder.
# This script performs:
#   1. Pearson correlation analysis (numeric variables vs G3)
#   2. Point-biserial correlation (binary variables vs G3)
#   3. Hypothesis tests: t-tests and Mann-Whitney U tests
#   4. Saves a summary table as CSV for your thesis
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats

# ── SETUP ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("student-mat.csv", sep=";")
df["pass_fail"] = (df["G3"] >= 10).astype(int)

print("=" * 65)
print("PHASE 3: STATISTICAL ANALYSIS")
print("=" * 65)
print()

# =============================================================================
# PART 1 — PEARSON CORRELATION ANALYSIS
# =============================================================================
# Pearson's r measures the LINEAR relationship between two numeric variables.
# r ranges from -1 (perfect negative) to +1 (perfect positive).
# We also get a p-value: if p < 0.05, the correlation is statistically
# significant (i.e. unlikely to be due to random chance).
#
# We test all numeric variables against G3 (excluding G1 and G2 — see thesis).
# =============================================================================

print("─" * 65)
print("PART 1: Pearson Correlation — numeric variables vs G3")
print("─" * 65)
print(f"{'Variable':<20} {'r':>8} {'p-value':>12} {'Significant':>14} {'Strength':>12}")
print("-" * 65)

numeric_vars = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences"
]

correlation_results = []

for var in numeric_vars:
    r, p = stats.pearsonr(df[var], df["G3"])

    # Determine significance
    significant = "Yes ***" if p < 0.001 else ("Yes **" if p < 0.01 else ("Yes *" if p < 0.05 else "No"))

    # Interpret strength of correlation (Cohen's conventions)
    abs_r = abs(r)
    if abs_r >= 0.50:
        strength = "Strong"
    elif abs_r >= 0.30:
        strength = "Moderate"
    elif abs_r >= 0.10:
        strength = "Weak"
    else:
        strength = "Negligible"

    direction = "positive" if r > 0 else "negative"

    print(f"{var:<20} {r:>8.4f} {p:>12.4f} {significant:>14} {strength:>12}")

    correlation_results.append({
        "variable": var,
        "r": round(r, 4),
        "p_value": round(p, 4),
        "significant": "Yes" if p < 0.05 else "No",
        "strength": strength,
        "direction": direction
    })

print()
print("Significance levels: * p<0.05  ** p<0.01  *** p<0.001")
print()

# Top 3 most correlated features (useful for thesis discussion)
corr_df = pd.DataFrame(correlation_results)
top3 = corr_df.reindex(corr_df["r"].abs().sort_values(ascending=False).index).head(3)
print("Top 3 predictors (by |r|):")
for _, row in top3.iterrows():
    print(f"  {row['variable']}: r = {row['r']} ({row['strength']}, {row['direction']})")
print()

# =============================================================================
# PART 2 — HYPOTHESIS TESTS: BINARY GROUPS vs G3
# =============================================================================
# For binary/categorical variables (e.g. sex, internet access), we split
# students into two groups and test whether their G3 means differ significantly.
#
# We use TWO tests:
#   t-test        → assumes normal distribution (parametric)
#   Mann-Whitney U → does NOT assume normal distribution (non-parametric)
# Reporting both is good academic practice. If they agree → robust finding.
#
# Effect size: Cohen's d tells you HOW LARGE the difference is, not just
# whether it's statistically significant. Small: 0.2, Medium: 0.5, Large: 0.8
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
    if abs_d >= 0.80: return "Large"
    elif abs_d >= 0.50: return "Medium"
    elif abs_d >= 0.20: return "Small"
    else: return "Negligible"

# Variables to test (binary or easily split into two groups)
binary_tests = [
    ("sex",        "F",  "M",   "Female",     "Male"),
    ("address",    "U",  "R",   "Urban",      "Rural"),
    ("famsup",     "yes","no",  "Family sup.", "No family sup."),
    ("paid",       "yes","no",  "Paid tutoring","No tutoring"),
    ("internet",   "yes","no",  "Internet",   "No internet"),
    ("romantic",   "yes","no",  "Romantic rel.","No rel."),
    ("schoolsup",  "yes","no",  "School sup.", "No school sup."),
    ("activities", "yes","no",  "Activities", "No activities"),
    ("higher",     "yes","no",  "Wants higher edu.","Does not"),
]

hypothesis_results = []

for var, val1, val2, label1, label2 in binary_tests:
    g1 = df[df[var] == val1]["G3"]
    g2 = df[df[var] == val2]["G3"]

    t_stat, t_p = stats.ttest_ind(g1, g2)
    u_stat, u_p  = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    d            = cohens_d(g1, g2)
    d_label      = interpret_d(d)
    sig          = "Yes *" if t_p < 0.05 else "No"

    print(f"Variable: {var}  ({label1} vs {label2})")
    print(f"  Mean G3 — {label1}: {g1.mean():.2f}  |  {label2}: {g2.mean():.2f}")
    print(f"  t-test:        t = {t_stat:.3f},  p = {t_p:.4f}")
    print(f"  Mann-Whitney:  U = {u_stat:.1f},  p = {u_p:.4f}")
    print(f"  Cohen's d:     {d:.3f}  ({d_label} effect)")
    print(f"  Significant:   {sig}")
    print()

    hypothesis_results.append({
        "variable": var,
        "group1": label1,
        "group2": label2,
        "mean_g1": round(g1.mean(), 2),
        "mean_g2": round(g2.mean(), 2),
        "t_stat": round(t_stat, 3),
        "t_p_value": round(t_p, 4),
        "mannwhitney_p": round(u_p, 4),
        "cohens_d": round(d, 3),
        "effect_size": d_label,
        "significant_t": "Yes" if t_p < 0.05 else "No"
    })

# =============================================================================
# PART 3 — NORMALITY CHECK (for thesis completeness)
# =============================================================================
# Before using parametric tests (t-test), you should check if G3 is normally
# distributed. The Shapiro-Wilk test does this.
# H0: the data IS normally distributed.
# If p < 0.05 → reject H0 → data is NOT normal → non-parametric tests preferred.
# This justifies reporting Mann-Whitney U alongside the t-test.
# =============================================================================

print("=" * 65)
print("PART 3: Normality check — Shapiro-Wilk test on G3")
print("=" * 65)
stat, p = stats.shapiro(df["G3"])
print(f"  Shapiro-Wilk: W = {stat:.4f},  p = {p:.6f}")
if p < 0.05:
    print("  Result: G3 is NOT normally distributed (p < 0.05).")
    print("  → This justifies using Mann-Whitney U (non-parametric) in your thesis.")
else:
    print("  Result: G3 appears normally distributed (p >= 0.05).")
print()

# =============================================================================
# PART 4 — SAVE RESULTS AS CSV
# =============================================================================
# Save both tables as CSV files — useful for pasting into your thesis document.

corr_df.to_csv("results_correlations.csv", index=False)
hyp_df = pd.DataFrame(hypothesis_results)
hyp_df.to_csv("results_hypothesis_tests.csv", index=False)

print("=" * 65)
print("PHASE 3 COMPLETE")
print("=" * 65)
print("Files saved:")
print("  results_correlations.csv     — Pearson correlation table")
print("  results_hypothesis_tests.csv — Hypothesis test table")
print()
print("Both tables can be pasted directly into your thesis Results section.")
print("Ready for Phase 4: Machine Learning Models")