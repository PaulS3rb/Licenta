# =============================================================================
# BACHELOR THESIS - Student Performance Prediction
# Portuguese Dataset — Phase 4b: SMOTE
# =============================================================================
# Run this AFTER por_phase4_ml_models.py.
# Reads from:  ../data/student_por_preprocessed.csv
# Writes to:   ../plots/por_plot_smote_comparison.png
#              ../plots/por_plot_smote_confusion.png
#              ../results/por_results_smote.csv
#
# Note: The Portuguese dataset has a more severe class imbalance (84.6% pass)
# with only 100 failing students total. SMOTE is expected to have an even
# larger effect here than in Mathematics.
#
# Install imbalanced-learn if needed:
#   pip install imbalanced-learn --break-system-packages
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ── SETUP ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/student_por_preprocessed.csv")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150

# ── FEATURES & TARGETS ────────────────────────────────────────────────────────
exclude_cols = ["G3", "pass_fail", "G1", "G2"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X     = df[feature_cols]
y_clf = df["pass_fail"]

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print("=" * 65)
print("PHASE 4b: SMOTE — CLASS IMBALANCE MITIGATION (Portuguese)")
print("=" * 65)
print()
print("Original training set class distribution:")
print(f"  Pass (1): {(y_train == 1).sum()}  ({(y_train == 1).mean()*100:.1f}%)")
print(f"  Fail (0): {(y_train == 0).sum()}  ({(y_train == 0).mean()*100:.1f}%)")
print()

# ── APPLY SMOTE ───────────────────────────────────────────────────────────────
# k_neighbors=5 is the SMOTE default. With only ~80 failing students in the
# training set, 5 neighbours is appropriate — going higher risks poor
# interpolation in sparse regions of the feature space.

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("After SMOTE training set class distribution:")
print(f"  Pass (1): {(y_train_sm == 1).sum()}  ({(y_train_sm == 1).mean()*100:.1f}%)")
print(f"  Fail (0): {(y_train_sm == 0).sum()}  ({(y_train_sm == 0).mean()*100:.1f}%)")
print(f"  Total samples: {len(y_train_sm)}  (was {len(y_train)})")
print()
print("Test set is unchanged:")
print(f"  Pass (1): {(y_test == 1).sum()}  |  Fail (0): {(y_test == 0).sum()}")
print()

# ── SCALING ───────────────────────────────────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

scaler_orig     = StandardScaler()
X_train_orig_sc = scaler_orig.fit_transform(X_train)
X_test_orig_sc  = scaler_orig.transform(X_test)

# ── MODELS ────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": {
        "orig":  LogisticRegression(max_iter=1000, random_state=42),
        "smote": LogisticRegression(max_iter=1000, random_state=42),
    },
    "Random Forest": {
        "orig":  RandomForestClassifier(n_estimators=100, random_state=42),
        "smote": RandomForestClassifier(n_estimators=100, random_state=42),
    },
    "Neural Network": {
        "orig":  MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                               max_iter=2000, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=20,
                               random_state=42),
        "smote": MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                               max_iter=2000, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=20,
                               random_state=42),
    },
}

models["Logistic Regression"]["orig"].fit(X_train_orig_sc, y_train)
models["Logistic Regression"]["smote"].fit(X_train_scaled, y_train_sm)
models["Random Forest"]["orig"].fit(X_train, y_train)
models["Random Forest"]["smote"].fit(X_train_sm, y_train_sm)
models["Neural Network"]["orig"].fit(X_train_orig_sc, y_train)
models["Neural Network"]["smote"].fit(X_train_scaled, y_train_sm)

preds = {
    "Logistic Regression": {
        "orig":  models["Logistic Regression"]["orig"].predict(X_test_orig_sc),
        "smote": models["Logistic Regression"]["smote"].predict(X_test_scaled),
    },
    "Random Forest": {
        "orig":  models["Random Forest"]["orig"].predict(X_test),
        "smote": models["Random Forest"]["smote"].predict(X_train_sm if False else X_test),
    },
    "Neural Network": {
        "orig":  models["Neural Network"]["orig"].predict(X_test_orig_sc),
        "smote": models["Neural Network"]["smote"].predict(X_test_scaled),
    },
}

# Fix Random Forest SMOTE predictions
preds["Random Forest"]["smote"] = models["Random Forest"]["smote"].predict(X_test)

# ── METRICS ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fail_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "f1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fail_recall": round(fail_recall, 4),
        "fail_caught": int(tn),
        "fail_missed": int(fn),
    }

results = []
model_names = list(models.keys())
total_fail  = (y_test == 0).sum()

print("─" * 65)
print("RESULTS COMPARISON: Original vs SMOTE")
print("─" * 65)

for name in model_names:
    m_orig  = metrics(y_test, preds[name]["orig"])
    m_smote = metrics(y_test, preds[name]["smote"])
    print(f"\n{name}:")
    print(f"  {'Metric':<25} {'Original':>10} {'SMOTE':>10} {'Δ':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for key, label in [("accuracy","Accuracy"), ("f1","F1 Score"),
                        ("fail_recall","Fail Recall"), ("fail_caught","Fail Caught")]:
        orig_val  = m_orig[key]
        smote_val = m_smote[key]
        delta     = smote_val - orig_val
        sign      = "+" if delta >= 0 else ""
        print(f"  {label:<25} {orig_val:>10.4f} {smote_val:>10.4f} {sign}{delta:>7.4f}")

    results.append({"Model": name, "Version": "Original",
                     **{k: m_orig[k] for k in ["accuracy","f1","fail_recall",
                                                "fail_caught","fail_missed"]}})
    results.append({"Model": name, "Version": "SMOTE",
                     **{k: m_smote[k] for k in ["accuracy","f1","fail_recall",
                                                  "fail_caught","fail_missed"]}})

print()

# ── PLOT 1: Comparison ────────────────────────────────────────────────────────
orig_recall  = [metrics(y_test, preds[n]["orig"])["fail_recall"]  for n in model_names]
smote_recall = [metrics(y_test, preds[n]["smote"])["fail_recall"] for n in model_names]
orig_f1      = [metrics(y_test, preds[n]["orig"])["f1"]           for n in model_names]
smote_f1     = [metrics(y_test, preds[n]["smote"])["f1"]          for n in model_names]

x = np.arange(len(model_names))
width = 0.32

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
b1 = ax.bar(x - width/2, orig_recall,  width, label="Original",  color="#4C72B0", edgecolor="white")
b2 = ax.bar(x + width/2, smote_recall, width, label="With SMOTE", color="#C44E52", edgecolor="white")
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_title("Fail Class Recall\n(at-risk student detection rate)", fontsize=12, fontweight="bold")
ax.set_ylabel("Fail Recall", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(["Log.\nRegression", "Random\nForest", "Neural\nNetwork"], fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)

ax = axes[1]
b3 = ax.bar(x - width/2, orig_f1,  width, label="Original",  color="#4C72B0", edgecolor="white")
b4 = ax.bar(x + width/2, smote_f1, width, label="With SMOTE", color="#C44E52", edgecolor="white")
for bars in [b3, b4]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_title("Pass Class F1 Score", fontsize=12, fontweight="bold")
ax.set_ylabel("F1 Score (Pass class)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(["Log.\nRegression", "Random\nForest", "Neural\nNetwork"], fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)

fig.suptitle("Effect of SMOTE on Classification Performance — Portuguese",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../plots/por_plot_smote_comparison.png", bbox_inches="tight")
plt.show()
print("Plot saved: ../plots/por_plot_smote_comparison.png")

# ── PLOT 2: Confusion matrices — Logistic Regression ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, y_pred, title in zip(
    axes,
    [preds["Logistic Regression"]["orig"], preds["Logistic Regression"]["smote"]],
    ["Logistic Regression\n(Original)", "Logistic Regression\n(With SMOTE)"]
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"],
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)

fig.suptitle("Confusion Matrices — Logistic Regression: Original vs SMOTE (Portuguese)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("../plots/por_plot_smote_confusion.png")
plt.show()
print("Plot saved: ../plots/por_plot_smote_confusion.png")

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv("../results/por_results_smote.csv", index=False)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("PHASE 4b COMPLETE — SMOTE Summary (Portuguese)")
print("=" * 65)
print(f"  Total failing students in test set: {total_fail}")
print()
print(f"  {'Model':<22} {'Orig Fail Recall':>17} {'SMOTE Fail Recall':>18} {'Δ Recall':>10}")
print(f"  {'-'*22} {'-'*17} {'-'*18} {'-'*10}")
for name in model_names:
    m_o = metrics(y_test, preds[name]["orig"])
    m_s = metrics(y_test, preds[name]["smote"])
    delta = m_s["fail_recall"] - m_o["fail_recall"]
    sign  = "+" if delta >= 0 else ""
    print(f"  {name:<22} {m_o['fail_recall']:>17.4f} {m_s['fail_recall']:>18.4f} "
          f"{sign}{delta:>9.4f}")
print()
print("Files saved:")
print("  ../plots/por_plot_smote_comparison.png")
print("  ../plots/por_plot_smote_confusion.png")
print("  ../results/por_results_smote.csv")