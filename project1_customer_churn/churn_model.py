"""
PROJECT 1: Customer Churn Prediction
Category  : Supervised Learning — Binary Classification
Algorithm : Random Forest Classifier

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY RANDOM FOREST?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alternatives considered:
  • Logistic Regression   → too linear, misses complex interactions
  • Decision Tree         → unstable, overfits on small changes in data
  • SVM                   → slow to train on large data, hard to interpret
  • XGBoost               → better performance but harder to explain to business

Random Forest wins here because:
  1. Handles MIXED data (numeric + categorical) without heavy preprocessing
  2. Built-in FEATURE IMPORTANCE tells business teams what actually drives churn
  3. CLASS WEIGHT='balanced' handles churn imbalance (~25% churn is common)
  4. Robust to outliers — one bad record won't throw off 200 trees
  5. No need for feature scaling (trees split on thresholds, not distances)
  6. Business-interpretable: "contract type is the #1 churn driver"

DATA FIX NOTES:
  • Increased dataset to 2000 rows (was 1000) for better AUC
  • Strengthened feature-label signal relationships for realistic ~0.80 AUC
  • Added interaction feature: charges_per_service
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)
import joblib

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# ─────────────────────────────────────────────
# C. DUMMY DATA GENERATION (fixed for realistic AUC)
# ─────────────────────────────────────────────
def generate_churn_data(n=2000, seed=42):
    """
    Synthetic telecom churn dataset.
    Signal strength tuned to produce realistic AUC ~0.78–0.83.
    Reduced noise std from 0.05 to 0.03; strengthened coefficient weights.
    """
    np.random.seed(seed)

    tenure          = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges   = np.round(monthly_charges * tenure + np.random.normal(0, 30, n), 2)
    num_services    = np.random.randint(1, 8, n)
    contract_type   = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
    )
    payment_method  = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )
    tech_support    = np.random.choice([0, 1], n)
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
    num_complaints  = np.random.poisson(1.0, n)

    # Stronger, cleaner signal for realistic ~0.80 AUC
    churn_prob = (
        0.50 * (contract_type == "Month-to-month").astype(float)
        + 0.25 * (num_complaints > 2).astype(float)
        + 0.20 * (monthly_charges > 85).astype(float)
        - 0.25 * (tenure > 36).astype(float)
        - 0.15 * tech_support
        + 0.10 * (senior_citizen == 1).astype(float)
        - 0.10 * (num_services > 4).astype(float)
        + np.random.normal(0, 0.03, n)   # reduced noise
    )
    churn_prob = np.clip(churn_prob, 0.03, 0.97)
    churn      = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "num_services":     num_services,
        "contract_type":    contract_type,
        "payment_method":   payment_method,
        "tech_support":     tech_support,
        "senior_citizen":   senior_citizen,
        "num_complaints":   num_complaints,
        "churn":            churn,
    })


# ─────────────────────────────────────────────
# D. MODEL IMPLEMENTATION
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    df["contract_type"]  = le.fit_transform(df["contract_type"])
    df["payment_method"] = le.fit_transform(df["payment_method"])
    # Interaction feature: high bill + low tenure = high risk
    df["charges_per_month_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)
    return df


def train_churn_model(df):
    df = preprocess(df)
    X  = df.drop("churn", axis=1)
    y  = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    print("=" * 60)
    print("PROJECT 1 — Customer Churn Prediction (Random Forest)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="roc_auc")
    print(f"ROC-AUC (test)     : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"5-Fold CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(ARTIFACT_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "churn_scaler.pkl"))

    return model, scaler, X_train_s, X_test_s, y_train, y_test, y_proba, X.columns.tolist()


# ─────────────────────────────────────────────
# E. EXPLAINABILITY VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    std         = np.std([t.feature_importances_ for t in model.estimators_], axis=0)
    idx         = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(importances)))
    ax.bar(range(len(importances)), importances[idx],
           yerr=std[idx], color=colors, capsize=4)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title("Feature Importance — What Drives Churn?\n"
                 "(Error bars = variance across 300 trees)", fontsize=13)
    for i in range(3):
        ax.text(i, importances[idx[i]] + std[idx[i]] + 0.002,
                f"#{i+1}", ha="center", color="darkred", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "churn_feature_importance.png"), dpi=130)
    plt.close()
    print("Saved: churn_feature_importance.png")


def plot_roc_and_pr(y_test, y_proba):
    fpr, tpr, _  = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr  = average_precision_score(y_test, y_proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AUC = {auc_roc:.3f}")
    ax1.fill_between(fpr, tpr, alpha=0.1, color="orange")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    ax1.set_xlabel("False Positive Rate\n(% loyal customers wrongly flagged)", fontsize=10)
    ax1.set_ylabel("True Positive Rate\n(% churners correctly caught)", fontsize=10)
    ax1.set_title("ROC Curve\n← Curve closer to top-left = better model", fontsize=11)
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(rec, prec, color="steelblue", lw=2.5, label=f"Avg Precision = {auc_pr:.3f}")
    ax2.fill_between(rec, prec, alpha=0.1, color="blue")
    baseline = float(np.array(y_test).mean())
    ax2.axhline(baseline, color="gray", lw=1, linestyle="--", label=f"Baseline = {baseline:.2f}")
    ax2.set_xlabel("Recall (% of all churners found)", fontsize=10)
    ax2.set_ylabel("Precision (% of flagged that really churn)", fontsize=10)
    ax2.set_title("Precision-Recall Curve\n(Better metric for imbalanced classes)", fontsize=11)
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Model Performance Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "churn_roc_pr.png"), dpi=130)
    plt.close()
    print("Saved: churn_roc_pr.png")


def plot_confusion_annotated(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix with Business Meaning", fontsize=12, pad=15)
    notes = [
        [f"TN={tn}\n✅ Correctly ignored\n(No wasted spend)", f"FP={fp}\n⚠️ False alarm\n(Wasted discount)"],
        [f"FN={fn}\n❌ Missed churner\n(Lost revenue)",       f"TP={tp}\n💰 Caught churner\n(Retention win)"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.38, notes[i][j], ha="center", va="center", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7))
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "churn_confusion_matrix.png"), dpi=130)
    plt.close()
    print("Saved: churn_confusion_matrix.png")


def plot_learning_curve(model, X, y):
    train_sizes, train_sc, test_sc = learning_curve(
        model, X, y, cv=5, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_sc.mean(1), "o-", color="#e74c3c", label="Training AUC")
    ax.fill_between(train_sizes, train_sc.mean(1) - train_sc.std(1),
                    train_sc.mean(1) + train_sc.std(1), alpha=0.15, color="#e74c3c")
    ax.plot(train_sizes, test_sc.mean(1), "o-", color="#2ecc71", label="Validation AUC")
    ax.fill_between(train_sizes, test_sc.mean(1) - test_sc.std(1),
                    test_sc.mean(1) + test_sc.std(1), alpha=0.15, color="#2ecc71")
    ax.set_xlabel("Training Set Size", fontsize=11)
    ax.set_ylabel("ROC-AUC Score", fontsize=11)
    ax.set_title("Learning Curve — Does more data help?", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "churn_learning_curve.png"), dpi=130)
    plt.close()
    print("Saved: churn_learning_curve.png")


def plot_churn_drivers(df):
    df = preprocess(df)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    ct = df.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
    axes[0].bar(["Month-to-Month", "Two Year", "One Year"], ct.values * 100,
                color=["#e74c3c", "#2ecc71", "#f39c12"])
    axes[0].set_ylabel("Churn Rate (%)"); axes[0].set_title("Churn Rate by Contract Type")
    for i, v in enumerate(ct.values * 100):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    axes[1].hist(df[df["churn"]==0]["tenure"], bins=20, alpha=0.6, color="#2ecc71",
                 label="Stay", density=True)
    axes[1].hist(df[df["churn"]==1]["tenure"], bins=20, alpha=0.6, color="#e74c3c",
                 label="Churn", density=True)
    axes[1].set_xlabel("Tenure (months)"); axes[1].set_title("Tenure by Churn Status")
    axes[1].legend(); axes[1].set_ylabel("Density")

    axes[2].hist(df[df["churn"]==0]["monthly_charges"], bins=20, alpha=0.6,
                 color="#2ecc71", label="Stay", density=True)
    axes[2].hist(df[df["churn"]==1]["monthly_charges"], bins=20, alpha=0.6,
                 color="#e74c3c", label="Churn", density=True)
    axes[2].set_xlabel("Monthly Charges ($)"); axes[2].set_title("Charges by Churn Status")
    axes[2].legend(); axes[2].set_ylabel("Density")

    df["complaint_bin"] = pd.cut(df["num_complaints"], bins=[-1,0,1,2,10],
                                  labels=["0","1","2","3+"])
    cb = df.groupby("complaint_bin", observed=True)["churn"].mean() * 100
    axes[3].bar(cb.index, cb.values, color=["#2ecc71","#f39c12","#e67e22","#e74c3c"])
    axes[3].set_xlabel("Complaints"); axes[3].set_ylabel("Churn Rate (%)")
    axes[3].set_title("More Complaints → More Churn")
    for i, v in enumerate(cb.values):
        axes[3].text(i, v + 0.5, f"{v:.1f}%", ha="center")

    ts = df.groupby("tech_support")["churn"].mean() * 100
    axes[4].bar(["No Support","Has Support"], ts.values, color=["#e74c3c","#2ecc71"])
    axes[4].set_ylabel("Churn Rate (%)"); axes[4].set_title("Tech Support Reduces Churn")
    for i, v in enumerate(ts.values):
        axes[4].text(i, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    cc = df["churn"].value_counts()
    axes[5].pie(cc, labels=["Stay","Churn"], colors=["#2ecc71","#e74c3c"],
                autopct="%1.1f%%", startangle=90, wedgeprops={"width":0.5})
    axes[5].set_title("Overall Churn Split")

    plt.suptitle("Churn Analysis — Understanding the Business Problem",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "churn_eda.png"), dpi=130)
    plt.close()
    print("Saved: churn_eda.png")


def plot_single_tree(model, feature_names):
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(model.estimators_[0], feature_names=feature_names,
              class_names=["Stay","Churn"], filled=True, max_depth=3,
              impurity=False, ax=ax, fontsize=9)
    ax.set_title("One Decision Tree from the Forest (depth=3 shown)\n"
                 "Blue=Stay · Orange=Churn · Darker=More confident", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "churn_single_tree.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("Saved: churn_single_tree.png")


def plot_churn_probability_distribution(y_proba, y_test):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(y_proba[y_test==0], bins=30, alpha=0.65, color="#2ecc71",
            label="Actual: Stay", density=True)
    ax.hist(y_proba[y_test==1], bins=30, alpha=0.65, color="#e74c3c",
            label="Actual: Churn", density=True)
    ax.axvline(0.5, color="black", lw=2, linestyle="--", label="Threshold (0.5)")
    ax.set_xlabel("Predicted Churn Probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Churn Probability Distribution\nBetter model = two peaks clearly separated",
                 fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "churn_prob_dist.png"), dpi=130)
    plt.close()
    print("Saved: churn_prob_dist.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_churn_data(2000)
    print(f"Dataset: {df.shape}  |  Churn rate: {df['churn'].mean():.1%}")

    model, scaler, X_train_s, X_test_s, y_train, y_test, y_proba, features = train_churn_model(df)
    y_pred = model.predict(X_test_s)

    plot_churn_drivers(df)
    plot_feature_importance(model, features)
    plot_roc_and_pr(y_test, y_proba)
    plot_confusion_annotated(y_test, y_pred)
    plot_churn_probability_distribution(y_proba, y_test.values)

    df_pre = preprocess(df)
    X_all  = scaler.transform(df_pre.drop("churn", axis=1))
    plot_learning_curve(model, X_all, df_pre["churn"])
    plot_single_tree(model, features)

    print("\n✅ All Project 1 artifacts saved.")
