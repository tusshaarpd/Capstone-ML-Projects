"""
PROJECT 1: Customer Churn Prediction
Category: Supervised Learning (Classification)
Algorithm: Random Forest Classifier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import joblib
import os

# ─────────────────────────────────────────────
# C. DUMMY DATA GENERATION
# ─────────────────────────────────────────────
def generate_churn_data(n=1000, seed=42):
    """
    Synthetic telecom customer churn dataset.
    Features mimic real telecom data (Telco churn problem).
    """
    np.random.seed(seed)

    tenure          = np.random.randint(1, 72, n)           # months as customer
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges   = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    num_services    = np.random.randint(1, 8, n)             # internet, phone, TV…
    contract_type   = np.random.choice(["Month-to-month", "One year", "Two year"], n,
                                       p=[0.55, 0.25, 0.20])
    payment_method  = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )
    tech_support    = np.random.choice([0, 1], n)            # 1 = has tech support
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
    num_complaints  = np.random.poisson(1.2, n)              # support tickets

    # Churn probability driven by business logic
    churn_prob = (
        0.40 * (contract_type == "Month-to-month").astype(float)
        + 0.20 * (num_complaints > 2).astype(float)
        + 0.15 * (monthly_charges > 80).astype(float)
        - 0.20 * (tenure > 36).astype(float)
        - 0.10 * tech_support
        + np.random.normal(0, 0.05, n)
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn      = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
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
    return df


# ─────────────────────────────────────────────
# D. MODEL IMPLEMENTATION
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    df["contract_type"]   = le.fit_transform(df["contract_type"])
    df["payment_method"]  = le.fit_transform(df["payment_method"])
    return df


def train_churn_model(df):
    df = preprocess(df)
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=" * 55)
    print("PROJECT 1 — Customer Churn Prediction")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_proba):.4f}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model,  "artifacts/churn_model.pkl")
    joblib.dump(scaler, "artifacts/churn_scaler.pkl")

    return model, scaler, X_test, y_test, y_proba, df.drop("churn", axis=1).columns.tolist()


# ─────────────────────────────────────────────
# E. EXPLAINABILITY
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1]

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(importances)), importances[idx], color="steelblue")
    plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=35, ha="right")
    plt.title("Feature Importance — What drives churn?", fontsize=14)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/churn_feature_importance.png", dpi=120)
    plt.close()
    print("Saved: artifacts/churn_feature_importance.png")


def plot_roc(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc          = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Churn Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/churn_roc_curve.png", dpi=120)
    plt.close()
    print("Saved: artifacts/churn_roc_curve.png")


def plot_confusion(y_test, model, X_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Churn"]); ax.set_yticklabels(["Stay", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("artifacts/churn_confusion_matrix.png", dpi=120)
    plt.close()
    print("Saved: artifacts/churn_confusion_matrix.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_churn_data(1000)
    print(f"Dataset shape: {df.shape}")
    print(df.head(3))

    model, scaler, X_test, y_test, y_proba, features = train_churn_model(df)
    plot_feature_importance(model, features)
    plot_roc(y_test, y_proba)
    plot_confusion(y_test, model, X_test)
    print("\nAll artifacts saved in artifacts/ folder.")
