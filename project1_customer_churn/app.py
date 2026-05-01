"""
PROJECT 1 — Streamlit App: Customer Churn Predictor (Enhanced)
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from churn_model import (
    generate_churn_data, train_churn_model, preprocess,
    plot_churn_drivers, plot_feature_importance, plot_roc_and_pr,
    plot_confusion_annotated, plot_churn_probability_distribution,
    plot_learning_curve, plot_single_tree, ARTIFACT_DIR,
)

# ── Bootstrap ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    mp = os.path.join(ARTIFACT_DIR, "churn_model.pkl")
    sp = os.path.join(ARTIFACT_DIR, "churn_scaler.pkl")
    if not os.path.exists(mp):
        df = generate_churn_data(1000)
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
    return joblib.load(mp), joblib.load(sp)

model, scaler = load_model()

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Prediction System")
st.markdown("""
> **Business Problem:** A telecom company wants to know *which customers will cancel next month*
> so retention teams can act before it's too late.
> — *Losing a customer costs 5× more than retaining one.*
""")

# ── Why Random Forest? box ────────────────────────────────
with st.expander("🧠 Why Random Forest? (Algorithm Explainer)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Random Forest = 200 Decision Trees voting together**

Each tree:
1. Sees a random *subset* of customers (bootstrap sampling)
2. At each split, considers a random *subset* of features
3. Votes: "This customer will Churn" or "Stay"

**Final decision = majority vote of 200 trees**

**Why is diversity good?** One doctor can be wrong.
200 doctors with *different expertise* are much harder to fool.
        """)
    with col_b:
        st.markdown("""
| Algorithm | Why ❌ Not Used |
|-----------|---------------|
| Logistic Regression | Too linear; misses complex interactions |
| Single Decision Tree | Overfits; unstable on small data changes |
| SVM | Slow on large data; black box |
| Neural Network | Overkill for tabular data; needs more data |
| **Random Forest** | ✅ Interpretable + Robust + Fast |

**Key hyperparameters:**
- `n_estimators=200` → 200 trees (more = better, slower)
- `max_depth=8` → prevents overfitting
- `class_weight='balanced'` → handles churn imbalance (20% positive)
        """)

st.markdown("---")

# ── Two columns: input + live prediction ─────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Customer Profile")
    tenure          = st.slider("Tenure (months as customer)", 1, 72, 12,
                                help="Longer = more loyal = less likely to churn")
    monthly_charges = st.slider("Monthly Bill ($)", 20.0, 120.0, 65.0)
    total_charges   = monthly_charges * tenure
    num_services    = st.slider("Bundled Services Count", 1, 7, 3)
    contract_type   = st.selectbox("Contract Type",
                                   ["Month-to-month", "One year", "Two year"],
                                   help="Month-to-month = lowest switching cost = highest churn risk")
    payment_method  = st.selectbox("Payment Method",
                                   ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])
    tech_support    = st.selectbox("Has Tech Support Add-on?", ["No", "Yes"])
    senior_citizen  = st.selectbox("Senior Citizen?", ["No", "Yes"])
    num_complaints  = st.slider("Support Complaints Filed", 0, 10, 1)

# ── Live prediction ───────────────────────────────────────
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map  = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}

input_data   = np.array([[
    tenure, monthly_charges, total_charges, num_services,
    contract_map[contract_type], payment_map[payment_method],
    1 if tech_support == "Yes" else 0,
    1 if senior_citizen == "Yes" else 0,
    num_complaints,
]])
churn_prob = model.predict_proba(scaler.transform(input_data))[0][1]

with col2:
    st.subheader("🔮 Live Prediction")

    # Gauge bar
    fig, ax = plt.subplots(figsize=(8, 1.4))
    cmap = plt.get_cmap("RdYlGn_r")
    for i in range(100):
        ax.barh(0, 0.01, left=i * 0.01, color=cmap(i / 100), height=0.8)
    ax.axvline(churn_prob, color="black", lw=3)
    ax.set_xlim(0, 1); ax.set_yticks([])
    ax.set_xlabel("Churn Probability →", fontsize=11)
    for x, lbl in [(0.15, "Safe"), (0.4, "Watch"), (0.65, "At Risk"), (0.85, "Critical")]:
        ax.text(x, 0.45, lbl, ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_title(f"Churn Probability: {churn_prob:.1%}", fontsize=13)
    st.pyplot(fig)

    # Result card
    if churn_prob >= 0.7:
        st.error(f"🚨 **CRITICAL** — {churn_prob:.1%} churn risk. Assign to retention specialist NOW.")
    elif churn_prob >= 0.5:
        st.warning(f"⚠️ **AT RISK** — {churn_prob:.1%}. Offer upgrade or discount this week.")
    elif churn_prob >= 0.3:
        st.info(f"👀 **WATCH** — {churn_prob:.1%}. Monitor; add to next campaign.")
    else:
        st.success(f"✅ **LOYAL** — {churn_prob:.1%}. No intervention needed.")

    # Per-feature contribution visualization
    st.subheader("🔍 What's Driving This Prediction?")
    feature_names = ["tenure", "monthly_charges", "total_charges", "num_services",
                     "contract_type", "payment_method", "tech_support",
                     "senior_citizen", "num_complaints"]
    importances = model.feature_importances_
    contrib = importances * np.abs(scaler.transform(input_data)[0])
    contrib = contrib / contrib.sum()

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if c > contrib.mean() else "#3498db" for c in contrib]
    ax2.barh(feature_names, contrib, color=colors)
    ax2.set_xlabel("Relative Contribution to This Prediction")
    ax2.set_title("Feature Contribution (this specific customer)")
    red_p  = mpatches.Patch(color="#e74c3c", label="Above average influence")
    blue_p = mpatches.Patch(color="#3498db", label="Below average influence")
    ax2.legend(handles=[red_p, blue_p])
    plt.tight_layout()
    st.pyplot(fig2)

# ── Tabs for deep dives ────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Business EDA", "🌲 Feature Importance",
    "📈 ROC + PR Curves", "🎯 Confusion Matrix",
    "📚 Learning Curve",
])

def show_img(path, caption=""):
    p = os.path.join(ARTIFACT_DIR, path)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.info("Run churn_model.py first to generate this chart.")

with tab1:
    st.markdown("**Understanding the raw data before modelling:**")
    show_img("churn_eda.png", "Churn Drivers — Business EDA")
    st.markdown("""
    - **Contract type** is the #1 predictor: month-to-month customers churn at 3× the rate of 2-year contracts
    - **Tenure** shows long-term customers are sticky — churn drops sharply after 24 months
    - **More complaints = more churn** — a linear signal the model can exploit
    """)

with tab2:
    show_img("churn_feature_importance.png", "What drives churn? (error bars = variance across 200 trees)")
    show_img("churn_single_tree.png", "One example tree — the forest uses 200 of these")
    st.markdown("""
    **Reading feature importance:** Each bar = how much that feature reduces uncertainty across all 200 trees.
    Error bars show variance — a tall error bar means trees disagree about this feature's importance.
    """)

with tab3:
    show_img("churn_roc_pr.png", "ROC + Precision-Recall curves")
    st.markdown("""
    - **ROC AUC = 1.0** is perfect; **0.5** = random guessing
    - **PR Curve** is more informative for imbalanced datasets like churn (only ~25% churn)
    - The gap between PR curve and baseline shows how much better than random the model is
    """)
    show_img("churn_prob_dist.png", "Probability distribution by true class")

with tab4:
    show_img("churn_confusion_matrix.png", "Confusion Matrix with business annotations")
    st.markdown("""
    | Cell | Business Meaning | Cost |
    |------|-----------------|------|
    | True Positive | Caught churner → can retain | Low (intervention cost) |
    | False Positive | Loyal flagged as churner | Medium (wasted discount) |
    | False Negative | Churner missed | **High** (lost revenue) |
    | True Negative | Loyal correctly ignored | None |
    """)

with tab5:
    show_img("churn_learning_curve.png", "Does more data help?")
    st.markdown("""
    - If training score >> validation score → **overfitting** (add regularization / more data)
    - If both scores plateau at low values → **underfitting** (need stronger model or better features)
    - Converging curves = **healthy model**
    """)

st.markdown("---")
st.subheader("📋 Sample Dataset")
st.dataframe(generate_churn_data(1000).head(12), use_container_width=True)

st.caption("""
**Simple Analogy:** Random Forest is like a medical board of 200 doctors.
Each doctor reviews the patient's history from a slightly different angle.
Most doctors saying 'this patient will churn' → the board rules: high churn risk.
The diversity of opinions makes the board far more accurate than any single doctor.
""")
