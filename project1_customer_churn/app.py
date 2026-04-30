"""
PROJECT 1 — Streamlit App: Customer Churn Predictor
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from churn_model import generate_churn_data, train_churn_model, preprocess

# ── bootstrap artifacts if missing ──────────────────────────
@st.cache_resource
def load_model():
    artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    model_path  = os.path.join(artifact_dir, "churn_model.pkl")
    scaler_path = os.path.join(artifact_dir, "churn_scaler.pkl")

    if not os.path.exists(model_path):
        df = generate_churn_data(1000)
        train_churn_model(df)

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

# ── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Predictor")
st.markdown(
    """
**What is this?** — A telecom company wants to know **which customers are likely to cancel** their
subscription so they can offer discounts or better service *before* the customer leaves.
"""
)

st.sidebar.header("Customer Profile")

tenure          = st.sidebar.slider("Tenure (months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
total_charges   = monthly_charges * tenure
num_services    = st.sidebar.slider("Number of Services", 1, 7, 3)
contract_type   = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method  = st.sidebar.selectbox(
    "Payment Method",
    ["Bank transfer", "Credit card", "Electronic check", "Mailed check"],
)
tech_support    = st.sidebar.selectbox("Has Tech Support?", ["No", "Yes"])
senior_citizen  = st.sidebar.selectbox("Senior Citizen?", ["No", "Yes"])
num_complaints  = st.sidebar.slider("Number of Complaints", 0, 10, 1)

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map  = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}

input_data = np.array([[
    tenure,
    monthly_charges,
    total_charges,
    num_services,
    contract_map[contract_type],
    payment_map[payment_method],
    1 if tech_support == "Yes" else 0,
    1 if senior_citizen == "Yes" else 0,
    num_complaints,
]])

input_scaled = scaler.transform(input_data)
churn_prob   = model.predict_proba(input_scaled)[0][1]
prediction   = "CHURN RISK" if churn_prob >= 0.5 else "LIKELY TO STAY"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    color = "#e74c3c" if churn_prob >= 0.5 else "#2ecc71"
    st.markdown(
        f"""
        <div style='background:{color};padding:20px;border-radius:10px;text-align:center'>
            <h2 style='color:white;margin:0'>{prediction}</h2>
            <p style='color:white;font-size:18px'>Churn Probability: {churn_prob:.1%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("What this means")
    if churn_prob >= 0.7:
        st.error("🚨 Very high churn risk. Offer retention package immediately.")
    elif churn_prob >= 0.5:
        st.warning("⚠️ Moderate churn risk. Proactive outreach recommended.")
    elif churn_prob >= 0.3:
        st.info("ℹ️ Low risk. Monitor this customer next quarter.")
    else:
        st.success("✅ Very loyal customer. Low priority for intervention.")

with col2:
    st.subheader("Churn Probability Gauge")
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["#2ecc71" if i / 10 < churn_prob else "#ecf0f1" for i in range(10)]
    ax.barh([0] * 10, [0.1] * 10, left=[i * 0.1 for i in range(10)], color=colors, height=0.5)
    ax.axvline(churn_prob, color="red", lw=2, linestyle="--", label=f"{churn_prob:.0%}")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Churn Probability")
    ax.set_title("Risk Gauge")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Feature Importance")
    img_path = os.path.join(os.path.dirname(__file__), "artifacts", "churn_feature_importance.png")
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)

st.markdown("---")
st.subheader("Sample Dataset (first 10 rows)")
df = generate_churn_data(1000)
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")
st.caption(
    "**Simple Analogy:** Random Forest works like a jury of 200 experts. "
    "Each expert looks at a customer slightly differently and votes 'Churn' or 'Stay'. "
    "The majority vote wins."
)
