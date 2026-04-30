"""
PROJECT 2 — Streamlit App: Customer Segmentation Dashboard
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
from segmentation_model import (
    generate_ecommerce_data, train_segmentation_model,
    plot_pca_clusters, plot_rfm_bars, CLUSTER_LABELS, COLORS,
)

@st.cache_resource
def load_model():
    artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    model_path  = os.path.join(artifact_dir, "seg_model.pkl")
    scaler_path = os.path.join(artifact_dir, "seg_scaler.pkl")
    if not os.path.exists(model_path):
        df = generate_ecommerce_data(800)
        train_segmentation_model(df, n_clusters=4)
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_model()

st.set_page_config(page_title="Customer Segmentation", page_icon="🎯", layout="wide")
st.title("🎯 E-Commerce Customer Segmentation")
st.markdown(
    """
**Goal:** Group customers into meaningful segments so marketing teams can run **targeted campaigns**
instead of sending the same message to everyone.
*(RFM = Recency · Frequency · Monetary — the gold standard in retail analytics)*
"""
)

st.sidebar.header("New Customer Profile")
recency   = st.sidebar.slider("Days since last purchase", 1, 365, 30)
frequency = st.sidebar.slider("Number of purchases", 1, 50, 8)
monetary  = st.sidebar.slider("Total spend ($)", 10.0, 3000.0, 400.0)
avg_order = st.sidebar.number_input("Avg order value ($)", 5.0, 1000.0, monetary / frequency)
clv       = st.sidebar.number_input("Customer Lifetime Value ($)", 10.0, 6000.0, monetary * 1.5)
sessions  = st.sidebar.slider("Total website sessions", 1, 300, frequency * 3)
returns   = st.sidebar.slider("Number of returns", 0, 10, 1)

input_vec = np.array([[recency, frequency, monetary, avg_order, clv, sessions, returns]])
scaled    = scaler.transform(input_vec)
cluster   = int(model.predict(scaled)[0])
seg_name  = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")

# ── Segment strategies ─────────────────────────────────────
strategies = {
    "Champions":       ("🏆", "#2ecc71", "Top customers! Reward with loyalty perks & early access."),
    "At Risk":         ("⚠️", "#e74c3c", "Haven't bought recently. Send win-back email with 20% off."),
    "New / Occasional":("🌱", "#3498db", "New users. Nurture with onboarding emails & first-order discount."),
    "Hibernating":     ("😴", "#95a5a6", "Long inactive. Try a big discount or sunset from active list."),
}

emoji, color, strategy = strategies.get(seg_name, ("❓", "#bdc3c7", "Unknown segment."))

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Predicted Segment")
    st.markdown(
        f"""
        <div style='background:{color};padding:20px;border-radius:10px;text-align:center'>
            <h1 style='color:white;margin:0'>{emoji}</h1>
            <h2 style='color:white;margin:4px'>{seg_name}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**Recommended Action:** {strategy}")

with col2:
    st.subheader("Segment Comparison")
    df_full = generate_ecommerce_data(800)
    _, _, df_clustered, X_scaled, _ = train_segmentation_model(df_full, n_clusters=4)
    profile = df_clustered.groupby("cluster")[["recency", "frequency", "monetary"]].mean().round(1)
    profile.index = [CLUSTER_LABELS.get(i, str(i)) for i in profile.index]
    profile.columns = ["Avg Recency (days)", "Avg Frequency", "Avg Monetary ($)"]
    st.dataframe(profile, use_container_width=True)

st.markdown("---")
img_pca = os.path.join(os.path.dirname(__file__), "artifacts", "seg_pca.png")
img_rfm = os.path.join(os.path.dirname(__file__), "artifacts", "seg_rfm_bars.png")

col3, col4 = st.columns(2)
with col3:
    if os.path.exists(img_pca):
        st.image(img_pca, caption="Customer Clusters — PCA View", use_container_width=True)
with col4:
    if os.path.exists(img_rfm):
        st.image(img_rfm, caption="RFM Profiles per Cluster", use_container_width=True)

st.markdown("---")
st.subheader("Full Dataset Sample")
st.dataframe(df_clustered.drop("true_segment", axis=1).head(15), use_container_width=True)

st.caption(
    "**Simple Analogy:** KMeans is like sorting fruit. "
    "You pick 4 bowls and keep moving each fruit to whichever bowl it's closest to. "
    "After a few rounds, similar fruits end up together automatically."
)
