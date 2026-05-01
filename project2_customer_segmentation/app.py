"""
PROJECT 2 — Streamlit App: Customer Segmentation Dashboard (Enhanced)
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from segmentation_model import (
    generate_ecommerce_data, train_segmentation_model, find_optimal_k,
    plot_elbow_silhouette, plot_pca_clusters, plot_rfm_radar,
    plot_rfm_bars, plot_segment_distribution,
    ARTIFACT_DIR, CLUSTER_LABELS, CLUSTER_COLORS, CLUSTER_STRATEGY, FEATURE_COLS,
)

@st.cache_resource
def load_all():
    mp = os.path.join(ARTIFACT_DIR, "seg_model.pkl")
    sp = os.path.join(ARTIFACT_DIR, "seg_scaler.pkl")
    lp = os.path.join(ARTIFACT_DIR, "seg_label_map.pkl")
    if not os.path.exists(mp):
        df = generate_ecommerce_data(1000)
        model, scaler, df_cl, X_scaled, label_map = train_segmentation_model(df, 4)
        k_range, in_, si_, db_ = find_optimal_k(X_scaled)
        plot_elbow_silhouette(k_range, in_, si_, db_)
        plot_pca_clusters(df_cl, X_scaled)
        plot_rfm_radar(df_cl)
        plot_rfm_bars(df_cl)
        plot_segment_distribution(df_cl)
    return (joblib.load(mp), joblib.load(sp), joblib.load(lp))

model, scaler, label_map = load_all()

st.set_page_config(page_title="Customer Segmentation", page_icon="🎯", layout="wide")
st.title("🎯 E-Commerce Customer Segmentation")
st.markdown("""
> **Business Goal:** Group millions of customers into actionable segments so marketing teams
> run **targeted campaigns** instead of blasting everyone with the same message.
> *(Targeted campaigns achieve 3–5× higher open rates)*
""")

# ── Algorithm explainer ───────────────────────────────────
with st.expander("🧠 Why KMeans? (Algorithm Explainer)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**KMeans Clustering — Step by Step:**

1. Pick K=4 random "centroids" (imaginary average customers)
2. Assign every customer to the nearest centroid
3. Move each centroid to the **mean** of its assigned customers
4. Repeat steps 2-3 until assignments stop changing

**Why K=4?**
We use the **Elbow Method** + **Silhouette Score** to objectively find the best K.

**Why StandardScaler first?**
Without scaling, `monetary` ($1000s) would dominate `num_returns` (0–3).
Scaling puts all features on equal footing.
        """)
    with col_b:
        st.markdown("""
| Algorithm | Why ❌ Not Used |
|-----------|---------------|
| DBSCAN | Hard to set parameters for business use |
| Hierarchical | Too slow for >50K customers |
| Gaussian Mixture | Probabilistic membership confuses marketers |
| **KMeans** | ✅ Fast · Interpretable · CRM-ready |

**What is RFM?**
| Letter | Meaning | Good Value |
|--------|---------|-----------|
| **R**ecency | Days since last purchase | Low (recent) |
| **F**requency | # of purchases | High |
| **M**onetary | Total $ spent | High |
        """)

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 New Customer Profile")
    recency   = st.slider("Days since last purchase (Recency)", 1, 365, 30)
    frequency = st.slider("Number of purchases (Frequency)", 1, 55, 8)
    monetary  = st.slider("Total spend $ (Monetary)", 10.0, 4000.0, 400.0)
    avg_order = st.number_input("Avg order value ($)", 5.0, 1000.0,
                                 float(round(monetary / max(frequency, 1), 2)))
    clv       = st.number_input("Lifetime Value ($)", 10.0, 8000.0, float(round(monetary * 1.4, 2)))
    sessions  = st.slider("Total website sessions", 1, 300, frequency * 3)
    returns   = st.slider("Number of returns", 0, 10, 1)

    input_vec = np.array([[recency, frequency, monetary, avg_order, clv, sessions, returns]])
    cluster   = int(model.predict(scaler.transform(input_vec))[0])
    seg_name  = label_map.get(cluster, f"Cluster {cluster}")
    color     = CLUSTER_COLORS.get(seg_name, "#bdc3c7")
    strategy  = CLUSTER_STRATEGY.get(seg_name, "Review manually")

    st.markdown(
        f"""
        <div style='background:{color};padding:18px;border-radius:10px;text-align:center;margin-top:12px'>
            <h2 style='color:white;margin:0'>Segment: {seg_name}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**💡 Recommended Action:** {strategy}")

with col2:
    st.subheader("📊 Segment Profiles")
    df = generate_ecommerce_data(1000)
    _, _, df_cl, X_scaled, lmap = train_segmentation_model(df, 4)

    profile = df_cl.groupby("segment")[
        ["recency", "frequency", "monetary", "customer_lifetime_value"]
    ].mean().round(1)
    profile.columns = ["Avg Recency (days)", "Avg Frequency", "Avg Monetary ($)", "Avg CLV ($)"]

    # Highlight current customer's segment row
    def highlight_seg(row):
        return ["background-color: #fffacd" if row.name == seg_name else "" for _ in row]

    st.dataframe(profile.style.apply(highlight_seg, axis=1), use_container_width=True)

    # Mini bar chart for this customer vs segment avg
    st.subheader("Your Customer vs Segment Average")
    seg_mask  = df_cl["segment"] == seg_name
    seg_avg_r = df_cl[seg_mask]["recency"].mean()
    seg_avg_f = df_cl[seg_mask]["frequency"].mean()
    seg_avg_m = df_cl[seg_mask]["monetary"].mean()

    comp_data = pd.DataFrame({
        "Metric": ["Recency", "Frequency", "Monetary"],
        "Customer":       [recency, frequency, monetary],
        "Segment Average": [seg_avg_r, seg_avg_f, seg_avg_m],
    }).set_index("Metric")

    # Normalize for fair visual comparison
    norm = comp_data / comp_data.max()
    fig, ax = plt.subplots(figsize=(7, 3))
    x  = np.arange(3)
    ax.bar(x - 0.18, norm.iloc[:, 0], 0.35, label="This Customer", color=color, alpha=0.9)
    ax.bar(x + 0.18, norm.iloc[:, 1], 0.35, label="Segment Avg",   color="gray", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["Recency", "Frequency", "Monetary"])
    ax.set_ylabel("Normalized Score"); ax.set_title("Customer vs Segment Average (normalized)")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📐 Choosing K", "🗺 PCA Cluster Map",
    "🕸 Radar Chart", "📊 RFM Bars", "🍩 Segment Distribution",
])

def show_img(name, caption=""):
    p = os.path.join(ARTIFACT_DIR, name)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.info("Run segmentation_model.py first.")

with tab1:
    show_img("seg_elbow.png", "Elbow + Silhouette + Davies-Bouldin to pick K=4")
    st.markdown("""
    - **Elbow:** Pick K where the curve bends — adding more clusters gives diminishing returns
    - **Silhouette > 0.5:** Clusters are well-separated (tight within, far apart from each other)
    - **Davies-Bouldin lower:** More compact and separated clusters
    - **All three methods point to K=4** → our choice is statistically justified
    """)

with tab2:
    show_img("seg_pca.png", "2D projection of 7 RFM features via PCA")
    st.markdown("""
    **PCA (Principal Component Analysis)** compresses our 7 features into 2 dimensions for visualization.
    - We keep the **dimensions with most variance** (most information)
    - Stars (★) mark cluster centroids — the 'average customer' of each segment
    - Clusters should appear as **distinct blobs** — good separation = good model
    """)

with tab3:
    show_img("seg_radar.png", "Radar chart comparing all 4 segments")
    st.markdown("""
    Each axis = one RFM metric (normalized 0–1). Larger polygon = stronger on that dimension.
    - **Champions** cover most area — high frequency, high monetary, high CLV
    - **Hibernating** are small and flat — low engagement across all metrics
    - **At Risk** have moderate frequency but declining (recency was cropped as it's inverse)
    """)

with tab4:
    show_img("seg_rfm_bars.png", "Average RFM metrics per segment")
    st.markdown("""
    Plain bar chart shows raw averages — easy to read in a stakeholder meeting:
    - *'Champions spend $2,000 on average vs Hibernating at $80'*
    - *'At Risk bought 14 times but haven't been back in 2+ months'*
    """)

with tab5:
    show_img("seg_distribution.png", "How customers split across segments")

st.caption("""
**Simple Analogy:** KMeans is like a librarian sorting books into 4 piles.
They glance at each book (customer) and put it with similar ones.
After a few rounds of adjusting — similar books end up together automatically.
The 'piles' are your customer segments.
""")
