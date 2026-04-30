"""
PROJECT 2: Customer Segmentation for E-Commerce
Category: Unsupervised Learning (Clustering)
Algorithm: KMeans + PCA for visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

# ─────────────────────────────────────────────
# C. DUMMY DATA GENERATION
# ─────────────────────────────────────────────
def generate_ecommerce_data(n=800, seed=42):
    """
    Synthetic e-commerce customer dataset using RFM framework:
    R = Recency   (days since last purchase — lower is better)
    F = Frequency (number of purchases)
    M = Monetary  (total spend in $)
    Plus additional behavioral features.
    """
    np.random.seed(seed)

    # Simulate 4 natural customer segments with distinct profiles
    segments = {
        "Champions":    dict(recency=(1, 15),   frequency=(20, 50), monetary=(800, 3000), n=int(n*0.20)),
        "At Risk":      dict(recency=(60, 180),  frequency=(5, 20),  monetary=(200, 800),  n=int(n*0.25)),
        "New Users":    dict(recency=(1, 30),    frequency=(1, 5),   monetary=(50, 300),   n=int(n*0.30)),
        "Hibernating":  dict(recency=(180, 365), frequency=(1, 5),   monetary=(50, 200),   n=int(n*0.25)),
    }

    rows = []
    for seg_name, props in segments.items():
        cnt = props["n"]
        recency   = np.random.randint(*props["recency"],   cnt)
        frequency = np.random.randint(*props["frequency"], cnt)
        monetary  = np.round(np.random.uniform(*props["monetary"], cnt), 2)
        avg_order = np.round(monetary / frequency + np.random.normal(0, 5, cnt), 2)
        clv       = np.round(monetary * np.random.uniform(1.0, 2.0, cnt), 2)
        sessions  = frequency * np.random.randint(2, 6, cnt)
        returns   = np.random.poisson(0.5, cnt)

        for i in range(cnt):
            rows.append({
                "recency":          recency[i],
                "frequency":        frequency[i],
                "monetary":         monetary[i],
                "avg_order_value":  max(avg_order[i], 5),
                "customer_lifetime_value": clv[i],
                "total_sessions":   sessions[i],
                "num_returns":      returns[i],
                "true_segment":     seg_name,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# D. MODEL IMPLEMENTATION
# ─────────────────────────────────────────────
def find_optimal_k(X_scaled, k_range=range(2, 9)):
    """Elbow method + silhouette score to find best k."""
    inertias    = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    return list(k_range), inertias, silhouettes


def train_segmentation_model(df, n_clusters=4):
    feature_cols = [
        "recency", "frequency", "monetary",
        "avg_order_value", "customer_lifetime_value",
        "total_sessions", "num_returns",
    ]
    X = df[feature_cols].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = model.fit_predict(X_scaled)

    df = df.copy()
    df["cluster"] = labels

    sil = silhouette_score(X_scaled, labels)
    print("=" * 55)
    print("PROJECT 2 — Customer Segmentation (KMeans)")
    print("=" * 55)
    print(f"Optimal clusters : {n_clusters}")
    print(f"Silhouette Score : {sil:.4f}  (closer to 1 = better separation)")
    print("\nCluster Sizes:")
    print(df["cluster"].value_counts().sort_index())
    print("\nCluster RFM Profiles:")
    print(df.groupby("cluster")[["recency", "frequency", "monetary"]].mean().round(1))

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model,  "artifacts/seg_model.pkl")
    joblib.dump(scaler, "artifacts/seg_scaler.pkl")

    return model, scaler, df, X_scaled, feature_cols


# ─────────────────────────────────────────────
# E. EXPLAINABILITY & VISUALIZATION
# ─────────────────────────────────────────────
CLUSTER_LABELS = {
    0: "Champions",
    1: "At Risk",
    2: "New / Occasional",
    3: "Hibernating",
}

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


def plot_elbow(k_range, inertias, silhouettes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(k_range, inertias, "bo-")
    ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method — Choose the 'elbow' point")

    ax2.plot(k_range, silhouettes, "rs-")
    ax2.set_xlabel("Number of Clusters (k)"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score — Higher = Better")

    plt.tight_layout()
    plt.savefig("artifacts/seg_elbow.png", dpi=120)
    plt.close()
    print("Saved: artifacts/seg_elbow.png")


def plot_pca_clusters(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(X2[mask, 0], X2[mask, 1],
                    label=CLUSTER_LABELS.get(c, f"Cluster {c}"),
                    color=COLORS[c % len(COLORS)], alpha=0.7, s=40)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("Customer Segments — PCA View")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/seg_pca.png", dpi=120)
    plt.close()
    print("Saved: artifacts/seg_pca.png")


def plot_rfm_bars(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(
        axes,
        ["recency", "frequency", "monetary"],
        ["Recency (days)", "Frequency (purchases)", "Monetary ($ spent)"],
    ):
        means = df.groupby("cluster")[col].mean()
        ax.bar(
            [CLUSTER_LABELS.get(i, str(i)) for i in means.index],
            means.values,
            color=COLORS[: len(means)],
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    plt.suptitle("Cluster Profiles — RFM Averages", fontsize=13)
    plt.tight_layout()
    plt.savefig("artifacts/seg_rfm_bars.png", dpi=120)
    plt.close()
    print("Saved: artifacts/seg_rfm_bars.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_ecommerce_data(800)
    print(f"Dataset shape: {df.shape}")
    print(df.head(3))

    model, scaler, df_clustered, X_scaled, features = train_segmentation_model(df, n_clusters=4)

    k_range, inertias, silhouettes = find_optimal_k(X_scaled)
    plot_elbow(k_range, inertias, silhouettes)
    plot_pca_clusters(X_scaled, df_clustered["cluster"].values)
    plot_rfm_bars(df_clustered)

    print("\nAll artifacts saved in artifacts/ folder.")
