"""
PROJECT 5 — Streamlit App: Movie Recommendation System
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from recommender_model import (
    generate_recommendation_data, build_user_item_matrix,
    train_svd_model, build_content_matrix,
    get_svd_recommendations, get_content_recommendations,
    get_hybrid_recommendations, plot_latent_space, plot_rating_distribution,
    MOVIE_NAMES,
)

@st.cache_resource
def load_model():
    artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    rec_path     = os.path.join(artifact_dir, "rec_model.pkl")

    if not os.path.exists(rec_path):
        df_movies, df_ratings = generate_recommendation_data(300, 50)
        matrix = build_user_item_matrix(df_ratings, 300, 50)
        svd, user_factors, item_factors, reconstructed = train_svd_model(matrix)
        content_matrix = build_content_matrix(df_movies)
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump({
            "svd": svd, "reconstructed": reconstructed,
            "user_factors": user_factors,
            "content_matrix": content_matrix,
            "df_movies": df_movies, "df_ratings": df_ratings,
        }, rec_path)
        plot_latent_space(user_factors)
        plot_rating_distribution(df_ratings)

    data = joblib.load(rec_path)
    return data

data = load_model()
reconstructed  = data["reconstructed"]
content_matrix = data["content_matrix"]
df_movies      = data["df_movies"]
df_ratings     = data["df_ratings"]
user_factors   = data.get("user_factors", None)

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown(
    """
**Goal:** Recommend the right movies to each user — just like Netflix's recommendation engine.
Uses **Collaborative Filtering** (what similar users like) + **Content-Based** (similar genres/ratings)
in a **Hybrid** system for best results.
"""
)

st.sidebar.header("Your Preferences")
user_id     = st.sidebar.slider("User ID", 0, 299, 42)
seed_movie  = st.sidebar.selectbox("Seed Movie (I liked this)", MOVIE_NAMES[:50])
rec_mode    = st.sidebar.radio("Recommendation Mode", ["Hybrid", "Collaborative Only", "Content Only"])
top_n       = st.sidebar.slider("Number of Recommendations", 3, 10, 5)
alpha       = st.sidebar.slider("CF ↔ Content Balance (α)", 0.0, 1.0, 0.6,
                                 help="α=1.0 → pure CF | α=0.0 → pure content")

# ── Get recommendations ────────────────────────────────────
if rec_mode == "Hybrid":
    recs = get_hybrid_recommendations(
        user_id, seed_movie, reconstructed, df_movies,
        df_ratings, content_matrix, top_n=top_n, alpha=alpha,
    )
elif rec_mode == "Collaborative Only":
    recs = get_svd_recommendations(user_id, reconstructed, df_movies, df_ratings, top_n=top_n)
else:
    recs = get_content_recommendations(seed_movie, df_movies, content_matrix, top_n=top_n)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(f"Top {top_n} Recommendations — Mode: {rec_mode}")
    if recs is not None and len(recs) > 0:
        for _, row in recs.iterrows():
            score_str = f"  ·  Score: {row['hybrid_score']:.3f}" if "hybrid_score" in recs.columns else ""
            stars     = "⭐" * round(row["avg_rating"] / 2)
            st.markdown(
                f"""
                <div style='background:#1e1e2e;padding:12px;border-radius:8px;margin:6px 0'>
                    <b style='color:#f0f0f0;font-size:15px'>🎬 {row['title']}</b>
                    <span style='color:#aaa'> — {row['genre']}</span><br>
                    <span style='color:#f1c40f'>{stars}</span>
                    <span style='color:#aaa'> {row['avg_rating']}/10{score_str}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No recommendations available for this combination.")

    st.markdown("---")
    st.subheader("User's Rating History")
    user_hist = df_ratings[df_ratings["user_id"] == user_id].merge(
        df_movies[["movie_id", "title", "genre"]], on="movie_id"
    )[["title", "genre", "rating"]].sort_values("rating", ascending=False)
    st.dataframe(user_hist.head(10), use_container_width=True)

with col2:
    st.subheader("How Hybrid Scoring Works")
    st.markdown(
        f"""
        | Component | Weight |
        |-----------|--------|
        | Collaborative Filtering (SVD) | **{alpha:.0%}** |
        | Content-Based Similarity | **{1-alpha:.0%}** |

        **CF says:** *"Users like you also enjoyed..."*

        **Content says:** *"Similar genre/rating to what you liked..."*

        **Hybrid combines both** for the most accurate picks.
        """
    )

    # Genre pie chart for user's watched movies
    if len(user_hist) > 0:
        genre_counts = user_hist["genre"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(genre_counts.values, labels=genre_counts.index,
               autopct="%1.0f%%", startangle=90, pctdistance=0.85)
        ax.set_title(f"User {user_id}'s Genre Profile")
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
img_col1, img_col2 = st.columns(2)
with img_col1:
    p = os.path.join(os.path.dirname(__file__), "artifacts", "rec_user_latent.png")
    if os.path.exists(p):
        st.image(p, caption="User Taste Map (SVD Latent Space)", use_container_width=True)

with img_col2:
    p = os.path.join(os.path.dirname(__file__), "artifacts", "rec_rating_dist.png")
    if os.path.exists(p):
        st.image(p, caption="Rating Distribution", use_container_width=True)

st.caption(
    "**Simple Analogy:** SVD is like asking 'What kind of person are you?' — it discovers hidden "
    "taste dimensions. If two users both love Sci-Fi and dislike Horror, SVD groups them together "
    "and recommends based on that shared taste — even without knowing WHY they like the same things."
)
