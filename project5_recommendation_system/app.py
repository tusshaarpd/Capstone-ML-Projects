"""
PROJECT 5 — Streamlit App: Movie Recommender (Enhanced)
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
    get_hybrid_recommendations, plot_matrix_sparsity,
    plot_svd_explained_variance, plot_user_latent_space,
    plot_rating_distribution, plot_genre_dist,
    plot_hybrid_score_breakdown, MOVIE_NAMES, ARTIFACT_DIR,
)

@st.cache_resource
def load_model():
    rp = os.path.join(ARTIFACT_DIR, "rec_model.pkl")
    if not os.path.exists(rp):
        df_movies, df_ratings = generate_recommendation_data(300, 50)
        matrix = build_user_item_matrix(df_ratings, 300, 50)
        svd, user_factors, item_factors, reconstructed = train_svd_model(matrix)
        content_matrix = build_content_matrix(df_movies)
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        joblib.dump({
            "svd": svd, "reconstructed": reconstructed, "user_factors": user_factors,
            "content_matrix": content_matrix, "df_movies": df_movies, "df_ratings": df_ratings,
        }, rp)
        plot_matrix_sparsity(matrix)
        plot_svd_explained_variance(svd)
        plot_user_latent_space(user_factors)
        plot_rating_distribution(df_ratings, df_movies)
        plot_genre_dist(df_movies)

    data = joblib.load(rp)
    return data

data           = load_model()
reconstructed  = data["reconstructed"]
content_matrix = data["content_matrix"]
df_movies      = data["df_movies"]
df_ratings     = data["df_ratings"]
user_factors   = data.get("user_factors")

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System — Hybrid CF + Content")
st.markdown("""
> **Business Goal:** Recommend the right movie to the right person at the right time.
> 80% of Netflix viewing comes from recommendations — this is the most high-value ML problem in consumer tech.
""")

with st.expander("🧠 How the Hybrid System Works (Algorithm Explainer)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Step 1 — SVD (Collaborative Filtering):**

The user-item rating matrix (300 users × 50 movies) is factorized:
```
R ≈ U × Σ × V^T
```
- **U** = User matrix: each user gets a 20-dim "taste vector"
- **V** = Item matrix: each movie gets a 20-dim "DNA vector"
- Predicted rating = dot product of user taste × movie DNA

**Why dot product?**
If you love action movies (high "action dimension" in your vector)
and a movie has a high action dimension → high predicted rating ✅

**Step 2 — Content Similarity:**
Each movie gets a feature vector: [genre one-hot, year, rating, popularity]
We use Cosine Similarity to find movies with similar feature vectors.
        """)
    with col_b:
        st.markdown("""
**Step 3 — Hybrid Score:**

```
Hybrid = α × CF_score + (1-α) × Content_score × 5.0
```

| α value | Behavior |
|---------|----------|
| α = 1.0 | Pure collaborative filtering (no content) |
| α = 0.0 | Pure content-based (ignore other users) |
| α = 0.6 | Balanced (default — good for active users) |

**Cold Start Problem:**
- New user → no ratings → CF fails → use Content-Based (α close to 0)
- Active user → many ratings → CF is reliable → use CF more (α close to 1)

**Why not just BERT/Deep Learning?**
SVD achieves 90%+ of deep model quality with 1% compute.
Netflix, Spotify originally used matrix factorization at scale.
        """)

st.markdown("---")
st.sidebar.header("🎬 Your Preferences")
user_id    = st.sidebar.slider("User ID", 0, 299, 42)
seed_movie = st.sidebar.selectbox("Seed Movie (I liked this)", MOVIE_NAMES[:50])
rec_mode   = st.sidebar.radio("Recommendation Mode",
                               ["Hybrid (Best)", "Collaborative Only (SVD)", "Content Only"])
top_n      = st.sidebar.slider("Number of Recommendations", 3, 10, 5)
alpha      = st.sidebar.slider("CF ↔ Content Balance (α)",
                                0.0, 1.0, 0.6, 0.05,
                                help="α=1.0 = pure collaborative · α=0.0 = pure content")

# ── Recommendations ────────────────────────────────────────
if "Hybrid" in rec_mode:
    recs = get_hybrid_recommendations(
        user_id, seed_movie, reconstructed, df_movies,
        df_ratings, content_matrix, top_n=top_n, alpha=alpha,
    )
elif "Collaborative" in rec_mode:
    recs = get_svd_recommendations(user_id, reconstructed, df_movies, df_ratings, top_n=top_n)
else:
    recs = get_content_recommendations(seed_movie, df_movies, content_matrix, top_n=top_n)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(f"🍿 Top {top_n} Recommendations")
    if recs is not None and len(recs) > 0:
        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            score_str = f" · Score: **{row['hybrid_score']:.3f}**" if "hybrid_score" in recs.columns else ""
            stars     = "⭐" * round(row["avg_rating"] / 2)
            genre_color = {"Action":"#e74c3c","Comedy":"#f39c12","Drama":"#3498db",
                           "Sci-Fi":"#9b59b6","Romance":"#e91e8c","Thriller":"#1abc9c",
                           "Horror":"#2c3e50","Animation":"#27ae60"}.get(row["genre"],"#95a5a6")
            st.markdown(
                f"""
                <div style='background:#1e1e2e;padding:12px 16px;border-radius:8px;
                            margin:5px 0;border-left:4px solid {genre_color}'>
                    <span style='color:#f0f0f0;font-size:15px;font-weight:bold'>
                        #{rank} {row['title']}</span>
                    <span style='color:{genre_color};margin-left:8px'>[{row['genre']}]</span>
                    <br><span style='color:#f1c40f'>{stars}</span>
                    <span style='color:#aaa'> {row['avg_rating']}/10{score_str}</span>
                </div>
                """, unsafe_allow_html=True,
            )
    else:
        st.info("No recommendations for this combination.")

    st.subheader("📚 This User's Watch History")
    hist = df_ratings[df_ratings["user_id"] == user_id].merge(
        df_movies[["movie_id","title","genre"]], on="movie_id"
    )[["title","genre","rating"]].sort_values("rating", ascending=False)
    st.dataframe(hist.head(10), use_container_width=True)

with col2:
    st.subheader("⚖️ Score Breakdown")
    st.markdown(
        f"""
        | Component | Weight |
        |-----------|--------|
        | 🤝 Collaborative (SVD) | **{alpha:.0%}** |
        | 🏷 Content-Based | **{1-alpha:.0%}** |
        """
    )

    # Genre pie for user taste
    if len(hist) > 0:
        gc = hist["genre"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = plt.cm.Set3(np.linspace(0, 1, len(gc)))
        ax.pie(gc.values, labels=gc.index, autopct="%1.0f%%",
               startangle=90, colors=colors, pctdistance=0.8)
        ax.set_title(f"User {user_id}'s Genre Taste Profile")
        plt.tight_layout(); st.pyplot(fig)

    # Rating summary for this user
    if len(hist) > 0:
        avg_r = hist["rating"].mean()
        st.metric("Avg Rating Given", f"{avg_r:.2f}/5.0")
        st.metric("Movies Watched", len(hist))

# ── Visualization tabs ────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🕳 Matrix Sparsity", "📉 SVD Variance",
    "🗺 User Taste Map", "📊 Rating Analysis",
    "📐 Hybrid Breakdown"
])

def show_img(name, caption=""):
    p = os.path.join(ARTIFACT_DIR, name)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)

with tab1:
    show_img("rec_sparsity.png")
    st.markdown("""
    **This is the core ML challenge.** Most cells are empty — we don't know what rating a user would give.
    SVD *fills in* these blanks by learning latent patterns.
    A model that can accurately reconstruct the known ratings will also predict the unknown ones well.
    """)

with tab2:
    show_img("rec_svd_variance.png")
    st.markdown("""
    - Each component = one "taste dimension" (e.g., 'loves action', 'avoids horror')
    - 20 components capture ~62% of all variance in ratings
    - Beyond ~15 components, adding more gives diminishing returns
    - This is why SVD with 20 components is a good production choice
    """)

with tab3:
    show_img("rec_user_latent.png")
    st.markdown("""
    **User Taste Map:** After SVD, each user is a point in 20D space.
    PCA projects it to 2D for visualization.
    Users who are close together have **similar taste profiles** — if you liked X, your neighbor likely liked X too.
    That's the insight that drives collaborative filtering.
    """)

with tab4:
    show_img("rec_rating_dist.png")
    st.markdown("Rating distribution, ratings per user, and average rating by genre.")

with tab5:
    st.markdown("**Hybrid Score Breakdown for current user + seed movie:**")
    bp = os.path.join(ARTIFACT_DIR, "rec_hybrid_breakdown.png")
    if os.path.exists(bp):
        st.image(bp, use_container_width=True)
    elif st.button("Generate Breakdown Chart"):
        plot_hybrid_score_breakdown(user_id, seed_movie, reconstructed,
                                     df_movies, df_ratings, content_matrix)
        st.image(bp, use_container_width=True)
    st.markdown("""
    - **Blue bars** = Collaborative Filtering score (what SVD predicts you'd rate this movie)
    - **Purple bars** = Content similarity score (how similar to your seed movie)
    - **Green bars** = Final hybrid score (weighted blend) — we recommend the highest green bars
    """)

st.caption("""
**Simple Analogy:** SVD is like a music DNA test (like Spotify's "taste profile").
It discovers you love 'high-energy guitar-driven tracks' without you ever saying so.
Then it finds other songs with the same DNA — even ones no one told you about.
That's matrix factorization: discovering hidden taste dimensions from listening history.
""")
