"""
PROJECT 5: Movie Recommendation System
Category  : Recommendation System — Hybrid (CF + Content-Based)
Algorithm : SVD Matrix Factorization + Cosine Similarity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY SVD + COSINE SIMILARITY (HYBRID)?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alternatives considered:
  • Pure Collaborative Filtering (CF)     → fails for new users/movies (cold start)
  • Pure Content-Based                    → misses 'wisdom of crowds' signal
  • ALS (Alternating Least Squares)       → better for implicit feedback (clicks)
  • Neural CF (NCF / Two-Tower)           → state-of-art but needs GPU + large data
  • Association Rules (Apriori)           → finds co-occurrence but no personalization

HYBRID wins because:
  1. CF (SVD) captures "users like you also liked..." — social signal
  2. Content-Based handles cold start — new movie can still get recommended
  3. Weighted blend (α) is tunable per use case (streaming vs e-commerce)
  4. SVD is fast and scalable — Netflix ran matrix factorization at scale

HOW SVD WORKS:
  User-Item matrix R (300×50) → decompose into U × Σ × V^T
  U  = user latent factors  (what "type" of viewer each user is)
  Σ  = importance of each latent dimension
  V  = item latent factors  (what "genre DNA" each movie has)
  Predicted rating = row of U · row of V = dot product of taste vectors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance",
          "Thriller", "Horror", "Animation"]

MOVIE_NAMES = [
    "Galactic Storm", "The Laughing Owl", "Tears in August", "Neon Odyssey",
    "Love in Paris", "Shadow Protocol", "Dark Whispers", "Tiny Heroes",
    "Iron Fist Rising", "Stand-Up Universe", "Broken Wings", "Quantum Drift",
    "Forever Yours", "Silent Code", "Night Screams", "Pixel Dreams",
    "Battle Horizon", "Comedy Carnival", "The Last Tear", "Deep Space 9",
    "Roses Never Die", "The Mole", "Haunted Manor", "Cartoon Kingdom",
    "Blaze Runner", "Joke Factory", "Echoes of War", "Stellar Minds",
    "First Kiss", "The Whistler", "Cry of the Banshee", "Robot Adventures",
    "Fury Road 2050", "Ha Ha Land", "Silent Tears", "Nebula Rising",
    "Sweet Surrender", "False Identity", "Midnight Terror", "Sunny Days",
    "Strike Force", "The Comedian", "A River Between", "Time Warp",
    "Garden of Hearts", "Undercover Boss", "Scream Street", "Mini Adventures",
    "Warrior Code", "Laugh Out Loud",
]


# ─────────────────────────────────────────────
# C. DUMMY DATA GENERATION
# ─────────────────────────────────────────────
def generate_recommendation_data(n_users=300, n_movies=50, seed=42):
    rng = np.random.default_rng(seed)

    movies = []
    for i, name in enumerate(MOVIE_NAMES[:n_movies]):
        genre  = GENRES[i % len(GENRES)]
        year   = int(rng.integers(2000, 2025))
        rating = round(float(rng.uniform(5.5, 9.5)), 1)
        movies.append({
            "movie_id":   i, "title": name, "genre": genre,
            "year": year, "avg_rating": rating,
            "popularity": int(rng.integers(1000, 500000)),
        })
    df_movies = pd.DataFrame(movies)

    # User taste vectors (Dirichlet gives realistic genre preferences)
    user_genre_pref = rng.dirichlet(np.ones(len(GENRES)), size=n_users)

    ratings = []
    for uid in range(n_users):
        n_rated   = int(rng.integers(10, 30))
        movie_ids = rng.choice(n_movies, size=n_rated, replace=False)
        for mid in movie_ids:
            g_idx  = GENRES.index(df_movies.loc[mid, "genre"])
            pref   = user_genre_pref[uid, g_idx]
            base_r = df_movies.loc[mid, "avg_rating"]
            r      = float(np.clip(base_r * pref * 1.4 + rng.normal(0, 0.6), 1, 5))
            ratings.append({"user_id": uid, "movie_id": int(mid), "rating": round(r, 1)})

    return df_movies, pd.DataFrame(ratings)


# ─────────────────────────────────────────────
# D. MODEL IMPLEMENTATION
# ─────────────────────────────────────────────
def build_user_item_matrix(df_ratings, n_users, n_movies):
    matrix = np.zeros((n_users, n_movies))
    for _, row in df_ratings.iterrows():
        matrix[int(row["user_id"]), int(row["movie_id"])] = row["rating"]
    return matrix


def train_svd_model(matrix, n_components=20):
    svd          = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)
    reconstructed = user_factors @ svd.components_

    known_mask = matrix > 0
    actual     = matrix[known_mask]
    predicted  = reconstructed[known_mask]
    rmse       = np.sqrt(mean_squared_error(actual, predicted))

    print("=" * 60)
    print("PROJECT 5 — Movie Recommender (SVD Hybrid)")
    print("=" * 60)
    print(f"SVD components      : {n_components}")
    print(f"Explained variance  : {svd.explained_variance_ratio_.sum():.2%}")
    print(f"Reconstruction RMSE : {rmse:.4f}  (on known ratings 1–5 scale)")

    return svd, user_factors, svd.components_.T, reconstructed


def build_content_matrix(df_movies):
    genre_dummies = pd.get_dummies(df_movies["genre"])
    scaler = StandardScaler()
    extra  = scaler.fit_transform(df_movies[["year", "avg_rating", "popularity"]])
    return np.hstack([genre_dummies.values.astype(float), extra])


def get_svd_recommendations(user_id, reconstructed, df_movies, df_ratings, top_n=5):
    rated = set(df_ratings[df_ratings["user_id"] == user_id]["movie_id"])
    scores = [(i, reconstructed[user_id, i]) for i in range(len(df_movies)) if i not in rated]
    scores.sort(key=lambda x: x[1], reverse=True)
    ids = [m[0] for m in scores[:top_n]]
    return df_movies[df_movies["movie_id"].isin(ids)][["title", "genre", "avg_rating"]]


def get_content_recommendations(movie_title, df_movies, content_matrix, top_n=5):
    idx = df_movies[df_movies["title"] == movie_title].index
    if len(idx) == 0:
        return pd.DataFrame()
    sim   = cosine_similarity(content_matrix[idx[0]].reshape(1, -1), content_matrix)[0]
    order = [i for i in np.argsort(sim)[::-1] if i != idx[0]][:top_n]
    return df_movies.iloc[order][["title", "genre", "avg_rating"]]


def get_hybrid_recommendations(user_id, movie_title, reconstructed, df_movies,
                               df_ratings, content_matrix, top_n=5, alpha=0.6):
    rated      = set(df_ratings[df_ratings["user_id"] == user_id]["movie_id"])
    svd_scores = reconstructed[user_id]
    seed_idx   = df_movies[df_movies["title"] == movie_title].index
    sim = (cosine_similarity(content_matrix[seed_idx[0]].reshape(1, -1), content_matrix)[0]
           if len(seed_idx) > 0 else np.ones(len(df_movies)) / len(df_movies))

    hybrid = {i: alpha * svd_scores[i] + (1 - alpha) * sim[i] * 5.0
              for i in range(len(df_movies)) if i not in rated}
    top_ids = sorted(hybrid, key=hybrid.get, reverse=True)[:top_n]
    result  = df_movies[df_movies["movie_id"].isin(top_ids)][
        ["movie_id", "title", "genre", "avg_rating"]].copy()
    result["hybrid_score"] = result["movie_id"].map(lambda x: round(hybrid.get(x, 0), 3))
    return result.sort_values("hybrid_score", ascending=False)


# ─────────────────────────────────────────────
# E. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_matrix_sparsity(matrix):
    """
    Sparsity map: white = rated, black = unrated.
    This is the core challenge — recommending from mostly empty data.
    """
    sample_m = matrix[:60, :]
    fig, ax  = plt.subplots(figsize=(12, 5))
    ax.imshow(sample_m == 0, cmap="gray_r", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Movie ID (0–49)", fontsize=11)
    ax.set_ylabel("User ID (first 60 users)", fontsize=11)
    ax.set_title(f"Rating Matrix Sparsity — Black = Rated, White = Unknown\n"
                 f"Sparsity: {(matrix==0).mean():.1%}  "
                 f"(this is what the model fills in!)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rec_sparsity.png"), dpi=130)
    plt.close()
    print("Saved: rec_sparsity.png")


def plot_svd_explained_variance(svd):
    """
    Each component captures a different 'taste dimension'.
    Cumulative line shows how many components you need to capture 80% of variance.
    """
    ev  = svd.explained_variance_ratio_
    cum = np.cumsum(ev)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(1, len(ev)+1), ev * 100, color="steelblue")
    ax1.set_xlabel("SVD Component (latent taste dimension)")
    ax1.set_ylabel("Variance Explained (%)");
    ax1.set_title("Per-Component Explained Variance\n"
                  "Each component = one 'taste dimension' (e.g., 'loves Sci-Fi')")
    ax1.grid(alpha=0.3)

    ax2.plot(range(1, len(cum)+1), cum * 100, "o-", color="darkorange", lw=2)
    ax2.axhline(80, color="red", lw=1.5, linestyle="--", label="80% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance — How many components do we need?")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("SVD Decomposition — Discovering Hidden Taste Dimensions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rec_svd_variance.png"), dpi=130)
    plt.close()
    print("Saved: rec_svd_variance.png")


def plot_user_latent_space(user_factors, n_show=100):
    """
    PCA on user latent factors → 2D map.
    Users clustered together have similar tastes.
    """
    from sklearn.decomposition import PCA
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(user_factors[:n_show])
    ev     = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=range(n_show),
                    cmap="tab20", alpha=0.75, s=55, edgecolors="white", lw=0.3)
    plt.colorbar(sc, ax=ax, label="User ID")
    ax.set_xlabel(f"Taste Dimension 1  ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"Taste Dimension 2  ({ev[1]*100:.1f}% var)")
    ax.set_title(f"User Taste Map — SVD Latent Space\n"
                 f"Nearby users = similar movie preferences\n"
                 f"(First {n_show} users shown)", fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rec_user_latent.png"), dpi=130)
    plt.close()
    print("Saved: rec_user_latent.png")


def plot_rating_distribution(df_ratings, df_movies):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Rating histogram
    axes[0].hist(df_ratings["rating"], bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Rating (1–5)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Overall Rating Distribution")

    # 2. Ratings per user
    rpu = df_ratings.groupby("user_id").size()
    axes[1].hist(rpu, bins=25, color="darkorchid", edgecolor="white")
    axes[1].set_xlabel("# Movies Rated per User"); axes[1].set_ylabel("# Users")
    axes[1].set_title(f"Ratings per User\nAvg = {rpu.mean():.1f}")

    # 3. Avg rating per genre
    merged = df_ratings.merge(df_movies[["movie_id", "genre"]], on="movie_id")
    genre_avg = merged.groupby("genre")["rating"].mean().sort_values(ascending=False)
    axes[2].barh(genre_avg.index, genre_avg.values, color="#f39c12")
    axes[2].set_xlabel("Average Rating"); axes[2].set_title("Average Rating by Genre")
    for i, v in enumerate(genre_avg.values):
        axes[2].text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)

    plt.suptitle("Rating Data Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rec_rating_dist.png"), dpi=130)
    plt.close()
    print("Saved: rec_rating_dist.png")


def plot_genre_dist(df_movies):
    genre_counts = df_movies["genre"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(genre_counts.index, genre_counts.values, color=plt.cm.Set2(np.linspace(0, 1, len(genre_counts))))
    ax.set_xlabel("Genre"); ax.set_ylabel("# Movies"); ax.set_title("Movie Genre Distribution")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rec_genre_dist.png"), dpi=130)
    plt.close()
    print("Saved: rec_genre_dist.png")


def plot_hybrid_score_breakdown(user_id, movie_title, reconstructed, df_movies,
                                df_ratings, content_matrix):
    """
    Visual breakdown: how CF score and Content score contribute to final hybrid score.
    """
    rated      = set(df_ratings[df_ratings["user_id"] == user_id]["movie_id"])
    svd_scores = reconstructed[user_id]
    seed_idx   = df_movies[df_movies["title"] == movie_title].index
    sim = (cosine_similarity(content_matrix[seed_idx[0]].reshape(1, -1), content_matrix)[0]
           if len(seed_idx) > 0 else np.ones(len(df_movies)) / len(df_movies))

    candidate_ids = [i for i in range(len(df_movies)) if i not in rated][:15]
    titles   = [df_movies.loc[df_movies["movie_id"] == i, "title"].values[0] for i in candidate_ids]
    cf_sc    = [svd_scores[i] for i in candidate_ids]
    cont_sc  = [sim[i] * 5.0 for i in candidate_ids]
    hybrid   = [0.6 * c + 0.4 * s for c, s in zip(cf_sc, cont_sc)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x       = np.arange(len(titles))
    w       = 0.28
    ax.bar(x - w, cf_sc,   w, label="CF Score (SVD)",     color="steelblue", alpha=0.85)
    ax.bar(x,     cont_sc, w, label="Content Score",       color="darkorchid", alpha=0.85)
    ax.bar(x + w, hybrid,  w, label="Hybrid (α=0.6 CF)",  color="#2ecc71", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(titles, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Score"); ax.set_title(
        f"Hybrid Score Breakdown — User {user_id} | Seed: {movie_title}\n"
        "Green bar = final recommendation score (higher = better)", fontsize=11)
    ax.legend(); ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rec_hybrid_breakdown.png"), dpi=130)
    plt.close()
    print("Saved: rec_hybrid_breakdown.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_movies, df_ratings = generate_recommendation_data(300, 50)
    print(f"Movies: {len(df_movies)}  |  Ratings: {len(df_ratings)}")

    matrix  = build_user_item_matrix(df_ratings, 300, 50)
    svd, user_factors, item_factors, reconstructed = train_svd_model(matrix)
    content_matrix = build_content_matrix(df_movies)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump({
        "svd": svd, "reconstructed": reconstructed,
        "user_factors": user_factors,
        "content_matrix": content_matrix,
        "df_movies": df_movies, "df_ratings": df_ratings,
    }, os.path.join(ARTIFACT_DIR, "rec_model.pkl"))

    print("\n=== Recommendations for User 42 ===")
    print(get_hybrid_recommendations(42, "Galactic Storm", reconstructed,
                                     df_movies, df_ratings, content_matrix))

    plot_matrix_sparsity(matrix)
    plot_svd_explained_variance(svd)
    plot_user_latent_space(user_factors)
    plot_rating_distribution(df_ratings, df_movies)
    plot_genre_dist(df_movies)
    plot_hybrid_score_breakdown(42, "Galactic Storm", reconstructed,
                                 df_movies, df_ratings, content_matrix)

    print("\n✅ All Project 5 artifacts saved.")
