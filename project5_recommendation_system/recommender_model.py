"""
PROJECT 5: Movie Recommendation System
Category: Recommendation System (Collaborative Filtering + Content-Based Hybrid)
Algorithm: SVD (Matrix Factorization) + Cosine Similarity (Content-Based)

Streaming platforms (Netflix, Prime Video, Spotify) live and die by recommendations.
A 1% improvement in click-through rate on recommendations = millions in revenue.
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

# ─────────────────────────────────────────────
# C. DUMMY DATA GENERATION
# ─────────────────────────────────────────────
GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Horror", "Animation"]

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


def generate_recommendation_data(n_users=300, n_movies=50, seed=42):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ── Movies catalog ───────────────────────
    movies = []
    for i, name in enumerate(MOVIE_NAMES[:n_movies]):
        genre  = GENRES[i % len(GENRES)]
        year   = rng.integers(2000, 2025)
        rating = round(float(rng.uniform(5.5, 9.5)), 1)
        movies.append({
            "movie_id":    i,
            "title":       name,
            "genre":       genre,
            "year":        year,
            "avg_rating":  rating,
            "popularity":  rng.integers(1000, 500000),
        })
    df_movies = pd.DataFrame(movies)

    # ── Genre preference per user (latent factor) ─────
    user_genre_pref = rng.dirichlet(np.ones(len(GENRES)), size=n_users)  # (n_users, n_genres)

    # ── Ratings matrix (sparse) ──────────────
    ratings = []
    for user_id in range(n_users):
        # Each user rates 8-25 movies
        n_rated  = rng.integers(8, 26)
        movie_ids = rng.choice(n_movies, size=n_rated, replace=False)
        for mid in movie_ids:
            genre_idx = GENRES.index(df_movies.loc[mid, "genre"])
            pref      = user_genre_pref[user_id, genre_idx]
            base_r    = df_movies.loc[mid, "avg_rating"]
            # User rating influenced by genre preference + noise
            rating    = np.clip(base_r * pref * 1.5 + rng.normal(0, 0.8), 1, 5)
            ratings.append({
                "user_id":  user_id,
                "movie_id": int(mid),
                "rating":   round(float(rating), 1),
            })

    df_ratings = pd.DataFrame(ratings)
    return df_movies, df_ratings


# ─────────────────────────────────────────────
# D. MODEL IMPLEMENTATION
# ─────────────────────────────────────────────
def build_user_item_matrix(df_ratings, n_users, n_movies):
    matrix = np.zeros((n_users, n_movies))
    for _, row in df_ratings.iterrows():
        matrix[int(row["user_id"]), int(row["movie_id"])] = row["rating"]
    return matrix


def train_svd_model(matrix, n_components=20):
    """
    SVD = Singular Value Decomposition
    Decomposes the user-item matrix into latent factors (taste dimensions).
    Think of it as: each user and movie gets a vector of hidden preferences.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors  = svd.fit_transform(matrix)
    item_factors  = svd.components_.T   # shape: (n_movies, n_components)
    reconstructed = user_factors @ svd.components_

    # Evaluate on known ratings (not a true train/test split — for illustration)
    known_mask    = matrix > 0
    actual        = matrix[known_mask]
    predicted     = reconstructed[known_mask]
    rmse          = np.sqrt(mean_squared_error(actual, predicted))

    print("=" * 55)
    print("PROJECT 5 — Movie Recommendation System (SVD Hybrid)")
    print("=" * 55)
    print(f"SVD components    : {n_components}")
    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    print(f"Reconstruction RMSE (known ratings): {rmse:.4f}")

    return svd, user_factors, item_factors, reconstructed


def build_content_matrix(df_movies):
    """One-hot encode genre + normalize year and popularity."""
    genre_dummies = pd.get_dummies(df_movies["genre"])
    scaler = StandardScaler()
    extra  = scaler.fit_transform(df_movies[["year", "avg_rating", "popularity"]])
    content_matrix = np.hstack([genre_dummies.values, extra])
    return content_matrix


def get_svd_recommendations(user_id, reconstructed, df_movies, df_ratings, top_n=5):
    """Pure collaborative filtering via SVD."""
    rated_movies = set(df_ratings[df_ratings["user_id"] == user_id]["movie_id"])
    scores       = reconstructed[user_id]
    scored_movies = [
        (i, scores[i]) for i in range(len(scores)) if i not in rated_movies
    ]
    scored_movies.sort(key=lambda x: x[1], reverse=True)
    top_ids = [m[0] for m in scored_movies[:top_n]]
    return df_movies[df_movies["movie_id"].isin(top_ids)][["title", "genre", "avg_rating"]]


def get_content_recommendations(movie_title, df_movies, content_matrix, top_n=5):
    """Content-based: find similar movies by genre/year/rating."""
    idx = df_movies[df_movies["title"] == movie_title].index
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[0]
    sim   = cosine_similarity(content_matrix[idx].reshape(1, -1), content_matrix)[0]
    order = np.argsort(sim)[::-1]
    order = [i for i in order if i != idx][:top_n]
    return df_movies.iloc[order][["title", "genre", "avg_rating"]]


def get_hybrid_recommendations(user_id, movie_title, reconstructed, df_movies,
                               df_ratings, content_matrix, top_n=5, alpha=0.6):
    """
    Hybrid score = alpha × SVD_score + (1 - alpha) × content_similarity
    alpha = weight given to collaborative filtering vs content-based
    """
    rated_movies = set(df_ratings[df_ratings["user_id"] == user_id]["movie_id"])
    svd_scores   = reconstructed[user_id]

    # Content similarity from seed movie
    seed_idx = df_movies[df_movies["title"] == movie_title].index
    if len(seed_idx) > 0:
        sim = cosine_similarity(
            content_matrix[seed_idx[0]].reshape(1, -1), content_matrix
        )[0]
    else:
        sim = np.ones(len(df_movies)) / len(df_movies)

    hybrid = {}
    for i in range(len(df_movies)):
        if i in rated_movies:
            continue
        score = alpha * svd_scores[i] + (1 - alpha) * sim[i] * 5.0
        hybrid[i] = score

    top_ids = sorted(hybrid, key=hybrid.get, reverse=True)[:top_n]
    result  = df_movies[df_movies["movie_id"].isin(top_ids)][["movie_id", "title", "genre", "avg_rating"]].copy()
    result["hybrid_score"] = result["movie_id"].map(lambda x: round(hybrid.get(x, 0), 3))
    return result.sort_values("hybrid_score", ascending=False)


# ─────────────────────────────────────────────
# E. EXPLAINABILITY
# ─────────────────────────────────────────────
def plot_latent_space(user_factors, n_show=50):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(user_factors[:n_show])
    plt.figure(figsize=(7, 5))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, c=range(n_show), cmap="tab20")
    plt.xlabel("Latent Factor 1 (taste dimension)"); plt.ylabel("Latent Factor 2")
    plt.title(f"User Taste Map — First {n_show} Users (SVD Latent Space)")
    plt.colorbar(label="User ID")
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/rec_user_latent.png", dpi=120)
    plt.close()
    print("Saved: artifacts/rec_user_latent.png")


def plot_rating_distribution(df_ratings):
    plt.figure(figsize=(6, 4))
    plt.hist(df_ratings["rating"], bins=20, color="steelblue", edgecolor="white")
    plt.xlabel("Rating (1-5)"); plt.ylabel("Count")
    plt.title("Rating Distribution")
    plt.tight_layout()
    plt.savefig("artifacts/rec_rating_dist.png", dpi=120)
    plt.close()
    print("Saved: artifacts/rec_rating_dist.png")


def plot_genre_popularity(df_movies):
    genre_counts = df_movies["genre"].value_counts()
    plt.figure(figsize=(7, 4))
    plt.bar(genre_counts.index, genre_counts.values, color="darkorchid")
    plt.xlabel("Genre"); plt.ylabel("Number of Movies"); plt.title("Genre Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("artifacts/rec_genre_dist.png", dpi=120)
    plt.close()
    print("Saved: artifacts/rec_genre_dist.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_movies, df_ratings = generate_recommendation_data(300, 50)
    print(f"Movies: {len(df_movies)}  |  Ratings: {len(df_ratings)}")
    print(df_ratings.head(5))

    matrix = build_user_item_matrix(df_ratings, 300, 50)
    svd, user_factors, item_factors, reconstructed = train_svd_model(matrix)
    content_matrix = build_content_matrix(df_movies)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({"svd": svd, "reconstructed": reconstructed,
                 "content_matrix": content_matrix,
                 "df_movies": df_movies, "df_ratings": df_ratings},
                "artifacts/rec_model.pkl")

    print("\n=== SVD Recommendations for User 0 ===")
    print(get_svd_recommendations(0, reconstructed, df_movies, df_ratings))

    print("\n=== Content Recommendations for 'Galactic Storm' ===")
    print(get_content_recommendations("Galactic Storm", df_movies, content_matrix))

    print("\n=== Hybrid Recommendations for User 0 (seed: Galactic Storm) ===")
    print(get_hybrid_recommendations(0, "Galactic Storm", reconstructed,
                                     df_movies, df_ratings, content_matrix))

    plot_latent_space(user_factors)
    plot_rating_distribution(df_ratings)
    plot_genre_popularity(df_movies)

    print("\nAll artifacts saved in artifacts/ folder.")
