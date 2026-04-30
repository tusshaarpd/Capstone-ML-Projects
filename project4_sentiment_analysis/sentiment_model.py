"""
PROJECT 4: Product Review Sentiment Analysis
Category: NLP-based System
Algorithm: TF-IDF + Logistic Regression (with SHAP-style word-level explanation)

E-commerce platforms (Amazon, Flipkart) analyze millions of reviews to:
- Surface product quality issues
- Rank sellers
- Alert operations teams to defects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os, re
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

# ─────────────────────────────────────────────
# C. DUMMY DATA GENERATION
# ─────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "absolutely love this product, works perfectly and exceeded my expectations",
    "excellent quality, fast delivery and great customer service",
    "amazing value for money, highly recommend to everyone",
    "this is fantastic, my family is very happy with the purchase",
    "superb build quality, looks premium and feels durable",
    "outstanding performance, exactly as described in the listing",
    "really impressed with the quality, will definitely buy again",
    "perfect gift, the recipient was very happy and satisfied",
    "works flawlessly, installation was easy and quick",
    "great product, does exactly what it promises",
]

NEGATIVE_TEMPLATES = [
    "terrible quality, broke after just one week of use",
    "very disappointed, product looks nothing like the pictures shown",
    "waste of money, completely stopped working within days",
    "awful experience, customer service was rude and unhelpful",
    "cheap material, falls apart easily and looks very fragile",
    "do not buy this, it is a total scam and fraud",
    "horrible product, arrived damaged and packaging was torn",
    "worst purchase ever made, returning immediately for refund",
    "poor quality control, missing parts and bad instructions",
    "extremely disappointed, not worth even half the price charged",
]

NEUTRAL_TEMPLATES = [
    "product is okay, nothing special but does the job",
    "average quality, meets basic requirements nothing more",
    "it is fine, some features are good and some are not",
    "decent for the price, but there is room for improvement",
    "mixed feelings, some aspects are great but delivery was slow",
    "works as expected, not amazing but not terrible either",
    "satisfactory product, would consider alternatives next time",
    "moderate quality, some issues but overall acceptable",
]

def augment(template, seed_offset=0):
    """Create slight variations to avoid exact duplicates."""
    rng = np.random.default_rng(seed_offset)
    fillers = ["honestly", "really", "quite", "pretty", "actually", ""]
    filler  = rng.choice(fillers)
    return f"{filler} {template}".strip()


def generate_review_data(n=900, seed=42):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n_pos = int(n * 0.45)
    n_neg = int(n * 0.35)
    n_neu = n - n_pos - n_neg

    reviews, labels = [], []
    for i in range(n_pos):
        t = POSITIVE_TEMPLATES[i % len(POSITIVE_TEMPLATES)]
        reviews.append(augment(t, i));    labels.append(2)   # Positive
    for i in range(n_neg):
        t = NEGATIVE_TEMPLATES[i % len(NEGATIVE_TEMPLATES)]
        reviews.append(augment(t, i+1000)); labels.append(0)  # Negative
    for i in range(n_neu):
        t = NEUTRAL_TEMPLATES[i % len(NEUTRAL_TEMPLATES)]
        reviews.append(augment(t, i+2000)); labels.append(1)  # Neutral

    # add some noise words
    noise_words = ["shipping", "box", "color", "size", "weight", "brand", "price"]
    noisy = []
    for r in reviews:
        if rng.random() < 0.3:
            w = rng.choice(noise_words)
            r = r + " " + w
        noisy.append(r)

    ratings = []
    for lbl in labels:
        if lbl == 2:   ratings.append(rng.integers(4, 6))
        elif lbl == 0: ratings.append(rng.integers(1, 3))
        else:          ratings.append(rng.integers(2, 5))

    df = pd.DataFrame({
        "review_text": noisy,
        "star_rating": ratings,
        "sentiment":   labels,
        "sentiment_label": ["Negative" if l == 0 else "Neutral" if l == 1 else "Positive"
                            for l in labels],
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────
# D. MODEL IMPLEMENTATION
# ─────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_sentiment_model(df):
    df = df.copy()
    df["clean_review"] = df["review_text"].apply(clean_text)

    X = df["clean_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    print("=" * 55)
    print("PROJECT 4 — Sentiment Analysis (TF-IDF + Logistic Regression)")
    print("=" * 55)
    print(classification_report(y_test, y_pred,
                                 target_names=["Negative", "Neutral", "Positive"]))
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    print(f"ROC-AUC (macro OvR): {auc:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/sentiment_pipeline.pkl")

    return pipeline, X_test, y_test, y_pred


# ─────────────────────────────────────────────
# E. EXPLAINABILITY
# ─────────────────────────────────────────────
def get_top_words(pipeline, n=15):
    """Extract top positive and negative words from LR coefficients."""
    vectorizer = pipeline.named_steps["tfidf"]
    clf        = pipeline.named_steps["clf"]
    features   = vectorizer.get_feature_names_out()

    # class order: 0=Negative, 1=Neutral, 2=Positive
    neg_coef = clf.coef_[0]
    pos_coef = clf.coef_[2]

    top_neg_idx = np.argsort(neg_coef)[::-1][:n]
    top_pos_idx = np.argsort(pos_coef)[::-1][:n]

    return (
        [(features[i], neg_coef[i]) for i in top_neg_idx],
        [(features[i], pos_coef[i]) for i in top_pos_idx],
    )


def plot_top_words(pipeline):
    neg_words, pos_words = get_top_words(pipeline)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    words, scores = zip(*neg_words)
    ax1.barh(words[::-1], scores[::-1], color="#e74c3c")
    ax1.set_title("Top Negative Sentiment Words")
    ax1.set_xlabel("LR Coefficient (strength)")

    words, scores = zip(*pos_words)
    ax2.barh(words[::-1], scores[::-1], color="#2ecc71")
    ax2.set_title("Top Positive Sentiment Words")
    ax2.set_xlabel("LR Coefficient (strength)")

    plt.suptitle("Model Explainability — Words that drive sentiment", fontsize=13)
    plt.tight_layout()
    plt.savefig("artifacts/sentiment_top_words.png", dpi=120)
    plt.close()
    print("Saved: artifacts/sentiment_top_words.png")


def explain_prediction(pipeline, text):
    """Word-level contribution to prediction (LIME-style manual)."""
    clean  = clean_text(text)
    words  = clean.split()
    probs  = pipeline.predict_proba([clean])[0]
    pred   = int(np.argmax(probs))
    label  = ["Negative", "Neutral", "Positive"][pred]

    vectorizer = pipeline.named_steps["tfidf"]
    clf        = pipeline.named_steps["clf"]

    word_contributions = []
    for w in set(words):
        if w in vectorizer.vocabulary_:
            idx  = vectorizer.vocabulary_[w]
            contrib = clf.coef_[pred, idx]
            word_contributions.append((w, contrib))

    word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return label, probs, word_contributions[:10]


def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Negative", "Neutral", "Positive"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=30); ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Sentiment")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("artifacts/sentiment_confusion.png", dpi=120)
    plt.close()
    print("Saved: artifacts/sentiment_confusion.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_review_data(900)
    print(f"Dataset shape: {df.shape}")
    print(df[["review_text", "sentiment_label"]].head(5))

    pipeline, X_test, y_test, y_pred = train_sentiment_model(df)
    plot_top_words(pipeline)
    plot_confusion(y_test, y_pred)

    test_reviews = [
        "This product is absolutely amazing and works perfectly!",
        "Complete waste of money, broke after one day.",
        "It is okay, nothing special but does the job.",
    ]
    print("\n=== Sample Predictions ===")
    for rev in test_reviews:
        label, probs, contribs = explain_prediction(pipeline, rev)
        print(f"\nReview: {rev}")
        print(f"Prediction: {label}  |  Probs: Neg={probs[0]:.2f} Neu={probs[1]:.2f} Pos={probs[2]:.2f}")
        print(f"Top words: {contribs[:3]}")

    print("\nAll artifacts saved in artifacts/ folder.")
