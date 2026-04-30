"""
PROJECT 4 — Streamlit App: Sentiment Analysis Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from sentiment_model import (
    generate_review_data, train_sentiment_model,
    explain_prediction, plot_top_words,
)

@st.cache_resource
def load_model():
    artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    pipe_path    = os.path.join(artifact_dir, "sentiment_pipeline.pkl")
    if not os.path.exists(pipe_path):
        df = generate_review_data(900)
        train_sentiment_model(df)
        plot_top_words(joblib.load(pipe_path))
    return joblib.load(pipe_path)

pipeline = load_model()

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
st.title("💬 Product Review Sentiment Analyzer")
st.markdown(
    """
**Goal:** Automatically classify customer reviews as **Positive / Neutral / Negative**
to help e-commerce teams monitor product quality, seller performance, and customer satisfaction at scale.
"""
)

SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}
SENTIMENT_EMOJI  = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter a Product Review")
    user_review = st.text_area(
        "Type or paste any review:",
        value="This product is absolutely amazing! Works perfectly and great value for money.",
        height=120,
    )

    if st.button("🔍 Analyze Sentiment"):
        label, probs, contribs = explain_prediction(pipeline, user_review)
        emoji  = SENTIMENT_EMOJI[label]
        color  = SENTIMENT_COLORS[label]

        st.markdown(
            f"""
            <div style='background:{color};padding:18px;border-radius:10px;text-align:center;margin:10px 0'>
                <h1 style='color:white;margin:0'>{emoji} {label}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bars = ax.barh(
            ["Negative", "Neutral", "Positive"],
            [probs[0], probs[1], probs[2]],
            color=["#e74c3c", "#f39c12", "#2ecc71"],
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Sentiment Probabilities")
        for bar, p in zip(bars, [probs[0], probs[1], probs[2]]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{p:.1%}", va="center", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

        # Word-level explanation
        if contribs:
            st.subheader("Key Words Driving This Prediction")
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            words_  = [c[0] for c in contribs]
            scores_ = [c[1] for c in contribs]
            bar_colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in scores_]
            ax2.barh(words_[::-1], scores_[::-1], color=bar_colors[::-1])
            ax2.axvline(0, color="black", lw=0.8)
            ax2.set_xlabel("Contribution Score")
            ax2.set_title("Word Contributions (green = positive push, red = negative push)")
            plt.tight_layout()
            st.pyplot(fig2)

with col2:
    st.subheader("Interpretation Guide")
    st.markdown("""
| Score | Meaning | Action |
|-------|---------|--------|
| **Positive** | Customer is satisfied | Flag as good review |
| **Neutral** | Mixed or indifferent | Monitor product |
| **Negative** | Customer is unhappy | Escalate to support |
    """)

    st.subheader("Top Signal Words")
    img_path = os.path.join(os.path.dirname(__file__), "artifacts", "sentiment_top_words.png")
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)

st.markdown("---")
st.subheader("Batch Analysis — Sample Reviews")
df = generate_review_data(200)
sample = df[["review_text", "sentiment_label", "star_rating"]].head(12)
st.dataframe(sample, use_container_width=True)

st.markdown("---")
st.subheader("Confusion Matrix")
cm_path = os.path.join(os.path.dirname(__file__), "artifacts", "sentiment_confusion.png")
if os.path.exists(cm_path):
    st.image(cm_path, width=400)

st.caption(
    "**Simple Analogy:** TF-IDF is like a highlighter that marks the most *unique* and *important* "
    "words in a review. Logistic Regression then reads those highlights and votes: "
    "'More highlighted positive words? → Positive review.' "
    "The model learned which words matter by reading thousands of examples."
)
