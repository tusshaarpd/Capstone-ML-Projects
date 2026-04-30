# 🎓 Machine Learning Capstone Projects

Five real-world, industry-relevant ML projects — each covering a **different ML category**, with full
implementation, explainability, and a working **Streamlit** app.

| # | Project | Category | Algorithm |
|---|---------|----------|-----------|
| 1 | Customer Churn Prediction | Supervised (Classification) | Random Forest |
| 2 | E-Commerce Customer Segmentation | Unsupervised (Clustering) | KMeans + PCA |
| 3 | Dynamic Ride Pricing Agent | Reinforcement Learning | Q-Learning |
| 4 | Product Review Sentiment Analyzer | NLP | TF-IDF + Logistic Regression |
| 5 | Movie Recommender | Recommendation (Hybrid) | SVD + Cosine Similarity |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# For each project:
cd project1_customer_churn
python churn_model.py        # train + save artifacts
streamlit run app.py         # launch web app
```

Each project folder contains:
- `*_model.py` — data generation, training, evaluation, explainability
- `app.py` — Streamlit web interface
- `artifacts/` — saved models and plots (auto-generated)

---

# 📘 PROJECT 1 — Customer Churn Prediction

### A. Problem Statement (Real Industry Use Case)
Telecom and SaaS companies lose **15–25% of customers every year**. Acquiring a new customer costs
**5× more** than retaining one. If we can predict *who* will churn next month, we can intervene with
discounts, better service, or upgraded plans.

**Business Impact:**
- 1% churn reduction at a $5B telecom = **$50M saved annually**
- Helps marketing teams prioritize retention budgets

### B. Algorithm: Random Forest Classifier
**How it works (simple terms):** A Random Forest is **200 small decision trees** that each look at the
customer slightly differently. Each tree votes "Churn" or "Stay" — the majority wins.

**Why Random Forest?**
- Handles mixed data types (numeric + categorical)
- Resistant to overfitting
- Provides feature importance out-of-the-box
- Works well even on noisy/imbalanced data

**Internal decisions:** Each tree splits the data on the most informative feature at every node
(e.g., "Is contract Month-to-Month?" → "Yes branch" vs "No branch").

### C. Dummy Data (1,000 rows)
| Feature | Description |
|---------|-------------|
| `tenure` | Months as customer (1–72) |
| `monthly_charges` | Monthly bill ($20–$120) |
| `total_charges` | Lifetime spend |
| `num_services` | Bundled services count |
| `contract_type` | Month-to-month / 1-year / 2-year |
| `payment_method` | Bank / Credit / Check |
| `tech_support` | 0/1 — has support add-on |
| `senior_citizen` | 0/1 |
| `num_complaints` | Support tickets opened |
| `churn` | **Target** — did customer leave? |

### D. Implementation
StandardScaler → train/test split (80/20) → Random Forest (200 trees, depth 8, balanced classes) →
classification report + ROC-AUC + confusion matrix.

### E. Explainability
- **Feature importance bar chart** — shows `contract_type`, `tenure`, `num_complaints` as top drivers
- **ROC curve** — visualizes trade-off between catching churners vs false alarms
- **Confusion matrix** — exact counts of correct/wrong predictions

### F. Streamlit App
Interactive sliders for all 9 customer attributes → live churn probability gauge with action recommendations.

### G. Output Interpretation
| Probability | Meaning | Action |
|-------------|---------|--------|
| ≥ 70% | Very high risk | Offer retention package immediately |
| 50–70% | Moderate risk | Proactive outreach |
| 30–50% | Low risk | Monitor next quarter |
| < 30% | Loyal customer | No action needed |

### H. Real-World Deployment
- Batch scoring: nightly Airflow job → CRM tags top 5% as "at risk"
- API: FastAPI endpoint integrated with customer service dashboard
- Monitoring: track prediction drift with Evidently AI; retrain monthly
- **Risk:** False positives waste retention budget; recalibrate threshold per quarter

### I. Simple Analogy
> Random Forest is like a **jury of 200 doctors**. Each doctor looks at slightly different symptoms.
> Most doctors saying "this patient will leave" → likely churn. The diversity of opinions = better diagnosis.

---

# 📘 PROJECT 2 — E-Commerce Customer Segmentation

### A. Problem Statement
A retailer with millions of customers can't run personalized campaigns for each one. They need to group
customers into **segments** with similar behavior (e.g., "Champions", "At Risk", "Hibernating") so each
segment gets a **tailored marketing strategy**.

**Business Impact:**
- 3–5× higher email open rates with segmentation
- Reduces marketing waste; focuses spend on high-LTV customers

### B. Algorithm: KMeans Clustering
**How it works:** Pick K (e.g., 4) "centroids" randomly, assign each customer to the nearest one,
then move centroids to the average of their group, and repeat until stable.

**Why KMeans?** Fast, scalable to millions of rows, easy to interpret. **PCA** is added to project the
high-dimensional clusters into a 2D plot for visualization.

### C. Dummy Data (800 rows) — Classic RFM Framework
| Feature | Meaning |
|---------|---------|
| `recency` | Days since last purchase |
| `frequency` | Number of purchases |
| `monetary` | Total $ spent |
| `avg_order_value` | $ per order |
| `customer_lifetime_value` | Predicted lifetime value |
| `total_sessions` | Website visits |
| `num_returns` | Returned orders |

### D. Implementation
StandardScaler → KMeans(k=4) → silhouette score → PCA(2D) for plotting → cluster profiling.

### E. Explainability
- **Elbow plot** — find optimal K (where inertia drops slow)
- **Silhouette score** — measures cluster separation quality
- **PCA scatter** — shows clusters as colored point clouds in 2D
- **RFM bar charts** — what makes each cluster unique

### F. Streamlit App
Enter a new customer's RFM stats → get assigned to a segment → see recommended marketing strategy.

### G. Output Interpretation
| Cluster | Profile | Strategy |
|---------|---------|----------|
| Champions | Recent + Frequent + High spend | Loyalty rewards |
| At Risk | Was active, now silent | Win-back campaign |
| New / Occasional | Low engagement | Onboarding nurture |
| Hibernating | Long inactive | Final discount or sunset |

### H. Real-World Deployment
- Daily Spark/SQL pipeline → segment table → push to email tool (Braze, Salesforce)
- A/B test segment-specific campaigns vs generic blast
- **Risk:** Segments drift over time → recluster monthly

### I. Simple Analogy
> KMeans is like **sorting laundry into 4 piles**. You glance at each item and toss it onto the pile
> it most resembles. After a few rounds, similar clothes end up together — automatically.

---

# 📘 PROJECT 3 — Dynamic Ride Pricing Agent

### A. Problem Statement
Ride-hailing platforms (Uber, Lyft, Ola) need to set surge prices in real time during high demand
(rain, concerts, rush hour). Too cheap → drivers leave the platform. Too expensive → riders cancel.
The goal: **maximize total revenue while keeping rider experience acceptable.**

**Business Impact:** Uber's surge pricing is estimated to add **$1B+ annually** in revenue.

### B. Algorithm: Q-Learning (Tabular Reinforcement Learning)
**How it works:** The agent tries different prices, observes the resulting revenue (reward), and
gradually learns a **Q-table**: "In state X, action Y gives expected reward Z." Over thousands of
simulated days, it converges on the optimal pricing policy.

**Why Q-Learning?**
- No need for labeled data — learns from interaction
- Naturally handles sequential decision-making
- Tabular Q-learning is interpretable (you can read the policy directly)

### C. Dummy Data — Simulated Environment
| Dimension | Values |
|-----------|--------|
| Demand | Low, Medium, High, Peak |
| Time of Day | Night, Morning, Afternoon, Evening |
| Competitor | Cheap, Similar, Expensive |
| Action (price multiplier) | 0.8×, 1.0×, 1.2×, 1.5×, 2.0× |
| Reward | Bookings × Price − low-booking penalty |

### D. Implementation
Custom `RidePricingEnv` (gym-style) → ε-greedy Q-Learning (3,000 episodes × 50 steps) → ε-decay from 1.0 → 0.05.

### E. Explainability
- **Learning curve** — revenue per episode improves over training
- **Policy heatmap** — at each demand × competitor combo, the chosen price multiplier
- **Q-value bar chart** — confidence in each action for the current state

### F. Streamlit App
Set demand/time/competitor → see recommended price + agent's confidence in each option → simulate 10 live steps.

### G. Output Interpretation
- Multiplier ≥ 1.5 → high demand surge
- Multiplier 1.0–1.2 → balanced pricing
- Multiplier < 1.0 → discount to attract riders in low demand

### H. Real-World Deployment
- Production: Deep Q-Network or PPO replaces tabular Q on real continuous state space
- Online learning with safety guardrails (caps on max price)
- **Risks:** Reward hacking, fairness concerns, regulatory backlash. A/B test against baseline; monitor rider complaints.

### I. Simple Analogy
> Q-Learning is like a **taxi driver experimenting with prices** every shift. After thousands of shifts,
> they learn: "It's a rainy Friday evening — I can charge double and still fill the car." That memory = the Q-table.

---

# 📘 PROJECT 4 — Product Review Sentiment Analyzer

### A. Problem Statement
Amazon, Flipkart, and Yelp process **millions of reviews daily**. Manual moderation is impossible.
Automated sentiment classification helps:
- Surface defective products (spike in negative reviews)
- Rank sellers by satisfaction
- Filter abusive content
- Generate executive sentiment dashboards

**Business Impact:** Catching a defective product 2 weeks earlier saves millions in returns + brand damage.

### B. Algorithm: TF-IDF + Logistic Regression
**How it works:**
1. **TF-IDF** converts text into numbers — each word gets a weight: high if frequent in *this* review
   but rare across all reviews (i.e., distinctive words).
2. **Logistic Regression** learns a coefficient per word for each class (Negative / Neutral / Positive).
3. Final prediction = words × coefficients → softmax probabilities.

**Why this combo?** Fast, interpretable (you can see exactly which words drive the prediction),
strong baseline that often beats heavier models on small data.

### C. Dummy Data (900 reviews)
- 45% Positive, 35% Negative, 20% Neutral
- Each review has: text, star rating (1–5), sentiment label
- Augmented with random fillers and noise words

### D. Implementation
Clean text → TF-IDF (1-2 grams, 5K features) → multinomial Logistic Regression → classification report + AUC.

### E. Explainability
- **Top positive/negative words** (LR coefficients) — `amazing`, `excellent`, `love` push positive;
  `terrible`, `waste`, `broke` push negative
- **Per-prediction word contributions** (LIME-style) — shows which words in *this* review tipped the decision
- **Confusion matrix**

### F. Streamlit App
Paste any review → see sentiment + probability bars + word-level contribution chart (green/red).

### G. Output Interpretation
| Label | Meaning | Operations Action |
|-------|---------|-------------------|
| Positive | Customer satisfied | Promote review |
| Neutral | Mixed | Flag for product team |
| Negative | Unhappy | Escalate to support |

### H. Real-World Deployment
- Kafka stream of new reviews → ML service → Elastic dashboard with sentiment trends
- Production upgrade: replace TF-IDF/LR with **DistilBERT** for nuanced reviews (sarcasm, emojis)
- **Risks:** Bias against certain demographics' writing style; sarcasm misclassification — needs human-in-the-loop for borderline cases

### I. Simple Analogy
> TF-IDF is a **highlighter** that marks the most distinctive words in a review.
> Logistic Regression then reads those highlights and votes:
> *"More green-highlighted words than red? → Positive."*

---

# 📘 PROJECT 5 — Movie Recommender (Hybrid)

### A. Problem Statement
**80% of Netflix viewing comes from recommendations.** Spotify, YouTube, Amazon all depend on accurate
recommendations to drive engagement and revenue. A 1% improvement in CTR = millions in retention.

**Business Impact:** Netflix estimates recommendations save **$1B/year** in churn reduction.

### B. Algorithm: SVD (Collaborative Filtering) + Content-Based Cosine Similarity
**How it works:**
1. **SVD (Singular Value Decomposition)** factorizes the giant user×movie ratings matrix into two
   smaller matrices: each user becomes a vector of "taste dimensions" (e.g., 20 hidden factors),
   each movie gets the same. Predicted rating = dot product of these vectors.
2. **Content similarity** — for each movie, build a feature vector (genre, year, rating, popularity)
   and compute cosine similarity to a seed movie.
3. **Hybrid** — final score = α × CF + (1−α) × content similarity. Best of both worlds.

**Why hybrid?**
- CF alone fails for **new users / new movies** (cold start)
- Content alone misses the wisdom of "users like you also liked..."
- Hybrid handles both gracefully

### C. Dummy Data
- 300 users × 50 movies → 4,800 ratings (sparse)
- Movies: `title`, `genre`, `year`, `avg_rating`, `popularity`
- Ratings generated from latent user-genre preferences (Dirichlet sampling)

### D. Implementation
Build user-item matrix → TruncatedSVD (20 components) → content matrix (one-hot genre + scaled features)
→ recommendation functions for CF / content / hybrid.

### E. Explainability
- **User latent space (PCA scatter)** — users with similar taste cluster together
- **Genre profile pie chart** — shows what user has watched
- **Hybrid score breakdown** — α slider explicitly shows CF vs content trade-off

### F. Streamlit App
Pick user ID + seed movie → choose mode (Hybrid / CF only / Content only) + α slider → see top-N recommendations
with star ratings, genre, and hybrid score.

### G. Output Interpretation
- High hybrid score → strong recommendation (both similar users liked it AND it matches your seed movie)
- Low score for a popular movie → already watched OR doesn't match your taste profile

### H. Real-World Deployment
- Offline batch: nightly Spark MLlib SVD → write top-100 per user to Redis/Cassandra
- Online: re-rank with real-time signals (current device, time, mood)
- Production upgrades: deep learning (Two-Tower models, transformers like SASRec, BERT4Rec)
- **Risks:** Filter bubble (users only see one type of content); popularity bias (head movies dominate); needs exploration via bandits

### I. Simple Analogy
> SVD is like a **dating app's compatibility score**. It discovers hidden personality dimensions
> (you may not even realize you like "slow-burn sci-fi with strong female leads"), then matches you
> with movies whose "personality" aligns. Two strangers who both love the same niche genres get matched
> automatically — even without knowing why.

---

## 📂 Repository Structure

```
Capstone-ML-Projects/
├── README.md
├── requirements.txt
├── project1_customer_churn/
│   ├── churn_model.py
│   ├── app.py
│   └── artifacts/
├── project2_customer_segmentation/
│   ├── segmentation_model.py
│   ├── app.py
│   └── artifacts/
├── project3_dynamic_pricing/
│   ├── pricing_agent.py
│   ├── app.py
│   └── artifacts/
├── project4_sentiment_analysis/
│   ├── sentiment_model.py
│   ├── app.py
│   └── artifacts/
└── project5_recommendation_system/
    ├── recommender_model.py
    ├── app.py
    └── artifacts/
```

## 🛠 Tech Stack
**Python · NumPy · Pandas · scikit-learn · Matplotlib · Streamlit · Joblib**

## 🎯 Learning Outcomes
After exploring this repo you will:
- Understand 5 different ML paradigms (supervised, unsupervised, RL, NLP, recsys)
- Have intuition for *why* each algorithm fits its problem
- Know how each model would actually ship in production
- Be able to demo any of these as a portfolio project
