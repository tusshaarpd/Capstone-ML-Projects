"""
PROJECT 3 — Streamlit App: Dynamic Pricing Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from pricing_agent import RidePricingEnv, QLearningAgent, train_agent, plot_heatmap

@st.cache_resource
def load_agent():
    artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    q_path       = os.path.join(artifact_dir, "q_table.npy")
    if not os.path.exists(q_path):
        agent, _ = train_agent(episodes=3000)
    else:
        env = RidePricingEnv(seed=0)
        agent = QLearningAgent(env)
        agent.q_table = np.load(q_path)
        agent.epsilon = 0.0
    return agent

agent = load_agent()
env   = agent.env

st.set_page_config(page_title="Dynamic Pricing", page_icon="🚗", layout="wide")
st.title("🚗 Dynamic Pricing Agent — Ride Hailing")
st.markdown(
    """
**Goal:** Automatically set the right surge price in real time based on demand,
time of day, and competitor pricing — just like Uber Surge or Lyft Prime Time.
"""
)

st.sidebar.header("Current Market Conditions")
demand_label = st.sidebar.selectbox("Demand Level", env.DEMAND_LABELS)
time_label   = st.sidebar.selectbox("Time of Day", env.TIME_LABELS)
comp_label   = st.sidebar.selectbox("Competitor Price", env.COMP_LABELS)

d = env.DEMAND_LABELS.index(demand_label)
t = env.TIME_LABELS.index(time_label)
c = env.COMP_LABELS.index(comp_label)

state       = (d, t, c)
best_action = int(np.argmax(agent.q_table[state]))
multiplier  = env.PRICE_MULTIPLIERS[best_action]
price       = env.BASE_PRICE * multiplier

# Q-values for all actions
q_vals = agent.q_table[state]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Recommended Price")
    color = "#e74c3c" if multiplier >= 1.5 else "#f39c12" if multiplier >= 1.2 else "#2ecc71"
    st.markdown(
        f"""
        <div style='background:{color};padding:20px;border-radius:10px;text-align:center'>
            <h1 style='color:white;margin:0'>${price:.2f}</h1>
            <p style='color:white;font-size:18px'>Surge multiplier: ×{multiplier}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if multiplier >= 1.5:
        st.error(f"🔴 High demand surge. Premium pricing applied.")
    elif multiplier >= 1.2:
        st.warning(f"🟡 Moderate surge. Balanced pricing.")
    elif multiplier == 1.0:
        st.info(f"🔵 Normal pricing. No surge needed.")
    else:
        st.success(f"🟢 Discount pricing to attract riders in low demand.")

    st.subheader("Q-Values (Agent Confidence)")
    fig, ax = plt.subplots(figsize=(5, 3))
    bar_colors = ["#e74c3c" if i == best_action else "#95a5a6" for i in range(env.n_actions)]
    ax.bar([f"×{m}" for m in env.PRICE_MULTIPLIERS], q_vals, color=bar_colors)
    ax.set_xlabel("Price Multiplier"); ax.set_ylabel("Q-Value (Expected Revenue)")
    ax.set_title("Action Values — Red = Chosen")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Policy Heatmap (Evening Demand)")
    heatmap_path = os.path.join(os.path.dirname(__file__), "artifacts", "rl_policy_heatmap.png")
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, use_container_width=True)

    st.subheader("Training Learning Curve")
    curve_path = os.path.join(os.path.dirname(__file__), "artifacts", "rl_training_curve.png")
    if os.path.exists(curve_path):
        st.image(curve_path, use_container_width=True)

st.markdown("---")
st.subheader("Simulate 10 Live Steps with This Condition")
if st.button("▶ Run Simulation"):
    sim_env = RidePricingEnv(seed=99)
    sim_env.state = state
    results = []
    for _ in range(10):
        act        = agent.choose_action(sim_env.state, greedy=True)
        ns, rew, _, info = sim_env.step(act)
        results.append({
            "Demand":       env.DEMAND_LABELS[state[0]],
            "Price ($)":    f"${info['price']:.2f}",
            "Rides":        info["n_rides"],
            "Revenue ($)":  f"${rew:.0f}",
        })
        state = ns
    import pandas as pd
    st.dataframe(pd.DataFrame(results), use_container_width=True)

st.caption(
    "**Simple Analogy:** Q-Learning is like a taxi driver who tries different prices every shift. "
    "Over thousands of shifts they remember: 'When it's raining on a Friday evening, "
    "I can charge 2× and still fill the car.' That memory = the Q-table."
)
