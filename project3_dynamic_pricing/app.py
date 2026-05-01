"""
PROJECT 3 — Streamlit App: Dynamic Pricing Agent (Enhanced)
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from pricing_agent import (
    RidePricingEnv, QLearningAgent, train_agent,
    plot_training_curve, plot_policy_heatmap,
    plot_epsilon_decay, plot_q_value_evolution, ARTIFACT_DIR,
)

@st.cache_resource
def load_agent():
    qp = os.path.join(ARTIFACT_DIR, "q_table.npy")
    if not os.path.exists(qp):
        agent, rewards, rand_rewards = train_agent(3000)
        plot_training_curve(rewards, rand_rewards)
        plot_policy_heatmap(agent, agent.env)
        plot_epsilon_decay(agent)
        plot_q_value_evolution(RidePricingEnv(seed=0))
    env   = RidePricingEnv(seed=0)
    agent = QLearningAgent(env)
    agent.q_table = np.load(qp)
    agent.epsilon = 0.0
    return agent

agent = load_agent()
env   = agent.env

st.set_page_config(page_title="Dynamic Pricing", page_icon="🚗", layout="wide")
st.title("🚗 Dynamic Ride Pricing Agent")
st.markdown("""
> **Business Goal:** Set the optimal surge price in real time — maximize revenue while keeping
> enough riders booking. Modelled after Uber Surge / Lyft Prime Time pricing systems.
""")

with st.expander("🧠 How Q-Learning Works (Algorithm Explainer)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Q-Learning — Step by Step:**

1. The agent observes the **state** (demand, time, competitor)
2. Chooses an **action** (price multiplier) — randomly at first (ε-greedy)
3. Environment returns a **reward** (revenue earned)
4. Agent updates its **Q-table**: *"In this situation, that price earned X revenue"*
5. Over thousands of trials, Q-values converge → the agent has MEMORIZED the best price for every scenario

**The Q-table is the 'brain':**
It's a 4×4×3×5 lookup table (demand × time × competitor × price)
— after training, just look up the row and pick the highest Q-value.
        """)
    with col_b:
        st.markdown("""
**Bellman Update Equation (simplified):**

```
Q(state, action) +=
  learning_rate × (
    reward_received
    + discount × best_future_Q
    - current_Q
  )
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α (learning rate) | 0.1 | How fast we update |
| γ (discount) | 0.95 | How much we value future revenue |
| ε start → end | 1.0 → 0.05 | Explore → Exploit transition |
        """)

st.markdown("---")
st.sidebar.header("🌍 Current Market Conditions")
demand_lbl = st.sidebar.selectbox("Demand Level", env.DEMAND_LABELS,
                                   help="Low=few riders, Peak=rush hour/concert")
time_lbl   = st.sidebar.selectbox("Time of Day", env.TIME_LABELS)
comp_lbl   = st.sidebar.selectbox("Competitor Price", env.COMP_LABELS,
                                   help="Is Uber/Lyft charging more or less than you?")

d = env.DEMAND_LABELS.index(demand_lbl)
t = env.TIME_LABELS.index(time_lbl)
c = env.COMP_LABELS.index(comp_lbl)

state       = (d, t, c)
best_action = int(np.argmax(agent.q_table[state]))
multiplier  = env.PRICE_MULTIPLIERS[best_action]
price       = env.BASE_PRICE * multiplier
q_vals      = agent.q_table[state]

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Agent's Recommendation")
    color = "#e74c3c" if multiplier >= 1.5 else "#f39c12" if multiplier >= 1.2 else "#2ecc71"
    st.markdown(
        f"""
        <div style='background:{color};padding:22px;border-radius:12px;text-align:center'>
            <h1 style='color:white;margin:0'>${price:.2f}</h1>
            <p style='color:white;font-size:18px;margin:4px'>Surge multiplier: ×{multiplier}</p>
            <p style='color:white;font-size:13px'>Base fare × {multiplier} = ${price:.2f}</p>
        </div>
        """, unsafe_allow_html=True,
    )
    if multiplier >= 1.5:
        st.error("🔴 Peak pricing engaged — high demand justifies premium rate.")
    elif multiplier >= 1.2:
        st.warning("🟡 Moderate surge — balanced revenue vs volume trade-off.")
    elif multiplier == 1.0:
        st.info("🔵 Normal rate — steady demand, no surge needed.")
    else:
        st.success("🟢 Discount pricing — attract riders during slow period.")

    st.subheader("🎯 Q-Values for All Prices")
    st.markdown("*Higher Q-value = agent expects more revenue from that price*")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bar_colors = [color if i == best_action else "#bdc3c7" for i in range(env.n_actions)]
    bars = ax.bar([f"×{m}" for m in env.PRICE_MULTIPLIERS], q_vals, color=bar_colors)
    ax.set_xlabel("Price Multiplier"); ax.set_ylabel("Q-Value (Expected Revenue $)")
    ax.set_title(f"Action Confidence — Best: ×{multiplier}")
    for bar, v in zip(bars, q_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{v:.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("🗺 Full Policy Heatmap")
    hp = os.path.join(ARTIFACT_DIR, "rl_policy_heatmap.png")
    if os.path.exists(hp):
        st.image(hp, use_container_width=True)
        st.markdown("""
        **How to read:** Each cell shows the price multiplier the agent LEARNED to use.
        - 🟥 Red cells = high surge (×1.5 or ×2.0)
        - 🟩 Green cells = discount (×0.8)
        - Pattern: Peak demand + expensive competitor → always charge maximum
        """)

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Learning Curve", "⚡ Epsilon Decay",
    "🔮 Q-Value Convergence", "▶ Live Simulation"
])

def show_img(name, caption=""):
    p = os.path.join(ARTIFACT_DIR, name)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)

with tab1:
    show_img("rl_training_curve.png")
    st.markdown("""
    - **Top chart:** Revenue per episode (smoothed). The agent improves as it explores more states.
    - **Bottom chart:** Revenue advantage over a random pricing policy.
    - Notice: the gap widens as training progresses — the agent learns increasingly smart prices.
    """)

with tab2:
    show_img("rl_epsilon_decay.png")
    st.markdown("""
    - **High ε (start):** Agent tries random prices to discover what works (exploration)
    - **ε decays:** Agent shifts to using its best-known prices (exploitation)
    - **Final ε = 0.05:** Agent still explores 5% of the time to avoid getting stuck
    """)

with tab3:
    show_img("rl_q_convergence.png")
    st.markdown("""
    Shows how Q-values for ONE state evolve during training.
    Early: all Q-values are 0 (agent knows nothing).
    Late: the highest Q-value clearly stands out — agent has learned the best action.
    """)

with tab4:
    st.subheader("Simulate 20 Pricing Steps")
    if st.button("▶ Run Live Simulation"):
        sim_env = RidePricingEnv(seed=99)
        sim_env.state = state
        results = []
        for step in range(20):
            s   = sim_env.state
            act = agent.choose_action(s, greedy=True)
            ns, rew, _, info = sim_env.step(act)
            results.append({
                "Step": step + 1,
                "Demand": env.DEMAND_LABELS[s[0]],
                "Time": env.TIME_LABELS[s[1]],
                "Competitor": env.COMP_LABELS[s[2]],
                "Price ($)": f"${info['price']:.2f}",
                "Rides": info["n_rides"],
                "Revenue ($)": f"${rew:.0f}",
            })
        df_sim = pd.DataFrame(results)
        st.dataframe(df_sim, use_container_width=True)
        total = sum(float(r["Revenue ($)"].replace("$","")) for r in results)
        st.success(f"Total simulated revenue: **${total:,.0f}**")

st.caption("""
**Simple Analogy:** Q-Learning is like a taxi driver running a personal experiment every shift.
First week: tries random prices. Keeps mental notes of what earned the most money.
After 3,000 shifts: "Rainy Friday evening with Uber charging surge? I can charge ×2.0 and still fill every seat."
That mental notebook = the Q-table.
""")
