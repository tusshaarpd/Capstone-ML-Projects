"""
PROJECT 3: Dynamic Ride Pricing Agent
Category  : Reinforcement Learning
Algorithm : Q-Learning (tabular)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY Q-LEARNING?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alternatives considered:
  • Deep Q-Network (DQN) → better for continuous state spaces but black box
  • PPO / Actor-Critic   → state-of-art but complex, needs many episodes
  • Bandit algorithms    → good for single-step, ignores future reward
  • Rule-based pricing   → no learning, misses edge cases

Q-Learning wins here because:
  1. TABULAR STATE SPACE — demand × time × competitor = 48 states; table fits perfectly
  2. FULLY INTERPRETABLE — you can literally read the Q-table and see every decision
  3. GUARANTEED convergence on finite MDP (Bellman equation)
  4. Easy to audit for regulatory compliance (no black box)
  5. Fast to train — 3,000 episodes in seconds on CPU

HOW Q-LEARNING WORKS (step by step):
  State  = current market condition (demand, time, competitor)
  Action = price multiplier to charge
  Reward = revenue earned in this time step
  Q(s,a) = expected TOTAL future reward if we take action a in state s

  Update rule (Bellman equation):
    Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') − Q(s,a)]
    where α = learning rate, γ = discount factor (values future reward)

  ε-greedy exploration:
    With probability ε → try a RANDOM price (explore)
    With probability 1-ε → use BEST KNOWN price (exploit)
    ε decays from 1.0 → 0.05 over training (more exploitation as we learn)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
class RidePricingEnv:
    """
    Simulated ride-hailing market environment.

    State  = (demand_level, time_of_day, competitor_price_level)
    Action = price multiplier index
    Reward = rides × price − low-booking penalty
    """
    PRICE_MULTIPLIERS = [0.8, 1.0, 1.2, 1.5, 2.0]
    BASE_PRICE        = 10.0

    DEMAND_LABELS    = ["Low", "Medium", "High", "Peak"]
    TIME_LABELS      = ["Night", "Morning", "Afternoon", "Evening"]
    COMP_LABELS      = ["Cheap", "Similar", "Expensive"]

    def __init__(self, seed=42):
        self.rng          = np.random.default_rng(seed)
        self.n_demand     = 4
        self.n_time       = 4
        self.n_competitor = 3
        self.n_actions    = len(self.PRICE_MULTIPLIERS)
        self.state        = None

    def reset(self):
        self.state = (
            self.rng.integers(0, self.n_demand),
            self.rng.integers(0, self.n_time),
            self.rng.integers(0, self.n_competitor),
        )
        return self.state

    def step(self, action):
        demand, time, competitor = self.state
        mult  = self.PRICE_MULTIPLIERS[action]
        price = self.BASE_PRICE * mult

        base_rate    = [0.30, 0.55, 0.75, 0.90][demand]
        price_pen    = (mult - 1.0) * 0.30
        comp_adj     = [-0.15, 0.0, 0.10][competitor]
        booking_prob = float(np.clip(base_rate - price_pen + comp_adj, 0.05, 0.95))
        n_rides      = int(self.rng.binomial(50, booking_prob))

        revenue = n_rides * price
        penalty = max(0, (10 - n_rides) * 5) if demand >= 2 else 0
        reward  = revenue - penalty

        next_state = (
            int(np.clip(demand + self.rng.integers(-1, 2), 0, 3)),
            (time + 1) % self.n_time,
            int(self.rng.integers(0, self.n_competitor)),
        )
        self.state = next_state
        return next_state, reward, False, {"n_rides": n_rides, "price": price}


# ─────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, env, lr=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.05):
        self.env           = env
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table = np.zeros((env.n_demand, env.n_time, env.n_competitor, env.n_actions))

        # Track training history
        self.epsilon_history = []
        self.reward_history  = []

    def choose_action(self, state, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_error  = reward + self.gamma * best_next - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─────────────────────────────────────────────
# D. TRAINING
# ─────────────────────────────────────────────
def train_agent(episodes=3000, steps_per_episode=50):
    env   = RidePricingEnv(seed=0)
    agent = QLearningAgent(env)

    episode_rewards = []
    random_rewards  = []   # baseline: random pricing

    for ep in range(episodes):
        state     = env.reset()
        total_rev = 0.0
        rand_rev  = 0.0

        for _ in range(steps_per_episode):
            # Learned agent
            action              = agent.choose_action(state)
            next_state, reward, _, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            total_rev += reward

            # Random baseline (same state, random action)
            rand_action = np.random.randint(env.n_actions)
            _, rand_r, _, _ = env.step(rand_action)
            rand_rev += rand_r

            state = next_state

        agent.decay_epsilon()
        episode_rewards.append(total_rev)
        random_rewards.append(rand_rev)
        agent.epsilon_history.append(agent.epsilon)

        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-500:])
            print(f"Episode {ep+1:4d} | Avg Revenue: ${avg:,.0f} | ε={agent.epsilon:.3f}")

    agent.reward_history = episode_rewards

    print("\n" + "=" * 60)
    print("PROJECT 3 — Dynamic Pricing Agent (Q-Learning)")
    print("=" * 60)
    final_avg   = np.mean(episode_rewards[-500:])
    random_avg  = np.mean(random_rewards[-500:])
    print(f"Learned policy avg revenue : ${final_avg:,.0f}")
    print(f"Random policy avg revenue  : ${random_avg:,.0f}")
    print(f"Improvement over random    : {(final_avg-random_avg)/random_avg*100:+.1f}%")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    np.save(os.path.join(ARTIFACT_DIR, "q_table.npy"), agent.q_table)

    return agent, episode_rewards, random_rewards


# ─────────────────────────────────────────────
# E. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_training_curve(episode_rewards, random_rewards):
    """
    Shows learning progress. Revenue should rise as ε falls.
    Compare agent vs random pricing to show what was learned.
    """
    window   = 100
    smoothed = pd.Series(episode_rewards).rolling(window).mean()
    rand_sm  = pd.Series(random_rewards).rolling(window).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    ax1.plot(episode_rewards, alpha=0.2, color="steelblue")
    ax1.plot(smoothed, color="steelblue", lw=2.5, label=f"Learned Agent ({window}-ep avg)")
    ax1.plot(rand_sm,  color="tomato",    lw=2,   linestyle="--", label="Random Policy")
    ax1.set_ylabel("Revenue per Episode ($)"); ax1.set_title("Learning Curve — Revenue over Time")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.annotate("Agent learns to beat\nrandom pricing →",
                 xy=(600, float(smoothed.iloc[600])), xytext=(800, float(smoothed.iloc[600]) * 0.85),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)

    ax2.plot(range(len(episode_rewards)),
             pd.Series(episode_rewards).rolling(window).mean() -
             pd.Series(random_rewards).rolling(window).mean(),
             color="#2ecc71", lw=2)
    ax2.axhline(0, color="black", lw=1, linestyle="--")
    ax2.fill_between(range(len(episode_rewards)),
                     (pd.Series(episode_rewards).rolling(window).mean() -
                      pd.Series(random_rewards).rolling(window).mean()),
                     0, alpha=0.2, color="#2ecc71")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Revenue Advantage ($)")
    ax2.set_title("Revenue Advantage of Learned Policy vs Random Pricing")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rl_training_curve.png"), dpi=130)
    plt.close()
    print("Saved: rl_training_curve.png")


def plot_policy_heatmap(agent, env):
    """
    For each demand × competitor combination, what price does the agent choose?
    Read: Peak demand + Expensive competitor → agent charges maximum (×2.0)
    """
    fig, axes = plt.subplots(1, env.n_time, figsize=(18, 4))

    for t_idx, ax in enumerate(axes):
        prices = np.zeros((env.n_demand, env.n_competitor))
        for d in range(env.n_demand):
            for c in range(env.n_competitor):
                best = int(np.argmax(agent.q_table[d, t_idx, c]))
                prices[d, c] = env.PRICE_MULTIPLIERS[best]

        im = ax.imshow(prices, cmap="RdYlGn_r", vmin=0.8, vmax=2.0, aspect="auto")
        ax.set_xticks(range(env.n_competitor)); ax.set_xticklabels(env.COMP_LABELS, fontsize=9)
        ax.set_yticks(range(env.n_demand));    ax.set_yticklabels(env.DEMAND_LABELS, fontsize=9)
        ax.set_xlabel("Competitor Price"); ax.set_ylabel("Demand" if t_idx == 0 else "")
        ax.set_title(f"{env.TIME_LABELS[t_idx]}")
        for d in range(env.n_demand):
            for c in range(env.n_competitor):
                ax.text(c, d, f"×{prices[d,c]}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if prices[d, c] >= 1.5 else "black")

    fig.colorbar(im, ax=axes, label="Price Multiplier", fraction=0.02)
    plt.suptitle("Optimal Price Multiplier Policy\n"
                 "Red = High surge pricing · Green = Discount / Normal",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rl_policy_heatmap.png"), dpi=130)
    plt.close()
    print("Saved: rl_policy_heatmap.png")


def plot_q_value_evolution(env):
    """
    Show Q-values for one key state converging as training progresses.
    Demonstrates how the agent's confidence improves with experience.
    """
    key_state = (3, 3, 1)  # Peak demand, Evening, Similar competitor
    snapshots = {ep: [] for ep in [100, 500, 1000, 3000]}

    agent = QLearningAgent(env)
    ep    = 0
    for _ in range(3000):
        state = env.reset()
        for _ in range(50):
            action = agent.choose_action(state)
            ns, r, _, _ = env.step(action)
            agent.update(state, action, r, ns)
            state = ns
        agent.decay_epsilon()
        ep += 1
        if ep in snapshots:
            snapshots[ep] = agent.q_table[key_state].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#bdc3c7", "#f39c12", "#e67e22", "#e74c3c"]
    for (ep_n, vals), col in zip(snapshots.items(), colors):
        if len(vals):
            ax.plot([f"×{m}" for m in env.PRICE_MULTIPLIERS], vals,
                    "o-", lw=2, color=col, label=f"Episode {ep_n}")

    ax.set_xlabel("Price Multiplier"); ax.set_ylabel("Q-Value (Expected Revenue)")
    ax.set_title(f"Q-Value Convergence for Peak Demand / Evening / Similar Competitor\n"
                 "As training progresses, the agent becomes confident about ×2.0")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rl_q_convergence.png"), dpi=130)
    plt.close()
    print("Saved: rl_q_convergence.png")


def plot_epsilon_decay(agent):
    """ε-greedy: shows the explore→exploit transition during training."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(agent.epsilon_history, color="purple", lw=2)
    ax.fill_between(range(len(agent.epsilon_history)),
                    agent.epsilon_history, alpha=0.2, color="purple")
    ax.axhline(0.05, color="red", lw=1.5, linestyle="--", label="Minimum ε = 0.05")
    ax.set_xlabel("Episode"); ax.set_ylabel("ε (Exploration Rate)")
    ax.set_title("ε-Greedy Decay: Exploration → Exploitation\n"
                 "High ε = tries random prices · Low ε = uses learned best price")
    ax.legend(); ax.grid(alpha=0.3)

    ax.annotate("Mostly exploring\n(random actions)", xy=(100, 0.6),
                fontsize=9, color="gray")
    ax.annotate("Mostly exploiting\n(best known action)", xy=(2500, 0.10),
                fontsize=9, color="red")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "rl_epsilon_decay.png"), dpi=130)
    plt.close()
    print("Saved: rl_epsilon_decay.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent, rewards, rand_rewards = train_agent(episodes=3000)
    env = agent.env

    plot_training_curve(rewards, rand_rewards)
    plot_policy_heatmap(agent, env)
    plot_epsilon_decay(agent)
    plot_q_value_evolution(RidePricingEnv(seed=0))

    print("\n✅ All Project 3 artifacts saved.")
