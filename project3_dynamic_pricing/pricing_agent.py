"""
PROJECT 3: Dynamic Pricing Agent
Category: Reinforcement Learning
Algorithm: Q-Learning (tabular)

A ride-hailing company (like Uber) wants to automatically set surge prices
based on current demand and time of day to maximize revenue.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# ENVIRONMENT DEFINITION
# ─────────────────────────────────────────────
class RidePricingEnv:
    """
    State  = (demand_level, time_of_day, competitor_price_level)
    Action = choose a price multiplier: [0.8, 1.0, 1.2, 1.5, 2.0]
    Reward = bookings × price (revenue) - penalty if demand drops too much

    Demand levels : 0=Low, 1=Medium, 2=High, 3=Peak
    Time of day   : 0=Night, 1=Morning, 2=Afternoon, 3=Evening
    Competitor    : 0=Cheap, 1=Similar, 2=Expensive
    """

    PRICE_MULTIPLIERS = [0.8, 1.0, 1.2, 1.5, 2.0]
    BASE_PRICE        = 10.0   # $10 base fare

    DEMAND_LABELS     = ["Low", "Medium", "High", "Peak"]
    TIME_LABELS       = ["Night", "Morning", "Afternoon", "Evening"]
    COMP_LABELS       = ["Cheap", "Similar", "Expensive"]

    def __init__(self, seed=42):
        self.rng               = np.random.default_rng(seed)
        self.n_demand          = 4
        self.n_time            = 4
        self.n_competitor      = 3
        self.n_actions         = len(self.PRICE_MULTIPLIERS)
        self.state             = None

    def reset(self):
        self.state = (
            self.rng.integers(0, self.n_demand),
            self.rng.integers(0, self.n_time),
            self.rng.integers(0, self.n_competitor),
        )
        return self.state

    def step(self, action):
        demand, time, competitor = self.state
        price_mult = self.PRICE_MULTIPLIERS[action]
        price      = self.BASE_PRICE * price_mult

        # Base booking rate depends on demand
        base_rate = [0.3, 0.55, 0.75, 0.90][demand]

        # Price sensitivity: higher price reduces bookings
        price_penalty = (price_mult - 1.0) * 0.30
        # Competitor adjustment: if competitor is cheap, we lose bookings
        comp_adjust = [-0.15, 0.0, 0.10][competitor]

        booking_prob = np.clip(base_rate - price_penalty + comp_adjust, 0.05, 0.95)
        n_rides      = self.rng.binomial(50, booking_prob)

        revenue = n_rides * price
        # Reward: revenue minus penalty for very low bookings (bad UX)
        low_booking_penalty = max(0, (10 - n_rides) * 5) if demand >= 2 else 0
        reward  = revenue - low_booking_penalty

        # Transition to next state (Markov)
        next_demand     = int(np.clip(demand + self.rng.integers(-1, 2), 0, 3))
        next_time       = (time + 1) % self.n_time
        next_competitor = self.rng.integers(0, self.n_competitor)
        self.state      = (next_demand, next_time, next_competitor)

        done = False   # continuous environment
        return self.state, reward, done, {"n_rides": n_rides, "price": price}


# ─────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, env, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.env           = env
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Q-table: shape (demand, time, competitor, action)
        self.q_table = np.zeros((env.n_demand, env.n_time, env.n_competitor, env.n_actions))

    def choose_action(self, state, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.env.n_actions)   # explore
        return int(np.argmax(self.q_table[state]))          # exploit

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state][action]
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

    for ep in range(episodes):
        state      = env.reset()
        total_rev  = 0.0

        for _ in range(steps_per_episode):
            action             = agent.choose_action(state)
            next_state, reward, _, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state      = next_state
            total_rev += reward

        agent.decay_epsilon()
        episode_rewards.append(total_rev)

        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-500:])
            print(f"Episode {ep+1:4d} | Avg Revenue: ${avg:,.0f} | ε={agent.epsilon:.3f}")

    print("\n" + "=" * 55)
    print("PROJECT 3 — Dynamic Pricing Agent (Q-Learning)")
    print("=" * 55)
    print(f"Training episodes : {episodes}")
    print(f"Final ε (explore) : {agent.epsilon:.4f}")
    print(f"Avg revenue (last 500 eps): ${np.mean(episode_rewards[-500:]):,.0f}")

    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/q_table.npy", agent.q_table)

    return agent, episode_rewards


# ─────────────────────────────────────────────
# E. EXPLAINABILITY
# ─────────────────────────────────────────────
def print_policy(agent, env):
    print("\n=== Learned Pricing Policy ===")
    print(f"{'Demand':<12} {'Time':<12} {'Competitor':<12} {'Best Price'}")
    print("-" * 55)
    for d in range(env.n_demand):
        for t in range(env.n_time):
            for c in range(env.n_competitor):
                best_action = int(np.argmax(agent.q_table[d, t, c]))
                mult        = env.PRICE_MULTIPLIERS[best_action]
                price       = env.BASE_PRICE * mult
                print(
                    f"{env.DEMAND_LABELS[d]:<12} {env.TIME_LABELS[t]:<12} "
                    f"{env.COMP_LABELS[c]:<12} ${price:.2f} (×{mult})"
                )


def plot_training_curve(episode_rewards):
    window = 100
    smoothed = pd.Series(episode_rewards).rolling(window).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, alpha=0.3, color="steelblue", label="Raw")
    plt.plot(smoothed, color="darkorange", lw=2, label=f"{window}-ep moving avg")
    plt.xlabel("Episode"); plt.ylabel("Total Revenue ($)")
    plt.title("Agent Learning Curve — Revenue improves over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/rl_training_curve.png", dpi=120)
    plt.close()
    print("Saved: artifacts/rl_training_curve.png")


def plot_heatmap(agent, env, time_idx=3):
    """Heatmap of best prices for Evening, varying demand vs competitor."""
    prices = np.zeros((env.n_demand, env.n_competitor))
    for d in range(env.n_demand):
        for c in range(env.n_competitor):
            best = int(np.argmax(agent.q_table[d, time_idx, c]))
            prices[d, c] = env.PRICE_MULTIPLIERS[best]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(prices, cmap="RdYlGn_r", vmin=0.8, vmax=2.0)
    ax.set_xticks(range(env.n_competitor)); ax.set_xticklabels(env.COMP_LABELS)
    ax.set_yticks(range(env.n_demand));    ax.set_yticklabels(env.DEMAND_LABELS)
    ax.set_xlabel("Competitor Price"); ax.set_ylabel("Demand Level")
    ax.set_title(f"Optimal Price Multiplier ({env.TIME_LABELS[time_idx]})")
    for d in range(env.n_demand):
        for c in range(env.n_competitor):
            ax.text(c, d, f"×{prices[d, c]}", ha="center", va="center", fontsize=11)
    plt.colorbar(im, label="Price Multiplier")
    plt.tight_layout()
    plt.savefig("artifacts/rl_policy_heatmap.png", dpi=120)
    plt.close()
    print("Saved: artifacts/rl_policy_heatmap.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent, rewards = train_agent(episodes=3000)
    env = agent.env
    print_policy(agent, env)
    plot_training_curve(rewards)
    plot_heatmap(agent, env)
    print("\nAll artifacts saved in artifacts/ folder.")
