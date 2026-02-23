"""
Complete Training Framework for ChronoEnv v2.0

Supports both PPO and PPO+PRM with 3-seed comparison.

Usage:
    # PPO only (baseline)
    python3 train_complete.py --method ppo --seeds 0,1,2 --episodes 5000
    
    # PPO + PRM (ours)
    python3 train_complete.py --method prm --seeds 0,1,2 --episodes 5000
    
    # Full comparison
    python3 train_complete.py --method both --seeds 0,1,2 --episodes 5000
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import json

from chrono_env_v2 import ChronoEnvTimeCritical


class PPOPolicy(nn.Module):
    """PPO Policy Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        self.action_dim = action_dim
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(state)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value
    
    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_logits, value = self.forward(state_tensor)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action and return log prob, value, entropy."""
        action_logits, value = self.forward(state)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        distribution = Categorical(action_probs)
        
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        
        return log_prob, value, entropy


class PRMLoss(nn.Module):
    """PRM Loss for Temporal Regret."""
    
    def __init__(
        self,
        redundancy_penalty: float = 0.5,
        critical_reward: float = 0.5,
        blind_penalty: float = 1.0,
        precision_reward: float = 1.0,
        critical_buffer_threshold: float = 20.0,
        uncertainty_threshold: float = 30.0,
    ):
        super().__init__()
        self.redundancy_penalty = redundancy_penalty
        self.critical_reward = critical_reward
        self.blind_penalty = blind_penalty
        self.precision_reward = precision_reward
        self.critical_buffer_threshold = critical_buffer_threshold
        self.uncertainty_threshold = uncertainty_threshold
    
    def compute_prm(self, state, action, info, previous_action=None) -> float:
        """Compute PRM reward for a single step."""
        prm_score = 0.0
        
        if action == 2:  # CHECK_TIME
            # 1. Penalty for redundant checks
            if previous_action == 2:
                prm_score -= self.redundancy_penalty
            
            # 2. Reward for critical moment queries
            estimated_buffer = info.get('time_remaining', 100)
            uncertainty = info.get('uncertainty', 0)
            
            if estimated_buffer < self.critical_buffer_threshold and uncertainty > self.uncertainty_threshold:
                prm_score += self.critical_reward
        
        elif action == 3:  # SUBMIT
            # 3. Penalty for blind submission
            work_done = info.get('work_done', 0)
            true_work_required = info.get('true_work_required', 100)
            
            if work_done < true_work_required * 0.8:  # Less than 80% done
                prm_score -= self.blind_penalty
            
            # 4. Reward for precision
            success = info.get('total_reward', 0) > 0
            if success:
                prm_score += self.precision_reward
        
        return prm_score


class PPOAgent:
    """PPO Agent with optional PRM."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        use_prm: bool = False,
        prm_config: Dict = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PPOPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.use_prm = use_prm
        
        if use_prm:
            prm_config = prm_config or {}
            self.prm = PRMLoss(**prm_config)
        else:
            self.prm = None
        
        self.last_action = None
    
    def get_action(self, state: np.ndarray) -> Tuple[int, Dict]:
        action, log_prob, value = self.policy.act(state)
        
        metadata = {
            "log_prob": log_prob,
            "value": value,
            "is_check": (action == 2),
        }
        
        if self.last_action is not None:
            metadata["previous_action"] = self.last_action
        
        self.last_action = action
        return action, metadata
    
    def compute_returns(self, rewards: List[float], values: List[float], dones: List[bool]) -> List[float]:
        returns = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
        
        return returns
    
    def update(self, states, actions, rewards, log_probs, values, dones, infos) -> Dict:
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        returns = torch.FloatTensor(self.compute_returns(rewards, values, dones)).to(self.device)
        advantages = returns - torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.optimizer.zero_grad()
        
        log_probs, values, entropy = self.policy.evaluate(states, actions)
        ratios = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        entropy_loss = -entropy.mean()
        
        # PRM regularization
        prm_loss = 0.0
        if self.use_prm and len(infos) > 1:
            prm_scores = []
            for i in range(len(infos) - 1):
                score = self.prm.compute_prm(
                    state=states[i],
                    action=actions[i],
                    info=infos[i],
                    previous_action=actions[i-1] if i > 0 else None,
                )
                prm_scores.append(score)
            prm_loss = torch.tensor(prm_scores).mean()
        
        total_loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss
        
        if self.use_prm:
            total_loss += 0.3 * prm_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "prm_loss": prm_loss.item() if self.use_prm else 0,
            "total_loss": total_loss.item(),
        }
    
    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)


def train_episodes(
    env: ChronoEnvTimeCritical,
    agent: PPOAgent,
    n_episodes: int = 5000,
    n_steps: int = 128,
    eval_every: int = 100,
    eval_episodes: int = 100,
    seed: int = 0,
    verbose: bool = True,
):
    """Train PPO agent and return training history."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    # Storage
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    infos = []
    
    # History
    history = {
        "episodes": [],
        "eval_episodes": [],  # Store eval episode indices
        "success_rate": [],
        "avg_reward": [],
        "avg_queries": [],
        "eval_success_rate": [],
        "eval_avg_reward": [],
        "eval_avg_queries": [],
    }
    
    episode_count = 0
    step_count = 0
    
    while episode_count < n_episodes:
        # Get action
        action, metadata = agent.get_action(state)
        
        # Execute
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Store
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(metadata["log_prob"])
        values.append(metadata["value"])
        dones.append(terminated or truncated)
        infos.append(info)
        
        # Update state
        state = next_state
        episode_reward += reward
        episode_steps += 1
        step_count += 1
        
        # Update policy
        if step_count % n_steps == 0 and len(states) > 0:
            training_info = agent.update(states, actions, rewards, log_probs, values, dones, infos)
            states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []
            infos = []
        
        # Check if done
        if terminated or truncated:
            episode_count += 1
            
            # Record metrics
            query_count = info.get("query_count", 0)
            
            history["episodes"].append(episode_count)
            history["success_rate"].append(1 if episode_reward > 0 else 0)
            history["avg_reward"].append(episode_reward)
            history["avg_queries"].append(query_count)
            
            # Print progress
            if verbose and (episode_count % 100 == 0 or episode_count == 1):
                avg_reward = np.mean(history["avg_reward"][-100:])
                success_rate = np.mean(history["success_rate"][-100:])
                avg_queries = np.mean(history["avg_queries"][-100:])
                
                print(f"Seed {seed}, Episode {episode_count:5d}: "
                      f"reward={avg_reward:7.2f}, "
                      f"success_rate={success_rate*100:5.1f}%, "
                      f"queries={avg_queries:.1f}")
            
            # Evaluate periodically
            if episode_count % eval_every == 0 or episode_count == 1:
                eval_results = evaluate_policy(env, agent, eval_episodes)
                history["eval_success_rate"].append(eval_results["success_rate"])
                history["eval_avg_reward"].append(eval_results["avg_reward"])
                history["eval_avg_queries"].append(eval_results["avg_queries"])
                # Track eval episode for plotting
                history["eval_episodes"].append(episode_count)
            
            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            agent.last_action = None
    
    return history


def evaluate_policy(env: ChronoEnvTimeCritical, agent: PPOAgent, n_episodes: int = 100) -> Dict:
    """Evaluate policy without training."""
    np.random.seed(42)  # Fixed seed for evaluation
    
    episode_rewards = []
    query_counts = []
    success_rates = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        
        while not terminated:
            action, _ = agent.get_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        query_counts.append(info.get("query_count", 0))
        success_rates.append(1 if episode_reward > 0 else 0)
    
    return {
        "success_rate": np.mean(success_rates),
        "avg_reward": np.mean(episode_rewards),
        "avg_queries": np.mean(query_counts),
        "success_std": np.std(success_rates),
    }


def plot_results(all_histories: Dict, method: str, output_dir: str = "."):
    """Generate the 3 key plots for the paper."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Learning Curve - Success Rate (use eval points only)
    ax = axes[0, 0]
    for method_name, histories in all_histories.items():
        all_success = [h["eval_success_rate"] for h in histories]
        # Get the eval episode indices (stored during training)
        all_eval_episodes = [h.get("eval_episodes", list(range(100, len(h["eval_success_rate"])*100+1, 100))) for h in histories]
        mean_eval_episodes = np.mean(all_eval_episodes, axis=0).astype(int)
        mean_success = np.mean(all_success, axis=0)
        std_success = np.std(all_success, axis=0)
        
        ax.plot(mean_eval_episodes, mean_success, label=method_name, linewidth=2, marker='o')
        ax.fill_between(mean_eval_episodes, mean_success - std_success, mean_success + std_success, alpha=0.2)
    
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Learning Curve - Success Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Query Efficiency (use eval points only)
    ax = axes[0, 1]
    for method_name, histories in all_histories.items():
        all_queries = [h["eval_avg_queries"] for h in histories]
        all_eval_episodes = [h.get("eval_episodes", list(range(100, len(h["eval_avg_queries"])*100+1, 100))) for h in histories]
        mean_eval_episodes = np.mean(all_eval_episodes, axis=0).astype(int)
        mean_queries = np.mean(all_queries, axis=0)
        std_queries = np.std(all_queries, axis=0)
        
        ax.plot(mean_eval_episodes, mean_queries, label=method_name, linewidth=2, marker='o')
        ax.fill_between(mean_eval_episodes, mean_queries - std_queries, mean_queries + std_queries, alpha=0.2)
    
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Avg Queries per Episode")
    ax.set_title("Query Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Performance Comparison
    ax = axes[1, 0]
    final_success = []
    final_queries = []
    labels = []
    
    for method_name, histories in all_histories.items():
        last_success = [h["eval_success_rate"][-1] for h in histories]
        last_queries = [h["eval_avg_queries"][-1] for h in histories]
        
        final_success.append(last_success)
        final_queries.append(last_queries)
        labels.append(method_name)
    
    # Pareto plot: Success Rate vs. Query Cost
    ax.scatter(final_queries, final_success, s=200, alpha=0.6, label=labels)
    
    for i, (q, s) in enumerate(zip(final_queries, final_success)):
        for j, (x, y) in enumerate(zip(q, s)):
            ax.annotate(f'{x:.1f}/{y*100:.1f}', (x, y), fontsize=8)
    
    ax.set_xlabel("Avg Queries per Episode (Cost)")
    ax.set_ylabel("Success Rate (Benefit)")
    ax.set_title("Pareto Front: Query Cost vs. Success")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Reward Comparison
    ax = axes[1, 1]
    x_pos = np.arange(len(all_histories))
    width = 0.3
    
    for i, (method_name, histories) in enumerate(all_histories.items()):
        all_rewards = [h["eval_avg_reward"] for h in histories]
        mean_reward = np.mean(all_rewards, axis=0)
        std_reward = np.std(all_rewards, axis=0)
        
        ax.bar(x_pos[i], mean_reward[-1], width, yerr=std_reward[-1], 
               label=method_name, capsize=5)
    
    ax.set_ylabel("Avg Reward (Final)")
    ax.set_title("Final Episode Reward Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(all_histories.keys()))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_comparison_{method}.png", dpi=150)
    plt.close()


def run_training(method: str, seeds: List[int], episodes: int, output_dir: str = "."):
    """Run complete training experiment."""
    
    print("=" * 60)
    print(f"Training Framework - ChronoEnv v2.0")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Seeds: {seeds}")
    print(f"Episodes: {episodes}")
    print()
    
    # Create environment
    env = ChronoEnvTimeCritical(max_steps=150, noise_std=20.0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print()
    
    all_histories = {}
    
    if method in ["ppo", "both"]:
        print("=" * 60)
        print("Running PPO Baseline")
        print("=" * 60)
        
        ppo_histories = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                use_prm=False,
            )
            
            history = train_episodes(
                env=env,
                agent=agent,
                n_episodes=episodes,
                seed=seed,
            )
            
            ppo_histories.append(history)
            agent.save(f"{output_dir}/models/ppo_seed_{seed}.pt")
        
        all_histories["PPO"] = ppo_histories
    
    if method in ["prm", "both"]:
        print("\n" + "=" * 60)
        print("Running PPO+PRM")
        print("=" * 60)
        
        prm_histories = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                use_prm=True,
                prm_config={
                    "redundancy_penalty": 0.5,
                    "critical_reward": 0.5,
                    "blind_penalty": 1.0,
                    "precision_reward": 1.0,
                    "critical_buffer_threshold": 20.0,
                    "uncertainty_threshold": 30.0,
                },
            )
            
            history = train_episodes(
                env=env,
                agent=agent,
                n_episodes=episodes,
                seed=seed,
            )
            
            prm_histories.append(history)
            agent.save(f"{output_dir}/models/prm_seed_{seed}.pt")
        
        all_histories["PPO+PRM"] = prm_histories
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    
    plot_results(all_histories, method, output_dir)
    
    # Save results
    results = {
        "method": method,
        "seeds": seeds,
        "episodes": episodes,
        "histories": all_histories,
    }
    
    with open(f"{output_dir}/results_{method}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_dir}/results_{method}.json")
    print(f"Plot saved to: {output_dir}/training_comparison_{method}.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for method_name, histories in all_histories.items():
        final_success = [h["eval_success_rate"][-1] for h in histories]
        final_queries = [h["eval_avg_queries"][-1] for h in histories]
        
        print(f"\n{method_name}:")
        print(f"  Success Rate: {np.mean(final_success)*100:.1f}% ± {np.std(final_success)*100:.1f}%")
        print(f"  Avg Queries: {np.mean(final_queries):.2f} ± {np.std(final_queries):.2f}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="both", choices=["ppo", "prm", "both"])
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seed values")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default=".")
    
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    
    run_training(
        method=args.method,
        seeds=seeds,
        episodes=args.episodes,
        output_dir=args.output_dir,
    )
