"""
PPO Agent with Temporal Regret for ChronoEnv

Combines PPO policy with PRM (Process Reward Modeling) for active temporal grounding.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Optional

from chrono_env import ChronoEnv, ChronoEnvWithPRM
from prm_temporal import TemporalRegretModule


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
        
        # Action probabilities
        self.action_dim = action_dim
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns action probabilities and value."""
        x = self.encoder(state)
        
        # Actor: action logits
        action_logits = self.actor(x)
        
        # Critic: state value
        value = self.critic(x)
        
        return action_logits, value
    
    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """Sample action and return log probability."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value = self.forward(state_tensor)
        
        # Sample action
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


class PPOAgent:
    """PPO Agent with Temporal Regret support."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        temporal_weight: float = 0.3,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PPOPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.temporal_weight = temporal_weight
        
        # Temporal regret module
        self.temporal_regret = TemporalRegretModule(uncertainty_threshold=30.0)
        
        # Tracking
        self.last_action_was_check = False
        self.time_since_check = 0
    
    def get_action(self, state: np.ndarray) -> Tuple[int, Dict]:
        """Get action with metadata."""
        action, log_prob, value = self.policy.act(state)
        
        return action, {
            "log_prob": log_prob,
            "value": value,
            "is_check": (action == 2),  # CHECK_TIME = 2
        }
    
    def compute_returns(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> List[float]:
        """Compute GAE returns."""
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
    
    def compute_temporal_regret(self, trajectory: Dict) -> float:
        """Compute temporal regret for a trajectory."""
        return self.temporal_regret.compute_trajectory_regret(
            uncertainties=trajectory.get("uncertainties", []),
            actions=trajectory.get("actions", []),
            time_since_checks=trajectory.get("time_since_checks", []),
        )
    
    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        log_probs: List[float],
        values: List[float],
        dones: List[bool],
        trajectory: Dict,
    ) -> Dict[str, float]:
        """Update PPO policy with temporal regret."""
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        returns = torch.FloatTensor(self.compute_returns(rewards, values, dones)).to(self.device)
        advantages = returns - torch.FloatTensor(values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute temporal regret
        temporal_regret = self.compute_temporal_regret(trajectory)
        
        # PPO update
        self.optimizer.zero_grad()
        
        log_probs, values, entropy = self.policy.evaluate(states, actions)
        
        # Ratio for clipped PPO
        ratios = torch.exp(log_probs - old_log_probs)
        
        # Standard PPO loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss (include temporal regret as regularization)
        total_loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss + self.temporal_weight * temporal_regret
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "temporal_regret": temporal_regret,
            "total_loss": total_loss.item(),
        }
    
    def save(self, path: str):
        """Save policy."""
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path: str):
        """Load policy."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


def train_ppo(
    env: ChronoEnv,
    agent: PPOAgent,
    total_timesteps: int = 100000,
    n_steps: int = 2048,
    print_interval: int = 1000,
) -> Dict:
    """Train PPO agent on ChronoEnv."""
    
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
    
    # Trajectory tracking
    trajectory = {
        "uncertainties": [],
        "actions": [],
        "time_since_checks": [],
    }
    
    # Results
    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "query_counts": [],
        "success_rates": [],
    }
    
    best_success_rate = 0
    episodes_completed = 0
    
    for timestep in range(total_timesteps):
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
        
        # Track trajectory
        trajectory["uncertainties"].append(info.get("uncertainty", 0))
        trajectory["actions"].append(action)
        trajectory["time_since_checks"].append(episode_steps - info.get("last_check_step", 0))
        
        # Update state
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        # Check if done
        if terminated or truncated:
            episodes_completed += 1
            
            # Record metrics
            results["episode_rewards"].append(episode_reward)
            results["episode_lengths"].append(episode_steps)
            results["query_counts"].append(info.get("query_count", 0))
            results["success_rates"].append(
                1 if episode_reward > 0 else 0
            )
            
            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            trajectory = {
                "uncertainties": [],
                "actions": [],
                "time_since_checks": [],
            }
        
        # Update policy
        if (timestep + 1) % n_steps == 0 or (timestep + 1) == total_timesteps:
            if len(states) > 0:
                training_info = agent.update(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    log_probs=log_probs,
                    values=values,
                    dones=dones,
                    trajectory=trajectory,
                )
                
                # Reset storage
                states = []
                actions = []
                rewards = []
                log_probs = []
                values = []
                dones = []
                trajectory = {
                    "uncertainties": [],
                    "actions": [],
                    "time_since_checks": [],
                }
        
        # Print progress
        if (timestep + 1) % print_interval == 0 and results["episode_rewards"]:
            avg_reward = np.mean(results["episode_rewards"][-10:])
            avg_queries = np.mean(results["query_counts"][-10:])
            success_rate = np.mean(results["success_rates"][-10:])
            
            print(f"Timestep {timestep+1}: "
                  f"avg_reward={avg_reward:.2f}, "
                  f"avg_queries={avg_queries:.2f}, "
                  f"success_rate={success_rate*100:.1f}%")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                agent.save(f"/content/best_model_{success_rate:.3f}.pt")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PPO + Temporal Regret on ChronoEnv")
    print("=" * 60)
    
    # Create environment
    env = ChronoEnv(max_steps=100, noise_std=10.0)
    print(f"State dim: {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.n}")
    
    # Create agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4,
        temporal_weight=0.3,
    )
    
    # Train
    print("\nTraining...")
    results = train_ppo(
        env=env,
        agent=agent,
        total_timesteps=50000,
        n_steps=128,
        print_interval=5000,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Summary
    print(f"\nFinal Results:")
    print(f"  Avg Reward: {np.mean(results['episode_rewards']):.2f}")
    print(f"  Avg Queries: {np.mean(results['query_counts']):.2f}")
    print(f"  Success Rate: {np.mean(results['success_rates'])*100:.1f}%")
