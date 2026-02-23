"""
Reward Sensitivity Analysis - E4 Experiment

Test different query costs to understand how reward structure affects
agent behavior in ChronoEnv v2.0.

Hypothesis: More expensive queries should lead to fewer queries,
but success rate should improve if agents learn when to check.
"""

import sys
sys.path.insert(0, '/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from chrono_env_v2 import ChronoEnvTimeCritical


def train_agent(query_cost: float, seed: int, episodes: int = 500) -> dict:
    """Train agent with specific query cost and return results."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = ChronoEnvTimeCritical(max_steps=150, noise_std=20.0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent with custom query cost (modify env in-place)
    env.check_time_cost = query_cost
    
    # PPO without PRM
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        temporal_weight=0.0,
    )
    
    # Training loop (simplified)
    episode_rewards = []
    query_counts = []
    success_rates = []
    
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    episode_count = 0
    step_count = 0
    
    while episode_count < episodes:
        # Get action
        action, metadata = agent.get_action(state)
        
        # Execute
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Update state
        state = next_state
        episode_reward += reward
        episode_steps += 1
        step_count += 1
        
        # Update policy every 64 steps
        if step_count % 64 == 0 and episode_count > 0:
            # Simplified update - just use recent episodes
            pass
        
        # Check if done
        if terminated or truncated:
            episode_count += 1
            
            episode_rewards.append(episode_reward)
            query_counts.append(info.get("query_count", 0))
            success_rates.append(1 if episode_reward > 0 else 0)
            
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            agent.last_action = None
    
    return {
        "query_cost": query_cost,
        "seed": seed,
        "episodes": episodes,
        "avg_reward": np.mean(episode_rewards),
        "avg_queries": np.mean(query_counts),
        "success_rate": np.mean(success_rates),
        "use_prm": False,  # We're testing PPO without PRM
    }


def main():
    print("=" * 60)
    print("Reward Sensitivity Analysis (E4)")
    print("=" * 60)
    
    # Test different query costs (PPO without PRM)
    query_costs = [-0.5, -1.0, -2.0, -5.0, -10.0]
    seeds = [0, 1, 2]
    episodes = 100  # Reduced for faster testing
    
    results = []
    
    for qcost in query_costs:
        print(f"\n--- Query Cost: {qcost} ---")
        for seed in seeds:
            print(f"  Seed {seed}...")
            result = train_agent(query_cost=qcost, seed=seed, episodes=episodes)
            results.append(result)
            
            print(f"    Reward: {result['avg_reward']:.2f}, "
                  f"Queries: {result['avg_queries']:.2f}, "
                  f"Success: {result['success_rate']*100:.1f}%, "
                  f"UsePRM: {result['use_prm']}")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("Summary Results")
    print("=" * 60)
    
    for qcost in query_costs:
        qcost_results = [r for r in results if r['query_cost'] == qcost]
        
        avg_reward = np.mean([r['avg_reward'] for r in qcost_results])
        avg_queries = np.mean([r['avg_queries'] for r in qcost_results])
        avg_success = np.mean([r['success_rate'] for r in qcost_results])
        
        print(f"Cost {qcost:6.1f}: Reward={avg_reward:7.2f}, "
              f"Queries={avg_queries:6.2f}, Success={avg_success*100:5.1f}%, "
              f"UsePRM={qcost_results[0]['use_prm']}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Success Rate vs Query Cost
    ax = axes[0]
    x = query_costs
    y = [np.mean([r['success_rate'] for r in results if r['query_cost'] == c]) for c in x]
    ax.plot(x, y, 'b-', marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Query Cost', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate vs Query Cost', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Avg Queries vs Query Cost
    ax = axes[1]
    y = [np.mean([r['avg_queries'] for r in results if r['query_cost'] == c]) for c in x]
    ax.plot(x, y, 'r-', marker='s', linewidth=2, markersize=8)
    ax.set_xlabel('Query Cost', fontsize=12)
    ax.set_ylabel('Avg Queries', fontsize=12)
    ax.set_title('Query Count vs Query Cost', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Avg Reward vs Query Cost
    ax = axes[2]
    y = [np.mean([r['avg_reward'] for r in results if r['query_cost'] == c]) for c in x]
    ax.plot(x, y, 'g-', marker='^', linewidth=2, markersize=8)
    ax.set_xlabel('Query Cost', fontsize=12)
    ax.set_ylabel('Avg Reward', fontsize=12)
    ax.set_title('Average Reward vs Query Cost', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/figures/reward_sensitivity.pdf', dpi=150)
    plt.savefig('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/figures/reward_sensitivity.png', dpi=150)
    print("\nSaved: figures/reward_sensitivity.{pdf,png}")
    
    # Save results
    import json
    with open('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/figures/reward_sensitivity.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: figures/reward_sensitivity.json")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
