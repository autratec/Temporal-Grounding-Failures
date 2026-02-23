"""
Temporal Debugger - Visualize Agent Decision Sequences

This tool helps analyze why agents fail in ChronoEnv by visualizing:
1. Time estimation vs ground truth over episodes
2. Query patterns
3. Submission timing
4. Work progress trajectories
"""

import sys
sys.path.insert(0, '/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from chrono_env_v2 import ChronoEnvTimeCritical
import json


def run_trajectory(env, agent, seed: int = 0):
    """Run single episode and return full trajectory."""
    np.random.seed(seed)
    
    state, _ = env.reset()
    
    trajectory = {
        'true_times': [0.0],
        'internal_estimates': [state[0]],
        'work_done': [0.0],
        'true_work_required': [env._true_work_required],
        'queries': [0],
        'actions': [],
        'rewards': [],
        'uncertainties': [0.0],
    }
    
    terminated = False
    episode_reward = 0
    
    while not terminated:
        action, metadata = agent.get_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        trajectory['true_times'].append(info['true_time'])
        trajectory['internal_estimates'].append(info['internal_estimate'])
        trajectory['work_done'].append(info['work_done'])
        trajectory['queries'].append(info['query_count'])
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['uncertainties'].append(info['uncertainty'])
        
        state = next_state
        episode_reward += reward
    
    trajectory['episode_reward'] = episode_reward
    trajectory['deadline'] = env._deadline
    
    return trajectory


def visualize_trajectory(trajectory, query_cost: float, seed: int, save_path: str):
    """Create comprehensive visualization of a single episode."""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    t = len(trajectory['true_times'])
    
    # Plot 1: Time Estimation vs Ground Truth
    ax = axes[0, 0]
    ax.plot(range(t), trajectory['true_times'], 'b-', linewidth=2, label='Ground Truth')
    ax.plot(range(t), trajectory['internal_estimates'], 'r--', linewidth=2, label='Internal Estimate')
    ax.axhline(y=trajectory['deadline'], color='g', linestyle=':', linewidth=2, label='Deadline')
    ax.set_xlabel('Steps', fontsize=10)
    ax.set_ylabel('Time (minutes)', fontsize=10)
    ax.set_title(f'Time Estimation (Seed {seed}, Reward={trajectory["episode_reward"]:.1f})', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Work Progress
    ax = axes[0, 1]
    ax.plot(range(t), trajectory['work_done'], 'b-', linewidth=2, label='Work Done')
    ax.axhline(y=trajectory['true_work_required'][0], color='r', linestyle='--', 
               linewidth=2, label=f'Target ({trajectory["true_work_required"][0]})')
    ax.set_xlabel('Steps', fontsize=10)
    ax.set_ylabel('Work Progress', fontsize=10)
    ax.set_title('Work Progress Over Time', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Query Pattern
    ax = axes[1, 0]
    query_times = [i for i, q in enumerate(trajectory['queries']) if q > 0 and i > 0]
    ax.vlines(query_times, 0, 1, colors='r', alpha=0.7, linewidth=2, label='CHECK_TIME')
    ax.set_xlabel('Steps', fontsize=10)
    ax.set_yticks([])
    ax.set_title(f'Query Timing (Total: {trajectory["queries"][-1]} queries)', fontsize=12)
    ax.legend(fontsize=8)
    
    # Plot 4: Action Distribution
    ax = axes[1, 1]
    actions = ['WORK', 'WAIT', 'CHECK_TIME', 'SUBMIT']
    action_counts = [trajectory['actions'].count(i) for i in range(4)]
    ax.bar(actions, action_counts, color=['blue', 'orange', 'green', 'red'])
    ax.set_xlabel('Actions', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Action Distribution', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Uncertainty Over Time
    ax = axes[2, 0]
    ax.plot(range(t), trajectory['uncertainties'], 'purple', linewidth=2)
    ax.axhline(y=20, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Steps', fontsize=10)
    ax.set_ylabel('Uncertainty (minutes)', fontsize=10)
    ax.set_title('Time Uncertainty Over Time', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Reward Accumulation
    ax = axes[2, 1]
    cumsum = np.cumsum(trajectory['rewards'])
    ax.plot(range(len(cumsum)), cumsum, 'b-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle=':', linewidth=2)
    ax.set_xlabel('Steps', fontsize=10)
    ax.set_ylabel('Cumulative Reward', fontsize=10)
    ax.set_title('Reward Accumulation', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("Temporal Debugger - Analyzing Agent Behavior")
    print("=" * 60)
    
    # Load trained models
    ppo_model = '/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code/models/ppo_seed_0.pt'
    prm_model = '/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code/models/prm_seed_0.pt'
    
    # Create environment
    env = ChronoEnvTimeCritical(max_steps=150, noise_std=20.0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load PPO agent
    ppo_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, temporal_weight=0.0)
    ppo_agent.policy.load_state_dict(torch.load(ppo_model))
    
    # Load PPO+PRM agent
    prm_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, temporal_weight=0.3)
    prm_agent.policy.load_state_dict(torch.load(prm_model))
    
    # Run trajectories for both agents
    seeds = [0, 1, 2, 3, 4]  # 5 different tasks
    
    print("\nRunning PPO agent trajectories...")
    ppo_trajectories = []
    for seed in seeds:
        print(f"  Seed {seed}...")
        traj = run_trajectory(env, ppo_agent, seed=seed)
        ppo_trajectories.append(traj)
        
        # Visualize each trajectory
        visualize_trajectory(
            traj, 
            query_cost=-1.0, 
            seed=seed, 
            save_path=f'/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/figures/ppo_traj_seed_{seed}.png'
        )
    
    print("\nRunning PPO+PRM agent trajectories...")
    prm_trajectories = []
    for seed in seeds:
        print(f"  Seed {seed}...")
        traj = run_trajectory(env, prm_agent, seed=seed)
        prm_trajectories.append(traj)
        
        # Visualize each trajectory
        visualize_trajectory(
            traj, 
            query_cost=-1.0, 
            seed=seed, 
            save_path=f'/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/figures/prm_traj_seed_{seed}.png'
        )
    
    # Generate summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    # PPO stats
    ppo_rewards = [t['episode_reward'] for t in ppo_trajectories]
    ppo_queries = [t['queries'][-1] for t in ppo_trajectories]
    ppo_success = [1 if r > 0 else 0 for r in ppo_rewards]
    
    print("\nPPO Agent:")
    print(f"  Avg Reward: {np.mean(ppo_rewards):.2f}")
    print(f"  Avg Queries: {np.mean(ppo_queries):.2f}")
    print(f"  Success Rate: {np.mean(ppo_success)*100:.1f}%")
    
    # PRM stats
    prm_rewards = [t['episode_reward'] for t in prm_trajectories]
    prm_queries = [t['queries'][-1] for t in prm_trajectories]
    prm_success = [1 if r > 0 else 0 for r in prm_rewards]
    
    print("\nPPO+PRM Agent:")
    print(f"  Avg Reward: {np.mean(prm_rewards):.2f}")
    print(f"  Avg Queries: {np.mean(prm_queries):.2f}")
    print(f"  Success Rate: {np.mean(prm_success)*100:.1f}%")
    
    # Save statistics
    stats = {
        'ppo': {
            'rewards': ppo_rewards,
            'queries': ppo_queries,
            'success': ppo_success,
        },
        'prm': {
            'rewards': prm_rewards,
            'queries': prm_queries,
            'success': prm_success,
        },
    }
    
    with open('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/figures/debugger_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nSaved: figures/debugger_stats.json")
    
    # Generate comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Reward comparison
    ax = axes[0]
    ax.boxplot([ppo_rewards, prm_rewards], tick_labels=['PPO', 'PPO+PRM'])
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Distribution')
    ax.grid(True, alpha=0.3)
    
    # Query comparison
    ax = axes[1]
    ax.boxplot([ppo_queries, prm_queries], tick_labels=['PPO', 'PPO+PRM'])
    ax.set_ylabel('Total Queries')
    ax.set_title('Query Distribution')
    ax.grid(True, alpha=0.3)
    
    # Success comparison
    ax = axes[2]
    ax.bar(['PPO', 'PPO+PRM'], [np.mean(ppo_success), np.mean(prm_success)])
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/figures/debugger_comparison.png', dpi=150)
    print("Saved: figures/debugger_comparison.png")
    
    print("\n" + "=" * 60)
    print("Debugging Complete!")
    print("=" * 60)
    print("\nGenerated visualizations:")
    print("  PPO trajectories: figures/ppo_traj_seed_{0-4}.png")
    print("  PRM trajectories: figures/prm_traj_seed_{0-4}.png")
    print("  Comparison: figures/debugger_comparison.png")
    print("  Statistics: figures/debugger_stats.json")
    print("\nUse these to analyze:")
    print("  - How agents estimate time")
    print("  - When they choose to query")
    print("  - Why they fail (submit too early/late)")
    print("  - How PRM changes behavior")


if __name__ == "__main__":
    import torch
    main()
