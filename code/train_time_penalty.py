"""
PPO Training on ChronoEnvTimePenalties - 500 Episodes

Tests active temporal grounding with time-based penalties.
Agent must learn when to check time to avoid early/late penalties.
"""

import sys
sys.path.insert(0, '/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code')

import numpy as np
import torch
from chrono_env_time_penalty import ChronoEnvTimePenalties
from ppo_agent import PPOAgent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("PPO Training on ChronoEnvTimePenalties - 500 Episodes")
    print("=" * 60)
    
    # Create environment with time penalties
    env = ChronoEnvTimePenalties(max_steps=100, noise_std=15.0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Time penalties: early=-5.0, late=-20.0")
    print(f"Success reward: +20.0, Good estimation bonus: +2.0")
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        temporal_weight=0.3,
    )
    
    # Train with custom loop for 500 episodes
    print("\nTraining for 500 episodes...")
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    query_counts = []
    success_rates = []
    avg_time_diffs = []
    
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    # Storage for training
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
    
    episode_count = 0
    step_count = 0
    
    while episode_count < 500:
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
        step_count += 1
        
        # Update policy every 64 steps
        if step_count % 64 == 0 and len(states) > 0:
            training_info = agent.update(
                states=states,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                values=values,
                dones=dones,
                trajectory=trajectory,
            )
            
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
        
        # Check if done
        if terminated or truncated:
            episode_count += 1
            
            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            query_counts.append(info.get("query_count", 0))
            avg_time_diffs.append(abs(info.get("time_diff", 0)))
            success_rates.append(1 if episode_reward >= 15 else 0)  # Reward >= 15 means success
            
            # Print progress
            if episode_count % 50 == 0 or episode_count == 1:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_queries = np.mean(query_counts[-50:])
                success_rate = np.mean(success_rates[-50:])
                avg_time_diff = np.mean(avg_time_diffs[-50:])
                
                print(f"Episode {episode_count:3d}: "
                      f"reward={avg_reward:7.2f}, "
                      f"queries={avg_queries:.2f}, "
                      f"success_rate={success_rate*100:.1f}%, "
                      f"avg_time_diff={avg_time_diff:.1f} min")
            
            # Save intermediate checkpoint every 100 episodes
            if episode_count % 100 == 0 and episode_count > 0:
                agent.save(f"/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code/models/ppo_time_penalty_checkpoint_{episode_count}.pt")
            
            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            trajectory = {
                "uncertainties": [],
                "actions": [],
                "time_since_checks": [],
            }
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Final statistics
    print(f"\nFinal Results (500 episodes):")
    print(f"  Avg Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Avg Episodes: {len(episode_rewards)}")
    print(f"  Avg Queries: {np.mean(query_counts):.2f}")
    print(f"  Success Rate: {np.mean(success_rates)*100:.1f}%")
    
    # Detailed stats
    print(f"\nDetailed Stats:")
    print(f"  Min Reward: {np.min(episode_rewards):.2f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")
    print(f"  Reward Std: {np.std(episode_rewards):.2f}")
    print(f"  Avg Length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Avg Time Diff: {np.mean(avg_time_diffs):.1f} min")
    
    # Save model
    agent.save("/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code/models/ppo_trained_500ep_time_penalty.pt")
    print("\nModel saved to: models/ppo_trained_500ep_time_penalty.pt")
    
    # Save full episode data for analysis
    np.savez('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code/full_episodes_500_time_penalty.npz',
             rewards=episode_rewards,
             lengths=episode_lengths,
             queries=query_counts,
             success=success_rates,
             time_diffs=avg_time_diffs)
    print("Episode data saved to: full_episodes_500_time_penalty.npz")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards (Time Penalties)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Query counts
    axes[0, 1].plot(query_counts)
    axes[0, 1].set_title('Query Counts (Time Penalties)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Queries')
    
    # Success rate (rolling)
    success_array = np.array(success_rates)
    rolling_success = np.convolve(success_array, np.ones(50)/50, mode='valid')
    axes[1, 0].plot(rolling_success)
    axes[1, 0].set_title('Success Rate (Rolling 50)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    
    # Distribution of queries
    axes[1, 1].hist(query_counts, bins=20, edgecolor='black')
    axes[1, 1].set_title('Query Distribution')
    axes[1, 1].set_xlabel('Queries')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code/ppo_results_500ep_time_penalty.png', dpi=150)
    print("Plot saved to: ppo_results_500ep_time_penalty.png")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
