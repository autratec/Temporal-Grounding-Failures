"""
ChronoEnv: Active Temporal Grounding Benchmark

A text-based task scheduling environment for testing active time query strategies.
Implements POMDP-style time perception where agent must learn when to check time.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional


class ChronoEnv(gym.Env):
    """
    ChronoEnv: A time-constrained task scheduling environment.
    
    The agent must complete tasks before deadlines, but time information
    is partially observable. The agent can:
    - WORK: Make progress toward deadline
    - WAIT: Wait (time passes, no progress)
    - CHECK_TIME: Query external clock (cost: -0.5 reward)
    - SUBMIT: Submit task (success if before deadline)
    
    State includes internal time estimate which drifts from ground truth.
    """
    
    def __init__(
        self,
        max_steps: int = 100,
        time_step_minutes: int = 5,
        noise_std: float = 10.0,  # minutes
        check_time_cost: float = -0.5,
        success_reward: float = 10.0,
        failure_reward: float = -10.0,
    ):
        super().__init__()
        
        # Configuration
        self.max_steps = max_steps
        self.time_step_minutes = time_step_minutes
        self.noise_std = noise_std
        self.check_time_cost = check_time_cost
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        
        # Track highest uncertainty seen (for PRM)
        self.highest_uncertainty = 0.0
        
        # Action space
        self.action_space = spaces.Discrete(4)  # WORK, WAIT, CHECK_TIME, SUBMIT
        
        # Observation space
        # [internal_estimate, deadline, step_count, last_check_step, query_count]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0, 0, 0]),
            high=np.array([1000.0, 1000.0, max_steps, max_steps, max_steps]),
            dtype=np.float32
        )
        
        # Reset
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Ground truth time (in minutes from start)
        self.true_time = 0.0
        
        # Agent's internal estimate (starts with some initial error)
        self.internal_estimate = 0.0 + self.np_random.uniform(-10, 10)
        
        # Task parameters
        self.deadline = self.np_random.uniform(200, 600)  # 3:20 to 10:00 in minutes
        self.progress = 0.0
        
        # Tracking
        self.current_step = 0
        self.last_check_step = 0
        self.query_count = 0
        self.episode_reward = 0
        self.last_check_time = 0.0
        
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            self.internal_estimate,
            self.deadline,
            self.current_step,
            self.current_step - self.last_check_step,
            self.query_count,
        ], dtype=np.float32)
    
    def _step_time(self):
        """Advance time by one step."""
        self.true_time += self.time_step_minutes
        self.internal_estimate += self.time_step_minutes
        
        # Add noise to internal estimate (simulates time perception error)
        noise = self.np_random.normal(0, self.noise_std)
        self.internal_estimate += noise
        
        self.current_step += 1
    
    def _check_time(self) -> float:
        """Query external clock and update internal estimate."""
        self.query_count += 1
        self.last_check_step = self.current_step
        self.last_check_time = self.true_time
        
        # Update internal estimate to ground truth
        self.internal_estimate = self.true_time
        
        return self.check_time_cost
    
    def _calculate_uncertainty(self) -> float:
        """Calculate current time uncertainty."""
        # Uncertainty increases with time since last check
        time_since_check = self.current_step - self.last_check_step
        base_uncertainty = time_since_check * self.time_step_minutes * 0.1
        
        # Add noise component
        noise_uncertainty = self.noise_std * np.sqrt(time_since_check)
        
        return base_uncertainty + noise_uncertainty
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return observation, reward, done, info."""
        reward = 0.0
        terminated = False
        truncated = False
        
        # Execute action
        if action == 0:  # WORK
            # Progress toward deadline (more progress closer to deadline)
            time_remaining = self.deadline - self.true_time
            if time_remaining > 0:
                # More progress possible when more time remains
                progress_step = 5 + (time_remaining / self.deadline) * 10
                self.progress = min(self.progress + progress_step, self.deadline)
            reward = 1.0  # Reward for making progress
        
        elif action == 1:  # WAIT
            reward = 0.0  # No penalty for waiting (encourage learning first)
        
        elif action == 2:  # CHECK_TIME
            # Query external clock and update internal estimate
            self.query_count += 1
            self.last_check_step = self.current_step
            self.last_check_time = self.true_time
            self.internal_estimate = self.true_time
            reward = 0.0  # No cost for checking (encourage learning time estimation)
        
        elif action == 3:  # SUBMIT
            # Check deadline BEFORE advancing time
            # Reward if progress is significant (at least 80% complete)
            if self.progress >= 0.8 * self.deadline:
                if self.true_time <= self.deadline:
                    reward = 20.0  # Large reward for successful submission
                else:
                    reward = -20.0  # Large penalty for missing deadline
            else:
                # Not ready to submit yet - small penalty
                reward = -1.0
            terminated = True
            self.progress = self.deadline
            
            # Create info dict before returning
            self.episode_reward += reward
            info = {
                "true_time": self.true_time,
                "internal_estimate": self.internal_estimate,
                "uncertainty": self._calculate_uncertainty(),
                "progress": self.progress,
                "deadline": self.deadline,
                "query_count": self.query_count,
                "total_reward": self.episode_reward,
            }
            # Don't advance time on submit
            return self._get_obs(), reward, terminated, truncated, info
        
        else:
            reward = -1.0  # Invalid action penalty
        
        # Advance time
        self._step_time()
        
        # Check if max steps reached (but only if not already terminated by SUBMIT)
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            # Final submission attempt
            if self.true_time <= self.deadline:
                reward = self.success_reward
            else:
                reward = self.failure_reward
            terminated = True
        
        self.episode_reward += reward
        
        info = {
            "true_time": self.true_time,
            "internal_estimate": self.internal_estimate,
            "uncertainty": self._calculate_uncertainty(),
            "progress": self.progress,
            "deadline": self.deadline,
            "query_count": self.query_count,
            "total_reward": self.episode_reward,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        """Render environment state."""
        print(f"True time: {self.true_time:.1f} min")
        print(f"Internal estimate: {self.internal_estimate:.1f} min")
        print(f"Deadline: {self.deadline:.1f} min")
        print(f"Progress: {self.progress:.1f} / {self.deadline:.1f}")
        print(f"Steps: {self.current_step} / {self.max_steps}")
        print(f"Query count: {self.query_count}")
        print(f"Uncertainty: {self._calculate_uncertainty():.1f} min")
        print("-" * 50)


class ChronoEnvWithPRM(ChronoEnv):
    """
    ChronoEnv with PRM (Process Reward Modeling) for temporal regret.
    
    PRM rewards/penalizes based on time decision quality:
    - Penalize: High uncertainty + no check
    - Penalize: Low uncertainty + redundant check
    """
    
    def __init__(
        self,
        prm_high_uncertainty_weight: float = 0.5,
        prm_redundant_check_weight: float = 0.3,
        uncertainty_threshold: float = 30.0,  # minutes
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prm_high_uncertainty_weight = prm_high_uncertainty_weight
        self.prm_redundant_check_weight = prm_redundant_check_weight
        self.uncertainty_threshold = uncertainty_threshold
        self.last_action_was_check = False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with PRM reward shaping."""
        # Get base reward
        obs, base_reward, terminated, truncated, info = super().step(action)
        
        # Calculate PRM reward
        prm_reward = 0.0
        
        if action == 2:  # CHECK_TIME
            # Penalize redundant checks (low uncertainty, just checked)
            if self.last_action_was_check:
                uncertainty = info.get("uncertainty", 0)
                if uncertainty < self.uncertainty_threshold:
                    prm_reward = -self.prm_redundant_check_weight
            self.last_action_was_check = True
        else:
            self.last_action_was_check = False
        
        # High uncertainty without checking
        if action not in [2, 3]:  # WORK or WAIT
            uncertainty = info.get("uncertainty", 0)
            if uncertainty > self.uncertainty_threshold:
                # Should have checked!
                prm_reward -= self.prm_high_uncertainty_weight
        
        total_reward = base_reward + prm_reward
        
        info["prm_reward"] = prm_reward
        info["total_reward"] = self.episode_reward + total_reward
        
        return obs, total_reward, terminated, truncated, info


class ChronoEnvHighUncertainty(ChronoEnv):
    """
    ChronoEnv with high time uncertainty to test active temporal grounding.
    
    Higher noise_std makes internal time estimate less reliable,
    forcing the agent to learn when to check time.
    """
    
    def __init__(self, **kwargs):
        # Override noise_std to be higher
        kwargs['noise_std'] = kwargs.get('noise_std', 30.0)  # 3x higher
        super().__init__(**kwargs)


class ChronoEnvWithPRMHighUncertainty(ChronoEnvWithPRM):
    """
    ChronoEnv with PRM and high uncertainty.
    """
    
    def __init__(self, **kwargs):
        # Override noise_std to be higher
        kwargs['noise_std'] = kwargs.get('noise_std', 30.0)  # 3x higher
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Test basic environment
    env = ChronoEnv()
    obs, info = env.reset()
    
    print("=" * 60)
    print("ChronoEnv Test")
    print("=" * 60)
    
    for step in range(10):
        print(f"\nStep {step}:")
        env.render()
        
        # Random actions
        action = env.action_space.sample()
        print(f"Action: {action} ({['WORK', 'WAIT', 'CHECK_TIME', 'SUBMIT'][action]})")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}, Total: {info['total_reward']:.3f}")
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    # Test PRM version
    print("\n" + "=" * 60)
    print("ChronoEnvWithPRM Test")
    print("=" * 60)
    
    env_prm = ChronoEnvWithPRM()
    obs, info = env_prm.reset()
    
    for step in range(10):
        action = env_prm.action_space.sample()
        obs, reward, terminated, truncated, info = env_prm.step(action)
        
        print(f"Step {step}: action={action}, reward={reward:.3f}, "
              f"prm={info.get('prm_reward', 0):.3f}, total={info['total_reward']:.3f}")
        
        if terminated or truncated:
            break
    
    print("\nPRM Test Complete")
