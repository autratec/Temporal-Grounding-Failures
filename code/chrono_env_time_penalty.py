"""
Enhanced ChronoEnv with Time-Based Penalties

To properly test active temporal grounding, we add:
1. Penalty for early submission (wasted effort)
2. Penalty for late submission (missed deadline)
3. Variable work speed (requires time estimation)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional


class ChronoEnvTimePenalties(gym.Env):
    """
    Enhanced ChronoEnv with time-based penalties for active temporal grounding.
    
    Key changes:
    1. Early submission penalty (wasted effort)
    2. Late submission penalty (missed deadline with extra cost)
    3. Variable work speed that depends on time estimation accuracy
    4. Reward for good time estimation (PRM signal)
    """
    
    def __init__(
        self,
        max_steps: int = 100,
        time_step_minutes: int = 5,
        noise_std: float = 15.0,
        check_time_cost: float = -0.5,
        early_penalty: float = -5.0,  # Penalty for submitting too early
        late_penalty: float = -20.0,  # Penalty for missing deadline
        success_reward: float = 20.0,
        good_estimation_bonus: float = 2.0,
    ):
        super().__init__()
        
        # Configuration
        self.max_steps = max_steps
        self.time_step_minutes = time_step_minutes
        self.noise_std = noise_std
        self.check_time_cost = check_time_cost
        self.early_penalty = early_penalty
        self.late_penalty = late_penalty
        self.success_reward = success_reward
        self.good_estimation_bonus = good_estimation_bonus
        
        # Action space
        self.action_space = spaces.Discrete(4)  # WORK, WAIT, CHECK_TIME, SUBMIT
        
        # Observation space
        # [internal_estimate, deadline, step_count, time_since_check, query_count]
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
    
    def _calculate_uncertainty(self) -> float:
        """Calculate current time uncertainty."""
        # Uncertainty increases with time since last check
        time_since_check = self.current_step - self.last_check_step
        base_uncertainty = time_since_check * self.time_step_minutes * 0.1
        
        # Add noise component
        noise_uncertainty = self.noise_std * np.sqrt(time_since_check)
        
        return base_uncertainty + noise_uncertainty
    
    def _estimate_completion_time(self) -> float:
        """Estimate when work will be completed based on internal time."""
        # Estimate based on how much work is left and current progress
        work_remaining = self.deadline - self.progress
        
        # Estimate time to finish (simplified: constant speed)
        # In reality, this would depend on internal state
        return self.internal_estimate + (work_remaining / 2)  # 2 min progress per step estimate
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return observation, reward, done, info."""
        reward = 0.0
        terminated = False
        truncated = False
        
        # Execute action
        if action == 0:  # WORK
            # Progress toward deadline
            # Work speed depends on time estimation quality
            time_uncertainty = self._calculate_uncertainty()
            
            # If uncertainty is high, work speed is reduced (agent hesitates)
            if time_uncertainty > self.noise_std * 3:
                progress_step = 3  # Slow work under high uncertainty
            else:
                progress_step = 5  # Normal work speed
            
            time_remaining = self.deadline - self.true_time
            if time_remaining > 0:
                self.progress = min(self.progress + progress_step, self.deadline)
            
            reward = 0.5  # Small reward for making progress
        
        elif action == 1:  # WAIT
            reward = 0.0
        
        elif action == 2:  # CHECK_TIME
            # Query external clock and update internal estimate
            self.query_count += 1
            self.last_check_step = self.current_step
            self.internal_estimate = self.true_time
            reward = self.check_time_cost  # Small cost for checking time
        
        elif action == 3:  # SUBMIT
            time_at_submit = self.true_time
            
            # Calculate time difference from deadline
            time_diff = time_at_submit - self.deadline
            
            if time_diff <= 0:
                # On time or early
                if time_diff >= -30:  # Within 30 minutes of deadline (good timing)
                    # Excellent timing - bonus for good time estimation
                    reward = self.success_reward + self.good_estimation_bonus + 5.0
                elif time_diff >= -60:  # Within 1 hour of deadline
                    reward = self.success_reward + self.good_estimation_bonus
                else:
                    # Too early - significant waste
                    reward = self.success_reward + self.early_penalty - 2.0
            else:
                # Late - missed deadline
                reward = self.late_penalty
            
            terminated = True
        
        else:
            reward = -1.0  # Invalid action penalty
        
        # Advance time
        self._step_time()
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
            # Force submission
            time_at_submit = self.true_time
            time_diff = time_at_submit - self.deadline
            
            if time_diff <= 0:
                reward = self.success_reward + 5.0  # Bonus for on-time
            else:
                reward = self.late_penalty
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
            "time_diff": self.true_time - self.deadline,
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


if __name__ == "__main__":
    # Test enhanced environment
    env = ChronoEnvTimePenalties(max_steps=100, noise_std=15.0)
    obs, info = env.reset()
    
    print("=" * 60)
    print("ChronoEnvTimePenalties Test")
    print("=" * 60)
    print(f"Deadline: {env.deadline:.1f} min")
    
    # Test various strategies
    print("\nTest 1: Submit immediately (should get early penalty)")
    obs, reward, terminated, truncated, info = env.step(3)
    print(f"Reward: {reward:.1f}")
    
    env.reset()
    print(f"\nDeadline: {env.deadline:.1f} min")
    
    print("\nTest 2: Work 20 steps then submit (might be early)")
    for i in range(20):
        obs, reward, terminated, truncated, info = env.step(0)
    obs, reward, terminated, truncated, info = env.step(3)
    print(f"Reward: {reward:.1f}, time_diff: {info['time_diff']:.1f}")
    
    env.reset()
    print(f"\nDeadline: {env.deadline:.1f} min")
    
    print("\nTest 3: Check time, then work, then submit")
    obs, reward, terminated, truncated, info = env.step(2)  # CHECK_TIME
    print(f"After CHECK_TIME: reward={reward:.1f}, internal={info['internal_estimate']:.1f}")
    
    for i in range(40):
        obs, reward, terminated, truncated, info = env.step(0)
    obs, reward, terminated, truncated, info = env.step(3)
    print(f"After WORK+SUBMIT: reward={reward:.1f}, time_diff: {info['time_diff']:.1f}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
