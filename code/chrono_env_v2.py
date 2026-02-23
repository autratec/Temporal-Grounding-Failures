"""
ChronoEnv v2.0 - Time-Critical Regime

Designed to FORCE active temporal grounding by eliminating time-independent shortcuts.

Core Design Principles:
1. Random work duration (80-120 steps) - Agent doesn't know exact requirement
2. Tight deadline (work_duration + buffer=10) - Must estimate remaining time precisely
3. No step count in observation - Agent cannot rely on internal counting
4. Work speed = 1 step/step - Progress is slow and must be monitored
5. CHECK_TIME returns noisy observation - Simulates imperfect clock

Reward Structure:
- Work: -0.1 (time cost)
- Check_Time: -1.0 (query cost)
- Success: 100 - (time_elapsed * 0.5) - Reward decays with time
- Early submit (incomplete): -50
- Late submit (missed): -50
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional


class ChronoEnvTimeCritical(gym.Env):
    """
    ChronoEnv v2.0 - Time-Critical Regime
    
    Forces agents to actively manage time uncertainty.
    Fixed policies fail - agent must CHECK_TIME to succeed.
    """
    
    def __init__(
        self,
        max_steps: int = 150,  # Allow enough time for longest tasks
        time_step_minutes: int = 5,
        noise_std: float = 20.0,  # 20 minute noise in CHECK_TIME
        check_time_cost: float = -1.0,
        work_cost: float = -0.1,
        success_reward: float = 100.0,
        failure_penalty: float = -50.0,
        time_decay: float = 0.5,  # Reward decays by 0.5 per step
        min_work_required: int = 80,
        max_work_required: int = 120,
        deadline_buffer_min: int = 5,
        deadline_buffer_max: int = 15,
    ):
        super().__init__()
        
        # Configuration
        self.max_steps = max_steps
        self.time_step_minutes = time_step_minutes
        self.noise_std = noise_std
        self.check_time_cost = check_time_cost
        self.work_cost = work_cost
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.time_decay = time_decay
        self.min_work_required = min_work_required
        self.max_work_required = max_work_required
        self.deadline_buffer_min = deadline_buffer_min
        self.deadline_buffer_max = deadline_buffer_max
        
        # Action space
        self.action_space = spaces.Discrete(4)  # WORK, WAIT, CHECK_TIME, SUBMIT
        
        # Observation space (NO step_count - forces time awareness!)
        # [internal_estimate, deadline_estimate, time_since_check, query_count]
        # Note: Internal estimate is noisy, deadline is unknown
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0, 0]),
            high=np.array([1000.0, 1000.0, max_steps, max_steps]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment - task parameters are now hidden!"""
        super().reset(seed=seed)
        
        # Generate hidden task parameters (agent doesn't see these)
        self._true_work_required = int(self.np_random.uniform(
            self.min_work_required, self.max_work_required + 1
        ))
        self._deadline_buffer = int(self.np_random.uniform(
            self.deadline_buffer_min, self.deadline_buffer_max + 1
        ))
        self._deadline = self._true_work_required + self._deadline_buffer
        
        # Time tracking
        self._true_time = 0.0  # Ground truth time elapsed
        self._work_done = 0.0  # Work completed
        self._episode_reward = 0.0
        
        # Agent's internal state
        self._internal_estimate = 0.0
        self._time_since_check = 0
        self._query_count = 0
        self._current_step = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """
        Get observation WITHOUT step_count!
        
        This forces agent to rely on CHECK_TIME or internal estimation.
        """
        return np.array([
            self._internal_estimate,
            self._deadline,  # Known (not hidden in this version)
            self._time_since_check,
            self._query_count,
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in Time-Critical regime."""
        reward = 0.0
        terminated = False
        truncated = False
        
        if action == 0:  # WORK
            # Progress toward deadline
            self._work_done += 1
            self._internal_estimate += self.time_step_minutes
            self._time_since_check += 1
            self._true_time += self.time_step_minutes
            self._current_step += 1
            
            reward = self.work_cost  # Time cost for working
            
            # Check if work is done
            if self._work_done >= self._true_work_required:
                reward = self.success_reward - (self._true_time * self.time_decay)
                terminated = True
        
        elif action == 1:  # WAIT
            # Just waste time
            self._internal_estimate += self.time_step_minutes
            self._time_since_check += 1
            self._true_time += self.time_step_minutes
            self._current_step += 1
            
            reward = self.work_cost - 0.2  # Waiting is inefficient
        
        elif action == 2:  # CHECK_TIME
            # Query external clock with noise (simulates imperfect clock)
            observed_time = self._true_time + self.np_random.normal(0, self.noise_std)
            observed_time = max(0, observed_time)  # No negative time
            
            # Update agent's estimate
            self._internal_estimate = observed_time
            self._time_since_check = 0
            self._query_count += 1
            
            reward = self.check_time_cost
        
        elif action == 3:  # SUBMIT
            # Agent submits based on current estimate
            # Success requires BOTH work complete AND time before deadline
            
            # Check if work is done
            if self._work_done >= self._true_work_required:
                # Work complete - check if on time
                if self._true_time <= self._deadline:
                    # On time! Reward decays with time
                    reward = self.success_reward - (self._true_time * self.time_decay)
                else:
                    # Missed deadline
                    reward = self.failure_penalty
            else:
                # Work not done - early submission penalty
                reward = self.failure_penalty
            
            terminated = True
        
        else:
            reward = -1.0  # Invalid action penalty
        
        # Check max steps
        if self._current_step >= self.max_steps:
            truncated = True
            if not terminated:
                # Force submission at max steps
                if self._work_done >= self._true_work_required:
                    reward = self.success_reward - (self._true_time * self.time_decay)
                else:
                    reward = self.failure_penalty
                terminated = True
        
        self._episode_reward += reward
        
        info = {
            "true_time": self._true_time,
            "true_work_required": int(self._true_work_required),
            "deadline": self._deadline,
            "work_done": int(self._work_done),
            "internal_estimate": self._internal_estimate,
            "time_remaining": self._deadline - self._true_time,
            "uncertainty": self.noise_std * np.sqrt(self._time_since_check),
            "query_count": self._query_count,
            "total_reward": self._episode_reward,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        """Render environment state."""
        print(f"True time: {self._true_time:.1f} min")
        print(f"True work required: {self._true_work_required} steps")
        print(f"Deadline: {self._deadline:.1f} min")
        print(f"Work done: {self._work_done:.0f} / {self._true_work_required}")
        print(f"Internal estimate: {self._internal_estimate:.1f} min")
        print(f"Time since check: {self._time_since_check} steps")
        print(f"Query count: {self._query_count}")
        print(f"Uncertainty: {self.noise_std * np.sqrt(self._time_since_check):.1f} min")
        print("-" * 50)


if __name__ == "__main__":
    # Test ChronoEnv v2.0
    env = ChronoEnvTimeCritical(max_steps=150, noise_std=20.0)
    
    print("=" * 60)
    print("ChronoEnv v2.0 - Time-Critical Regime Test")
    print("=" * 60)
    
    # Test 1: Fixed 100-step strategy (will fail on 120-step tasks)
    print("\n" + "-" * 60)
    print("Test 1: Fixed 100-step strategy (should fail on hard tasks)")
    print("-" * 60)
    
    success_count = 0
    for i in range(20):
        obs, info = env.reset()
        work_req = env._true_work_required
        deadline = env._deadline
        print(f"Task {i+1}: work_required={work_req}, deadline={deadline}")
        total_reward = 0
        
        # Work exactly 100 steps
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(0)  # WORK
            total_reward += reward
            if terminated or truncated:
                break
        
        # Submit
        obs, reward, terminated, truncated, info = env.step(3)
        total_reward += reward
        
        success = info['work_done'] >= info['true_work_required']
        if success:
            success_count += 1
            print(f"Task {i+1}: SUCCESS (work={info['work_done']:.0f}, required={info['true_work_required']})")
        else:
            print(f"Task {i+1}: FAILED (work={info['work_done']:.0f}, required={info['true_work_required']}, deadline={info['deadline']})")
    
    print(f"\nSuccess rate: {success_count}/20 = {success_count/20*100:.1f}%")
    print("(Expected: ~50% due to random work requirements 80-120)")
    
    # Test 2: CHECK_TIME helps
    print("\n" + "-" * 60)
    print("Test 2: CHECK_TIME strategy (should succeed more)")
    print("-" * 60)
    
    success_count = 0
    query_count = 0
    for i in range(20):
        obs, info = env.reset()
        work_req = env._true_work_required
        deadline = env._deadline
        print(f"Task {i+1}: work_required={work_req}, deadline={deadline}")
        total_reward = 0
        
        # Check time every 30 steps
        for step in range(150):
            obs, reward, terminated, truncated, info = env.step(2)  # CHECK_TIME
            total_reward += reward
            query_count += 1
            
            # If work is done, submit
            if info['work_done'] >= info['true_work_required']:
                obs, reward, terminated, truncated, info = env.step(3)  # SUBMIT
                total_reward += reward
                if info['true_time'] <= info['deadline']:
                    success_count += 1
                break
            
            if terminated or truncated:
                break
        
        # If still going, submit
        if not terminated and env._work_done >= env._true_work_required:
            obs, reward, terminated, truncated, info = env.step(3)
            total_reward += reward
            if info['true_time'] <= info['deadline']:
                success_count += 1
    
    print(f"\nSuccess rate: {success_count}/20 = {success_count/20*100:.1f}%")
    print(f"Avg queries per task: {query_count/20:.1f}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
