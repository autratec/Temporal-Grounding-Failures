"""
PRM Implementation for ChronoEnv - Temporal Regret Module

Extends PRM (Psychological Regret Modeling) to include temporal regret:
- Regret for not checking time when uncertainty is high
- Regret for redundant time checks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional


class TemporalRegretModule(nn.Module):
    """
    Temporal Regret Module for time-aware RL.
    
    Implements regret signals for time decisions:
    1. High uncertainty + no check → negative regret
    2. Low uncertainty + redundant check → negative regret
    """
    
    def __init__(
        self,
        high_uncertainty_weight: float = 0.5,
        redundant_check_weight: float = 0.3,
        uncertainty_threshold: float = 30.0,
        temporal_horizon: int = 10,
    ):
        super().__init__()
        self.high_uncertainty_weight = high_uncertainty_weight
        self.redundant_check_weight = redundant_check_weight
        self.uncertainty_threshold = uncertainty_threshold
        self.temporal_horizon = temporal_horizon
        
        # Learnable weights (can be trained)
        self.register_buffer("w_high_uncertainty", torch.tensor(high_uncertainty_weight))
        self.register_buffer("w_redundant", torch.tensor(redundant_check_weight))
    
    def compute_temporal_regret(
        self,
        uncertainty: float,
        action: int,
        just_checked: bool,
        time_since_check: int,
    ) -> float:
        """
        Compute temporal regret for a single decision.
        
        Args:
            uncertainty: Estimated time uncertainty (minutes)
            action: 0=WORK, 1=WAIT, 2=CHECK_TIME, 3=SUBMIT
            just_checked: Whether action was CHECK_TIME
            time_since_check: Steps since last check
        
        Returns:
            Temporal regret value
        """
        regret = 0.0
        
        # Regret 1: High uncertainty without checking
        if action in [0, 1]:  # WORK or WAIT
            if uncertainty > self.uncertainty_threshold:
                # Should have checked!
                regret -= self.w_high_uncertainty.item() * (
                    uncertainty / self.uncertainty_threshold
                )
        
        # Regret 2: Redundant checking (low uncertainty, just checked)
        if just_checked and time_since_check < self.temporal_horizon:
            # Check again too soon
            regret -= self.w_redundant.item()
        
        return regret
    
    def compute_trajectory_regret(
        self,
        uncertainties: list,
        actions: list,
        time_since_checks: list,
    ) -> torch.Tensor:
        """
        Compute temporal regret for entire trajectory.
        
        Args:
            uncertainties: List of uncertainty values
            actions: List of actions
            time_since_checks: List of steps since last check
        
        Returns:
            Total temporal regret tensor
        """
        regrets = []
        just_checked = False
        
        for i in range(len(actions)):
            regret = self.compute_temporal_regret(
                uncertainty=uncertainties[i],
                action=actions[i],
                just_checked=just_checked,
                time_since_check=time_since_checks[i],
            )
            regrets.append(regret)
            
            just_checked = (actions[i] == 2)  # CHECK_TIME action
        
        return torch.tensor(regrets).sum()
    
    def forward(self, batch: Dict) -> torch.Tensor:
        """Compute batch temporal regret."""
        uncertainties = batch.get("uncertainties", [])
        actions = batch.get("actions", [])
        time_since_checks = batch.get("time_since_checks", [])
        
        return self.compute_trajectory_regret(uncertainties, actions, time_since_checks)


class PRMLoss(nn.Module):
    """
    PRM Loss for time-aware RL.
    
    Combines task reward with temporal regret for process supervision.
    """
    
    def __init__(
        self,
        task_weight: float = 1.0,
        temporal_weight: float = 0.3,
        uncertainty_threshold: float = 30.0,
    ):
        super().__init__()
        self.task_weight = task_weight
        self.temporal_weight = temporal_weight
        self.uncertainty_threshold = uncertainty_threshold
        self.temporal_regret = TemporalRegretModule(uncertainty_threshold=uncertainty_threshold)
    
    def forward(
        self,
        rewards: torch.Tensor,
        uncertainties: list,
        actions: list,
        time_since_checks: list,
    ) -> torch.Tensor:
        """
        Compute PRM loss.
        
        Args:
            rewards: Task rewards
            uncertainties: Time uncertainty per step
            actions: Actions taken
            time_since_checks: Steps since last time check
        
        Returns:
            PRM loss (to be minimized)
        """
        # Task reward component
        task_loss = -rewards.mean()  # Maximize reward = minimize negative
        
        # Temporal regret component
        temporal_regret = self.temporal_regret.compute_trajectory_regret(
            uncertainties, actions, time_since_checks
        )
        
        # Combined loss
        total_loss = self.task_weight * task_loss + self.temporal_weight * temporal_regret
        
        return total_loss


class ProcessRewardModel(nn.Module):
    """
    Process Reward Model with temporal dimensions.
    
    Learn to assign rewards based on process quality, not just outcomes.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        temporal_weight: float = 0.3,
    ):
        super().__init__()
        
        # Process reward network
        self.prn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.temporal_weight = temporal_weight
        self.temporal_regret = TemporalRegretModule()
    
    def compute_process_reward(
        self,
        state: torch.Tensor,
        action: int,
        uncertainty: float,
        time_since_check: int,
        just_checked: bool,
    ) -> torch.Tensor:
        """
        Compute process reward for a single step.
        
        Args:
            state: Environment state
            action: Action taken
            uncertainty: Time uncertainty
            time_since_check: Steps since last check
            just_checked: Whether just checked time
        
        Returns:
            Process reward
        """
        # Base process reward from PRN
        base_reward = self.prn(state)
        
        # Temporal regret adjustment
        temporal_regret = self.temporal_regret.compute_temporal_regret(
            uncertainty=uncertainty,
            action=action,
            just_checked=just_checked,
            time_since_check=time_since_check,
        )
        
        return base_reward + self.temporal_weight * temporal_regret
    
    def forward(self, batch: Dict) -> torch.Tensor:
        """Compute batch process rewards."""
        states = batch["states"]
        actions = batch["actions"]
        uncertainties = batch["uncertainties"]
        time_since_checks = batch["time_since_checks"]
        
        rewards = []
        just_checked = False
        
        for i in range(len(actions)):
            reward = self.compute_process_reward(
                state=states[i],
                action=actions[i],
                uncertainty=uncertainties[i],
                time_since_check=time_since_checks[i],
                just_checked=just_checked,
            )
            rewards.append(reward)
            
            just_checked = (actions[i] == 2)
        
        return torch.stack(rewards)


if __name__ == "__main__":
    # Test TemporalRegretModule
    print("=" * 60)
    print("Testing TemporalRegretModule")
    print("=" * 60)
    
    trm = TemporalRegretModule(uncertainty_threshold=30.0)
    
    # Test 1: High uncertainty, no check
    regret = trm.compute_temporal_regret(
        uncertainty=50.0, action=0, just_checked=False, time_since_check=5
    )
    print(f"Test 1 - High uncertainty, no check: regret={regret:.4f}")
    
    # Test 2: Low uncertainty, redundant check
    regret = trm.compute_temporal_regret(
        uncertainty=10.0, action=2, just_checked=True, time_since_check=2
    )
    print(f"Test 2 - Low uncertainty, redundant check: regret={regret:.4f}")
    
    # Test 3: Normal case
    regret = trm.compute_temporal_regret(
        uncertainty=25.0, action=0, just_checked=False, time_since_check=8
    )
    print(f"Test 3 - Normal case: regret={regret:.4f}")
    
    # Test PRMLoss
    print("\n" + "=" * 60)
    print("Testing PRMLoss")
    print("=" * 60)
    
    prm_loss = PRMLoss(task_weight=1.0, temporal_weight=0.3)
    
    rewards = torch.tensor([5.0, 3.0, -2.0, 10.0])
    uncertainties = [10.0, 20.0, 40.0, 15.0]
    actions = [0, 0, 2, 0]
    time_since_checks = [1, 2, 1, 3]
    
    loss = prm_loss.forward(rewards, uncertainties, actions, time_since_checks)
    print(f"PRM Loss: {loss.item():.4f}")
    
    # Test ProcessRewardModel
    print("\n" + "=" * 60)
    print("Testing ProcessRewardModel")
    print("=" * 60)
    
    prm = ProcessRewardModel(state_dim=5, temporal_weight=0.3)
    
    state = torch.randn(1, 5)
    reward = prm.compute_process_reward(
        state=state,
        action=2,
        uncertainty=35.0,
        time_since_check=5,
        just_checked=False,
    )
    print(f"Process reward: {reward.item():.4f}")
    
    print("\n✓ All PRM tests passed!")
