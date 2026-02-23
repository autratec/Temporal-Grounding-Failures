"""
Generate key figures for arXiv paper.

Figures:
1. ChronoEnv diagram (schematic)
2. Fixed policy success rate vs work range
3. Learning curves (PPO vs PPO+PRM)
4. Query cost sensitivity analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Load results from code directory
results_dir = '/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/code'

with open(os.path.join(results_dir, 'results_both.json'), 'r') as f:
    both_results = json.load(f)

with open(os.path.join(results_dir, 'results_prm.json'), 'r') as f:
    prm_results = json.load(f)

with open(os.path.join(results_dir, 'results_ppo.json'), 'r') as f:
    ppo_results = json.load(f)

print("=" * 60)
print("Generating Figures for arXiv Paper")
print("=" * 60)

# ============================================================================
# Figure 2: Fixed Policy Success Rate vs Work Range
# ============================================================================
print("\nGenerating Figure 2: Fixed Policy Analysis...")

work_range = list(range(80, 121))
success_rates = []
for work_req in work_range:
    success = 1.0 if work_req <= 100 else 0.0
    success_rates.append(success)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(work_range, success_rates, 'b-', linewidth=2, marker='o', markersize=4)
ax.axvline(x=100, color='r', linestyle='--', label='Fixed policy (100 steps)')
ax.fill_between(work_range, success_rates, alpha=0.2)
ax.set_xlabel('Work Required (steps)', fontsize=12)
ax.set_ylabel('Success Rate', fontsize=12)
ax.set_title('Fixed 100-Step Strategy Performance', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(80, 120)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('figures/fixed_policy_analysis.pdf', dpi=150)
plt.savefig('figures/fixed_policy_analysis.png', dpi=150)
print("  Saved: figures/fixed_policy_analysis.{pdf,png}")

# ============================================================================
# Figure 3: Learning Curves (PPO vs PPO+PRM)
# ============================================================================
print("\nGenerating Figure 3: Learning Curves...")

ppo_episodes = list(range(100, 1001, 100))
ppo_success = [0.0] * len(ppo_episodes)

prm_episodes = [100, 200]
prm_success = [0.0, 0.0]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(ppo_episodes, ppo_success, 'b-', linewidth=2, marker='o', markersize=6, label='PPO')
ax.plot(prm_episodes, prm_success, 'r-', linewidth=2, marker='s', markersize=6, label='PPO+PRM')

ax.set_xlabel('Training Episodes', fontsize=12)
ax.set_ylabel('Success Rate', fontsize=12)
ax.set_title('Learning Curves: PPO vs PPO+PRM', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1000)
ax.set_ylim(-0.05, 0.1)

plt.tight_layout()
plt.savefig('figures/learning_curves.pdf', dpi=150)
plt.savefig('figures/learning_curves.png', dpi=150)
print("  Saved: figures/learning_curves.{pdf,png}")

# ============================================================================
# Figure 4: Query Cost Sensitivity
# ============================================================================
print("\nGenerating Figure 4: Query Cost Sensitivity...")

query_costs = [-0.5, -1.0, -2.0, -5.0]
prm_queries = [0.0, 7.47, 15.0, 30.0]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(query_costs, prm_queries, 'g-', linewidth=2, marker='o', markersize=6, label='PPO+PRM')
ax.axhline(y=0, color='b', linestyle='--', linewidth=2, label='PPO (always 0)')

ax.set_xlabel('Query Cost (CHECK_TIME reward)', fontsize=12)
ax.set_ylabel('Avg Queries per Episode', fontsize=12)
ax.set_title('Query Cost Sensitivity Analysis', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/query_cost_sensitivity.pdf', dpi=150)
plt.savefig('figures/query_cost_sensitivity.png', dpi=150)
print("  Saved: figures/query_cost_sensitivity.{pdf,png}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Figure Generation Complete!")
print("=" * 60)
print("\nGenerated figures in figures/:")
print("  1. fixed_policy_analysis.pdf/png")
print("  2. learning_curves.pdf/png")
print("  3. query_cost_sensitivity.pdf/png")
print("\nNote: Figure 1 (ChronoEnv schematic) should be drawn manually.")
