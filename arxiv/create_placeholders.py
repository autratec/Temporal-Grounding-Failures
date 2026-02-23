#!/usr/bin/env python3
"""
Create placeholder figures for paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a placeholder diagram for ChronoEnv
fig, ax = plt.subplots(figsize=(8, 6))

ax.add_patch(plt.Rectangle((0.1, 0.2), 0.2, 0.6, facecolor='blue', alpha=0.5, label='Agent'))
ax.add_patch(plt.Rectangle((0.7, 0.2), 0.2, 0.6, facecolor='red', alpha=0.5, label='Environment'))
ax.add_patch(plt.Rectangle((0.4, 0.4), 0.2, 0.3, facecolor='green', alpha=0.5, label='Clock'))

ax.text(0.2, 0.5, 'WORK/CHECK', ha='center', fontsize=12)
ax.text(0.8, 0.5, 'Reward/Time', ha='center', fontsize=12)
ax.text(0.5, 0.55, 'Time Query', ha='center', fontsize=12, rotation=90)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('ChronoEnv Architecture', fontsize=14)

plt.tight_layout()
plt.savefig('figures/env_diagram.pdf', dpi=150)
plt.savefig('figures/env_diagram.png', dpi=150)
print("Created: figures/env_diagram.{pdf,png}")

# Create a simple placeholder for learning curves
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot([0, 1000], [0, 0], 'b-', linewidth=2, label='PPO')
ax.plot([0, 1000], [0, 0], 'r--', linewidth=2, label='PPO+PRM')
ax.set_xlabel('Training Episodes')
ax.set_ylabel('Success Rate')
ax.set_title('Learning Curves (Both Failed)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/learning_curves_placeholder.pdf', dpi=150)
plt.savefig('figures/learning_curves_placeholder.png', dpi=150)
print("Created: figures/learning_curves_placeholder.{pdf,png}")

print("\nDone!")
