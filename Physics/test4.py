import matplotlib.pyplot as plt
import numpy as np

# Data from your proactive analysis
densities = [10, 20, 30, 35, 37, 40]
opt_gp = [0.01, 0.01, 0.51, 0.51, 0.66, 0.44]
est_scores = [58.9, 114.3, 190.1, 233.5, 239.1, 242.6]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Optimal gp
color = 'tab:red'
ax1.set_xlabel('Obstacle Density (%)', fontsize=12)
ax1.set_ylabel('Optimal Proactive Switch Rate ($gp^*$)', color=color, fontsize=12)
ax1.step(densities, opt_gp, where='mid', color=color, linewidth=3, label='Optimal Switching Rate')
ax1.scatter(densities, opt_gp, color=color, s=100, zorder=5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Annotate the regimes
ax1.annotate('Regime 1: Ballistic\n(Pure Geo)', xy=(12, 0.05), color='darkred', fontweight='bold')
ax1.annotate('Regime 2: Intermittent\n(Hybrid)', xy=(31, 0.55), color='darkred', fontweight='bold')

# Plot EST Score
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Expected Search Time (Score)', color=color, fontsize=12)
ax2.plot(densities, est_scores, 's-', color=color, alpha=0.5, label='EST Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Discovery of Strategy Phase Transition in Mapless Navigation', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()