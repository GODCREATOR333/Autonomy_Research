import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from config import RESULTS_FILE, RESULTS_DIR, N_EPISODES, EVAL_EVERY

with open(RESULTS_FILE, 'rb') as f:
    all_results = pickle.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Algorithm Comparison — N16 p=0100', fontsize=14)

COLORS = {
    'qlearning':  '#E74C3C',
    'sarsa':      '#3498DB',
    'td_lambda':  '#2ECC71',
    'montecarlo': '#F39C12',
}

metrics = [
    ('success_rate', 'Success Rate (%)',    True,  axes[0,0]),
    ('avg_steps',    'Avg Steps to Goal',   False, axes[0,1]),
    ('qtable_size',  'Q-Table Size',        False, axes[1,0]),
    ('train_reward', 'Avg Train Reward',    False, axes[1,1]),
]

for metric, ylabel, as_percent, ax in metrics:
    for name, seed_results in all_results.items():

        if metric == 'train_reward':
            # train reward logged every episode, smooth it
            all_rewards = np.array([
                seed_results[seed]['train_reward']
                for seed in seed_results
            ])
            window   = 1000
            smoothed = np.array([
                np.convolve(r, np.ones(window)/window, mode='valid')
                for r in all_rewards
            ])
            episodes = np.arange(window, N_EPISODES + 1)
            mean     = smoothed.mean(axis=0)
            std      = smoothed.std(axis=0)
        else:
            # eval metrics logged every EVAL_EVERY episodes
            data     = np.array([
                seed_results[seed][metric]
                for seed in seed_results
            ])
            episodes = seed_results[0]['eval_episodes']
            mean     = data.mean(axis=0)
            std      = data.std(axis=0)
            if as_percent:
                mean *= 100
                std  *= 100

        ax.plot(episodes, mean,
                label=name, color=COLORS[name], linewidth=2)
        ax.fill_between(episodes,
                        mean - std, mean + std,
                        color=COLORS[name], alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparison_N16_p0100.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {RESULTS_DIR}/comparison_N16_p0100.png")
