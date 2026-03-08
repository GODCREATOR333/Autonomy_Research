# train_all.py
import numpy as np
import pickle
from config import *
from environment import evaluate
from algorithms import (
    run_episode_qlearning,
    run_episode_sarsa,
    run_episode_td_lambda,
    run_episode_montecarlo,
)
import os
os.makedirs(RESULTS_DIR, exist_ok=True)

train_mazes = np.load('./data/maze_dataset/N16_p0100_train.npy')
test_mazes  = np.load('./data/maze_dataset/N16_p0100_test.npy')
print(f"Train: {len(train_mazes)} | Test: {len(test_mazes)}")

#np.random.seed(42)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

ALGORITHMS = {
    'qlearning':   run_episode_qlearning,
    'sarsa':       run_episode_sarsa,
    'td_lambda':   run_episode_td_lambda,
    'montecarlo':  run_episode_montecarlo,
}

all_results = {}

for name, run_episode in ALGORITHMS.items():
    all_results[name] = {}

    for seed in SEEDS:
        np.random.seed(seed)
        print(f"\n{name} | seed {seed}")

        Q       = {}
        epsilon = EPSILON_START
        results = {
            'train_reward':  [],
            'success_rate':  [],
            'avg_steps':     [],
            'qtable_size':   [],
            'eval_episodes': [],
        }

        for episode in range(N_EPISODES):
            maze      = train_mazes[np.random.randint(len(train_mazes))]
            Q, reward = run_episode(maze, Q, epsilon, ALPHA, GAMMA)
            epsilon   = max(EPSILON_END, epsilon * EPSILON_DECAY)
            results['train_reward'].append(reward)

            if (episode + 1) % EVAL_EVERY == 0:
                success_rate, avg_steps = evaluate(Q, test_mazes)
                results['success_rate'].append(success_rate)
                results['avg_steps'].append(avg_steps)
                results['qtable_size'].append(len(Q))
                results['eval_episodes'].append(episode + 1)

                print(f"Ep {episode+1:6d} | "
                      f"Success: {success_rate*100:5.1f}% | "
                      f"Steps: {avg_steps:6.1f} | "
                      f"Q-size: {len(Q):5d} | "
                      f"Eps: {epsilon:.4f}")

        # save per seed
        with open(f'q_table_{name}_seed{seed}_N16_p0100.pkl', 'wb') as f:
            pickle.dump(Q, f)

        all_results[name][seed] = results

with open(RESULTS_FILE, 'wb') as f:
    pickle.dump(all_results, f)
print("\nAll results saved.")