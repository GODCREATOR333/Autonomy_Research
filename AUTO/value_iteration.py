import numpy as np
import sys

# Load dataset
grids = np.load("data_jax/N16_P0100_test_solvable_random.npy")

num_grids, grid_rows, grid_cols = grids.shape

print(f"Loaded {num_grids} mazes of size {grid_rows}x{grid_cols}")

gamma = 1
iterations = 100

# Actions
actions = {
    "up": (-1,0),
    "down": (1,0),
    "left": (0,-1),
    "right": (0,1)
}

action_names = list(actions.keys())

start = (0,0)
goal = (grid_rows-1, grid_cols-1)


def value_iteration(grid):

    V = np.zeros((grid_rows, grid_cols))
    policy = np.full((grid_rows, grid_cols), 'none', dtype=object)

    for k in range(iterations):

        new_V = V.copy()

        for r in range(grid_rows):
            for c in range(grid_cols):

                if grid[r,c] == 1:
                    continue

                if (r,c) == goal:
                    new_V[r,c] = 0
                    policy[r,c] = "GOAL"
                    continue

                action_values = []

                for action in action_names:

                    dr,dc = actions[action]

                    nr = r + dr
                    nc = c + dc

                    if nr < 0 or nr >= grid_rows or nc < 0 or nc >= grid_cols:
                        nr,nc = r,c

                    if grid[nr,nc] == 1:
                        nr,nc = r,c

                    if (nr,nc) == goal:
                        reward = 10
                    else:
                        reward = -1

                    value = reward + gamma * V[nr,nc]
                    action_values.append(value)

                best_value = np.max(action_values)
                best_action = action_names[np.argmax(action_values)]

                new_V[r,c] = best_value
                policy[r,c] = best_action

        V = new_V.copy()

    return V, policy


# Run value iteration on all grids
all_values = []
all_policies = []

for i in range(num_grids):

    grid = grids[i]

    V, policy = value_iteration(grid)

    all_values.append(V)
    all_policies.append(policy)

    # Progress display
    sys.stdout.write(f"\rProgress: {i+1}/{num_grids} mazes solved")
    sys.stdout.flush()

print("\nAll mazes solved.")