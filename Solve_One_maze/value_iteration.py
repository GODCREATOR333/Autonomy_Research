import numpy as np
import pickle

# --- Load all mazes (train + test) ---
test_mazes  = np.load('./data/maze_dataset/N16_p0100_test.npy')
train_mazes = np.load('./data/maze_dataset/N16_p0100_train.npy')

GRID_SIZE = 16
START     = (0, 0)
GOAL      = (GRID_SIZE-1, GRID_SIZE-1)
gamma     = 0.99
theta     = 1e-6   # convergence threshold

action_to_delta = {
    0: (-1,  0),
    1: ( 1,  0),
    2: ( 0, -1),
    3: ( 0,  1),
}

def value_iteration(maze):
    # init V to zeros everywhere
    V = np.zeros((GRID_SIZE, GRID_SIZE))

    while True:
        delta = 0.0

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):

                # skip obstacles and goal
                if maze[r, c] == 1:
                    continue
                if (r, c) == GOAL:
                    V[r, c] = 100.0
                    continue

                old_v = V[r, c]
                action_values = []

                for action, (dr, dc) in action_to_delta.items():
                    nr, nc = r+dr, c+dc

                    # out of bounds or obstacle — stay in place
                    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
                        reward  = -1.0
                        next_v  = V[r, c]
                    elif maze[nr, nc] == 1:
                        reward  = -1.0
                        next_v  = V[r, c]
                    else:
                        reward  = -0.1
                        next_v  = V[nr, nc]

                    action_values.append(reward + gamma * next_v)

                V[r, c] = max(action_values)
                delta   = max(delta, abs(old_v - V[r, c]))

        if delta < theta:
            break

    # extract optimal policy
    policy = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if maze[r, c] == 1:
                continue
            if (r, c) == GOAL:
                continue
            action_values = []
            for action, (dr, dc) in action_to_delta.items():
                nr, nc = r+dr, c+dc
                if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
                    reward = -1.0
                    next_v = V[r, c]
                elif maze[nr, nc] == 1:
                    reward = -1.0
                    next_v = V[r, c]
                else:
                    reward = -0.1
                    next_v = V[nr, nc]
                action_values.append(reward + gamma * next_v)
            policy[r, c] = np.argmax(action_values)

    return V, policy


# --- Solve all test mazes ---
print("Solving test mazes with value iteration...")
test_results = []
for i, maze in enumerate(test_mazes):
    V, policy = value_iteration(maze)
    test_results.append({'V': V, 'policy': policy})
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(test_mazes)} done")

with open('vi_test_N16_p0100.pkl', 'wb') as f:
    pickle.dump(test_results, f)
print(f"Saved vi_test_N16_p0100.pkl")

# --- Solve all train mazes ---
print("Solving train mazes with value iteration...")
train_results = []
for i, maze in enumerate(train_mazes):
    V, policy = value_iteration(maze)
    train_results.append({'V': V, 'policy': policy})
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(train_mazes)} done")

with open('vi_train_N16_p0100.pkl', 'wb') as f:
    pickle.dump(train_results, f)
print(f"Saved vi_train_N16_p0100.pkl")