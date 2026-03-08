import numpy as np
import pickle

# --- Load mazes ---
all_mazes = np.load('./data/maze_dataset/N16_p0100_test.npy')
maze = all_mazes[10]   # maze number 10

GRID_SIZE = 16
START = (0,0)
GOAL  = (GRID_SIZE-1, GRID_SIZE-1)
N_ACTIONS = 4

action_to_delta = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
action_symbols = {0:'↑', 1:'↓', 2:'←', 3:'→'}

# --- Q-learning hyperparameters ---
alpha = 0.1
gamma = 0.99
epsilon = 0.1
n_iterations = 1000

HALF_W = 2
WINDOW_SIZE = 5

# --- Helpers ---
def get_state(maze, pos):
    row, col = pos
    window = []
    for dr in range(-HALF_W, HALF_W+1):
        for dc in range(-HALF_W, HALF_W+1):
            r, c = row+dr, col+dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                window.append(maze[r,c])
            else:
                window.append(1)
    dr_goal = int(np.sign(GOAL[0] - row))
    dc_goal = int(np.sign(GOAL[1] - col))
    return tuple(window) + (dr_goal, dc_goal)

def step(maze, pos, action):
    row, col = pos
    dr, dc = action_to_delta[action]
    nr, nc = row+dr, col+dc
    if nr<0 or nr>=GRID_SIZE or nc<0 or nc>=GRID_SIZE:
        return pos, -1.0, False
    if maze[nr,nc] == 1:
        return pos, -1.0, False
    if (nr,nc) == GOAL:
        return (nr,nc), 0.0, True
    return (nr,nc), -0.1, False

def get_path(maze, Q):
    pos = START
    path = [pos]
    for _ in range(GRID_SIZE*GRID_SIZE*2):
        state = get_state(maze, pos)
        q_vals = Q.get(state, np.zeros(N_ACTIONS))
        action = np.argmax(q_vals)
        pos, _, done = step(maze, pos, action)
        path.append(pos)
        if done:
            return path, True
    return path, False

# --- Initialize Q-table ---
Q = {}

# --- Training ---
for it in range(n_iterations):
    pos = START
    for _ in range(GRID_SIZE*GRID_SIZE*2):
        state = get_state(maze, pos)
        if np.random.rand() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            q_vals = Q.get(state, np.zeros(N_ACTIONS))
            action = np.argmax(q_vals)
        
        next_pos, reward, done = step(maze, pos, action)
        next_state = get_state(maze, next_pos)
        
        q_vals = Q.get(state, np.zeros(N_ACTIONS))
        next_q = Q.get(next_state, np.zeros(N_ACTIONS))
        q_vals[action] = q_vals[action] + alpha * (reward + gamma*np.max(next_q) - q_vals[action])
        Q[state] = q_vals
        
        pos = next_pos
        if done:
            break

print("Training completed.")

# --- Test 100 times ---
successes = 0
for _ in range(100):
    _, success = get_path(maze, Q)
    if success:
        successes += 1

print(f"Success rate over 100 tests: {successes}%")

# --- Print learned policy ---
policy_grid = np.full((GRID_SIZE, GRID_SIZE), ' ')
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if maze[r,c]==1:
            policy_grid[r,c] = '#'
        elif (r,c)==GOAL:
            policy_grid[r,c] = 'G'
        elif (r,c)==START:
            policy_grid[r,c] = 'S'
        else:
            state = get_state(maze, (r,c))
            action = np.argmax(Q.get(state, np.zeros(N_ACTIONS)))
            policy_grid[r,c] = action_symbols[action]

print("\nLearned Policy (maze 10):")
for row in policy_grid:
    print(' '.join(row))