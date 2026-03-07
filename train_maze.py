import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Load mazes ---
mazes = np.load('./data/maze_dataset/N16_p0100_train.npy')
print(f"Loaded {mazes.shape[0]} mazes of size {mazes.shape[1]}x{mazes.shape[2]}")

# --- Parameters ---
GRID_SIZE = 16
N_ACTIONS = 4
START = (0, 0)
GOAL = (GRID_SIZE-1, GRID_SIZE-1)

# --- Action to movement mapping ---
action_to_delta = {
    0: (-1,  0),
    1: ( 1,  0),
    2: ( 0, -1),
    3: ( 0,  1),
}

# --- Dictionary Q-table ---
Q = {}

def get_Q(Q, state):
    if state not in Q:
        Q[state] = np.zeros(N_ACTIONS)
    return Q[state]

# --- Get local window state ---
def get_state(maze, pos):
    row, col = pos
    window = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            r, c = row+dr, col+dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                window.append(maze[r, c])
            else:
                window.append(1)
    dr_goal = int(np.sign(GOAL[0] - row))
    dc_goal = int(np.sign(GOAL[1] - col))
    return tuple(window) + (dr_goal, dc_goal)

# --- Step function ---
def step(maze, pos, action):
    row, col = pos
    dr, dc = action_to_delta[action]
    new_row = row + dr
    new_col = col + dc
    if new_row < 0 or new_row >= GRID_SIZE or new_col < 0 or new_col >= GRID_SIZE:
        return pos, -50.0, False
    if maze[new_row, new_col] == 1:
        return pos, -50.0, False
    if (new_row, new_col) == GOAL:
        return (new_row, new_col), +500.0, True
    return (new_row, new_col), -1, False

# --- Epsilon greedy ---
def select_action(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    else:
        return np.argmax(get_Q(Q, state))

# --- Q-learning loop ---
def train(mazes, Q, n_episodes, alpha, gamma):
    rewards_per_episode = []
    print("Training Started")

    epsilon       = 1.0
    epsilon_end   = 0.05
    epsilon_decay = 0.99995

    for episode in range(n_episodes):

        maze  = mazes[np.random.randint(len(mazes))]
        pos   = START
        state = get_state(maze, pos)
        total_reward = 0
        max_steps = GRID_SIZE * GRID_SIZE * 4

        for _ in range(max_steps):
            action    = select_action(Q, state, epsilon)
            new_pos, reward, done = step(maze, pos, action)
            new_state = get_state(maze, new_pos)

            current_Q = get_Q(Q, state)[action]
            target_Q  = reward + gamma * np.max(get_Q(Q, new_state))
            td_error  = target_Q - current_Q
            get_Q(Q, state)[action] += alpha * td_error

            pos   = new_pos
            state = new_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        # --- this was outside the loop before, now it is inside ---
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-1000:])
            print(f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | Q-table size: {len(Q)} | Epsilon: {epsilon:.3f}")

    return Q, rewards_per_episode

# --- Run ---
Q, rewards = train(mazes, Q, n_episodes=50000, alpha=0.1, gamma=0.99)

# --- Plot ---
plt.figure(figsize=(10, 4))
window   = 500
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(smoothed)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Q-learning Training Curve')
plt.grid(True)
plt.show()

# --- Save ---
with open('q_table_N16_p0100.pkl', 'wb') as f:
    pickle.dump(Q, f)
print(f"Q-table saved | Total unique states: {len(Q)}")