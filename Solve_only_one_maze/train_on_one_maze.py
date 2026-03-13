
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Load mazes ---

mazes = np.load('./data/maze_dataset/N16_p0100_train.npy')
print(f"Loaded {mazes.shape[0]} mazes of size {mazes.shape[1]}x{mazes.shape[2]}")
train_maze = mazes[10]
print(np.shape(train_maze))

np.random.seed(42)


# --- Parameters ---
GRID_SIZE = 16
N_ACTIONS = 4
START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
WINDOW_SIZE = 3   
HALF_W = WINDOW_SIZE // 2

# --- Action to movement mapping ---
action_to_delta = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}



# --- Dictionary Q-table ---
Q = {}


def get_Q(Q, state):
    if state not in Q:
        Q[state] = np.zeros(N_ACTIONS)
    return Q[state]


#--- Get local window state ---
def get_state(maze, pos):
    # row, col = pos
    # window = []
    return pos

    for dr in range(-HALF_W, HALF_W + 1):
        for dc in range(-HALF_W, HALF_W + 1):
            r = row + dr
            c = col + dc

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
        return pos, -10.0, False

    if maze[new_row, new_col] == 1:
        return pos, -10.0, False

    if (new_row, new_col) == GOAL: 
        return (new_row, new_col), +100.0, True

    return (new_row, new_col), -50, False


# --- Epsilon greedy ---
def select_action(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    else:
        q = get_Q(Q, state)
        max_q = q.max()
        best_actions = np.where(q == max_q)[0]
        return np.random.choice(best_actions)


# --- Q-learning loop ---
def train(mazes, Q, n_episodes, alpha, gamma):

    target_maze = mazes[10]
    rewards_per_episode = []
    print("Training Started")

    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.99997

    for episode in range(n_episodes):

        # maze = mazes[np.random.randint(len(mazes))]
        maze = target_maze
        pos = START
        state = get_state(maze, pos)

        total_reward = 0
        max_steps = GRID_SIZE * GRID_SIZE * 8

        # visit count for this episode only
        visit_count = {}    

        for _ in range(max_steps):

            action = select_action(Q, state, epsilon)

            new_pos, reward, done = step(maze, pos, action)
            new_state = get_state(maze, new_pos)

            # visit penalty
            count = visit_count.get(new_pos, 0)

            if not done and count > 0:
                reward -= 2

            visit_count[new_pos] = count + 1

            q_values = get_Q(Q, state)
            current_Q = q_values[action]
            if done:
                target_Q = reward
            else:
                target_Q = reward + gamma * np.max(get_Q(Q, new_state))

            td_error = target_Q - current_Q
            q_values[action] += alpha * td_error
            pos = new_pos
            state = new_state

            total_reward += reward

            if done:
                break
        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-1000:])
            print(
                f"Episode {episode+1}/{n_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Q-table size: {len(Q)} | "
                f"Epsilon: {epsilon:.3f}"
            )

    return Q, rewards_per_episode

print(f"Q-table saved | Total states: {len(Q)}")

# --- Run ---
Q, rewards = train(
    mazes,
    Q,
    n_episodes=100000,
    alpha=0.1,
    gamma=0.99
)

# --- Plot ---
plt.figure(figsize=(10, 4))

window = 500
smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')

plt.plot(smoothed)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Q-learning Training Curve')
plt.grid(True)

plt.show()

# --- Save Q-table ---
with open('q_table_N16_p0100.pkl', 'wb') as f:
    pickle.dump(Q, f)
print("Training Done! Q-Table Saved")



def test_policy(maze, Q):

    pos = START
    state = get_state(maze, pos)

    path = [pos]

    max_steps = GRID_SIZE * GRID_SIZE * 4

    for _ in range(max_steps):

        q = get_Q(Q, state)
        action = np.argmax(q)

        new_pos, reward, done = step(maze, pos, action)

        pos = new_pos
        state = get_state(maze, pos)

        path.append(pos)

        if done:
            break

    return path, done


arrow_map = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→"
}

def print_policy_arrows(maze, Q):

    policy_grid = []

    for r in range(GRID_SIZE):
        row = []

        for c in range(GRID_SIZE):

            if maze[r, c] == 1:
                row.append("#")
                continue

            if (r, c) == START:
                row.append("S")
                continue

            if (r, c) == GOAL:
                row.append("G")
                continue

            state = get_state(maze, (r, c))
            q = get_Q(Q, state)

            best_actions = np.where(q == q.max())[0]
            action = np.random.choice(best_actions)

            row.append(arrow_map[action])

        policy_grid.append(row)

    for row in policy_grid:
        print(" ".join(row))

maze = mazes[10]

path, success = test_policy(maze, Q)

print("\nPolicy success:", success)
print("Steps taken:", len(path))

maze = mazes[10]

print("\nLearned Policy:\n")
print_policy_arrows(maze, Q)