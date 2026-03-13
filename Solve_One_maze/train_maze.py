import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Load mazes ---
mazes = np.load('./data/maze_dataset/N16_p0100_train.npy')
print(f"Loaded {mazes.shape[0]} mazes of size {mazes.shape[1]}x{mazes.shape[2]}")

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
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
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

    for dr in range(-HALF_W, HALF_W + 1):
        for dc in range(-HALF_W, HALF_W + 1):
            r = row + dr
            c = col + dc

            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                window.append(maze[r, c])
            else:
                window.append(1)  # Out of bounds = blocked

    dr_goal = int(np.sign(GOAL[0] - row))
    dc_goal = int(np.sign(GOAL[1] - col))

    return tuple(window) + (dr_goal, dc_goal)

# --- Step function with progress-based reward shaping ---
def step(maze, pos, action):
    row, col = pos
    dr, dc = action_to_delta[action]

    new_row = row + dr
    new_col = col + dc

    # Wall or boundary collision
    if new_row < 0 or new_row >= GRID_SIZE or new_col < 0 or new_col >= GRID_SIZE:
        return pos, -10.0, False
    if maze[new_row, new_col] == 1:
        return pos, -10.0, False

    # Goal reached
    if (new_row, new_col) == GOAL: 
        return (new_row, new_col), +100.0, True

    # --- Progress-based reward shaping ---
    old_dist = abs(row - GOAL[0]) + abs(col - GOAL[1])          # Manhattan distance
    new_dist = abs(new_row - GOAL[0]) + abs(new_col - GOAL[1])
    progress_reward = 0.5 * (old_dist - new_dist)               # +0.5 for getting closer
    
    # Base step penalty (discourages wandering)
    step_penalty = -0.2
    
    return (new_row, new_col), step_penalty + progress_reward, False

# --- Epsilon greedy with tie-breaking toward goal ---
def select_action(Q, state, epsilon, pos=None):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    else:
        q = get_Q(Q, state)
        max_q = q.max()
        best_actions = np.where(q == max_q)[0]
        
        # Tie-breaking: prefer actions aligned with goal direction
        if pos is not None and len(best_actions) > 1:
            row, col = pos
            dr_goal = np.sign(GOAL[0] - row)
            dc_goal = np.sign(GOAL[1] - col)
            
            action_deltas = [(-1,0), (1,0), (0,-1), (0,1)]  # UP, DOWN, LEFT, RIGHT
            preferred = []
            for act in best_actions:
                d_row, d_col = action_deltas[act]
                # Prefer if action moves toward goal direction
                if (dr_goal != 0 and d_row == dr_goal) or (dc_goal != 0 and d_col == dc_goal):
                    preferred.append(act)
            if preferred:
                return preferred[0]
        
        return np.random.choice(best_actions)

# --- Q-learning loop ---
def train(mazes, Q, n_episodes, alpha, gamma):
    rewards_per_episode = []
    print("Training Started")

    # Epsilon schedule: decay precisely to epsilon_end at final episode
    epsilon = 1.0
    epsilon_end = 0.01
    epsilon_decay = (epsilon_end / epsilon) ** (1 / n_episodes)

    for episode in range(n_episodes):
        maze = mazes[np.random.randint(len(mazes))]
        pos = START
        state = get_state(maze, pos)
        total_reward = 0
        max_steps = GRID_SIZE * GRID_SIZE * 8

        for _ in range(max_steps):
            action = select_action(Q, state, epsilon, pos=pos)  # Pass pos for tie-breaking
            new_pos, reward, done = step(maze, pos, action)
            new_state = get_state(maze, new_pos)

            # Q-learning update
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
                f"Epsilon: {epsilon:.4f}"
            )

    return Q, rewards_per_episode

# --- Evaluation function ---
def evaluate(Q, mazes, n_test=20, max_steps_factor=4):
    """Evaluate trained agent on random mazes (no exploration)"""
    success = 0
    total_steps = 0
    
    for i in range(n_test):
        maze = mazes[np.random.randint(len(mazes))]
        pos = START
        state = get_state(maze, pos)
        done = False
        steps = 0
        max_steps = GRID_SIZE * GRID_SIZE * max_steps_factor
        
        while not done and steps < max_steps:
            action = select_action(Q, state, epsilon=0.0, pos=pos)  # Greedy
            pos, reward, done = step(maze, pos, action)
            state = get_state(maze, pos)
            steps += 1
        
        if done:
            success += 1
        total_steps += steps
    
    success_rate = 100 * success / n_test
    avg_steps = total_steps / n_test if n_test > 0 else 0
    print(f"\n✅ Evaluation ({n_test} mazes):")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Avg Steps: {avg_steps:.1f}")
    return success_rate, avg_steps

# --- Main execution ---
if __name__ == "__main__":
    # Train
    Q, rewards = train(
        mazes,
        Q,
        n_episodes=50000,
        alpha=0.1,
        gamma=0.99
    )

    # Plot training curve
    plt.figure(figsize=(10, 4))
    window = 500
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.plot(smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (500-ep window)')
    plt.title('Q-learning Training Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    print("📈 Training curve saved as 'training_curve.png'")
    plt.show()

    # Evaluate
    print("\n🔍 Evaluating trained agent...")
    evaluate(Q, mazes, n_test=50)

    # Save Q-table
    with open('q_table_N16_p0100_fixed.pkl', 'wb') as f:
        pickle.dump(Q, f)
    print(f"💾 Q-table saved | Total states: {len(Q)}")
    print("🎉 Training Complete!")