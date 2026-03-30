import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physics Parameters ---
D_R = 0.15 
P_straight_ideal = np.exp(-D_R)
P_turn_ideal = (1.0 - P_straight_ideal) / 2.0

# --- 2. Analytical Stochastic Policy ---
def get_ego_probabilities(win_id, last_a):
    # Fast bit decoding (no string conversion)
    is_wall = np.array([
        (win_id >> 7) & 1,  # UP
        (win_id >> 1) & 1,  # DOWN
        (win_id >> 5) & 1,  # LEFT
        (win_id >> 3) & 1   # RIGHT
    ])
    
    probs = np.zeros(4)

    if last_a == -1:
        probs = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        opposites = {0:1, 1:0, 2:3, 3:2}
        for a in range(4):
            if a == last_a:
                probs[a] = P_straight_ideal
            elif a == opposites[last_a]:
                probs[a] = 0.0
            else:
                probs[a] = P_turn_ideal

    probs[is_wall == 1] = 0.0
    
    s = np.sum(probs)
    if s > 0:
        probs = probs / s
    else:
        rev = {0:1, 1:0, 2:3, 3:2}[last_a]
        probs[rev] = 1.0
        
    return probs


def calculate_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


# --- 3. Simulation ---
def run_stochastic_sim(maze):
    pos = np.array([0, 0])
    last_a = -1
    path = [tuple(pos)]
    actions = []
    entropies = []
    
    deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    
    padded = np.pad(maze, 1, constant_values=1)

    for _ in range(400):
        window = padded[pos[0]:pos[0]+3, pos[1]:pos[1]+3].flatten()
        win_id = int(np.sum(window * np.array([256,128,64,32,16,8,4,2,1])))

        probs = get_ego_probabilities(win_id, last_a)
        entropy = calculate_entropy(probs)

        action = np.random.choice(4, p=probs)

        dr, dc = deltas[action]
        pos = pos + np.array([dr, dc])
        last_a = action

        path.append(tuple(pos))
        actions.append(action)
        entropies.append(entropy)

        if np.array_equal(pos, [15, 15]):
            break

    return np.array(path), np.array(actions), np.array(entropies)


# --- 4. Evaluation ---
def evaluate_agent(mazes, trials=20):
    success = 0
    steps_to_goal = []

    total_runs = len(mazes) * trials
    run_count = 0

    for i, maze in enumerate(mazes):
        print(f"Maze {i+1}/{len(mazes)}")

        for t in range(trials):
            run_count += 1
            if run_count % 50 == 0:
                print(f"  Simulation {run_count}/{total_runs}")

            path, _, _ = run_stochastic_sim(maze)

            if np.array_equal(path[-1], [15, 15]):
                success += 1
                steps_to_goal.append(len(path))

    success_rate = success / total_runs
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else None

    return success_rate, avg_steps


# --- Load Mazes ---
DATA_PATH = "data_jax/N16_P0100_test_solvable_random.npy"
print("Loading mazes...")
test_mazes = np.load(DATA_PATH)
print("Loaded mazes:", len(test_mazes))

# --- Evaluate First ---
print("Starting evaluation...")
rate, steps = evaluate_agent(test_mazes, trials=20)
print("Evaluation done")
print("Success rate:", rate)
print("Average steps to goal:", steps)


# --- 5. Interactive Viewer ---
current_idx = 0
fig, ax = plt.subplots(figsize=(8, 8))

def update_plot():
    ax.clear()
    maze = test_mazes[current_idx]
    path, actions, ents = run_stochastic_sim(maze)
    
    ax.imshow(maze, cmap='gray_r', extent=[-0.5, 15.5, 15.5, -0.5])
    ax.plot(path[:, 1], path[:, 0], color='white', linewidth=1, alpha=0.3)
    
    dy = np.array([-1, 1, 0, 0])[actions]
    dx = np.array([0, 0, -1, 1])[actions]
    
    colors = plt.cm.plasma(ents / 1.38)
    
    ax.quiver(path[:-1, 1], path[:-1, 0], dx, -dy,
              color=colors, scale=25, width=0.006, headwidth=4)
    
    ax.set_title(f"Stochastic ABP | Maze #{current_idx}")
    plt.draw()

def on_key(event):
    global current_idx
    if event.key == 'right':
        current_idx = (current_idx + 1) % len(test_mazes)
    elif event.key == 'left':
        current_idx = (current_idx - 1) % len(test_mazes)
    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)

update_plot()
plt.show()