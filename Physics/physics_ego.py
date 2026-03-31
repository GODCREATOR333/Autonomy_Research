import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Physics Parameters
# =========================
D_R = 0.15
P_straight_ideal = np.exp(-D_R)
P_turn_ideal = (1.0 - P_straight_ideal) / 2.0

GOAL_POS = np.array([15, 15])
MAX_STEPS = 400
DATA_PATH = "data_jax/N16_P0100_test_solvable_random.npy"


# =========================
# 2. Analytical Stochastic Policy
# =========================
def get_ego_probabilities(win_id, last_a):
    # Bit decoding of 3x3 local window walls
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
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        for a in range(4):
            if a == last_a:
                probs[a] = P_straight_ideal
            elif a == opposites[last_a]:
                probs[a] = 0.0
            else:
                probs[a] = P_turn_ideal

    # Remove wall directions
    probs[is_wall == 1] = 0.0

    s = np.sum(probs)
    if s > 0:
        probs = probs / s
    else:
        # Dead-end fallback: reverse
        rev = {0: 1, 1: 0, 2: 3, 3: 2}[last_a]
        probs[rev] = 1.0

    return probs


def calculate_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


# =========================
# 3. Simulation
# =========================
def run_stochastic_sim(maze, max_steps=MAX_STEPS):
    pos = np.array([0, 0])
    last_a = -1
    path = [tuple(pos)]
    actions = []
    entropies = []
    collisions = 0

    deltas = {
        0: np.array([-1, 0]),  # UP
        1: np.array([1, 0]),   # DOWN
        2: np.array([0, -1]),  # LEFT
        3: np.array([0, 1])    # RIGHT
    }

    padded = np.pad(maze, 1, constant_values=1)

    for _ in range(max_steps):
        window = padded[pos[0]:pos[0]+3, pos[1]:pos[1]+3].flatten()
        win_id = int(np.sum(window * np.array([256,128,64,32,16,8,4,2,1])))

        probs = get_ego_probabilities(win_id, last_a)
        entropy = calculate_entropy(probs)

        action = np.random.choice(4, p=probs)

        next_pos = pos + deltas[action]

        # Safety check (should almost never fail because policy masks walls)
        if (
            0 <= next_pos[0] < maze.shape[0]
            and 0 <= next_pos[1] < maze.shape[1]
            and maze[next_pos[0], next_pos[1]] == 0
        ):
            pos = next_pos
        else:
            collisions += 1

        last_a = action

        path.append(tuple(pos))
        actions.append(action)
        entropies.append(entropy)

        if np.array_equal(pos, GOAL_POS):
            break

    return (
        np.array(path),
        np.array(actions),
        np.array(entropies),
        collisions
    )


# =========================
# 4. Evaluation
# =========================
def evaluate_agent(mazes, trials=20, max_steps=MAX_STEPS):
    success = 0
    steps_to_goal = []

    collision_counts = []
    final_distances = []
    unique_cells_visited = []
    path_efficiencies = []
    mean_entropies = []

    total_runs = len(mazes) * trials
    run_count = 0

    start_dist = np.linalg.norm(GOAL_POS - np.array([0, 0]))

    for i, maze in enumerate(mazes):
        print(f"Maze {i+1}/{len(mazes)}")

        for t in range(trials):
            run_count += 1
            if run_count % 50 == 0:
                print(f"  Simulation {run_count}/{total_runs}")

            path, actions, ents, collisions = run_stochastic_sim(maze, max_steps=max_steps)

            # Success
            reached_goal = np.array_equal(path[-1], GOAL_POS)
            if reached_goal:
                success += 1
                steps_to_goal.append(len(path) - 1)

            # Collisions
            collision_counts.append(collisions)

            # Final distance to goal
            final_dist = np.linalg.norm(path[-1] - GOAL_POS)
            final_distances.append(final_dist)

            # Unique cells visited
            n_unique = len(set(map(tuple, path)))
            unique_cells_visited.append(n_unique)

            # Path efficiency = net progress / steps
            progress = start_dist - final_dist
            efficiency = progress / max(1, len(path) - 1)
            path_efficiencies.append(efficiency)

            # Mean entropy of decisions
            mean_entropy = np.mean(ents) if len(ents) > 0 else 0.0
            mean_entropies.append(mean_entropy)

    success_rate = success / total_runs
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else None

    results = {
        "success_rate": success_rate,
        "avg_steps_to_goal": avg_steps,
        "avg_collisions": np.mean(collision_counts),
        "avg_final_distance": np.mean(final_distances),
        "avg_unique_cells": np.mean(unique_cells_visited),
        "avg_path_efficiency": np.mean(path_efficiencies),
        "avg_entropy": np.mean(mean_entropies),
        "total_runs": total_runs,
        "successful_runs": success
    }

    return results


# =========================
# 5. Load Mazes
# =========================
print("Loading mazes...")
test_mazes = np.load(DATA_PATH)
print("Loaded mazes:", len(test_mazes))
print("Maze shape:", test_mazes[0].shape)

assert test_mazes.ndim == 3, "Expected shape: (num_mazes, H, W)"
assert test_mazes.shape[1:] == (16, 16), "Expected 16x16 mazes"
assert test_mazes[0, 0, 0] == 0, "Start cell [0,0] should be free"
assert test_mazes[0, 15, 15] == 0, "Goal cell [15,15] should be free"


# =========================
# 6. Evaluate First
# =========================
print("Starting evaluation...")
results = evaluate_agent(test_mazes, trials=20, max_steps=MAX_STEPS)
print("Evaluation done\n")

print("===== EGO AGENT RESULTS =====")
print(f"Success rate         : {results['success_rate']:.4f}")
print(f"Average steps        : {results['avg_steps_to_goal']}")
print(f"Average collisions   : {results['avg_collisions']:.2f}")
print(f"Average final dist   : {results['avg_final_distance']:.2f}")
print(f"Average unique cells : {results['avg_unique_cells']:.2f}")
print(f"Average efficiency   : {results['avg_path_efficiency']:.4f}")
print(f"Average entropy      : {results['avg_entropy']:.4f}")
print(f"Successful runs      : {results['successful_runs']}/{results['total_runs']}")


# =========================
# 7. Interactive Viewer
# =========================
current_idx = 0
fig, ax = plt.subplots(figsize=(8, 8))

def update_plot():
    ax.clear()
    maze = test_mazes[current_idx]
    path, actions, ents, collisions = run_stochastic_sim(maze)

    ax.imshow(maze, cmap='gray_r', extent=[-0.5, 15.5, 15.5, -0.5])
    ax.plot(path[:, 1], path[:, 0], color='white', linewidth=1, alpha=0.3)

    if len(actions) > 0:
        dy = np.array([-1, 1, 0, 0])[actions]
        dx = np.array([0, 0, -1, 1])[actions]
        colors = plt.cm.plasma(ents / np.log(4))  # normalize by max entropy
        ax.quiver(
            path[:-1, 1], path[:-1, 0],
            dx, -dy,
            color=colors,
            scale=25,
            width=0.006,
            headwidth=4
        )

    reached = np.array_equal(path[-1], GOAL_POS)

    ax.plot(0, 0, 'go', markersize=8, label='Start')
    ax.plot(GOAL_POS[1], GOAL_POS[0], 'r*', markersize=15, label='Goal')

    title = f"Stochastic ABP | Maze #{current_idx}"
    title += "\nSUCCESS" if reached else "\nFAILED"
    title += f" | Collisions: {collisions}"

    ax.set_title(title)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(15.5, -0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    plt.draw()

def on_key(event):
    global current_idx
    if event.key == 'right':
        current_idx = (current_idx + 1) % len(test_mazes)
    elif event.key == 'left':
        current_idx = (current_idx - 1) % len(test_mazes)
    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)

print("\nPress Left/Right arrows to switch mazes.")
update_plot()
plt.show()