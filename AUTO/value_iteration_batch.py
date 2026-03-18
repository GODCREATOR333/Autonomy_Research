# ********************************* NUMPY LOOP *******************
# for k in range(10000):
    
#     # 1. Setup the shifted matrices (Clone V to handle walls)
#     V_up = np.copy(V)
#     V_down = np.copy(V)
#     V_left = np.copy(V)
#     V_right = np.copy(V)

#     # 2. Perform the shifts (Destination = Source)
#     V_up    = V_up.at[:, 1:, :].set(V[:, :-1, :])
#     V_down  = V_down.at[:, :-1, :].set(V[:, 1:, :])
#     V_left  = V_left.at[:, :, 1:].set(V[:, :, :-1])
#     V_right = V_right.at[:, :, :-1].set(V[:, :, 1:])

#     # 3. Stack into a 4D Tensor
#     # V_all_actions becomes a 4D Tensor with the shape:
#     # (1000 mazes, 16 rows, 16 cols, 4 actions)
#     V_all_actions = np.stack([V_up, V_down, V_left, V_right], axis=-1)

#     # 4. Find the best action value for every single cell
#     best_next_V = np.max(V_all_actions, axis=-1)

#     # 5. Bellman Equation: Value = Reward + Gamma * Max(Next State)
#     new_V = reward + gamma * best_next_V

#     # 6. Re-apply the rules (Walls are -inf, Goal is 0)
#     new_V[data == 1] = -np.inf
#     new_V[:, 15, 15] = 0.0

#     # 7. Overwrite V for the next iteration
#     V = new_V

# print("Iteration complete!")
# # Let's check a cell exactly 1 step away from the goal in the first maze!
# print("Value of cell (14, 15) in Maze 0:", V[0, 14, 15])
# Solvers 

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings

# Ignore warnings about plotting NaNs (we use NaNs to hide -inf walls)
warnings.filterwarnings("ignore")

# Import Data
data = np.load("data_jax/N16_P0100_test_solvable_random.npy")
data_jax = jnp.array(data)
print("Data Shape : ", data.shape)

# Create a 3D tensor for State-Value for each cell.
V = jnp.zeros((1000,16,16))
V = jnp.where(data_jax == 1, -jnp.inf, V)
V = V.at[:,15,15].set(0.0)  # Goal

# Constants
gamma = 1.0
reward = -1.0
iterations = 10000

# Value Iteration
def body_fun(i, V):
    # Shift matrices using roll
    V_up    = jnp.roll(V, shift=1, axis=1)
    V_down  = jnp.roll(V, shift=-1, axis=1)
    V_left  = jnp.roll(V, shift=1, axis=2)
    V_right = jnp.roll(V, shift=-1, axis=2)

    # Mask edges so rolled values don’t wrap
    V_up    = V_up.at[:,0,:].set(-jnp.inf)
    V_down  = V_down.at[:,-1,:].set(-jnp.inf)
    V_left  = V_left.at[:,:,0].set(-jnp.inf)
    V_right = V_right.at[:,:,-1].set(-jnp.inf)

    # Bellman update
    best_next_V = jnp.max(jnp.stack([V_up, V_down, V_left, V_right], axis=-1), axis=-1)
    new_V = reward + gamma * best_next_V

    # Re-apply walls and goal
    new_V = jnp.where(data_jax == 1, -jnp.inf, new_V)
    new_V = new_V.at[:,15,15].set(0.0)

    return new_V

@jax.jit
def run_value_iteration(V_init):
    return jax.lax.fori_loop(0, iterations, body_fun, V_init)

# Run Value Iteration
V = run_value_iteration(V)

# Extract Optimal Policy
V_up    = jnp.roll(V, shift=1, axis=1)
V_down  = jnp.roll(V, shift=-1, axis=1)
V_left  = jnp.roll(V, shift=1, axis=2)
V_right = jnp.roll(V, shift=-1, axis=2)

V_up    = V_up.at[:,0,:].set(-jnp.inf)
V_down  = V_down.at[:,-1,:].set(-jnp.inf)
V_left  = V_left.at[:,:,0].set(-jnp.inf)
V_right = V_right.at[:,:,-1].set(-jnp.inf)

V_all_actions = jnp.stack([V_up, V_down, V_left, V_right], axis=-1)
optimal_policy = jnp.argmax(V_all_actions, axis=-1)

print("Policy Shape:", optimal_policy.shape)
print(optimal_policy[0])

# ----------------- Visualization -----------------
arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

fig, axes = plt.subplots(1,3,figsize=(18,6))
plt.subplots_adjust(bottom=0.2)

def update(val):
    idx = int(slider.val)

    for ax in axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot 1: Raw Maze
    axes[0].imshow(data[idx], cmap='gray_r')
    axes[0].set_title(f"Maze {idx} - Raw Data", fontsize=14)
    axes[0].text(0,0,"S", color='green', ha='center', va='center', fontsize=16, fontweight='bold')
    axes[0].text(15,15,"G", color='red', ha='center', va='center', fontsize=16, fontweight='bold')

    # Plot 2: Value Function
    v_maze = np.array(V[idx])
    v_maze[np.isinf(v_maze)] = np.nan
    axes[1].imshow(v_maze, cmap='viridis')
    axes[1].set_title(f"Maze {idx} - Optimal Values (V*)", fontsize=14)

    # Plot 3: Optimal Policy
    axes[2].imshow(data[idx], cmap='gray_r', alpha=0.3)
    axes[2].set_title(f"Maze {idx} - Optimal Policy (π*)", fontsize=14)

    for r in range(16):
        for c in range(16):
            if r == 15 and c == 15:
                axes[2].text(c,r,"★", color='gold', ha='center', va='center', fontsize=18)
            elif data[idx,r,c] == 0:
                pol = optimal_policy[idx,r,c]
                axes[2].text(c,r,arrow_map[int(pol)], color='blue', ha='center', va='center', fontsize=12)

    fig.canvas.draw_idle()

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, label='Maze Index', valmin=0, valmax=999, valinit=0, valstep=1)
slider.on_changed(update)
update(0)
plt.show()