
# Solvers 

import numpy as np

# Import Data
data = np.load("data_jax/N16_P0100_test_solvable_random.npy")
print("Data Shape : ",data.shape)
print(data[0])

# Create a 3D tensor for State-Value for each cell.
# Assign 1 to -inf so walls have value -inf and terminal 15,15 = 0 
V = np.zeros((1000,16,16))
print(V.shape)
V[data == 1] = -np.inf
V[:,15,15] = 0
print(V[0]) 


# Shifting Matrices to get next state
V_up = np.copy(V)
V_down = np.copy(V)
V_left = np.copy(V)
V_right = np.copy(V)

V_up[:, 1:, :] = V[:, :-1, :]
V_down[:, :-1, :] = V[:, 1:, :]
V_left[:, :, 1:] = V[:, :, :-1]
V_right[:, :, :-1] = V[:, :, 1:]


# Calculating 
# Set our constants
gamma = 1.0
reward = -1
for k in range(100):
    
    # 1. Setup the shifted matrices (Clone V to handle walls)
    V_up = np.copy(V)
    V_down = np.copy(V)
    V_left = np.copy(V)
    V_right = np.copy(V)

    # 2. Perform the shifts (Destination = Source)
    V_up[:, 1:, :] = V[:, :-1, :]       # Shift down (simulate moving Up)
    V_down[:, :-1, :] = V[:, 1:, :]     # Shift up (simulate moving Down)
    V_left[:, :, 1:] = V[:, :, :-1]     # Shift right (simulate moving Left)
    V_right[:, :, :-1] = V[:, :, 1:]    # Shift left (simulate moving Right)

    # 3. Stack into a 4D Tensor
    # V_all_actions becomes a 4D Tensor with the shape:
    # (1000 mazes, 16 rows, 16 cols, 4 actions)
    V_all_actions = np.stack([V_up, V_down, V_left, V_right], axis=-1)

    # 4. Find the best action value for every single cell
    best_next_V = np.max(V_all_actions, axis=-1)

    # 5. Bellman Equation: Value = Reward + Gamma * Max(Next State)
    new_V = reward + gamma * best_next_V

    # 6. Re-apply the rules (Walls are -inf, Goal is 0)
    new_V[data == 1] = -np.inf
    new_V[:, 15, 15] = 0.0

    # 7. Overwrite V for the next iteration
    V = new_V

print("Iteration complete!")
# Let's check a cell exactly 1 step away from the goal in the first maze!
print("Value of cell (14, 15) in Maze 0:", V[0, 14, 15])


# Extract Optimal Policy

# 1. Do one final shift using the fully optimized V matrix
V_up = np.copy(V)
V_down = np.copy(V)
V_left = np.copy(V)
V_right = np.copy(V)

V_up[:, 1:, :] = V[:, :-1, :]
V_down[:, :-1, :] = V[:, 1:, :]
V_left[:, :, 1:] = V[:, :, :-1]
V_right[:, :, :-1] = V[:, :, 1:]

# 2. Stack into the 4D tensor: shape (1000, 16, 16, 4)
# Index mapping: 0=Up, 1=Down, 2=Left, 3=Right
V_all_actions = np.stack([V_up, V_down, V_left, V_right], axis=-1)

# 3. Extract the optimal policy using argmax
optimal_policy = np.argmax(V_all_actions, axis=-1)

# Let's print the results for the very first maze!
print("Policy Shape:", optimal_policy.shape)
print("\nOptimal Policy for Maze 0 (0=Up, 1=Down, 2=Left, 3=Right):")
print(optimal_policy[0])


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings

# Ignore warnings about plotting NaNs (we use NaNs to hide the -inf walls)
warnings.filterwarnings("ignore")

# Map our action indices back to arrows
arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

# Create the figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(bottom=0.2)  # Make room for the slider at the bottom

def update(val):
    # Get the current slider value (maze index)
    idx = int(slider.val)
    
    # Clear previous plots
    for ax in axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        
    # --- Plot 1: Raw Data ---
    # cmap='gray_r' makes 0 (open) white and 1 (wall) black
    axes[0].imshow(data[idx], cmap='gray_r')
    axes[0].set_title(f"Maze {idx} - Raw Data", fontsize=14)
    axes[0].text(0, 0, "S", color='green', ha='center', va='center', fontsize=16, fontweight='bold')
    axes[0].text(15, 15, "G", color='red', ha='center', va='center', fontsize=16, fontweight='bold')

    # --- Plot 2: Value Function ---
    v_maze = np.copy(V[idx])
    v_maze[np.isinf(v_maze)] = np.nan  # Convert -inf to NaN so it doesn't break the colors
    # cmap='viridis' gives a cool heat map (yellow=high value/goal, purple=low value/start)
    axes[1].imshow(v_maze, cmap='viridis')
    axes[1].set_title(f"Maze {idx} - Optimal Values (V*)", fontsize=14)

    # --- Plot 3: Optimal Policy ---
    axes[2].imshow(data[idx], cmap='gray_r', alpha=0.3) # Faded background
    axes[2].set_title(f"Maze {idx} - Optimal Policy (π*)", fontsize=14)
    
    # Draw the arrows
    for r in range(16):
        for c in range(16):
            if r == 15 and c == 15:
                axes[2].text(c, r, "★", color='gold', ha='center', va='center', fontsize=18)
            elif data[idx, r, c] == 0:  # Only draw arrows on open paths
                pol = optimal_policy[idx, r, c]
                axes[2].text(c, r, arrow_map[pol], color='blue', ha='center', va='center', fontsize=12)

    fig.canvas.draw_idle()

# Create the slider axis and the slider itself
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(
    ax=ax_slider,
    label='Maze Index',
    valmin=0,
    valmax=999,
    valinit=0,
    valstep=1
)

# Link the slider to our update function
slider.on_changed(update)

# Initialize the first view
update(0)

plt.show()