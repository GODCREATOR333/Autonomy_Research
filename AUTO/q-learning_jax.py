import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.lax as lax
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize
# Change from (256, 4) to (1024, 4)
# Change from 1024 to 26244
q_table = jnp.full((26244, 4), 150.0)
rng_key = jrandom.PRNGKey(42)

epsilon = 1.0
min_epsilon = 0.01
decay = 0.9997

print("Starting JAX Q-Learning...")
start_time = time.perf_counter()

data = np.load("data_jax/N16_P0100_test_solvable_random.npy")
maze = jnp.array(data[0])


@jax.jit
def get_state(maze, r, c):
    # 1. Inject the Goal into the maze as a '2'
    # (Assuming goal is always at 15, 15)
    maze_with_goal = maze.at[15, 15].set(2)
    
    # 2. Pad the maze with walls (1s) to handle borders
    padded_maze = jnp.pad(maze_with_goal, pad_width=1, mode='constant', constant_values=1)
    
    # 3. Extract 3x3 window 
    window = jax.lax.dynamic_slice(padded_maze, (r, c), (3, 3))
    
    # 4. Flatten and remove the center (index 4)
    flat = window.flatten()
    neighbors = jnp.concatenate([flat[:4], flat[5:]])
    
    # 5. Base-3 to Integer (0 to 6560) using powers of 3
    # 3^7, 3^6, 3^5, 3^4, 3^3, 3^2, 3^1, 3^0
    powers = jnp.array([2187, 729, 243, 81, 27, 9, 3, 1])
    local_id = jnp.sum(neighbors * powers)
    
    # 6. Compass (0 to 3)
    dr = jnp.where(r < 15, 1, 0)
    dc = jnp.where(c < 15, 1, 0)
    compass_id = dr * 2 + dc
    
    # Return unique ID (0 to 26243)
    return (local_id * 4 + compass_id).astype(jnp.int32)

@jax.jit
def step(maze, r, c, last_r, last_c, action):
    # 1. Map action
    dr_array = jnp.array([-1, 1, 0, 0])
    dc_array = jnp.array([0, 0, -1, 1])
    
    dr = dr_array[action]
    dc = dc_array[action]
    
    next_r = r + dr
    next_c = c + dc
    
    # 2. Validity Check
    out_of_bounds = (next_r < 0) | (next_r >= 16) | (next_c < 0) | (next_c >= 16)
    safe_r = jnp.clip(next_r, 0, 15)
    safe_c = jnp.clip(next_c, 0, 15)
    hit_wall = maze[safe_r, safe_c] == 1
    
    is_invalid = out_of_bounds | hit_wall
    
    final_r = jnp.where(is_invalid, r, next_r)
    final_c = jnp.where(is_invalid, c, next_c)
    
    # 3. Base Reward
    is_goal = (final_r == 15) & (final_c == 15)
    base_reward = jnp.where(is_goal, 100.0, 
                            jnp.where(is_invalid, -10.0, -1.0))
    
    # 4. Dense Reward Shaping (Distance)
    old_dist = jnp.abs(r - 15) + jnp.abs(c - 15)
    new_dist = jnp.abs(final_r - 15) + jnp.abs(final_c - 15)
    
    # If closer: +0.5. If further: -0.5. If hit wall (no change): 0.0
    dist_reward = jnp.where(new_dist < old_dist, 0.5, 
                            jnp.where(new_dist > old_dist, -0.5, 0.0))
    
    # 5. Anti-Oscillation Penalty (U-Turn Check)
    # If final position is exactly where we were last step (and we didn't just bounce off a wall)
    is_u_turn = (final_r == last_r) & (final_c == last_c) & jnp.logical_not(is_invalid)
    oscillation_penalty = jnp.where(is_u_turn, -1.0, 0.0)
    
    # Combine all rewards
    total_reward = base_reward + dist_reward + oscillation_penalty
    
    return final_r, final_c, total_reward, is_goal


@jax.jit
def get_action(q_table, state, epsilon, key):
    # 1. SPLIT THE KEY
    # We need 2 random numbers (one for the coin flip, one for the random action).
    # We split the main key into 3 parts: 2 for us to use now, and 1 to return for later!
    key, subkey_coin, subkey_action = jrandom.split(key, 3)
    
    # 2. The Coin Flip (Float between 0.0 and 1.0)
    coin_flip = jrandom.uniform(subkey_coin)
    
    # 3. The Random Action (Int between 0 and 3)
    # shape=() means it returns a single scalar number, not an array
    random_action = jrandom.randint(subkey_action, shape=(), minval=0, maxval=4)
    
    # 4. The Greedy Action (Best known action)
    greedy_action = jnp.argmax(q_table[state])
    
    # 5. The Decision (No Python 'if' statements allowed!)
    # If coin_flip < epsilon, return random_action. Else, return greedy_action.
    chosen_action = jnp.where(coin_flip < epsilon, random_action, greedy_action)
    
    # We MUST return the new key so the next step of the loop has fresh randomness!
    return chosen_action, key


# Constants for the math
ALPHA = 0.1
GAMMA = 0.99
MAX_STEPS = 500

@jax.jit
def play_episode(q_table, maze, epsilon, key):
    
    # 1. The Condition Function
    def cond_fun(backpack):
        q, r, c, last_r, last_c, steps, done, k = backpack
        # Stop if done is True OR steps reach 1000
        return jnp.logical_not(done) & (steps < 1000)

    # 2. The Body Function (One Step of the Game)
    def body_fun(backpack):
        q, r, c, last_r, last_c, steps, done, k = backpack
        
        
        state = get_state(maze, r, c)
        
        # Get Action
        action, k = get_action(q, state, epsilon, k)
        
        # Take Step
        next_r, next_c, reward, next_done = step(maze, r, c, last_r, last_c, action)
        
        next_state = get_state(maze, next_r, next_c)
        
        best_next_q = jnp.where(next_done, 0.0, jnp.max(q[next_state]))
        td_target = reward + 0.99 * best_next_q
        
        new_q_val = q[state, action] + 1.0 * (td_target - q[state, action])
        q = q.at[state, action].set(new_q_val)
        
        # --- REPACK BACKPACK (current r, c become the new last_r, last_c) ---
        return (q, next_r, next_c, r, c, steps + 1, next_done, k)

    # 3. Initialize the Backpack for Step 0
    # Start at r=0, c=0, steps=0, done=False
    init_backpack = (q_table, 0, 0, 0, 0, 0, False, key)
    
   # 4. Run the loop
    final_backpack = jax.lax.while_loop(cond_fun, body_fun, init_backpack)
    
    # 5. Unpack
    final_q_table, final_r, final_c, final_lr, final_lc, total_steps, final_done, final_key = final_backpack
    
    return final_q_table, final_key

# Play 20,000 episodes
for episode in range(50000):
    # Decay epsilon in Python
    epsilon = max(min_epsilon, epsilon * decay)
    
    # Play one full episode at lightning speed
    q_table, rng_key = play_episode(q_table, maze, epsilon, rng_key)

end_time = time.perf_counter()
print(f"Finished 20,000 episodes in {end_time - start_time:.4f} seconds!")

# Sanity Check
print("Sum of Q-table:", jnp.sum(q_table))

# ------------------------------
# Visualization of Learned Policy
# ------------------------------

print("Plotting the results...")

# Map our action indices back to arrows
# Actions: 0=Up, 1=Down, 2=Left, 3=Right
arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("JAX Q-Learning Results", fontsize=20, fontweight='bold')

# --- Plot 1: Raw Maze ---
axes[0].imshow(maze, cmap='gray_r')  # 0=open (white), 1=wall (black)
axes[0].set_title("Environment", fontsize=16)
axes[0].text(0, 0, "S", color='green', ha='center', va='center', fontsize=18, fontweight='bold')
axes[0].text(15, 15, "G", color='red', ha='center', va='center', fontsize=18, fontweight='bold')
axes[0].set_xticks([])
axes[0].set_yticks([])

# --- Plot 2: Learned Policy ---
axes[1].imshow(maze, cmap='gray_r', alpha=0.3) # Faded background
axes[1].set_title("Optimal Policy (Greedy Actions)", fontsize=16)
axes[1].set_xticks([])
axes[1].set_yticks([])

# Loop through every cell to extract the best action from the Q-table
for r in range(16):
    for c in range(16):
        if (r, c) == (15, 15):
            axes[1].text(c, r, "★", color='gold', ha='center', va='center', fontsize=22)
        elif maze[r, c] == 1:
            pass
        else:
            # --- NEW STATE CALL ---
            state = get_state(maze, r, c)
            best_action = int(jnp.argmax(q_table[state]))
            axes[1].text(c, r, arrow_map[best_action], color='blue', ha='center', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.9) # Adjust for the main title
plt.show()


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

print("Launching Interactive Aliasing Probe...")

# 1. Fast Standalone Value Iteration (For God View)
@jax.jit
def solve_vi_for_maze(maze):
    V = jnp.zeros((16, 16))
    V = jnp.where(maze == 1, -jnp.inf, V)
    V = V.at[15, 15].set(0.0)

    def vi_step(i, V_curr):
        V_up = jnp.pad(V_curr[:-1, :], ((1,0), (0,0)), constant_values=-jnp.inf)
        V_down = jnp.pad(V_curr[1:, :], ((0,1), (0,0)), constant_values=-jnp.inf)
        V_left = jnp.pad(V_curr[:, :-1], ((0,0), (1,0)), constant_values=-jnp.inf)
        V_right = jnp.pad(V_curr[:, 1:], ((0,0), (0,1)), constant_values=-jnp.inf)
        
        best_v = jnp.max(jnp.stack([V_up, V_down, V_left, V_right], axis=-1), axis=-1)
        new_V = -1.0 + 0.99 * best_v
        new_V = jnp.where(maze == 1, -jnp.inf, new_V)
        return new_V.at[15, 15].set(0.0)

    V_final = jax.lax.fori_loop(0, 500, vi_step, V)
    
    # Extract Policy
    V_up = jnp.pad(V_final[:-1, :], ((1,0), (0,0)), constant_values=-jnp.inf)
    V_down = jnp.pad(V_final[1:, :], ((0,1), (0,0)), constant_values=-jnp.inf)
    V_left = jnp.pad(V_final[:, :-1], ((0,0), (1,0)), constant_values=-jnp.inf)
    V_right = jnp.pad(V_final[:, 1:], ((0,0), (0,1)), constant_values=-jnp.inf)
    
    return jnp.argmax(jnp.stack([V_up, V_down, V_left, V_right], axis=-1), axis=-1)


# 2. Probe UI Setup
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(bottom=0.2)
arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

# Global variables to store UI state
highlight_boxes =[]
current_state_matrix = np.zeros((16, 16), dtype=int)
current_maze_idx = 0

def draw_maze(idx):
    global current_state_matrix, highlight_boxes
    current_maze_idx = idx
    maze_layout = data[idx]
    
    # Clear old plots
    for ax in axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(maze_layout, cmap='gray_r', alpha=0.4)
        ax.text(0, 0, "S", color='green', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(15, 15, "G", color='red', ha='center', va='center', fontsize=16, fontweight='bold')

    axes[0].set_title(f"Value Iteration (God View) - Maze {idx}", fontsize=14)
    axes[1].set_title(f"Q-Learning (POMDP) - Maze {idx}\nCLICK a cell to probe Aliasing!", fontsize=14)

    # Calculate VI Policy
    vi_policy = solve_vi_for_maze(jnp.array(maze_layout))
    
    # Draw Arrows
    for r in range(16):
        for c in range(16):
            if (r, c) == (15, 15) or maze_layout[r, c] == 1:
                current_state_matrix[r, c] = -1 # Ignore walls/goal
                continue
                
            # VI Arrow (Left)
            axes[0].text(c, r, arrow_map[int(vi_policy[r, c])], color='darkgreen', ha='center', va='center', fontsize=12)
            
            # Q-Learning Arrow (Right)
            state_id = int(get_state(jnp.array(maze_layout), r, c))
            current_state_matrix[r, c] = state_id
            best_action = int(jnp.argmax(q_table[state_id]))
            
            # Color code: Red if POMDP disagrees with God View, Blue if they agree
            color = 'blue' if best_action == int(vi_policy[r, c]) else 'red'
            axes[1].text(c, r, arrow_map[best_action], color=color, ha='center', va='center', fontsize=12, fontweight='bold')

    highlight_boxes =[] # Reset highlights
    fig.canvas.draw_idle()

# 3. Interactive Click Event for Aliasing
def on_click(event):
    global highlight_boxes
    
    # Only trigger if clicking on the Right panel (Q-Learning)
    if event.inaxes == axes[1]:
        # Get rounded grid coordinates
        c, r = int(round(event.xdata)), int(round(event.ydata))
        
        # Check bounds
        if 0 <= r < 16 and 0 <= c < 16 and current_state_matrix[r, c] != -1:
            clicked_state_id = current_state_matrix[r, c]
            
            # Clear old red boxes
            for box in highlight_boxes:
                box.remove()
            highlight_boxes.clear()
            
            # Find ALL cells in this maze with the exact same State ID
            aliased_cells = np.argwhere(current_state_matrix == clicked_state_id)
            
            for (ar, ac) in aliased_cells:
                # Draw a bright red box around aliased cells
                rect = patches.Rectangle((ac - 0.5, ar - 0.5), 1, 1, linewidth=3, edgecolor='red', facecolor='none')
                axes[1].add_patch(rect)
                highlight_boxes.append(rect)
                
            axes[1].set_title(f"State ID: {clicked_state_id} | Aliased Locations: {len(aliased_cells)}", fontsize=14, color='red')
            fig.canvas.draw_idle()

# 4. Slider Setup
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Maze', 0, 999, valinit=0, valstep=1)

slider.on_changed(lambda val: draw_maze(int(val)))
fig.canvas.mpl_connect('button_press_event', on_click)

draw_maze(0)
plt.show()