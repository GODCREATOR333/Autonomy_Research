import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the Two "Pure" Brains ---
# Using the files we registered in Alios
geo_q = jnp.array(np.load("../Alios/Artifacts/PURE_GEO_ANGLE_policy.npy"))
ego_q = jnp.array(np.load("../Alios/Artifacts/EGO_REACTIVE_SURVIVOR_512_policy.npy"))

@jit
def get_mdp_state(pos):
    return pos[0] * 16 + pos[1]

@jit
def get_ego_state(maze, pos):
    padded = jnp.pad(maze, 1, constant_values=1)
    window = lax.dynamic_slice(padded, (pos[0], pos[1]), (3, 3)).flatten()
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    return jnp.sum(window.astype(jnp.int32) * powers)

@jit
def dmp_step(maze, pos, mode, timer, key):
    """
    DMP logic:
    Mode 0 (Geo): Ballistic pursuit.
    Mode 1 (Ego): Stochastic avoidance (Tumble).
    """
    key, k1, k2 = jax.random.split(key, 3)
    
    # --- POLICY SELECT ---
    # Geo Action: Greedy
    a_geo = jnp.argmax(geo_q[get_mdp_state(pos)])
    
    # Ego Action: Stochastic (Softmax over Ego Q-values)
    # This turns 'Avoidance' into a 'Random walk that hates walls'
    q_ego = ego_q[get_ego_state(maze, pos)]
    probs = jax.nn.softmax(q_ego * 1.0) # Temperature = 1.0
    a_ego = jax.random.choice(k1, jnp.arange(4), p=probs)
    
    # Choose action based on mode
    action = jnp.where(mode == 0, a_geo, a_ego)
    
    # --- PHYSICS ---
    deltas = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    new_pos = pos + deltas[action]
    out = (new_pos[0]<0)|(new_pos[0]>=16)|(new_pos[1]<0)|(new_pos[1]>=16)
    safe_rc = jnp.clip(new_pos, 0, 15)
    hit = out | (maze[safe_rc[0], safe_rc[1]] == 1)
    
    # Final resolution
    actual_pos = jnp.where(hit, pos, new_pos)
    
    # --- DMP SWITCHING LOGIC ---
    # 1. Trigger: If Geo hits a wall, switch to Ego (Tumble)
    switch_to_ego = (mode == 0) & hit
    
    # 2. Timer: Stay in Ego mode for 15 steps to clear the obstacle
    new_mode = jnp.where(switch_to_ego, 1, mode)
    new_timer = jnp.where(switch_to_ego, 15, timer - 1)
    
    # 3. Reset: When timer hits 0, switch back to Geo (Run)
    final_mode = jnp.where(new_timer <= 0, 0, new_mode)
    final_timer = jnp.where(new_timer <= 0, 0, new_timer)
    
    return actual_pos, final_mode, final_timer, key

# --- Interactive Visualizer ---
test_mazes = np.load("data_jax/N16_P0100_test_solvable_random.npy")
current_idx = 0

def run_simulation(idx):
    maze = test_mazes[idx]
    pos = jnp.array([0, 0])
    mode, timer = 0, 0
    key = jax.random.PRNGKey(idx)
    path = [tuple(pos)]
    modes = [0]
    
    for _ in range(1000): # Allow time for search
        pos, mode, timer, key = dmp_step(maze, pos, mode, timer, key)
        path.append(tuple(map(int, pos)))
        modes.append(int(mode))
        if path[-1] == (15, 15): break
    return np.array(path), np.array(modes)

fig, ax = plt.subplots(figsize=(8,8))

def update():
    ax.clear()
    path, modes = run_simulation(current_idx)
    ax.imshow(test_mazes[current_idx], cmap='gray_r')
    
    # Plotting logic: Red = Geo (Run), Cyan = Ego (Tumble)
    for i in range(len(path)-1):
        color = 'red' if modes[i] == 0 else 'cyan'
        ax.plot(path[i:i+2, 1], path[i:i+2, 0], color=color, linewidth=2, alpha=0.8)
    
    success = "SUCCESS" if tuple(path[-1]) == (15, 15) else "FAILED"
    ax.set_title(f"DMP Switcher | Maze {current_idx} | {success}\nRed: Geo (Run) | Cyan: Ego (Stochastic Tumble)")
    plt.draw()

def on_key(event):
    global current_idx
    if event.key == 'right': current_idx = (current_idx + 1) % 100
    elif event.key == 'left': current_idx = (current_idx - 1) % 100
    update()

fig.canvas.mpl_connect('key_press_event', on_key)
update()
plt.show()