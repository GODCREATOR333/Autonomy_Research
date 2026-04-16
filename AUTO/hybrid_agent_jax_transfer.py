import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# 1. CONFIGURATION
# =========================================================
DENSITIES = ["10%", "20%", "30%", "35%", '37%' "40%"]
TAGS = ["P0100", "P0200", "P0300","P0350","P0370" "P0400"]

KAPPA = 2.5
D_R = 0.15
P_STRAIGHT = jnp.exp(-D_R)
P_TURN = (1.0 - P_STRAIGHT) / 2.0

Q_TABLE_SIZE = 11520 
TRAIN_EPISODES = 60000  # Balanced for speed and convergence
ALPHA, GAMMA = 0.1, 0.98
EPS_START, EPS_MIN, EPS_DECAY = 1.0, 0.05, 0.99995

# =========================================================
# 2. CORE ENGINES (JIT COMPLIED)
# =========================================================

@jax.jit
def get_meta_state(maze, r, c, last_action):
    padded = jnp.pad(maze, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    neighbors = jnp.concatenate([window[:4], window[5:]])
    window_id = jnp.sum(neighbors * jnp.array([128, 64, 32, 16, 8, 4, 2, 1]))
    dr_sign, dc_sign = jnp.sign(15 - r) + 1, jnp.sign(15 - c) + 1
    goal_id = dr_sign * 3 + dc_sign
    return (window_id * 45 + goal_id * 5 + last_action).astype(jnp.int32)

@jax.jit
def get_geo_probs(r, c):
    vr, vc = 15.0 - r, 15.0 - c
    norm = jnp.sqrt(vr**2 + vc**2) + 1e-5
    q = jnp.array([-1, 1, 0, 0])*(vr/norm) + jnp.array([0, 0, -1, 1])*(vc/norm)
    return jax.nn.softmax(KAPPA * q)

@jax.jit
def get_ego_probs(maze, r, c, last_a):
    padded = jnp.pad(maze, 1, constant_values=1)
    is_w = jnp.array([padded[r, c+1], padded[r+2, c+1], padded[r+1, c], padded[r+1, c+2]])
    base = jnp.where(last_a == 4, jnp.full(4, 0.25), jnp.full(4, P_TURN))
    safe_la = jnp.clip(last_a, 0, 3)
    base = jnp.where(last_a < 4, base.at[safe_la].set(P_STRAIGHT), base)
    opp = jnp.array([1, 0, 3, 2])[safe_la]
    base = jnp.where(last_a < 4, base.at[opp].set(0.0), base)
    masked = jnp.where(is_w == 1, 0.0, base)
    s = jnp.sum(masked)
    return jnp.where(s > 0, masked / (s + 1e-8), jnp.zeros(4).at[opp].set(1.0))

@jax.jit
def env_step(maze, r, c, last_a, meta_a, key):
    probs = jnp.where(meta_a == 0, get_geo_probs(r, c), get_ego_probs(maze, r, c, last_a))
    key, subkey = jrandom.split(key)
    a = jrandom.choice(subkey, 4, p=probs)
    nr, nc = r + jnp.array([-1, 1, 0, 0])[a], c + jnp.array([0, 0, -1, 1])[a]
    invalid = (nr < 0) | (nr >= 16) | (nc < 0) | (nc >= 16) | (maze[jnp.clip(nr,0,15), jnp.clip(nc,0,15)] == 1)
    fr, fc = jnp.where(invalid, r, nr), jnp.where(invalid, c, nc)
    is_goal = (fr == 15) & (fc == 15)
    reward = jnp.where(is_goal, 100.0, jnp.where(invalid, -10.0, -1.0)) # Higher penalty for wall
    return fr, fc, a, reward, is_goal, key

# =========================================================
# 3. TRAINING & EVALUATION WRAPPERS
# =========================================================

@jax.jit
def train_step(q, mazes, eps, k):
    k, sk1, sk2 = jrandom.split(k, 3)
    maze = mazes[jrandom.randint(sk1, (), 0, len(mazes))]
    def body(s):
        q, r, c, la, st, done, k = s
        state = get_meta_state(maze, r, c, la)
        k, sk = jrandom.split(k)
        meta_a = jnp.where(jrandom.uniform(sk) < eps, jrandom.randint(sk, (), 0, 2), jnp.argmax(q[state]))
        nr, nc, na, rew, d, k = env_step(maze, r, c, la, meta_a, k)
        target = rew + GAMMA * jnp.where(d, 0.0, jnp.max(q[get_meta_state(maze, nr, nc, na)]))
        q = q.at[state, meta_a].set(q[state, meta_a] + ALPHA * (target - q[state, meta_a]))
        return (q, nr, nc, na, st + 1, d, k)
    return jax.lax.while_loop(lambda s: (~s[5]) & (s[4] < 300), body, (q, 0, 0, 4, 0, False, k))[0], k

@jax.jit
def evaluate_batch(mazes, q, keys):
    def run_one(maze, k):
        def body(s):
            r, c, la, st, ego_c, d, k = s
            state = get_meta_state(maze, r, c, la)
            meta_a = jnp.argmax(q[state])
            nr, nc, na, _, d, k = env_step(maze, r, c, la, meta_a, k)
            return (nr, nc, na, st + 1, ego_c + meta_a, d, k)
        res = jax.lax.while_loop(lambda s: (~s[5]) & (s[4] < 400), body, (0, 0, 4, 0, 0, False, k))
        return res[5], res[3], res[4] 
    return jax.vmap(run_one)(mazes, keys)

# =========================================================
# 4. CROSS-DENSITY SIMULATION LOOP
# =========================================================
matrix_sr = np.zeros((4, 4))
matrix_mfpt = np.zeros((4, 4))
matrix_ego = np.zeros((4, 4))

trained_q_tables = [] # Store all 4 Q-tables

for i, train_tag in enumerate(TAGS):
    print(f"\n--- Phase {i+1}/4: Training on {DENSITIES[i]} Density ---")
    train_mazes = jnp.array(np.load(f"data_jax/N16_{train_tag}_train_solvable.npy"))
    
    q_table = jnp.full((Q_TABLE_SIZE, 2), -50.0) # Optimistic init
    rng_key = jrandom.PRNGKey(i)
    
    # Train
    for ep in range(TRAIN_EPISODES):
        curr_eps = max(EPS_MIN, EPS_START * (EPS_DECAY**ep))
        q_table, rng_key = train_step(q_table, train_mazes, curr_eps, rng_key)
    
    trained_q_tables.append(q_table)
    
    # Cross-Evaluate
    for j, test_tag in enumerate(TAGS):
        test_mazes = jnp.array(np.load(f"data_jax/N16_{test_tag}_test_solvable_random.npy"))
        keys = jrandom.split(rng_key, len(test_mazes))
        
        successes, steps, ego_steps = evaluate_batch(test_mazes, q_table, keys)
        
        matrix_sr[i, j] = jnp.mean(successes)
        matrix_mfpt[i, j] = jnp.mean(steps[successes]) if jnp.any(successes) else 400
        matrix_ego[i, j] = jnp.mean(ego_steps / steps)
        print(f" Tested on {DENSITIES[j]}: SR={matrix_sr[i,j]*100:.1f}%")

# =========================================================
# 5. PLOTTING THE TRANSFER MATRICES
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Success Rate (%)", "MFPT (Steps)", "EGO Usage Fraction"]
mats = [matrix_sr * 100, matrix_mfpt, matrix_ego]
cmaps = ["RdYlGn", "viridis_r", "coolwarm"]

for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
    im = ax.imshow(mat, cmap=cmap)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels(DENSITIES)
    ax.set_yticks(range(4)); ax.set_yticklabels(DENSITIES)
    ax.set_xlabel("Test Density"); ax.set_ylabel("Train Density")
    # Annotate values
    for (r, c), val in np.ndenumerate(mat):
        ax.text(c, r, f'{val:.1f}', ha='center', va='center', color='white' if im.norm(val) < 0.5 else 'black')

plt.tight_layout()
plt.show()

# =========================================================
# 6. INTERACTIVE VIEWER (Visualizing the 40% Trained Agent)
# =========================================================
current_maze_idx = 0
active_q_table = trained_q_tables[3] # Index 3 is the 40% density table
active_maze_set = jnp.array(np.load("data_jax/N16_P0300_test_solvable_random.npy"))

fig_v, ax_v = plt.subplots(figsize=(8, 8))

def update_plot():
    ax_v.clear()
    maze = active_maze_set[current_maze_idx]
    
    # Run a greedy rollout
    r, c, la = 0, 0, 4
    path, modes, actions = [(0,0)], [], []
    k = jrandom.PRNGKey(99)
    for _ in range(400):
        state = get_meta_state(maze, r, c, la)
        meta_a = int(jnp.argmax(active_q_table[state]))
        nr, nc, na, _, d, k = env_step(maze, r, c, la, meta_a, k)
        path.append((int(nr), int(nc)))
        modes.append(meta_a); actions.append(int(na))
        r, c, la = nr, nc, na
        if d: break
    path = np.array(path); modes = np.array(modes); actions = np.array(actions)
    
    ax_v.imshow(maze, cmap='gray_r', alpha=0.2, extent=[-0.5, 15.5, 15.5, -0.5])
    ax_v.set_xticks(np.arange(-0.5, 16, 1), minor=True); ax_v.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax_v.grid(which='minor', color='w', linestyle='-', linewidth=1)

    for i in range(len(modes)):
        color = 'blue' if modes[i] == 0 else 'red'
        ax_v.plot(path[i:i+2, 1], path[i:i+2, 0], color=color, lw=3)
        dr = [-0.3, 0.3, 0, 0][actions[i]]; dc = [0, 0, -0.3, 0.3][actions[i]]
        ax_v.arrow(path[i, 1], path[i, 0], dc, dr, head_width=0.2, head_length=0.2, fc=color, ec=color)

    ax_v.plot(0, 0, 'go'); ax_v.plot(15, 15, 'r*', ms=15)
    ax_v.set_title(f"40% Trained Agent on 40% Maze #{current_maze_idx}\nSteps: {len(modes)} | EGO Usage: {(np.sum(modes==1)/len(modes))*100:.1f}%")
    plt.draw()

def on_key(event):
    global current_maze_idx
    if event.key == 'right': current_maze_idx = (current_maze_idx + 1) % 1000
    elif event.key == 'left': current_maze_idx = (current_maze_idx - 1) % 1000
    update_plot()

fig_v.canvas.mpl_connect('key_press_event', on_key)
update_plot()
plt.show()