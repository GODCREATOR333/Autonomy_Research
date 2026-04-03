import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# 1. PHYSICS & RL CONFIGURATION
# =========================================================
KAPPA = 10.0           # Higher Kappa = Softmax logic with near-zero noise
D_R = 0.15             
P_STRAIGHT = jnp.exp(-D_R)
P_TURN = (1.0 - P_STRAIGHT) / 2.0

Q_TABLE_SIZE = 11520 
DENSITIES = ["10%", "20%", "30%", "35%", "37%", "40%"]
TAGS = ["P0100", "P0200", "P0300", "P0350", "P0370", "P0400"]

# Curriculum Hyperparams
EPISODES_PER_STAGE = 100000 
ALPHA, GAMMA = 0.1, 0.99
EPS_START, EPS_MIN, EPS_DECAY = 1.0, 0.01, 0.99996

# =========================================================
# 2. PHYSICS ENGINES (JAX JIT)
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
    # High collision penalty to discourage GEO bumping
    reward = jnp.where(is_goal, 100.0, jnp.where(invalid, -10.0, -1.0))
    return fr, fc, a, reward, is_goal, key

# =========================================================
# 3. TRAINING LOOP WRAPPERS
# =========================================================

@jax.jit
def train_loop(q, mazes, eps, k):
    k, sk1 = jrandom.split(k)
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
    res = jax.lax.while_loop(lambda s: (~s[5]) & (s[4] < 400), body, (q, 0, 0, 4, 0, False, k))
    return res[0], res[6]

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
# 4. EXECUTION: CURRICULUM TRAINING
# =========================================================
q_table = jnp.full((Q_TABLE_SIZE, 2), -50.0)
rng_key = jrandom.PRNGKey(42)
epsilon = EPS_START

print("Starting Curriculum Training Sequence...")
t_wall = time.perf_counter()

for i, tag in enumerate(TAGS):
    print(f"\n[Stage {i+1}/6] Training on {DENSITIES[i]} Density...")
    train_data = jnp.array(np.load(f"data_jax/N16_{tag}_train_solvable.npy"))
    
    # "Re-heat" epsilon slightly for new density to encourage exploration of new corners
    if i > 0: epsilon = max(epsilon, 0.25)

    for ep in range(EPISODES_PER_STAGE):
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        q_table, rng_key = train_loop(q_table, train_data, epsilon, rng_key)
        if ep % 10000 == 0: print(f"  Episode {ep}...")

print(f"\nCurriculum Complete in {time.perf_counter()-t_wall:.2f}s")

# =========================================================
# 5. FINAL EVALUATION (GENERALIZATION)
# =========================================================
print("\n--- FINAL TEST RESULTS (MASTER AGENT) ---")
for tag, name in zip(TAGS, DENSITIES):
    test_mazes = jnp.array(np.load(f"data_jax/N16_{tag}_test_solvable_random.npy"))
    keys = jrandom.split(rng_key, len(test_mazes))
    succ, steps, ego = evaluate_batch(test_mazes, q_table, keys)
    print(f"[{name}] Success: {jnp.mean(succ)*100:.1f}% | MFPT: {jnp.mean(steps[succ]):.1f} | EGO Usage: {jnp.mean(ego/steps)*100:.1f}%")

# =========================================================
# 6. INTERACTIVE VIEWER
# =========================================================
current_maze_idx = 0
active_set = jnp.array(np.load("data_jax/N16_P0400_test_solvable_random.npy"))
fig, ax = plt.subplots(figsize=(8, 8))

def update_plot():
    ax.clear()
    maze = active_set[current_maze_idx]
    
    # Rollout
    r, c, la = 0, 0, 4
    path, modes, actions = [(0,0)], [], []
    k = jrandom.PRNGKey(99)
    for _ in range(400):
        state = get_meta_state(maze, r, c, la)
        meta_a = int(jnp.argmax(q_table[state]))
        nr, nc, na, _, d, k = env_step(maze, r, c, la, meta_a, k)
        path.append((int(nr), int(nc))); modes.append(meta_a); actions.append(int(na))
        r, c, la = nr, nc, na
        if d: break
    path = np.array(path); modes = np.array(modes); actions = np.array(actions)
    
    ax.imshow(maze, cmap='gray_r', alpha=0.2, extent=[-0.5, 15.5, 15.5, -0.5])
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True); ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    for i in range(len(modes)):
        color = 'blue' if modes[i] == 0 else 'red'
        ax.plot(path[i:i+2, 1], path[i:i+2, 0], color=color, lw=3)
        dr, dc = ([-0.3, 0.3, 0, 0][actions[i]], [0, 0, -0.3, 0.3][actions[i]])
        ax.arrow(path[i, 1], path[i, 0], dc, dr, head_width=0.2, head_length=0.2, fc=color, ec=color)

    ax.plot(0, 0, 'go'); ax.plot(15, 15, 'y*', ms=15)
    ax.set_title(f"Master Agent | {DENSITIES[-1]} Maze #{current_maze_idx}\nSteps: {len(modes)} | EGO Usage: {(np.sum(modes==1)/len(modes))*100:.1f}%")
    plt.draw()

def on_key(event):
    global current_maze_idx
    if event.key == 'right': current_maze_idx = (current_maze_idx + 1) % 1000
    elif event.key == 'left': current_maze_idx = (current_maze_idx - 1) % 1000
    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot()
plt.show()