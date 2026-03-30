import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# --- 1. PARAMETERS ---
GRID_SIZE = 16
N_META_ACTIONS = 2 
N_META_STATES = 4608
EPISODES = 5000000        
WARMUP_EPS = 2000000      
NUM_MAZES_TRAIN = 4500
EPS_PER_MAZE = 800        
MAX_STEPS = 1024          
GOAL = jnp.array([15, 15])
ALPHA = 0.1
GAMMA = 0.99

# --- 2. LOAD SUB-POLICIES ---
# Ensure these files exist in your Artifacts folder
ego_q_sub = jnp.array(np.load("../Alios/Artifacts/EGO_REACTIVE_SURVIVOR_512_policy.npy"))
geo_q_sub = jnp.array(np.load("../Alios/Artifacts/PURE_GEO_ANGLE_policy.npy"))

# --- 3. JAX CORE LOGIC ---

@jit
def get_meta_state(maze, pos):
    r, c = pos
    padded = jnp.pad(maze, 1, constant_values=1)
    window = lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    win_id = jnp.sum(window.astype(jnp.int32) * jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1]))
    dr, dc = jnp.sign(15 - r), jnp.sign(15 - c)
    comp_id = (dr + 1) * 3 + (dc + 1)
    return (win_id * 9 + comp_id).astype(jnp.int32)

@jit
def get_physical_move(maze, pos, meta_choice):
    a_geo = jnp.argmax(geo_q_sub[pos[0] * 16 + pos[1]])
    padded = jnp.pad(maze, 1, constant_values=1)
    window = lax.dynamic_slice(padded, (pos[0], pos[1]), (3, 3)).flatten()
    win_id = jnp.sum(window.astype(jnp.int32) * jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1]))
    a_ego = jnp.argmax(ego_q_sub[win_id])
    return jnp.where(meta_choice == 1, a_geo, a_ego)

@jit
def step_env(maze, pos, meta_choice):
    move = get_physical_move(maze, pos, meta_choice)
    deltas = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    new_pos = pos + deltas[move]
    out = (new_pos[0]<0)|(new_pos[0]>=16)|(new_pos[1]<0)|(new_pos[1]>=16)
    safe_rc = jnp.clip(new_pos, 0, 15)
    hit = out | (maze[safe_rc[0], safe_rc[1]] == 1)
    
    fr, fc = jnp.where(hit, pos[0], new_pos[0]), jnp.where(hit, pos[1], new_pos[1])
    actual_pos = jnp.array([fr, fc])
    is_goal = (fr == 15) & (fc == 15)
    
    # Dung Beetle Shaping
    old_dist = jnp.sum(jnp.abs(pos - GOAL))
    new_dist = jnp.sum(jnp.abs(actual_pos - GOAL))
    progress_reward = (old_dist - new_dist) * 2.0 - 0.2
    reward = jnp.where(is_goal, 100.0, jnp.where(hit, -10.0, progress_reward))
    return actual_pos, reward, is_goal

# --- 4. TRAINING ENGINE ---

@jit
def train_episode(carry, episode_idx):
    q_table, rng, mazes = carry
    key, k_maze, k_l = jax.random.split(rng, 3)

    # Curriculum Logic
    ordered_end = NUM_MAZES_TRAIN * EPS_PER_MAZE
    m_idx = jnp.where(episode_idx < ordered_end, (episode_idx // EPS_PER_MAZE) % NUM_MAZES_TRAIN, jax.random.randint(k_maze, (), 0, NUM_MAZES_TRAIN))
    maze = mazes[m_idx]

    # Linear Epsilon
    eps_decay = 1.0 - (1.0 - 0.01) * (episode_idx - WARMUP_EPS) / (EPISODES - WARMUP_EPS)
    epsilon = jnp.where(episode_idx < WARMUP_EPS, 1.0, jnp.maximum(0.01, eps_decay))

    def body_fun(state):
        p, q, stp, done, total_loss, k = state
        k, k1, k2 = jax.random.split(k, 3)
        sid = get_meta_state(maze, p)
        a_meta = jnp.where(jax.random.uniform(k1) < epsilon, jax.random.randint(k2, (), 0, 2), jnp.argmax(q[sid]))
        
        next_p, rew, is_done = step_env(maze, p, a_meta)
        nsid = get_meta_state(maze, next_p)
        
        target = rew + GAMMA * jnp.where(is_done, 0.0, jnp.max(q[nsid]))
        td_error = target - q[sid, a_meta]
        q = q.at[sid, a_meta].add(ALPHA * td_error)
        
        return next_p, q, stp + 1, is_done | (stp > MAX_STEPS), total_loss + jnp.abs(td_error), k

    init = (jnp.array([0,0]), q_table, 0, False, 0.0, k_l)
    final = lax.while_loop(lambda s: ~s[3], body_fun, init)
    
    success = jnp.where((final[0][0] == 15) & (final[0][1] == 15), 1.0, 0.0)
    avg_loss = final[4] / (final[2] + 1)
    return (final[1], key, mazes), (success, avg_loss)

# --- 5. EXECUTION ---

def run_main():
    train_data = jnp.array(np.load("data_jax/N16_P0100_train_solvable.npy"))
    q_table = jnp.zeros((N_META_STATES, N_META_ACTIONS)) + 5.0 
    key = jax.random.PRNGKey(42)

    print(f"🚀 Training for {EPISODES} episodes...")
    start_t = time.time()
    
    # Store metrics for final plotting
    history_success = []
    history_loss = []
    
    chunk_size = 500000
    for chunk in range(EPISODES // chunk_size):
        chunk_range = jnp.arange(chunk * chunk_size, (chunk + 1) * chunk_size)
        (q_table, key, _), (successes, losses) = lax.scan(train_episode, (q_table, key, train_data), chunk_range)
        
        chunk_succ = float(jnp.mean(successes))
        chunk_loss = float(jnp.mean(losses))
        history_success.append(chunk_succ)
        history_loss.append(chunk_loss)
        
        print(f"Chunk {chunk+1}/{EPISODES // chunk_size} | Success: {chunk_succ*100:.1f}% | Loss: {chunk_loss:.4f}")

    print(f"✅ Total Time: {time.time()-start_t:.2f}s")
    np.save('META_SWITCHER_5M.npy', np.array(q_table))

    # --- 6. PLOT ANALYTICS ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(history_success, marker='o', color='green')
    ax1.set_title("Training Success Rate (per 500k eps)")
    ax1.set_ylabel("Success Rate"); ax1.grid(alpha=0.3)
    
    ax2.plot(history_loss, marker='x', color='red')
    ax2.set_title("TD-Error Loss (per 500k eps)")
    ax2.set_ylabel("Mean Abs TD Error"); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # --- 7. INTERACTIVE TESTER ---
    test_mazes = np.load("data_jax/N16_P0100_test_solvable_random.npy")
    current_idx = [0] # List for mutability in callback

    def rollout(idx):
        maze = test_mazes[idx]
        pos, path, modes = jnp.array([0, 0]), [(0,0)], []
        for _ in range(MAX_STEPS):
            sid = get_meta_state(maze, pos)
            mode = int(np.argmax(q_table[sid]))
            move = int(get_physical_move(maze, pos, mode))
            pos, _, done = step_env(maze, pos, mode)
            path.append(tuple(map(int, pos))); modes.append(mode)
            if done: break
        return np.array(path), np.array(modes)

    viz_fig, viz_ax = plt.subplots(figsize=(7,7))
    def redraw():
        viz_ax.clear()
        p, m = rollout(current_idx[0])
        viz_ax.imshow(test_mazes[current_idx[0]], cmap='gray_r')
        for i in range(len(p)-1):
            color = 'cyan' if m[i] == 0 else 'red'
            viz_ax.plot(p[i:i+2, 1], p[i:i+2, 0], color=color, linewidth=2)
        success = "✓" if (p[-1] == [15,15]).all() else "✗"
        viz_ax.set_title(f"Interactive Test | Maze {current_idx[0]} | {success}\nRED: Geo | CYAN: Ego")
        plt.draw()

    def on_key(event):
        if event.key == 'right': current_idx[0] = (current_idx[0] + 1) % 100
        elif event.key == 'left': current_idx[0] = (current_idx[0] - 1) % 100
        redraw()

    viz_fig.canvas.mpl_connect('key_press_event', on_key)
    redraw()
    print("\n🎮 Use LEFT/RIGHT ARROWS to verify mazes.")
    plt.show()

if __name__ == "__main__":
    run_main()