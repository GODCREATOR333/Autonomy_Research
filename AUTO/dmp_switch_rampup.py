import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
import numpy as np
import time
import sys

# --- 1. PARAMETERS ---
GRID_SIZE = 16
N_META_ACTIONS = 2 
N_META_STATES = 4608
EPISODES = 500000   
MAX_STEPS = 256
ALPHA = 0.1
GAMMA = 0.99
LAMBDA = 0.8        # Trace decay
RESET_PROB = 0.01   # Stochastic reset rate (1 per 100 steps avg)
GOAL = jnp.array([15, 15])

# Load sub policies
ego_q_sub = jnp.array(np.load("../Alios/Artifacts/EGO_REACTIVE_SURVIVOR_512_policy.npy"))
geo_q_sub = jnp.array(np.load("../Alios/Artifacts/PURE_GEO_ANGLE_policy.npy"))

# --- 2. JAX CORE LOGIC ---

@jit
def get_meta_state(maze, pos):
    r, c = pos
    padded = jnp.pad(maze, 1, constant_values=1)
    window = lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1])
    win_id = jnp.sum(window.astype(jnp.int32) * powers)
    dr, dc = jnp.sign(15 - r), jnp.sign(15 - c)
    comp_id = (dr + 1) * 3 + (dc + 1)
    return (win_id * 9 + comp_id).astype(jnp.int32)

@jit
def step_env(maze, pos, meta_choice):
    # Resolve physical move
    a_geo = jnp.argmax(geo_q_sub[pos[0] * 16 + pos[1]])
    r, c = pos
    padded = jnp.pad(maze, 1, constant_values=1)
    window = lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    win_id = jnp.sum(window.astype(jnp.int32) * jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1]))
    a_ego = jnp.argmax(ego_q_sub[win_id])
    move = jnp.where(meta_choice == 1, a_geo, a_ego)

    deltas = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    new_pos = pos + deltas[move]
    out = (new_pos[0]<0)|(new_pos[0]>=16)|(new_pos[1]<0)|(new_pos[1]>=16)
    safe_rc = jnp.clip(new_pos, 0, 15)
    hit = out | (maze[safe_rc[0], safe_rc[1]] == 1)
    
    fr, fc = jnp.where(hit, pos[0], new_pos[0]), jnp.where(hit, pos[1], new_pos[1])
    actual_pos = jnp.array([fr, fc])
    is_goal = (fr == 15) & (fc == 15)
    
    # Reference Reward Shaping
    old_dist = jnp.sum(jnp.abs(pos - GOAL))
    new_dist = jnp.sum(jnp.abs(actual_pos - GOAL))
    progress_reward = (old_dist - new_dist) * 2.0 - 0.2
    reward = jnp.where(is_goal, 100.0, jnp.where(hit, -10.0, progress_reward))
    return actual_pos, reward, is_goal

# --- 3. TRAINING EPISODE ---

@jit
def train_episode(carry, episode_idx):
    q_table, rng, eps_decay = carry
    key, k_maze, k_step = jax.random.split(rng, 3)

    # Density Curriculum: 0.0 to 0.1 over 400k episodes
    p = jnp.minimum(0.10, (episode_idx / 400000.0) * 0.10)
    maze = jax.random.bernoulli(k_maze, p, shape=(16, 16)).astype(jnp.int32)
    maze = maze.at[0,0].set(0).at[15,15].set(0)

    epsilon = jnp.maximum(0.01, 1.0 * (eps_decay ** episode_idx))
    
    # Backpack: (pos, q_table, e_table, steps, done, key)
    e_init = jnp.zeros_like(q_table)
    init_val = (jnp.array([0,0]), q_table, e_init, 0, False, k_step)

    def body_fun(state):
        p_pos, q, e, stp, done, k = state
        k, k_reset, k_eps, k_act = jax.random.split(k, 4)

        # --- STOCHASTIC RESET ---
        reset_roll = jax.random.uniform(k_reset) < RESET_PROB
        curr_p = jnp.where(reset_roll, jnp.array([0,0]), p_pos)
        curr_e = jnp.where(reset_roll, jnp.zeros_like(e), e) # Wipe traces on reset

        sid = get_meta_state(maze, curr_p)
        a_meta = jnp.where(jax.random.uniform(k_eps) < epsilon, 
                          jax.random.randint(k_act, (), 0, 2), 
                          jnp.argmax(q[sid]))
        
        next_p, rew, is_done = step_env(maze, curr_p, a_meta)
        nsid = get_meta_state(maze, next_p)
        
        # --- Q(LAMBDA) UPDATE ---
        # 1. Calculate TD Error (Delta)
        target = rew + GAMMA * jnp.where(is_done, 0.0, jnp.max(q[nsid]))
        delta = target - q[sid, a_meta]
        
        # 2. Update Eligibility Trace (Accumulating trace)
        new_e = curr_e.at[sid, a_meta].add(1.0)
        
        # 3. Apply Delta to all states using the trace
        new_q = q + ALPHA * delta * new_e
        
        # 4. Decay the trace
        new_e = new_e * (GAMMA * LAMBDA)
        
        return next_p, new_q, new_e, stp + 1, is_done | (stp > MAX_STEPS), k

    final = lax.while_loop(lambda s: ~s[4], body_fun, init_val)
    success = jnp.where((final[0][0] == 15) & (final[0][1] == 15), 1.0, 0.0)
    return (final[1], key, eps_decay), success

# --- 4. EXECUTION ---

def run_study():
    q_table = jnp.zeros((N_META_STATES, N_META_ACTIONS)) + 10.0 # Optimistic
    key = jax.random.PRNGKey(42)
    eps_decay = (0.01 / 1.0) ** (1 / EPISODES)

    print(f"🚀 Training Meta-Q(λ) Switcher with Stochastic Resetting...")
    start_t = time.time()
    
    # Training in chunks for monitoring
    chunk_size = 50000
    for chunk in range(EPISODES // chunk_size):
        chunk_range = jnp.arange(chunk * chunk_size, (chunk + 1) * chunk_size)
        (q_table, key, _), successes = lax.scan(train_episode, (q_table, key, eps_decay), chunk_range)
        print(f"Chunk {chunk+1} | Success: {jnp.mean(successes)*100:.1f}% | p: {min(0.1, (chunk*chunk_size/400000)*0.1):.3f}")

    print(f"✅ Total Time: {time.time()-start_t:.2f}s")
    # np.save('META_SWITCHER_QLAMBDA.npy', np.array(q_table))

if __name__ == "__main__":
    run_study()