import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
import numpy as np
import sys
import time

# --- 1. Link to Alios Registry ---
sys.path.append("../Alios") 
from alios_db import register_run

# --- 2. Parameters ---
GRID_SIZE = 16
N_ACTIONS = 4
N_STATES = 512 # 2^9 (The 3x3 window)
EPISODES = 100000
MAX_STEPS = 50 
ALPHA = 0.2
GAMMA = 0.9
DELTAS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

# --- 3. Analytical Safety Prior (Consistently 9-bit) ---
def get_safety_prior_512():
    """
    Creates a (512, 4) Q-table. 
    Indices in the flattened 3x3 window:
    [0 1 2]  <- 1 is UP
    [3 4 5]  <- 3 is LEFT, 4 is CENTER, 5 is RIGHT
    [6 7 8]  <- 7 is DOWN
    """
    # Generate all 512 combinations of 9 bits
    patterns = jnp.array([[int(x) for x in format(i, '09b')] for i in range(512)])
    
    # Identify which patterns have walls in cardinal directions
    up_walls    = patterns[:, 1]
    down_walls  = patterns[:, 7]
    left_walls  = patterns[:, 3]
    right_walls = patterns[:, 5]
    
    prior = jnp.zeros((512, 4))
    # Fill walls with -100 to ensure 'Reflexive' avoidance
    prior = prior.at[:, 0].set(jnp.where(up_walls == 1, -100.0, 0.0))    # UP
    prior = prior.at[:, 1].set(jnp.where(down_walls == 1, -100.0, 0.0))  # DOWN
    prior = prior.at[:, 2].set(jnp.where(left_walls == 1, -100.0, 0.0))  # LEFT
    prior = prior.at[:, 3].set(jnp.where(right_walls == 1, -100.0, 0.0)) # RIGHT
    return prior

# --- 4. JAX Logic (Mirrors your core_logic.py) ---
@jit
def get_state(maze, pos):
    r, c = pos
    padded = jnp.pad(maze, 1, constant_values=1)
    window = lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    return jnp.sum(window.astype(jnp.int32) * powers)

@jit
def step(maze, pos, action):
    new_pos = pos + DELTAS[action]
    out = (new_pos[0]<0)|(new_pos[0]>=16)|(new_pos[1]<0)|(new_pos[1]>=16)
    safe_rc = jnp.clip(new_pos, 0, 15)
    hit = out | (maze[safe_rc[0], safe_rc[1]] == 1)
    # Reward: -100 for wall, +1 for safe move
    reward = jnp.where(hit, -100.0, 1.0)
    return jnp.where(hit, pos, new_pos), reward, hit

@jit
def train_step(carry, episode_idx):
    q, mazes, key = carry
    key, km, ks, kl = jax.random.split(key, 4)
    maze = mazes[jax.random.randint(km, (), 0, mazes.shape[0])]
    
    # Spawn at random open positions to cover the 512-state space
    rand_pos = jax.random.randint(ks, (2,), 0, 16)
    pos = jnp.where(maze[rand_pos[0], rand_pos[1]] == 0, rand_pos, jnp.array([0,0]))
    
    epsilon = jnp.maximum(0.01, 1.0 * (0.99992 ** episode_idx))

    def body(s):
        p, q_tab, steps, done, k = s
        k, k1, k2 = jax.random.split(k, 3)
        sid = get_state(maze, p)
        
        # Epsilon-greedy
        a = jnp.where(jax.random.uniform(k1) < epsilon, 
                      jax.random.randint(k2, (), 0, 4), 
                      jnp.argmax(q_tab[sid]))
        
        np_pos, rew, hit = step(maze, p, a)
        nsid = get_state(maze, np_pos)
        
        # Terminal Q-update logic
        target = rew + GAMMA * jnp.max(q_tab[nsid]) * (1.0 - hit.astype(jnp.float32))
        q_tab = q_tab.at[sid, a].add(ALPHA * (target - q_tab[sid, a]))
        return np_pos, q_tab, steps + 1, hit, k

    init = (pos, q, 0, False, kl)
    final = lax.while_loop(lambda s: (s[2] < MAX_STEPS) & (~s[3]), body, init)
    return (final[1], mazes, key), 0.0

# --- 5. Run Training and Register ---
def main():
    print("🧠 Building Analytical Survival Prior (512 states)...")
    initial_q = get_safety_prior_512()
    
    print("🚀 Training Pure Reactive Ego via JAX...")
    train_data = jnp.array(np.load("data_jax/N16_P0100_train_solvable.npy"))
    key = jax.random.PRNGKey(42)
    
    start_t = time.time()
    (final_q, _, _), _ = lax.scan(train_step, (initial_q, train_data, key), jnp.arange(EPISODES))
    print(f"✅ Completed 100,000 episodes in {time.time()-start_t:.2f}s")

    # Register in Alios
    register_run(
        run_id="EGO_REACTIVE_SURVIVOR_512",
        algo="Reactive Survivor",
        state_repr="pure_ego", # Matches your core_logic key
        config_dict={
            "states": 512, 
            "mode": "9-bit Reactive", 
            "memory": "None", 
            "prior": "Analytical Wall-Avoidance"
        },
        q_table=np.array(final_q)
    )

if __name__ == "__main__":
    main()