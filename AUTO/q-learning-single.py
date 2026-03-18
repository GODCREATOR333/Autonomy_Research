import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# --- 1. Load Data ---
# Replace with your actual path. Creating a dummy maze if file not found for demo.
try:
    data = np.load("data_jax/N16_P0100_test_solvable_random.npy")
    maze = data[0]
except:
    maze = np.zeros((16, 16)) # Fallback
    
GOAL = (15, 15)
GRID_SIZE = 16

# --- 2. State & Direction Logic ---

def get_geocentric_hint(r, c):
    """Calculates which of the 4 actions points most toward the goal."""
    if (r, c) == GOAL: return 0
    dy, dx = GOAL[0] - r, GOAL[1] - c
    angle = np.arctan2(dy, dx)
    # Discretize 360 degrees into 4 bins centered on axes
    shifted_angle = (angle + np.pi/4) % (2 * np.pi)
    bin_idx = int(shifted_angle / (np.pi/2))
    # Map: 0:Right(3), 1:Down(1), 2:Left(2), 3:Up(0)
    return {0: 3, 1: 1, 2: 2, 3: 0}.get(bin_idx, 1)

def get_state(maze, r, c):
    """Combines 3x3 local walls (256 combos) with compass hint (4 combos)."""
    padded = np.pad(maze, 1, constant_values=1)
    window = padded[r:r+3, c:c+3].flatten()
    neighbors = np.delete(window, 4) # Remove center
    
    local_id = 0
    for bit in neighbors:
        local_id = (local_id << 1) | int(bit)
        
    hint = get_geocentric_hint(r, c)
    return int(local_id * 4 + hint)

# --- 3. Environment & Reward ---

def step(maze, r, c, action):
    dr, dc = [(-1,0), (1,0), (0,-1), (0,1)][action] # Up, Down, Left, Right
    nr, nc = r + dr, c + dc
    
    # Boundary/Wall Check
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE or maze[nr, nc] == 1:
        # Hit wall: stay put, high penalty
        return r, c, -5.0, False
    
    if (nr, nc) == GOAL:
        return nr, nc, 100.0, True
    
    # Progress-based reward shaping to prevent loops
    old_dist = abs(r - GOAL[0]) + abs(c - GOAL[1])
    new_dist = abs(nr - GOAL[0]) + abs(nc - GOAL[1])
    # +0.5 for closer, -0.5 for further, -0.1 base penalty
    reward = 0.5 * (old_dist - new_dist) - 0.1
    return nr, nc, reward, False

# --- 4. Training ---

Q = np.zeros((1024, 4))
alpha, gamma = 0.2, 0.99
epsilon, min_epsilon, decay = 1.0, 0.01, 0.9995

print("Training agent...")
for episode in range(20000):
    r, c = 0, 0
    done = False
    steps = 0
    epsilon = max(min_epsilon, epsilon * decay)
    
    while not done and steps < 1000:
        s = get_state(maze, r, c)
        
        # Action Selection with Geocentric Tie-breaking
        if np.random.rand() < epsilon:
            a = np.random.randint(4)
        else:
            q_s = Q[s]
            if np.all(q_s == 0): # No knowledge yet? Follow compass.
                a = get_geocentric_hint(r, c)
            else:
                a = np.argmax(q_s)
        
        nr, nc, rew, done = step(maze, r, c, a)
        ns = get_state(maze, nr, nc)
        
        # Q-Update
        max_next_q = 0 if done else np.max(Q[ns])
        Q[s, a] += alpha * (rew + gamma * max_next_q - Q[s, a])
        
        r, c, steps = nr, nc, steps + 1

# --- 5. Visualization ---

arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(maze, cmap='gray_r')

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if (r, c) == GOAL:
            ax.text(c, r, "★", ha='center', va='center', color='red', fontsize=15)
        elif maze[r, c] == 0:
            s = get_state(maze, r, c)
            a = np.argmax(Q[s])
            ax.text(c, r, arrow_map[a], ha='center', va='center', color='blue', alpha=0.6)

ax.set_title("Optimized Geocentric Policy")
plt.show()