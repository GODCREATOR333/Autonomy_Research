import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# 1. Your Exact Physics Parameters
# =========================
D_R = 0.15
P_straight_ideal = np.exp(-D_R)
P_turn_ideal = (1.0 - P_straight_ideal) / 2.0
KAPPA = 2.5

GOAL_POS = np.array([15, 15])
MAX_STEPS = 500  # Increased for search fairness
DATA_DIR = "data_jax"
DENSITIES = ["P0100", "P0200", "P0300" , "P0350" ,"P0370" ,"P0400"] # Testing a few for the graph

DELTAS = {
    0: np.array([-1, 0]),  # UP
    1: np.array([1, 0]),   # DOWN
    2: np.array([0, -1]),  # LEFT
    3: np.array([0, 1])    # RIGHT
}

# =========================
# 2. Your Exact Ego/Geo Policies
# =========================

def get_ego_probabilities(win_id, last_a):
    """ Your exact ABP bit-decoding logic """
    is_wall = np.array([
        (win_id >> 7) & 1,  # UP
        (win_id >> 1) & 1,  # DOWN
        (win_id >> 5) & 1,  # LEFT
        (win_id >> 3) & 1   # RIGHT
    ])
    probs = np.zeros(4)
    if last_a == -1:
        probs = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        for a in range(4):
            if a == last_a: probs[a] = P_straight_ideal
            elif a == opposites[last_a]: probs[a] = 0.0
            else: probs[a] = P_turn_ideal

    probs[is_wall == 1] = 0.0
    s = np.sum(probs)
    if s > 0: probs /= s
    else:
        rev = {0: 1, 1: 0, 2: 3, 3: 2}[last_a]
        probs[rev] = 1.0
    return probs

def get_geo_probabilities(pos, kappa=KAPPA):
    """ Your exact discrete Von Mises logic """
    v_g = GOAL_POS - pos
    u_g = v_g / (np.linalg.norm(v_g) + 1e-6)
    actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    q_values = np.dot(actions, u_g)
    exp_q = np.exp(kappa * q_values)
    return exp_q / np.sum(exp_q)

# =========================
# 3. Proactive Hybrid Simulation
# =========================

def run_proactive_sim(maze, gamma_p, gamma_r=0.2):
    """
    gamma_p: Prob(Run -> Tumble) - Tune this!
    gamma_r: Prob(Tumble -> Run) - Persistence length in EGO mode
    """
    pos = np.array([0, 0])
    last_a = -1
    mode = 0  # 0: GEO (Run), 1: EGO (Tumble)
    padded = np.pad(maze, 1, constant_values=1)
    
    for step in range(MAX_STEPS):
        # --- Stochastic Switching (The Physics Paper's Logic) ---
        if mode == 0:
            if np.random.rand() < gamma_p: mode = 1
        else:
            if np.random.rand() < gamma_r: mode = 0

        # --- Policy Selection ---
        if mode == 0:
            probs = get_geo_probabilities(pos)
        else:
            # Get 3x3 local window for win_id
            window = padded[pos[0]:pos[0]+3, pos[1]:pos[1]+3].flatten()
            win_id = int(np.sum(window * np.array([256,128,64,32,16,8,4,2,1])))
            probs = get_ego_probabilities(win_id, last_a)

        # --- Movement ---
        action = np.random.choice(4, p=probs)
        next_pos = pos + DELTAS[action]

        if (0 <= next_pos[0] < 16 and 0 <= next_pos[1] < 16 and maze[next_pos[0], next_pos[1]] == 0):
            pos = next_pos
            last_a = action
        else:
            # Reactive safeguard: force EGO mode on collision
            mode = 1
            last_a = action

        if np.array_equal(pos, GOAL_POS):
            return step + 1
            
    return MAX_STEPS # Failure penalty

# =========================
# 4. Sweep & Analysis
# =========================

results_summary = []
gamma_values = np.linspace(0.01, 0.8, 12)

for tag in DENSITIES:
    print(f"\nAnalyzing Density: {tag}")
    mazes = np.load(os.path.join(DATA_DIR, f"N16_{tag}_train_solvable.npy"))[:200]
    
    avg_scores = []
    for gp in gamma_values:
        # Run multiple trials per maze to get a clean Expected Search Time
        scores = [run_proactive_sim(m, gp) for m in mazes for _ in range(5)]
        mean_score = np.mean(scores)
        avg_scores.append(mean_score)
        print(f"  gp: {gp:.2f} | Score (EST): {mean_score:.1f}")
        
    opt_gamma = gamma_values[np.argmin(avg_scores)]
    results_summary.append((tag, opt_gamma))

# =========================
# 5. Graphing Optimal Switch Frequency
# =========================

tags, opts = zip(*results_summary)
density_floats = [float(t[1:])/1000 for t in tags]

plt.figure(figsize=(8, 5))
plt.plot(density_floats, opts, 'D--', color='firebrick', label='Experimental $\gamma^*$')
plt.title("PhD Research: Optimal Switching Frequency $\gamma^*$ vs. Maze Density")
plt.xlabel("Obstacle Density")
plt.ylabel("Optimal Proactive Switching Rate ($\gamma_p$)")
plt.grid(True, alpha=0.2)
plt.legend()
plt.show()