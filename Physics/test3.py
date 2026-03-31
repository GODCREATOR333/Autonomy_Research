import numpy as np
import random
import matplotlib.pyplot as plt
import cma

# =============================
# PARAMETERS
# =============================
L = 100
v = 1.0
D = 0.5
dt = 0.1
runs = 50

rho_vals = np.linspace(0, 1, 12)
trap_length = 5
num_env = 1   # environments per rho

# =============================
# TRAP GENERATION
# =============================
def generate_traps(rho):
    traps = []
    total = int(rho * L)
    covered = 0

    while covered < total:
        start = random.uniform(0, L - trap_length)
        traps.append((start, start + trap_length))
        covered += trap_length

    return traps

def in_trap(x, traps):
    for a, b in traps:
        if a <= x <= b:
            return True
    return False

# =============================
# MFPT SIMULATION
# =============================
def simulate_mfpt(alpha, beta, traps):
    times = []

    for _ in range(runs):
        x = 0
        state = "G"
        t = 0

        while x < L and t < 500:
            if state == "G":
                if not in_trap(x, traps):
                    x += v * dt
                if random.random() < alpha * dt:
                    state = "E"

            else:
                x += random.gauss(0, np.sqrt(2 * D * dt))
                if random.random() < beta * dt:
                    state = "G"

            if x < 0:
                x = 0

            t += dt

        times.append(t)

    return np.mean(times)

# =============================
# CMA-ES OPTIMIZER
# =============================
def optimize_cma(traps):

    def objective(log_params):
        log_alpha, log_beta = log_params
        alpha = 10**log_alpha
        beta  = 10**log_beta

        return simulate_mfpt(alpha, beta, traps)

    # initial guess in log space
    x0 = [0, 0]   # α=1, β=1
    sigma = 1.0

    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'bounds': [[-2, -2], [2, 2]],  # α,β ∈ [0.01, 100]
        'verb_disp': 0
    })

    while not es.stop():
        solutions = es.ask()
        values = [objective(x) for x in solutions]
        es.tell(solutions, values)

    best = es.result.xbest
    alpha_opt = 10**best[0]
    beta_opt  = 10**best[1]

    return alpha_opt, beta_opt

# =============================
# MAIN LOOP
# =============================
alpha_star = []
beta_star = []

for rho in rho_vals:
    print(f"\nρ = {rho:.2f}")

    alpha_list = []
    beta_list = []

    for _ in range(num_env):
        traps = generate_traps(rho)
        a_opt, b_opt = optimize_cma(traps)

        alpha_list.append(a_opt)
        beta_list.append(b_opt)

    alpha_star.append(np.mean(alpha_list))
    beta_star.append(np.mean(beta_list))

    print(f"α* ≈ {alpha_star[-1]:.3f}, β* ≈ {beta_star[-1]:.3f}")

# =============================
# PLOT
# =============================
plt.figure(figsize=(8,6))

plt.plot(rho_vals, alpha_star, 'o-', label='α*')
plt.plot(rho_vals, beta_star, 's-', label='β*')

plt.yscale('log')
plt.xlabel('Trap density ρ')
plt.ylabel('Optimal switching rate')
plt.title('CMA-ES Optimized Switching Strategy')
plt.grid(True, which='both')
plt.legend()
plt.show()