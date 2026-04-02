import numpy as np
import matplotlib.pyplot as plt

# =============================
# PARAMETERS
# =============================
L = 100       # goal position
v = 1.0       # drift speed in G
D = 0.5       # diffusion coefficient in E
N = 200       # number of spatial points
dx = L / N

# Grid of α and β
alpha_vals = np.logspace(-2, 1.7, 30)
beta_vals  = np.logspace(-2, 1.7, 30)

# =============================
# BACKWARD EQUATION SOLVER (WITH TRAPS)
# =============================
def solve_mfpt_with_traps(alpha, beta, traps):
    T_size = 2*(N+1)
    A = np.zeros((T_size, T_size))
    b = np.zeros(T_size)
    
    for i in range(N+1):
        g = i
        e = i + N + 1
        
        # 1. Absorbing Boundary (Goal)
        if i == N:
            A[g, g] = 1
            A[e, e] = 1
            b[g] = 0
            b[e] = 0
            
        # 2. Reflecting Boundary (Start)
        elif i == 0:
            if traps[i]:
                A[g, g] = -alpha
                A[g, e] = alpha
            else:
                A[g, g] = -v/dx - alpha
                A[g, g+1] = v/dx
                A[g, e] = alpha
            b[g] = -1
            
            A[e, e] = 1
            A[e, e+1] = -1
            b[e] = 0
            
        # 3. Interior Points
        else:
            # --- G MODE (Drift) ---
            if traps[i]:
                # TRAP: Velocity is 0. Agent is stuck until it switches to E.
                A[g, g] = -alpha
                A[g, e] = alpha
                b[g] = -1
            else:
                # CLEAR: Normal drift.
                A[g, g] = -v/dx - alpha
                A[g, g+1] = v/dx
                A[g, e] = alpha
                b[g] = -1
            
            # --- E MODE (Diffusion) ---
            # E mode can always diffuse, even in traps.
            A[e, e-1] = D/dx**2
            A[e, e]   = -2*D/dx**2 - beta
            A[e, e+1] = D/dx**2
            A[e, g]   = beta
            b[e] = -1
            
    T = np.linalg.solve(A, b)
    return T[:N+1][0]  # Return MFPT starting at x=0 in state G

# =============================
# SWEEP OVER TRAP DENSITIES (The Phase Transition Hunt)
# =============================
# Let's test densities from 0% to 30%
densities = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
num_mazes = 10 # Average over 10 random mazes per density to smooth noise

optimal_alphas = []
optimal_betas = []
min_mfpts = []

for p in densities:
    print(f"\nEvaluating Trap Density: {p*100}%")
    
    # Pre-generate mazes for this density
    mazes = []
    for _ in range(num_mazes):
        maze = np.random.rand(N+1) < p
        maze[0] = False # Start is never a trap
        maze[N] = False # Goal is never a trap
        mazes.append(maze)
        
    MFPT_avg = np.zeros((len(alpha_vals), len(beta_vals)))
    
    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            
            # Average the MFPT across the random mazes
            mfpt_sum = 0
            for maze in mazes:
                mfpt_sum += solve_mfpt_with_traps(alpha, beta, maze)
            MFPT_avg[i, j] = mfpt_sum / num_mazes
            
    # Find Optimal for this density
    min_idx = np.unravel_index(np.argmin(MFPT_avg), MFPT_avg.shape)
    alpha_opt = alpha_vals[min_idx[0]]
    beta_opt  = beta_vals[min_idx[1]]
    mfpt_min  = MFPT_avg[min_idx]
    
    optimal_alphas.append(alpha_opt)
    optimal_betas.append(beta_opt)
    min_mfpts.append(mfpt_min)
    
    print(f"Optimal Strategy -> α(G→E): {alpha_opt:.3f}, β(E→G): {beta_opt:.3f} | MFPT: {mfpt_min:.1f}")

# =============================
# PLOT THE PHASE TRANSITION
# =============================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(densities, optimal_alphas, '-o', color='blue', label='α (G→E)')
plt.plot(densities, optimal_betas, '-o', color='red', label='β (E→G)')
plt.yscale('log')
plt.xlabel('Trap Density (p)')
plt.ylabel('Optimal Switching Rate')
plt.title('Phase Transition in Strategy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(densities, min_mfpts, '-ok', label='Minimum MFPT')
plt.xlabel('Trap Density (p)')
plt.ylabel('Time to Goal')
plt.title('Cost Function vs Density')
plt.legend()

plt.tight_layout()
plt.show()