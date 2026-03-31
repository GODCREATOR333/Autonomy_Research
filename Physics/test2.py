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
alpha_vals = np.logspace(-2, 1.7, 30)  # 0.01 -> ~50
beta_vals  = np.logspace(-2, 1.7, 30)  # 0.01 -> ~50

# =============================
# BACKWARD EQUATION SOLVER
# =============================
def solve_mfpt(alpha, beta):
    x = np.linspace(0, L, N+1)
    T_size = 2*(N+1)
    
    A = np.zeros((T_size, T_size))
    b = np.zeros(T_size)
    
    for i in range(N+1):
        g = i
        e = i + N + 1
        
        if i == N:
            # Absorbing at goal
            A[g, g] = 1
            A[e, e] = 1
            b[g] = 0
            b[e] = 0
        elif i == 0:
            # Reflecting at start for E (diffusion)
            # For G, standard backward eq
            A[g, g] = -v/dx - alpha
            A[g, g+1] = v/dx
            A[g, e] = alpha
            b[g] = -1
            
            A[e, e] = 1
            A[e, e+1] = -1
            b[e] = 0
        else:
            # Interior points
            # G mode (drift)
            A[g, g] = -v/dx - alpha
            A[g, g+1] = v/dx
            A[g, e] = alpha
            b[g] = -1
            
            # E mode (diffusion)
            A[e, e-1] = D/dx**2
            A[e, e]   = -2*D/dx**2 - beta
            A[e, e+1] = D/dx**2
            A[e, g]   = beta
            b[e] = -1
    
    T = np.linalg.solve(A, b)
    T_G = T[:N+1]
    T_E = T[N+1:]
    
    return T_G[0]  # MFPT starting at x=0, state G

# =============================
# SWEEP ALPHA AND BETA
# =============================
MFPT = np.zeros((len(alpha_vals), len(beta_vals)))

for i, alpha in enumerate(alpha_vals):
    for j, beta in enumerate(beta_vals):
        MFPT[i,j] = solve_mfpt(alpha, beta)
        print(f"α={alpha:.3f}, β={beta:.3f}, MFPT={MFPT[i,j]:.3f}")

# =============================
# FIND MINIMUM
# =============================
min_idx = np.unravel_index(np.argmin(MFPT), MFPT.shape)
alpha_opt = alpha_vals[min_idx[0]]
beta_opt  = beta_vals[min_idx[1]]
mfpt_min  = MFPT[min_idx]

print("\n===== OPTIMAL SWITCHING =====")
print(f"Minimum MFPT ≈ {mfpt_min:.3f} at α ≈ {alpha_opt:.3f}, β ≈ {beta_opt:.3f}")

# =============================
# PLOT HEATMAP
# =============================
plt.figure(figsize=(8,6))
plt.contourf(beta_vals, alpha_vals, MFPT, levels=50, cmap='viridis')
plt.colorbar(label='MFPT (x=0, G)')
plt.scatter(beta_opt, alpha_opt, color='red', label='Optimal')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('β (E→G)')
plt.ylabel('α (G→E)')
plt.title('MFPT Heatmap vs α and β')
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# =============================
# 3D SURFACE PLOT
# =============================
Alpha_grid, Beta_grid = np.meshgrid(alpha_vals, beta_vals, indexing='ij')

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(np.log10(Beta_grid), np.log10(Alpha_grid), MFPT, 
                       cmap='viridis', edgecolor='none', alpha=0.9)

# Mark optimal point
ax.scatter(np.log10(beta_opt), np.log10(alpha_opt), mfpt_min, 
           color='red', s=50, label='Optimal MFPT')

ax.set_xlabel('log10(β)')
ax.set_ylabel('log10(α)')
ax.set_zlabel('MFPT (x=0, G)')
ax.set_title('3D Surface of MFPT vs α and β')
ax.view_init(elev=30, azim=45)
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=10, label='MFPT')
plt.show()