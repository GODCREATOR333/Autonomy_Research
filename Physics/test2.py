import numpy as np
import matplotlib.pyplot as plt

# =============================
# PARAMETERS
# =============================
L = 100       # goal position
v = 1.0       # velocity in G
D = 0.5       # diffusion in E
beta = 1.0    # switching rate E -> G
N = 200       # number of grid points
dx = L / N

# Alpha values: extend to high alpha for power law
alphas = np.logspace(-2, 1.7, 50)  # 0.01 -> ~50

# =============================
# FUNCTION TO SOLVE BACKWARD EQUATION
# =============================
def solve_mfpt(alpha):
    x = np.linspace(0, L, N+1)
    T_size = 2*(N+1)
    
    A = np.zeros((T_size, T_size))
    b = np.zeros(T_size)
    
    for i in range(N+1):
        g = i
        e = i + N + 1
        
        if i == N:
            A[g, g] = 1
            A[e, e] = 1
            b[g] = 0
            b[e] = 0
        elif i == 0:
            A[g, g] = -v/dx - alpha
            A[g, g+1] = v/dx
            A[g, e] = alpha
            b[g] = -1
            
            A[e, e] = 1
            A[e, e+1] = -1
            b[e] = 0
        else:
            A[g, g] = -v/dx - alpha
            A[g, g+1] = v/dx
            A[g, e] = alpha
            b[g] = -1
            
            A[e, e-1] = D/dx**2
            A[e, e]   = -2*D/dx**2 - beta
            A[e, e+1] = D/dx**2
            A[e, g]   = beta
            b[e] = -1
    
    T = np.linalg.solve(A, b)
    T_G = T[:N+1]
    T_E = T[N+1:]
    
    return x, T_G, T_E

# =============================
# COMPUTE MFPT VS ALPHA
# =============================
mfpt_G0 = []

for alpha in alphas:
    x, T_G, T_E = solve_mfpt(alpha)
    mfpt_G0.append(T_G[0])

mfpt_G0 = np.array(mfpt_G0)

# =============================
# PLOT LOG-LOG AND FIT POWER LAW AT HIGH ALPHA
# =============================
plt.figure()
plt.scatter(alphas, mfpt_G0, label="MFPT Data")

# Pick high-alpha regime for power-law fit (α > 5)
mask = alphas > 5
log_alpha = np.log10(alphas[mask])
log_mfpt = np.log10(mfpt_G0[mask])

# Linear fit in log-log → power law
slope, intercept = np.polyfit(log_alpha, log_mfpt, 1)
fit_power = 10**intercept * alphas[mask]**slope

plt.plot(alphas[mask], fit_power, 'r--', label=f"Power-law fit: slope={slope:.3f}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("α (G → E)")
plt.ylabel("MFPT starting at x=0, G")
plt.title("Log-Log MFPT vs α (High α power-law check)")
plt.legend()
plt.grid(True)
plt.show()

# =============================
# PRINT FIT RESULTS
# =============================
print("===== POWER-LAW FIT RESULTS (High α) =====")
print(f"Slope (exponent) ≈ {slope:.3f}")
print(f"Intercept ≈ {intercept:.3f}")