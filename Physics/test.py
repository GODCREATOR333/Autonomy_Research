import numpy as np
import random
import matplotlib.pyplot as plt

# =============================
# 🔹 PARAMETERS
# =============================
L = 100
v = 1.0
D = 0.5
dt = 0.1
beta = 1.0
runs = 2000

# Trap region
traps = [(20, 25)]

# =============================
# 🔹 HELPER FUNCTIONS
# =============================
def in_trap(x):
    for a, b in traps:
        if a <= x <= b:
            return True
    return False

# =============================
# 🔹 SINGLE TRAJECTORY (VISUAL)
# =============================
def simulate_trajectory(alpha):
    x = 0
    state = "G"
    t = 0

    xs, ts, states = [], [], []

    while x < L and t < 200:
        xs.append(x)
        ts.append(t)
        states.append(state)

        if state == "G":
            if not in_trap(x):
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

    return ts, xs, states

# =============================
# 🔹 MFPT SIMULATION
# =============================
def run_sim(alpha):
    times = []

    for _ in range(runs):
        x = 0
        state = "G"
        t = 0

        while x < L:
            if state == "G":
                if not in_trap(x):
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
# 🔹 RUN EXPERIMENT
# =============================
alphas = np.logspace(-2, 1, 40)
mfpt = []

for a in alphas:
    print("alpha:", a)
    mfpt.append(run_sim(a))

alphas = np.array(alphas)
mfpt = np.array(mfpt)

# =============================
# 🔹 REMOVE INVALID VALUES
# =============================
mask = (mfpt > 0)
alphas = alphas[mask]
mfpt = mfpt[mask]

# =============================
# 🔹 POWER LAW FIT
# =============================
log_a = np.log10(alphas)
log_m = np.log10(mfpt)

slope, intercept = np.polyfit(log_a, log_m, 1)
fit_power = 10**intercept * alphas**slope

# R² power
fit_vals = slope * log_a + intercept
ss_res = np.sum((log_m - fit_vals)**2)
ss_tot = np.sum((log_m - np.mean(log_m))**2)
r2_power = 1 - ss_res / ss_tot

# =============================
# 🔹 EXPONENTIAL FIT
# =============================
slope_exp, intercept_exp = np.polyfit(alphas, np.log(mfpt), 1)
fit_exp = np.exp(intercept_exp) * np.exp(slope_exp * alphas)

# R² exponential
fit_vals_exp = slope_exp * alphas + intercept_exp
ss_res_exp = np.sum((np.log(mfpt) - fit_vals_exp)**2)
ss_tot_exp = np.sum((np.log(mfpt) - np.mean(np.log(mfpt)))**2)
r2_exp = 1 - ss_res_exp / ss_tot_exp

# =============================
# 🔹 FIND OPTIMAL ALPHA
# =============================
opt_alpha = alphas[np.argmin(mfpt)]

# =============================
# 🔹 PLOTS
# =============================

# Linear plot
plt.figure()
plt.plot(alphas, mfpt, 'o-', label="Data")
plt.axvline(opt_alpha, color='r', linestyle='--', label=f"Optimal α ≈ {opt_alpha:.2f}")
plt.xlabel("α")
plt.ylabel("MFPT")
plt.title("MFPT vs α (Linear Scale)")
plt.legend()
plt.show()

# Log-log plot
plt.figure()
plt.scatter(alphas, mfpt, label="Data")
plt.plot(alphas, fit_power, 'r--', label=f"Power fit (R²={r2_power:.3f})")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("α (log)")
plt.ylabel("MFPT (log)")
plt.title(f"Log-Log (slope ≈ {slope:.2f})")
plt.legend()
plt.show()

# Semi-log plot
plt.figure()
plt.scatter(alphas, mfpt, label="Data")
plt.plot(alphas, fit_exp, 'g--', label=f"Exp fit (R²={r2_exp:.3f})")
plt.yscale("log")
plt.xlabel("α")
plt.ylabel("MFPT (log)")
plt.title("Semi-log Plot")
plt.legend()
plt.show()

# =============================
# 🔹 PRINT RESULTS
# =============================
print("\n===== FIT RESULTS =====")
print(f"Power-law exponent ≈ {slope:.3f}")
print(f"Power-law R²       ≈ {r2_power:.4f}")
print(f"Exponential slope  ≈ {slope_exp:.3f}")
print(f"Exponential R²     ≈ {r2_exp:.4f}")
print(f"Optimal α          ≈ {opt_alpha:.4f}")

# =============================
# 🔹 VISUALIZE ONE TRAJECTORY
# =============================
ts, xs, states = simulate_trajectory(alpha=opt_alpha)

plt.figure()

for i in range(len(ts) - 1):
    if states[i] == "G":
        plt.plot(ts[i:i+2], xs[i:i+2], 'b')
    else:
        plt.plot(ts[i:i+2], xs[i:i+2], 'r')

plt.axhspan(20, 25, color='gray', alpha=0.3, label="Trap")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Single Trajectory (Blue=Geo, Red=Ego)")
plt.legend()
plt.show()