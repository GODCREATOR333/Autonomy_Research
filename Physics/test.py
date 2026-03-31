import numpy as np
import random
import matplotlib.pyplot as plt
import time

# =============================
# PARAMETERS
# =============================
L = 100
v = 1.0
D = 0.5
dt = 0.1
beta = 1.0
runs = 200
alphas = np.logspace(-2, 1, 40)  # 0.01 -> 10

# =============================
# HELPER FUNCTIONS
# =============================
def simulate_trajectory(alpha):
    x = 0
    state = "G"
    t = 0
    while x < L:
        if state == "G":
            x += v*dt
            if random.random() < alpha*dt:
                state = "E"
        else:
            x += random.gauss(0, np.sqrt(2*D*dt))
            if random.random() < beta*dt:
                state = "G"
        if x < 0:
            x = 0
        t += dt
    return t

def run_sim(alpha):
    times = []
    for _ in range(runs):
        times.append(simulate_trajectory(alpha))
    return np.mean(times)

# =============================
# SWEEP ALPHA WITH PRINTS
# =============================
mfpt = []
start_time = time.time()
for i, a in enumerate(alphas):
    t0 = time.time()
    mfpt_val = run_sim(a)
    mfpt.append(mfpt_val)
    elapsed = time.time() - t0
    print(f"[{i+1}/{len(alphas)}] α={a:.3f}, MFPT={mfpt_val:.3f} (took {elapsed:.2f}s)")
total_elapsed = time.time() - start_time
print(f"\nTotal simulation time: {total_elapsed:.2f}s")

alphas = np.array(alphas)
mfpt = np.array(mfpt)

# =============================
# FIND OPTIMAL ALPHA
# =============================
opt_alpha = alphas[np.argmin(mfpt)]
print(f"Optimal α ≈ {opt_alpha:.4f}, MFPT ≈ {mfpt.min():.3f}")

# =============================
# PLOTS
# =============================

# Linear plot
plt.figure()
plt.plot(alphas, mfpt, 'o-', label="MFPT")
plt.axvline(opt_alpha, color='r', linestyle='--', label=f"Optimal α ≈ {opt_alpha:.2f}")
plt.xlabel("α")
plt.ylabel("MFPT")
plt.title("MFPT vs α (Linear Scale)")
plt.legend()
plt.show()

# Log-Log plot
log_a = np.log10(alphas)
log_m = np.log10(mfpt)
slope, intercept = np.polyfit(log_a, log_m, 1)
fit_power = 10**intercept * alphas**slope

ss_res = np.sum((log_m - (slope*log_a + intercept))**2)
ss_tot = np.sum((log_m - np.mean(log_m))**2)
r2_power = 1 - ss_res/ss_tot

plt.figure()
plt.scatter(alphas, mfpt, label="Data")
plt.plot(alphas, fit_power, 'r--', label=f"Power-law fit (slope={slope:.2f}, R²={r2_power:.3f})")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("α (log)")
plt.ylabel("MFPT (log)")
plt.title("Log-Log Plot of MFPT vs α")
plt.legend()
plt.show()

# Semi-log plot
slope_exp, intercept_exp = np.polyfit(alphas, np.log(mfpt), 1)
fit_exp = np.exp(intercept_exp) * np.exp(slope_exp*alphas)

ss_res_exp = np.sum((np.log(mfpt) - (slope_exp*alphas + intercept_exp))**2)
ss_tot_exp = np.sum((np.log(mfpt) - np.mean(np.log(mfpt)))**2)
r2_exp = 1 - ss_res_exp/ss_tot_exp

plt.figure()
plt.scatter(alphas, mfpt, label="Data")
plt.plot(alphas, fit_exp, 'g--', label=f"Exponential fit (R²={r2_exp:.3f})")
plt.yscale("log")
plt.xlabel("α")
plt.ylabel("MFPT (log)")
plt.title("Semi-log Plot of MFPT vs α")
plt.legend()
plt.show()

# =============================
# HIGH α POWER-LAW FIT
# =============================
# Select high α region (e.g., α > 1)
high_alpha_mask = alphas > 1
alphas_high = alphas[high_alpha_mask]
mfpt_high = mfpt[high_alpha_mask]

# Log-log fit
log_a_high = np.log10(alphas_high)
log_m_high = np.log10(mfpt_high)
slope_high, intercept_high = np.polyfit(log_a_high, log_m_high, 1)
fit_power_high = 10**intercept_high * alphas_high**slope_high

# R² for high alpha
fit_vals_high = slope_high * log_a_high + intercept_high
ss_res_high = np.sum((log_m_high - fit_vals_high)**2)
ss_tot_high = np.sum((log_m_high - np.mean(log_m_high))**2)
r2_high = 1 - ss_res_high / ss_tot_high

# =============================
# PLOT HIGH α LOG-LOG ONLY
# =============================
plt.figure()
plt.scatter(alphas_high, mfpt_high, label="Data (high α)", color='blue')
plt.plot(alphas_high, fit_power_high, 'r--', label=f"Power-law fit (slope={slope_high:.2f}, R²={r2_high:.3f})")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("α (log)")
plt.ylabel("MFPT (log)")
plt.title("High α Log-Log Power Law Fit")
plt.legend()
plt.show()

print(f"High α power-law slope ≈ {slope_high:.3f}, intercept ≈ {intercept_high:.3f}, R² ≈ {r2_high:.4f}")