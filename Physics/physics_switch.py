"""
hybrid_agent.py
===============
Hybrid GEO ↔ EGO agent with obstacle-triggered switching.

Physics
-------
  GEO mode  : Von Mises / softmax goal-directed drift (KAPPA concentration).
               Blind to walls — chooses purely by goal direction.
  EGO mode  : Persistent ABP with rotational diffusion D_R.
               Wall-aware (masks blocked directions from local 3×3 window).
               Carries inertia via last_a memory.

Switching rule
--------------
  1. Agent starts in GEO.
  2. GEO wall collision → immediately engage EGO.
       last_a = blocked direction (seeds EGO inertia toward the wall,
       so EGO naturally turns to find a way around).
  3. Every step while in EGO: return to GEO with prob γ.
       γ = 0  →  EGO forever after first collision  (pure EGO)
       γ = 1  →  single EGO recovery step, then back to GEO
  4. Scan γ ∈ [0, 1] → find minimum MFPT.

MFPT policy
-----------
  Reported only over successful runs (reached goal).
  Success rate reported separately.

Usage
-----
  python hybrid_agent.py
  ← → arrow keys in the interactive viewer to browse mazes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════
# 1.  Shared Parameters
# ═══════════════════════════════════════════════════════════════
KAPPA      = 2.5          # GEO: Von Mises concentration  (physics_geo.py)
D_R        = 0.15         # EGO: rotational diffusion      (physics_ego.py)
P_STRAIGHT = np.exp(-D_R)
P_TURN     = (1.0 - P_STRAIGHT) / 2.0

GOAL_POS   = np.array([15, 15])
MAX_STEPS  = 400
DATA_PATH  = "data_jax/N16_P0370_test_solvable_random.npy"

# (row, col) deltas for UP / DOWN / LEFT / RIGHT
DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

# ── Gamma scan ────────────────────────────────────────────────
GAMMA_VALUES = np.linspace(0.0, 1.0, 21)   # 0.00, 0.05, …, 1.00
SCAN_TRIALS  = 10   # fast sweep
FINAL_TRIALS = 20   # validation at optimal γ

# ── Colours ───────────────────────────────────────────────────
GEO_COLOR = "royalblue"
EGO_COLOR = "tomato"

# ═══════════════════════════════════════════════════════════════
# 2.  GEO Policy   (physics_geo.py — unchanged)
# ═══════════════════════════════════════════════════════════════
def get_geo_probs(pos):
    """Von Mises / softmax goal-directed distribution over 4 actions."""
    if np.array_equal(pos, GOAL_POS):
        return np.ones(4) / 4.0
    v = GOAL_POS.astype(float) - pos
    u = v / np.linalg.norm(v)
    q = DELTAS @ u          # dot each action with unit goal vector
    e = np.exp(KAPPA * q)
    return e / e.sum()


# ═══════════════════════════════════════════════════════════════
# 3.  EGO Policy   (physics_ego.py — unchanged)
# ═══════════════════════════════════════════════════════════════
_OPP = {0: 1, 1: 0, 2: 3, 3: 2}   # opposite directions

def get_ego_probs(win_id, last_a):
    """
    Persistent ABP: biased toward last action, no reverse,
    walls masked from 3×3 local window encoding.
    """
    is_wall = np.array([
        (win_id >> 7) & 1,   # UP
        (win_id >> 1) & 1,   # DOWN
        (win_id >> 5) & 1,   # LEFT
        (win_id >> 3) & 1,   # RIGHT
    ], dtype=float)

    probs = np.zeros(4)
    if last_a == -1:
        probs[:] = 0.25
    else:
        for a in range(4):
            if   a == last_a:        probs[a] = P_STRAIGHT
            elif a == _OPP[last_a]:  probs[a] = 0.0
            else:                     probs[a] = P_TURN

    probs[is_wall == 1] = 0.0
    s = probs.sum()
    if s > 0:
        probs /= s
    else:
        # True dead-end: reverse
        probs[_OPP.get(last_a, 0)] = 1.0
    return probs


def _entropy(p):
    p = p[p > 1e-12]
    return -np.sum(p * np.log(p))


def _win_id(padded, pos):
    """3×3 binary window encoded as a 9-bit integer."""
    w = padded[pos[0]: pos[0]+3, pos[1]: pos[1]+3].flatten()
    return int((w * np.array([256, 128, 64, 32, 16, 8, 4, 2, 1])).sum())


# ═══════════════════════════════════════════════════════════════
# 4.  Hybrid Simulation
# ═══════════════════════════════════════════════════════════════
def run_hybrid(maze, gamma, max_steps=MAX_STEPS):
    """
    Simulate one trajectory of the GEO↔EGO hybrid agent.

    Parameters
    ----------
    maze  : (16,16) binary array  (0=free, 1=wall)
    gamma : float ∈ [0,1]  — prob. of returning GEO per EGO step
    max_steps : int

    Returns
    -------
    path  : (T+1, 2) int   positions in (row, col)
    acts  : (T,)    int    action taken at each step
    ents  : (T,)    float  entropy of the chosen policy distribution
    modes : (T,)    int    0=GEO, 1=EGO  — mode that generated that action
    """
    H, W    = maze.shape
    padded  = np.pad(maze, 1, constant_values=1)

    pos     = np.array([0, 0])
    mode    = 0      # 0 = GEO, 1 = EGO
    last_a  = -1     # EGO inertia (−1 = no history)

    path  = [tuple(pos)]
    acts  = []
    ents  = []
    modes = []

    for _ in range(max_steps):
        mode_used = mode   # mode that will generate this step's action

        # ── Choose action ────────────────────────────────────
        if mode == 0:
            probs = get_geo_probs(pos)
        else:
            probs = get_ego_probs(_win_id(padded, pos), last_a)

        ent = _entropy(probs)
        a   = np.random.choice(4, p=probs)

        # ── Apply action (physics enforces walls) ─────────────
        npos = pos + DELTAS[a]
        if (0 <= npos[0] < H and 0 <= npos[1] < W
                and maze[npos[0], npos[1]] == 0):
            pos    = npos      # successful move
            last_a = a
        elif mode == 0:
            # GEO wall collision → engage EGO
            # Use the blocked direction as last_a so EGO inertia points
            # toward the wall; EGO will mask it and naturally turn around it.
            mode   = 1
            last_a = a

        # ── EGO → GEO return (only when EGO was active this step) ──
        if mode_used == 1 and np.random.random() < gamma:
            mode = 0

        # ── Record ───────────────────────────────────────────
        path.append(tuple(pos))
        acts.append(a)
        ents.append(ent)
        modes.append(mode_used)   # mode that *generated* this action

        if np.array_equal(pos, GOAL_POS):
            break

    return (np.array(path),
            np.array(acts),
            np.array(ents),
            np.array(modes))


# ═══════════════════════════════════════════════════════════════
# 5.  MFPT Evaluator  (successful runs only + success rate)
# ═══════════════════════════════════════════════════════════════
def eval_gamma(all_mazes, gamma, trials, max_steps=MAX_STEPS):
    """
    Returns
    -------
    success_rate : fraction of runs that reached goal
    mfpt         : mean first-passage time (steps) over successful runs;
                   np.nan if no successes
    """
    hit_steps = []
    n_total   = len(all_mazes) * trials

    for maze in all_mazes:
        for _ in range(trials):
            path, *_ = run_hybrid(maze, gamma, max_steps)
            if np.array_equal(path[-1], GOAL_POS):
                hit_steps.append(len(path) - 1)

    sr   = len(hit_steps) / n_total
    mfpt = float(np.mean(hit_steps)) if hit_steps else np.nan
    return sr, mfpt


# ═══════════════════════════════════════════════════════════════
# 6.  Load Mazes
# ═══════════════════════════════════════════════════════════════
print("Loading mazes …")
mazes = np.load(DATA_PATH)
print(f"  {len(mazes)} mazes | shape {mazes[0].shape}\n")

assert mazes.ndim == 3 and mazes.shape[1:] == (16, 16), \
    "Expected shape (N, 16, 16)"
assert mazes[0,  0,  0] == 0, "Start cell [0,0] must be free"
assert mazes[0, 15, 15] == 0, "Goal cell [15,15] must be free"


# ═══════════════════════════════════════════════════════════════
# 7.  Gamma Scan
# ═══════════════════════════════════════════════════════════════
total_sims = len(GAMMA_VALUES) * len(mazes) * SCAN_TRIALS
print(f"Gamma scan")
print(f"  γ values  : {len(GAMMA_VALUES)}  ({GAMMA_VALUES[0]:.2f} … {GAMMA_VALUES[-1]:.2f})")
print(f"  Trials    : {SCAN_TRIALS} per maze")
print(f"  Total sims: {total_sims:,}")
print("─" * 54)

# 7.  Gamma Scan (EST Corrected)
# ═══════════════════════════════════════════════════════════════
print(f"Gamma scan (Optimizing for Expected Search Time)")
# ... (headers)

scan_sr   = np.zeros(len(GAMMA_VALUES))
scan_mfpt = np.full(len(GAMMA_VALUES), np.nan)
scan_est  = np.zeros(len(GAMMA_VALUES))

for i, g in enumerate(GAMMA_VALUES):
    sr, mfpt = eval_gamma(mazes, g, SCAN_TRIALS)
    scan_sr[i]   = sr
    scan_mfpt[i] = mfpt
    
    # Calculate EST: (SuccessRate * MFPT) + (FailureRate * Penalty)
    # We use MAX_STEPS as the penalty for failed runs.
    if sr > 0:
        scan_est[i] = (sr * mfpt) + ((1.0 - sr) * MAX_STEPS)
    else:
        scan_est[i] = MAX_STEPS

    mfpt_str = f"{mfpt:7.1f}" if not np.isnan(mfpt) else "    N/A"
    est_str  = f"{scan_est[i]:7.1f}" 
    print(f"  γ = {g:.2f}  │  SR = {sr:.4f}  │  MFPT = {mfpt_str}  │  EST = {est_str}")

# Find index of minimum Expected Search Time
opt_idx       = int(np.argmin(scan_est))
OPTIMAL_GAMMA = float(GAMMA_VALUES[opt_idx])

print("\n" + "═" * 54)
print(f"  OPTIMAL  γ  =  {OPTIMAL_GAMMA:.2f}")
print(f"  MFPT        =  {scan_mfpt[opt_idx]:.2f} steps")
print(f"  Success     =  {scan_sr[opt_idx]:.4f}")
print("═" * 54)

# ── Final validation at optimal γ ────────────────────────────
print(f"\nValidating at γ = {OPTIMAL_GAMMA:.2f}  ({FINAL_TRIALS} trials/maze) …")
final_sr, final_mfpt = eval_gamma(mazes, OPTIMAL_GAMMA, FINAL_TRIALS)
print(f"  Final SR   = {final_sr:.4f}")
print(f"  Final MFPT = {final_mfpt:.2f} steps")


# ═══════════════════════════════════════════════════════════════
# 8.  Plot — MFPT & Success Rate vs γ
# ═══════════════════════════════════════════════════════════════
fig_scan, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig_scan.suptitle("Hybrid GEO↔EGO Agent — MFPT & Success Rate vs Switching Rate γ",
                  fontsize=13, fontweight="bold")

# ── MFPT panel ────────────────────────────────────────────────
ax_m.plot(GAMMA_VALUES, scan_mfpt, "o-",
          color="steelblue", lw=2.5, ms=6,
          label="MFPT (successful runs only)")
ax_m.axvline(OPTIMAL_GAMMA, color="crimson", ls="--", lw=1.8,
             label=f"Optimal γ = {OPTIMAL_GAMMA:.2f}  "
                   f"(MFPT = {scan_mfpt[opt_idx]:.1f})")
ax_m.set_ylabel("Mean First Passage Time (steps)", fontsize=11)
ax_m.legend(fontsize=9)
ax_m.grid(True, alpha=0.3)
ax_m.set_xlim(-0.02, 1.02)

# ── Success rate panel ────────────────────────────────────────
ax_s.plot(GAMMA_VALUES, scan_sr, "s-",
          color="darkorange", lw=2.5, ms=6,
          label="Success Rate")
ax_s.axvline(OPTIMAL_GAMMA, color="crimson", ls="--", lw=1.8,
             label=f"Optimal γ = {OPTIMAL_GAMMA:.2f}  "
                   f"(SR = {scan_sr[opt_idx]:.3f})")
ax_s.set_xlabel("γ  —  probability of returning GEO from EGO per step",
                fontsize=11)
ax_s.set_ylabel("Success Rate", fontsize=11)
ax_s.set_ylim(0, 1.05)
ax_s.legend(fontsize=9)
ax_s.grid(True, alpha=0.3)

# Reference lines at γ = 0 (pure EGO after collision) and γ = 1 (near-pure GEO)
for ax in (ax_m, ax_s):
    ax.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.axvline(1.0, color="gray", ls=":", lw=1.0, alpha=0.6)

plt.tight_layout()
plt.savefig("mfpt_vs_gamma.png", dpi=150, bbox_inches="tight")
print("\nSaved: mfpt_vs_gamma.png")
plt.show()


# ═══════════════════════════════════════════════════════════════
# 9.  Interactive Viewer at Optimal γ
# ═══════════════════════════════════════════════════════════════
current_idx = 0
fig_v, ax_v = plt.subplots(figsize=(9, 9))
fig_v.patch.set_facecolor("#1a1a2e")
ax_v.set_facecolor("#16213e")


def update_plot():
    ax_v.clear()
    ax_v.set_facecolor("#16213e")

    maze = mazes[current_idx]
    path, acts, ents, ms = run_hybrid(maze, OPTIMAL_GAMMA)

    n_steps = len(acts)
    n_g     = int((ms == 0).sum())
    n_e     = int((ms == 1).sum())

    # ── Maze ──────────────────────────────────────────────────
    ax_v.imshow(maze, cmap="gray_r", extent=[-0.5, 15.5, 15.5, -0.5],
                alpha=0.85, zorder=1)

    if n_steps > 0:
        # ── Path segments coloured by mode ────────────────────
        path_xy  = path[:, ::-1].astype(float)   # (row,col) → (x,y)
        segs     = np.stack([path_xy[:-1], path_xy[1:]], axis=1)
        seg_clrs = [GEO_COLOR if m == 0 else EGO_COLOR for m in ms]
        lc = LineCollection(segs, colors=seg_clrs,
                            linewidths=1.8, alpha=0.55, zorder=2)
        ax_v.add_collection(lc)

        # ── Action arrows coloured by mode ────────────────────
        dy = np.array([-1,  1,  0,  0])[acts]
        dx = np.array([ 0,  0, -1,  1])[acts]

        for mode_val, color in [(0, GEO_COLOR), (1, EGO_COLOR)]:
            mask = ms == mode_val
            if not mask.any():
                continue
            ax_v.quiver(
                path[:-1][mask, 1], path[:-1][mask, 0],
                dx[mask], -dy[mask],
                color=color, alpha=0.85,
                scale=22, width=0.007, headwidth=4, zorder=3
            )

        # ── Mode-switch markers ───────────────────────────────
        # Mark every point where mode changes (GEO→EGO or EGO→GEO)
        switches = np.where(np.diff(ms) != 0)[0]   # indices in acts
        if switches.size:
            sw_pts = path[switches + 1]             # position after switch
            ax_v.scatter(sw_pts[:, 1], sw_pts[:, 0],
                         marker="D", s=28, c="gold",
                         zorder=4, alpha=0.8, linewidths=0.5,
                         edgecolors="white", label="mode switch")

    # ── Start / Goal markers ──────────────────────────────────
    ax_v.plot(0, 0, "go", ms=11, zorder=5)
    ax_v.plot(GOAL_POS[1], GOAL_POS[0], "r*", ms=18, zorder=5)

    # ── Legend ────────────────────────────────────────────────
    reached = np.array_equal(path[-1], GOAL_POS)
    status  = "✓  SUCCESS" if reached else "✗  FAILED / TIMED OUT"

    legend_elems = [
        Line2D([0],[0], color="lime",    marker="o",  lw=0, ms=9,
               label="Start [0,0]"),
        Line2D([0],[0], color="red",     marker="*",  lw=0, ms=13,
               label="Goal [15,15]"),
        Line2D([0],[0], color=GEO_COLOR, lw=2.5,
               label=f"GEO steps ({n_g})  — goal drift"),
        Line2D([0],[0], color=EGO_COLOR, lw=2.5,
               label=f"EGO steps ({n_e})  — obstacle escape"),
        Line2D([0],[0], color="gold",    marker="D",  lw=0, ms=7,
               label="Mode switch"),
    ]
    ax_v.legend(handles=legend_elems, loc="upper left",
                fontsize=8.5, framealpha=0.75)

    # ── Title ─────────────────────────────────────────────────
    ax_v.set_title(
        f"Hybrid GEO↔EGO  │  γ = {OPTIMAL_GAMMA:.2f}  │  Maze #{current_idx}\n"
        f"{status}  │  Steps: {n_steps}  │  GEO: {n_g}  EGO: {n_e}",
        fontsize=11, color="white"
    )
    ax_v.set_xlim(-0.5, 15.5)
    ax_v.set_ylim(15.5, -0.5)
    ax_v.set_aspect("equal")
    ax_v.tick_params(colors="white")
    for spine in ax_v.spines.values():
        spine.set_edgecolor("gray")

    plt.draw()


def on_key(event):
    global current_idx
    if event.key == "right":
        current_idx = (current_idx + 1) % len(mazes)
    elif event.key == "left":
        current_idx = (current_idx - 1) % len(mazes)
    update_plot()


fig_v.canvas.mpl_connect("key_press_event", on_key)

print(f"\nInteractive viewer  —  Optimal γ = {OPTIMAL_GAMMA:.2f}")
print("  ←  /  →  arrow keys to browse mazes")
print("  Gold diamonds = mode switch points\n")
update_plot()
plt.show()