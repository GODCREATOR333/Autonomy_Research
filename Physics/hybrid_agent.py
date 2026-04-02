"""
hybrid_sweep.py
===============
Physically accurate GEO ↔ EGO hybrid agent.

Physics summary
---------------
  GEO  : Discretised Von Mises on Z² — goal-directed drift, wall-blind.
           P_G(k|r) ∝ exp(κ · ê_k · û_g).  Wall hit = wasted step (steric).
  EGO  : Discrete Active Brownian Particle with rotational diffusion D_R.
           P_straight = exp(-D_R), P_turn = (1-exp(-D_R))/2, P_reverse = 0.
           Hard-core steric: wall directions masked, probability renormalised.
           1-step orientational memory via last_a.
  Switch: GEO → EGO on wall collision (deterministic, steric trigger).
          EGO → GEO with probability γ per step (geometric holding time,
          mean 1/γ steps in EGO — equivalent to CTMC tumbling rate).

Cost functions (both computed, both plotted)
--------------------------------------------
  C_EST(γ)  = SR · MFPT + (1−SR) · T_max          [Expected Search Time]
  C_WGT(γ)  = (1−SR)^β · T_max + SR · MFPT        [Failure-amplified EST]
               β > 1 makes failure super-linearly costly.

T_max (density-adaptive)
------------------------
  BFS every train maze → 95th percentile of shortest path lengths × k.
  Physically: agent allowed k× the hardest-but-legitimate path.

Configuration
-------------
  Edit SELECTED_DENSITIES, BETA, K_TMAX, SCAN_TRIALS, TEST_TRIALS below.
  Only selected densities are computed — run one at a time or all at once.

Usage
-----
  python hybrid_sweep.py
  ← → arrows to browse mazes in the interactive viewer.
"""

import os
import sys
import time
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════════
# ███  USER CONFIGURATION  ██████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════
DATA_DIR = "data_jax"

# ── Select which densities to compute ─────────────────────────────────────
# Keys   : human-readable label
# Values : file tag (P0100 = 10%, P0200 = 20%, etc.)
# Comment out any density you do NOT want to run this session.
DENSITY_MAP = {
    "10%": "P0100",
    "20%": "P0200",
    "30%": "P0300",
    "35%": "P0350",
    "37%": "P0370",
    "40%": "P0400",
}

# ← Edit this list to select which densities to run.
# Example: SELECTED_DENSITIES = ["10%", "40%"]  runs only those two.
SELECTED_DENSITIES = ["10%", "20%", "30%", "35%", "37%", "40%"]

# ── Physics ────────────────────────────────────────────────────────────────
KAPPA      = 2.5     # Von Mises concentration κ (GEO)
D_R        = 0.15    # Rotational diffusion coefficient (EGO)
GOAL_POS   = np.array([15, 15])
DELTAS     = np.array([[-1,0],[1,0],[0,-1],[0,1]])  # UP DOWN LEFT RIGHT

P_STRAIGHT = np.exp(-D_R)
P_TURN     = (1.0 - P_STRAIGHT) / 2.0

# ── Adaptive T_max ─────────────────────────────────────────────────────────
K_TMAX        = 5      # T_max = K_TMAX × BFS_95th_percentile
BFS_PERCENTILE = 95    # percentile of BFS path lengths used as reference
MIN_TMAX      = 90     # floor: at least 3× Manhattan distance (=30)

# ── Cost function ──────────────────────────────────────────────────────────
BETA = 2  # failure amplification exponent β  (1 = pure EST, 2 = quadratic)

# ── Gamma sweep ────────────────────────────────────────────────────────────
GAMMA_VALUES    = np.linspace(0.0, 1.0, 21)
TRAIN_SUBSAMPLE = 1500   # mazes from train set used for sweep (speed knob)
SCAN_TRIALS     = 10    # trials per maze during sweep
TEST_TRIALS     = 20    # trials per maze during test evaluation

# ── Maze types to test ─────────────────────────────────────────────────────
TEST_TYPES = ["random", "shapes", "symmetric"]

# ── Colours ────────────────────────────────────────────────────────────────
GEO_COLOR = "#4e9af1"
EGO_COLOR = "#f76e6e"
DENSITY_COLORS = {
    "10%": "#4e79a7", "20%": "#f28e2b", "30%": "#e15759",
    "35%": "#76b7b2", "37%": "#59a14f", "40%": "#b07aa1",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PHYSICS POLICIES
# ═══════════════════════════════════════════════════════════════════════════

# ── GEO: Discretised Von Mises ─────────────────────────────────────────────
def get_geo_probs(pos):
    """
    P_G(k|r) = softmax( κ · ê_k · û_g )
    Memoryless, wall-blind, goal-directed.
    """
    if np.array_equal(pos, GOAL_POS):
        return np.ones(4) / 4.0
    v = GOAL_POS.astype(float) - pos
    u = v / np.linalg.norm(v)
    q = DELTAS @ u                  # projection of each action onto goal dir
    e = np.exp(KAPPA * q)
    return e / e.sum()


# ── EGO: Discrete ABP with steric repulsion ─────────────────────────────────
_OPP = {0: 1, 1: 0, 2: 3, 3: 2}   # opposite directions

def get_ego_probs(win_id, last_a):
    """
    P_E(k | a_{t-1}, w) ∝ {P_straight if k==last_a,
                             P_turn    if k is lateral,
                             0         if k==opp(last_a)}
    × 1[w_k == 0]    (hard steric: wall directions zeroed, renormalised)
    Dead-end fallback: reverse if all forward+lateral blocked.
    """
    is_wall = np.array([
        (win_id >> 7) & 1,   # UP
        (win_id >> 1) & 1,   # DOWN
        (win_id >> 5) & 1,   # LEFT
        (win_id >> 3) & 1,   # RIGHT
    ], dtype=float)

    probs = np.zeros(4)
    if last_a == -1:          # no memory yet (start of episode)
        probs[:] = 0.25
    else:
        for k in range(4):
            if   k == last_a:         probs[k] = P_STRAIGHT
            elif k == _OPP[last_a]:   probs[k] = 0.0
            else:                      probs[k] = P_TURN

    probs[is_wall == 1] = 0.0  # hard steric repulsion
    s = probs.sum()
    if s > 0:
        probs /= s
    else:                       # true dead-end → reverse
        probs[_OPP.get(last_a, 0)] = 1.0
    return probs


def _win_id(padded, pos):
    """Encode 3×3 local window as 9-bit integer."""
    w = padded[pos[0]:pos[0]+3, pos[1]:pos[1]+3].flatten()
    return int((w * np.array([256,128,64,32,16,8,4,2,1])).sum())


def _entropy(p):
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


# ═══════════════════════════════════════════════════════════════════════════
# 2.  ADAPTIVE T_MAX via BFS
# ═══════════════════════════════════════════════════════════════════════════

def bfs_shortest_path(maze):
    """
    BFS from [0,0] to GOAL_POS on binary maze (0=free, 1=wall).
    Returns path length (int) or None if unsolvable.
    """
    H, W = maze.shape
    start = (0, 0)
    goal  = tuple(GOAL_POS)
    if maze[0,0] == 1 or maze[goal[0],goal[1]] == 1:
        return None
    dist = {start: 0}
    q    = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return dist[goal]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (0<=nr<H and 0<=nc<W
                    and maze[nr,nc] == 0
                    and (nr,nc) not in dist):
                dist[(nr,nc)] = dist[(r,c)] + 1
                q.append((nr,nc))
    return None


def compute_adaptive_tmax(train_mazes, percentile=BFS_PERCENTILE, k=K_TMAX):
    """
    Run BFS on each train maze, collect shortest path lengths,
    return k × p-th percentile (floor MIN_TMAX).
    """
    lengths = []
    for maze in train_mazes:
        L = bfs_shortest_path(maze)
        if L is not None:
            lengths.append(L)
    if not lengths:
        return MIN_TMAX
    ref = np.percentile(lengths, percentile)
    tmax = max(MIN_TMAX, int(np.ceil(k * ref)))
    return tmax, float(np.mean(lengths)), float(np.median(lengths)), float(ref)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  HYBRID SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def run_hybrid(maze, gamma, max_steps):
    """
    One episode of the GEO↔EGO hybrid agent.

    Switching physics
    -----------------
      GEO wall collision → GEO→EGO (deterministic steric trigger, last_a = blocked dir)
      EGO step           → EGO→GEO with prob γ (geometric holding time, mean 1/γ)

    Returns
    -------
    path    (T+1, 2)  int    positions (row, col)
    acts    (T,)      int    action taken each step
    modes   (T,)      int    0=GEO, 1=EGO  (mode that generated the action)
    ents    (T,)      float  entropy of action distribution
    """
    H, W   = maze.shape
    padded = np.pad(maze, 1, constant_values=1)

    pos    = np.array([0, 0])
    mode   = 0      # start in GEO
    last_a = -1     # no orientational memory yet

    path  = [tuple(pos)]
    acts  = []
    modes = []
    ents  = []

    for _ in range(max_steps):
        mode_used = mode

        # ── Sample action from active policy ─────────────────
        if mode == 0:
            probs = get_geo_probs(pos)
        else:
            probs = get_ego_probs(_win_id(padded, pos), last_a)

        ents.append(_entropy(probs))
        a    = np.random.choice(4, p=probs)
        npos = pos + DELTAS[a]

        # ── Physics: hard-wall steric enforcement ─────────────
        moved = (0 <= npos[0] < H and 0 <= npos[1] < W
                 and maze[npos[0], npos[1]] == 0)
        if moved:
            pos    = npos
            last_a = a
        elif mode == 0:
            # GEO wall collision → engage EGO
            # last_a = blocked dir: EGO masks it and naturally turns laterally
            mode   = 1
            last_a = a

        # ── Mode transition: EGO→GEO with rate γ ─────────────
        # Executed only after an EGO step (geometric sojourn time)
        if mode_used == 1 and np.random.random() < gamma:
            mode = 0

        acts.append(a)
        modes.append(mode_used)
        path.append(tuple(pos))

        if np.array_equal(pos, GOAL_POS):
            break

    return (np.array(path),
            np.array(acts),
            np.array(modes),
            np.array(ents))


# ═══════════════════════════════════════════════════════════════════════════
# 4.  FULL METRICS EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(mazes, gamma, trials, max_steps):
    """
    Run trials × len(mazes) episodes and return a comprehensive metrics dict.

    Metrics
    -------
    success_rate      : fraction of runs reaching goal
    timeout_rate      : fraction hitting max_steps without goal
    mfpt              : mean steps (successful runs only); np.nan if none
    est               : Expected Search Time = SR·MFPT + (1-SR)·T_max
    cost_wgt          : Failure-amplified EST = (1-SR)^β · T_max + SR·MFPT
    avg_final_dist    : mean Euclidean distance to goal at run end (all runs)
    avg_ego_steps     : mean EGO steps per trajectory
    avg_geo_steps     : mean GEO steps per trajectory
    avg_ego_fraction  : mean fraction of steps spent in EGO mode
    avg_ego_switches  : mean number of GEO→EGO transitions per trajectory
    avg_entropy_geo   : mean action entropy during GEO steps
    avg_entropy_ego   : mean action entropy during EGO steps
    avg_steps_total   : mean total steps per trajectory (all runs)
    n_total           : total number of runs
    n_success         : number of successful runs
    """
    hit_steps     = []
    total_steps   = []
    final_dists   = []
    ego_steps_all = []
    geo_steps_all = []
    ego_frac_all  = []
    ego_sw_all    = []
    ent_geo_all   = []
    ent_ego_all   = []

    n_total = len(mazes) * trials

    for maze in mazes:
        for _ in range(trials):
            path, acts, modes, ents = run_hybrid(maze, gamma, max_steps)
            T      = len(acts)
            reached = np.array_equal(path[-1], GOAL_POS)

            if reached:
                hit_steps.append(T)

            total_steps.append(T)
            final_dists.append(
                float(np.linalg.norm(path[-1].astype(float) - GOAL_POS)))

            n_ego = int((modes == 1).sum())
            n_geo = int((modes == 0).sum())
            ego_steps_all.append(n_ego)
            geo_steps_all.append(n_geo)
            ego_frac_all.append(n_ego / max(1, T))

            # Count GEO→EGO switches
            if T > 1:
                sw = int(np.sum((modes[:-1] == 0) & (modes[1:] == 1)))
            else:
                sw = 0
            ego_sw_all.append(sw)

            # Entropy per mode
            if n_geo > 0:
                ent_geo_all.append(float(np.mean(ents[modes == 0])))
            if n_ego > 0:
                ent_ego_all.append(float(np.mean(ents[modes == 1])))

    n_success = len(hit_steps)
    sr        = n_success / n_total
    mfpt      = float(np.mean(hit_steps)) if hit_steps else np.nan
    est       = sr * (mfpt if not np.isnan(mfpt) else max_steps) + (1-sr)*max_steps
    cost_wgt  = (1-sr)**BETA * max_steps + sr * (mfpt if not np.isnan(mfpt) else max_steps)

    return {
        "success_rate":      sr,
        "timeout_rate":      1.0 - sr,
        "mfpt":              mfpt,
        "est":               est,
        "cost_wgt":          cost_wgt,
        "avg_final_dist":    float(np.mean(final_dists)),
        "avg_ego_steps":     float(np.mean(ego_steps_all)),
        "avg_geo_steps":     float(np.mean(geo_steps_all)),
        "avg_ego_fraction":  float(np.mean(ego_frac_all)),
        "avg_ego_switches":  float(np.mean(ego_sw_all)),
        "avg_entropy_geo":   float(np.mean(ent_geo_all)) if ent_geo_all else np.nan,
        "avg_entropy_ego":   float(np.mean(ent_ego_all)) if ent_ego_all else np.nan,
        "avg_steps_total":   float(np.mean(total_steps)),
        "n_total":           n_total,
        "n_success":         n_success,
        "tmax_used":         max_steps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FILE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_maze_file(tag, split, maze_type=None):
    """Load a .npy maze file.  Returns array or None if missing."""
    if split == "train":
        fname = f"N16_{tag}_train_solvable.npy"
    else:
        fname = f"N16_{tag}_test_solvable_{maze_type}.npy"
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MAIN SWEEP LOOP
# ═══════════════════════════════════════════════════════════════════════════

results_store = {}   # density_label → full results dict (sweep + test)

wall = time.time()
print("═"*72)
print("  Hybrid GEO↔EGO — Density Sweep with Adaptive T_max")
print(f"  Densities selected : {SELECTED_DENSITIES}")
print(f"  β (failure amplif) : {BETA}")
print(f"  k × BFS-{BFS_PERCENTILE}th pct   : T_max")
print("═"*72)

for label in SELECTED_DENSITIES:
    if label not in DENSITY_MAP:
        print(f"\n[WARN] Unknown density '{label}' — skipped.")
        continue
    tag = DENSITY_MAP[label]

    print(f"\n{'━'*72}")
    print(f"  DENSITY  {label}  ({tag})")
    print(f"{'━'*72}")

    # ── Load train ───────────────────────────────────────────────────────
    train_all = load_maze_file(tag, "train")
    if train_all is None:
        print(f"  [SKIP] Train file missing for {tag}")
        continue

    # Subsample for sweep speed
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(train_all),
                      min(TRAIN_SUBSAMPLE, len(train_all)),
                      replace=False)
    train_sub = train_all[idx]
    print(f"  Train: {len(train_all)} mazes  → subsampled {len(train_sub)}")

    # ── Adaptive T_max ────────────────────────────────────────────────────
    t0 = time.time()
    tmax_result = compute_adaptive_tmax(train_all)   # use full train for BFS
    tmax, bfs_mean, bfs_med, bfs_p95 = tmax_result
    print(f"  BFS paths  — mean: {bfs_mean:.1f}  median: {bfs_med:.1f}"
          f"  p95: {bfs_p95:.1f}  → T_max = {K_TMAX}×{bfs_p95:.0f} = {tmax}"
          f"  ({time.time()-t0:.1f}s)")

    # ── Gamma sweep on train ──────────────────────────────────────────────
    n_sweep_sims = len(GAMMA_VALUES) * len(train_sub) * SCAN_TRIALS
    print(f"  Sweeping {len(GAMMA_VALUES)} γ values "
          f"({SCAN_TRIALS} trials × {len(train_sub)} mazes"
          f" = {n_sweep_sims:,} sims) …")

    sweep = {
        "gamma":    GAMMA_VALUES.copy(),
        "sr":       np.zeros(len(GAMMA_VALUES)),
        "mfpt":     np.full(len(GAMMA_VALUES), np.nan),
        "est":      np.zeros(len(GAMMA_VALUES)),
        "cost_wgt": np.zeros(len(GAMMA_VALUES)),
    }

    t0 = time.time()
    print(f"  {'γ':>5}  {'SR':>7}  {'MFPT':>8}  {'EST':>8}  {'C_wgt':>8}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")

    for i, g in enumerate(GAMMA_VALUES):
        m = evaluate(train_sub, g, SCAN_TRIALS, tmax)
        sweep["sr"][i]       = m["success_rate"]
        sweep["mfpt"][i]     = m["mfpt"]
        sweep["est"][i]      = m["est"]
        sweep["cost_wgt"][i] = m["cost_wgt"]

        mfpt_s = f"{m['mfpt']:8.1f}" if not np.isnan(m['mfpt']) else "     N/A"
        print(f"  {g:5.2f}  {m['success_rate']:7.4f}  "
              f"{mfpt_s}  {m['est']:8.1f}  {m['cost_wgt']:8.1f}")

    elapsed = time.time() - t0
    print(f"  Sweep done in {elapsed:.1f}s")

    # ── Optimal γ by each cost ─────────────────────────────────────────────
    opt_est_idx  = int(np.nanargmin(sweep["est"]))
    opt_wgt_idx  = int(np.nanargmin(sweep["cost_wgt"]))
    opt_est_g    = float(GAMMA_VALUES[opt_est_idx])
    opt_wgt_g    = float(GAMMA_VALUES[opt_wgt_idx])

    print(f"\n  Optimal γ (EST cost)  : {opt_est_g:.2f}"
          f"  →  EST={sweep['est'][opt_est_idx]:.1f}"
          f"  SR={sweep['sr'][opt_est_idx]:.4f}")
    print(f"  Optimal γ (Wgt cost)  : {opt_wgt_g:.2f}"
          f"  →  C_wgt={sweep['cost_wgt'][opt_wgt_idx]:.1f}"
          f"  SR={sweep['sr'][opt_wgt_idx]:.4f}")

    # Use weighted cost as primary (higher SR weight); EST as reference
    opt_gamma = opt_wgt_g

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f"\n  Testing  γ={opt_gamma:.2f}  ({TEST_TRIALS} trials/maze) …")
    test_metrics = {}

    for ttype in TEST_TYPES:
        test_mazes = load_maze_file(tag, "test", maze_type=ttype)
        if test_mazes is None:
            print(f"    [{ttype:10s}]  file missing — skipped")
            continue

        m = evaluate(test_mazes, opt_gamma, TEST_TRIALS, tmax)
        test_metrics[ttype] = m

        mfpt_s = f"{m['mfpt']:.2f}" if not np.isnan(m['mfpt']) else "  N/A  "
        print(f"    [{ttype:10s}]  "
              f"SR={m['success_rate']:.4f}  "
              f"TO={m['timeout_rate']:.4f}  "
              f"MFPT={mfpt_s:>8}  "
              f"EST={m['est']:7.1f}  "
              f"EGO%={m['avg_ego_fraction']*100:5.1f}  "
              f"dist={m['avg_final_dist']:.3f}")

    results_store[label] = {
        "tag":          tag,
        "tmax":         tmax,
        "bfs_mean":     bfs_mean,
        "bfs_p95":      bfs_p95,
        "sweep":        sweep,
        "opt_gamma_est": opt_est_g,
        "opt_gamma_wgt": opt_wgt_g,
        "opt_gamma":    opt_gamma,
        "test":         test_metrics,
    }

total_elapsed = time.time() - wall
print(f"\n{'═'*72}")
print(f"  All selected densities done in {total_elapsed/60:.1f} min")
print(f"{'═'*72}")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "═"*80)
print("  SUMMARY — Optimal γ and Test Metrics per Density")
print("═"*80)

metric_cols = ["success_rate","timeout_rate","mfpt","est",
               "avg_ego_fraction","avg_ego_switches","avg_final_dist"]
col_w = 10

header = (f"{'Density':>8}  {'T_max':>6}  {'γ*':>5}  "
          + "".join(f"  {c[:col_w]:>{col_w}}" for c in metric_cols))

for label in SELECTED_DENSITIES:
    if label not in results_store:
        continue
    R   = results_store[label]
    og  = R["opt_gamma"]
    tmx = R["tmax"]
    print(f"\n  {label} (T_max={tmx}, γ*={og:.2f})")
    print(f"  {'Type':<12}" +
          "".join(f"  {c[:col_w]:>{col_w}}" for c in metric_cols))
    print("  " + "─"*(12 + (col_w+2)*len(metric_cols)))

    for ttype, m in R["test"].items():
        vals = []
        for c in metric_cols:
            v = m[c]
            if np.isnan(v):
                vals.append(f"{'N/A':>{col_w}}")
            elif c in ("success_rate","timeout_rate","avg_ego_fraction"):
                vals.append(f"{v*100:>{col_w-1}.2f}%")
            elif c == "avg_ego_switches":
                vals.append(f"{v:>{col_w}.2f}")
            else:
                vals.append(f"{v:>{col_w}.1f}")
        print(f"  {ttype:<12}" + "".join(f"  {v}" for v in vals))


# ═══════════════════════════════════════════════════════════════════════════
# 8.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════
computed = [d for d in SELECTED_DENSITIES if d in results_store]
n_dens   = len(computed)

if n_dens == 0:
    print("No results to plot.")
    sys.exit(0)

# ── 8a. Per-density gamma sweep panels (EST + WGT cost + SR) ─────────────
ncols = min(3, n_dens)
nrows = (n_dens + ncols - 1) // ncols
fig_sw, axes_sw = plt.subplots(nrows, ncols,
                               figsize=(5.5*ncols, 4.5*nrows),
                               squeeze=False)
fig_sw.suptitle(
    f"Gamma Sweep per Density — EST & Failure-Amplified Cost (β={BETA})",
    fontsize=13, fontweight="bold")

for ai, label in enumerate(computed):
    R    = results_store[label]
    sw   = R["sweep"]
    col  = DENSITY_COLORS.get(label, "steelblue")
    r, c = divmod(ai, ncols)
    ax   = axes_sw[r][c]

    lns = []
    l1, = ax.plot(sw["gamma"], sw["est"],  "o-", color=col,
                  lw=2.2, ms=5, label="EST  (β=1)")
    l2, = ax.plot(sw["gamma"], sw["cost_wgt"], "s--", color=col,
                  alpha=0.65, lw=1.8, ms=4, label=f"C_wgt (β={BETA})")
    lns += [l1, l2]

    ax2 = ax.twinx()
    l3, = ax2.plot(sw["gamma"], sw["sr"], "^:", color="gray",
                   lw=1.4, ms=4, label="Success Rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("SR", fontsize=8, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

    og_est = R["opt_gamma_est"]
    og_wgt = R["opt_gamma_wgt"]
    ax.axvline(og_wgt, color="crimson", ls="-", lw=1.8,
               label=f"γ*(C_wgt)={og_wgt:.2f}")
    ax.axvline(og_est, color="navy",    ls=":", lw=1.4,
               label=f"γ*(EST)={og_est:.2f}")

    ax.set_title(f"Density {label}  |  T_max={R['tmax']}", fontsize=10)
    ax.set_xlabel("γ", fontsize=9)
    ax.set_ylabel("Cost (steps)", fontsize=9)
    ax.grid(True, alpha=0.25)
    handles = lns + [l3]
    labels  = [h.get_label() for h in handles]
    ax.legend(handles, labels, fontsize=7, loc="upper right")

for ai in range(n_dens, nrows*ncols):
    r, c = divmod(ai, ncols)
    axes_sw[r][c].set_visible(False)

plt.tight_layout()
plt.savefig("sweep_per_density.png", dpi=150, bbox_inches="tight")
print("\nSaved: sweep_per_density.png")


# ── 8b. Optimal γ vs Density (both cost functions) ────────────────────────
dens_pct  = []
og_est_arr = []
og_wgt_arr = []
for label in computed:
    pct_str = DENSITY_MAP[label][1:]       # "P0100" → "0100"
    dens_pct.append(int(pct_str) / 10000)  # → 0.10
    og_est_arr.append(results_store[label]["opt_gamma_est"])
    og_wgt_arr.append(results_store[label]["opt_gamma_wgt"])

fig_og, ax_og = plt.subplots(figsize=(8, 4.5))
ax_og.plot(dens_pct, og_est_arr, "D-",  color="navy",   lw=2.2, ms=8,
           label="γ*(EST)")
ax_og.plot(dens_pct, og_wgt_arr, "o--", color="crimson", lw=2.2, ms=8,
           label=f"γ*(C_wgt, β={BETA})")
for x, y1, y2, lbl in zip(dens_pct, og_est_arr, og_wgt_arr, computed):
    ax_og.annotate(lbl, (x, max(y1, y2)),
                   textcoords="offset points", xytext=(4, 6), fontsize=8)
ax_og.set_xlabel("Obstacle Density", fontsize=11)
ax_og.set_ylabel("Optimal Switching Rate γ*", fontsize=11)
ax_og.set_title("Optimal Switching Rate vs Obstacle Density", fontsize=12)
ax_og.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax_og.set_ylim(0, 1.1)
ax_og.legend(fontsize=10)
ax_og.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("optimal_gamma_vs_density.png", dpi=150, bbox_inches="tight")
print("Saved: optimal_gamma_vs_density.png")


# ── 8c. Test metrics across densities × maze type ─────────────────────────
metrics_to_plot = {
    "MFPT (steps)":         "mfpt",
    "Success Rate":         "success_rate",
    "EGO Fraction (%)":     "avg_ego_fraction",
    "Avg Final Distance":   "avg_final_dist",
}
type_colors  = {"random": "#4e79a7", "shapes": "#e15759", "symmetric": "#59a14f"}
type_markers = {"random": "o",       "shapes": "s",       "symmetric": "^"}

fig_tm, axs_tm = plt.subplots(2, 2, figsize=(13, 9))
fig_tm.suptitle("Test Metrics at Optimal γ* per Density",
                fontsize=13, fontweight="bold")
axs_flat = axs_tm.flatten()

for ax, (mname, mkey) in zip(axs_flat, metrics_to_plot.items()):
    for ttype in TEST_TYPES:
        xs, ys = [], []
        for label in computed:
            tdata = results_store[label]["test"]
            if ttype not in tdata:
                continue
            pct_str = DENSITY_MAP[label][1:]
            xs.append(int(pct_str) / 10000)
            val = tdata[ttype][mkey]
            if mkey == "avg_ego_fraction":
                val *= 100   # → percent
            ys.append(val)
        if xs:
            ax.plot(xs, ys, marker=type_markers[ttype],
                    color=type_colors[ttype], lw=2.0, ms=7,
                    label=ttype.capitalize())

    ax.set_xlabel("Obstacle Density", fontsize=9)
    ax.set_ylabel(mname, fontsize=9)
    ax.set_title(mname, fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("test_metrics_per_density.png", dpi=150, bbox_inches="tight")
print("Saved: test_metrics_per_density.png")


# ═══════════════════════════════════════════════════════════════════════════
# 9.  INTERACTIVE VIEWER
# ═══════════════════════════════════════════════════════════════════════════
# Build a flat list of (density_label, maze_array) for browsing.
# User presses ← → to browse mazes, D to cycle density.

viewer_densities = computed
viewer_dens_idx  = 0
viewer_maze_idx  = 0

# Pre-load one test set (random) per density for the viewer
viewer_mazes = {}
for label in viewer_densities:
    tag  = DENSITY_MAP[label]
    m    = load_maze_file(tag, "test", maze_type="random")
    if m is None:
        m = load_maze_file(tag, "train")
    viewer_mazes[label] = m

fig_v, ax_v = plt.subplots(figsize=(9, 9))
fig_v.patch.set_facecolor("#0d1117")
ax_v.set_facecolor("#161b22")


def viewer_update():
    ax_v.clear()
    ax_v.set_facecolor("#161b22")

    label  = viewer_densities[viewer_dens_idx]
    R      = results_store[label]
    gamma  = R["opt_gamma"]
    tmax   = R["tmax"]
    mazes  = viewer_mazes[label]

    if mazes is None or len(mazes) == 0:
        ax_v.set_title("No mazes available for this density", color="white")
        plt.draw()
        return

    midx = viewer_maze_idx % len(mazes)
    maze = mazes[midx]

    path, acts, modes, ents = run_hybrid(maze, gamma, tmax)
    T    = len(acts)
    n_g  = int((modes == 0).sum())
    n_e  = int((modes == 1).sum())
    sw   = int(np.sum((modes[:-1]==0) & (modes[1:]==1))) if T > 1 else 0
    reached = np.array_equal(path[-1], GOAL_POS)

    # ── Maze background ───────────────────────────────────────
    ax_v.imshow(maze, cmap="gray_r",
                extent=[-0.5, 15.5, 15.5, -0.5], alpha=0.82, zorder=1)

    if T > 0:
        # ── Coloured path segments ────────────────────────────
        xy   = path[:, ::-1].astype(float)   # (row,col)→(x,y)
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        clrs = [GEO_COLOR if m == 0 else EGO_COLOR for m in modes]
        lc   = LineCollection(segs, colors=clrs, lw=1.8, alpha=0.5, zorder=2)
        ax_v.add_collection(lc)

        # ── Action arrows ─────────────────────────────────────
        dy_arr = np.array([-1, 1, 0, 0])[acts]
        dx_arr = np.array([ 0, 0,-1, 1])[acts]
        for mv, col in [(0, GEO_COLOR), (1, EGO_COLOR)]:
            msk = modes == mv
            if not msk.any():
                continue
            ax_v.quiver(
                path[:-1][msk, 1], path[:-1][msk, 0],
                dx_arr[msk], -dy_arr[msk],
                color=col, alpha=0.88,
                scale=22, width=0.007, headwidth=4, zorder=3)

        # ── Mode switch markers ───────────────────────────────
        if T > 1:
            sw_idx = np.where((modes[:-1]==0) & (modes[1:]==1))[0]
            if sw_idx.size:
                spts = path[sw_idx + 1]
                ax_v.scatter(spts[:, 1], spts[:, 0],
                             marker="D", s=30, c="gold",
                             zorder=4, alpha=0.85, linewidths=0.4,
                             edgecolors="white")

    # ── Start / goal ──────────────────────────────────────────
    ax_v.plot(0, 0, "go", ms=11, zorder=5)
    ax_v.plot(GOAL_POS[1], GOAL_POS[0], "r*", ms=18, zorder=5)

    # ── Legend ────────────────────────────────────────────────
    status = "✓ SUCCESS" if reached else "✗ FAILED"
    legend_elems = [
        Line2D([0],[0], color="lime",    marker="o", lw=0, ms=9,
               label="Start [0,0]"),
        Line2D([0],[0], color="red",     marker="*", lw=0, ms=13,
               label="Goal [15,15]"),
        Line2D([0],[0], color=GEO_COLOR, lw=2.5,
               label=f"GEO ({n_g} steps)"),
        Line2D([0],[0], color=EGO_COLOR, lw=2.5,
               label=f"EGO ({n_e} steps)"),
        Line2D([0],[0], color="gold",    marker="D", lw=0, ms=7,
               label=f"Mode switches ({sw})"),
    ]
    ax_v.legend(handles=legend_elems, loc="upper left",
                fontsize=8.5, framealpha=0.78)

    ax_v.set_title(
        f"Density {label}  │  γ*={gamma:.2f}  │  T_max={tmax}  │  Maze #{midx}\n"
        f"{status}  │  Steps={T}  │  GEO={n_g}  EGO={n_e}  "
        f"Switches={sw}\n"
        f"[← →] browse mazes     [D] cycle density",
        fontsize=10, color="white", linespacing=1.5)

    ax_v.set_xlim(-0.5, 15.5)
    ax_v.set_ylim(15.5, -0.5)
    ax_v.set_aspect("equal")
    ax_v.tick_params(colors="white")
    for sp in ax_v.spines.values():
        sp.set_edgecolor("#444")
    plt.draw()


def on_key(event):
    global viewer_maze_idx, viewer_dens_idx
    if event.key == "right":
        viewer_maze_idx += 1
    elif event.key == "left":
        viewer_maze_idx = max(0, viewer_maze_idx - 1)
    elif event.key in ("d", "D"):
        viewer_dens_idx = (viewer_dens_idx + 1) % len(viewer_densities)
        viewer_maze_idx = 0
    viewer_update()


fig_v.canvas.mpl_connect("key_press_event", on_key)

print("\nInteractive viewer ready.")
print("  ← →  : browse mazes")
print("  D     : cycle density\n")
viewer_update()
plt.show()