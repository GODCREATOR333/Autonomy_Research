"""
visualize_mazes.py
==================
For every .npy file in data_dir (each containing 100 mazes):
    For each of the 100 mazes → one figure with 3 panels side by side:
        1. Raw maze          (walls=black, free=white, start=red, goal=green)
        2. BFS solution      (same + magenta path)
        3. Value iteration   (inferno heatmap, walls masked dark)

Output: one PNG per maze, organised as:
    output_dir/
        N16_p0100/
            maze_000.png
            maze_001.png
            ...
            maze_099.png
        N32_p0200/
            ...

Usage:
    python visualize_mazes.py \
        --data_dir   ~/Autonomy_research/data/maze_dataset \
        --output_dir ~/Autonomy_research/debug_plots
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from collections import deque
from pathlib import Path
import re


# ── BFS ───────────────────────────────────────────────────────────────────────

def bfs_path(maze):
    N     = maze.shape[0]
    start = (0, 0)
    goal  = (N-1, N-1)
    queue   = deque([(start, [start])])
    visited = {start}
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (0 <= nr < N and 0 <= nc < N
                    and maze[nr, nc] == 0
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


# ── Value Iteration ───────────────────────────────────────────────────────────

def run_value_iteration(maze, gamma=0.9, theta=1e-4):
    N            = maze.shape[0]
    V            = np.zeros((N, N))
    step_reward  = -1
    wall_penalty = -50
    goal_reward  = 500

    V[N-1, N-1] = goal_reward

    while True:
        delta = 0
        new_V = np.copy(V)
        new_V[N-1, N-1] = goal_reward

        for r in range(N):
            for c in range(N):
                if maze[r, c] == 1 or (r == N-1 and c == N-1):
                    continue
                vals = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < N and 0 <= nc < N:
                        if maze[nr, nc] == 0:
                            if nr == N-1 and nc == N-1:
                                vals.append(goal_reward)
                            else:
                                vals.append(step_reward + gamma * V[nr, nc])
                        else:
                            vals.append(wall_penalty + gamma * V[r, c])
                    else:
                        vals.append(wall_penalty + gamma * V[r, c])
                new_V[r, c] = max(vals)
                delta = max(delta, abs(new_V[r,c] - V[r,c]))
        V = new_V
        if delta < theta:
            break
    return V


# ── Panel helpers ─────────────────────────────────────────────────────────────

WALL_COLOR = '#1c1c1e'
FREE_COLOR = '#f7f3ed'
MAZE_CMAP  = ListedColormap([FREE_COLOR, WALL_COLOR])


def _base_maze(ax, maze):
    """Draw raw maze — walls black, free off-white."""
    ax.imshow(maze, cmap=MAZE_CMAP, origin='upper',
              interpolation='nearest', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_edgecolor('#1c1c1e')
        sp.set_linewidth(1.5)


def _start_goal(ax, N):
    """Plot start (red) and goal (green) markers."""
    ax.scatter(0,   0,   s=90, color='#e74c3c',
               edgecolors='white', linewidths=0.8, zorder=6)
    ax.scatter(N-1, N-1, s=90, color='#2ecc71',
               edgecolors='white', linewidths=0.8, zorder=6)


def panel_raw(ax, maze, N, p, maze_idx):
    """Panel 1 — plain maze."""
    _base_maze(ax, maze)
    _start_goal(ax, N)

    ax.set_title(f'Maze  {maze_idx:03d}',
                 fontsize=9, pad=5, color='#1c1c1e',
                 fontfamily='monospace')

    legend = [
        mpatches.Patch(facecolor=WALL_COLOR, label='Wall'),
        mpatches.Patch(facecolor=FREE_COLOR,
                       edgecolor='#bbbbbb', label='Free'),
        mpatches.Patch(facecolor='#e74c3c', label='Start (0,0)'),
        mpatches.Patch(facecolor='#2ecc71',
                       label=f'Goal ({N-1},{N-1})'),
    ]
    ax.legend(handles=legend, fontsize=6.5, loc='upper right',
              framealpha=0.92, edgecolor='#cccccc',
              fancybox=False, borderpad=0.5)


def panel_bfs(ax, maze, path, N):
    """Panel 2 — maze + magenta BFS path."""
    _base_maze(ax, maze)
    _start_goal(ax, N)

    if path is not None:
        arr = np.array(path)
        ax.plot(arr[:, 1], arr[:, 0],
                color='#ff00cc', linewidth=1.8,
                alpha=0.9, zorder=4, solid_capstyle='round')
        ax.scatter(arr[:, 1], arr[:, 0],
                   s=6, color='#ff00cc', alpha=0.6, zorder=4)
        status   = 'solvable'
        path_len = len(path) - 1
    else:
        status   = 'unsolvable'
        path_len = 0

    ax.set_title(f'BFS Path  [{status}  ·  {path_len} steps]',
                 fontsize=9, pad=5, color='#1c1c1e',
                 fontfamily='monospace')

    legend = [
        mpatches.Patch(facecolor='#e74c3c', label='Start'),
        mpatches.Patch(facecolor='#2ecc71', label='Goal'),
        Line2D([0],[0], color='#ff00cc', linewidth=2, label='BFS path'),
    ]
    ax.legend(handles=legend, fontsize=6.5, loc='upper right',
              framealpha=0.92, edgecolor='#cccccc',
              fancybox=False, borderpad=0.5)


def panel_vi(ax, maze, V, path, N):
    """Panel 3 — VI value heatmap, walls masked, BFS path overlay."""
    V_show             = V.astype(float).copy()
    V_show[maze == 1]  = np.nan

    cmap_vi = plt.cm.inferno.copy()
    cmap_vi.set_bad(color=WALL_COLOR)

    norm = Normalize(vmin=np.nanmin(V_show), vmax=np.nanmax(V_show))
    im   = ax.imshow(V_show, cmap=cmap_vi, origin='upper',
                     norm=norm, interpolation='nearest')

    # BFS path in cyan for contrast over inferno
    if path is not None:
        arr = np.array(path)
        ax.plot(arr[:, 1], arr[:, 0],
                color='#00e5ff', linewidth=1.6,
                alpha=0.85, zorder=4, solid_capstyle='round')

    # Start / goal
    ax.scatter(0,   0,   s=70, color='#e74c3c',
               edgecolors='white', linewidths=0.8, zorder=5)
    ax.scatter(N-1, N-1, s=70, color='#2ecc71',
               edgecolors='white', linewidths=0.8, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_edgecolor('#1c1c1e')
        sp.set_linewidth(1.5)

    ax.set_title('Value Iteration  V(s)',
                 fontsize=9, pad=5, color='#1c1c1e',
                 fontfamily='monospace')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('V(s)', rotation=270, labelpad=14,
                   fontsize=8, color='#1c1c1e')
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_edgecolor('#cccccc')


# ── Single maze figure ────────────────────────────────────────────────────────

def make_figure(maze, maze_idx, N, p, save_path):
    path = bfs_path(maze)
    V    = run_value_iteration(maze)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=300)

    fig.suptitle(
        f'N={N}×{N}   p={p:.3f}   maze {maze_idx:03d}',
        fontsize=11, fontweight='bold',
        color='#1c1c1e', y=1.02,
        fontfamily='monospace',
    )

    panel_raw(axes[0], maze, N, p, maze_idx)
    panel_bfs(axes[1], maze, path, N)
    panel_vi (axes[2], maze, V, path, N)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--filter_N',   type=int, default=None,
                        help='Only process this grid size')
    parser.add_argument('--filter_p',   type=float, default=None,
                        help='Only process this density e.g. 0.3')
    args = parser.parse_args()

    data_path   = Path(args.data_dir)
    output_path = Path(args.output_dir)

    npy_files = sorted(data_path.glob('*.npy'))
    if not npy_files:
        print(f'No .npy files in {data_path}')
        return

    print(f'Found {len(npy_files)} files\n')

    for fpath in npy_files:
        match = re.match(r'N(\d+)_p(\d+)\.npy', fpath.name)
        if not match:
            print(f'Skipping {fpath.name}')
            continue

        N = int(match.group(1))
        p = int(match.group(2)) / 1000.0

        if args.filter_N is not None and N != args.filter_N:
            continue
        if args.filter_p is not None and abs(p - args.filter_p) > 1e-4:
            continue

        mazes = np.load(fpath)
        if mazes.ndim != 3:
            print(f'Unexpected shape {mazes.shape} — skipping {fpath.name}')
            continue

        n_mazes   = mazes.shape[0]
        out_dir   = output_path / f'N{N}_p{int(p*1000):04d}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'N={N:3d}  p={p:.3f}  →  {n_mazes} mazes  →  {out_dir}')

        for i in range(n_mazes):
            save_path = out_dir / f'maze_{i:03d}.png'
            make_figure(mazes[i], i, N, p, save_path)

            if (i + 1) % 10 == 0:
                print(f'  {i+1:3d}/{n_mazes}')

        print(f'  Done\n')

    print('All done.')


if __name__ == '__main__':
    main()