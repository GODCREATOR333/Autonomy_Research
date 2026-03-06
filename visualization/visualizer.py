import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

def plot_research_triplet(maze, path, V, maze_idx, save_path):
    """
    Enhanced aesthetic visualization for maze + path + value iteration.
    """
    N = maze.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)

    # --- Custom colormap for maze ---
    maze_cmap = ListedColormap(["white", "black"])  # free=white, wall=black

    # 1. Maze + Path
    ax1 = axes[0]
    ax1.imshow(maze, cmap=maze_cmap, origin='upper')
    
    # Overlay BFS path
    if path:
        path = np.array(path)
        ax1.plot(path[:,1], path[:,0], color='magenta', linewidth=2.5, label='BFS Path', alpha=0.8)
        ax1.scatter(path[:,1], path[:,0], color='magenta', s=15, alpha=0.8)

    # Start & Goal markers
    ax1.scatter(0, 0, color='lime', s=120, edgecolors='black', label='Start', zorder=5)
    ax1.scatter(N-1, N-1, color='red', s=120, edgecolors='black', label='Goal', zorder=5)

    ax1.set_title(f"Maze {maze_idx}: Ground Truth & BFS Path", fontsize=16)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)

    # 2. Value Function Heatmap
    ax2 = axes[1]
    norm = Normalize(vmin=V.min(), vmax=V.max())
    heatmap = ax2.imshow(V, cmap='inferno', origin='upper', norm=norm)
    
    # Overlay path subtly on top
    if path:
        ax2.plot(path[:,1], path[:,0], color='cyan', linewidth=2, alpha=0.7, label='BFS Path')

    ax2.set_title(f"Maze {maze_idx}: Value Iteration $V(s)$", fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('equal')

    # Colorbar on the side of heatmap
    cbar = fig.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("State Value", rotation=270, labelpad=20, fontsize=12)

    # Layout adjustments
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)