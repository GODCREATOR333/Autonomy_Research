import matplotlib.pyplot as plt
import numpy as np

def plot_research_triplet(maze, path, V, maze_idx, save_path):
    """
    Renders 3 panels: Maze+Path, Value Heatmap, and Title/Legends.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Maze + BFS Path
    # Map: 1 (Wall) -> Gray (0.5), 0 (Free) -> White (1.0)
    maze_viz = np.where(maze == 1, 0.5, 1.0)
    ax1.imshow(maze_viz, cmap='gray', vmin=0, vmax=1)
    
    if path:
        path = np.array(path)
        # Plot path in Magenta
        ax1.plot(path[:,1], path[:,0], color='magenta', linewidth=2, label='BFS Path')
    
    # Start/Goal markers
    ax1.scatter(0, 0, color='lime', s=100, label='Start', edgecolors='black')
    ax1.scatter(maze.shape[0]-1, maze.shape[1]-1, color='red', s=100, label='Goal', edgecolors='black')
    
    ax1.set_title(f"Maze {maze_idx}: Ground Truth & BFS")
    ax1.legend(loc='upper right')
    ax1.set_xticks([]); ax1.set_yticks([])

    # 2. Value Function Heatmap
    im = ax2.imshow(V, cmap='turbo', interpolation='nearest')
    ax2.set_title(f"Maze {maze_idx}: Value Iteration $V(s)$")
    fig.colorbar(im, ax=ax2, label='State Value')
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)