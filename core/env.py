# core/env.py
import numpy as np
from pathlib import Path
from collections import deque

def load_maze_dataset(N: int, p: float, data_dir: str = "data/maze_dataset") -> np.ndarray:
    """Loads precomputed solvable mazes."""
    p_str = f"{int(round(p * 1000)):04d}"
    filepath = Path(data_dir) / f"N{N}_p{p_str}.npy"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
        
    return np.load(filepath)

def solve_maze_shortest_path(maze: np.ndarray) -> list:
    """
    Finds the shortest path from (0,0) to (N-1, N-1) using BFS.
    Returns a list of (row, col) tuples.
    """
    N = maze.shape[0]
    start, goal = (0, 0), (N - 1, N - 1)
    
    queue = deque([(start, [start])]) # Stores (current_node, path_to_node)
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        
        if (r, c) == goal:
            return path
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
                
    return [] # Should not happen since your dataset is pre-filtered