import numpy as np
from pathlib import Path

def load_maze_dataset(N: int, p: float, data_dir: str = "data/maze_dataset") -> np.ndarray:
    """
    Loads precomputed solvable mazes.
    Returns: np.ndarray of shape (num_mazes, N, N)
    """
    p_str = f"{int(round(p * 1000)):04d}"
    filepath = Path(data_dir) / f"N{N}_p{p_str}.npy"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
        
    mazes = np.load(filepath)
    return mazes