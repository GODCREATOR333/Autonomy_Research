import numpy as np
from collections import deque
from pathlib import Path

def make_seed(N: int, p: float, trial_index: int) -> int:
    """Deterministic seed from your original formula."""
    return N * 100_000 + int(round(p * 1000)) * 100 + trial_index

def generate_maze(N: int, p: float, seed: int) -> np.ndarray:
    if p == 0.0:
        return np.zeros((N, N), dtype=np.int32)
    rng = np.random.default_rng(seed)
    maze = rng.choice([0, 1], size=(N, N), p=[1.0 - p, p]).astype(np.int32)
    maze[0, 0] = 0
    maze[N - 1, N - 1] = 0
    return maze

def is_solvable(maze: np.ndarray) -> bool:
    N = maze.shape[0]
    start, goal = (0, 0), (N - 1, N - 1)
    if maze[start] == 1 or maze[goal] == 1:
        return False
    
    queue, visited = deque([start]), {start}
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False

def precompute_dataset(N: int, p: float, num_trials: int = 100, max_attempts: int = 50000, save_dir: str = "data/maze_dataset"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Single RNG for the whole dataset
    rng = np.random.default_rng(make_seed(N, p, 0))

    # Percolation threshold check
    if p > 0.59:
        print(f"Generating N={N}, p={p}...")
        print(f"  -> p exceeds percolation threshold. Saved 0 mazes.")

        p_str = f"{int(round(p * 1000)):04d}"
        filepath = f"{save_dir}/N{N}_p{p_str}.npy"

        np.save(filepath, np.empty((0, N, N), dtype=np.int32))
        return

    found_mazes = []
    trial_index = 0

    print(f"Generating N={N}, p={p}...")

    while len(found_mazes) < num_trials:

        for attempt in range(max_attempts):

            # sample a random seed each attempt
            seed = rng.integers(0, 2**32)

            maze = generate_maze(N, p, seed)

            if is_solvable(maze):
                found_mazes.append(maze)
                break

        else:
            print(f"  [Gave up] Density p={p} is too high. Only found {len(found_mazes)} mazes.")
            break

        trial_index += 1

    if len(found_mazes) > 0:
        mazes_array = np.array(found_mazes, dtype=np.int32)
    else:
        mazes_array = np.empty((0, N, N), dtype=np.int32)

    p_str = f"{int(round(p * 1000)):04d}"
    filepath = f"{save_dir}/N{N}_p{p_str}.npy"
    np.save(filepath, mazes_array)

    print(f"  -> Saved {len(found_mazes)} mazes to {filepath}")

    
if __name__ == "__main__":
    densities = [0.10, 0.20, 0.30, 0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.57,0.58,0.59,0.60,0.65]
    for p in densities:
        precompute_dataset(N=32, p=p, num_trials=100)