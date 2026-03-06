import numpy as np
from collections import deque
from pathlib import Path

def generate_maze(N: int, p: float) -> np.ndarray:
    rng = np.random.default_rng()
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

def precompute_dataset(N: int, p: float, num_train: int = 1000, num_test: int = 200,
                       save_dir: str = "data/maze_dataset", max_attempts: int = 1000000):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for dataset_type, target_num in zip(['train', 'test'], [num_train, num_test]):
        p_str = f"{int(round(p * 1000)):04d}"
        filepath = f"{save_dir}/N{N}_p{p_str}_{dataset_type}.npy"

        # Load existing mazes if file exists
        if Path(filepath).exists():
            existing_mazes = np.load(filepath)
            seen_hashes = {m.tobytes() for m in existing_mazes}
        else:
            existing_mazes = np.empty((0, N, N), dtype=np.int32)
            seen_hashes = set()

        found_mazes = []
        attempts = 0
        print(f"Generating {dataset_type} set for N={N}, p={p}, aiming for {target_num} unique mazes...")

        while len(existing_mazes) + len(found_mazes) < target_num and attempts < max_attempts:
            maze = generate_maze(N, p)
            maze_hash = maze.tobytes()

            if maze_hash in seen_hashes:
                attempts += 1
                continue

            if is_solvable(maze):
                found_mazes.append(maze)
                seen_hashes.add(maze_hash)

            attempts += 1

        if attempts >= max_attempts:
            print(f"  [Stopped] Max attempts reached. Generated {len(found_mazes)} new unique mazes.")

        all_mazes = np.array(list(existing_mazes) + found_mazes, dtype=np.int32)
        np.save(filepath, all_mazes)
        print(f"  -> Total {len(all_mazes)} mazes saved to {filepath}")

if __name__ == "__main__":
    densities = [0.10, 0.20, 0.30, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.57, 0.58, 0.59]
    sizes = [16, 32]

    for N in sizes:
        for p in densities:
            precompute_dataset(N=N, p=p, num_train=1000, num_test=200)