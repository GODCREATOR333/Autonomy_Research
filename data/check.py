import numpy as np
from pathlib import Path

def check_unique_mazes(dataset_dir: str = "data/maze_dataset"):
    dataset_dir = Path(dataset_dir)
    files = list(dataset_dir.glob("N*_p*.npy"))

    for file in files:
        mazes = np.load(file)
        print(f"Checking {file.name} with {len(mazes)} mazes...")

        if len(mazes) == 0:
            print("  -> No mazes to check, skipping.")
            continue

        # Convert each maze to bytes for hashing
        hashes = set()
        duplicates = 0
        for maze in mazes:
            h = maze.tobytes()
            if h in hashes:
                duplicates += 1
            else:
                hashes.add(h)

        if duplicates == 0:
            print(f"  -> All mazes are unique ✅")
        else:
            print(f"  -> Found {duplicates} duplicate mazes ❌")

if __name__ == "__main__":
    check_unique_mazes("data/maze_dataset")