from core.env import load_maze_dataset
from visualization.visualizer import MazeVisualizer

def main():
    N = 16
    p = 0.40  # Test loading the 0.40 density
    
    print(f"Loading dataset for N={N}, p={p}...")
    maze_batch = load_maze_dataset(N, p, data_dir="./data/maze_dataset")
    
    print(f"Loaded {maze_batch.shape[0]} mazes. Launching Rerun...")
    vis = MazeVisualizer()
    vis.log_maze_batch(maze_batch)

if __name__ == "__main__":
    main()