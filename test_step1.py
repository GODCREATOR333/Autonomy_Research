from core.env import load_maze_dataset, solve_maze_shortest_path
from visualization.visualizer import MazeBlueprint
from core.solver import run_value_iteration
import rerun as rr

def main():
    N = 16
    p = 0.10
    data_dir = "data/maze_dataset"

    print(f"Loading real dataset for N={N}, p={p}...")
    try:
        maze_batch = load_maze_dataset(N, p, data_dir)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Successfully loaded {len(maze_batch)} mazes.")
    print("Solving mazes...")

    paths = [solve_maze_shortest_path(m) for m in maze_batch]

    print("Launching visualizer...")
    rr.init("maze_app")
    rr.spawn()

    for i, maze in enumerate(maze_batch):
        V = run_value_iteration(maze)

        blueprint = MazeBlueprint(N)
        blueprint.add_maze_view(maze, path=paths[i])
        blueprint.add_vi_view(V)

        rr.set_time("maze_index", sequence=i)
        rr.log("maze/combined", rr.Image(blueprint.get_blueprint()))

    print(f"Visualized {len(maze_batch)} mazes.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()