import numpy as np
import os

# --- Configuration (Matching your training script) ---
GRID_SIZE = 16
HALF_W = 1  # For 3x3 window
GOAL = (15, 15) # n-1, n-1
DATA_PATH = "data/maze_dataset/N16_p0100_train.npy"

# --- State Function (Exactly as provided) ---
def get_state(maze, pos):
    row, col = pos
    window = []

    for dr in range(-HALF_W, HALF_W + 1):
        for dc in range(-HALF_W, HALF_W + 1):
            r = row + dr
            c = col + dc

            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                window.append(maze[r, c])
            else:
                window.append(1) # Out of bounds treated as blocked

    dr_goal = int(np.sign(GOAL[0] - row))
    dc_goal = int(np.sign(GOAL[1] - col))

    # Return tuple so it is hashable for the set
    return tuple(window) + (dr_goal, dc_goal)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    print(f"Loading dataset from {DATA_PATH}...")
    try:
        mazes = np.load(DATA_PATH)
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        return

    print(f"Dataset shape: {mazes.shape}")
    num_mazes = mazes.shape[0]
    
    unique_states = set()
    total_valid_positions = 0

    print("Iterating through all mazes and cells...")

    for i in range(num_mazes):
        maze = mazes[i]
        # Iterate every cell in the 16x16 grid
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                
                # Agent cannot exist inside a wall (obstacle)
                # So we only count states where the center cell is 0 (Open)
                if maze[r, c] == 1:
                    continue
                
                total_valid_positions += 1
                state = get_state(maze, (r, c))
                unique_states.add(state)
        
        # Optional: Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_mazes} mazes...")

    print("-" * 30)
    print(f"Total Mazes: {num_mazes}")
    print(f"Total Valid Positions Checked: {total_valid_positions}")
    print(f"Total UNIQUE States Found: {len(unique_states)}")
    print("-" * 30)
    
    # Analysis Hint
    if len(unique_states) < 500:
        print("OBSERVATION: Unique state count is low.")
        print("This suggests the dataset might be locally repetitive, OR")
        print("your state representation (especially goal direction) has low variance.")
    else:
        print("OBSERVATION: Unique state count is high.")
        print("If your Q-table is still small (280), your agent has an EXPLORATION issue.")

if __name__ == "__main__":
    main()