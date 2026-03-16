#!/usr/bin/env python3
"""
compute_VI.py - Compute Value Iteration benchmark for all maze datasets.
Simple, clean implementation based on reference code.
"""

import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "save_dir": "data_jax",
    "gamma": 0.9,
    "theta": 1e-4,
    "step_reward": -1,
    "wall_penalty": -10,
}

# =============================================================================
# Value Iteration (Your Reference Code + Policy Extraction)
# =============================================================================
def run_value_iteration(maze, gamma=0.9, theta=1e-4, step_reward=-1, wall_penalty=-10):
    """
    Run Value Iteration on a single maze.
    Returns V-table and optimal policy.
    """
    N = maze.shape[0]
    V = np.zeros((N, N))
    policy = np.zeros((N, N), dtype=np.int32)  # 0=Up, 1=Down, 2=Left, 3=Right
    
    action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for r in range(N):
            for c in range(N):
                # Skip walls and goal
                if maze[r, c] == 1 or (r == N-1 and c == N-1):
                    continue
                
                vals = []
                for a, (dr, dc) in enumerate(action_deltas):
                    nr, nc = r + dr, c + dc
                    
                    if 0 <= nr < N and 0 <= nc < N:
                        if maze[nr, nc] == 0:
                            reward = step_reward
                            vals.append((reward + gamma * V[nr, nc], a))
                        else:
                            # Hit wall
                            reward = wall_penalty
                            vals.append((reward + gamma * V[r, c], a))
                    else:
                        # Out of bounds
                        vals.append((wall_penalty + gamma * V[r, c], a))
                
                # Get best value and action
                best_val, best_action = max(vals, key=lambda x: x[0])
                new_V[r, c] = best_val
                policy[r, c] = best_action
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        
        V = new_V
        if delta < theta:
            break
    
    return V, policy

# =============================================================================
# BFS for Optimal Path Length (Verification)
# =============================================================================
def compute_optimal_path_length(maze, start, goal):
    """BFS to find shortest path length."""
    from collections import deque
    N = maze.shape[0]
    
    if maze[start] == 1 or maze[goal] == 1:
        return -1
    
    queue = deque([(start[0], start[1], 0)])
    visited = np.zeros((N, N), dtype=bool)
    visited[start] = True
    
    while queue:
        r, c, dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N:
                if maze[nr, nc] == 0 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc, dist + 1))
    
    return -1

# =============================================================================
# MAIN: Process All Maze Files
# =============================================================================
def main():
    save_dir = Path(CONFIG["save_dir"])
    
    if not save_dir.exists():
        print(f"❌ Directory not found: {save_dir}")
        return
    
    maze_files = sorted(save_dir.glob("*.npy"))
    
    if not maze_files:
        print(f"❌ No .npy files found in {save_dir}")
        return
    
    print(f"🔍 Found {len(maze_files)} maze files\n")
    
    start, goal = (0, 0), (15, 15)
    
    for filepath in maze_files:
        print(f"📂 Processing: {filepath.name}")
        
        mazes = np.load(filepath)
        print(f"   Loaded {mazes.shape[0]} mazes")
        
        output_path = filepath.with_name(filepath.stem + "_VI.npz")
        
        if output_path.exists():
            print(f"   ⏭️  Exists: {output_path.name}\n")
            continue
        
        all_V = []
        all_policies = []
        all_lengths = []
        
        for i, maze in enumerate(mazes):
            V, policy = run_value_iteration(
                maze,
                gamma=CONFIG["gamma"],
                theta=CONFIG["theta"],
                step_reward=CONFIG["step_reward"],
                wall_penalty=CONFIG["wall_penalty"]
            )
            
            length = compute_optimal_path_length(maze, start, goal)
            
            all_V.append(V)
            all_policies.append(policy)
            all_lengths.append(length)
            
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{len(mazes)}")
        
        # Save
        np.savez(
            output_path,
            mazes=mazes,
            V_tables=np.array(all_V),
            policies=np.array(all_policies),
            optimal_lengths=np.array(all_lengths),
            metadata={
                'gamma': CONFIG["gamma"],
                'theta': CONFIG["theta"],
                'step_reward': CONFIG["step_reward"],
                'wall_penalty': CONFIG["wall_penalty"],
            }
        )
        
        lengths = np.array(all_lengths)
        print(f"   ✅ Saved {output_path.name}")
        print(f"      Path lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
        print()
    
    print(f"{'='*60}")
    print(f"🎉 Value Iteration benchmark complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()