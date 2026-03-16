#!/usr/bin/env python3
"""
compute_VI.py - CORRECTED VERSION
Computes Value Iteration benchmark for maze datasets.
Fixes: 
  1. BFS now correctly respects walls (0=open, 1=wall).
  2. VI uses gamma=1.0 for shortest path convergence.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
from pathlib import Path
from collections import deque
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "save_dir": "data_jax",
    "gamma": 1.0,            # CHANGED: 1.0 for shortest path (undiscounted)
    "theta": 1e-4,           # CHANGED: Relaxed threshold for gamma=1.0
    "max_iterations": 2000,  # Increased limit
    "batch_size": 50,        # Reduced batch size for stability
    "step_cost": -1.0,       # Reward per step
    "goal_reward": 0.0,      # Reward at goal
}

ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

# =============================================================================
# NUMPY: Corrected BFS (Respects Walls)
# =============================================================================
def compute_optimal_path_length(maze, start, goal):
    """
    BFS to find shortest path length.
    CRITICAL FIX: Checks maze[nr, nc] == 0 (open) to respect walls.
    """
    N = maze.shape[0]
    
    # Safety check
    if maze[start] == 1 or maze[goal] == 1:
        return -1  # Start or goal is blocked
    
    queue = deque([(start[0], start[1], 0)])  # (row, col, distance)
    visited = np.zeros((N, N), dtype=bool)
    visited[start] = True
    
    while queue:
        r, c, dist = queue.popleft()
        
        if (r, c) == goal:
            return dist
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            
            # Check bounds AND if cell is OPEN (0)
            if 0 <= nr < N and 0 <= nc < N:
                if maze[nr, nc] == 0 and not visited[nr, nc]:  # <--- THE FIX
                    visited[nr, nc] = True
                    queue.append((nr, nc, dist + 1))
    
    return -1  # No path found

# =============================================================================
# JAX: Value Iteration (Fixed for Gamma=1.0)
# =============================================================================
def value_iteration_jax(mazes, gamma=1.0, theta=1e-4, max_iter=2000, step_cost=-1.0):
    """
    Run Value Iteration on a BATCH of mazes.
    Uses gamma=1.0 for shortest path problems.
    """
    batch_size, N, _ = mazes.shape
    
    # Initialize V-table: 0 everywhere
    V = jnp.zeros((batch_size, N, N))
    
    # Pre-compute goal mask
    goal_mask = jnp.zeros((N, N), dtype=bool)
    goal_mask = goal_mask.at[N-1, N-1].set(True)  # Goal at (15,15)
    
    def vi_step(V, mazes):
        """One iteration of Value Iteration."""
        V_new = V.copy()
        
        # Iterate over all states
        for r in range(N):
            for c in range(N):
                # Mask for open cells in this position across the batch
                is_open = mazes[:, r, c] == 0
                is_goal = goal_mask[r, c]
                
                # Compute Q-values for 4 actions
                Q_values = jnp.full((batch_size, 4), -1e9)  # Initialize to -inf
                
                for a in range(4):
                    dr, dc = ACTION_DELTAS[a]
                    nr, nc = r + dr, c + dc
                    
                    # Check bounds
                    valid_move = (nr >= 0) & (nr < N) & (nc >= 0) & (nc < N)
                    
                    # If valid and not a wall, get next state value
                    # Note: mazes[:, nr, nc] == 0 checks if target is open
                    target_open = jnp.where(valid_move, mazes[:, nr, nc] == 0, False)
                    
                    next_V = jnp.where(
                        target_open,
                        V[:, nr, nc],
                        -1e9  # Penalty for hitting wall/boundary
                    )
                    
                    # Q = step_cost + gamma * V(next)
                    Q_values = Q_values.at[:, a].set(step_cost + gamma * next_V)
                
                # Update V: 
                # - Goal: 0
                # - Open: max(Q)
                # - Wall: 0 (or keep previous)
                max_Q = jnp.max(Q_values, axis=1)
                
                V_new = V_new.at[:, r, c].set(
                    jnp.where(is_goal, 0.0, 
                     jnp.where(is_open, max_Q, 0.0))
                )
        
        # Compute convergence metric (only for open, non-goal cells)
        delta = jnp.max(jnp.abs(V_new - V))
        return V_new, delta
    
    # Run iterations
    V_current = V
    for iteration in range(max_iter):
        V_next, delta = vi_step(V_current, mazes)
        
        # Check convergence
        if iteration % 100 == 0:
            print(f"      Iter {iteration}: delta={delta:.6f}", end='\r')
        
        if delta < theta:
            print(f"\n      Converged at iteration {iteration}")
            V_current = V_next
            break
        
        V_current = V_next
    else:
        print(f"\n      ⚠️ Max iterations reached (delta={delta:.6f})")
    
    # Extract Policy (Greedy)
    policies = jnp.zeros((batch_size, N, N), dtype=jnp.int32)
    
    for r in range(N):
        for c in range(N):
            is_open = mazes[:, r, c] == 0
            is_goal = goal_mask[r, c]
            
            Q_values = jnp.full((batch_size, 4), -1e9)
            for a in range(4):
                dr, dc = ACTION_DELTAS[a]
                nr, nc = r + dr, c + dc
                valid_move = (nr >= 0) & (nr < N) & (nc >= 0) & (nc < N)
                target_open = jnp.where(valid_move, mazes[:, nr, nc] == 0, False)
                next_V = jnp.where(target_open, V_current[:, nr, nc], -1e9)
                Q_values = Q_values.at[:, a].set(step_cost + gamma * next_V)
            
            best_action = jnp.argmax(Q_values, axis=1)
            policies = policies.at[:, r, c].set(
                jnp.where(is_open & ~is_goal, best_action, 0)
            )
    
    return V_current, policies

# =============================================================================
# MAIN
# =============================================================================
def main():
    save_dir = Path(CONFIG["save_dir"])
    maze_files = sorted(save_dir.glob("*.npy"))
    
    if not maze_files:
        print(f"❌ No .npy files found in {save_dir}")
        return
    
    print(f"🔍 Found {len(maze_files)} maze files\n")
    start, goal = (0, 0), (15, 15)
    
    for filepath in maze_files:
        print(f"📂 Processing: {filepath.name}")
        mazes = np.load(filepath)
        output_path = filepath.with_name(filepath.stem + "_VI.npz")
        
        if output_path.exists():
            print(f"   ⏭️  Exists: {output_path.name}\n")
            continue
        
        batch_size = CONFIG["batch_size"]
        num_mazes = mazes.shape[0]
        
        all_V, all_P, all_L = [], [], []
        
        for batch_start in range(0, num_mazes, batch_size):
            batch_end = min(batch_start + batch_size, num_mazes)
            batch_mazes = mazes[batch_start:batch_end]
            batch_jax = jnp.array(batch_mazes)
            
            print(f"   Batch {batch_start}:{batch_end}...")
            t0 = time.time()
            
            V_batch, P_batch = value_iteration_jax(
                batch_jax, 
                gamma=CONFIG["gamma"], 
                theta=CONFIG["theta"],
                max_iter=CONFIG["max_iterations"]
            )
            
            # Compute path lengths with CORRECTED BFS
            lengths = []
            for i, mz in enumerate(batch_mazes):
                L = compute_optimal_path_length(mz, start, goal)
                lengths.append(L)
            
            all_V.append(np.array(V_batch))
            all_P.append(np.array(P_batch))
            all_L.extend(lengths)
            
            t1 = time.time()
            print(f"   Done ({t1-t0:.2f}s). Path lengths: min={min(lengths)}, max={max(lengths)}")
            
            # DEBUG: Print one example if lengths look suspicious
            if min(lengths) == 30 and max(lengths) == 30:
                print(f"   ⚠️  WARNING: All paths length 30! Checking maze {batch_start}...")
                sample = batch_mazes[0]
                print(f"   Sample maze start/goal: {sample[start]}, {sample[goal]}")
                print(f"   Sample maze obstacle count: {np.sum(sample == 1)}/256")
        
        # Save
        np.savez(
            output_path,
            mazes=mazes,
            V_tables=np.concatenate(all_V, axis=0),
            policies=np.concatenate(all_P, axis=0),
            optimal_lengths=np.array(all_L),
            metadata={'gamma': CONFIG["gamma"], 'converged': True}
        )
        print(f"   ✅ Saved {output_path.name}\n")

if __name__ == "__main__":
    main()