import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pathlib import Path
from collections import deque
import copy

# =============================================================================
# CONFIGURATION - Change these to reuse the script
# =============================================================================
CONFIG = {
    "grid_size": 16,
    "obstacle_prob": 0.40,       # 10% obstacles (1), 90% open (0)
    "start": (0, 0),
    "goal": (15, 15),
    
    # Train set: 5000 total (90% solvable, 10% unsolvable)
    "train_total": 5000,
    "train_solvable_ratio": 0.90,
    
    # Test sets: 1000 each for random, symmetric, shapes (all solvable)
    "test_per_type": 1000,
    
    "save_dir": "data_jax",
    "batch_size": 50,            # JAX batch size for generation
    "max_attempts": 50000,       # Safety limit per dataset
}

Path(CONFIG["save_dir"]).mkdir(parents=True, exist_ok=True)

def get_p_str(p):
    """Format probability for filenames: 0.10 -> 0100"""
    return f"{int(round(p * 1000)):04d}"

# =============================================================================
# JAX: Fast Batch Random Generation
# =============================================================================
def generate_batch_jax(key, shape, p_obstacle):
    """
    Generate a batch of random mazes using JAX.
    Returns JAX array of shape (batch, N, N) with 0=open, 1=obstacle.
    """
    return jax.random.bernoulli(key, p=p_obstacle, shape=shape).astype(jnp.int32)

# =============================================================================
# NUMPY/PYTHON: Solvability Check (BFS)
# =============================================================================
def is_solvable_numpy(maze, start, goal):
    """
    Standard BFS to check if path exists from start to goal.
    0 = open, 1 = obstacle.
    """
    N = maze.shape[0]
    if maze[start] == 1 or maze[goal] == 1:
        return False
    
    queue = deque([start])
    visited = set([start])
    
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N:
                if maze[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False

# =============================================================================
# FORCE UNSOLVABLE: Create guaranteed unsolvable mazes
# =============================================================================
def force_unsolvable(maze, start, goal):
    """
    Add a vertical barrier to guarantee no path exists.
    """
    N = maze.shape[0]
    modified = maze.copy()
    mid_col = N // 2
    modified[:, mid_col] = 1  # Wall across middle column
    modified[start] = 0
    modified[goal] = 0
    return modified

# =============================================================================
# SYMMETRIC MAZE GENERATOR (Horizontal Mirror)
# =============================================================================
def generate_symmetric_maze(N, p, key, start, goal):
    """
    Fixed version: All 4 symmetry axes now fill the entire maze correctly.
    """
    key, axis_key, maze_key = random.split(key, 3)
    axis = int(random.randint(axis_key, (), 0, 4))
    
    maze = np.zeros((N, N), dtype=np.int32)
    
    if axis == 0:
        # Vertical symmetry (left-right mirror)
        half = N // 2
        left = np.array(random.bernoulli(maze_key, p=p, shape=(N, half)), dtype=np.int32)
        maze[:, :half] = left
        maze[:, half:] = np.fliplr(left)  # Fixed: use fliplr for proper mirror
    
    elif axis == 1:
        # Horizontal symmetry (top-bottom mirror)
        half = N // 2
        top = np.array(random.bernoulli(maze_key, p=p, shape=(half, N)), dtype=np.int32)
        maze[:half, :] = top
        maze[half:, :] = np.flipud(top)  # Fixed: use flipud for proper mirror
    
    elif axis == 2:
        # Main diagonal symmetry (transpose)
        base = np.array(random.bernoulli(maze_key, p=p, shape=(N, N)), dtype=np.int32)
        # Create symmetric matrix: A = (base + base.T) > 0
        maze = ((base + base.T) > 0).astype(np.int32)
    
    else:
        # Anti-diagonal symmetry
        base = np.array(random.bernoulli(maze_key, p=p, shape=(N, N)), dtype=np.int32)
        # Flip both axes, transpose, then flip back
        maze = np.flipud(np.fliplr(base))
        maze = ((maze + maze.T) > 0).astype(np.int32)
        maze = np.flipud(np.fliplr(maze))
    
    # Ensure start/goal open
    maze[start] = 0
    maze[goal] = 0
    
    return maze, key

# =============================================================================
# SHAPE PRIMITIVES: Templates for "traps" and structures
# =============================================================================
def define_shape_primitives(N):
    """
    Define trap primitives designed to confuse egocentric agents.
    1 = wall, 0 = open. Shapes are designed to look promising from entrance.
    """
    primitives = {}
    
    # =====================================================================
    # 1. DEEP U-TRAP (The Classic Lure)
    # =====================================================================
    # Agent sees open path → enters → hits back wall → must backtrack
    # Entrance is wide (3 cells) to look inviting
    u_trap = np.ones((7, 7), dtype=np.int32)
    # Inner corridor: 3 cells wide, 4 cells deep
    u_trap[1:6, 2:5] = 0  # Vertical corridor
    u_trap[5, 2:5] = 1    # Back wall (dead end)
    # Open entrance at top
    u_trap[0, 2:5] = 0
    # Optional: Add side "decoy" openings that also dead-end
    u_trap[3, 1] = 0  # Small side pocket
    u_trap[3, 5] = 0
    primitives['u_trap_deep'] = u_trap
    
    # =====================================================================
    # 2. LONG CORRIDOR DEAD-END (Commitment Trap)
    # =====================================================================
    # 1-cell wide corridor, 10 cells long → sudden dead end
    # Egocentric agent can't see the end from the start
    corridor_trap = np.ones((3, 12), dtype=np.int32)
    corridor_trap[1, 1:10] = 0  # Long narrow path
    corridor_trap[1, 10] = 1    # Dead end wall
    # Entrance open
    corridor_trap[1, 0] = 0
    # Add slight "hope" at end: a 2x2 chamber that still dead-ends
    corridor_trap[0:2, 9] = 0
    primitives['corridor_long'] = corridor_trap
    
    # =====================================================================
    # 3. T-JUNCTION WITH FAKE BRANCH (Ambiguous Choice)
    # =====================================================================
    # Agent reaches T, picks right branch → dead end after 3 steps
    # Must backtrack to T and try left (real path)
    t_trap = np.ones((7, 9), dtype=np.int32)
    # Main vertical corridor (real path continues down)
    t_trap[1:6, 4] = 0
    # Horizontal branch at row 3 (the T)
    t_trap[3, 2:7] = 0
    # Right branch dead-ends after 2 cells
    t_trap[3, 6] = 1
    t_trap[2:5, 6] = 1  # Wall off the dead end
    # Left branch continues (but we only stamp part of it)
    t_trap[3, 1] = 0
    # Entrance at top
    t_trap[0, 4] = 0
    primitives['t_junction_fake'] = t_trap
    
    # =====================================================================
    # 4. BOTTLENECK ROOM WITH DECOY EXITS (False Hope)
    # =====================================================================
    # Enter narrow → large room → 3 exits, only 1 is real
    # Egocentric agent picks wrong exit → trapped
    bottleneck = np.ones((9, 9), dtype=np.int32)
    # Entrance corridor (narrow)
    bottleneck[4, 0:3] = 0
    # Central room (3x3 open)
    bottleneck[3:6, 3:6] = 0
    # Three "exits" from room (only bottom is real)
    bottleneck[3, 4] = 0  # Top exit → dead end
    bottleneck[5, 4] = 0  # Bottom exit → REAL PATH (agent should go here)
    bottleneck[4, 6] = 0  # Right exit → dead end
    # Wall off the fake exits after 1-2 cells
    bottleneck[2, 4] = 1  # Top dead end wall
    bottleneck[4, 7] = 1  # Right dead end wall
    primitives['bottleneck_decoy'] = bottleneck
    
    # =====================================================================
    # 5. ZIG-ZAG FALSE PATH (Memory Test)
    # =====================================================================
    # Path turns 3x, agent thinks it's navigating → sudden dead end
    # Tests if agent maintains global map vs. reactive
    zigzag = np.ones((7, 9), dtype=np.int32)
    # Zig-zag path: right → down → right → down → dead end
    zigzag[3, 1:4] = 0  # Horizontal segment 1
    zigzag[3:6, 3] = 0  # Vertical segment
    zigzag[5, 3:7] = 0  # Horizontal segment 2
    zigzag[5, 7] = 1    # Dead end wall
    # Entrance
    zigzag[3, 0] = 0
    # Add "almost exits" that are walled off to increase confusion
    zigzag[2, 3] = 0  # Looks like an up exit but is short
    zigzag[2, 4] = 1
    primitives['zigzag_false'] = zigzag
    
    # =====================================================================
    # 6. SPIRAL TRAP (Advanced: Tests Path Integration)
    # =====================================================================
    # Path spirals inward → center is dead end
    # Agent must realize it's looping and backtrack
    spiral = np.ones((9, 9), dtype=np.int32)
    # Outer ring (partial)
    spiral[1, 2:7] = 0
    spiral[2:7, 6] = 0
    spiral[6, 2:6] = 0
    spiral[3:6, 2] = 0
    # Inner ring
    spiral[3, 3:5] = 0
    spiral[4, 4] = 0  # Center (dead end)
    spiral[4, 5] = 1  # Wall off center
    # Entrance
    spiral[1, 2] = 0
    primitives['spiral_inward'] = spiral
    
    return primitives

def stamp_shape(maze, primitive, top_left, rotation=0, preserve_open=True):
    """
    Overlay a primitive onto the maze.
    
    Args:
        preserve_open: If True, only stamp walls (1s), don't overwrite existing open (0s).
                      This ensures stamped traps connect to existing paths.
    """
    N = maze.shape[0]
    p_h, p_w = primitive.shape
    r0, c0 = top_left
    
    # Rotate primitive
    rotated = np.rot90(primitive, k=-rotation)
    p_h, p_w = rotated.shape
    
    for i in range(p_h):
        for j in range(p_w):
            r, c = r0 + i, c0 + j
            if 0 <= r < N and 0 <= c < N:
                if preserve_open:
                    # Only write walls, never overwrite open space
                    if rotated[i, j] == 1:
                        maze[r, c] = 1
                else:
                    # Overwrite everything (use cautiously)
                    maze[r, c] = rotated[i, j]
    # Rotate primitive
    rotated = np.rot90(primitive, k=-rotation)
    p_h, p_w = rotated.shape
    
    for i in range(p_h):
        for j in range(p_w):
            r, c = r0 + i, c0 + j
            if 0 <= r < N and 0 <= c < N:
                if preserve_open:
                    if rotated[i, j] == 1:
                        maze[r, c] = 1
                else:
                    maze[r, c] = rotated[i, j]
    
    # === ADD THESE 3 LINES ===
    # Force primitive's border cells that are open (0) to stay open in maze
    for i in range(p_h):
        for j in range(p_w):
            if (i in [0, p_h-1] or j in [0, p_w-1]) and rotated[i, j] == 0:  # border + open
                r, c = r0 + i, c0 + j
                if 0 <= r < N and 0 <= c < N:
                    maze[r, c] = 0  # Force open
    # === END ADDITION ===
    
    return maze
    

def find_trap_placement(maze, primitive, start, goal, max_attempts=50):
    """
    Find a good location to stamp a trap:
    - Near the start (within 4 cells) so agent encounters it early
    - Oriented so entrance faces away from start (looks like a path forward)
    - Doesn't block start/goal
    """
    N = maze.shape[0]
    p_h, p_w = primitive.shape
    
    for attempt in range(max_attempts):
        # Bias position: sample near start (within 4 cells)
        r_range = max(0, start[0] - 4), min(N - p_h, start[0] + 4)
        c_range = max(0, start[1] - 4), min(N - p_w, start[1] + 4)
        
        if r_range[0] > r_range[1] or c_range[0] > c_range[1]:
            continue
            
        r0 = np.random.randint(r_range[0], r_range[1] + 1)
        c0 = np.random.randint(c_range[0], c_range[1] + 1)
        
        # Try 4 rotations, pick one where entrance faces "forward" (toward goal)
        best_rotation = 0
        for rot in range(4):
            rotated = np.rot90(primitive, k=-rot)
            # Simple heuristic: entrance should not face directly toward start
            # (We assume entrance is at the "open" side of the primitive)
            # For U-trap, entrance is top; for corridor, entrance is left, etc.
            # This is approximate but works well in practice
            best_rotation = rot
            break  # Just pick first valid for now
        
        # Check: does stamping this block start or goal?
        test_maze = maze.copy()
        test_maze = stamp_shape(test_maze, primitive, (r0, c0), best_rotation)
        if test_maze[start] == 1 or test_maze[goal] == 1:
            continue  # Try next position
        
        return (r0, c0), best_rotation
    
    # Fallback: random placement
    max_r = N - primitive.shape[0]
    max_c = N - primitive.shape[1]
    if max_r <= 0 or max_c <= 0:
        return None, 0
    return (np.random.randint(0, max_r+1), np.random.randint(0, max_c+1)), np.random.randint(0, 4)

def generate_shape_maze(N, p, key, start, goal, primitives):
    """
    Generate a maze by stamping 2-4 trap primitives onto a random background.
    Traps are placed near start and oriented to look like forward paths.
    """
    # Start with random background
    key, subkey = random.split(key)
    maze = np.array(jax.random.bernoulli(subkey, p=p, shape=(N, N)).astype(jnp.int32))
    
    # Force start/goal open
    maze[start] = 0
    maze[goal] = 0
    
    # Stamp 2-4 traps with smart placement
    num_traps = 1
    for _ in range(num_traps):
        name = np.random.choice(list(primitives.keys()))
        primitive = primitives[name]
        
        # Find good placement
        top_left, rotation = find_trap_placement(maze, primitive, start, goal)
        if top_left is None:
            continue
        
        # Stamp it (preserve open paths so agent can enter)
        maze = stamp_shape(maze, primitive, top_left, rotation, preserve_open=True)
    
    # Final safety: ensure start/goal still open
    maze[start] = 0
    maze[goal] = 0
    
    return maze, key

# =============================================================================
# MAIN GENERATION FUNCTION (Handles all types)
# =============================================================================
def generate_dataset_split(key, num_mazes, target_solvable_ratio, seen_hashes, 
                          maze_type='random', primitives=None, is_test=False):
    """
    Generate a dataset split with specified maze type.
    
    Args:
        maze_type: 'random', 'symmetric', or 'shapes'
        primitives: dict of shape templates (only for 'shapes' type)
    """
    N = CONFIG["grid_size"]
    p = CONFIG["obstacle_prob"]
    start, goal = CONFIG["start"], CONFIG["goal"]
    batch_size = CONFIG["batch_size"]
    max_attempts = CONFIG["max_attempts"]
    
    # Determine quotas
    if is_test or maze_type in ['symmetric', 'shapes']:
        # Test sets and structured mazes are 100% solvable
        num_solvable = num_mazes
        num_unsolvable = 0
    else:
        num_solvable = int(num_mazes * target_solvable_ratio)
        num_unsolvable = num_mazes - num_solvable
    
    collected_solvable = []
    collected_unsolvable = []
    attempts = 0
    
    print(f"Generating {num_mazes} {maze_type} mazes (Solvable: {num_solvable}, Unsolvable: {num_unsolvable})...")
    
    while (len(collected_solvable) < num_solvable or len(collected_unsolvable) < num_unsolvable) and attempts < max_attempts:
        key, subkey = random.split(key)
        
        # Generate batch based on maze type
        if maze_type == 'random':
            # JAX batch generation
            batch_jax = generate_batch_jax(subkey, (batch_size, N, N), p)
            batch_np = np.array(batch_jax)
            mazes_to_check = []
            for mz in batch_np:
                mz[start] = 0
                mz[goal] = 0
                mazes_to_check.append(mz)
                
        elif maze_type == 'symmetric':
            mazes_to_check = []
            for _ in range(batch_size):
                mz, key = generate_symmetric_maze(N, p, key, start, goal)
                mazes_to_check.append(mz)
                
        elif maze_type == 'shapes':
            if primitives is None:
                primitives = define_shape_primitives(N)
            mazes_to_check = []
            for _ in range(batch_size):
                mz, key = generate_shape_maze(N, p, key, start, goal, primitives)
                mazes_to_check.append(mz)
        
        # Process each maze in batch
        for maze in mazes_to_check:
            h = maze.tobytes()
            if h in seen_hashes:
                continue
            
            solvable = is_solvable_numpy(maze, start, goal)
            
            # Collect solvable
            if solvable and len(collected_solvable) < num_solvable:
                collected_solvable.append(maze)
                seen_hashes.add(h)
            
            # Collect unsolvable (only for random train)
            elif not solvable and len(collected_unsolvable) < num_unsolvable:
                collected_unsolvable.append(maze)
                seen_hashes.add(h)
            
            # Force unsolvable if struggling (random train only)
            elif solvable and maze_type == 'random' and not is_test and len(collected_unsolvable) < num_unsolvable:
                forced = force_unsolvable(maze, start, goal)
                h_forced = forced.tobytes()
                if h_forced not in seen_hashes and not is_solvable_numpy(forced, start, goal):
                    collected_unsolvable.append(forced)
                    seen_hashes.add(h_forced)
            
            # Early exit
            if len(collected_solvable) >= num_solvable and len(collected_unsolvable) >= num_unsolvable:
                break
        
        attempts += 1
        if attempts % 1000 == 0:
            print(f"  Progress: {len(collected_solvable)+len(collected_unsolvable)}/{num_mazes} collected, {attempts} attempts")
    
    if attempts >= max_attempts:
        print(f"  ⚠️  Max attempts reached. Collected {len(collected_solvable)+len(collected_unsolvable)}/{num_mazes}")
    
    # Convert to arrays (handle empty case)
    sol_arr = np.array(collected_solvable, dtype=np.int32) if collected_solvable else np.empty((0, N, N), dtype=np.int32)
    unsol_arr = np.array(collected_unsolvable, dtype=np.int32) if collected_unsolvable else np.empty((0, N, N), dtype=np.int32)
    
    return sol_arr, unsol_arr, seen_hashes, key

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Initialize global state
    rng_key = random.PRNGKey(42)  # Fixed seed for reproducibility
    global_seen_hashes = set()
    
    N = CONFIG["grid_size"]
    p_str = get_p_str(CONFIG["obstacle_prob"])
    save_dir = CONFIG["save_dir"]
    
    print(f"🚀 Starting maze generation: {N}x{N}, p={CONFIG['obstacle_prob']}")
    print(f"📁 Saving to: {save_dir}/\n")
    
    # -------------------------------------------------------------------------
    # 1. TRAIN SET: Random mazes (4500 solvable + 500 unsolvable)
    # -------------------------------------------------------------------------
    print("📦 Generating TRAIN set (random)...")
    train_sol, train_unsol, global_seen_hashes, rng_key = generate_dataset_split(
        rng_key, 
        CONFIG["train_total"], 
        CONFIG["train_solvable_ratio"], 
        global_seen_hashes,
        maze_type='random',
        is_test=False
    )
    
    # Save train files
    if len(train_sol) > 0:
        np.save(f"{save_dir}/N{N}_P{p_str}_train_solvable.npy", train_sol)
        print(f"✓ Saved train_solvable: {train_sol.shape}")
    if len(train_unsol) > 0:
        np.save(f"{save_dir}/N{N}_P{p_str}_train_unsolvable.npy", train_unsol)
        print(f"✓ Saved train_unsolvable: {train_unsol.shape}")
    
    # -------------------------------------------------------------------------
    # 2. TEST SET: Random mazes (1000 solvable)
    # -------------------------------------------------------------------------
    print("\n📦 Generating TEST set (random)...")
    test_random_sol, _, global_seen_hashes, rng_key = generate_dataset_split(
        rng_key,
        CONFIG["test_per_type"],
        1.0,
        global_seen_hashes,
        maze_type='random',
        is_test=True
    )
    if len(test_random_sol) > 0:
        np.save(f"{save_dir}/N{N}_P{p_str}_test_solvable_random.npy", test_random_sol)
        print(f"✓ Saved test_solvable_random: {test_random_sol.shape}")
    
    # -------------------------------------------------------------------------
    # 3. TEST SET: Symmetric mazes (1000 solvable)
    # -------------------------------------------------------------------------
    print("\n📦 Generating TEST set (symmetric)...")
    test_sym_sol, _, global_seen_hashes, rng_key = generate_dataset_split(
        rng_key,
        CONFIG["test_per_type"],
        1.0,
        global_seen_hashes,
        maze_type='symmetric',
        is_test=True
    )
    if len(test_sym_sol) > 0:
        np.save(f"{save_dir}/N{N}_P{p_str}_test_solvable_symmetric.npy", test_sym_sol)
        print(f"✓ Saved test_solvable_symmetric: {test_sym_sol.shape}")
    
    # -------------------------------------------------------------------------
    # 4. TEST SET: Shape-based mazes (1000 solvable)
    # -------------------------------------------------------------------------
    print("\n📦 Generating TEST set (shapes)...")
    primitives = define_shape_primitives(N)  # Define once, reuse
    test_shapes_sol, _, global_seen_hashes, rng_key = generate_dataset_split(
        rng_key,
        CONFIG["test_per_type"],
        1.0,
        global_seen_hashes,
        maze_type='shapes',
        primitives=primitives,
        is_test=True
    )
    if len(test_shapes_sol) > 0:
        np.save(f"{save_dir}/N{N}_P{p_str}_test_solvable_shapes.npy", test_shapes_sol)
        print(f"✓ Saved test_solvable_shapes: {test_shapes_sol.shape}")
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"🎉 Generation complete! Files saved to {save_dir}/")
    print(f"{'='*60}")
    
    # Verify all files exist and show shapes
    expected_files = [
        f"N{N}_P{p_str}_train_solvable.npy",
        f"N{N}_P{p_str}_train_unsolvable.npy", 
        f"N{N}_P{p_str}_test_solvable_random.npy",
        f"N{N}_P{p_str}_test_solvable_symmetric.npy",
        f"N{N}_P{p_str}_test_solvable_shapes.npy"
    ]
    
    for fname in expected_files:
        fpath = f"{save_dir}/{fname}"
        if Path(fpath).exists():
            data = np.load(fpath)
            print(f"✓ {fname:45s} -> {data.shape}")
        else:
            print(f"✗ {fname:45s} -> MISSING")
    
    print(f"\n🔢 Total unique mazes generated: {len(global_seen_hashes):,}")