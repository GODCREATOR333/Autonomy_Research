#!/usr/bin/env python3
"""
view.py - Interactive maze viewer for generated datasets.

Now supports viewing Value Iteration data (V-table + Policy) alongside mazes.

Usage:
    python view.py                                    # Interactive mode
    python view.py --file train_solvable              # Load specific file
    python view.py --file train_solvable --idx 42     # View specific maze
    python view.py --list                             # List available files
    python view.py --file test_shapes --save maze.png # Save as image
    python view.py --vi                               # Show VI data if available
"""

import numpy as np
import argparse
from pathlib import Path
import random
import sys

# Try to import matplotlib for nice visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not found. Using ASCII terminal display.")
    print("   Install with: pip install matplotlib")

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_SAVE_DIR = "data_jax"
GRID_SIZE = 16
ACTION_NAMES = {0: '↑', 1: '↓', 2: '←', 3: '→'}  # Up, Down, Left, Right

# =============================================================================
# COLORMAPS
# =============================================================================
def get_maze_colormap():
    """Colormap for maze: 0=open, 1=wall, 2=start, 3=goal"""
    colors = ['white', 'black', 'green', 'red']
    return mcolors.ListedColormap(colors)

def get_v_colormap():
    """Colormap for V-table: blue (low) to red (high)"""
    return plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed

# =============================================================================
# ASCII DISPLAY (Fallback)
# =============================================================================
def print_maze_ascii(maze, V=None, policy=None, start=(0,0), goal=(15,15), idx=None):
    """Print maze, V-table, and policy to terminal."""
    N = maze.shape[0]
    
    print(f"\n{'='*80}")
    if idx is not None:
        print(f"Maze Index: {idx}")
    print(f"Grid: {N}x{N} | Start: {start} | Goal: {goal}")
    print('='*80)
    
    # Print header
    print(f"{'MAZE':^26} | {'V-TABLE':^26} | {'POLICY':^26}")
    print('-'*80)
    
    for r in range(N):
        # Maze row
        maze_row = ""
        for c in range(N):
            if (r, c) == start:
                maze_row += "S"
            elif (r, c) == goal:
                maze_row += "G"
            elif maze[r, c] == 1:
                maze_row += "█"
            else:
                maze_row += "·"
        
        # V-table row
        if V is not None:
            v_row = ""
            for c in range(N):
                if maze[r, c] == 1:
                    v_row += "  █"
                elif (r, c) == goal:
                    v_row += "  G"
                else:
                    v_row += f"{V[r,c]:3.0f}"
        else:
            v_row = "N/A"
        
        # Policy row
        if policy is not None:
            p_row = ""
            for c in range(N):
                if maze[r, c] == 1:
                    p_row += " █"
                elif (r, c) == goal:
                    p_row += " G"
                else:
                    p_row += f" {ACTION_NAMES.get(policy[r,c], '?')}"
        else:
            p_row = "N/A"
        
        print(f"{r:2d} |{maze_row}| {v_row} |{p_row}")
    
    print('-'*80)
    print("Legend: █=Wall  ·=Open  S=Start  G=Goal  ↑↓←→=Policy")
    print()

# =============================================================================
# MATPLOTLIB: Single Maze Display
# =============================================================================
def show_maze_matplotlib(maze, start=(0,0), goal=(15,15), idx=None, save_path=None):
    """Display maze only (original functionality)."""
    N = maze.shape[0]
    display = maze.copy().astype(float)
    display[start] = 2
    display[goal] = 3
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = get_maze_colormap()
    ax.imshow(display, cmap=cmap, vmin=0, vmax=3)
    
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    title = f"Maze #{idx}" if idx is not None else "Maze"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved maze to {save_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# =============================================================================
# MATPLOTLIB: Maze + V-Table + Policy (Side-by-Side)
# =============================================================================
def show_maze_vi_matplotlib(maze, V, policy, start=(0,0), goal=(15,15), 
                            idx=None, save_path=None, optimal_length=None):
    """
    Display maze, V-table, and policy side-by-side in one figure.
    """
    N = maze.shape[0]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ==========================================================================
    # Panel 1: Maze
    # ==========================================================================
    ax = axes[0]
    display = maze.copy().astype(float)
    display[start] = 2
    display[goal] = 3
    
    cmap_maze = get_maze_colormap()
    ax.imshow(display, cmap=cmap_maze, vmin=0, vmax=3)
    
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.set_title(f"Maze #{idx}" if idx is not None else "Maze", fontsize=14, fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # ==========================================================================
    # Panel 2: V-Table (Heatmap)
    # ==========================================================================
    ax = axes[1]
    
    # Mask walls for better visualization
    V_display = V.copy()
    V_display[maze == 1] = np.nan  # Hide walls
    
    cmap_v = get_v_colormap()
    im = ax.imshow(V_display, cmap=cmap_v, vmin=np.nanmin(V), vmax=np.nanmax(V))
    
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.set_title("V-Table (Value Function)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Column")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    # ==========================================================================
    # Panel 3: Policy (Arrows)
    # ==========================================================================
    ax = axes[2]
    
    # Display maze background (light gray for open, black for walls)
    bg = np.ones((N, N)) * 0.9
    bg[maze == 1] = 0.0
    ax.imshow(bg, cmap='gray', vmin=0, vmax=1)
    
    # Draw arrows for policy
    for r in range(N):
        for c in range(N):
            if maze[r, c] == 1:
                continue  # Skip walls
            
            if (r, c) == goal:
                # Mark goal with 'G'
                ax.text(c, r, 'G', ha='center', va='center', fontsize=10, 
                       fontweight='bold', color='red')
            elif (r, c) == start:
                # Mark start with 'S'
                ax.text(c, r, 'S', ha='center', va='center', fontsize=10, 
                       fontweight='bold', color='green')
            else:
                # Draw arrow based on policy
                action = policy[r, c]
                if action == 0:  # Up
                    ax.arrow(c, r+0.3, 0, -0.5, head_width=0.15, head_length=0.2, 
                            fc='blue', ec='blue')
                elif action == 1:  # Down
                    ax.arrow(c, r-0.3, 0, 0.5, head_width=0.15, head_length=0.2, 
                            fc='blue', ec='blue')
                elif action == 2:  # Left
                    ax.arrow(c-0.3, r, -0.5, 0, head_width=0.15, head_length=0.2, 
                            fc='blue', ec='blue')
                elif action == 3:  # Right
                    ax.arrow(c+0.3, r, 0.5, 0, head_width=0.15, head_length=0.2, 
                            fc='blue', ec='blue')
    
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.set_title("Optimal Policy", fontsize=14, fontweight='bold')
    ax.set_xlabel("Column")
    
    # ==========================================================================
    # Add optimal path length info
    # ==========================================================================
    if optimal_length is not None:
        fig.suptitle(f"Optimal Path Length: {optimal_length} steps", 
                    fontsize=12, fontweight='bold', y=1.02)
    
    # ==========================================================================
    # Save or show
    # ==========================================================================
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# =============================================================================
# LOADING LOGIC
# =============================================================================
def list_available_files(save_dir=DEFAULT_SAVE_DIR):
    """List all .npy and .npz (VI) files."""
    path = Path(save_dir)
    if not path.exists():
        return []
    
    files = sorted([f.stem for f in path.glob("*.npy")])
    vi_files = sorted([f.stem for f in path.glob("*_VI.npz")])
    
    return files, vi_files

def load_maze_file(filename, save_dir=DEFAULT_SAVE_DIR):
    """Load a .npy maze file."""
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    filepath = Path(save_dir) / filename
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        mazes = np.load(filepath)
        print(f"✓ Loaded {filepath.name}")
        print(f"  Shape: {mazes.shape} → {mazes.shape[0]} mazes")
        return mazes
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def load_vi_file(maze_filename, save_dir=DEFAULT_SAVE_DIR):
    """
    Load the corresponding VI file for a maze file.
    E.g., 'train_solvable.npy' → 'train_solvable_VI.npz'
    """
    # Construct VI filename
    if maze_filename.endswith('.npy'):
        base_name = maze_filename[:-4]  # Remove .npy
    else:
        base_name = maze_filename
    
    vi_filename = f"{base_name}_VI.npz"
    filepath = Path(save_dir) / vi_filename
    
    if not filepath.exists():
        print(f"⚠️  VI file not found: {vi_filename}")
        print(f"   Run compute_VI.py first to generate VI data")
        return None
    
    try:
        vi_data = np.load(filepath, allow_pickle=True)
        print(f"✓ Loaded VI data: {vi_filename}")
        return vi_data
    except Exception as e:
        print(f"❌ Error loading VI file: {e}")
        return None

def pick_random_maze(mazes, exclude_indices=None):
    """Pick a random maze index."""
    exclude = exclude_indices or []
    valid_indices = [i for i in range(len(mazes)) if i not in exclude]
    if not valid_indices:
        return None
    return random.choice(valid_indices)

# =============================================================================
# INTERACTIVE MODE (With VI Support)
# =============================================================================
def interactive_mode(save_dir=DEFAULT_SAVE_DIR, show_vi=False):
    """Run interactive terminal viewer."""
    print(f"\n🔍 Maze Viewer - Interactive Mode")
    print(f"📁 Looking in: {save_dir}/\n")
    
    files, vi_files = list_available_files(save_dir)
    
    if not files:
        print("No .npy files found. Run the generator first!")
        return
    
    print("Available maze datasets:")
    for i, f in enumerate(files, 1):
        vi_status = "✓ VI" if f in vi_files else "✗ No VI"
        print(f"  {i}. {f}  [{vi_status}]")
    
    # Let user pick a file
    while True:
        choice = input(f"\nSelect file (1-{len(files)}) or 'q' to quit: ").strip()
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Please enter a valid number or 'q'")
    
    # Load mazes
    mazes = load_maze_file(selected_file, save_dir)
    if mazes is None:
        return
    
    # Load VI data if requested
    vi_data = None
    if show_vi:
        vi_data = load_vi_file(selected_file, save_dir)
    
    # Interactive viewing loop
    viewed_indices = set()
    start, goal = (0, 0), (15, 15)
    
    while True:
        # Pick a maze
        idx = pick_random_maze(mazes, exclude_indices=viewed_indices if len(viewed_indices) < len(mazes) else None)
        if idx is None:
            print("\n🎉 You've viewed all mazes! Resetting...")
            viewed_indices.clear()
            idx = pick_random_maze(mazes)
        
        viewed_indices.add(idx)
        maze = mazes[idx]
        
        # Get VI data for this maze if available
        V, policy, opt_length = None, None, None
        if vi_data is not None:
            V = vi_data['V_tables'][idx]
            policy = vi_data['policies'][idx]
            opt_length = vi_data['optimal_lengths'][idx]
        
        # Display
        if not HAS_MATPLOTLIB:
            print_maze_ascii(maze, V, policy, start, goal, idx)
        elif vi_data is not None:
            show_maze_vi_matplotlib(maze, V, policy, start, goal, idx, 
                                   optimal_length=opt_length)
        else:
            show_maze_matplotlib(maze, start, goal, idx)
        
        # Next action
        status = f"Viewed {len(viewed_indices)}/{len(mazes)}"
        if vi_data is not None:
            status += f" | Optimal length: {opt_length}"
        print(f"{status}")
        
        if vi_data is not None:
            next_action = input("Enter=next | 's'=save | 'm'=maze-only | 'q'=quit: ").strip().lower()
        else:
            next_action = input("Enter=next | 's'=save | 'q'=quit: ").strip().lower()
        
        if next_action == 'q':
            print("👋 Goodbye!")
            return
        elif next_action == 's' and HAS_MATPLOTLIB:
            save_name = input("Enter filename (e.g., maze_vi.png): ").strip()
            if save_name:
                if not save_name.endswith('.png'):
                    save_name += '.png'
                if vi_data is not None:
                    show_maze_vi_matplotlib(maze, V, policy, start, goal, idx, 
                                           save_path=save_name, optimal_length=opt_length)
                else:
                    show_maze_matplotlib(maze, start, goal, idx, save_path=save_name)
        elif next_action == 'm' and vi_data is not None:
            # Show maze only (toggle)
            show_maze_matplotlib(maze, start, goal, idx)

# =============================================================================
# COMMAND-LINE MODE
# =============================================================================
def command_line_mode(args):
    """Handle command-line arguments."""
    
    if args.list:
        files, vi_files = list_available_files(args.dir)
        if files:
            print("Available datasets:")
            for f in files:
                vi_status = "✓ VI" if f in vi_files else "✗"
                print(f"  • {f}  [{vi_status}]")
        else:
            print("No .npy files found.")
        return
    
    if not args.file:
        print("❌ Specify --file <filename> or use interactive mode")
        return
    
    mazes = load_maze_file(args.file, args.dir)
    if mazes is None:
        return
    
    # Pick index
    if args.idx is not None:
        idx = args.idx
        if idx < 0 or idx >= len(mazes):
            print(f"❌ Index {idx} out of range [0, {len(mazes)-1}]")
            return
    else:
        idx = pick_random_maze(mazes)
    
    maze = mazes[idx]
    start, goal = (0, 0), (15, 15)
    
    # Load VI if requested
    vi_data = None
    if args.vi:
        vi_data = load_vi_file(args.file, args.dir)
    
    if vi_data is not None:
        V = vi_data['V_tables'][idx]
        policy = vi_data['policies'][idx]
        opt_length = vi_data['optimal_lengths'][idx]
        
        if args.save:
            show_maze_vi_matplotlib(maze, V, policy, start, goal, idx, 
                                   save_path=args.save, optimal_length=opt_length)
        else:
            show_maze_vi_matplotlib(maze, V, policy, start, goal, idx, 
                                   optimal_length=opt_length)
    else:
        if args.save:
            show_maze_matplotlib(maze, start, goal, idx, save_path=args.save)
        else:
            show_maze_matplotlib(maze, start, goal, idx)

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="View generated maze datasets with optional VI visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive mode
  %(prog)s --list                       # List available files
  %(prog)s --file train_solvable        # Random maze from file
  %(prog)s --file train_solvable --idx 42  # Specific maze
  %(prog)s --file train_solvable --vi   # Show with VI data (if available)
  %(prog)s --file test_shapes --vi --save output.png  # Save VI visualization
        """
    )
    
    parser.add_argument('--file', '-f', type=str, help='Maze file to load (without .npy)')
    parser.add_argument('--idx', '-i', type=int, help='Specific maze index to view')
    parser.add_argument('--dir', '-d', type=str, default=DEFAULT_SAVE_DIR, 
                       help=f'Data directory (default: {DEFAULT_SAVE_DIR})')
    parser.add_argument('--list', '-l', action='store_true', help='List available files')
    parser.add_argument('--save', '-s', type=str, help='Save visualization as PNG')
    parser.add_argument('--vi', '-v', action='store_true', 
                       help='Show VI data (V-table + Policy) if available')
    
    args = parser.parse_args()
    
    if args.file or args.list or args.idx or args.save:
        command_line_mode(args)
    else:
        interactive_mode(args.dir, show_vi=args.vi)

if __name__ == "__main__":
    main()