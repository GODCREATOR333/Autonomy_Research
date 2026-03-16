#!/usr/bin/env python3
"""
view_vi.py - Side-by-side visualization of Maze + V-Table + Policy.

Usage:
    python view_vi.py                                    # Interactive mode
    python view_vi.py --file train_solvable --idx 42     # View specific maze
    python view_vi.py --file train_solvable --save out.png  # Save as image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import argparse
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "save_dir": "data_jax",
    "grid_size": 16,
    "start": (0, 0),
    "goal": (15, 15),
}

ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
ACTION_ARROWS = {0: (0, -0.4), 1: (0, 0.4), 2: (-0.4, 0), 3: (0.4, 0)}

# Colors
BG = '#1A1A2E'
COLOR_WALL = '#2C3E50'
COLOR_OPEN = '#ECF0F1'
COLOR_START = '#3498DB'
COLOR_GOAL = '#2ECC71'
COLOR_TEXT = 'white'

# =============================================================================
# LOADING FUNCTIONS
# =============================================================================
def load_maze_file(filename, save_dir):
    """Load maze .npy file."""
    filepath = Path(save_dir) / f"{filename}.npy"
    if not filepath.exists():
        print(f"❌ Not found: {filepath}")
        return None
    mazes = np.load(filepath)
    print(f"✓ Loaded {filename}.npy: {mazes.shape[0]} mazes")
    return mazes

def load_vi_file(filename, save_dir):
    """Load corresponding VI .npz file."""
    filepath = Path(save_dir) / f"{filename}_VI.npz"
    if not filepath.exists():
        print(f"⚠️  VI file not found: {filepath}")
        return None
    data = np.load(filepath, allow_pickle=True)
    print(f"✓ Loaded {filename}_VI.npz")
    return data

# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================
def draw_maze_panel(ax, maze, start, goal):
    """Draw maze with start/goal markers."""
    N = maze.shape[0]
    
    # Base image
    img = np.ones((N, N, 3)) * 0.95  # Open = light gray
    img[maze == 1] = [0.17, 0.24, 0.31]  # Wall = dark blue-gray
    img[start] = [0.20, 0.60, 0.86]  # Start = blue
    img[goal] = [0.18, 0.80, 0.44]   # Goal = green
    
    ax.imshow(img, extent=[-0.5, N-0.5, N-0.5, -0.5], 
              interpolation='nearest', aspect='equal', zorder=0)
    
    # Grid lines
    for i in range(N + 1):
        ax.axhline(i - 0.5, color='#BDC3C7', linewidth=0.3, zorder=1)
        ax.axvline(i - 0.5, color='#BDC3C7', linewidth=0.3, zorder=1)
    
    # Labels
    ax.text(start[1], start[0], 'S', ha='center', va='center', 
            color='white', fontsize=10, fontweight='bold', zorder=3)
    ax.text(goal[1], goal[0], 'G', ha='center', va='center', 
            color='white', fontsize=10, fontweight='bold', zorder=3)
    
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Maze', color=COLOR_TEXT, fontsize=12, pad=8)

def draw_v_table_panel(ax, V, maze, start, goal):
    """Draw V-table as heatmap with walls masked."""
    N = maze.shape[0]
    
    # Mask walls for visualization
    V_display = V.copy()
    V_display[maze == 1] = np.nan
    
    # Colorbar limits
    vmin = np.nanmin(V_display)
    vmax = np.nanmax(V_display)
    
    im = ax.imshow(V_display, cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='equal', zorder=0)
    
    # Overlay walls as dark rectangles
    for r in range(N):
        for c in range(N):
            if maze[r, c] == 1:
                ax.add_patch(patches.Rectangle(
                    (c-0.5, r-0.5), 1, 1, facecolor=COLOR_WALL, 
                    linewidth=0, zorder=1
                ))
    
    # Start/goal markers
    ax.text(start[1], start[0], 'S', ha='center', va='center', 
            color='white', fontsize=9, fontweight='bold', zorder=3)
    ax.text(goal[1], goal[0], 'G', ha='center', va='center', 
            color='white', fontsize=9, fontweight='bold', zorder=3)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=COLOR_TEXT, labelsize=7)
    cbar.set_label('Value', color=COLOR_TEXT, rotation=270, labelpad=12)
    
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('V-Table (Value Function)', color=COLOR_TEXT, fontsize=12, pad=8)
    ax.tick_params(colors=COLOR_TEXT, labelsize=6)

def draw_policy_panel(ax, policy, maze, start, goal):
    """Draw policy as arrows on maze background."""
    N = maze.shape[0]
    
    # Background: walls vs open
    for r in range(N):
        for c in range(N):
            color = COLOR_WALL if maze[r, c] == 1 else '#2E4057'
            ax.add_patch(patches.Rectangle(
                (c-0.5, r-0.5), 1, 1, facecolor=color, 
                linewidth=0.3, edgecolor='#4A4A6A', zorder=0
            ))
    
    # Draw arrows for policy
    for r in range(N):
        for c in range(N):
            if maze[r, c] == 1 or (r, c) == goal:
                continue
            action = policy[r, c]
            if action in ACTION_ARROWS:
                du, dv = ACTION_ARROWS[action]
                ax.arrow(c, r, du, dv, head_width=0.12, head_length=0.15, 
                        fc='#F1C40F', ec='#F1C40F', zorder=2, width=0.04)
    
    # Start/goal markers
    for (pos, color, label) in [(start, COLOR_START, 'S'), (goal, COLOR_GOAL, 'G')]:
        ax.add_patch(patches.Rectangle(
            (pos[1]-0.5, pos[0]-0.5), 1, 1, facecolor=color, linewidth=0, zorder=3
        ))
        ax.text(pos[1], pos[0], label, ha='center', va='center', 
                color='white', fontsize=9, fontweight='bold', zorder=4)
    
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Optimal Policy', color=COLOR_TEXT, fontsize=12, pad=8)
    ax.tick_params(colors=COLOR_TEXT, labelsize=6)

# =============================================================================
# MAIN VIEWER CLASS
# =============================================================================
class VIViewer:
    def __init__(self, mazes, vi_data, save_dir):
        self.mazes = mazes
        self.vi_data = vi_data
        self.save_dir = save_dir
        self.N = CONFIG["grid_size"]
        self.start = CONFIG["start"]
        self.goal = CONFIG["goal"]
        self.idx = 0
        
        self._setup_figure()
        self._update_display()
    
    def _setup_figure(self):
        """Create figure with 3 panels + controls."""
        self.fig = plt.figure(figsize=(16, 6))
        self.fig.patch.set_facecolor(BG)
        
        # Panel positions (3 equal columns)
        left_margin = 0.05
        right_margin = 0.98
        top_margin = 0.92
        bot_margin = 0.12
        gap = 0.03
        
        total_w = right_margin - left_margin
        total_h = top_margin - bot_margin
        panel_w = (total_w - 2 * gap) / 3
        
        col0 = left_margin
        col1 = left_margin + panel_w + gap
        col2 = left_margin + 2 * (panel_w + gap)
        
        # Create panels
        self.ax_maze = self.fig.add_axes([col0, bot_margin + 0.08, panel_w, total_h - 0.08])
        self.ax_v = self.fig.add_axes([col1, bot_margin + 0.08, panel_w, total_h - 0.08])
        self.ax_policy = self.fig.add_axes([col2, bot_margin + 0.08, panel_w, total_h - 0.08])
        
        for ax in [self.ax_maze, self.ax_v, self.ax_policy]:
            ax.set_facecolor(BG)
        
        # Slider for maze index
        ax_slider = self.fig.add_axes([0.25, 0.04, 0.5, 0.03])
        ax_slider.set_facecolor('#2C3E50')
        self.slider = Slider(ax_slider, 'Maze Index', 0, len(self.mazes)-1, 
                            valinit=0, valstep=1, color='#E74C3C')
        self.slider.label.set_color(COLOR_TEXT)
        self.slider.valtext.set_color(COLOR_TEXT)
        
        # Navigation buttons
        btn_w, btn_h = 0.04, 0.04
        self.btn_prev = Button(self.fig.add_axes([0.12, 0.035, btn_w, btn_h]), 
                              '◀', color='#2C3E50', hovercolor='#3D566E')
        self.btn_next = Button(self.fig.add_axes([0.17, 0.035, btn_w, btn_h]), 
                              '▶', color='#2C3E50', hovercolor='#3D566E')
        self.btn_save = Button(self.fig.add_axes([0.85, 0.035, btn_w, btn_h]), 
                              '💾', color='#2C3E50', hovercolor='#3D566E')
        
        for btn in [self.btn_prev, self.btn_next, self.btn_save]:
            btn.label.set_color(COLOR_TEXT)
            btn.label.set_fontsize(11)
        
        # Connect callbacks
        self.slider.on_changed(self._on_slider_change)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_save.on_clicked(self._on_save)
        
        # Title
        self.title = self.fig.suptitle('', color=COLOR_TEXT, fontsize=13, fontweight='bold', y=0.99)
    
    def _update_display(self):
        """Redraw all panels for current maze index."""
        maze = self.mazes[self.idx]
        
        # Get VI data
        if self.vi_data is not None:
            V = self.vi_data['V_tables'][self.idx]
            policy = self.vi_data['policies'][self.idx]
            opt_len = self.vi_data['optimal_lengths'][self.idx]
        else:
            V = np.zeros((self.N, self.N))
            policy = np.zeros((self.N, self.N), dtype=int)
            opt_len = None
        
        # Clear and redraw panels
        self.ax_maze.clear()
        self.ax_v.clear()
        self.ax_policy.clear()
        
        draw_maze_panel(self.ax_maze, maze, self.start, self.goal)
        draw_v_table_panel(self.ax_v, V, maze, self.start, self.goal)
        draw_policy_panel(self.ax_policy, policy, maze, self.start, self.goal)
        
        # Update title
        status = f"Optimal Length: {opt_len}" if opt_len is not None else "VI data not available"
        self.title.set_text(f"Maze #{self.idx} | {status}")
        
        self.fig.canvas.draw_idle()
    
    def _on_slider_change(self, val):
        """Handle slider movement."""
        self.idx = int(val)
        self._update_display()
    
    def _on_prev(self, event):
        """Go to previous maze."""
        self.idx = max(0, self.idx - 1)
        self.slider.set_val(self.idx)
    
    def _on_next(self, event):
        """Go to next maze."""
        self.idx = min(len(self.mazes) - 1, self.idx + 1)
        self.slider.set_val(self.idx)
    
    def _on_save(self, event):
        """Save current view as PNG."""
        filename = f"maze_{self.idx:04d}_vi.png"
        filepath = Path(self.save_dir) / filename
        self.fig.savefig(filepath, dpi=150, facecolor=BG, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    
    def show(self):
        """Display the viewer."""
        plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="View Maze + VI + Policy side-by-side")
    parser.add_argument('--file', '-f', type=str, required=True, 
                       help='Maze file to load (without .npy)')
    parser.add_argument('--idx', '-i', type=int, default=None, 
                       help='Specific maze index to view')
    parser.add_argument('--dir', '-d', type=str, default=CONFIG["save_dir"], 
                       help=f'Data directory (default: {CONFIG["save_dir"]})')
    parser.add_argument('--save', '-s', type=str, default=None, 
                       help='Save initial view as PNG')
    
    args = parser.parse_args()
    
    # Load mazes
    mazes = load_maze_file(args.file, args.dir)
    if mazes is None:
        return
    
    # Load VI data (optional)
    vi_data = load_vi_file(args.file, args.dir)
    
    # Create viewer
    viewer = VIViewer(mazes, vi_data, args.dir)
    
    # Set initial index if specified
    if args.idx is not None:
        if 0 <= args.idx < len(mazes):
            viewer.idx = args.idx
            viewer.slider.set_val(args.idx)
            viewer._update_display()
    
    # Save if requested
    if args.save:
        filepath = Path(args.dir) / args.save
        viewer.fig.savefig(filepath, dpi=150, facecolor=BG, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        return
    
    # Show interactive viewer
    viewer.show()

if __name__ == "__main__":
    main()