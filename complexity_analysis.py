import numpy as np
import networkx as nx
import os
import pandas as pd

def analyze_maze(grid):
    rows, cols = grid.shape
    G = nx.grid_2d_graph(rows, cols)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                G.remove_node((r, c))
    
    start, goal = (0,0), (rows-1, cols-1)
    
    if start not in G or goal not in G or not nx.has_path(G, start, goal):
        # technically unreachable
        return float('inf'), 0, 0

    # Tortuosity
    path_len = nx.shortest_path_length(G, start, goal)
    manhattan_dist = (rows - 1) + (cols - 1)
    tortuosity = path_len / manhattan_dist

    # Spectral gap
    eigenvalues = nx.laplacian_spectrum(G)
    spectral_gap = sorted(eigenvalues)[1] if len(eigenvalues) > 1 else 0

    # Dead-end density
    degrees = dict(G.degree())
    dead_ends = sum(1 for v in degrees if degrees[v] == 1)
    dead_end_density = dead_ends / len(G)

    return tortuosity, spectral_gap, dead_end_density

# --- process dataset ---
data_dir = './data/maze_dataset'
results = []

for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        filepath = os.path.join(data_dir, filename)
        mazes = np.load(filepath)
        for i, maze in enumerate(mazes):
            t, s, d = analyze_maze(maze)
            results.append({
                'file': filename,
                'maze_idx': i,
                'tortuosity': t,
                'spectral_gap': s,
                'dead_end_density': d
            })

df = pd.DataFrame(results)
# clip tiny or negative spectral_gap to a small positive number
# clip negatives to zero, then scale
df['spectral_gap_clipped'] = df['spectral_gap'].clip(lower=0)

# normalize between 0 and 1
df['spectral_inv_norm'] = (1 / (df['spectral_gap_clipped'] + 1e-5))  # avoid divide by 0
df['spectral_inv_norm'] = df['spectral_inv_norm'] / df['spectral_inv_norm'].max()

# weighted complexity score
df['complexity_score'] = 0.5 * df['tortuosity'] + 0.3 * df['spectral_inv_norm'] + 0.2 * df['dead_end_density']
df.to_csv('maze_analysis.csv', index=False)