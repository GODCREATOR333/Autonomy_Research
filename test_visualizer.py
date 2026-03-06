import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for headless
import matplotlib.pyplot as plt

def maze_to_graph(grid):
    rows, cols = grid.shape
    G = nx.grid_2d_graph(rows, cols)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                G.remove_node((r, c))
    return G

def plot_maze_to_file(G, filename='maze.png', start=(0,0), goal=None, highlight_path=True):
    if goal is None:
        goal = max(G.nodes, key=lambda x: x[0]+x[1])
    pos = {node: (node[1], -node[0]) for node in G.nodes()}

    degrees = dict(G.degree())
    node_color = ['red' if degrees[n]==1 else 'lightgray' for n in G.nodes()]

    plt.figure(figsize=(8,8))
    nx.draw(G, pos=pos, node_color=node_color, node_size=120, with_labels=False, edge_color='black')

    if highlight_path and nx.has_path(G, start, goal):
        path = nx.shortest_path(G, start, goal)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='blue', node_size=150)
        nx.draw_networkx_edges(G, pos, edgelist=list(zip(path[:-1], path[1:])), edge_color='blue', width=2)

    plt.title(f"Maze Visualization ({G.number_of_nodes()} nodes)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Load maze and save visualization ---
maze_file = './data/maze_dataset/N16_p0400.npy'
mazes = np.load(maze_file)
maze_idx = 5  # pick the first maze
grid = mazes[maze_idx]

G = maze_to_graph(grid)
plot_maze_to_file(G, filename='maze_0.png')