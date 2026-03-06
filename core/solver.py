import numpy as np

def run_value_iteration(maze, gamma=0.9, theta=1e-4):
    N = maze.shape[0]
    V = np.zeros((N, N))
    
    # Parameters
    step_reward = -1       # normal step
    wall_penalty = -10     # hitting a wall

    while True:
        delta = 0
        new_V = np.copy(V)
        for r in range(N):
            for c in range(N):
                if maze[r, c] == 1 or (r == N-1 and c == N-1):
                    continue

                vals = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        if maze[nr, nc] == 0:
                            reward = step_reward
                            vals.append(reward + gamma * V[nr, nc])
                        else:
                            # Hit wall → big penalty
                            reward = wall_penalty
                            vals.append(reward + gamma * V[r, c])
                    else:
                        # Out of bounds → treat same as wall
                        vals.append(wall_penalty + gamma * V[r, c])
                
                new_V[r, c] = max(vals)
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        V = new_V
        if delta < theta:
            break
    return V