# environment.py
import numpy as np
from config import *

action_to_delta = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}

def get_state(maze, pos, last_action=None):
    row, col = pos
    window   = []
    for dr in range(-HALF, HALF + 1):
        for dc in range(-HALF, HALF + 1):
            r, c = row+dr, col+dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                window.append(maze[r, c])
            else:
                window.append(1)
    dr_goal = int(np.sign(GOAL[0] - row))
    dc_goal = int(np.sign(GOAL[1] - col))
    last_a  = last_action if last_action is not None else -1
    return tuple(window) + (dr_goal, dc_goal, last_a)

def step(maze, pos, action):
    row, col = pos
    dr, dc   = action_to_delta[action]
    nr, nc   = row+dr, col+dc
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        return pos, R_WALL, False
    if maze[nr, nc] == 1:
        return pos, R_WALL, False
    if (nr, nc) == GOAL:
        return (nr, nc), R_GOAL, True
    return (nr, nc), R_STEP, False

def evaluate(Q, test_mazes):
    """
    Run greedy policy on all test mazes.
    Returns success rate and average steps.
    """
    successes  = 0
    step_counts = []

    for maze in test_mazes:
        pos         = START
        last_action = None
        success     = False

        for t in range(MAX_STEPS):
            state  = get_state(maze, pos, last_action)

            if state in Q:
                action = np.argmax(Q[state])
            else:
                # unseen state fallback — go toward goal
                dr = GOAL[0] - pos[0]
                dc = GOAL[1] - pos[1]
                if abs(dr) >= abs(dc):
                    action = 1 if dr > 0 else 0
                else:
                    action = 3 if dc > 0 else 2

            pos, _, done = step(maze, pos, action)
            last_action  = action

            if done:
                successes += 1
                step_counts.append(t + 1)
                success = True
                break

        if not success:
            step_counts.append(MAX_STEPS)

    success_rate = successes / len(test_mazes)
    avg_steps    = np.mean(step_counts)
    return success_rate, avg_steps