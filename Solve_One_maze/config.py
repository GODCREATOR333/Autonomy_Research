# config.py — single source of truth for everything
GRID_SIZE     = 16
N_ACTIONS     = 4
START         = (0, 0)
GOAL          = (GRID_SIZE-1, GRID_SIZE-1)
WINDOW_SIZE   = 3
HALF          = WINDOW_SIZE // 2

# training
N_EPISODES    = 100000
MAX_STEPS     = GRID_SIZE * GRID_SIZE * 8
ALPHA         = 0.1
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.999990   # slower decay for better coverage

# evaluation
EVAL_EVERY    = 1000       # evaluate every N episodes
EVAL_MAZES    = 200        # how many test mazes to evaluate on

# rewards
R_GOAL        = +100.0
R_WALL        = -10.0
R_STEP        = -0.1
R_REVISIT     = -1.0       # per visit count