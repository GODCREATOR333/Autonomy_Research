# config.py
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
EPSILON_DECAY = 0.999990
LAMBDA        = 0.9        # for TD(lambda) only

# evaluation
EVAL_EVERY    = 1000
SEEDS         = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# rewards
R_GOAL        = +100.0
R_WALL        = -10.0
R_STEP        = -0.1
R_REVISIT     = -1.0

# data
TRAIN_PATH    = './data/N16_p0100_train.npy'
TEST_PATH     = './data/N16_p0100_test.npy'

# results
RESULTS_DIR   = './results'
RESULTS_FILE  = './results/training_results_N16_p0100.pkl'