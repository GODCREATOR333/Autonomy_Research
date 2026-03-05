import rerun as rr
import numpy as np

class MazeVisualizer:
    def __init__(self, experiment_name: str = "Autonomy_Research"):
        rr.init("my_app", spawn=False)
        rr.log("maze", rr.ViewCoordinates.RDF, timeless=True)

    def log_maze_batch(self, maze_batch: np.ndarray):
        num_trials, N, _ = maze_batch.shape
        
        for trial_idx in range(num_trials):
            maze = maze_batch[trial_idx]
            rr.set_time_sequence("maze_index", trial_idx)
            
            # 0=Free (White), 1=Wall (Black)
            img_array = np.where(maze == 1, 0, 255).astype(np.uint8)
            rr.log("maze/grid", rr.Image(img_array))
            
            # Start (Green), Goal (Red)
            rr.log("maze/start_goal", rr.Points2D(
                positions=[[0.5, 0.5], [N-0.5, N-0.5]], 
                colors=[[0, 255, 0], [255, 0, 0]], 
                radii=0.4
            ))