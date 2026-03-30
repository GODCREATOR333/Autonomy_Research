import numpy as np
import math
import sys
import os

# Link to Alios Registry
sys.path.append("../Alios") 
from alios_db import register_run

# 1. Setup Parameters
GRID_SIZE = 16
GOAL = (15, 15)

def get_angle_based_action(r, c):
    """Calculates the best grid action using the Cartesian Angle to Goal."""
    if (r, c) == GOAL:
        return 1 # Stay/Default
    
    # --- YOUR REFERENCE CODE LOGIC ---
    # Map Grid Row index to Cartesian Y (Negative because Row increases downwards)
    delta_y = -(GOAL[0] - r)
    delta_x = GOAL[1] - c
    
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad) # Range: -180 to 180
    
    # --- DISCRETIZATION INTO 4 BINS ---
    # 0 deg is Right (+x). -90 deg is Down (-y). 90 deg is Up (+y). 180 deg is Left (-x).
    
    # 1. Right: -45 to 45
    if -45 <= angle_deg < 45:
        return 3 # RIGHT
    # 2. Up: 45 to 135
    elif 45 <= angle_deg < 135:
        return 0 # UP
    # 3. Down: -135 to -45
    elif -135 <= angle_deg < -45:
        return 1 # DOWN
    # 4. Left: Everything else
    else:
        return 2 # LEFT

# 2. Build the Q-Table for MDP (256 states)
geo_q = np.zeros((256, 4))

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        s_id = r * GRID_SIZE + c
        
        target_action = get_angle_based_action(r, c)
        
        # High contrast Q-values for the "Magnet" effect
        geo_q[s_id, :] = -100.0
        geo_q[s_id, target_action] = 100.0

# 3. Register with Alios
register_run(
    run_id="PURE_GEO_ANGLE",
    algo="Analytical Magnet",
    state_repr="mdp",
    config_dict={
        "logic": "Trigonometric atan2",
        "Goal": str(GOAL),
        "physics": "Pure Vector Field (No Wall Awareness)"
    },
    q_table=geo_q
)

print(f"✅ PURE_GEO_ANGLE registered. Arrows are now mathematically aligned to {GOAL}")