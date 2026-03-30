import numpy as np
import sys
import os

# Link to Alios
sys.path.append("../Alios/Database")
from alios_db import register_run

# 1. Load the two policies you already trained/built
geo_q = np.load("../Alios/Artifacts/PURE_GEO_ANGLE_policy.npy") # (256, 4) or similar
ego_q = np.load("../Alios/Artifacts/EGO_REACTIVE_SURVIVOR_512_policy.npy") # (512, 4)

# 2. Create the Meta-Table (Size 1024 to accommodate both modes)
# Rows 0-511: The Tumble (Ego)
# Rows 512-1023: The Run (Geo)
meta_q = np.zeros((1024, 4))

# Fill Ego part
meta_q[0:512] = ego_q

# Fill Geo part 
# We need to map the 9 compass IDs to rows 512-520
# For each compass state, we put the 'Magnet' actions
for comp_id in range(9):
    # This assumes your geo_q was mdp-based. We'll extract the 
    # directional logic and apply it to these rows.
    # [Up, Down, Left, Right]
    # Example for comp_id 8 (South-East): Prefer Down or Right
    # (For simplicity, we can just hardcode these 9 rows to match your magnet logic)
    meta_q[512 + comp_id] = [-100, 100, -100, 100] 

# 3. Register as a single "DMP Agent"
register_run(
    run_id="DMP_REACTIVE_SWITCHER_V1",
    algo="DMP (Run-and-Tumble)",
    state_repr="reactive_dmp", # Points to our new modular decoder
    config_dict={"mode": "Reactive Switch", "trigger": "Path Blocked"},
    q_table=meta_q
)