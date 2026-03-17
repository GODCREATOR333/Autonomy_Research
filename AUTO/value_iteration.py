import numpy as np

# 1. Define Grid Environment
grid_rows = 3
grid_cols = 3
start = (0, 0)
goal = (2, 2)
gamma = 1

# 2. Define Actions
actions = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}
action_names = list(actions.keys())




# 3. Initialize Value Table and Policy Table    
V = np.zeros((grid_rows, grid_cols))
policy = np.full((grid_rows, grid_cols), 'none', dtype=object)

# 4. Storage for Plotting Later
V_history = [V.copy()]
policy_history = [policy.copy()]

# 5. Value Iteration Loop
for k in range(6):  # 5 iterations is enough for a 3x3 grid
    new_V = np.zeros((grid_rows, grid_cols))
    new_policy = np.full((grid_rows, grid_cols), 'none', dtype=object)
    
    # Iterate through every state in the grid
    for r in range(grid_rows):
        for c in range(grid_cols):
            
            # If we are at the goal, it's a terminal state. Value is 0.
            if (r, c) == goal:
                new_V[r, c] = 0
                new_policy[r, c] = "GOAL"
                continue
            
            action_values =[]
            
            # Evaluate all 4 actions
            for action_name in action_names:
                dr, dc = actions[action_name]
                next_r = r + dr
                next_c = c + dc
                
                # Boundary check: If moving hits a wall, stay in current state
                if next_r < 0 or next_r >= grid_rows or next_c < 0 or next_c >= grid_cols:
                    next_r, next_c = r, c 
                    
                # Assign rewards
                if (next_r, next_c) == goal:
                    reward = 10
                else:
                    reward = -1
                    
                # Bellman Equation: R + gamma * Vk(s')
                # Notice we use `V` (old iteration), not `new_V`
                value = reward + gamma * V[next_r, next_c]
                action_values.append(value)
                
            # Extract highest value and corresponding action
            best_value = np.max(action_values)
            best_action_idx = np.argmax(action_values)
            
            # Update the new tables
            new_V[r, c] = best_value
            new_policy[r, c] = action_names[best_action_idx]
            
    # Overwrite old V with new_V for the next iteration
    V = new_V.copy()
    
    # Save to history for our plots
    V_history.append(V.copy())
    policy_history.append(new_policy.copy())
    
    print(f"\n--- Iteration {k+1} ---")
    print(np.round(V, 1))


import matplotlib.pyplot as plt

# 1. Map string actions to Unicode arrows for visual appeal
arrow_map = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "none": "-",   # For initialization where no action is chosen
    "GOAL": "★"    # For the terminal state
}

# 2. Create the figure: 2 rows (Value & Policy) and 5 columns (Iterations)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
fig.suptitle("Value Iteration & Policy Extraction Progression", fontsize=22, fontweight='bold')

for k in range(5):
    ax_v = axes[0, k]  # Top row for Values
    ax_p = axes[1, k]  # Bottom row for Policies
    
    # 3. Setup grid limits and titles
    for ax in[ax_v, ax_p]:
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5) # Inverted so (0,0) is top-left
        
        # Draw gridlines
        ax.set_xticks(np.arange(-0.5, 3, 1))
        ax.set_yticks(np.arange(-0.5, 3, 1))
        ax.grid(color='black', linewidth=2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)

    ax_v.set_title(f"Iteration {k}\nValues", fontsize=16)
    ax_p.set_title(f"Iteration {k}\nPolicy", fontsize=16)
    
    # 4. Populate the cells with data
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Highlight the GOAL state
            if (r, c) == goal:
                ax_v.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='lightgreen', alpha=0.5))
                ax_p.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='lightgreen', alpha=0.5))
            
            # Extract and write the Value
            val = V_history[k][r, c]
            ax_v.text(c, r, f"{val:.0f}", ha='center', va='center', fontsize=20, fontweight='bold')
            
            # Extract and write the Policy Arrow
            pol = policy_history[k][r, c]
            ax_p.text(c, r, arrow_map[pol], ha='center', va='center', fontsize=28, fontweight='bold')
            
    # 5. Draw progression arrows between the columns (skip after the last column)
    if k < 4:
        # Arrow for Value row
        ax_v.annotate("", xy=(1.25, 0.5), xytext=(1.05, 0.5), 
                      xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=4, color='gray'))
        # Arrow for Policy row
        ax_p.annotate("", xy=(1.25, 0.5), xytext=(1.05, 0.5), 
                      xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=4, color='gray'))

plt.tight_layout()
plt.subplots_adjust(top=0.85, wspace=0.4) # Add spacing for the arrows and title
plt.show()