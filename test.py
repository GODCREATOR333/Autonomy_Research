import numpy as np

# To load a solved file
data = np.load("value_iteration/N16_P0100_train_unsolvable_VI_solved.npz")

# Access them specifically
v_matrix = data['values']  # Shape: (1000, 16, 16)
p_matrix = data['policy']  # Shape: (1000, 16, 16)

# Example: Get Value and Policy for Maze 5, Row 15, Col 10
print(f"Value: {v_matrix[5, 15, 10]}")
print(f"Action: {p_matrix[5, 15, 10]}")

print(v_matrix[0])
print(p_matrix[0])