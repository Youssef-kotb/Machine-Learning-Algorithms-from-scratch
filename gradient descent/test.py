import numpy as np
from gradient_descent import gradient_descent

# --- Small dataset for testing ---
X = np.array([
    [1, 2],
    [2, 3],
    [4, 5]
])                         # Shape (3 samples, 2 features)

y = np.array([5, 8, 14])   # Shape (3,)

# --- Initial parameters ---
w = np.array([0.0, 0.0])   # Start from zeros

# --- Create optimizer ---
opt = gradient_descent(learning_rate=0.01, max_iterations=100)
opt.cost_function = 'MSE'

# --- Run gradient descent manually ---
print("Initial w:", w)

for i in range(20):  # run 20 update steps
    w = opt.step(actual_outputs=y, values_vector=X, parameters_vector=w)
    print(f"Step {i+1} → w = {w}")

print("\nFinal learned parameters:", w)
