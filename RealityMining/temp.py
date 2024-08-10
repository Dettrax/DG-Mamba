import matplotlib.pyplot as plt
import numpy as np
import random

# Assuming delta, B, and C matrices have already been extracted

# Sample data for demonstration (replace with actual data)
# Shape: (nodes, lookback, features)
delta = np.random.rand(96, 3, 96)  # Example shape: (96, 3, 96)
B = np.random.rand(96, 3, 96)  # Example shape: (96, 3, 96)
C = np.random.rand(96, 3, 96)  # Example shape: (96, 3, 96)

# Randomly select 3 features
random_features = random.sample(range(delta.shape[-1]), 3)

# Visualization for Delta matrix
plt.figure(figsize=(14, 4))
for i in random_features:
    plt.plot(delta[:, :, i].flatten(), label=f'Delta feature {i+1}')
plt.title('Delta Matrix Values for Random Features')
plt.xlabel('Nodes * Lookback')
plt.ylabel('Delta Values')
plt.legend()
plt.grid(True)
plt.show()

# Visualization for B matrix
plt.figure(figsize=(14, 4))
for i in random_features:
    plt.plot(B[:, :, i].flatten(), label=f'B feature {i+1}')
plt.title('B Matrix Values for Random Features')
plt.xlabel('Nodes * Lookback')
plt.ylabel('B Values')
plt.legend()
plt.grid(True)
plt.show()

# Visualization for C matrix
plt.figure(figsize=(14, 4))
for i in random_features:
    plt.plot(C[:, :, i].flatten(), label=f'C feature {i+1}')
plt.title('C Matrix Values for Random Features')
plt.xlabel('Nodes * Lookback')
plt.ylabel('C Values')
plt.legend()
plt.grid(True)
plt.show()
