import numpy as np

# Given values
x = np.array([1, 3, 4])
A = np.array([4, 2])
C = np.array([1, 3, 2])
delta = np.array([1, 3, 4])
delta_A = np.array([1, 3, 4, 2])

# Functions f_A and f_B
def f_A(delta_i, A):
    return np.exp(delta_i * A)

def f_B(delta_i, B_i):
    return delta_i * B_i

# Compute Ā and B̄ for each time step
A_bar = [f_A(delta[i], A) for i in range(len(delta))]
B_bar = [f_B(delta[i], C[i]) for i in range(len(delta))]

# To correct the previous approach, let's compute each element of alpha_tilde as a sum of scalar products
# For each entry, we need to multiply the appropriate scalar values and sum them up

# Initialize alpha_tilde as a matrix of the correct shape
alpha_tilde = np.zeros((len(x), len(x)))

# Fill alpha_tilde matrix with scalar values
for i in range(len(x)):
    for j in range(i+1):
        if j == i:
            alpha_tilde[i, j] = C[i] * B_bar[i].sum()
        else:
            product_A = np.prod([A_bar[k].sum() for k in range(j+1, i+1)])
            alpha_tilde[i, j] = C[i] * product_A * B_bar[j].sum()

# Compute the output y
y = np.dot(alpha_tilde, x)
alpha_tilde, y
