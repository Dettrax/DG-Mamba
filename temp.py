import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def precompute_zero_entries(sparse_matrix):
    """ Precomputes zero entries in the given sparse matrix. """
    # Convert to COO format for easy row/col access
    sparse_coo = sparse_matrix.tocoo()
    # Generate all possible (row, col) pairs
    all_indices = {(i, j) for i in range(sparse_matrix.shape[0]) for j in range(sparse_matrix.shape[1])}
    # Remove all non-zero indices
    nonzero_indices = {(i, j) for i, j in zip(sparse_coo.row, sparse_coo.col)}
    zero_indices = list(all_indices - nonzero_indices)
    return zero_indices

def sample_negative_entries(sparse_matrix, n_samples, precomputed_zeros=None):
    """ Samples n_samples negative entries from sparse_matrix. """
    if precomputed_zeros is None:
        precomputed_zeros = precompute_zero_entries(sparse_matrix)
    # Sample from the zero entries
    sampled_indices = np.random.choice(len(precomputed_zeros), n_samples, replace=False)
    return [precomputed_zeros[i] for i in sampled_indices]

# Example usage
# Define a small sparse matrix
rows = np.array([0, 1, 2, 3])
cols = np.array([1, 2, 3, 4])
data = np.ones(len(rows))
sparse_matrix = csr_matrix((data, (rows, cols)), shape=(5, 5))

# Sample 3 negative entries
negative_samples = sample_negative_entries(sparse_matrix, 3)
print("Sampled Negative Entries:", negative_samples)
print(sparse_matrix.toarray())

import numpy as np
from scipy.sparse import csr_matrix


def find_and_sample_zero_entries(sparse_matrix, num_samples=None):
    # Convert sparse matrix to dense format
    dense_matrix = sparse_matrix.toarray()

    # Create a boolean array: True where elements are zero
    zero_mask = dense_matrix == 0

    # Get the indices of zero elements
    zero_indices = np.argwhere(zero_mask)

    # Check if sampling is requested
    if num_samples is not None and num_samples > 0:
        # Ensure that there are enough zero entries to sample from
        if num_samples > len(zero_indices):
            raise ValueError("Requested more samples than available zeros.")
        # Sample indices randomly without replacement
        sampled_indices = zero_indices[np.random.choice(len(zero_indices), num_samples, replace=False)]
        return sampled_indices
    return zero_indices


# Sample 3 zero entries
sampled_zero_entries = find_and_sample_zero_entries(sparse_matrix, 3)
print("Sampled zero entries indices:", sampled_zero_entries)
