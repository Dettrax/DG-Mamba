import numpy as np
import torch
import time
from scipy.sparse import random as sparse_random
from scipy.sparse import coo_matrix

# Generate a large random sparse adjacency matrix
np.random.seed(0)
adj_matrix_sparse_np = sparse_random(11000, 11000, density=0.001, format='coo', dtype=np.int64)

# Convert to a dense matrix for PyTorch
rows = adj_matrix_sparse_np.row
cols = adj_matrix_sparse_np.col
data = adj_matrix_sparse_np.data

# Create the same sparse matrix in PyTorch
indices = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
values = torch.tensor(data, dtype=torch.float32)
adj_matrix_torch_cpu = torch.sparse.FloatTensor(indices, values, torch.Size([11000, 11000]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adj_matrix_torch_gpu = adj_matrix_torch_cpu.to(device)


def sample_positive_negative_nodes_sparse_np(adj_matrix, num_samples):
    # Find all positive edges (existing edges)
    pos_edges = np.column_stack((adj_matrix.row, adj_matrix.col))
    # Find all negative edges (non-existing edges)
    total_elements = adj_matrix.shape[0] * adj_matrix.shape[1]
    neg_mask = np.ones(total_elements, dtype=bool)
    neg_mask[adj_matrix.row * adj_matrix.shape[1] + adj_matrix.col] = False
    neg_indices = np.nonzero(neg_mask)[0]
    neg_edges = np.column_stack((neg_indices // adj_matrix.shape[1], neg_indices % adj_matrix.shape[1]))

    # Sample positive edges
    pos_samples = pos_edges[np.random.choice(pos_edges.shape[0], num_samples, replace=False)]
    # Sample negative edges
    neg_samples = neg_edges[np.random.choice(neg_edges.shape[0], num_samples, replace=False)]

    return pos_samples, neg_samples


def sample_positive_negative_nodes_sparse_torch(adj_matrix, num_samples, device='cpu'):
    adj_matrix = adj_matrix.coalesce().to(device)

    # Find all positive edges (existing edges)
    pos_edges = adj_matrix.indices().t()
    # Find all negative edges (non-existing edges)
    total_elements = adj_matrix.size(0) * adj_matrix.size(1)
    neg_mask = torch.ones(total_elements, dtype=torch.bool, device=device)
    neg_mask[pos_edges[:, 0] * adj_matrix.size(1) + pos_edges[:, 1]] = False
    neg_indices = torch.nonzero(neg_mask).squeeze()
    neg_edges = torch.stack((neg_indices // adj_matrix.size(1), neg_indices % adj_matrix.size(1)), dim=1)

    # Sample positive edges
    pos_samples = pos_edges[torch.randperm(len(pos_edges))[:num_samples]]
    # Sample negative edges
    neg_samples = neg_edges[torch.randperm(len(neg_edges))[:num_samples]]

    return pos_samples, neg_samples


num_samples = 100000  # Number of samples to take

# Measure time for NumPy (scipy.sparse)
start_time = time.time()
pos_samples_np, neg_samples_np = sample_positive_negative_nodes_sparse_np(adj_matrix_sparse_np, num_samples)
end_time = time.time()
print(f"NumPy (scipy.sparse) sampling time: {end_time - start_time:.6f} seconds")

# Measure time for PyTorch on GPU
if torch.cuda.is_available():
    start_time = time.time()
    pos_samples_torch_gpu, neg_samples_torch_gpu = sample_positive_negative_nodes_sparse_torch(adj_matrix_torch_gpu,
                                                                                               num_samples,
                                                                                               device=device)
    end_time = time.time()
    print(f"PyTorch (GPU) sampling time: {end_time - start_time:.6f} seconds")
else:
    print("GPU not available.")

# Measure time for PyTorch on CPU
start_time = time.time()
pos_samples_torch_cpu, neg_samples_torch_cpu = sample_positive_negative_nodes_sparse_torch(adj_matrix_torch_cpu,
                                                                                           num_samples, device='cpu')
end_time = time.time()
print(f"PyTorch (CPU) sampling time: {end_time - start_time:.6f} seconds")
