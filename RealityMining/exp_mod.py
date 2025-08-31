# transformerg2g_eval_fast_compat.py
# Keeps the original paper-style structure, but:
#  - uses diagonal-free negative sampling
#  - aligns adj size with embedding size
#  - vectorizes pair feature building
#  - preserves original metric function signatures and logging format

from scipy.sparse import csr_matrix
import tarfile

import torch
import numpy as np
import scipy.sparse as sp
from scipy import sparse
from scipy.sparse import coo_matrix

import random
import time
import warnings
import argparse
import yaml
import os
import logging
import pickle
import pandas as pd
import json

from sklearn.metrics import average_precision_score

# original imports (keep as-is for your env)
from utils import *
from models import *

warnings.filterwarnings('ignore')

# ----------------------- Config & Logging -----------------------

# Read parameters from json file
with open("config.json") as f:
    config = json.load(f)

K                 = config["K"]
p_val             = config["p_val"]
p_nodes           = config["p_nodes"]
n_hidden          = config["n_hidden"]
max_iter          = config["max_iter"]
tolerance_init    = config["tolerance"]
time_list         = config["time_list"]
L_list            = config["L_list"]
save_time_complex = config["save_time_complex"]
save_MRR_MAP      = config["save_MRR_MAP"]
save_sigma_mu     = config["save_sigma_mu"]
scale             = config["scale"]
seed              = config["seed"]
verbose           = config["verbose"]
lookback          = config["lookback"]

# Evaluation toggles (keep defaults to mirror the paper; flip if you want)
DIAG_FREE_NEGATIVES = True     # exclude (i,i) from negatives
CONSISTENT_NEG_RATIO = False   # if True uses same mult for train/test; if False keep 10(train)/50(test)

def init_logging_handler(exp_name):
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(exp_name, current_time))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

name = 'Results/RealityMining'
init_logging_handler(name)
logging.debug(str(config))

def check_if_gpu():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = check_if_gpu()
logging.debug('The code will be running on {}'.format(device))

# Reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ----------------------- Helpers -----------------------

class Namespace(object):
    """ Reference dict entries as attributes: d.k instead of d['k'] """
    def __init__(self, adict):
        self.__dict__.update(adict)

def aggregate_by_time(time_vector, time_win_aggr):
    time_vector = time_vector - time_vector.min()
    time_vector = time_vector // time_win_aggr
    return time_vector

def load_data_from_tar(file, tar_archive, replace_unknow=False, starting_line=1,
                       sep=',', type_fn=float, tensor_const=torch.DoubleTensor):
    f = tar_archive.extractfile(file)
    lines = f.read().decode('utf-8')
    if replace_unknow:
        lines = lines.replace('unknow', '-1').replace('-1n', '-1')
    lines = lines.splitlines()
    data = [[type_fn(r) for r in row.split()] for row in lines[starting_line:]]
    data = tensor_const(data)
    return data

def is_compatible(filename):
    return any(filename.endswith(extension) for extension in ['.txt'])

class dataset_mit(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.Adj_arr = []
        self.X_Sparse_arr = []
        count = 0
        max_size = 0
        tar_file = os.path.join(self.root_dir, 'datasets', 'download.tsv.mit.tar.bz2')
        tar_archive = tarfile.open(tar_file, 'r:bz2')

        data = load_data_from_tar('mit/out.mit', tar_archive, starting_line=2, sep=' ')
        cols = Namespace({'source': 0, 'target': 1, 'weight': 2, 'time': 3})

        data = data.long()
        num_nodes = int(data[:, [cols.source, cols.target]].max())
        data[:, [cols.source, cols.target]] -= 1  # make 0-based contiguous ids

        # aggregate time
        data[:, cols.time] = aggregate_by_time(data[:, cols.time], 222400)

        idx = data[:, [cols.source, cols.target, cols.time]]
        df = pd.DataFrame(idx.numpy(), columns=['source', 'target', 'time'])

        for i in range(df['time'].max() + 1):
            print(count)
            df1 = df[df['time'] == i]
            arr = df1[['source', 'target']].to_numpy()
            A, X_Sparse, size = self.get_graph(arr, max_size)
            if size > max_size:
                max_size = size
            self.Adj_arr.append(A)
            self.X_Sparse_arr.append(X_Sparse)
            count += 1

    def __len__(self):
        # fix: original had an undefined attribute
        return len(self.Adj_arr)

    def __getitem__(self, idx):
        return self.Adj_arr[idx], self.X_Sparse_arr[idx]

    def get_graph(self, arr, max_size):
        if arr.size == 0:
            # handle empty slice; build a minimal 1x1 zero graph
            new_max = max_size if max_size > 0 else 1
            arr_zero = np.zeros((new_max, new_max))
        else:
            if max_size > arr.max() + 1:
                new_max = max_size
                arr_zero = np.zeros((max_size, max_size))
            else:
                new_max = arr.max() + 1
                arr_zero = np.zeros((new_max, new_max))

            for i, j in arr:
                arr_zero[int(i)][int(j)] = 1

        # no self loops in adjacency
        np.fill_diagonal(arr_zero, 0)
        A = csr_matrix(arr_zero)

        X = A + sp.eye(A.shape[0])
        X_Sparse = sparse_feeder(X)
        X_Sparse = spy_sparse2torch_sparse(X_Sparse)
        return A, X_Sparse, new_max

# Build dataset (same as original)
data = dataset_mit('..')

# ----------------------- Negative Samplers -----------------------

def sample_zero_forever(mat):
    """Original sampler: includes diagonal pairs."""
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    N = mat.shape[0]
    while True:
        t = tuple(np.random.randint(0, N, 2))
        if t not in nonzero_or_sampled:
            yield t
            nonzero_or_sampled.add(t)

def sample_zero_n(mat, n=2000):
    itr = sample_zero_forever(mat)
    return [next(itr) for _ in range(n)]

def sample_zero_no_diag_n(mat, n=2000, rng=None):
    """Diagonal-free sampler: excludes (i,i) and existing edges, unique sampling."""
    if rng is None:
        rng = np.random.default_rng(seed)
    N = mat.shape[0]
    E = set(zip(*mat.nonzero()))
    seen = set()
    out = []
    while len(out) < n:
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N - 1))
        if j >= i:
            j += 1  # ensure j != i
        t = (i, j)
        if t in E or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

# ----------------------- Metrics (paper-style signatures) -----------------------

def get_row_MRR(probs, true_classes):
    existing_mask = true_classes == 1
    ordered_indices = np.flip(probs.argsort())  # descending by prob
    ordered_existing_mask = existing_mask[ordered_indices]
    existing_ranks = np.arange(1, true_classes.shape[0] + 1, dtype=np.float64)[ordered_existing_mask]
    MRR = (1.0 / existing_ranks).sum() / max(1, existing_ranks.shape[0])
    return MRR

def get_MRR(predictions, true_classes, adj):
    probs = torch.sigmoid(predictions).detach().cpu().numpy()
    # ensure numpy arrays
    true_classes = np.asarray(true_classes.detach().cpu().numpy())
    adj = adj  # shape (2, E)

    pred_matrix = coo_matrix((probs, (adj[0], adj[1]))).toarray()
    true_matrix = coo_matrix((true_classes, (adj[0], adj[1]))).toarray()

    row_MRRs = []
    for i, pred_row in enumerate(pred_matrix):
        if np.any(true_matrix[i] == 1):
            row_MRRs.append(get_row_MRR(pred_row, true_matrix[i]))

    if len(row_MRRs) == 0:
        return torch.tensor(0.0)
    return torch.tensor(row_MRRs).mean()

def get_MAP_e(predictions, true_classes, adj):
    probs = torch.sigmoid(predictions).detach().cpu().numpy()
    true_classes = np.asarray(true_classes.detach().cpu().numpy())
    return average_precision_score(true_classes, probs)

# ----------------------- Load learned embeddings -----------------------

name_loaded = 'Results/RealityMining'
with open(os.path.join(name_loaded, 'Eval_Results', 'saved_array', 'mu_as'), 'rb') as f:
    mu_arr = pickle.load(f)
with open(os.path.join(name_loaded, 'Eval_Results', 'saved_array', 'sigma_as'), 'rb') as f:
    sigma_arr = pickle.load(f)

# ----------------------- Training + Evaluation -----------------------
#
# MAP_l = []
# MRR_l = []
#
# for l_num in range(len(L_list)):
#     mu_64 = mu_arr[l_num]
#     sigma_64 = sigma_arr[l_num]
#
#     # Simple MLP classifier as in the paper
#     class Classifier(torch.nn.Module):
#         def __init__(self):
#             super(Classifier, self).__init__()
#             activation = torch.nn.ReLU()
#             d = int(np.array(mu_64[0]).shape[1])
#             self.mlp = torch.nn.Sequential(
#                 torch.nn.Linear(in_features=2 * d, out_features=d),
#                 activation,
#                 torch.nn.Linear(in_features=d, out_features=1),
#             )
#         def forward(self, x):
#             return self.mlp(x)
#
#     classify = Classifier().to(device)
#     loss = torch.nn.BCEWithLogitsLoss(reduction='none')
#     optim = torch.optim.Adam(classify.parameters(), lr=1e-3)
#
#     # Neg ratios (paper vs consistent)
#     if CONSISTENT_NEG_RATIO:
#         mult = 20
#         mult_test = 20
#     else:
#         mult = 10
#         mult_test = 50
#
#     # -------------- Train evaluator on earlier bins --------------
#     num_epochs = 50
#     torch.cuda.empty_cache()
#     for epoch in range(num_epochs):
#         count = 0
#         A_prev_node = None
#         for ctr in range(lookback + 1, 63):
#             A = data[ctr][0]
#             A_node = A.shape[0]
#
#             # align adjacency to embedding size
#             N_emb = int(np.array(mu_64[0]).shape[0])
#             if A_node > N_emb:
#                 A = A[:N_emb, :N_emb]
#                 A_node = N_emb
#
#             if count > 0 and A_prev_node is not None and A_node > A_prev_node:
#                 A = A[:A_prev_node, :A_prev_node]
#                 A_node = A_prev_node
#
#             if ctr < 63 and ctr > 0:
#                 logging.debug('Training')
#                 logging.debug(ctr)
#
#                 ones_edj = A.nnz
#                 # max possible off-diagonal pairs = N*(N-1)
#                 max_offdiag = A.shape[0] * (A.shape[0] - 1)
#                 zeroes_edj = min(A.shape[0] * mult, max_offdiag - ones_edj)
#
#                 tot = ones_edj + zeroes_edj
#                 if tot == 0:
#                     A_prev_node = A_node
#                     count += 1
#                     continue
#
#                 # positives
#                 val_ones = list(set(zip(*A.nonzero())))
#                 if ones_edj > 0:
#                     val_ones = random.sample(val_ones, ones_edj)
#                 val_ones = [list(ele) for ele in val_ones]
#
#                 # negatives
#                 if DIAG_FREE_NEGATIVES:
#                     val_zeros = sample_zero_no_diag_n(A, zeroes_edj)
#                 else:
#                     val_zeros = sample_zero_n(A, zeroes_edj)
#                 val_zeros = [list(ele) for ele in val_zeros]
#
#                 val_edges = np.row_stack((val_ones, val_zeros))
#                 val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1
#
#                 # shuffle deterministically per count
#                 np.random.seed(count)
#                 idx_perm = np.random.permutation(len(val_edges))
#                 a = val_edges[idx_perm]
#                 b = val_ground_truth[idx_perm]
#
#                 # embeddings at time ctr-(lookback+1) (paper’s choice)
#                 mu_np = np.array(mu_64[ctr - (lookback + 1)], dtype=np.float32)
#
#                 # vectorized pair features
#                 left = mu_np[a[:, 0].astype(int)]
#                 right = mu_np[a[:, 1].astype(int)]
#                 inp_clf = np.concatenate([left, right], axis=1)
#
#                 # tensors
#                 X = torch.from_numpy(inp_clf).to(device)
#                 y = torch.from_numpy(b.astype(np.float32)).to(device)
#
#                 classify.train()
#                 out = classify(X).squeeze()
#
#                 # class weighting (same as paper spirit)
#                 weight = torch.tensor([0.1, 0.9], device=device)
#                 weight_ = weight[y.long()]
#                 l = (loss(out, y) * weight_).mean()
#
#                 optim.zero_grad()
#                 l.backward()
#                 optim.step()
#
#                 # paper-style metrics: predictions first, then labels, then adj
#                 MRR_val = get_MRR(out.detach().cpu(), y.detach().cpu(), np.transpose(a))
#                 MAP_val = get_MAP_e(out.detach().cpu(), y.detach().cpu(), None)
#
#                 logging.debug('L:{}, Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}'.format(
#                     int(mu_np.shape[1]), epoch, ctr, l.item(), MAP_val, MRR_val.item()))
#
#             A_prev_node = A_node
#             count += 1
#
#     # -------------- Evaluate on later bins --------------
#     num_epochs = 1
#     MAP_time = []
#     MRR_time = []
#     time_ctr = 0
#
#     for epoch in range(num_epochs):
#         get_MAP_avg = []
#         get_MRR_avg = []
#         count = 0
#         A_prev_node = None
#
#         for ctr in range(72, 90):
#             A = data[ctr][0]
#             A_node = A.shape[0]
#
#             # align adjacency to embedding size
#             N_emb = int(np.array(mu_64[0]).shape[0])
#             if A_node > N_emb:
#                 A = A[:N_emb, :N_emb]
#                 A_node = N_emb
#
#             if count > 0 and A_prev_node is not None and A_node > A_prev_node:
#                 A = A[:A_prev_node, :A_prev_node]
#                 A_node = A_prev_node
#
#             if ctr >= 72:
#                 logging.debug('Testing')
#                 logging.debug(ctr)
#
#                 ones_edj = A.nnz
#                 max_offdiag = A.shape[0] * (A.shape[0] - 1)
#                 zeroes_edj = min(A.shape[0] * mult_test, max_offdiag - ones_edj)
#                 tot = ones_edj + zeroes_edj
#                 if tot == 0:
#                     A_prev_node = A_node
#                     count += 1
#                     continue
#
#                 # positives
#                 val_ones = list(set(zip(*A.nonzero())))
#                 if ones_edj > 0:
#                     val_ones = random.sample(val_ones, ones_edj)
#                 val_ones = [list(ele) for ele in val_ones]
#
#                 # negatives
#                 if DIAG_FREE_NEGATIVES:
#                     val_zeros = sample_zero_no_diag_n(A, zeroes_edj)
#                 else:
#                     val_zeros = sample_zero_n(A, zeroes_edj)
#                 val_zeros = [list(ele) for ele in val_zeros]
#
#                 val_edges = np.row_stack((val_ones, val_zeros))
#                 val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1
#
#                 # shuffle deterministically per count
#                 np.random.seed(count)
#                 idx_perm = np.random.permutation(len(val_edges))
#                 a = val_edges[idx_perm]
#                 b = val_ground_truth[idx_perm]
#
#                 # embeddings at time ctr-(lookback+1)
#                 mu_np = np.array(mu_64[ctr - (lookback + 1)], dtype=np.float32)
#
#                 left = mu_np[a[:, 0].astype(int)]
#                 right = mu_np[a[:, 1].astype(int)]
#                 inp_clf = np.concatenate([left, right], axis=1)
#
#                 X = torch.from_numpy(inp_clf).to(device)
#                 y = torch.from_numpy(b.astype(np.float32)).to(device)
#
#                 classify.eval()
#                 with torch.no_grad():
#                     out = classify(X).squeeze()
#
#                 # evaluate
#                 l = (loss(out, y) * torch.tensor([0.1, 0.9], device=device)[y.long()]).mean()
#
#                 MAP_val = get_MAP_e(out.detach().cpu(), y.detach().cpu(), None)
#                 MRR_val = get_MRR(out.detach().cpu(), y.detach().cpu(), np.transpose(a))
#
#                 get_MAP_avg.append(MAP_val)
#                 get_MRR_avg.append(MRR_val.item())
#
#                 # capture at specific time points if provided
#                 try:
#                     if ctr == time_list[time_ctr]:
#                         MAP_time.append(MAP_val)
#                         MRR_time.append(MRR_val.item())
#                         time_ctr += 1
#                 except Exception:
#                     pass
#
#                 logging.debug('Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}, Running Mean MAP: {}, Running Mean MRR: {}'.format(
#                     epoch, ctr, l.item(), MAP_val, MRR_val.item(),
#                     float(np.mean(get_MAP_avg)) if len(get_MAP_avg) else 0.0,
#                     float(np.mean(get_MRR_avg)) if len(get_MRR_avg) else 0.0
#                 ))
#
#             A_prev_node = A_node
#             count += 1
#
#     MAP_l.append(MAP_time)
#     MRR_l.append(MRR_time)
#
#     logging.debug('Saving model')
#     torch.save(classify.state_dict(), os.path.join(name, 'classifier_{}.pth'.format(int(np.array(mu_64[0]).shape[1]))))

# ----------------------- Persist results -----------------------

# outdir = os.path.join(name, 'saved_array')
# os.makedirs(outdir, exist_ok=True)
# with open(os.path.join(outdir, 'MRR'), 'wb') as f:
#     pickle.dump(MRR_l, f)
# with open(os.path.join(outdir, 'MAP'), 'wb') as f:
#     pickle.dump(MAP_l, f)

def get_MAP_avg(mu_arr, sigma_arr, lookback, data, *, seed=5, device=None):
    """
    Faithful to the original transformerg2g evaluation:
      - metrics: get_MAP_e(predictions, true_labels, adj), get_MRR(predictions, true_labels, adj)
      - time indexing: embeddings at t - (lookback + 1) for both train and test
      - negatives: includes diagonal (i,i) via sample_zero_n (paper behavior)
      - train/test ratios: mult=10 (train), mult_test=50 (test)
    Returns:
      mean_MAP_over_L, mean_MRR_over_L
    """
    import numpy as np
    import torch
    import random

    # expect these to be in scope exactly as in the original script
    #   - sample_zero_n(mat, n)
    #   - get_MAP_e(predictions, true_classes, adj)
    #   - get_MRR(predictions, true_classes, adj)

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # reproducibility (mirror original spirit)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def unison_shuffled_copies(a, b, seed_):
        assert len(a) == len(b)
        np.random.seed(seed_)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    class Classifier(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * dim, dim),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, 1)
            )
        def forward(self, x):
            return self.mlp(x)

    # paper ratios (train/test)
    mult = 10
    mult_test = 50

    MAP_means = []
    MRR_means = []

    for l_num in range(len(mu_arr)):
        mu_64 = mu_arr[l_num]
        sigma_64 = sigma_arr[l_num]  # not used here, kept for parity with original

        dim = int(np.array(mu_64[0]).shape[1])
        classify = Classifier(dim).to(device)

        loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        optim = torch.optim.Adam(classify.parameters(), lr=1e-3)

        # ---------- TRAIN ----------
        num_epochs = 50
        for epoch in range(num_epochs):
            count = 0
            for ctr in range(lookback + 1, 63):
                A = data[ctr][0]
                A_node = A.shape[0]

                if count > 0:
                    # keep size non-increasing across time as in original
                    if A_node > A_prev_node:
                        A = A[:A_prev_node, :A_prev_node]
                        A_node = A_prev_node

                if ctr < 63 and ctr > 0:
                    ones_edj = A.nnz
                    # original's max capacity formula (kept as-is)
                    if A.shape[0] * mult <= (A.shape[0] - 1) * (A.shape[0] - 1):
                        zeroes_edj = A.shape[0] * mult
                    else:
                        zeroes_edj = (A.shape[0] - 1) * (A.shape[0] - 1) - A.nnz

                    tot = ones_edj + zeroes_edj
                    if tot == 0:
                        A_prev_node = A_node
                        count += 1
                        continue

                    # positives (randomized order; effectively a shuffle)
                    val_ones = list(set(zip(*A.nonzero())))
                    if ones_edj > 0:
                        val_ones = random.sample(val_ones, ones_edj)
                    val_ones = [list(ele) for ele in val_ones]

                    # negatives (includes diagonal, paper behavior)
                    val_zeros = sample_zero_n(A, zeroes_edj)
                    val_zeros = [list(ele) for ele in val_zeros]

                    val_edges = np.row_stack((val_ones, val_zeros))
                    val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                    a, b = unison_shuffled_copies(val_edges, val_ground_truth, count)

                    # embeddings from ctr - (lookback + 1) (paper choice)
                    mu_np = np.array(mu_64[ctr - (lookback + 1)], dtype=np.float32)

                    # vectorized pair features (same math as original loop)
                    left = mu_np[a[:, 0].astype(int)]
                    right = mu_np[a[:, 1].astype(int)]
                    inp_clf = np.concatenate([left, right], axis=1)

                    X = torch.from_numpy(inp_clf).to(device)
                    y = torch.from_numpy(b.astype(np.float32)).to(device)

                    classify.train()
                    out = classify(X).squeeze()

                    # class weights (same spirit as original)
                    weight = torch.tensor([0.1, 0.9], device=device)
                    weight_ = weight[y.long()]
                    l = (loss(out, y) * weight_).mean()

                    optim.zero_grad()
                    l.backward()
                    optim.step()

                    # optional: compute metrics during train (paper logs them)
                    # MRR_train = get_MRR(out.detach().cpu(), y.detach().cpu(), np.transpose(a))
                    # MAP_train = get_MAP_e(out.detach().cpu(), y.detach().cpu(), None)

                A_prev_node = A_node
                count += 1

        # ---------- TEST ----------
        get_MAP_vals = []
        get_MRR_vals = []
        count = 0
        for ctr in range(72, 90):
            A = data[ctr][0]
            A_node = A.shape[0]

            if count > 0:
                if A_node > A_prev_node:
                    A = A[:A_prev_node, :A_prev_node]
                    A_node = A_prev_node

            if ctr >= 72:
                ones_edj = A.nnz
                if A.shape[0] * mult_test <= (A.shape[0] - 1) * (A.shape[0] - 1):
                    zeroes_edj = A.shape[0] * mult_test
                else:
                    zeroes_edj = (A.shape[0] - 1) * (A.shape[0] - 1) - A.nnz

                tot = ones_edj + zeroes_edj
                if tot == 0:
                    A_prev_node = A_node
                    count += 1
                    continue

                val_ones = list(set(zip(*A.nonzero())))
                if ones_edj > 0:
                    val_ones = random.sample(val_ones, ones_edj)
                val_ones = [list(ele) for ele in val_ones]

                val_zeros = sample_zero_n(A, zeroes_edj)
                val_zeros = [list(ele) for ele in val_zeros]

                val_edges = np.row_stack((val_ones, val_zeros))
                val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                a, b = unison_shuffled_copies(val_edges, val_ground_truth, count)

                mu_np = np.array(mu_64[ctr - (lookback + 1)], dtype=np.float32)

                left = mu_np[a[:, 0].astype(int)]
                right = mu_np[a[:, 1].astype(int)]
                inp_clf = np.concatenate([left, right], axis=1)

                X = torch.from_numpy(inp_clf).to(device)
                y = torch.from_numpy(b.astype(np.float32)).to(device)

                classify.eval()
                with torch.no_grad():
                    out = classify(X).squeeze()

                # paper-style metrics: predictions first
                MAP_val = get_MAP_e(out.detach().cpu(), y.detach().cpu(), None)
                MRR_val = get_MRR(out.detach().cpu(), y.detach().cpu(), np.transpose(a))

                get_MAP_vals.append(float(MAP_val))
                get_MRR_vals.append(float(MRR_val))

            A_prev_node = A_node
            count += 1

        # per-L means
        MAP_means.append(np.mean(get_MAP_vals) if len(get_MAP_vals) else 0.0)
        MRR_means.append(np.mean(get_MRR_vals) if len(get_MRR_vals) else 0.0)

    # across-L means
    return float(np.mean(MAP_means)), float(np.mean(MRR_means))
