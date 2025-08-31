# utils.py
# Reality Mining utilities: robust preprocessing, snapshot graphs, hop rings,
# triplet sampling, and metrics. Includes dataset_mit.

from __future__ import annotations

import os
import io
import tarfile
import logging
import itertools
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F

# ----------------------------
# Column indices helper
# ----------------------------
Cols = SimpleNamespace(
    source=0,
    target=1,
    weight=2,
    time=3,
)

# ----------------------------
# Logging
# ----------------------------
def init_logging_handler(name: str = "reality_mining", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

# ----------------------------
# Loaders
# ----------------------------
def load_data_from_tar(
    tar_path: str,
    member_name: str,
    *,
    starting_line: int = 0,
    sep: Optional[str] = None,
    dtype=float,
    tensor_const=torch.DoubleTensor,
) -> torch.Tensor:
    """
    Load a text member from a tar(.gz|.bz2) into a torch tensor.

    Each row is split by `sep` if provided; otherwise split on whitespace.
    Lines before `starting_line` are skipped.
    """
    with tarfile.open(tar_path, "r:*") as tar:
        member = tar.getmember(member_name)
        f = tar.extractfile(member)
        content = f.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()[starting_line:]
    rows = [
        [dtype(tok) for tok in (line.split(sep) if sep is not None else line.split())]
        for line in lines if line.strip()
    ]
    return tensor_const(rows)

# ----------------------------
# Time
# ----------------------------
def aggregate_by_time(time_vector: torch.Tensor, time_win_aggr: int) -> torch.Tensor:
    """
    Convert absolute times to discrete bins: floor((t - min_t) / window).
    """
    t = time_vector - time_vector.min()
    return (t // int(time_win_aggr)).to(dtype=torch.long)

# ----------------------------
# Sparse converters
# ----------------------------
def spy_sparse2torch_sparse(M: sp.spmatrix) -> torch.Tensor:
    C = M.tocoo()
    indices = torch.tensor([C.row, C.col], dtype=torch.long)
    values = torch.tensor(C.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=C.shape)

def sparse_feeder(M: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Return TF-style (indices, values, shape) for SciPy sparse."""
    C = M.tocoo()
    idx = np.vstack((C.row, C.col)).T.astype(np.int64, copy=False)
    val = C.data.astype(np.float32, copy=False)
    shp = C.shape
    return idx, val, shp

# ----------------------------
# dataset_mit
# ----------------------------
class dataset_mit(torch.utils.data.Dataset):
    """
    Reality Mining dataset wrapper that builds per-time-bin adjacency snapshots.

    __getitem__(i) -> (A_i: CSR, X_i: torch.sparse (I + A_i))
    """
    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        *,
        time_bin: int = 222400,
        undirected: bool = False,
        dedup: bool = True,
    ):
        self.root_dir = root_dir
        self.time_bin = int(time_bin)
        self.undirected = bool(undirected)
        self.dedup = bool(dedup)

        self.Adj_arr: List[csr_matrix] = []
        self.X_Sparse_arr: List[torch.Tensor] = []

        # load edges from tar
        tar_file = os.path.join(self.root_dir, "datasets", "download.tsv.mit.tar.bz2")
        if not os.path.exists(tar_file):
            raise FileNotFoundError(f"Could not find tar file at {tar_file}")
        member = "mit/out.mit"
        data = load_data_from_tar(
            tar_file,
            member,
            starting_line=2,
            sep=None,
            dtype=float,
            tensor_const=torch.DoubleTensor,
        ).long()

        SRC, TGT, W, T = Cols.source, Cols.target, Cols.weight, Cols.time

        # Robust zero-basing
        min_id = int(data[:, [SRC, TGT]].min().item())
        if min_id not in (0, 1):
            raise ValueError(f"Unexpected min node id {min_id}; expected 0 or 1.")
        data[:, [SRC, TGT]] -= min_id
        max_id = int(data[:, [SRC, TGT]].max().item())
        self.N = max_id + 1

        # Optional symmetrization
        if self.undirected:
            data = torch.cat([data, data[:, [TGT, SRC, W, T]]], dim=0)

        # Time binning
        data[:, T] = aggregate_by_time(data[:, T], self.time_bin)

        # De-duplicate within each bin (binary adjacency)
        idx = data[:, [SRC, TGT, T]]
        if self.dedup:
            idx = torch.unique(idx, dim=0)

        df = pd.DataFrame(idx.numpy(), columns=["source", "target", "time"]).astype(int)
        self.T_bins = int(df["time"].max()) + 1 if len(df) > 0 else 0

        # Build snapshots
        running_max = self.N
        for t in range(self.T_bins):
            sub = df.loc[df["time"] == t, ["source", "target"]]
            arr = sub.to_numpy() if len(sub) else np.empty((0, 2), dtype=np.int64)
            A, Xs, running_max = self.get_graph(arr, running_max)
            self.Adj_arr.append(A)
            self.X_Sparse_arr.append(Xs)

    def __len__(self) -> int:
        return len(self.Adj_arr)

    def __getitem__(self, idx: int):
        return self.Adj_arr[idx], self.X_Sparse_arr[idx]

    def get_graph(self, arr: np.ndarray, max_size: int) -> Tuple[csr_matrix, torch.Tensor, int]:
        """
        Build binary adjacency (CSR) and (I + A) torch sparse for a single time bin.
        Keeps shape consistent across bins, growing to the max encountered id.
        """
        if arr.size == 0:
            N = max(1, int(max_size))
            A = csr_matrix((N, N), dtype=np.float32)
        else:
            local_max = int(arr.max()) + 1
            N = max(int(max_size), local_max)
            r = arr[:, 0].astype(np.int64, copy=False)
            c = arr[:, 1].astype(np.int64, copy=False)
            v = np.ones(len(r), dtype=np.float32)
            A = csr_matrix((v, (r, c)), shape=(N, N))
            A.setdiag(0)
            A.eliminate_zeros()

        X = A + sp.eye(A.shape[0], dtype=np.float32, format="csr")
        X_Sparse = spy_sparse2torch_sparse(X)
        return A, X_Sparse, N

# ----------------------------
# Hops & sampling
# ----------------------------
def _csr_neighbors(A: csr_matrix) -> List[np.ndarray]:
    """Row-wise neighbors as numpy arrays."""
    A = A.tocsr()
    indptr, indices = A.indptr, A.indices
    out: List[np.ndarray] = []
    for i in range(A.shape[0]):
        out.append(indices[indptr[i]:indptr[i + 1]])
    return out

def get_hops(A: csr_matrix, K: int) -> Dict[int, List[np.ndarray]]:
    """
    Compute exact-hop rings (distance = 1..K) for every node using BFS layers.

    Returns:
        hops[h]: list of length N; hops[h][i] is np.array of nodes at distance h from i
        hops[-1]: list with one element [union_mat_csr], marking (union≤K ∪ {self}) per row
    """
    N = A.shape[0]
    nbrs = _csr_neighbors(A)

    # prepare containers
    hops: Dict[int, List[np.ndarray]] = {
        h: [np.empty(0, dtype=np.int64) for _ in range(N)] for h in range(1, K + 1)
    }
    union_sets: List[set] = [set() for _ in range(N)]

    # BFS per node to build exact layers
    for i in range(N):
        visited = {i}
        frontier = set(nbrs[i])
        # 1-hop
        exact = frontier - visited
        if exact:
            arr = np.fromiter(exact, dtype=np.int64)
            hops[1][i] = arr
        union = set(exact)
        visited |= exact

        cur_frontier = frontier
        for h in range(2, K + 1):
            next_frontier = set()
            for u in cur_frontier:
                # iterate neighbors(u)
                for v in nbrs[u]:
                    next_frontier.add(v)
            exact_h = next_frontier - visited - {i}
            if exact_h:
                arr_h = np.fromiter(exact_h, dtype=np.int64)
                hops[h][i] = arr_h
            union |= exact_h
            visited |= exact_h
            cur_frontier = next_frontier

        union_sets[i] = union

    # Build union matrix safely via LIL then convert to CSR
    union_mat = sp.lil_matrix((N, N), dtype=np.bool_)
    for i, uset in enumerate(union_sets):
        if uset:
            union_mat[i, list(uset)] = True
        union_mat[i, i] = True  # block self in outside sampler
    union_mat = union_mat.tocsr(copy=False)

    hops[-1] = [union_mat]
    return hops

def sample_last_hop(union_mat: csr_matrix, nodes: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    For each anchor in `nodes`, sample a node NOT in the union-of-<=K ring (and not itself).
    """
    rng = rng or np.random.default_rng()
    union_mat = union_mat.tocsr()
    N = union_mat.shape[0]
    sampled = rng.integers(0, N, size=len(nodes))
    invalid = np.array(union_mat[nodes][:, sampled].diagonal(), dtype=bool)
    while invalid.any():
        sampled[invalid] = rng.integers(0, N, size=int(invalid.sum()))
        invalid = np.array(union_mat[nodes][:, sampled].diagonal(), dtype=bool)
    return sampled

def sample_hops(A: csr_matrix, K: int, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    For each node, sample one node from each exact-hop ring (1..K),
    and one node from 'outside' (not in union ≤K).

    Returns:
        sampled_hops: [N, 1 + K + 1] with columns [anchor, hop1, ..., hopK, outside]
        scale_terms:  dict {1..K, K+1} -> (N,) ring sizes per anchor (outside = size of complement)
    """
    rng = rng or np.random.default_rng()
    hops = get_hops(A, K)
    union_mat = hops[-1][0]  # CSR
    N = A.shape[0]

    sampled = np.full((N, K + 1), -1, dtype=np.int64)  # K hops + outside (to append later)
    scale_terms: Dict[int, np.ndarray] = {}

    # exact-hop sampling and sizes
    for h in range(1, K + 1):
        sizes = np.fromiter((len(hops[h][i]) for i in range(N)), dtype=np.int64, count=N)
        scale_terms[h] = sizes.copy()
        for i in range(N):
            if sizes[i] > 0:
                sampled[i, h - 1] = rng.choice(hops[h][i])

    # outside sampling and sizes
    anchors = np.arange(N, dtype=np.int64)
    outside = sample_last_hop(union_mat, anchors, rng)
    outside_sizes = (N - union_mat.getnnz(axis=1)).astype(np.int64)
    sampled[:, K] = outside
    scale_terms[K + 1] = outside_sizes

    sampled_hops = np.column_stack([anchors, sampled])
    return sampled_hops, scale_terms

# ----------------------------
# Triplets
# ----------------------------
def to_triplets(sampled_hops: np.ndarray, scale_terms: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (anchor, pos, neg) triplets from sampled hops.
    i<j means ring i (closer) is positive and ring j (farther/outside) is negative.
    """
    N, W = sampled_hops.shape
    K = W - 2
    triplets = []
    scales = []
    for i, j in itertools.combinations(range(1, K + 2), 2):
        T = sampled_hops[:, [0, i, j]]
        mask = (T[:, 1] != -1) & (T[:, 2] != -1)
        T = T[mask]
        if len(T) == 0:
            continue
        mask2 = (T[:, 0] != T[:, 1]) & (T[:, 0] != T[:, 2])
        T = T[mask2]
        if len(T) == 0:
            continue
        triplets.append(T)
        anchors = T[:, 0]
        s = scale_terms[i][anchors] * scale_terms[j][anchors]
        scales.append(s.astype(np.float32))
    if len(triplets) == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.float32)
    return np.row_stack(triplets).astype(np.int64), np.concatenate(scales).astype(np.float32)

# ----------------------------
# Scoring & metrics
# ----------------------------
def cosine_score(emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    u = F.normalize(emb[pairs[:, 0]], dim=1)
    v = F.normalize(emb[pairs[:, 1]], dim=1)
    return (u * v).sum(dim=1)

def l2_similarity_score(emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    u = emb[pairs[:, 0]]
    v = emb[pairs[:, 1]]
    dist2 = ((u - v) ** 2).sum(dim=1)
    return -dist2

def _to_probs(scores: np.ndarray, assume_logits: bool) -> np.ndarray:
    if assume_logits:
        return 1.0 / (1.0 + np.exp(-scores))
    return scores

def get_MAP_e(y_true: np.ndarray, scores: np.ndarray, assume_logits: bool = True) -> float:
    from sklearn.metrics import average_precision_score
    p = _to_probs(np.asarray(scores, dtype=np.float64), assume_logits)
    y = np.asarray(y_true, dtype=np.int32)
    return float(average_precision_score(y, p))

def get_MRR(y_true: np.ndarray, scores: np.ndarray, assume_logits: bool = True) -> float:
    p = _to_probs(np.asarray(scores, dtype=np.float64), assume_logits)
    y = np.asarray(y_true, dtype=np.int32)
    order = np.argsort(-p)
    y_sorted = y[order]
    pos = np.where(y_sorted == 1)[0]
    return 0.0 if len(pos) == 0 else 1.0 / float(pos[0] + 1)

# ----------------------------
# Quick sanity report (optional)
# ----------------------------
def sanity_report(df_edges: pd.DataFrame, Adj_arr: List[csr_matrix], N: int, undirected: bool) -> None:
    print("=== Preprocessing sanity report ===")
    print(f"Nodes (N): {N}")
    if len(df_edges) == 0:
        print("No edges after processing.")
        return

    pairs_all = df_edges[["source", "target"]].drop_duplicates()
    E_unique = len(pairs_all)
    possible_pairs = N * (N - 1)  # no self-loops
    print(f"Unique directed pairs across all time: {E_unique} / {possible_pairs} "
          f"({E_unique/possible_pairs:.4f} density)")

    per_bin = df_edges.groupby("time").size()
    print("Top 5 bins by edge count:")
    print(per_bin.sort_values(ascending=False).head(5).to_string())

    for t, A in enumerate(Adj_arr[:3]):
        nnz = int(A.nnz)
        if undirected:
            sym_mismatch = (A != A.T).nnz
            if sym_mismatch != 0:
                print(f"[WARN] bin {t}: undirected=True but A != A.T (mismatch nnz={sym_mismatch})")
        print(f"  bin {t}: edges={nnz}")

