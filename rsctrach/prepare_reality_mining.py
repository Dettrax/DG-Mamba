#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reality Mining (KONECT) loader + preprocessor for TransformerG2G-style pipelines.

Outputs
-------
out_dir/
  snapshots_csr.npz      # list of scipy.sparse.csr_matrix, length T
  node_list.npy          # np.array of node ids (0..n-1)
  time_bins.npy          # np.array of bin edges (T+1,)
  splits.json            # {"train": [...], "val": [...], "test": [...]}
  tea_stats.json         # novelty, repeated/new edges counts per t
  tea_plot.png           # optional TEA bar chart
  triplets_t{t}.npy      # optional, per-t array of shape (M_t, 3) with (ref, near, far)

Notes
-----
- Reads KONECT `out.mit` lines with 2–4 whitespace columns:
  u v [weight/multiplicity] [unix_timestamp]
- Converts to undirected, unweighted edges per snapshot (A_t ∈ {0,1}).
- Bins timestamps into T equal-width bins (default T=90).
- Removes self-loops; deduplicates multi-edges within a snapshot.
"""

import argparse
import os
import tarfile
import io
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import csgraph


# ----------------------------
# Utilities
# ----------------------------

def _extract_if_archive(input_path: Path, out_dir: Path) -> Path:
    """
    If input is a .tar.bz2, extract to out_dir and return extracted folder; otherwise return input_path.
    """
    input_path = input_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if input_path.suffixes[-2:] == ['.tar', '.bz2'] or input_path.suffix == '.bz2':
        with tarfile.open(input_path, mode='r:bz2') as tf:
            tf.extractall(path=out_dir)
        # Heuristics: KONECT bundles a top-level folder (e.g., 'mit')
        # Find it:
        subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
        return out_dir
    return input_path


def _find_out_mit(root: Path) -> Path:
    candidates = []
    for p in root.rglob("out.mit"):
        candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No out.mit found under {root}")
    # Prefer the shortest path (likely ./mit/out.mit)
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def _read_out_mit(out_mit: Path) -> pd.DataFrame:
    """
    Robust reader for KONECT out.mit (whitespace separated).
    Returns columns: u, v, w (float or 1.0), t (int or None).
    Node ids will be returned as ints as they appear (often 1-based).
    """
    rows = []
    with out_mit.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('%') or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            u = int(parts[0]); v = int(parts[1])
            w = 1.0
            t = None
            if len(parts) == 3:
                # Could be weight or timestamp; in Reality Mining it's likely weight if time present elsewhere.
                # Heuristic: timestamps here should be >= 10**8 (seconds since 1970), weight is usually small.
                x = float(parts[2])
                if x >= 1e8:
                    t = int(x); w = 1.0
                else:
                    w = x
            elif len(parts) >= 4:
                w = float(parts[2])
                try:
                    t = int(float(parts[3]))
                except:
                    t = None
            rows.append((u, v, w, t))
    df = pd.DataFrame(rows, columns=["u", "v", "w", "t"])
    return df


def _map_nodes(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Map original node ids to 0..n-1, return updated df and node_list.
    """
    nodes = np.unique(np.concatenate([df["u"].values, df["v"].values]))
    nodes_sorted = np.sort(nodes)
    mapping = {orig: i for i, orig in enumerate(nodes_sorted)}
    df2 = df.copy()
    df2["u"] = df2["u"].map(mapping)
    df2["v"] = df2["v"].map(mapping)
    return df2, nodes_sorted


def _bin_timestamps(ts: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build equal-width bins from min..max and assign each timestamp to [0..T-1].
    Returns (bin_ids, bin_edges)
    """
    tmin = int(np.nanmin(ts))
    tmax = int(np.nanmax(ts))
    if tmin == tmax:
        # Edge case: all same timestamp -> put all in last bin
        edges = np.linspace(tmin - 1, tmax + 1, T + 1)
        bins = np.full_like(ts, T - 1, dtype=int)
        return bins, edges
    edges = np.linspace(tmin, tmax, T + 1)
    # np.digitize returns 1..T for right=False, subtract 1 to get 0..T-1
    bins = np.digitize(ts, edges[1:-1], right=False)
    return bins.astype(int), edges


def _edges_to_snapshots(
    df: pd.DataFrame,
    T: int,
    undirected: bool = True,
    remove_self_loops: bool = True,
) -> Tuple[List[csr_matrix], np.ndarray]:
    """
    Convert (u,v,w,t) rows to list of CSR adjacency matrices A_t (binary, unweighted).
    """
    if df["t"].isna().all():
        raise ValueError("No timestamps found. Reality Mining should have per-edge Unix timestamps.")

    bins, edges = _bin_timestamps(df["t"].values.astype(int), T)
    df = df.assign(bin=bins)

    n = int(np.max(np.r_[df["u"].values, df["v"].values])) + 1
    snapshots: List[csr_matrix] = []

    for t in range(T):
        sub = df[df["bin"] == t]
        if len(sub) == 0:
            snapshots.append(csr_matrix((n, n), dtype=np.uint8))
            continue

        u = sub["u"].values
        v = sub["v"].values

        if remove_self_loops:
            mask = u != v
            u, v = u[mask], v[mask]

        # binary edges
        data = np.ones_like(u, dtype=np.uint8)
        A = coo_matrix((data, (u, v)), shape=(n, n), dtype=np.uint8).tocsr()
        A.sum_duplicates()
        A.data[:] = 1

        if undirected:
            A = A.maximum(A.T)

        # zero diagonal
        A.setdiag(0)
        A.eliminate_zeros()

        snapshots.append(A)

    return snapshots, edges


def _compute_tea(snapshots: List[csr_matrix]) -> dict:
    """
    Compute TEA counts (new vs repeated edges per t) + novelty index.
    """
    seen = set()
    new_counts = []
    rep_counts = []

    for t, A in enumerate(snapshots):
        A_coo = A.tocoo()
        edges = set(zip(A_coo.row.tolist(), A_coo.col.tolist()))
        # undirected: ensure (i<j) unique pairs
        edges = set((min(a, b), max(a, b)) for a, b in edges if a != b)

        new = sum((e not in seen) for e in edges)
        rep = len(edges) - new

        new_counts.append(int(new))
        rep_counts.append(int(rep))

        seen |= edges

    novelty = float(np.mean([nc / (nc + rc) if (nc + rc) > 0 else 0.0
                             for nc, rc in zip(new_counts, rep_counts)]))
    return {
        "new_per_t": new_counts,
        "repeated_per_t": rep_counts,
        "novelty": novelty,
    }


def _plot_tea(tea: dict, out_path: Path):
    new = np.array(tea["new_per_t"])
    rep = np.array(tea["repeated_per_t"])
    T = len(new)
    x = np.arange(T)
    plt.figure(figsize=(12, 4))
    plt.bar(x, rep, label="Repeated edges")
    plt.bar(x, new, bottom=rep, label="New edges")
    plt.xlabel("Time")
    plt.ylabel("Edges")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _split_time(T: int, train=0.7, val=0.1, test=0.2):
    assert abs((train + val + test) - 1.0) < 1e-6
    t_train = int(np.floor(T * train))
    t_val = int(np.floor(T * (train + val)))
    idx = {
        "train": list(range(0, t_train)),
        "val": list(range(t_train, t_val)),
        "test": list(range(t_val, T)),
    }
    return idx


def _triplets_for_snapshot(
    A: csr_matrix,
    k_small: int = 2,
    k_large: int = 3,
    per_node: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample (ref, near, far) triplets using shortest-path k-hop neighborhoods.
    - ref node is any node present at this time (degree>0).
    - near from dist in [1..k_small], far from dist >= k_large (finite).
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    present = np.where(deg > 0)[0]
    if len(present) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Shortest-path distances (unweighted)
    D = csgraph.shortest_path(A, directed=False, unweighted=True, return_predecessors=False)
    triplets = []

    for ref in present:
        d = D[ref]
        near_cand = np.where((d >= 1) & (d <= k_small) & np.isfinite(d))[0]
        far_cand  = np.where((d >= k_large) & np.isfinite(d))[0]
        if len(near_cand) == 0 or len(far_cand) == 0:
            continue
        take = min(per_node, len(near_cand))
        picks_near = rng.choice(near_cand, size=take, replace=len(near_cand) < take)
        picks_far  = rng.choice(far_cand,  size=take, replace=len(far_cand)  < take)
        for a, b in zip(picks_near, picks_far):
            triplets.append((int(ref), int(a), int(b)))

    if not triplets:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(triplets, dtype=np.int32)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="Path to download.tsv.mit.tar.bz2 or extracted folder containing mit/out.mit")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--snapshots", type=int, default=90,
                    help="Number of discrete time bins (T). Paper used 90 for Reality Mining.")
    ap.add_argument("--undirected", action="store_true", help="Force undirected adjacency (recommended).")
    ap.add_argument("--tea_plot", action="store_true", help="Save TEA bar chart.")
    ap.add_argument("--triplets", action="store_true", help="Also generate triplets per snapshot.")
    ap.add_argument("--k_small", type=int, default=2)
    ap.add_argument("--k_large", type=int, default=3)
    ap.add_argument("--triplets_per_node", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(">> Extract / locate data")
    data_root = _extract_if_archive(input_path, out_dir / "_extracted")
    out_mit = _find_out_mit(data_root)
    print(f"   using: {out_mit}")

    print(">> Parse out.mit")
    df = _read_out_mit(out_mit)
    if df["t"].isna().all():
        raise SystemExit("Parsed file has no timestamps; Reality Mining should include per-edge time.")

    print(">> Map nodes to 0..n-1")
    df, node_list = _map_nodes(df)
    np.save(out_dir / "node_list.npy", node_list)

    print(">> Build snapshots")
    snapshots, bin_edges = _edges_to_snapshots(
        df, T=args.snapshots, undirected=args.undirected, remove_self_loops=True
    )
    np.save(out_dir / "time_bins.npy", bin_edges)

    # Save snapshots efficiently
    print(">> Save snapshots (CSR list)")
    # store as a npz with object array of csr matrices
    np.savez_compressed(out_dir / "snapshots_csr.npz",
                        snapshots=np.array(snapshots, dtype=object))

    print(">> TEA stats")
    tea = _compute_tea(snapshots)
    with (out_dir / "tea_stats.json").open("w") as f:
        json.dump(tea, f, indent=2)
    if args.tea_plot:
        _plot_tea(tea, out_dir / "tea_plot.png")

    print(">> Train/Val/Test split (70/10/20)")
    splits = _split_time(len(snapshots), 0.7, 0.1, 0.2)
    with (out_dir / "splits.json").open("w") as f:
        json.dump(splits, f, indent=2)

    if args.triplets:
        print(">> Triplet sampling per snapshot")
        for t, A in enumerate(tqdm(snapshots, desc="triplets")):
            tri = _triplets_for_snapshot(
                A,
                k_small=args.k_small,
                k_large=args.k_large,
                per_node=args.triplets_per_node,
                seed=args.seed + t,
            )
            np.save(out_dir / f"triplets_t{t}.npy", tri)

    # Quick sanity prints
    n = snapshots[0].shape[0]
    m_edges_total = int(sum(A.nnz for A in snapshots) // 2) if args.undirected else int(sum(A.nnz for A in snapshots))
    print(f">> Done. n={n}, T={len(snapshots)}, total (binary) edge-entries across T ≈ {m_edges_total}")
    print(f"   novelty ≈ {tea['novelty']:.4f}")
    print(f"   train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"   artifacts in: {out_dir}")

if __name__ == "__main__":
    main()
