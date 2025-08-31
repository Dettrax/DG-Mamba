#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reality Mining → TransformerG2G-style dataset.

Loads:
  data_dir/
    snapshots_csr.npz   # list[csr_matrix] length T
    node_list.npy       # original node ids (sorted)
    splits.json         # contiguous time splits
    triplets_t{t}.npy   # optional per-t triplets (ref, near, far)

Yields for split S and lookback ℓ:
  history: float32 [ℓ, n]   = [A_{t-ℓ}[i], ..., A_{t-1}[i]]
  target:  float32 [n]      =  A_t[i]
  meta:    dict { "t": int, "node": int, "present": bool, ... }
  (optional) triplets for node i at time t
"""
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------- Utils ----------------

def _load_snapshots(data_dir: Path) -> List:
    npz = np.load(data_dir / "snapshots_csr.npz", allow_pickle=True)
    snaps = list(npz["snapshots"])
    return snaps

def _densify(snaps, dtype=np.float32) -> np.ndarray:
    dense = [A.toarray().astype(dtype, copy=False) for A in snaps]
    return np.stack(dense, axis=0)  # [T, n, n]

def _load_splits(data_dir: Path) -> Dict[str, List[int]]:
    with (data_dir / "splits.json").open("r") as f:
        return json.load(f)

def _maybe_load_triplets(data_dir: Path, T: int) -> Optional[List[np.ndarray]]:
    triplets = []
    have_any = False
    for t in range(T):
        p = data_dir / f"triplets_t{t}.npy"
        if p.exists():
            triplets.append(np.load(p))
            have_any = True
        else:
            triplets.append(None)
    return triplets if have_any else None

def _index_triplets_by_anchor(triplets: List[Optional[np.ndarray]], n: int):
    """idx[t][i] -> list[(near, far)] or None"""
    if triplets is None:
        return None
    out: List[Optional[List[List[Tuple[int,int]]]]] = []
    for t, arr in enumerate(triplets):
        if arr is None or arr.size == 0:
            out.append(None)
            continue
        bucket = [[] for _ in range(n)]
        for ref, near, far in arr:
            bucket[int(ref)].append((int(near), int(far)))
        out.append(bucket)
    return out

# ---------------- Dataset ----------------

class RealityMiningSeqDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 lookback: int = 8,
                 active_only: bool = True,
                 include_triplets: bool = False,
                 device: Optional[torch.device] = None):
        """
        data_dir: path to preprocessed artifacts from prepare_reality_mining.py
        split: 'train'|'val'|'test'
        lookback (ℓ): number of past time steps
        active_only: only yield (t,i) where node i has degree>0 at time t (target)
        include_triplets: attach (near, far) list for each sample if available
        """
        self.data_dir = Path(data_dir)
        snaps = _load_snapshots(self.data_dir)
        self.T = len(snaps)
        self.n = snaps[0].shape[0]
        self.A = _densify(snaps, dtype=np.float32)  # [T, n, n]
        self.splits = _load_splits(self.data_dir)
        assert split in self.splits, f"split must be one of {list(self.splits)}"
        self.times: List[int] = self.splits[split]
        self.lookback = int(lookback)
        self.device = device

        # Node present at t if degree>0 in A_t
        self.deg = self.A.sum(axis=2)  # [T, n]
        if active_only:
            self.indices = [(t, i) for t in self.times for i in range(self.n)
                            if self.deg[t, i] > 0.0]
        else:
            self.indices = [(t, i) for t in self.times for i in range(self.n)]

        # Triplets (optional)
        self.triplets_by_anchor = None
        if include_triplets:
            triplets = _maybe_load_triplets(self.data_dir, self.T)
            self.triplets_by_anchor = _index_triplets_by_anchor(triplets, self.n)

    def __len__(self):
        return len(self.indices)

    def _history_window(self, t: int, i: int) -> np.ndarray:
        """
        Return [ℓ, n]: zero-left-padded if t-ℓ < 0.
        """
        l = self.lookback
        if l <= 0:
            return np.zeros((0, self.n), dtype=np.float32)
        start = t - l
        if start >= 0:
            H = self.A[start:t, i, :]  # [ℓ, n]
        else:
            pad = np.zeros((-start, self.n), dtype=np.float32)
            H = self.A[0:t, i, :]
            H = np.concatenate([pad, H], axis=0)
        return H

    def __getitem__(self, idx):
        t, i = self.indices[idx]
        H = self._history_window(t, i)           # [ℓ, n]
        y = self.A[t, i, :]                      # [n]
        present = bool(self.deg[t, i] > 0.0)

        sample = {
            "history": torch.from_numpy(H),      # float32 [ℓ,n]
            "target":  torch.from_numpy(y),      # float32 [n]
            "t":       t,
            "node":    i,
            "present": present,
        }

        if self.triplets_by_anchor is not None:
            pairs = None
            if self.triplets_by_anchor[t] is not None:
                pairs = self.triplets_by_anchor[t][i]  # list[(near,far)]
            sample["triplets"] = pairs  # may be None or empty
        return sample

# ---------------- Collate & Loader ----------------

def collate_samples(batch: List[Dict]):
    """
    Custom collate that stacks tensors and leaves `triplets` ragged.
    """
    history = torch.stack([b["history"] for b in batch], dim=0)  # [B, ℓ, n]
    target  = torch.stack([b["target"]  for b in batch], dim=0)  # [B, n]
    t       = torch.tensor([b["t"]    for b in batch], dtype=torch.long)
    node    = torch.tensor([b["node"] for b in batch], dtype=torch.long)
    present = torch.tensor([b["present"] for b in batch], dtype=torch.bool)
    # keep ragged triplets as a Python list (or None)
    triplets = [b.get("triplets", None) for b in batch]
    out = {
        "history": history,
        "target": target,
        "t": t,
        "node": node,
        "present": present,
    }
    # only include key if any sample asked for it
    if any(k is not None for k in triplets):
        out["triplets"] = triplets
    return out

def make_loader(data_dir: str,
                split: str,
                lookback: int,
                batch_size: int = 64,
                shuffle: bool = True,
                active_only: bool = True,
                include_triplets: bool = False,
                num_workers: int = 0,
                pin_memory: bool = False):
    ds = RealityMiningSeqDataset(
        data_dir=data_dir,
        split=split,
        lookback=lookback,
        active_only=active_only,
        include_triplets=include_triplets,
    )
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                    collate_fn=collate_samples)  # <- custom collate
    return ds, dl

# ---------------- CLI sanity check ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--lookback", default=8, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--active_only", action="store_true")
    ap.add_argument("--include_triplets", action="store_true")
    args = ap.parse_args()

    ds, dl = make_loader(
        data_dir=args.data_dir,
        split=args.split,
        lookback=args.lookback,
        batch_size=args.batch_size,
        shuffle=True,
        active_only=args.active_only,
        include_triplets=args.include_triplets,
    )

    print(f"Dataset split={args.split}, lookback={args.lookback}")
    print(f"  T={ds.T}, n={ds.n}, samples={len(ds)}")
    for batch in dl:
        H = batch["history"]   # [B, ℓ, n]
        y = batch["target"]    # [B, n]
        print("Batch shapes:", tuple(H.shape), tuple(y.shape))
        pos = y.sum(dim=1).float().mean().item()
        print(f"Avg positives in target row ≈ {pos:.2f}")
        nonempty = (H.sum(dim=(1,2)) > 0).float().mean().item()
        print(f"Fraction non-empty histories ≈ {nonempty:.3f}")
        if "triplets" in batch:
            cnt = sum(len(p) if p is not None else 0 for p in batch["triplets"])
            print(f"Triplet pairs in batch: {cnt}")
        break

if __name__ == "__main__":
    main()
