#!/usr/bin/env python3
# tg2g_mlp_free_fast.py
# One-file, fast training script for time-evolving graph → graph link prediction (Reality Mining–style)
# Core ideas:
#  - Precompute all input sequences ONCE (vectorized), so the loader just indexes arrays.
#  - Vectorized Gaussian similarity scoring (L2 or symmetric KL) against a learnable node bank.
#  - AMP (fp16/bf16) + torch.compile + TF32 to squeeze GPU throughput without dropping MAP.
#  - Pinned-memory, multi-worker data pipeline with small, non-Pythonic collation cost.
#  - Honest eval: MAP/MRR overall and for NEW edges (not seen in the lookback window). Also simple baselines.
#
# Expected data format:
#   A: numpy array of shape [T, n, n], binary (0/1) or float in [0,1]. Row A[t, i, :] are neighbors of i at time t.
#   Load from:  --data path/to/A.npy   or  --data path/to/A.npz  (expects key "A")
#
# Example:
#   python tg2g_mlp_free_fast.py --data A.npy --lookback 4 --train_mode next_time --eval_mode next_time \
#       --loss_nce_w 1.0 --loss_triplet_w 0.2 --pos_cap 8 --neg_cap 64 --hard_neg_k 0 \
#       --tau 0.07 --use_posenc --epochs 40 --batch_size 256 --amp bf16 --compile
#
# NOTE: This script is self-contained; it does not rely on your previous modules.
#
import os
import math
import json
import time
import argparse
import random
from typing import Tuple, Optional, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- Repro & CUDA knobs --------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_cuda_fastmath():
    # TF32 gives big wins on Ampere+ with near-FP32 quality
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# -------------------------- Sequence builders --------------------------
# All builders return float32 arrays shaped [L, n]

def build_seq_same_time(A: np.ndarray, t: int, i: int, lookback: int) -> np.ndarray:
    """History [t-l+1 .. t] INCLUDING current (leaky)."""
    l, n = lookback, A.shape[2]
    start = max(0, t - l + 1)
    H = A[start:t+1, i, :]  # shape [<=L, n]
    if H.shape[0] < l:
        pad = np.zeros((l - H.shape[0], n), dtype=A.dtype)
        H = np.concatenate([pad, H], axis=0)
    return H.astype(np.float32)


def build_seq_same_time_no_leak(A: np.ndarray, t: int, i: int, lookback: int) -> np.ndarray:
    """History [t-l .. t-1] (no leak of current)."""
    l, n = lookback, A.shape[2]
    start = max(0, t - l)
    end = max(0, t)  # exclusive
    H = A[start:end, i, :]
    if H.shape[0] < l:
        pad = np.zeros((l - H.shape[0], n), dtype=A.dtype)
        H = np.concatenate([pad, H], axis=0)
    return H.astype(np.float32)


def build_seq_next_time(A: np.ndarray, t: int, i: int, lookback: int) -> np.ndarray:
    """Predict next slice: sequence is [t-l .. t-1], target is at time t."""
    return build_seq_same_time_no_leak(A, t, i, lookback)


# -------------------------- NEW-edge helper --------------------------

def compute_new_mask(A_hist: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mark edges in y as NEW if they did not occur at any time in A_hist.
       A_hist: [L, n], y: [n] -> returns [n] boolean mask.
    """
    prev = (A_hist > 0.5).any(axis=0)  # [n]
    return (~prev) & (y > 0.5)


# -------------------------- Scoring (Gaussian) --------------------------

def score_matrix_L2(mu_A: torch.Tensor, mu_B: torch.Tensor) -> torch.Tensor:
    """Return scores (higher is better): negative L2^2 distances.
       mu_A: [B, d], mu_B: [K, d] -> [B, K]
    """
    a2 = (mu_A**2).sum(dim=-1, keepdim=True)   # [B,1]
    b2 = (mu_B**2).sum(dim=-1).unsqueeze(0)    # [1,K]
    # -||a-b||^2 = -a^2 - b^2 + 2 a.b
    return -a2 - b2 + 2.0 * (mu_A @ mu_B.t())


def _sym_kl_terms(mu_A, sg_A, mu_B, sg_B, eps=1e-8):
    """Helper to compute symmetric KL terms efficiently.
       All shapes: A is [B,d], B is [K,d].
       Returns the NEGATIVE sym-KL (so higher is better): -0.5*(tr + quad - 2d)
    """
    d = mu_A.size(-1)
    # Clamp sigmas to avoid blowups; work with variances
    sg_A = torch.clamp(sg_A, min=eps)
    sg_B = torch.clamp(sg_B, min=eps)
    inv_A = 1.0 / sg_A                 # [B,d]
    inv_B = 1.0 / sg_B                 # [K,d]

    # tr(SA^{-1} SB + SB^{-1} SA) = sum_d (invA*SB + invB*SA)
    tr_term = (inv_A.unsqueeze(1) * sg_B.unsqueeze(0) +
               inv_B.unsqueeze(0) * sg_A.unsqueeze(1)).sum(dim=-1)  # [B,K]

    # (muA - muB)^T (invA + invB) (muA - muB)
    diff2 = (mu_A.unsqueeze(1) - mu_B.unsqueeze(0))**2              # [B,K,d]
    quad = ((inv_A.unsqueeze(1) + inv_B.unsqueeze(0)) * diff2).sum(dim=-1)  # [B,K]

    symkl = 0.5 * (tr_term + quad - 2.0 * d)    # [B,K], >= 0
    return -symkl                                # score = negative distance


def score_matrix_symKL(mu_A: torch.Tensor, sg_A: torch.Tensor,
                       mu_B: torch.Tensor, sg_B: torch.Tensor,
                       eps: float = 1e-8) -> torch.Tensor:
    return _sym_kl_terms(mu_A, sg_A, mu_B, sg_B, eps)


def score_vector_L2(mu_a: torch.Tensor, mu_B: torch.Tensor) -> torch.Tensor:
    """Single anchor vs bank -> [K]"""
    return score_matrix_L2(mu_a.unsqueeze(0), mu_B).squeeze(0)


def score_vector_symKL(mu_a: torch.Tensor, sg_a: torch.Tensor,
                       mu_B: torch.Tensor, sg_B: torch.Tensor,
                       eps: float = 1e-8) -> torch.Tensor:
    return _sym_kl_terms(mu_a.unsqueeze(0), sg_a.unsqueeze(0), mu_B, sg_B, eps).squeeze(0)


# -------------------------- MAP/MRR (dtype-safe) --------------------------

@torch.no_grad()
def _map_mrr_for_node(scores: torch.Tensor, labels: torch.Tensor, self_index: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    # Optionally mask self index from ranking
    if self_index is not None and 0 <= self_index < scores.numel():
        scores = scores.clone()
        scores[self_index] = -1e9

    order = torch.argsort(scores, descending=True)
    labs = labels[order]
    pos_idx = (labs > 0.5).nonzero(as_tuple=False).squeeze(-1)
    if pos_idx.numel() == 0:
        return None, None
    precisions = []
    for k in pos_idx.tolist():
        precisions.append(labs[:k+1].float().mean().item())
    ap = float(np.mean(precisions))
    rr = float(1.0 / (pos_idx[0].item() + 1))
    return ap, rr


@torch.no_grad()
def average_map_mrr(scores: torch.Tensor, labels: torch.Tensor, self_indices: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """scores, labels: [B, n]"""
    B = scores.size(0)
    aps, rrs = [], []
    for b in range(B):
        si = int(self_indices[b].item()) if self_indices is not None else None
        ap, rr = _map_mrr_for_node(scores[b], labels[b], si)
        if ap is not None:
            aps.append(ap); rrs.append(rr)
    if len(aps) == 0:
        return 0.0, 0.0
    return float(np.mean(aps)), float(np.mean(rrs))


# -------------------------- Baselines --------------------------

@torch.no_grad()
def baseline_last_seen(seq_hist: torch.Tensor) -> torch.Tensor:
    """Use the last row of the sequence as the score. seq_hist: [B,L,n] -> [B,n]"""
    return seq_hist[:, -1, :].float()


@torch.no_grad()
def baseline_popularity(seq_hist: torch.Tensor) -> torch.Tensor:
    """Use frequency in the window as score. seq_hist: [B,L,n] -> [B,n]"""
    return seq_hist.float().sum(dim=1)


# -------------------------- Model --------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class TG2GEncoder(nn.Module):
    """Sequence encoder -> Gaussian (mu, sigma) for the anchor node at time t."""
    def __init__(self, n: int, d_model: int = 128, emb_dim: int = 64, hidden_after: int = 128,
                 layers: int = 2, dropout: float = 0.1, use_posenc: bool = True):
        super().__init__()
        self.row_proj = nn.Linear(n, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=max(1, d_model // 32), dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.posenc = PositionalEncoding(d_model) if use_posenc else nn.Identity()
        self.after = nn.Sequential(
            nn.Linear(d_model, hidden_after),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_after, 2 * emb_dim)  # -> [mu | log_sigma]
        )

    def forward(self, seq_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq_hist: [B, L, n]
        x = self.row_proj(seq_hist)          # [B, L, d_model]
        x = self.posenc(x)
        x = self.encoder(x)                  # [B, L, d_model]
        x = x[:, -1, :]                      # take last token
        out = self.after(x)                  # [B, 2*emb_dim]
        mu, log_sigma = out.chunk(2, dim=-1) # [B,d], [B,d]
        sigma = F.softplus(log_sigma) + 1e-5
        return mu, sigma


class NodeBank(nn.Module):
    """Learnable Gaussian for each node j in {1..n}."""
    def __init__(self, n: int, emb_dim: int):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(n, emb_dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(n, emb_dim))

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu, F.softplus(self.log_sigma) + 1e-5


# -------------------------- Dataset with precomputation --------------------------

class SeqCache:
    """Precompute sequences for all (t,i) and the 3 modes. Keeps them in CPU RAM.
       Memory: ~ T*n*L*n floats per mode. For MIT (T~62,n~96,L=5) ≈ 11.4MB/mode.
    """
    def __init__(self, A: np.ndarray, lookback: int):
        self.A = A
        self.T, self.n, _ = A.shape
        self.L = lookback
        self._next = {}
        self._st   = {}
        self._st0  = {}

    def _add_key(self, d, key, arr):
        d[key] = arr.astype(np.float32, copy=False)

    def get(self, mode: str, t: int, i: int) -> np.ndarray:
        if mode == "next_time":
            if t <= 0:
                raise ValueError("next_time requires t>=1")
            key = (t, i)
            d = self._next
            if key in d:
                return d[key]
            arr = build_seq_next_time(self.A, t, i, self.L)
            self._add_key(d, key, arr)
            return arr
        elif mode == "same_time_leaky":
            key = (t, i)
            d = self._st
            if key in d:
                return d[key]
            arr = build_seq_same_time(self.A, t, i, self.L)
            self._add_key(d, key, arr)
            return arr
        elif mode == "same_time_no_leak":
            key = (t, i)
            d = self._st0
            if key in d:
                return d[key]
            arr = build_seq_same_time_no_leak(self.A, t, i, self.L)
            self._add_key(d, key, arr)
            return arr
        else:
            raise ValueError(f"Unknown mode {mode}")


class RealityMiningSeqDataset(torch.utils.data.Dataset):
    """Index over all valid (t,i). Returns:
       seq_hist: [L, n] float32
       y:        [n] float32
       i_index:  int (self index)
       new_mask: [n] bool (optional)
    """
    def __init__(self, A: np.ndarray, lookback: int, mode: str = "next_time"):
        assert mode in ("next_time", "same_time_leaky", "same_time_no_leak")
        self.A = A.astype(np.float32, copy=False)
        self.T, self.n, _ = A.shape
        self.L = lookback
        self.mode = mode
        self.cache = SeqCache(self.A, lookback=lookback)
        # Build indices (t, i)
        self.indices: List[Tuple[int,int]] = []
        t_start = lookback if mode in ("next_time", "same_time_no_leak") else (lookback - 1)
        t_start = max(1, t_start)  # avoid t=0 for next_time
        for t in range(t_start, self.T):
            for i in range(self.n):
                self.indices.append((t, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        t, i = self.indices[idx]
        seq = self.cache.get(self.mode, t, i)             # [L, n]
        y = self.A[t, i, :].astype(np.float32)            # [n]
        new_mask = compute_new_mask(seq, y).astype(np.bool_)
        return {
            "seq_hist": torch.from_numpy(seq),            # [L,n]
            "y": torch.from_numpy(y),                     # [n]
            "i_index": torch.tensor(i, dtype=torch.long), # []
            "new_mask": torch.from_numpy(new_mask),       # [n] bool
        }


def collate_samples(batch: List[dict]) -> dict:
    # Simple, zero-copy stacking for tensors
    seq_hist = torch.stack([b["seq_hist"] for b in batch], dim=0)  # [B,L,n]
    y = torch.stack([b["y"] for b in batch], dim=0)                # [B,n]
    i_index = torch.stack([b["i_index"] for b in batch], dim=0)    # [B]
    new_mask = torch.stack([b["new_mask"] for b in batch], dim=0)  # [B,n]
    return {"seq_hist": seq_hist, "y": y, "i_index": i_index, "new_mask": new_mask}


# -------------------------- Losses --------------------------

def nce_multi_positive(scores: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Multi-positive InfoNCE on full-bank scores.
       scores: [B, n], y in {0,1}^n
       loss = - E[log sum_{p in P} exp(s_p / tau) - log sum_{all} exp(s / tau)]
    """
    s = scores / max(tau, 1e-8)
    # clamp for stability
    s = s - s.max(dim=1, keepdim=True).values
    # positives
    pos_mask = (y > 0.5)
    # avoid empty positives: skip such items from the mean
    pos_logsumexp = torch.where(
        pos_mask.any(dim=1, keepdim=True),
        torch.logsumexp(torch.where(pos_mask, s, torch.full_like(s, -1e9)), dim=1, keepdim=True),
        torch.zeros_like(s[:, :1])
    )
    all_logsumexp = torch.logsumexp(s, dim=1, keepdim=True)
    # Only include valid rows
    valid = pos_mask.any(dim=1)
    loss = -(pos_logsumexp[valid] - all_logsumexp[valid]).mean()
    return loss


def triplet_from_scores(scores: torch.Tensor, y: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Triplet margin (on scores, higher is better): choose one pos/neg per row.
       scores: [B, n], y in {0,1}
    """
    with torch.no_grad():
        pos_scores = torch.where(y > 0.5, scores, torch.full_like(scores, -1e9))
        neg_scores = torch.where(y < 0.5, scores, torch.full_like(scores, -1e9))
        p = pos_scores.argmax(dim=1)  # [B]
        n = neg_scores.argmax(dim=1)  # [B]
    s_p = scores[torch.arange(scores.size(0), device=scores.device), p]
    s_n = scores[torch.arange(scores.size(0), device=scores.device), n]
    return F.relu(margin - (s_p - s_n)).mean()


# -------------------------- Training loop --------------------------

def make_autocast(amp_dtype: str):
    from contextlib import nullcontext
    if amp_dtype == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if amp_dtype == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to A.npy or A.npz (key A)")
    ap.add_argument("--train_mode", type=str, default="next_time", choices=["next_time", "same_time_leaky", "same_time_no_leak"])
    ap.add_argument("--eval_mode", type=str, default="next_time", choices=["next_time", "same_time_leaky", "same_time_no_leak"])
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--hidden_after", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--amp", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--pin_memory", action="store_true", default=True)
    ap.add_argument("--tau", type=float, default=0.07, help="temperature for InfoNCE")
    ap.add_argument("--loss_nce_w", type=float, default=1.0)
    ap.add_argument("--loss_triplet_w", type=float, default=0.2)
    ap.add_argument("--score", type=str, default="symkl", choices=["symkl", "l2"])
    ap.add_argument("--save_best", type=str, default="", help="path to save best checkpoint")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--early_stop_metric", type=str, default="map", choices=["map", "mrr"])

    args = ap.parse_args()

    set_seed(args.seed)
    setup_cuda_fastmath()

    # ------------------ Load data ------------------
    path = args.data
    if path.endswith(".npz"):
        A = np.load(path)["A"]
    else:
        A = np.load(path)
    assert A.ndim == 3 and A.shape[1] == A.shape[2], "A must be [T, n, n]"
    A = (A > 0.0).astype(np.float32)  # binarize

    T, n, _ = A.shape
    print(f"[data] A shape = {A.shape}, density = {A.mean():.5f}")

    # ------------------ Dataset / Loader ------------------
    ds_train = RealityMiningSeqDataset(A, lookback=args.lookback, mode=args.train_mode)
    ds_eval  = RealityMiningSeqDataset(A, lookback=args.lookback, mode=args.eval_mode)

    loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0), collate_fn=collate_samples
    )

    # ------------------ Model ------------------
    device = torch.device(args.device)
    enc = TG2GEncoder(n=n, d_model=args.d_model, emb_dim=args.emb_dim,
                      hidden_after=args.hidden_after, layers=args.layers,
                      dropout=args.dropout, use_posenc=True).to(device)
    bank = NodeBank(n=n, emb_dim=args.emb_dim).to(device)

    if args.compile and hasattr(torch, "compile"):
        enc = torch.compile(enc, mode="reduce-overhead")
        bank = bank  # embedding bank is simple; compiling not needed

    params = list(enc.parameters()) + list(bank.parameters())
    # Fused AdamW when available on CUDA
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=(device.type=="cuda")) \
          if hasattr(torch.optim, "AdamW") else torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp in ("fp16","bf16")))

    # ------------------ Training ------------------
    best_metric = -1.0
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        enc.train(); bank.train()
        t0 = time.time()
        total_loss = 0.0
        n_steps = 0

        for batch in loader:
            seq_hist = batch["seq_hist"].to(device, non_blocking=True)  # [B,L,n]
            y = batch["y"].to(device, non_blocking=True)                # [B,n]

            with make_autocast(args.amp):
                mu_a, sg_a = enc(seq_hist)                # [B,d], [B,d]
                mu_B, sg_B = bank.get()                   # [n,d], [n,d]
                if args.score == "symkl":
                    S = score_matrix_symKL(mu_a, sg_a, mu_B, sg_B)  # [B,n]
                else:
                    S = score_matrix_L2(mu_a, mu_B)                 # [B,n]

                # Loss: InfoNCE multi-positive + optional triplet on scores
                loss = 0.0
                if args.loss_nce_w > 0:
                    loss = loss + args.loss_nce_w * nce_multi_positive(S, y, args.tau)
                if args.loss_triplet_w > 0:
                    loss = loss + args.loss_triplet_w * triplet_from_scores(S, y, margin=0.2)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            total_loss += float(loss.item())
            n_steps += 1

        elapsed = time.time() - t0
        print(f"[train] epoch={epoch} steps={n_steps} loss={total_loss/max(1,n_steps):.4f} time={elapsed:.1f}s")

        # ------------------ Eval ------------------
        enc.eval(); bank.eval()
        with torch.no_grad():
            mu_B, sg_B = bank.get()  # [n,d],[n,d]
            mu_B = mu_B.to(device); sg_B = sg_B.to(device)

            # Evaluate in minibatches to avoid OOM
            B = 1024
            all_scores = []
            all_labels = []
            all_self = []

            for k in range(0, len(ds_eval), B):
                sl = slice(k, min(k+B, len(ds_eval)))
                batch_items = [ds_eval[idx] for idx in range(sl.start, sl.stop)]
                cb = collate_samples(batch_items)
                seq_hist = cb["seq_hist"].to(device)
                y = cb["y"].to(device)
                i_index = cb["i_index"].to(device)

                mu_a, sg_a = enc(seq_hist)
                if args.score == "symkl":
                    S = score_matrix_symKL(mu_a, sg_a, mu_B, sg_B)  # [b,n]
                else:
                    S = score_matrix_L2(mu_a, mu_B)

                all_scores.append(S)
                all_labels.append(y)
                all_self.append(i_index)

            scores = torch.cat(all_scores, dim=0)   # [N, n]
            labels = torch.cat(all_labels, dim=0)   # [N, n]
            self_idx = torch.cat(all_self, dim=0)   # [N]

            MAP, MRR = average_map_mrr(scores, labels, self_indices=self_idx)
            print(f"[eval] MAP={MAP:.4f} | MRR={MRR:.4f}")

            current = MAP if args.early_stop_metric == "map" else MRR
            if current > best_metric + 1e-6:
                best_metric = current
                best_epoch = epoch
                epochs_since_improve = 0
                if args.save_best:
                    torch.save({
                        "enc": enc.state_dict(),
                        "bank": bank.state_dict(),
                        "args": vars(args),
                        "metric": best_metric,
                        "epoch": epoch
                    }, args.save_best)
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= args.patience:
                    print(f"[early-stop] epoch={epoch} no_improve={epochs_since_improve} best_{args.early_stop_metric}={best_metric:.4f} @ epoch {best_epoch}")
                    break

    print(f"[done] best_{args.early_stop_metric}={best_metric:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()