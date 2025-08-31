# mamba_mlp_free_final.py
# One-file training script for MambaG2G on Reality Mining-style data.
# Goals:
#   • Correct, leak-safe sequence construction for {next_time, same_time_no_leak, same_time_leaky}
#   • Stable losses: KL-triplet + (optional) InfoNCE on Gaussian similarity (L2 or sym-KL)
#   • AMP-safe (fp16/bf16/none) without half-overflow, with dtype-aware masking
#   • Fast(er): sequence caching, fused optimizer if available, fewer Python loops
#   • Honest evaluation: MAP/MRR for L2 & sym-KL, plus NEW-edge breakdown and baselines
#
# Usage examples
#   python mamba_mlp_free_final.py --lookback 4 --train_mode next_time --eval_mode next_time
#   python mamba_mlp_free_final.py --lookback 4 --train_mode next_time --eval_mode next_time \
#       --loss_nce_w 1.0 --loss_triplet_w 0.2 --pos_cap 8 --neg_cap 64 --hard_neg_k 0 \
#       --new_pos_weight 2.0 --tau 0.07 --use_posenc
#   python mamba_mlp_free_final.py --sanity random|oracle

import argparse, json, random, os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from mit_dataset import RealityMiningSeqDataset, collate_samples
from mambag2g import MambaG2G, triplet_contrastive_loss  # Gaussian KL triplet

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
    """History [t-l .. t-1] + CURRENT row A[t,i,:] (leaky)."""
    l, n = lookback, A.shape[2]
    start = t - l
    if start >= 0:
        H = A[start:t, i, :]
    else:
        pad = np.zeros((-start, n), dtype=np.float32)
        H = A[0:t, i, :]
        H = np.concatenate([pad, H], axis=0)
    cur = A[t, i, :]
    return np.concatenate([H, cur[None, :]], axis=0).astype(np.float32)


def build_seq_same_time_no_leak(A: np.ndarray, t: int, i: int, lookback: int) -> np.ndarray:
    """History [t-l .. t-1] + ZERO row at t (non-leaky)."""
    l, n = lookback, A.shape[2]
    start = t - l
    if start >= 0:
        H = A[start:t, i, :]
    else:
        pad = np.zeros((-start, n), dtype=np.float32)
        H = A[0:t, i, :]
        H = np.concatenate([pad, H], axis=0)
    zero = np.zeros((1, n), dtype=np.float32)
    return np.concatenate([H, zero], axis=0).astype(np.float32)


def build_seq_next_time(A: np.ndarray, t: int, i: int, lookback: int) -> np.ndarray:
    """Use up to t-1 by reusing 'same_time' at (t-1). Labels use A[t]."""
    assert t - 1 >= 0, "next_time requires t>=1"
    return build_seq_same_time(A, t - 1, i, lookback)


# -------------------------- Sequence cache (speeds up training) --------------------------

class SeqCache:
    """Precompute sequences for all (t,i) and the 3 modes. Keeps them in CPU RAM.
       Memory: ~ T*n*L*n floats per mode. For MIT (T~62,n~96,L=5) ≈ 11.4MB/mode.
    """
    def __init__(self, A: np.ndarray, lookback: int):
        self.A = A
        self.T, self.n, _ = A.shape
        self.L = lookback + 1
        self.lb = lookback
        self._next = {}
        self._st   = {}
        self._st0  = {}

    def get(self, mode: str, t: int, i: int) -> np.ndarray:
        if mode == "next_time":
            if t <= 0:
                raise ValueError("next_time requires t>=1")
            key = (t, i)
            d = self._next
            if key in d:
                return d[key]
            arr = build_seq_next_time(self.A, t, i, self.lb)
            d[key] = arr
            return arr
        elif mode == "same_time_leaky":
            key = (t, i)
            d = self._st
            if key in d:
                return d[key]
            arr = build_seq_same_time(self.A, t, i, self.lb)
            d[key] = arr
            return arr
        elif mode == "same_time_no_leak":
            key = (t, i)
            d = self._st0
            if key in d:
                return d[key]
            arr = build_seq_same_time_no_leak(self.A, t, i, self.lb)
            d[key] = arr
            return arr
        else:
            raise ValueError(f"Unknown mode {mode}")


# -------------------------- Gaussian score helpers --------------------------

@torch.no_grad()
def score_matrix_L2(mu: torch.Tensor) -> torch.Tensor:
    # -||μ_i - μ_j||2, higher is better
    D = torch.cdist(mu, mu, p=2)
    return -D


@torch.no_grad()
def score_matrix_symKL(mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric KL between diagonal Gaussians, vectorized.
       Returns *scores* = -d_sym (higher is better)."""
    s = sigma.clamp_min(eps)  # [n,d]
    inv_s = 1.0 / s
    n, d = mu.shape
    diff2 = (mu.unsqueeze(1) - mu.unsqueeze(0)) ** 2  # [n,n,d]
    invsj = inv_s.unsqueeze(0)
    invsi = inv_s.unsqueeze(1)
    sj = s.unsqueeze(0)
    si = s.unsqueeze(1)
    term_sigma = (si / sj + sj / si).sum(dim=-1)                # [n,n]
    term_mu = (diff2 * (invsj + invsi)).sum(dim=-1)             # [n,n]
    d_sym = 0.5 * (term_sigma + term_mu - 2.0 * d)
    d_sym = 0.5 * (d_sym + d_sym.T)
    d_sym.fill_diagonal_(0.0)
    return -d_sym


def score_vector_symKL(mu_a: torch.Tensor, sg_a: torch.Tensor,
                       mu_B: torch.Tensor, sg_B: torch.Tensor,
                       eps: float = 1e-8) -> torch.Tensor:
    """Scores from single anchor to a set of targets (higher is better). Shapes:
       mu_a, sg_a: [d]; mu_B, sg_B: [K,d] -> returns [K]."""
    s_i = sg_a.clamp_min(eps)
    s_B = sg_B.clamp_min(eps)
    inv_i = 1.0 / s_i
    inv_B = 1.0 / s_B
    term_sigma = (s_i / s_B + s_B / s_i).sum(dim=-1)           # [K]
    term_mu = ((mu_B - mu_a)**2 * (inv_B + inv_i)).sum(dim=-1) # [K]
    d = mu_a.numel()
    d_sym = 0.5 * (term_sigma + term_mu - 2.0 * d)
    return -d_sym


def score_vector_L2(mu_a: torch.Tensor, mu_B: torch.Tensor) -> torch.Tensor:
    return -torch.cdist(mu_a[None, :], mu_B, p=2).squeeze(0)


def neg_large_for(dtype: torch.dtype) -> float:
    # Safe, dtype-aware large negative for masking & log-softmax, AMP-safe.
    if dtype == torch.float16:
        return -1e4
    if dtype == torch.bfloat16:
        return -1e6
    return -1e9


# -------------------------- MAP/MRR (dtype-safe) --------------------------

@torch.no_grad()
def _map_mrr_for_node(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    order = torch.argsort(scores, descending=True)
    labs = labels[order]
    pos_idx = (labs > 0.5).nonzero(as_tuple=False).squeeze(-1)
    if pos_idx.numel() == 0:
        return None, None
    precisions = []
    for k in pos_idx.tolist():
        precisions.append(labs[:k+1].float().mean().item())
    AP = float(np.mean(precisions))
    MRR = 1.0 / float(pos_idx[0].item() + 1)
    return AP, MRR


@torch.no_grad()
def _avg_map_mrr_from_S(S: torch.Tensor, A_t: torch.Tensor) -> Tuple[float, float]:
    n = S.size(0)
    APs, MRRs = [], []
    ng = neg_large_for(S.dtype)
    for i in range(n):
        labels = A_t[i].clone()
        labels[i] = 0.0
        scores = S[i].clone()
        # use a dtype-safe negative to avoid half/bfloat overflows
        scores[i] = ng
        AP, MRR = _map_mrr_for_node(scores, labels)
        if AP is not None:
            APs.append(AP); MRRs.append(MRR)
    return (float(np.mean(APs)) if APs else 0.0,
            float(np.mean(MRRs)) if MRRs else 0.0)


@torch.no_grad()
def average_map_mrr_L2(embs: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                       A: np.ndarray, times: List[int], device: torch.device,
                       thr: float = 0.5) -> Tuple[float, float]:
    APs, MRRs = [], []
    for t in times:
        if t not in embs:
            continue
        mu_t, _ = embs[t]
        # Always evaluate in float32 for numerical sanity
        mu_t = mu_t.to(device=device, dtype=torch.float32)
        A_t = torch.from_numpy((A[t] >= thr).astype(np.float32)).to(device)
        S = score_matrix_L2(mu_t)
        AP, MRR = _avg_map_mrr_from_S(S, A_t)
        APs.append(AP); MRRs.append(MRR)
    return (float(np.mean(APs)) if APs else 0.0,
            float(np.mean(MRRs)) if MRRs else 0.0)


@torch.no_grad()
def average_map_mrr_KL(embs: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                       A: np.ndarray, times: List[int], device: torch.device,
                       thr: float = 0.5) -> Tuple[float, float]:
    APs, MRRs = [], []
    for t in times:
        if t not in embs:
            continue
        mu_t, sg_t = embs[t]
        mu_t = mu_t.to(device=device, dtype=torch.float32)
        sg_t = sg_t.to(device=device, dtype=torch.float32)
        A_t = torch.from_numpy((A[t] >= thr).astype(np.float32)).to(device)
        S = score_matrix_symKL(mu_t, sg_t)
        AP, MRR = _avg_map_mrr_from_S(S, A_t)
        APs.append(AP); MRRs.append(MRR)
    return (float(np.mean(APs)) if APs else 0.0,
            float(np.mean(MRRs)) if MRRs else 0.0)


@torch.no_grad()
def average_map_mrr_L2_new_edges(embs: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                                 A: np.ndarray, times: List[int], device: torch.device,
                                 thr: float = 0.5) -> Tuple[float, float]:
    APs, MRRs = [], []
    for t in times:
        if t not in embs or t == 0:
            continue
        mu_t, _ = embs[t]
        mu_t = mu_t.to(device=device, dtype=torch.float32)
        A_t   = torch.from_numpy((A[t]   >= thr).astype(np.float32)).to(device)
        A_tm1 = torch.from_numpy((A[t-1] >= thr).astype(np.float32)).to(device)
        S = score_matrix_L2(mu_t)
        n = mu_t.size(0)
        ng = neg_large_for(S.dtype)
        for i in range(n):
            new_mask = (A_tm1[i] == 0.0).float()
            labels = (A_t[i] * new_mask)
            if labels.sum() == 0:
                continue
            scores = S[i].clone()
            scores[i] = ng
            scores = scores + (1 - new_mask) * ng
            AP, MRR = _map_mrr_for_node(scores, labels)
            if AP is not None:
                APs.append(AP); MRRs.append(MRR)
    return (float(np.mean(APs)) if APs else 0.0,
            float(np.mean(MRRs)) if MRRs else 0.0)


@torch.no_grad()
def average_map_mrr_KL_new_edges(embs: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                                 A: np.ndarray, times: List[int], device: torch.device,
                                 thr: float = 0.5) -> Tuple[float, float]:
    APs, MRRs = [], []
    for t in times:
        if t not in embs or t == 0:
            continue
        mu_t, sg_t = embs[t]
        mu_t = mu_t.to(device=device, dtype=torch.float32)
        sg_t = sg_t.to(device=device, dtype=torch.float32)
        A_t   = torch.from_numpy((A[t]   >= thr).astype(np.float32)).to(device)
        A_tm1 = torch.from_numpy((A[t-1] >= thr).astype(np.float32)).to(device)
        S = score_matrix_symKL(mu_t, sg_t)
        n = mu_t.size(0)
        ng = neg_large_for(S.dtype)
        for i in range(n):
            new_mask = (A_tm1[i] == 0.0).float()
            labels = (A_t[i] * new_mask)
            if labels.sum() == 0:
                continue
            scores = S[i].clone()
            scores[i] = ng
            scores = scores + (1 - new_mask) * ng
            AP, MRR = _map_mrr_for_node(scores, labels)
            if AP is not None:
                APs.append(AP); MRRs.append(MRR)
    return (float(np.mean(APs)) if APs else 0.0,
            float(np.mean(MRRs)) if MRRs else 0.0)


# -------------------------- Baselines --------------------------

@torch.no_grad()
def baseline_last_seen(A: np.ndarray, times: List[int], thr: float = 0.5) -> Tuple[float, float]:
    APs, MRRs = [], []
    for t in times:
        if t == 0:  # no previous
            continue
        A_prev = torch.from_numpy((A[t-1] >= thr).astype(np.float32))
        A_t = torch.from_numpy((A[t] >= thr).astype(np.float32))
        n = A_t.size(0)
        ng = neg_large_for(torch.float32)
        for i in range(n):
            labels = A_t[i].clone(); labels[i] = 0.0
            scores = A_prev[i].clone(); scores[i] = ng
            AP, MRR = _map_mrr_for_node(scores, labels)
            if AP is not None:
                APs.append(AP); MRRs.append(MRR)
    return (float(np.mean(APs)) if APs else 0.0,
            float(np.mean(MRRs)) if MRRs else 0.0)


@torch.no_grad()
def baseline_popularity_new(A: np.ndarray, times: List[int], thr: float = 0.5) -> Tuple[float, float]:
    """Rank by (t-1) indegree among *new* candidates."""
    APs, MRRs = [], []
    for t in times:
        if t == 0:
            continue
        A_t   = torch.from_numpy((A[t]   >= thr).astype(np.float32))
        A_tm1 = torch.from_numpy((A[t-1] >= thr).astype(np.float32))
        indeg_tm1 = A_tm1.sum(dim=0)  # [n]
        n = A_t.size(0)
        ng = neg_large_for(torch.float32)
        for i in range(n):
            new_mask = (A_tm1[i] == 0.0).float()
            labels = (A_t[i] * new_mask)
            if labels.sum() == 0:
                continue
            scores = indeg_tm1.clone()
            scores[i] = ng
            scores = scores + (1 - new_mask) * ng
            AP, MRR = _map_mrr_for_node(scores, labels)
            if AP is not None:
                APs.append(AP); MRRs.append(MRR)
    return (float(np.mean(APs)) if APs else 0.0,
            float(np.mean(MRRs)) if MRRs else 0.0)


# -------------------------- Embedding computation --------------------------

@torch.no_grad()
def compute_embeddings(model: MambaG2G, A: np.ndarray, times: List[int], lookback: int,
                       device: torch.device, mode: str = "next_time", batch_nodes: int = 1024,
                       seq_cache: Optional[SeqCache] = None) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    T, n, _ = A.shape
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for t in times:
        if mode == "next_time" and t - 1 < 0:
            continue
        # Build sequences for all nodes at time t
        if seq_cache is None:
            if mode == "next_time":
                seqs = np.stack([build_seq_next_time(A, t, i, lookback) for i in range(n)], axis=0)
            elif mode == "same_time_no_leak":
                seqs = np.stack([build_seq_same_time_no_leak(A, t, i, lookback) for i in range(n)], axis=0)
            elif mode == "same_time_leaky":
                seqs = np.stack([build_seq_same_time(A, t, i, lookback) for i in range(n)], axis=0)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            seqs = np.stack([seq_cache.get(mode, t, i) for i in range(n)], axis=0)

        seqs_t = torch.from_numpy(seqs).to(device)
        mus, sigmas = [], []
        for s in range(0, n, batch_nodes):
            mu, sg = model(seqs_t[s:s+batch_nodes])
            mus.append(mu.detach().cpu())
            sigmas.append(sg.detach().cpu())
        out[t] = (torch.cat(mus, dim=0), torch.cat(sigmas, dim=0))
    return out


# -------------------------- NCE loss (stable, masked) --------------------------

def nce_loss_from_pairs(
    mu_a: torch.Tensor, sg_a: torch.Tensor,
    mu_pos: torch.Tensor, sg_pos: Optional[torch.Tensor],
    mu_neg: torch.Tensor, sg_neg: Optional[torch.Tensor],
    metric: str = "kl", tau: float = 0.07,
    pos_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute InfoNCE for a *single* anchor given sets of pos/neg.
    Works in fp32 regardless of AMP to avoid half log-softmax weirdness.
    """
    assert mu_pos.ndim == 2 and mu_neg.ndim == 2
    mu_a = mu_a.float(); sg_a = (sg_a.float() if sg_a is not None else None)
    mu_pos = mu_pos.float(); mu_neg = mu_neg.float()
    sg_pos = (sg_pos.float() if sg_pos is not None else None)
    sg_neg = (sg_neg.float() if sg_neg is not None else None)

    if metric == "kl":
        sp = score_vector_symKL(mu_a, sg_a, mu_pos, sg_pos)
        sn = score_vector_symKL(mu_a, sg_a, mu_neg, sg_neg)
    else:
        sp = score_vector_L2(mu_a, mu_pos)
        sn = score_vector_L2(mu_a, mu_neg)

    logits = torch.cat([sp, sn], dim=0) / max(1e-6, float(tau))  # [P+K]
    # Masking not needed here; callers only pass valid pos/neg
    log_probs = F.log_softmax(logits, dim=-1)  # [P+K]
    logp_pos = log_probs[:sp.numel()]
    if pos_weights is None:
        loss = -(logp_pos.mean())
    else:
        w = pos_weights.float()
        w = w / (w.sum().clamp_min(1.0))
        loss = -torch.sum(w * logp_pos)
    return loss


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
    ap.add_argument("--data_dir", type=str, default="./data/mit_reality_mining")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lookback", type=int, default=4)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden_after", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_pairs_per_anchor", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=10)

    ap.add_argument("--eval_mode", type=str, default="next_time",
                    choices=["next_time", "same_time_no_leak", "same_time_leaky"])
    ap.add_argument("--train_mode", type=str, default="next_time",
                    choices=["same_time", "next_time"])

    # NEW: label binarization for eval & new-edge definitions
    ap.add_argument("--label_threshold", type=float, default=0.5)

    # Loss mixing
    ap.add_argument("--loss_nce_w", type=float, default=1.0)
    ap.add_argument("--loss_triplet_w", type=float, default=1.0)
    ap.add_argument("--nce_metric", type=str, default="kl", choices=["kl", "l2"])
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--new_pos_weight", type=float, default=1.0)
    ap.add_argument("--pos_cap", type=int, default=8)
    ap.add_argument("--neg_cap", type=int, default=64)
    ap.add_argument("--hard_neg_k", type=int, default=0, help="extra mined negatives per anchor (0=off)")

    # AMP & speed
    ap.add_argument("--amp_dtype", type=str, default="none", choices=["none", "fp16", "bf16"])
    ap.add_argument("--num_workers", type=int, default=max(2, os.cpu_count() // 2))
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--compile", action="store_true")

    # Alignment + stability
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--sigma_reg", type=float, default=1e-4)

    # Early stopping
    ap.add_argument("--early_stop_metric", type=str, default="KL_MAP", choices=["L2_MAP", "KL_MAP"])
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--save_best", type=str, default="")

    # Mamba core hparams
    ap.add_argument("--d_state", type=int, default=32)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--use_posenc", action="store_true")

    # Sanity checks
    ap.add_argument("--sanity", type=str, default="", choices=["", "random", "oracle"])

    # Extra baseline prints
    ap.add_argument("--print_popularity_new", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)
    setup_cuda_fastmath()
    device = torch.device(args.device)

    # Dataset
    ds = RealityMiningSeqDataset(
        data_dir=args.data_dir, split="train", lookback=args.lookback,
        active_only=True, include_triplets=True
    )
    A = ds.A  # [T, n, n] float32
    T, n, _ = A.shape

    # Seq cache
    seq_cache = SeqCache(A, args.lookback)

    # Model
    model = MambaG2G(
        n_nodes=n, lookback=args.lookback, d_model=args.d_model,
        hidden_after=args.hidden_after, emb_dim=args.emb_dim,
        num_layers=args.layers, dropout=args.dropout,
        d_state=args.d_state, d_conv=args.d_conv, expand=args.expand,
        use_posenc=args.use_posenc,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    # Loader
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0), collate_fn=collate_samples
    )

    # Optimizer (try fused AdamW on CUDA)
    fused_ok = (device.type == "cuda") and hasattr(torch.optim, "AdamW")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            fused=fused_ok)

    # Splits
    splits = json.load(open(Path(args.data_dir) / "splits.json"))
    times_test = splits["test"]

    print(f"Training MambaG2G on train timestamps={len(ds.times)} (n={n})")
    print(f"Modes: train_mode={args.train_mode} | eval_mode={args.eval_mode} | lookback={args.lookback}")
    thr = float(args.label_threshold)
    print(f"Threshold thr={thr}")

    # Sanity-only runs (no training)
    if args.sanity:
        # random baseline on test split
        if args.sanity == "random":
            # Pretend random scores ~ N(0,1)
            rng = torch.Generator().manual_seed(args.seed)
            embs = {}
            for t in times_test:
                mu = torch.randn(n, args.emb_dim, generator=rng)
                sg = torch.rand(n, args.emb_dim, generator=rng) + 0.5
                embs[t] = (mu, sg)
            MAP_l2, MRR_l2 = average_map_mrr_L2(embs, A, times_test, device, thr)
            MAP_kl, MRR_kl = average_map_mrr_KL(embs, A, times_test, device, thr)
            print(f"[SANITY random] L2  MAP={MAP_l2:.4f}, MRR={MRR_l2:.4f}")
            print(f"[SANITY random] KL  MAP={MAP_kl:.4f}, MRR={MRR_kl:.4f}")
            return
        if args.sanity == "oracle":
            # Perfect ranking by using A[t] as scores
            embs = {}
            for t in times_test:
                mu = torch.from_numpy(A[t]).float()  # abuse: treat rows as embeddings with perfect distances
                sg = torch.ones_like(mu) * 1.0
                embs[t] = (mu, sg)
            MAP_l2, MRR_l2 = average_map_mrr_L2(embs, A, times_test, device, thr)
            MAP_kl, MRR_kl = average_map_mrr_KL(embs, A, times_test, device, thr)
            print(f"[SANITY oracle] L2  MAP={MAP_l2:.4f}, MRR={MRR_l2:.4f}")
            print(f"[SANITY oracle] KL  MAP={MAP_kl:.4f}, MRR={MRR_kl:.4f}")
            return

    use_amp = (args.amp_dtype != "none" and device.type == "cuda")
    autocast = make_autocast(args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp_dtype == "fp16"))

    best_metric = -1.0
    best_epoch = 0
    epochs_since_improve = 0

    # -------------- Training --------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        sg_mean = float("nan"); sg_max = float("nan")

        for batch in loader:
            # Gather anchors
            B = batch["history"].shape[0]
            t_list = batch["t"].tolist()
            i_list = batch["node"].tolist()
            pairs_list = batch.get("triplets", [None] * B)

            # Build candidate lists per anchor (caps)
            all_anchor_keys = []
            per_anchor_pos_keys = []
            per_anchor_neg_keys = []
            per_anchor_pos_weights = []

            for b in range(B):
                t = int(t_list[b]); i = int(i_list[b])
                if args.train_mode == "next_time" and t == 0:
                    continue
                all_anchor_keys.append((t, i))
                pairs = pairs_list[b] or []
                # shuffle to avoid order bias
                if len(pairs) > 0:
                    random.shuffle(pairs)
                # select up to pos_cap pairs
                take = min(args.pos_cap, len(pairs))
                pos_keys = []
                neg_keys = []
                pos_wts = []
                for k in range(take):
                    ne, fa = pairs[k]
                    pos_keys.append((t, int(ne)))
                    neg_keys.append((t, int(fa)))
                    # NEW-edge weight (optional)
                    if t > 0 and args.new_pos_weight != 1.0:
                        was_edge = A[t-1, i, int(ne)] >= thr
                        is_edge  = A[t,   i, int(ne)] >= thr
                        w = args.new_pos_weight if (not was_edge and is_edge) else 1.0
                    else:
                        w = 1.0
                    pos_wts.append(w)
                # pad if no triplets
                per_anchor_pos_keys.append(pos_keys)
                per_anchor_neg_keys.append(neg_keys)
                per_anchor_pos_weights.append(pos_wts)

            if not all_anchor_keys:
                continue

            # Deduplicate all target keys to avoid repeated forwards
            uniq_keys = {}
            uniq_list: List[Tuple[int,int]] = []
            def _add_key(key):
                if key not in uniq_keys:
                    uniq_keys[key] = len(uniq_list)
                    uniq_list.append(key)
                return uniq_keys[key]

            # fill indices
            anc_indices = [_add_key(k) for k in all_anchor_keys]
            pos_indices_per_anchor: List[List[int]] = []
            neg_indices_per_anchor: List[List[int]] = []
            for pos_keys, neg_keys in zip(per_anchor_pos_keys, per_anchor_neg_keys):
                pos_indices_per_anchor.append([_add_key(k) for k in pos_keys])
                neg_indices_per_anchor.append([_add_key(k) for k in neg_keys])

            # materialize sequences for all uniq keys in the chosen train_mode
            mode_for_targets = "next_time" if args.train_mode == "next_time" else "same_time_leaky"
            seqs_np = np.stack([seq_cache.get(mode_for_targets, t, i) for (t, i) in uniq_list], axis=0)
            seqs = torch.from_numpy(seqs_np).to(device)

            # forward once for all uniq nodes
            with autocast:
                mu_all, sg_all = model(seqs)

            # slice python lists now
            mu_all32 = mu_all.float(); sg_all32 = sg_all.float()

            # Update sigma stats
            sg_mean = float(sg_all32.mean().item()); sg_max = float(sg_all32.max().item())

            # -------------- Build losses --------------
            loss_terms = []

            # Triplet (KL) from original pairs if requested
            if args.loss_triplet_w > 0.0:
                trip_mu_a = []
                trip_sg_a = []
                trip_mu_n = []
                trip_sg_n = []
                trip_mu_f = []
                trip_sg_f = []
                for ai, pos_idx_list, neg_idx_list in zip(anc_indices, pos_indices_per_anchor, neg_indices_per_anchor):
                    # each selected pair gives one triplet
                    m = min(len(pos_idx_list), len(neg_idx_list), args.max_pairs_per_anchor)
                    for k in range(m):
                        pidx = pos_idx_list[k]
                        nidx = neg_idx_list[k]
                        trip_mu_a.append(mu_all32[ai]); trip_sg_a.append(sg_all32[ai])
                        trip_mu_n.append(mu_all32[pidx]); trip_sg_n.append(sg_all32[pidx])
                        trip_mu_f.append(mu_all32[nidx]); trip_sg_f.append(sg_all32[nidx])
                if trip_mu_a:
                    mu_a = torch.stack(trip_mu_a, dim=0)
                    sg_a = torch.stack(trip_sg_a, dim=0)
                    mu_n = torch.stack(trip_mu_n, dim=0)
                    sg_n = torch.stack(trip_sg_n, dim=0)
                    mu_f = torch.stack(trip_mu_f, dim=0)
                    sg_f = torch.stack(trip_sg_f, dim=0)
                    with autocast:
                        tloss = triplet_contrastive_loss(mu_a, sg_a, mu_n, sg_n, mu_f, sg_f,
                                                          margin=args.margin, reduction="mean")
                    loss_terms.append(args.loss_triplet_w * tloss)

            # InfoNCE if requested
            if args.loss_nce_w > 0.0:
                nce_losses = []
                for ai, pos_idx_list, neg_idx_list, pos_wts in zip(anc_indices, pos_indices_per_anchor, neg_indices_per_anchor, per_anchor_pos_weights):
                    if len(pos_idx_list) == 0 or len(neg_idx_list) == 0:
                        continue
                    # cap
                    pos_idx_list = pos_idx_list[:args.pos_cap]
                    neg_idx_list = neg_idx_list[:args.neg_cap]
                    mu_pos = mu_all32[pos_idx_list]
                    mu_neg = mu_all32[neg_idx_list]
                    sg_pos = sg_all32[pos_idx_list]
                    sg_neg = sg_all32[neg_idx_list]
                    wts = torch.tensor(pos_wts[:len(pos_idx_list)], device=mu_pos.device)
                    loss_i = nce_loss_from_pairs(mu_all32[ai], sg_all32[ai],
                                                 mu_pos, sg_pos, mu_neg, sg_neg,
                                                 metric=args.nce_metric, tau=args.tau,
                                                 pos_weights=wts if (args.new_pos_weight != 1.0) else None)
                    nce_losses.append(loss_i)
                if nce_losses:
                    loss_terms.append(args.loss_nce_w * torch.stack(nce_losses, dim=0).mean())

            if not loss_terms:
                continue

            loss = sum(loss_terms)
            # Optional variance regularizer: discourage variance blow-up
            if args.sigma_reg > 0:
                loss = loss + (sg_all32.mean() * args.sigma_reg)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            total_loss += float(loss.detach().item())
            steps += 1

        print(f"[Epoch {epoch:03d}] loss: {total_loss/max(steps,1):.4f} | sigma(mean,max)≈({sg_mean:.4f},{sg_max:.4f})")

        # -------- Evaluation --------
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            embs_te = compute_embeddings(model, A, times_test, args.lookback, device,
                                         mode=args.eval_mode, seq_cache=seq_cache)

            with torch.no_grad():
                MAP_l2, MRR_l2 = average_map_mrr_L2(embs_te, A, times_test, device, thr)
                MAP_kl, MRR_kl = average_map_mrr_KL(embs_te, A, times_test, device, thr)
                print(f"==> [{args.eval_mode}] L2   MAP={MAP_l2:.4f}, MRR={MRR_l2:.4f}")
                print(f"==> [{args.eval_mode}] KL   MAP={MAP_kl:.4f}, MRR={MRR_kl:.4f}")

                MAP_new_l2, MRR_new_l2 = average_map_mrr_L2_new_edges(embs_te, A, times_test, device, thr)
                MAP_new_kl, MRR_new_kl = average_map_mrr_KL_new_edges(embs_te, A, times_test, device, thr)
                print(f"==> [{args.eval_mode}] NEW  L2-MAP={MAP_new_l2:.4f}, L2-MRR={MRR_new_l2:.4f}")
                print(f"==> [{args.eval_mode}] NEW  KL-MAP={MAP_new_kl:.4f}, KL-MRR={MRR_new_kl:.4f}")

                MAP_b_w, MRR_b_w = baseline_last_seen(A, times_test, thr)
                print(f"==> Baseline last-seen (binary)   MAP={MAP_b_w:.4f}, MRR={MRR_b_w:.4f}")
                if args.print_popularity_new:
                    MAP_pop, MRR_pop = baseline_popularity_new(A, times_test, thr)
                    print(f"==> Baseline popularity (NEW)     MAP={MAP_pop:.4f}, MRR={MRR_pop:.4f}")

            # Early stop logic
            current = MAP_l2 if args.early_stop_metric == "L2_MAP" else MAP_kl
            if current > best_metric + 1e-6:
                best_metric = current
                best_epoch = epoch
                epochs_since_improve = 0
                if args.save_best:
                    torch.save({"model": model.state_dict(),
                                "args": vars(args),
                                "metric": best_metric,
                                "epoch": epoch}, args.save_best)
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch} | best {args.early_stop_metric}={best_metric:.4f} @ epoch {best_epoch}")
                    return


if __name__ == "__main__":
    main()
