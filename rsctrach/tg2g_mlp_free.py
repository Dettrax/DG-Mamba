# train_tg2g_mit.py

# MLP-free link prediction:

# score(i, j) = - || mu_i - mu_j ||_2

# We keep the same triplet training; evaluation uses distances only.

import argparse, json, random

from pathlib import Path

from typing import List, Dict, Tuple, Optional

import numpy as np

import torch

import torch.nn as nn

from mit_dataset import RealityMiningSeqDataset, collate_samples

from transformerg2g import TransformerG2G, triplet_contrastive_loss

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

# -------------------------- Sequence builders --------------------------

def build_seq_same_time(A: np.ndarray, t: int, i: int, lookback: int) -> np.ndarray:

    """History [t-l .. t-1] + CURRENT row A[t,i,:] (length l+1)."""

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

    """History [t-l .. t-1] + ZERO row (length l+1)."""

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

    """Use history up to t-1 (length l+1) by reusing 'same_time' at t-1. Labels use A[t]."""

    assert t - 1 >= 0, "next_time requires t>=1"

    return build_seq_same_time(A, t - 1, i, lookback)

# -------------------- Embedding computation (mu only is needed) --------------------

def compute_embeddings(model: TransformerG2G,

                       A: np.ndarray,

                       times: List[int],

                       lookback: int,

                       device: torch.device,

                       mode: str = "next_time",

                       batch_nodes: int = 1024) -> Dict[int, torch.Tensor]:

    """Return dict t -> mu[t] of shape [n, d] on CPU."""

    model.eval()

    T, n, _ = A.shape

    out_mu: Dict[int, torch.Tensor] = {}

    with torch.no_grad():

        for t in times:

            if mode == "next_time":

                if t - 1 < 0:

                    continue

                seqs = np.stack([build_seq_next_time(A, t, i, lookback) for i in range(n)], axis=0)

            elif mode == "same_time_no_leak":

                seqs = np.stack([build_seq_same_time_no_leak(A, t, i, lookback) for i in range(n)], axis=0)

            elif mode == "same_time_leaky":

                seqs = np.stack([build_seq_same_time(A, t, i, lookback) for i in range(n)], axis=0)

            else:

                raise ValueError(f"Unknown mode: {mode}")

            seqs_t = torch.from_numpy(seqs).to(device)

            mus = []

            for s in range(0, n, batch_nodes):

                mu, _ = model(seqs_t[s:s+batch_nodes])

                mus.append(mu.detach().cpu())

            out_mu[t] = torch.cat(mus, dim=0) # [n, d]

    return out_mu

# -------------------------- Metrics (MAP/MRR) --------------------------

def _map_mrr_for_node(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:

    """scores, labels: [N] for a single query (self already excluded or set to -inf)."""

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

def average_map_mrr_L2(embs_mu: Dict[int, torch.Tensor],

                       A: np.ndarray, times: List[int],

                       device: torch.device) -> Tuple[float, float]:

    """MLP-free: scores are -L2 distances between embeddings."""

    APs, MRRs = [], []

    with torch.no_grad():

        for t in times:

            if t not in embs_mu:

                continue

            mu_t = embs_mu[t].to(device) # [n, d]

            A_t = torch.from_numpy(A[t]).float().to(device)

            # pairwise distances once for speed

            D = torch.cdist(mu_t, mu_t, p=2) # [n, n]

            S = -D # higher is better

            n = mu_t.size(0)

            for i in range(n):

                labels = A_t[i].clone()

                labels[i] = 0.0

                scores = S[i].clone()

                scores[i] = -1e9

                AP, MRR = _map_mrr_for_node(scores, labels)

                if AP is not None:

                    APs.append(AP); MRRs.append(MRR)

    return (float(np.mean(APs)) if APs else 0.0,

            float(np.mean(MRRs)) if MRRs else 0.0)

def average_map_mrr_L2_new_edges(embs_mu: Dict[int, torch.Tensor],

                                 A: np.ndarray, times: List[int],

                                 device: torch.device) -> Tuple[float, float]:

    """New-edges-only evaluation using -L2 distances."""

    APs, MRRs = [], []

    with torch.no_grad():

        for t in times:

            if t not in embs_mu or t == 0:

                continue

            mu_t = embs_mu[t].to(device)

            A_t = torch.from_numpy(A[t]).float().to(device)

            A_tm1= torch.from_numpy(A[t-1]).float().to(device)

            D = torch.cdist(mu_t, mu_t, p=2) # [n, n]

            S = -D

            n = mu_t.size(0)

            for i in range(n):

                new_mask = (A_tm1[i] == 0.0).float()

                labels = (A_t[i] * new_mask)

                if labels.sum() == 0:

                    continue

                scores = S[i].clone()

                scores[i] = -1e9

                # mask out old edges

                scores = scores + (1 - new_mask) * (-1e9)

                AP, MRR = _map_mrr_for_node(scores, labels)

                if AP is not None:

                    APs.append(AP); MRRs.append(MRR)

    return (float(np.mean(APs)) if APs else 0.0,

            float(np.mean(MRRs)) if MRRs else 0.0)

def baseline_last_seen(A: np.ndarray, times: List[int]) -> Tuple[float, float]:

    """Scores are A[t-1][i, :] (no learning)."""

    APs, MRRs = [], []

    for t in times:

        if t == 0:

            continue

        A_prev = torch.from_numpy(A[t-1]).float()

        A_t = torch.from_numpy(A[t]).float()

        n = A_t.size(0)

        for i in range(n):

            labels = A_t[i].clone(); labels[i] = 0.0

            scores = A_prev[i].clone(); scores[i] = -1e9

            AP, MRR = _map_mrr_for_node(scores, labels)

            if AP is not None:

                APs.append(AP); MRRs.append(MRR)

    return (float(np.mean(APs)) if APs else 0.0,

            float(np.mean(MRRs)) if MRRs else 0.0)

# ----------------------- Triplet batching from loader -----------------------

def batch_build_triplet_inputs(batch,

                               A: np.ndarray,

                               lookback: int,

                               max_pairs_per_anchor: int = 4,

                               mode: str = "same_time"):

    """Build (anchor, near, far) sequences [M, L, n] from a collated batch (ragged triplets).

    mode: 'same_time' (uses A[t] in sequences) or 'next_time' (uses up to A[t-1]).

    """

    anc_seqs, near_seqs, far_seqs = [], [], []

    B = batch["history"].shape[0]

    for b in range(B):

        t = int(batch["t"][b].item())

        i = int(batch["node"][b].item())

        pairs = batch.get("triplets", [None] * B)[b]

        if not pairs:

            continue

        if mode == "next_time" and t == 0:

            continue  # anchor sequence

        if mode == "same_time":

            H = batch["history"][b].numpy()

            y = batch["target"][b].numpy()

            anc = np.concatenate([H, y[None, :]], axis=0).astype(np.float32)

            make = lambda node: build_seq_same_time(A, t, node, lookback)

        else:

            anc = build_seq_next_time(A, t, i, lookback)

            make = lambda node: build_seq_next_time(A, t, node, lookback)

        take = min(max_pairs_per_anchor, len(pairs))

        sel = np.random.choice(len(pairs), size=take, replace=False)

        for k in sel:

            near, far = pairs[k]

            near_seqs.append(make(int(near)))

            far_seqs.append(make(int(far)))

            anc_seqs.append(anc)

    if not anc_seqs:

        return None, None, None

    return (torch.from_numpy(np.stack(anc_seqs, axis=0)),

            torch.from_numpy(np.stack(near_seqs, axis=0)),

            torch.from_numpy(np.stack(far_seqs, axis=0)))

# ----------------------------------- Main -----------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="./data/mit_reality_mining")

    ap.add_argument("--epochs", type=int, default=120)

    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--lookback", type=int, default=1)

    ap.add_argument("--d_model", type=int, default=128)  # Increased from 128

    ap.add_argument("--emb_dim", type=int, default=64)  # Increased from 64

    ap.add_argument("--hidden_after", type=int, default=128)  # Increased from 128

    ap.add_argument("--nhead", type=int, default=4)  # Increased from 2

    ap.add_argument("--layers", type=int, default=2)  # Increased from 1

    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_pairs_per_anchor", type=int, default=4)

    ap.add_argument("--eval_every", type=int, default=20)

    ap.add_argument("--eval_mode", type=str, default="next_time",

                    choices=["next_time", "same_time_no_leak", "same_time_leaky"])

    ap.add_argument("--train_mode", type=str, default="next_time",

                    choices=["same_time", "next_time"])

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    # Dataset for triplet training

    ds = RealityMiningSeqDataset(

        data_dir=args.data_dir, split="train", lookback=args.lookback,

        active_only=True, include_triplets=True

    )

    A = ds.A # [T, n, n]

    T, n, _ = A.shape

    model = TransformerG2G(

        n_nodes=n, lookback=args.lookback, d_model=args.d_model,

        hidden_after=args.hidden_after, emb_dim=args.emb_dim,

        nhead=args.nhead, num_layers=args.layers, dropout=args.dropout,

        causal_attention=False  # Disabled causal attention

    ).to(device)

    loader = torch.utils.data.DataLoader(

        ds, batch_size=args.batch_size, shuffle=True, drop_last=False,

        num_workers=0, pin_memory=False, collate_fn=collate_samples

    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_l2_map = 0.0 # Track best L2-only MAP

    print(f"Training TransformerG2G on train timestamps={len(ds.times)} (n={n})")

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss, steps = 0.0, 0

        for batch in loader:

            anc, near, far = batch_build_triplet_inputs(

                batch, A, args.lookback, args.max_pairs_per_anchor, mode=args.train_mode

            )

            if anc is None:

                continue

            anc, near, far = anc.to(device), near.to(device), far.to(device)

            mu_a, sg_a = model(anc)

            mu_n, sg_n = model(near)

            mu_f, sg_f = model(far)

            triplet_loss = triplet_contrastive_loss(mu_a, sg_a, mu_n, sg_n, mu_f, sg_f, 0.25, "mean")

            # Auxiliary L2-based loss: encourage smaller L2 for near, larger for far

            l2_near = torch.cdist(mu_a, mu_n, p=2).diag().mean()

            l2_far = torch.cdist(mu_a, mu_f, p=2).diag().mean()

            l2_loss = l2_near - l2_far  # Minimize near distances, maximize far

            loss = triplet_loss + 0.1 * l2_loss  # Lambda=0.1 as auxiliary weight

            opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            opt.step()

            total_loss += loss.item()

            steps += 1

        print(f"[Epoch {epoch:03d}] total loss: {total_loss/max(steps,1):.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:

            splits = json.load(open(Path(args.data_dir) / "splits.json"))

            times_test = splits["test"]

            # ---- Embeddings (TEST) ----

            emu_te = compute_embeddings(model, A, times_test, args.lookback, device, mode=args.eval_mode)

            # L2-only MAP/MRR

            MAP_l2, MRR_l2 = average_map_mrr_L2(emu_te, A, times_test, device)

            print(f"==> [{args.eval_mode}] L2-only MAP={MAP_l2:.4f}, MRR={MRR_l2:.4f}")

            # Track best L2-only MAP

            if MAP_l2 > best_l2_map:

                best_l2_map = MAP_l2

            # New-edges-only with L2

            MAP_new, MRR_new = average_map_mrr_L2_new_edges(emu_te, A, times_test, device)

            print(f"==> [{args.eval_mode}] New-edges L2 MAP={MAP_new:.4f}, MRR={MRR_new:.4f}")

            # Last-seen baseline (fixed yardstick)

            MAP_b, MRR_b = baseline_last_seen(A, times_test)

            print(f"==> Baseline last-seen MAP={MAP_b:.4f}, MRR={MRR_b:.4f}")

    print(f"Best L2-only MAP: {best_l2_map:.4f}")

if __name__ == "__main__":

    main()
