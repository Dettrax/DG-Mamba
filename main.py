# --- main.py ---
import time, copy, argparse, random
import numpy as np
import torch, torch.nn.functional as F
import math
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import get_dataset, negative_sampling
from model import STFormerGCN

# ---------------- reproducibility ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---------------- args ----------------
parser = argparse.ArgumentParser(description="Temporal Spatial Mamba + Graph2Gauss")
parser.add_argument('--dataset_name', type=str, default='uci')
parser.add_argument('--dataset_interval', default='D', choices=['W','D','M'])
parser.add_argument('--train_ratio', type=float, default=0.70)
parser.add_argument('--val_ratio',   type=float, default=0.15)
parser.add_argument('--window_size', type=int,   default=3)
parser.add_argument('--d_model',     type=int,   default=32)
parser.add_argument('--device',      type=str,   default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--epochs',      type=int,   default=10)
parser.add_argument('--patience',    type=int,   default=5)
parser.add_argument('--lr',          type=float, default=5e-4)
parser.add_argument('--weight_decay',type=float, default=1e-5)
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--use_pos_emb', action='store_true',default=True, help='add positional tokens before Mamba')
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(args.device)

# ---------------- data ----------------
dataset = get_dataset(args)
print(f"Total Snapshots: {len(dataset.snapshots)}, Total Nodes: {dataset.num_nodes}, Total Edges: {dataset.num_edges}")

# ---------------- model ----------------
W = args.window_size
# Force Transformer on CPU to avoid Mamba CUDA-only ops
use_mamba = (device.type == 'cuda') and False
model = STFormerGCN(
    in_dim=128, gcn_dim=128, d_model=args.d_model,
    num_nodes=dataset.num_nodes, use_id_emb=True,
    gcn_layers=3, num_tlayers=3, dropout=0.15, max_len=W, temporal_type="mamba",
    proj_dim=args.d_model,
).to(device)
print(f"Temporal encoder: {model.temporal.__class__.__name__}")
model.margin  = 1.0  # Larger margin
model.sym_kl  = False
model.var_reg = 5e-4  # Variance regularization
model.sigma_floor = 1e-3

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)


def compute_loss(logits, labels):
    fn = torch.nn.BCEWithLogitsLoss()
    logits = logits.flatten()
    labels = labels.flatten().float()
    return fn(logits, labels)

def compute_roc_auc(logits, labels):
    probabilities = torch.sigmoid(logits)
    return roc_auc_score(y_true=labels.detach().cpu().numpy(), y_score=probabilities.detach().cpu().numpy())

def compute_ap(logits, labels):
    probabilities = torch.sigmoid(logits)
    return average_precision_score(y_true=labels.detach().cpu().numpy(), y_score=probabilities.detach().cpu().numpy())

# =====================================================================================
# STFormerGCN path — KL energy, leak-free, hard negative mining
# =====================================================================================

def _delta_seq_from_winB(winB, device, N):
    """
    Strictly past time encoding:
      Δt per step computed relative to the last step in the window (winB[-1]).
    Returns: [N, W] tensor (days).
    """
    W = len(winB)
    has_ts = all(hasattr(s, 'snapshot_ts') for s in winB)
    if has_ts:
        win_ts = torch.tensor([float(s.snapshot_ts) for s in winB], device=device, dtype=torch.float32)
        ref_ts = win_ts[-1]
        delta_days = (ref_ts - win_ts) / 86400.0
        delta_days = torch.clamp(delta_days, min=0.0)
    else:
        delta_days = torch.arange(W - 1, -1, -1, device=device, dtype=torch.float32)
    return delta_days.unsqueeze(0).expand(N, -1)  # [N, W]

@torch.no_grad()
def _precompute_spatial_seq(model, dataset):
    """Precompute spatial embeddings H_t for all snapshots to speed training/eval."""
    H_all = []
    for s in dataset.snapshots:
        H_all.append(model.encode_one_snapshot(s))  # [N,d]
    return H_all  # list of length T of [N,d]

@torch.no_grad()
def _collect_logits_labels(model, dataset, device, W, split):
    """
    For AP/AUC reporting only (pos vs random negs).
    Uses window-B embeddings with strictly past Δt.
    """
    begin, end = dataset.get_range_by_split(split)
    t_start = begin + (W - 1)
    t_stop = end - 1
    logits_all, labels_all = [], []

    H_all = _precompute_spatial_seq(model, dataset)

    for t in range(t_start, t_stop):
        winB = dataset.snapshots[t - (W - 1): t + 1]
        if len(winB) != W:
            continue

        N = dataset.snapshots[0].node_feature.size(0)
        delta_seq = _delta_seq_from_winB(winB, device, N)
        H_seq = torch.stack(H_all[t - (W - 1): t + 1], dim=1)  # [N,W,d]
        z = model.temporal(H_seq, delta_seq=delta_seq)  # [N, d]

        next_snapshot = dataset.snapshots[t + 1]
        pos_ei = next_snapshot.edge_index.to(device)
        neg_ei = negative_sampling(pos_ei, num_nodes=dataset.num_nodes).to(device)

        ei_all = torch.cat([pos_ei, neg_ei], dim=1)
        y = torch.cat([torch.ones(pos_ei.size(1)), torch.zeros(neg_ei.size(1))], dim=0).to(device)
        logits = model.score_edges(z, ei_all)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    if len(logits_all) == 0:
        return None, None
    return torch.cat(logits_all), torch.cat(labels_all)

@torch.no_grad()
def _mine_hard_negatives(mu_src, mu_dst, var_src, var_dst, pos_ei, k=500, topk=200, N=None):
    """
    Hard negative mining: sample k random negatives, compute energy, select topk hardest (lowest energy).
    """
    device = mu_src.device
    src = pos_ei[0]
    dst_pos = pos_ei[1]
    P = src.size(0)
    if N is None:
        N = mu_dst.size(0)

    # Sample k random negatives per positive edge
    neg_pool = torch.randint(0, N, (P, k), device=device)
    # Avoid sampling the true positive
    neg_pool = torch.where(neg_pool == dst_pos.unsqueeze(1), (neg_pool + 1) % N, neg_pool)

    # Compute energy for all candidates
    mu_u = mu_src[src].unsqueeze(1)  # [P, 1, d]
    var_u = var_src[src].unsqueeze(1)  # [P, 1, d]
    mu_neg = mu_dst[neg_pool]  # [P, k, d]
    var_neg = var_dst[neg_pool]  # [P, k, d]

    d = mu_u.size(-1)
    ratio = var_u / var_neg
    trace = ratio.sum(dim=-1)
    delta = mu_neg - mu_u
    quad = (delta * delta / var_neg).sum(dim=-1)
    logdet = (torch.log(var_neg) - torch.log(var_u)).sum(dim=-1)
    E_neg_pool = 0.5 * (trace + quad - d + logdet)  # [P, k]

    # Select topk hardest (lowest energy = hardest negatives)
    topk_actual = min(topk, k)
    _, hard_idx = torch.topk(E_neg_pool, topk_actual, dim=1, largest=False)  # [P, topk]
    hard_negs = torch.gather(neg_pool, 1, hard_idx)  # [P, topk]

    return hard_negs

def _triplet_hinge_kl(E_pos, E_neg, margin=0.5):
    """Triplet loss with hinge margin"""
    hinge = F.relu(margin + E_pos.unsqueeze(1) - E_neg)   # [P, k]
    return hinge.mean()

def _train_epoch_stformer(model, optimizer, dataset, device, W, margin, k, topk, pos_cap):
    model.train()
    begin, end = dataset.get_range_by_split('train')
    t_start = begin + (W - 1)
    t_stop  = end - 1

    total_loss, steps = 0.0, 0
    ap_list, auc_list = [], []

    for t in tqdm(range(t_start, t_stop), leave=False):
        winB = dataset.snapshots[t - (W - 1): t + 1]
        if len(winB) != W:
            continue

        N = dataset.snapshots[0].node_feature.size(0)
        delta_seq = _delta_seq_from_winB(winB, device, N)

        # Encode spatial features with gradients enabled
        H_seq = []
        for s in winB:
            H_seq.append(model.encode_one_snapshot(s))
        H_seq = torch.stack(H_seq, dim=1)  # [N,W,d]

        z = model.temporal(H_seq, delta_seq=delta_seq)
        mu_src, var_src, mu_dst, var_dst = model.gaussian_params(z)
        total_nodes = mu_src.size(0)

        next_snapshot = dataset.snapshots[t + 1]
        pos_ei_full = next_snapshot.edge_index.to(device)
        P = pos_ei_full.size(1)
        if (pos_cap is not None) and (P > pos_cap):
            idx = torch.randperm(P, device=device)[:pos_cap]
            pos_ei = pos_ei_full[:, idx]
        else:
            pos_ei = pos_ei_full

        src = pos_ei[0]
        dst_pos = pos_ei[1]
        E_pos = model.kl_diag(mu_src[src], var_src[src], mu_dst[dst_pos], var_dst[dst_pos])

        # Mixed strategy: 70% hard negatives, 30% random negatives
        hard_ratio = 0.7
        num_hard = int(topk * hard_ratio)
        num_random = topk - num_hard

        # Hard negative mining
        hard_negs = _mine_hard_negatives(
            mu_src.detach(), mu_dst.detach(), var_src.detach(), var_dst.detach(),
            pos_ei, k=k, topk=num_hard, N=total_nodes
        )

        # Random negatives
        random_negs = torch.randint(0, N, (P, num_random), device=device)
        random_negs = torch.where(random_negs == dst_pos.unsqueeze(1), (random_negs + 1) % N, random_negs)

        # Combine hard and random negatives
        all_negs = torch.cat([hard_negs, random_negs], dim=1)  # [P, topk]

        # Compute energy for all negatives
        mu_neg = mu_dst[all_negs]
        var_neg = var_dst[all_negs]
        mu_u = mu_src[src].unsqueeze(1).expand_as(mu_neg)
        var_u = var_src[src].unsqueeze(1).expand_as(var_neg)
        d = mu_u.size(-1)
        ratio = var_u / var_neg
        trace = ratio.sum(dim=-1)
        delta = mu_neg - mu_u
        quad = (delta * delta / var_neg).sum(dim=-1)
        logdet = (torch.log(var_neg) - torch.log(var_u)).sum(dim=-1)
        E_neg = 0.5 * (trace + quad - d + logdet)

        triplet_loss = _triplet_hinge_kl(E_pos, E_neg, margin=margin)

        # Variance regularization - keep variance from collapsing or exploding
        var_reg_w = getattr(model, 'var_reg', 0.0)
        if var_reg_w > 0:
            logv_src = torch.log(var_src)
            logv_dst = torch.log(var_dst)
            var_reg_term = (logv_src.pow(2).mean() + logv_dst.pow(2).mean()) * 0.5
            loss = triplet_loss + var_reg_w * var_reg_term
        else:
            loss = triplet_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute metrics with random negatives for monitoring
        with torch.no_grad():
            neg_ei = negative_sampling(pos_ei_full, num_nodes=dataset.num_nodes).to(device)
            ei_all = torch.cat([pos_ei_full, neg_ei], dim=1)
            y = torch.cat([torch.ones(pos_ei_full.size(1)), torch.zeros(neg_ei.size(1))], dim=0).to(device)
            logits = model.score_edges(z, ei_all)
            roc_auc = compute_roc_auc(logits, y)
            ap = compute_ap(logits, y)

        total_loss += float(loss.detach())
        ap_list.append(ap)
        auc_list.append(roc_auc)
        steps += 1

    if steps == 0:
        return {"loss": 0.0, "ap": 0.0, "auc": 0.0}

    return {"loss": total_loss / steps, "ap": float(np.mean(ap_list)), "auc": float(np.mean(auc_list))}

@torch.no_grad()
def _evaluate_stformer(model, dataset, device, W, split):
    model.eval()
    logits, labels = _collect_logits_labels(model, dataset, device, W, split)
    if logits is None:
        return {"loss": 0.0, "ap": 0.0, "auc": 0.0}
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    auc = compute_roc_auc(logits, labels)
    ap = compute_ap(logits, labels)
    return {"loss": loss, "ap": ap, "auc": auc}

# =====================================================================================
# TOP-LEVEL DISPATCH
# =====================================================================================

def train(model, optimizer, dataset, n_epoch, patience, device, auto_scale_epochs=True, target_updates=4000):
    """
    Training with hard negative mining and proper Graph2Gauss framework
    """
    if getattr(model, 'is_stformer', False):
        W = getattr(model, 'window_size', 10)
        margin = getattr(model, 'margin', 0.5)
        k = 500  # Sample pool size
        topk = 200  # Hard negatives to keep
        pos_cap = 2000

        begin, end = dataset.get_range_by_split('train')
        steps_per_epoch = max(1, (end - begin) - W)
        n_epoch_eff = n_epoch
        if auto_scale_epochs:
            n_epoch_eff = 10

        best_ap = -1.0
        best_epoch = 0
        best_state_dict = None
        best_model_unchanged = 0

        for epoch in range(n_epoch_eff):
            start_time = time.time()

            train_metric = _train_epoch_stformer(
                model, optimizer, dataset, device,
                W=W, margin=margin, k=k, topk=topk, pos_cap=pos_cap
            )
            val_metric   = _evaluate_stformer(model, dataset, device, W, 'val')
            test_metric  = _evaluate_stformer(model, dataset, device, W, 'test')

            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{n_epoch_eff}, Time: {epoch_time:.2f}s')
            print(f"Train: loss: {train_metric['loss']:.4f}, roc_auc: {train_metric['auc']:.4f}, ap: {train_metric['ap']:.4f}")
            print(f"Valid: loss: {val_metric['loss']:.4f}, roc_auc: {val_metric['auc']:.4f}, ap: {val_metric['ap']:.4f}")
            print(f"Test : loss: {test_metric['loss']:.4f}, roc_auc: {test_metric['auc']:.4f}, ap: {test_metric['ap']:.4f}")
            print('======' * 20)

            # Update learning rate based on validation AP
            scheduler.step(val_metric['ap'])

            # Early stop on AP
            if val_metric['ap'] > best_ap:
                best_ap = val_metric['ap']
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                best_model_unchanged = 0
            else:
                best_model_unchanged += 1

            if best_model_unchanged >= patience:
                print(f'Saving Model At Epoch {best_epoch + 1}')
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        final_metric = _evaluate_stformer(model, dataset, device, W, 'test')
        print('Final Test Results: roc_auc: {:.4f}, ap: {:.4f}'.format(final_metric['auc'], final_metric['ap']))
        return final_metric['auc'], final_metric['ap']

avg_auc, avg_ap = train(model, optimizer, dataset, n_epoch=args.epochs, patience=5, device=device, auto_scale_epochs=True)
print(f'Average AUC: {avg_auc}, Average AP: {avg_ap}')
