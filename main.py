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
parser.add_argument('--window_size', type=int,   default=5)
parser.add_argument('--d_model',     type=int,   default=32)
parser.add_argument('--device',      type=str,   default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--epochs',      type=int,   default=10)
parser.add_argument('--patience',    type=int,   default=5)
parser.add_argument('--lr',          type=float, default=3e-4)
parser.add_argument('--weight_decay',type=float, default=1e-5)
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--use_pos_emb', action='store_true',default=True, help='add positional tokens before Mamba')
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(args.device)

# ---------------- data ----------------
dataset = get_dataset(args)
print(f"Total Snapshots: {len(dataset.snapshots)}, Total Nodes: {dataset.num_nodes}, Total Edges: {dataset.num_edges}")

model = STFormerGCN(
    in_dim=128,
    gcn_dim=128,
    d_model=args.d_model,
    proj_dim=args.d_model,
    num_nodes=dataset.num_nodes,
    use_id_emb=True,
    nhead=4,
    num_tlayers=3,
    gcn_layers=2,
    dropout=0.3,
    max_len=args.window_size,
    sigma_floor=1e-4,
    temporal_type="mamba"  # if mamba-ssm not available, set to "transformer"
).to(device)

# training knobs for KL triplet
model.window_size = args.window_size
model.margin = 0.2
model.triplet_k = 200
model.triplet_pool = 4000
model.block_h = 3
model.pos_cap = 1500
model.mine_chunk = 512
model.use_amp_mine = True
model.bce_weight = 1.0
model.sym_kl = False
model.var_reg = 0.0

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
# ORIGINAL DY* PATH (kept for completeness; unused for STFormerGCN)
# =====================================================================================

def train_step(model, optimizer, dataset, device):
    start, end = dataset.get_range_by_split('train')
    train_loss = 0
    count = 0
    train_ap = []
    train_auc = []

    model.train()
    if hasattr(model, 'reset_memory'):
        model.reset_memory()
    init_state = torch.zeros(
        dataset.snapshots[0].node_feature.shape[0],
        getattr(model, 'hidden_dim', dataset.snapshots[0].node_feature.shape[1])
    )

    for t in tqdm(range(start, end - 1), leave=False):
        current_snapshot = dataset.snapshots[t]
        next_snapshot = dataset.snapshots[t + 1]

        x = current_snapshot.node_feature.to(device)
        init_state = init_state.to(device)
        edge_index = current_snapshot.edge_index.to(device)
        edge_feature = getattr(current_snapshot, 'edge_time', None)
        if edge_feature is None:
            edge_feature = torch.ones(current_snapshot.edge_index.size(1), device=device)
        else:
            edge_feature = edge_feature.to(device)

        negative_edge_index = negative_sampling(next_snapshot.edge_index, num_nodes=dataset.num_nodes)

        edge_label_index = torch.cat([next_snapshot.edge_index, negative_edge_index], dim=-1).to(device)
        label = torch.cat([
            torch.ones(next_snapshot.edge_index.shape[1]),
            torch.zeros(negative_edge_index.shape[1])
        ]).to(device)

        prediction, new_state = model(x, edge_index, edge_label_index, edge_feature, init_state)

        loss = compute_loss(prediction, label)

        init_state = new_state.detach().cpu().clone()

        roc_auc = compute_roc_auc(prediction, label)
        ap = compute_ap(prediction, label)

        train_ap.append(ap)
        train_auc.append(roc_auc)

        train_loss += loss
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / count

    train_metric = {
        'loss': train_loss,
        'ap': float(np.mean(train_ap)),
        'auc': float(np.mean(train_auc)),
    }
    return train_metric, init_state, getattr(model, 'memory_edge_index', []), getattr(model, 'memory_edge_time', [])

def evaluate_step(model, dataset, previous_state, previous_memory_edge_index, previous_memory_edge_time, spilt, device):
    start, end = dataset.get_range_by_split(spilt)
    test_loss = 0
    count = 0
    test_ap = []
    test_auc = []

    init_state = previous_state.clone()

    if hasattr(model, 'read_memory'):
        model.read_memory(copy.deepcopy(previous_memory_edge_index), copy.deepcopy(previous_memory_edge_time))

    model.eval()
    with torch.no_grad():
        for t in tqdm(range(start - 1, end - 1), leave=False):
            current_snapshot = dataset.snapshots[t]
            next_snapshot = dataset.snapshots[t + 1]

            x = current_snapshot.node_feature.to(device)
            init_state = init_state.to(device)
            edge_index = current_snapshot.edge_index.to(device)
            edge_feature = getattr(current_snapshot, 'edge_time', None)
            if edge_feature is None:
                edge_feature = torch.ones(current_snapshot.edge_index.size(1), device=device)
            else:
                edge_feature = edge_feature.to(device)

            edge_label_index = next_snapshot.edge_label_index.to(device) if hasattr(next_snapshot, 'edge_label_index') else \
                torch.cat([next_snapshot.edge_index, negative_sampling(next_snapshot.edge_index, num_nodes=dataset.num_nodes)], dim=-1).to(device)
            label = next_snapshot.edge_label.to(device) if hasattr(next_snapshot, 'edge_label') else \
                torch.cat([torch.ones(next_snapshot.edge_index.shape[1]),
                           torch.zeros(next_snapshot.edge_index.shape[1])]).to(device)

            prediction, new_state = model(x, edge_index, edge_label_index, edge_feature, init_state)

            loss = compute_loss(prediction, label)

            init_state = new_state.detach().cpu().clone()

            roc_auc = compute_roc_auc(prediction, label)
            ap = compute_ap(prediction, label)

            test_loss += loss
            count += 1
            test_ap.append(ap)
            test_auc.append(roc_auc)

    test_loss = test_loss / count

    test_metric = {
        'loss': test_loss,
        'ap': float(np.mean(test_ap)),
        'auc': float(np.mean(test_auc)),
    }

    if spilt == 'val':
        return test_metric, init_state, getattr(model, 'memory_edge_index', []), getattr(model, 'memory_edge_time', [])
    else:
        return test_metric

# =====================================================================================
# STFormerGCN path â€” KL energy, leak-free, consistent time-encoding
# =====================================================================================

def _delta_seq_from_winB(winB, device, N):
    """
    Strictly past time encoding:
      Î”t per step computed relative to the last step in the window (winB[-1]).
    Returns: [N, W] tensor (days).
    """
    W = len(winB)
    has_ts = all(hasattr(s, 'snapshot_ts') for s in winB)
    if has_ts:
        # make a device tensor first, then take the last entry on the SAME device
        win_ts = torch.tensor([float(s.snapshot_ts) for s in winB], device=device, dtype=torch.float32)
        ref_ts = win_ts[-1]
        delta_days = (ref_ts - win_ts) / 86400.0
        delta_days = torch.clamp(delta_days, min=0.0)
    else:
        delta_days = torch.arange(W - 1, -1, -1, device=device, dtype=torch.float32)
    return delta_days.unsqueeze(0).expand(N, -1)  # [N, W]

@torch.no_grad()
def _collect_logits_labels(model, dataset, device, W, split):
    """
    For AP/AUC reporting only (pos vs random negs).
    Uses window-B embeddings with strictly past Î”t.
    """
    begin, end = dataset.get_range_by_split(split)
    # IMPORTANT: align with train start
    t_start = begin + (W - 1)
    t_stop = end - 1
    logits_all, labels_all = [], []

    for t in range(t_start, t_stop):
        winB = dataset.snapshots[t - (W - 1): t + 1]
        if len(winB) != W:
            continue

        N = dataset.snapshots[0].node_feature.size(0)
        delta_seq = _delta_seq_from_winB(winB, device, N)
        z = model.encode_window(winB, delta_seq=delta_seq)  # [N, d]

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
def _select_random_negatives(mu_src, mu_dst, pos_ei, k=200, N=None):
    device = mu_src.device
    src = pos_ei[0]
    dst = pos_ei[1]
    P = src.size(0)
    if N is None:
        N = mu_dst.size(0)

    neg = torch.randint(0, N, (P, k), device=device)
    neg = torch.where(neg == dst.unsqueeze(1), (neg + 1) % N, neg)
    have_semihard = torch.zeros(P, dtype=torch.bool, device=device)
    return neg, have_semihard

def _triplet_hinge_kl(E_pos, E_neg, margin=0.2):
    hinge = F.relu(margin + E_pos.unsqueeze(1) - E_neg)   # [P, k]
    return hinge.mean()

def _train_epoch_stformer(model, optimizer, dataset, device, W, margin, k, pool, block_h,
                          pos_cap, mine_chunk, use_amp_mine):
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

        z = model.encode_window(winB, delta_seq=delta_seq)
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

        neg_targets, have_semihard = _select_random_negatives(
            mu_src.detach(), mu_dst.detach(), pos_ei, k=k, N=total_nodes
        )

        src = pos_ei[0]
        dst_pos = pos_ei[1]
        E_pos = model.kl_diag(mu_src[src], var_src[src], mu_dst[dst_pos], var_dst[dst_pos])  # [P]

        mu_neg = mu_dst[neg_targets]   # [P,k,d]
        var_neg= var_dst[neg_targets]  # [P,k,d]
        mu_u   = mu_src[src].unsqueeze(1).expand_as(mu_neg)
        var_u  = var_src[src].unsqueeze(1).expand_as(var_neg)
        d = mu_u.size(-1)
        ratio = var_u / var_neg
        trace = ratio.sum(dim=-1)
        delta = mu_neg - mu_u
        quad  = (delta * delta / var_neg).sum(dim=-1)
        logdet= (torch.log(var_neg) - torch.log(var_u)).sum(dim=-1)
        E_neg = 0.5 * (trace + quad - d + logdet)

        triplet_loss = _triplet_hinge_kl(E_pos, E_neg, margin=margin)

        aux_weight = getattr(model, 'bce_weight', 0.0)
        var_reg_w = getattr(model, 'var_reg', 0.0)
        if aux_weight > 0:
            logit_scale = model.logit_scale.clamp(0.05, 50.0)
            logits_pos = -E_pos * logit_scale
            logits_neg = -E_neg * logit_scale
            logits_all = torch.cat([logits_pos, logits_neg.view(-1)], dim=0)
            labels_all = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg.view(-1))], dim=0)
            bce_loss = F.binary_cross_entropy_with_logits(logits_all, labels_all)
            loss = triplet_loss + aux_weight * bce_loss
        else:
            bce_loss = torch.tensor(0.0, device=device)
            loss = triplet_loss
        if var_reg_w > 0:
            _, var_src_full, _, var_dst_full = model.gaussian_params(z.detach())
            logv_src = torch.log(var_src_full)
            logv_dst = torch.log(var_dst_full)
            var_reg_term = (logv_src.pow(2).mean() + logv_dst.pow(2).mean()) * 0.5
            loss = loss + var_reg_w * var_reg_term

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

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
    If model.is_stformer == True, use the STFormerGCN path (KL triplet, leak-free).
    Else, keep the original Dy* path.
    """
    if getattr(model, 'is_stformer', False):
        W            = getattr(model, 'window_size', 10)
        margin       = getattr(model, 'margin', 0.2)
        k            = getattr(model, 'triplet_k', 200)
        pool         = getattr(model, 'triplet_pool', 4000)
        block_h      = getattr(model, 'block_h', 3)
        pos_cap      = getattr(model, 'pos_cap', 1500)
        mine_chunk   = getattr(model, 'mine_chunk', 512)
        use_amp_mine = getattr(model, 'use_amp_mine', True)

        # keep updates roughly constant across W (optional but helpful)
        begin, end = dataset.get_range_by_split('train')
        steps_per_epoch = max(1, (end - begin) - W)
        n_epoch_eff = n_epoch
        if auto_scale_epochs:
            n_epoch_eff = max(n_epoch, math.ceil(target_updates / steps_per_epoch))

        best_ap = -1.0
        best_epoch = 0
        best_state_dict = None
        best_model_unchanged = 0

        for epoch in range(n_epoch_eff):
            start_time = time.time()

            train_metric = _train_epoch_stformer(
                model, optimizer, dataset, device,
                W=W, margin=margin, k=k, pool=pool, block_h=block_h,
                pos_cap=pos_cap, mine_chunk=mine_chunk, use_amp_mine=use_amp_mine
            )
            val_metric   = _evaluate_stformer(model, dataset, device, W, 'val')
            test_metric  = _evaluate_stformer(model, dataset, device, W, 'test')

            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{n_epoch_eff}, Time: {epoch_time:.2f}s')
            print(f"Train: loss: {train_metric['loss']:.4f}, roc_auc: {train_metric['auc']:.4f}, ap: {train_metric['ap']:.4f}")
            print(f"Valid: loss: {val_metric['loss']:.4f}, roc_auc: {val_metric['auc']:.4f}, ap: {val_metric['ap']:.4f}")
            print(f"Test : loss: {test_metric['loss']:.4f}, roc_auc: {test_metric['auc']:.4f}, ap: {test_metric['ap']:.4f}")
            print('======' * 20)

            # Early stop on AP (your target)
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

    # default Dy* path
    best_model = copy.deepcopy(model.state_dict())
    best_model_unchanged = 0
    best_ap = -1.0
    best_epoch = 0
    best_state = None
    best_memory_edge_index = []
    best_memory_edge_time = []

    for epoch in range(n_epoch):
        start_time = time.time()
        train_metric, train_state, train_memory_edge_index, train_memory_edge_time = train_step(model, optimizer, dataset, device)
        val_metric, val_state, val_memory_edge_index, val_memory_edge_time = evaluate_step(
            model, dataset, train_state, train_memory_edge_index, train_memory_edge_time, 'val', device
        )
        test_metric = evaluate_step(model, dataset, val_state, val_memory_edge_index, val_memory_edge_time, 'test', device)
        epoch_time = time.time() - start_time

        print('Epoch {}, Time: {}s'.format(epoch + 1, epoch_time))
        print('Train: loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(train_metric['loss'], train_metric['auc'], train_metric['ap']))
        print('Valid: loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(val_metric['loss'], val_metric['auc'], val_metric['ap']))
        print('Test : loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(test_metric['loss'], test_metric['auc'], test_metric['ap']))
        print('======' * 20)

        if val_metric['ap'] > best_ap:
            best_ap = val_metric['ap']
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_state = val_state
            best_memory_edge_index = copy.deepcopy(val_memory_edge_index)
            best_memory_edge_time = copy.deepcopy(val_memory_edge_time)
            best_model_unchanged = 0
        else:
            best_model_unchanged += 1

        if best_model_unchanged >= patience:
            print('Saving Model At Epoch {}'.format(best_epoch + 1))
            break

    model.load_state_dict(best_model)
    final_metric = evaluate_step(model, dataset, best_state, best_memory_edge_index, best_memory_edge_time, 'test', device)
    print('Final Test Results: roc_auc: {:.4f}, ap: {:.4f}'.format(final_metric['auc'], final_metric['ap']))
    return final_metric['auc'], final_metric['ap']

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

avg_auc, avg_ap = train(model, optimizer, dataset, n_epoch=50, patience=10, device=device, auto_scale_epochs=True)
print(f'Average AUC: {avg_auc}, Average AP: {avg_ap}')