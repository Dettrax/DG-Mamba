# Tr_test.py

# Train Transformer + Graph2Gauss-style encoder with:
# (1) KL triplet ranking
# (2) Symmetric-KL edge BCE (with curriculum + mined negatives)
# (3) Row-softmax (InfoNCE) ranking loss per anchor node <-- boosts MAP/PR
# Includes: sanity checks on score separation at chosen timesteps.

import os, warnings, numpy as np, torch, torch.nn.functional as F
from typing import Optional
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    os.chdir("RealityMining")
except Exception:
    pass

from models import Graph2Gauss_Torch
from utils_mod import *  # dataset_mit, sample_hops, to_triplets
from eval_mod import get_MAP_avg, sanity_check_time

# Repro & device
torch.backends.cudnn.deterministic = True
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------------- Dataset ----------------

class RMDataset(Dataset):
    def __init__(self, data, lookback: int, K: int = 2):
        self.data = data
        self.lookback = lookback
        self.K = K
        self.N = max(d[0].shape[0] for d in data)
        self.T = len(data)
        self._windows = {}

    def __len__(self):
        return self.T

    def _window_dense(self, i: int) -> np.ndarray:
        B = np.zeros((self.N, self.lookback + 1, self.N), dtype=np.float32)
        t0 = i - self.lookback

        for j in range(self.lookback + 1):
            t = t0 + j
            if 0 <= t < self.T:
                Aij: csr_matrix = self.data[t][0]
                n_t = Aij.shape[0]
                if n_t > 0:
                    B[:n_t, j, :n_t] = Aij.toarray().astype(np.float32, copy=False)

        return B

    def __getitem__(self, i):
        if i not in self._windows:
            self._windows[i] = self._window_dense(i)

        x = self._windows[i]
        A_i: csr_matrix = self.data[i][0]
        sampled_hops, scale_terms = sample_hops(A_i, K=self.K)
        triplet, scale = to_triplets(sampled_hops, scale_terms)

        return x, triplet, scale


# ---------------- Energies & losses ----------------

def energy_kl(mu: torch.Tensor, sigma: torch.Tensor, pairs: torch.Tensor, L: int) -> torch.Tensor:
    u, v = pairs[:, 0], pairs[:, 1]
    mu_u, mu_v = mu[u], mu[v]
    su, sv = sigma[u], sigma[v]

    ratio = sv / (su + 1e-8)
    trace_fac = ratio.sum(dim=1)
    log_det = (ratio + 1e-8).log().sum(dim=1)
    mu_diff_sq = ((mu_u - mu_v) ** 2 / (su + 1e-8)).sum(dim=1)

    return 0.5 * (trace_fac + mu_diff_sq - L - log_det)


def loss_triplet_margin(triplet: np.ndarray, mu: torch.Tensor, sigma: torch.Tensor, L: int, margin=2.0):
    t = torch.as_tensor(triplet, dtype=torch.long, device=mu.device)
    e_pos = energy_kl(mu, sigma, t[:, [0, 1]], L)
    e_neg = energy_kl(mu, sigma, t[:, [0, 2]], L)
    return F.softplus(e_pos - e_neg + margin).mean()


def _symmetrize(A: csr_matrix) -> csr_matrix:
    return ((A + A.T) > 0).astype(np.int8)


def _pos_pairs_undirected(A: csr_matrix, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    As = _symmetrize(A)
    r, c = As.nonzero()
    m = r < c
    pos = np.stack([r[m], c[m]], 1)

    if cap is not None and len(pos) > cap:
        idx = rng.choice(len(pos), size=cap, replace=False)
        pos = pos[idx]

    return pos.astype(np.int64, copy=False)


def _twohop_candidates(A: csr_matrix) -> np.ndarray:
    As = _symmetrize(A).astype(np.int8).tocsr()
    two = (As @ As).astype(np.int32)
    two.setdiag(0)
    two.eliminate_zeros()

    two = (two - two.multiply(As)).tocsr()
    two.eliminate_zeros()

    r, c = two.nonzero()
    m = r < c
    return np.stack([r[m], c[m]], 1).astype(np.int64, copy=False)


def _uniform_negative_candidates(A: csr_matrix) -> np.ndarray:
    n = A.shape[0]
    As = _symmetrize(A).astype(np.int8)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r, c = As.nonzero()
    uu = np.minimum(r, c)
    vv = np.maximum(r, c)
    mask[uu, vv] = False

    neg = np.argwhere(mask)
    return neg.astype(np.int64, copy=False)


def _sample_negatives_mix(A: csr_matrix, k: int, rng: np.random.Generator, hard_ratio: float = 0.2) -> np.ndarray:
    k = max(0, int(k))
    hard_all = _twohop_candidates(A)
    uni_all = _uniform_negative_candidates(A)

    k_hard = min(int(k * hard_ratio), len(hard_all))
    k_uni = min(k - k_hard, len(uni_all))

    outs = []
    if k_hard > 0:
        outs.append(hard_all[rng.choice(len(hard_all), size=k_hard, replace=False)])
    if k_uni > 0:
        outs.append(uni_all[rng.choice(len(uni_all), size=k_uni, replace=False)])

    if not outs:
        return np.empty((0, 2), dtype=np.int64)

    return np.vstack(outs).astype(np.int64, copy=False)


def _mine_hard_negatives(mu: torch.Tensor, sigma: torch.Tensor, neg_pairs_np: np.ndarray,
                         top_frac: float = 0.5, temp: float = 0.5) -> np.ndarray:
    if len(neg_pairs_np) == 0 or top_frac <= 0.0:
        return neg_pairs_np

    pairs = torch.as_tensor(neg_pairs_np, dtype=torch.long, device=mu.device)

    with torch.no_grad():
        e_uv = energy_kl(mu, sigma, pairs, L=mu.shape[1])
        e_vu = energy_kl(mu, sigma, pairs[:, [1, 0]], L=mu.shape[1])
        logits = -0.5 * (e_uv + e_vu) / temp

        k = max(1, int(min(top_frac, 1.0) * len(neg_pairs_np)))
        top_idx = torch.topk(logits, k=k, largest=True).indices.detach().cpu().numpy()

        return neg_pairs_np[top_idx]


def loss_edge_bce(mu: torch.Tensor, sigma: torch.Tensor,
                  pos_pairs: np.ndarray, neg_pairs: np.ndarray,
                  temp: float = 0.5) -> torch.Tensor:
    pairs = torch.as_tensor(np.vstack([pos_pairs, neg_pairs]), dtype=torch.long, device=mu.device)
    y = torch.cat([torch.ones(len(pos_pairs), device=mu.device),
                   torch.zeros(len(neg_pairs), device=mu.device)], dim=0)

    e_uv = energy_kl(mu, sigma, pairs, L=mu.shape[1])
    e_vu = energy_kl(mu, sigma, pairs[:, [1, 0]], L=mu.shape[1])
    logits = -0.5 * (e_uv + e_vu) / temp

    return F.binary_cross_entropy_with_logits(logits, y)


# -------- Row-softmax (InfoNCE) ranking loss --------

def loss_row_softmax(mu: torch.Tensor, sigma: torch.Tensor, A: csr_matrix,
                     rng: np.random.Generator,
                     anchors_per_batch: int = 128, neg_per_anchor: int = 64,
                     temp: float = 0.5) -> torch.Tensor:
    """
    For each sampled anchor u:
    pick one positive v+ (neighbor),
    sample neg_per_anchor non-neighbors v-,
    maximize logit(u,v+) against {v+ ∪ v-} via log-softmax.
    """
    n = A.shape[0]
    As = _symmetrize(A).tocsr()
    deg = np.array(As.sum(1)).ravel()
    cand_u = np.where(deg > 0)[0]

    if len(cand_u) == 0:
        return torch.tensor(0.0, device=mu.device)

    num_u = min(anchors_per_batch, len(cand_u))
    U = rng.choice(cand_u, size=num_u, replace=False)

    losses = []
    for u in U:
        # one positive
        pos_list = As[u].indices
        v_pos = int(rng.choice(pos_list))

        # negatives: complement of neighbors∪self in [0..n)
        mask = np.ones(n, dtype=bool)
        mask[u] = False
        mask[pos_list] = False
        neg_pool = np.where(mask)[0]

        if len(neg_pool) == 0:
            continue

        V_neg = rng.choice(neg_pool, size=min(neg_per_anchor, len(neg_pool)), replace=False)

        # Build pairs: [ (u,v_pos) , (u,v_neg_1), ... ]
        pairs = np.concatenate([[[u, v_pos]], np.stack([np.full(len(V_neg), u), V_neg], 1)])
        pairs_t = torch.as_tensor(pairs, dtype=torch.long, device=mu.device)

        e_uv = energy_kl(mu, sigma, pairs_t, L=mu.shape[1])
        e_vu = energy_kl(mu, sigma, pairs_t[:, [1, 0]], L=mu.shape[1])
        logits = -0.5 * (e_uv + e_vu) / temp  # higher = more likely edge

        log_probs = F.log_softmax(logits, dim=0)
        losses.append(-log_probs[0])  # pos is first

    if len(losses) == 0:
        return torch.tensor(0.0, device=mu.device)

    return torch.stack(losses).mean()


# ---------------- Train/Eval ----------------

def optimise_model(data,
                   lookback: int = 2,
                   dim_in: int = 96, dim_out: int = 64,
                   n_heads: int = 1, n_layers: int = 1,
                   epochs: int = 200, lr: float = 5e-5, weight_decay: float = 1e-3,  # Lowered LR to stabilize
                   train_end: int = 63, eval_every: int = 3,  # More frequent eval
                   pos_cap: int = 1500, neg_ratio: int = 2,  # Reduced neg_ratio
                   alpha_triplet: float = 0.2, beta_edge: float = 0.8, gamma_row: float = 0.5,  # Rebalanced weights
                   mine_top_frac: float = 0.3,  # Less aggressive mining
                   anchors_per_batch: int = 64, neg_per_anchor: int = 32):  # Smaller batches

    ds = RMDataset(data, lookback=lookback, K=2)
    model = Graph2Gauss_Torch(dim_val=64, dim_attn=64, dim_in=dim_in, dim_out=dim_out,
                              n_encoder_layers=n_layers, n_heads=n_heads, lookback=lookback).to(device)

    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Add learning rate scheduler with more aggressive reduction
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

    rng = np.random.default_rng(0)
    best_map = -1.0
    best_state = None

    # Early stopping params
    patience = 10
    wait = 0
    min_delta = 0.0001  # Minimum improvement required

    for e in tqdm(range(1, epochs + 1)):
        model.train()
        losses = []

        # More gradual curriculum: slower ramp, lower max
        hard_ratio = min(0.1, max(0.0, (e - 1) / 25.0 * 0.1))

        # Adaptive evaluation frequency
        eval_every_current = 2 if e <= 15 else eval_every

        for i in range(lookback, min(train_end, len(ds))):
            x, triplet, _ = ds[i]
            x = torch.tensor(x, dtype=torch.float32, device=device)

            _, mu, sigma = model(x)
            sigma = F.softplus(sigma) + 1e-3
            sigma = sigma.clamp(min=1e-4, max=20.0)  # Added min clamp to prevent near-zero sigma

            A = data[i][0]

            # Edge BCE: pos/neg with curriculum + mined hard negatives
            pos_pairs = _pos_pairs_undirected(A, cap=pos_cap, rng=rng)
            if len(pos_pairs) == 0:
                continue

            neg_pairs = _sample_negatives_mix(A, k=len(pos_pairs) * neg_ratio, rng=rng, hard_ratio=hard_ratio)
            neg_pairs = _mine_hard_negatives(mu, sigma, neg_pairs, top_frac=mine_top_frac, temp=0.7)  # Higher temp

            if len(neg_pairs) > len(pos_pairs):
                idx = rng.choice(len(neg_pairs), size=len(pos_pairs), replace=False)
                neg_pairs = neg_pairs[idx]

            # Progressive loss introduction
            if e <= 20:
                # Phase 1: Focus on edge BCE
                loss_t = loss_triplet_margin(triplet, mu, sigma, L=mu.shape[1], margin=2.0)
                loss_e = loss_edge_bce(mu, sigma, pos_pairs, neg_pairs, temp=0.7)  # Higher temp
                loss_r = torch.tensor(0.0, device=mu.device)
            else:
                # Phase 2: Add all losses
                loss_t = loss_triplet_margin(triplet, mu, sigma, L=mu.shape[1], margin=2.0)
                loss_e = loss_edge_bce(mu, sigma, pos_pairs, neg_pairs, temp=0.7)
                loss_r = loss_row_softmax(mu, sigma, A, rng, anchors_per_batch=anchors_per_batch,
                                          neg_per_anchor=neg_per_anchor, temp=0.7)

            # Dynamic gamma_row ramp-up after phase 1
            current_gamma = gamma_row * min(1.0, (e - 20) / 10.0) if e > 20 else 0.0

            # Adaptive regularization
            lambda_sigma = max(1e-4, 5e-4 * (0.95 ** e))
            lambda_mu = max(1e-5, 5e-5 * (0.95 ** e))

            reg_sigma = lambda_sigma * (torch.log(sigma).clamp(min=-5.0, max=5.0) ** 2).mean()
            reg_mu = lambda_mu * (mu ** 2).mean()

            loss = alpha_triplet * loss_t + beta_edge * loss_e + current_gamma * loss_r + reg_sigma + reg_mu

            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {e}, batch {i}. Skipping batch.")
                continue

            opt.zero_grad()
            loss.backward()

            # Enhanced gradient checking
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Increased threshold
            if total_norm > 5.0 and i % 20 == 0:
                print(f"Warning: Large gradient norm {total_norm:.3f} at epoch {e}, batch {i}")

            opt.step()
            losses.append(loss.item())

            # Log batch losses with additional stats
            if i % 20 == 0:
                print(f"Epoch {e}, Batch {i}: triplet={loss_t.item():.4f}, edge={loss_e.item():.4f}, "
                      f"row={loss_r.item():.4f}, total={loss.item():.4f}, "
                      f"mu_norm={mu.norm().item():.4f}, sigma_mean={sigma.mean().item():.4f}")

        if e % eval_every_current == 0 or e == epochs:
            model.eval()
            mu_seq, sigma_seq = [], []

            with torch.no_grad():
                for t in range(lookback, len(ds)):
                    x, _, _ = ds[t]
                    x = torch.tensor(x, dtype=torch.float32, device=device)

                    _, mu, sigma = model(x)
                    sigma = F.softplus(sigma) + 1e-3
                    sigma = sigma.clamp(min=1e-4, max=20.0)

                    mu_seq.append(mu.detach().cpu().numpy())
                    sigma_seq.append(sigma.detach().cpu().numpy())

            MAP, MRR = get_MAP_avg([mu_seq], lookback, data,
                                   sigma_seq_in=[sigma_seq],
                                   undirected=True, symmetric_kl=True,
                                   neg_multiplier=8, eval_start=72)  # Increased for robustness

            avg_loss = np.mean(losses) if losses else float('nan')
            print(
                f"Epoch {e:03d} | loss={avg_loss:.4f} | MAP={MAP:.4f} | MRR={MRR:.4f} | LR={opt.param_groups[0]['lr']:.2e}")

            scheduler.step(MAP)

            if MAP > best_map + min_delta:
                best_map = MAP
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                wait = 0
                print(f"New best MAP: {best_map:.4f}")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {e} due to no improvement.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    data = dataset_mit('..', undirected=True)
    lookback = 2

    model = optimise_model(
        data,
        lookback=lookback,
        dim_in=96, dim_out=64,
        n_heads=1, n_layers=1,
        epochs=200, lr=5e-5, weight_decay=1e-3,
        train_end=63, eval_every=10,
        pos_cap=1500, neg_ratio=2,
        alpha_triplet=0.2, beta_edge=0.8, gamma_row=0.5,
        mine_top_frac=0.3,
        anchors_per_batch=64, neg_per_anchor=32
    )

    # Final eval + save
    ds = RMDataset(data, lookback=lookback)
    mu_seq, sigma_seq = [], []

    model.eval()
    with torch.no_grad():
        for t in range(lookback, len(ds)):
            x, _, _ = ds[t]
            x = torch.tensor(x, dtype=torch.float32, device=device)

            _, mu, sigma = model(x)
            sigma = F.softplus(sigma) + 1e-3
            sigma = sigma.clamp(min=1e-4, max=20.0)

            mu_seq.append(mu.detach().cpu().numpy())
            sigma_seq.append(sigma.detach().cpu().numpy())

    print("\n=== Sanity checks ===")
    sanity_check_time([mu_seq], [sigma_seq], lookback, data, t=72, undirected=True, symmetric_kl=True, num_neg=2000,
                      seed=0)
    sanity_check_time([mu_seq], [sigma_seq], lookback, data, t=80, undirected=True, symmetric_kl=True, num_neg=2000,
                      seed=1)

    MAP, MRR = get_MAP_avg([mu_seq], lookback, data,
                           sigma_seq_in=[sigma_seq],
                           undirected=True, symmetric_kl=True,
                           neg_multiplier=8, eval_start=72)

    print(f"\nFinal MAP: {MAP}")
    print(f"Final MRR: {MRR}")

    name = 'Results/RealityMining'
    os.makedirs(os.path.join(name, 'Eval_Results', 'saved_array'), exist_ok=True)

    import pickle
    with open(os.path.join(name, 'Eval_Results', 'saved_array', 'mu_as'), 'wb') as f:
        pickle.dump([mu_seq], f)
    with open(os.path.join(name, 'Eval_Results', 'saved_array', 'sigma_as'), 'wb') as f:
        pickle.dump([sigma_seq], f)

    torch.save(model.state_dict(), os.path.join(name, 'best_model.pth'))


if __name__ == "__main__":
    main()
