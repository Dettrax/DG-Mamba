# eval_mod.py

"""
MAP/MRR evaluation + sanity checks for temporal graph embeddings (Graph2Gauss-style).

Key defaults for RealityMining:
- undirected=True
- symmetric_kl=True
- neg_multiplier=6 # moderate difficulty; avoids prior-dominated AP

Exposed API:
- get_MAP_avg(..) -> (MAP_avg, MRR_avg)
- sanity_check_time(..) -> dict of metrics and prints a small report
- _scores_from_embeddings(..) # used by sanity check or external scripts
"""

from typing import List, Optional, Sequence, Tuple, Dict, Any
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-14


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _energy_kl_pairs(mu: np.ndarray, sigma: np.ndarray, pairs: np.ndarray, L: int) -> np.ndarray:
    mu_u = mu[pairs[:, 0]]
    mu_v = mu[pairs[:, 1]]
    su = sigma[pairs[:, 0]]
    sv = sigma[pairs[:, 1]]

    ratio = sv / (su + EPS)
    trace_fac = ratio.sum(axis=1)
    log_det = np.log(ratio + EPS).sum(axis=1)
    mu_diff_sq = ((mu_u - mu_v) ** 2 / (su + EPS)).sum(axis=1)

    return 0.5 * (trace_fac + mu_diff_sq - L - log_det)


def _scores_from_embeddings(mu: np.ndarray,
                            sigma: Optional[np.ndarray],
                            pairs: np.ndarray,
                            symmetric_kl: bool = True) -> np.ndarray:
    """
    Higher score => more likely edge.
    If sigma is None: uses -0.5*||mu_u - mu_v||^2
    Else: uses -KL (symmetric if symmetric_kl=True)
    """
    mu = _to_numpy(mu).astype(np.float32, copy=False)

    if sigma is None:
        mu_u = mu[pairs[:, 0]]
        mu_v = mu[pairs[:, 1]]
        return -0.5 * ((mu_u - mu_v) ** 2).sum(axis=1)

    sigma = _to_numpy(sigma).astype(np.float32, copy=False)
    L = mu.shape[1]

    e_uv = _energy_kl_pairs(mu, sigma, pairs, L)
    if not symmetric_kl:
        return -e_uv

    e_vu = _energy_kl_pairs(mu, sigma, pairs[:, [1, 0]], L)
    return -0.5 * (e_uv + e_vu)


def _sample_negatives_directed(A: csr_matrix, k: int, rng: np.random.Generator) -> np.ndarray:
    n = A.shape[0]
    k = max(0, int(k))

    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    r, c = A.nonzero()
    mask[r, c] = False

    neg = np.argwhere(mask)
    if neg.shape[0] == 0 or k == 0:
        return np.empty((0, 2), dtype=np.int64)

    idx = rng.choice(neg.shape[0], size=min(k, neg.shape[0]), replace=False)
    return neg[idx].astype(np.int64, copy=False)


def _sample_negatives_undirected(A: csr_matrix, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Undirected negatives: sample only i < j pairs from non-edges
    """
    n = A.shape[0]
    k = max(0, int(k))

    As = ((A + A.T) > 0).astype(np.int8)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    r, c = As.nonzero()
    uu = np.minimum(r, c)
    vv = np.maximum(r, c)
    mask[uu, vv] = False

    neg = np.argwhere(mask)
    if neg.shape[0] == 0 or k == 0:
        return np.empty((0, 2), dtype=np.int64)

    idx = rng.choice(neg.shape[0], size=min(k, neg.shape[0]), replace=False)
    return neg[idx].astype(np.int64, copy=False)


def _row_mrr(pred_row: np.ndarray, true_row: np.ndarray) -> float:
    if true_row.sum() == 0:
        return np.nan

    order = np.argsort(-pred_row)
    ranks = np.nonzero(true_row[order].astype(bool))[0] + 1
    return (1.0 / ranks).mean()


def _eval_single_time(mu_t: np.ndarray,
                      sigma_t: Optional[np.ndarray],
                      A_t: csr_matrix,
                      neg_multiplier: int,
                      rng: np.random.Generator,
                      undirected: bool = True,
                      symmetric_kl: bool = True) -> Tuple[Optional[float], Optional[float]]:
    n = A_t.shape[0]

    if undirected:
        As = ((A_t + A_t.T) > 0).astype(np.int8)
        ur, uc = As.nonzero()
        mask = ur < uc
        pos_pairs = np.stack([ur[mask], uc[mask]], axis=1)
        num_pos = pos_pairs.shape[0]
        total_pairs = n * (n - 1) // 2
        sampler = _sample_negatives_undirected
    else:
        r, c = A_t.nonzero()
        mask = r != c
        pos_pairs = np.stack([r[mask], c[mask]], axis=1)
        num_pos = pos_pairs.shape[0]
        total_pairs = n * (n - 1)
        sampler = _sample_negatives_directed

    if num_pos == 0:
        return None, None

    num_neg_avail = total_pairs - num_pos
    if num_neg_avail <= 0:
        return None, None

    num_neg = min(n * neg_multiplier, num_neg_avail)
    neg_pairs = sampler(A_t, num_neg, rng)

    pairs = np.vstack([pos_pairs, neg_pairs])
    labels = np.concatenate([
        np.ones(len(pos_pairs), dtype=np.int32),
        np.zeros(len(neg_pairs), dtype=np.int32)
    ])

    scores = _scores_from_embeddings(mu_t, sigma_t, pairs, symmetric_kl=symmetric_kl)

    try:
        ap = float(average_precision_score(labels, scores))
    except Exception:
        ap = None

    pred = np.zeros((n, n), dtype=np.float32)
    truth = np.zeros((n, n), dtype=np.int8)
    pred[pairs[:, 0], pairs[:, 1]] = scores
    truth[pairs[:, 0], pairs[:, 1]] = labels

    row_mrrs = []
    for i in range(n):
        if truth[i].sum() > 0:
            row_mrrs.append(_row_mrr(pred[i], truth[i]))

    mrr = float(np.nanmean(row_mrrs)) if row_mrrs else None
    return ap, mrr


def get_MAP_avg(mu_seq_in: Sequence,
                lookback: int,
                data: Sequence,
                sigma_seq_in: Optional[Sequence] = None,
                eval_start: Optional[int] = 72,
                eval_end: Optional[int] = None,
                neg_multiplier: int = 6,
                undirected: bool = True,
                symmetric_kl: bool = True,
                seed: int = 5) -> Tuple[float, float]:
    """
    Average AP (MAP proxy) and MRR over timesteps.
    """
    mu_seq = mu_seq_in[0] if (isinstance(mu_seq_in, (list, tuple)) and mu_seq_in and isinstance(mu_seq_in[0], (list,
                                                                                                               tuple))) else mu_seq_in

    if sigma_seq_in is not None:
        sigma_seq = sigma_seq_in[0] if (
                    isinstance(sigma_seq_in, (list, tuple)) and sigma_seq_in and isinstance(sigma_seq_in[0], (list,
                                                                                                              tuple))) else sigma_seq_in
        assert len(sigma_seq) == len(mu_seq)
    else:
        sigma_seq = None

    T = len(data)
    Lmu = len(mu_seq)
    t_lo = lookback
    t_hi = lookback + Lmu - 1

    eval_start = max(eval_start if eval_start is not None else t_lo, t_lo)
    eval_end = min((eval_end if eval_end is not None else t_hi), t_hi, T - 1)

    if eval_end < eval_start:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    aps: List[float] = []
    mrrs: List[float] = []

    for t in range(eval_start, eval_end + 1):
        mu_t = _to_numpy(mu_seq[t - lookback])
        sigma_t = _to_numpy(sigma_seq[t - lookback]) if sigma_seq is not None else None
        A_t: csr_matrix = data[t][0]

        n_eff = min(A_t.shape[0], mu_t.shape[0])
        if n_eff <= 1:
            continue

        A_crop = A_t[:n_eff, :n_eff]
        mu_crop = mu_t[:n_eff]
        sigma_crop = sigma_t[:n_eff] if sigma_t is not None else None

        ap, mrr = _eval_single_time(mu_crop, sigma_crop, A_crop, neg_multiplier, rng, undirected, symmetric_kl)

        if ap is not None:
            aps.append(ap)
        if mrr is not None:
            mrrs.append(mrr)

    return (float(np.mean(aps)) if aps else float("nan"),
            float(np.mean(mrrs)) if mrrs else float("nan"))


# -------------------- Sanity check --------------------

def sanity_check_time(mu_seq_in: Sequence,
                      sigma_seq_in: Optional[Sequence],
                      lookback: int,
                      data: Sequence,
                      t: int = 72,
                      undirected: bool = True,
                      symmetric_kl: bool = True,
                      num_neg: int = 2000,
                      seed: int = 0) -> Dict[str, Any]:
    """
    Inspect score separation at a single timestep.
    Prints stats and returns a dict with metrics.
    """
    mu_seq = mu_seq_in[0] if (isinstance(mu_seq_in, (list, tuple)) and mu_seq_in and isinstance(mu_seq_in[0], (list,
                                                                                                               tuple))) else mu_seq_in

    sigma_seq = None
    if sigma_seq_in is not None:
        sigma_seq = sigma_seq_in[0] if (
                    isinstance(sigma_seq_in, (list, tuple)) and sigma_seq_in and isinstance(sigma_seq_in[0], (list,
                                                                                                              tuple))) else sigma_seq_in

    mu = _to_numpy(mu_seq[t - lookback])
    sigma = _to_numpy(sigma_seq[t - lookback]) if sigma_seq is not None else None

    A = data[t][0]
    n = min(A.shape[0], mu.shape[0])

    if undirected:
        As = ((A[:n, :n] + A[:n, :n].T) > 0).astype(np.int8)
        r, c = As.nonzero()
        m = r < c
        pos = np.stack([r[m], c[m]], 1)

        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        uu = np.minimum(r, c)
        vv = np.maximum(r, c)
        mask[uu, vv] = False
        cand = np.argwhere(mask)

        rng = np.random.default_rng(seed)
        if len(cand) == 0:
            return {}
        neg = cand[rng.choice(len(cand), size=min(num_neg, len(cand)), replace=False)]
    else:
        r, c = A[:n, :n].nonzero()
        m = r != c
        pos = np.stack([r[m], c[m]], 1)

        rng = np.random.default_rng(seed)
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        mask[r, c] = False
        cand = np.argwhere(mask)
        neg = cand[rng.choice(len(cand), size=min(num_neg, len(cand)), replace=False)]

    pairs = np.vstack([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=np.int32),
                             np.zeros(len(neg), dtype=np.int32)])

    scores = _scores_from_embeddings(mu[:n], sigma[:n] if sigma is not None else None, pairs, symmetric_kl=symmetric_kl)

    pos_scores = scores[:len(pos)]
    neg_scores = scores[len(pos):]

    pos_mean, pos_std = float(pos_scores.mean()), float(pos_scores.std())
    neg_mean, neg_std = float(neg_scores.mean()), float(neg_scores.std())
    pos_frac = float(len(pos) / (len(pos) + len(neg))) if (len(pos) + len(neg)) > 0 else float("nan")

    try:
        auprc = float(average_precision_score(labels, scores))
    except:
        auprc = float("nan")

    try:
        auroc = float(roc_auc_score(labels, scores))
    except:
        auroc = float("nan")

    print(f"[Sanity t={t}] pos mean/std: {pos_mean:.4f}/{pos_std:.4f} | "
          f"neg mean/std: {neg_mean:.4f}/{neg_std:.4f}")
    print(f"[Sanity t={t}] AUPRC: {auprc:.4f} | AUROC: {auroc:.4f} | pos_fraction: {pos_frac:.4f}")

    return dict(t=t, pos_mean=pos_mean, pos_std=pos_std, neg_mean=neg_mean, neg_std=neg_std,
                AUPRC=auprc, AUROC=auroc, pos_fraction=pos_frac,
                n_pos=int(len(pos)), n_neg=int(len(neg)))
