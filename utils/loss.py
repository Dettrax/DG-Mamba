import torch
from sklearn.metrics import roc_auc_score, average_precision_score

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

# ---------- Gaussian utilities ----------

def gaussian_symmetric_kl(mu_i: torch.Tensor, lv_i: torch.Tensor, mu_j: torch.Tensor, lv_j: torch.Tensor) -> torch.Tensor:
    """
    Symmetric KL divergence between two diagonal Gaussians per row.
    Inputs are [L, D]; returns [L].
    """
    var_i = torch.exp(lv_i)
    var_j = torch.exp(lv_j)
    inv_var_i = torch.exp(-lv_i)
    inv_var_j = torch.exp(-lv_j)

    diff_ij = mu_j - mu_i
    diff_ji = -diff_ij
    k = mu_i.size(-1)

    kl_ij = 0.5 * (
        torch.sum(var_i * inv_var_j, dim=-1) + torch.sum(diff_ij * diff_ij * inv_var_j, dim=-1) - k + torch.sum(lv_j - lv_i, dim=-1)
    )
    kl_ji = 0.5 * (
        torch.sum(var_j * inv_var_i, dim=-1) + torch.sum(diff_ji * diff_ji * inv_var_i, dim=-1) - k + torch.sum(lv_i - lv_j, dim=-1)
    )
    return 0.5 * (kl_ij + kl_ji)


def gaussian_pair_distance(mu: torch.Tensor, logvar: torch.Tensor, edge_index: torch.Tensor, mode: str = 'kl') -> torch.Tensor:
    """
    Compute distance between endpoints for each edge in edge_index.
    Returns tensor [L], where L = edge_index.size(1).
    mode: 'kl' for symmetric KL, 'l2' for Euclidean between means.
    """
    i = edge_index[0]
    j = edge_index[1]
    mu_i, mu_j = mu[i], mu[j]
    lv_i, lv_j = logvar[i], logvar[j]
    if mode == 'l2':
        return torch.sum((mu_i - mu_j) ** 2, dim=-1)
    else:
        return gaussian_symmetric_kl(mu_i, lv_i, mu_j, lv_j)


def triplet_margin_loss(d_pos: torch.Tensor, d_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Triplet hinge loss where distances should satisfy d_pos + margin <= d_neg.
    Inputs: [L] each. Returns scalar.
    """
    return torch.relu(d_pos - d_neg + margin).mean()


def kl_prior_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(mu, diag(exp(logvar))) and N(0, I), averaged over nodes.
    Returns scalar.
    """
    # Standard analytical KL(q||p) for diagonal Gaussian and unit Gaussian prior
    kld = 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar)
    return kld.sum(dim=-1).mean()
