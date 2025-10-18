# utils/smooth_ap.py
import torch
import torch.nn.functional as F

def smooth_ap_loss(scores: torch.Tensor, labels: torch.Tensor, tau: float = 0.01, eps: float = 1e-8):
    """
    scores: (M,) unnormalized scores (logits or any real scores)
    labels: (M,) binary {0,1}; must contain at least one positive
    returns: scalar loss ~ (1 - SmoothAP)

    Implements the ECCV'20 Smooth-AP differentiable AP surrogate.
    Ref: Brown et al., "Smooth-AP" (ECCV 2020), Algorithm 1 (supplement).
    """
    scores = scores.view(-1)
    labels = labels.view(-1).float()
    pos_cnt = labels.sum()
    if pos_cnt <= 0:
        # no positives in this batch; give zero loss so training doesn’t blow up
        return scores.new_zeros(())
    m = scores.numel()

    # pairwise score differences, sigmoid temperature
    s1 = scores.unsqueeze(0).expand(m, m)
    s2 = scores.unsqueeze(1).expand(m, m)
    D = s1 - s2
    # approximate heaviside with temperature tau
    H = torch.sigmoid(D / (tau + eps))

    # zero out diagonal contributions
    eye = torch.eye(m, device=scores.device, dtype=H.dtype)
    H = H * (1.0 - eye)

    # soft rank of each sample
    R = 1.0 + H.sum(dim=1)  # (m,)

    # precision at ranks of positives, averaged
    R_pos = R * labels
    ap_smooth = (R_pos / (R + eps)).sum() / (pos_cnt + eps)

    # As a loss, minimize (1 - AP)
    return 1.0 - ap_smooth
