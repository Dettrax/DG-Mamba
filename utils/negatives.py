# utils/negatives.py
import torch
import torch.nn.functional as F

def self_adversarial_bpr(scores: torch.Tensor, labels: torch.Tensor, tau: float = 0.07):
    """
    scores: (M,) edge scores for this snapshot’s candidates
    labels: (M,) in {0,1}
    tau: temperature for adversarial weighting; smaller -> harder negatives emphasized

    Returns a scalar BPR-style ranking loss where negatives are weighted by their
    current predicted hardness (self-adversarial).
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_zeros(())

    # weights for negatives: w_j ∝ exp(neg_j / tau)
    w = F.softmax(neg / tau, dim=0)  # (N_neg,)

    # pairwise BPR: -log σ(pos - neg), expectation over adversarial negs
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)          # (N_pos, N_neg)
    bpr = F.softplus(-diff)                             # = -log σ(diff)
    loss = (bpr * w.unsqueeze(0)).sum(dim=1).mean()     # expectation over weighted negs
    return loss
