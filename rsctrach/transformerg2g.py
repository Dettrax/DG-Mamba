# transformerg2g.py
# TransformerG2G: Gaussian node embeddings for temporal graphs.
# - Robust to all-masked sequences (avoids NaNs for small lookback).
# - Dropout, masking, and simple heads for mean/variance.
# - KL-based triplet loss as in the paper.

from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding (batch_first)."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        return x + self.pe[:, : x.size(1), :]


def gaussian_kl_div(mu_p: torch.Tensor, sig_p: torch.Tensor,
                    mu_q: torch.Tensor, sig_q: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    """KL( N(mu_p, diag(sig_p)) || N(mu_q, diag(sig_q)) ) per-sample.
    mu_*:  [B, D]
    sig_*: [B, D]  (variances, must be > 0)
    returns: [B] KL values
    """
    sig_p = sig_p.clamp_min(eps)
    sig_q = sig_q.clamp_min(eps)
    term1 = torch.sum(sig_p / sig_q, dim=-1)
    term2 = torch.sum((mu_q - mu_p) ** 2 / sig_q, dim=-1)
    logdet_ratio = torch.sum(torch.log(sig_q) - torch.log(sig_p), dim=-1)
    k = mu_p.size(-1)
    return 0.5 * (term1 + term2 - k + logdet_ratio)


def triplet_contrastive_loss(mu_ref, sig_ref, mu_near, sig_near, mu_far, sig_far,margin: float = 0.5,
                             reduction: str = "mean") -> torch.Tensor:
    """Paper-style loss:  E_pos^2 + exp(-E_neg)  where E is Gaussian KL-divergence."""
    E_pos = gaussian_kl_div(mu_ref, sig_ref, mu_near, sig_near)   # [M]
    E_neg = gaussian_kl_div(mu_ref, sig_ref, mu_far,  sig_far)    # [M]
    loss = E_pos**2 + torch.exp(-(E_neg - margin))  # margin encourages larger E_neg
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class TransformerG2G(nn.Module):
    """Transformer encoder over per-node row-history -> Gaussian embedding (mu, sigma)."""
    def __init__(self,
                 n_nodes: int,
                 lookback: int,
                 d_model: int = 256,
                 hidden_after: int = 512,
                 emb_dim: int = 64,
                 nhead: int = 2,                 # even heads avoids nested-tensor warning path
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 var_eps: float = 1e-6,
                 causal_attention: bool = True  # <--- new argument
                 ):
        super().__init__()
        self.n = n_nodes
        self.l = lookback
        self.L = lookback + 1
        self.var_eps = var_eps
        self.causal_attention = causal_attention

        self.in_proj = nn.Linear(n_nodes, d_model, bias=True)
        self.posenc = PositionalEncoding(d_model, max_len=self.L)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.after = nn.Sequential(
            nn.Linear(d_model, hidden_after),
            nn.Tanh(),
        )
        self.head_mu = nn.Linear(hidden_after, emb_dim)
        self.head_sigma = nn.Linear(hidden_after, emb_dim)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """seq: [B, L, n], rows for a single node across time (including current or zero).
        returns:
          mu:    [B, emb_dim]
          sigma: [B, emb_dim] (positive variances)
        """
        # Build padding mask: rows that are exactly zero get masked (no attention paid).
        with torch.no_grad():
            pad_mask = (seq.abs().sum(dim=-1) == 0.0)  # [B, L], True=mask

        x = self.in_proj(seq)          # [B, L, d_model]
        x = self.dropout(x)
        x = self.posenc(x)
        x = self.dropout(x)

        # Avoid the "all tokens masked" case (can yield NaNs in attention)
        all_masked = pad_mask.all(dim=1)                # [B]
        if all_masked.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_masked, -1] = False           # unmask last token

        # Causal attention mask (optional)
        attn_mask = None
        if self.causal_attention:
            L = seq.size(1)
            attn_mask = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()  # [L, L], True=mask

        x = self.encoder(x, src_key_padding_mask=pad_mask, mask=attn_mask)  # [B, L, d_model]
        h = self.after(x[:, -1, :])  # take the last timestep’s representation
        mu = self.head_mu(h)
        # ELU + 1 + tiny eps keeps variance strictly > 0
        sigma = F.softplus(self.head_sigma(h)) + self.var_eps
        sigma = sigma.clamp(max=10.0)

        return mu, sigma
