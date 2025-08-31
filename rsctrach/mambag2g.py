# mambag2g.py
# MambaG2G: Gaussian node embeddings for temporal graphs with a Mamba backbone.
# - API-compatible with TransformerG2G: forward(seq[B, L, n]) -> (mu[B, d], sigma[B, d])
# - Robust to all-zero histories: falls back to a learned token for entirely padded sequences.
# - Keeps KL-based triplet loss for compatibility with existing training loops.

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError(
        "mamba_ssm is required. Install with: pip install mamba-ssm"
    ) from e


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding (batch_first)."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # even
        pe[:, 1::2] = torch.cos(position * div_term)   # odd
        self.register_buffer("pe", pe.unsqueeze(0))    # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        return x + self.pe[:, : x.size(1), :]


def gaussian_kl_div(mu_p: torch.Tensor, sig_p: torch.Tensor,
                    mu_q: torch.Tensor, sig_q: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    """KL( N(mu_p, diag(sig_p)) || N(mu_q, diag(sig_q)) ) per-sample. Returns [B]."""
    sig_p = sig_p.clamp_min(eps)
    sig_q = sig_q.clamp_min(eps)
    term1 = torch.sum(sig_p / sig_q, dim=-1)
    term2 = torch.sum((mu_q - mu_p) ** 2 / sig_q, dim=-1)
    logdet_ratio = torch.sum(torch.log(sig_q) - torch.log(sig_p), dim=-1)
    k = mu_p.size(-1)
    return 0.5 * (term1 + term2 - k + logdet_ratio)


def triplet_contrastive_loss(mu_ref, sig_ref, mu_near, sig_near, mu_far, sig_far,
                             margin: float = 0.5, reduction: str = "mean") -> torch.Tensor:
    """Paper-style loss: E_pos^2 + exp(- (E_neg - margin))."""
    E_pos = gaussian_kl_div(mu_ref, sig_ref, mu_near, sig_near)   # [M]
    E_neg = gaussian_kl_div(mu_ref, sig_ref, mu_far,  sig_far)    # [M]
    loss = E_pos**2 + torch.exp(-(E_neg - margin))
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class _MambaBlock(nn.Module):
    """LayerNorm -> Mamba -> Dropout with residual."""
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.core = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(self.norm(x))     # [B, L, d_model]
        return x + self.drop(y)         # residual


class MambaG2G(nn.Module):
    """Sequence encoder over per-node row-history -> Gaussian embedding (mu, sigma) via Mamba.

    Args:
      n_nodes:     number of nodes (input feature dim per timestep)
      lookback:    how many past rows, so L = lookback + 1
      d_model:     model width
      hidden_after: hidden size in post-seq MLP before heads
      emb_dim:     dimension of output embedding (mu, sigma each [B, emb_dim])
      num_layers:  number of Mamba blocks
      dropout:     dropout rate (applied in blocks and on projections)
      var_eps:     small epsilon added to variances to keep them > 0
      use_posenc:  if True, adds sine/cosine positional encoding
      d_state, d_conv, expand: Mamba hyperparameters (see mamba-ssm docs)

    Notes:
      - Accepts **unused to swallow Transformer-only kwargs like nhead/causal_attention,
        so you can plug it into existing scripts without touching the arg parser.
    """
    def __init__(self,
                 n_nodes: int,
                 lookback: int,
                 d_model: int = 256,
                 hidden_after: int = 512,
                 emb_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 var_eps: float = 1e-6,
                 use_posenc: bool = False,
                 # Mamba core hparams (match your example defaults)
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 **unused):
        super().__init__()
        self.n = n_nodes
        self.l = lookback
        self.L = lookback + 1
        self.var_eps = var_eps
        self.use_posenc = use_posenc

        # Input projection from [n_nodes] to [d_model]
        self.in_proj = nn.Linear(n_nodes, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Optional positional encoding (Mamba is sequence-aware; this is a harmless boost)
        self.posenc = PositionalEncoding(d_model, max_len=self.L)

        # A small learned fallback token to avoid degenerate “all zero rows” sequences
        self.fallback_token = nn.Parameter(1e-3 * torch.randn(n_nodes))

        # Mamba stack
        blocks = []
        for _ in range(num_layers):
            blocks.append(_MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout))
        self.backbone = nn.Sequential(*blocks)

        # Heads
        self.after = nn.Sequential(
            nn.Linear(d_model, hidden_after),
            nn.Tanh(),
        )
        self.head_mu = nn.Linear(hidden_after, emb_dim)
        self.head_sigma = nn.Linear(hidden_after, emb_dim)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq:   [B, L, n] — per-node row history (current or zero at last row).
        returns:
          mu:    [B, emb_dim]
          sigma: [B, emb_dim] (positive variances)
        """
        B, L, N = seq.shape
        assert L == self.L and N == self.n, f"Expected [B,{self.L},{self.n}] but got {seq.shape}"

        # Detect “padded” rows (exactly zero). Keep behavior aligned with TransformerG2G.
        with torch.no_grad():
            row_is_zero = (seq.abs().sum(dim=-1) == 0.0)  # [B, L], True means padded row
            all_masked = row_is_zero.all(dim=1)           # [B], True means every row was zero

        # For entirely masked sequences, drop in a learned token at the last step
        if all_masked.any():
            seq = seq.clone()
            seq[all_masked, -1, :] = self.fallback_token  # [n]

        # Project to model width and (optionally) add positional encoding
        x = self.in_proj(seq)              # [B, L, d_model]
        x = self.dropout(x)
        if self.use_posenc:
            x = self.posenc(x)
        x = self.dropout(x)

        # Mamba stack (causal by construction; no explicit masks required)
        x = self.backbone(x)               # [B, L, d_model]

        # Use the last timestep representation
        h = self.after(x[:, -1, :])        # [B, hidden_after]
        mu = self.head_mu(h)               # [B, emb_dim]
        sigma = F.softplus(self.head_sigma(h)) + self.var_eps
        sigma = sigma.clamp(max=10.0)

        return mu, sigma


# ------------------------- Quick self-test -------------------------
if __name__ == "__main__":
    # Shape sanity check against mamba_ssm example
    import torch
    B, L, N = 2, 64, 16
    seq = torch.randn(B, L, N).to("cuda" if torch.cuda.is_available() else "cpu")

    model = MambaG2G(
        n_nodes=N, lookback=L - 1, d_model=64,
        hidden_after=64, emb_dim=32,
        num_layers=2, dropout=0.1,
        d_state=16, d_conv=4, expand=2,
    ).to(seq.device)

    mu, sigma = model(seq)
    assert mu.shape == (B, 32) and sigma.shape == (B, 32)
    print("MambaG2G OK:", mu.shape, sigma.shape)
