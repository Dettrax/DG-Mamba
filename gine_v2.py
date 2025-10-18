# model/DyGSTA.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv
from torch_geometric.utils import coalesce

from layers import LinkDecoder

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


# ---------------- Spatial encoder: residual GINE ----------------
class SpatialGINE(nn.Module):
    """
    Simple, sturdy spatial encoder:
      - L stacked GINEConv layers (L = num_hop you pass to the model)
      - Residual + LayerNorm per layer
    Input:  x [N, Din], edge_index [2, E], edge_attr [E, edge_dim]
    Output: [N, Dh]
    """
    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int = 1, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.layers = layers
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * layers
        # GINEConv uses an MLP (nn) for the update function
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.Linear(dims[i + 1], dims[i + 1])
                ),
                edge_dim=edge_dim
            ) for i in range(layers)
        ])
        self.res_projs = nn.ModuleList(
            [nn.Identity() if dims[i] == dims[i + 1] else nn.Linear(dims[i], dims[i + 1]) for i in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(layers)])
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index, edge_attr=None):
        # Ensure edge_attr has same dtype/device as node features to avoid
        # "mat1 and mat2 must have the same dtype" runtime error.
        if edge_attr is not None:
            # Move to same device and dtype as x (handles Long -> Float conversion)
            edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)

        for li in range(self.layers):
            h = self.convs[li](x, edge_index, edge_attr)  # message passing with edge attributes
            if li < self.layers - 1:
                h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.norms[li](self.res_projs[li](x) + h)
        return x


# ---------------- Temporal encoder: Mamba (or Transformer fallback) ----------------
class TemporalST(nn.Module):
    """
    Temporal encoder over the last W node embeddings (one vector per node per time step).
    If mamba-ssm is available, uses Mamba blocks; otherwise uses nn.TransformerEncoder.

    Input:  x_seq [N, W, D]
    Output: [N, D]  (recency-weighted pooled)
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        max_len: int = 64,
        dropout: float = 0.2,
        nhead: int = 4,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        # Prefer Mamba when available; will be gated in forward by tensor device
        self.has_mamba = Mamba is not None
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.recency_logit = nn.Parameter(torch.zeros(max_len))  # learned pooling over time
        self.dropout = nn.Dropout(dropout)

        if self.has_mamba:
            self.blocks = nn.ModuleList(
                [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        # Always have a Transformer fallback
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [N, W, D]
        N, W, D = x_seq.shape
        W_clamped = min(W, self.pos_emb.num_embeddings)

        pos = torch.arange(W_clamped, device=x_seq.device)
        x = x_seq[:, :W_clamped, :] + self.pos_emb(pos)[None, :, :]

        if self.has_mamba and x.is_cuda:
            z = x
            for ln, blk in zip(self.norms, self.blocks):
                z = z + self.dropout(blk(ln(z)))  # pre-norm + residual
        else:
            z = self.encoder(x)  # [N, W, D]

        # Learned recency weighting
        w = torch.softmax(self.recency_logit[:z.size(1)], dim=0)  # [W]
        out = (z * w.view(1, -1, 1)).sum(dim=1)                   # [N, D]
        return out


# ---------------- Main model: DyGSTA (Spatial-Temporal Mamba) ----------------
class DyGSTA(nn.Module):
    """
    Drop-in replacement:
      - Same __init__ args and forward signature as the original DyGSTA.
      - Spatial encoder: residual GINE (with edge attributes).
      - Temporal encoder: Mamba (or Transformer fallback).
      - Optional GRUCell for state carry (recurrent=True keeps behavior identical).
      - Maintains memory_edge_index / memory_edge_time lists for compatibility; embeddings are
        also cached internally to build the temporal window.

    forward inputs/outputs unchanged:
      (x, edge_index, edge_label_index, edge_feature, previous_state, is_updated=False)
      -> returns (prediction, new_state)
    """
    def __init__(
        self,
        dim_in,
        hidden_dim,
        dim_out,
        num_hop,          # re-used as number of GINE layers (spatial depth)
        num_heads,        # used by Transformer fallback (nhead); Mamba ignores it
        window_size,
        ratio=0.8,        # kept for API compatibility; not used here
        time_encode=True, # kept for API compatibility; not used here
        recurrent=True,
        device='cuda',
        use_gaussian: bool = False,
        gauss_score: str = 'kl',
        gauss_scale: float = 1.0,
        gauss_aux_only: bool = False,
        edge_dim: int = 1,  # edge attribute dimension for UCI dataset
    ):
        super().__init__()

        # ---- public attributes expected elsewhere (train loop) ----
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.window_size = window_size
        self.ratio = ratio
        self.num_hop = num_hop
        self.time_encode = time_encode
        self.recurrent = recurrent
        self.num_heads = num_heads
        self.device = device
        self.edge_dim = edge_dim

        # Gaussian options
        self.use_gaussian = use_gaussian
        self.gauss_score = gauss_score
        self.gauss_scale = gauss_scale
        self.gauss_aux_only = gauss_aux_only

        # ---- input projection ----
        self.mlp_transform = nn.Sequential(
            nn.Linear(self.dim_in, self.hidden_dim),
            nn.ReLU(),
        )

        # ---- spatial & temporal encoders ----
        self.spatial = SpatialGINE(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            edge_dim=self.edge_dim,
            layers=max(1, self.num_hop),
            dropout=0.2
        )
        self.temporal = TemporalST(
            d_model=self.hidden_dim,
            num_layers=2,
            max_len=max(2, self.window_size),
            dropout=0.2,
            nhead=max(1, self.num_heads),
            d_state=48,
            d_conv=4,
            expand=2,
        )

        # ---- optional recurrent state (keeps original behavior) ----
        if self.recurrent:
            self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        # ---- edge decoder (unchanged) ----
        self.decoder = LinkDecoder(self.hidden_dim, self.dim_out)

        # ---- Gaussian head (optional) ----
        if self.use_gaussian:
            self.mu_head = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.logvar_head = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.gauss_bias = nn.Parameter(torch.zeros(1))

        # ---- memory (keep same API as before) ----
        self.memory_edge_index = []
        self.memory_edge_time = []

        # Embedding memory for the temporal window (we keep it CPU-side like edges)
        self.memory_embedding = []

        # cache for last gaussian params for loss computation
        self.last_mu = None
        self.last_logvar = None

    # -------- Memory helpers (compatible with original train/eval loops) --------
    def update_memory(self, edge_index, edge_time):
        if self.window_size == 1:
            return
        if len(self.memory_edge_index) >= self.window_size:
            self.memory_edge_index.pop(0)
            self.memory_edge_time.pop(0)
            # (Keep invariant with original behavior.)
            del self.memory_edge_index[0], self.memory_edge_time[0]
        self.memory_edge_index.append(edge_index.clone().detach().cpu())
        self.memory_edge_time.append(edge_time.clone().detach().cpu())

    def reset_memory(self):
        self.memory_edge_index = []
        self.memory_edge_time = []
        self.memory_embedding = []
        self.last_mu = None
        self.last_logvar = None

    def read_memory(self, edge_index, edge_time):
        # Used by eval loop to warm-start with train memory (edges & times).
        self.memory_edge_index = edge_index
        self.memory_edge_time = edge_time
        # Embeddings are rebuilt on-the-fly during the loop.

    def merge_graphs(self, edge_index_list, edge_time_list):
        merge = torch.cat(edge_index_list, dim=-1)
        merge_time = torch.cat(edge_time_list, dim=-1)
        merge, merge_time = coalesce(merge, merge_time, reduce='max')
        return merge, merge_time

    def merge_memory(self):
        merge_graph, merge_time = self.merge_graphs(self.memory_edge_index, self.memory_edge_time)
        return merge_graph.to(self.device), merge_time.to(self.device)

    def _gaussian_edge_logits(self, mu: torch.Tensor, logvar: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Compute symmetric KL (or L2) distance per candidate edge and map to logits via negative scaling.
        i = edge_label_index[0]
        j = edge_label_index[1]
        mu_i, mu_j = mu[i], mu[j]
        lv_i, lv_j = logvar[i], logvar[j]

        if self.gauss_score == 'l2':
            d = torch.sum((mu_i - mu_j) ** 2, dim=-1)  # [L]
        else:
            # KL(i||j)
            var_i = torch.exp(lv_i)
            inv_var_j = torch.exp(-lv_j)
            diff_ij = mu_j - mu_i
            k = mu_i.size(-1)
            kl_ij = 0.5 * (
                torch.sum(var_i * inv_var_j, dim=-1)
                + torch.sum(diff_ij * diff_ij * inv_var_j, dim=-1)
                - k
                + torch.sum(lv_j - lv_i, dim=-1)
            )
            # KL(j||i)
            var_j = torch.exp(lv_j)
            inv_var_i = torch.exp(-lv_i)
            diff_ji = mu_i - mu_j
            kl_ji = 0.5 * (
                torch.sum(var_j * inv_var_i, dim=-1)
                + torch.sum(diff_ji * diff_ji * inv_var_i, dim=-1)
                - k
                + torch.sum(lv_i - lv_j, dim=-1)
            )
            d = 0.5 * (kl_ij + kl_ji)

        logits = - self.gauss_scale * d + getattr(self, 'gauss_bias', 0.0)
        return logits.unsqueeze(-1)  # shape [L, 1]

    # --------- Forward (drop-in compatible) ---------
    def forward(self, x, edge_index, edge_label_index, edge_feature, previous_state, is_updated: bool = False):
        """
        x:                 [N, dim_in]
        edge_index:        [2, E]
        edge_label_index:  [2, L]  (candidate edges to score)
        edge_feature:      [E, edge_dim] edge attributes (e.g., timestamps for UCI dataset)
        previous_state:    [N, hidden_dim]
        is_updated:        same flag used by original code
        """
        device = x.device

        # Maintain original "memory update" call order for edges/times
        if not is_updated and self.window_size != 1:
            self.update_memory(edge_index, edge_feature)

        # 1) Project + spatial encode on current graph
        x = self.mlp_transform(x)                       # [N, H]

        # Prepare edge_attr: for UCI, edge_feature is typically timestamps [E] or [E, 1]
        edge_attr = edge_feature
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # [E, 1]

        h_spatial = self.spatial(x, edge_index, edge_attr)  # [N, H] with edge attributes

        # 2) Build temporal window of embeddings (last W-1 from memory + current)
        if self.window_size != 1 and len(self.memory_embedding) > 0:
            # gather up to window_size - 1 previous embeddings
            prev_list = self.memory_embedding[-(self.window_size - 1):]
            prev_seq = torch.stack([t.to(device) for t in prev_list], dim=1)  # [N, W_prev, H]
            x_seq = torch.cat([prev_seq, h_spatial.unsqueeze(1)], dim=1)      # [N, W, H]
        else:
            x_seq = h_spatial.unsqueeze(1)                                     # [N, 1, H]

        # 3) Temporal aggregation (Mamba or Transformer fallback)
        z_t = self.temporal(x_seq)
        # if self.recurrent:
        #     previous_state = previous_state.to(device)
        #     z_t = self.gru(z_t, previous_state)

        # Compute Gaussian params if enabled (for scoring or aux losses)
        mu = logvar = None
        if self.use_gaussian:
            mu = self.mu_head(z_t)
            logvar = self.logvar_head(z_t)
            self.last_mu = mu
            self.last_logvar = logvar

        # Choose scoring head
        if self.use_gaussian and not self.gauss_aux_only:
            prediction = self._gaussian_edge_logits(mu, logvar, edge_label_index)
        else:
            prediction = self.decoder(z_t, edge_label_index)

        # 6) Append current embedding to memory (after computing outputs)
        if not is_updated and self.window_size != 1:
            self.memory_embedding.append(h_spatial.detach().cpu())
            if len(self.memory_embedding) > self.window_size:
                self.memory_embedding = self.memory_embedding[-self.window_size:]

        return prediction, z_t
