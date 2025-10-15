import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv , SAGEConv

# ---- Freeze all seeds for reproducibility ----
def freeze_all_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

freeze_all_seeds(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Spatial encoder ----------------
class SpatialGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.num_layers = num_layers
        self.dropout = dropout

    def edge_dropout(self, edge_index, p):
        if not self.training or p == 0.0:
            return edge_index
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges, device=edge_index.device) > p
        return edge_index[:, mask]

    def forward(self, x, edge_index):
        for li, conv in enumerate(self.convs):
            # Apply edge dropout before each layer except the last
            if li < self.num_layers - 1:
                edge_index_drop = self.edge_dropout(edge_index, self.dropout)
            else:
                edge_index_drop = edge_index
            x = conv(x, edge_index_drop)
            if li < self.num_layers - 1:
                x = F.leaky_relu(x)
        return x  # [N, out_dim]

class SpatialBiSAGE(nn.Module):
    """
    Direction-aware Spatial Encoder:
      - SAGE over out-edges (u->v) and over in-edges (v->u).
      - Learned gate mixes them per feature.
      - Residual path projects when dims change.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # dims[0] -> dims[1] -> ... -> dims[num_layers]
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        self.out_convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(num_layers)])
        self.in_convs  = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(num_layers)])

        # gate to mix [h_out || h_in] -> h
        self.gates = nn.ModuleList([nn.Linear(2 * dims[i+1], dims[i+1]) for i in range(num_layers)])

        # residual projection when dims change
        self.res_projs = nn.ModuleList([
            nn.Identity() if dims[i] == dims[i+1] else nn.Linear(dims[i], dims[i+1])
            for i in range(num_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(num_layers)])
        self.act = nn.LeakyReLU(0.1)

    def edge_dropout(self, edge_index, p):
        if (not self.training) or p <= 0.0:
            return edge_index
        m = edge_index.size(1)
        keep = torch.rand(m, device=edge_index.device) > p
        return edge_index[:, keep]

    def forward(self, x, edge_index):
        ei = edge_index
        for li in range(self.num_layers):
            # drop edges before this layer (not on last layer)
            ei_drop = self.edge_dropout(ei, self.dropout) if li < self.num_layers - 1 else ei
            ei_rev  = torch.stack([ei_drop[1], ei_drop[0]], dim=0)

            h_out = self.out_convs[li](x, ei_drop)   # u -> v
            h_in  = self.in_convs [li](x, ei_rev)    # v -> u

            H = torch.cat([h_out, h_in], dim=-1)
            g = torch.sigmoid(self.gates[li](H))
            h = g * h_out + (1.0 - g) * h_in

            # residual with projection if needed
            x_res = self.res_projs[li](x)
            x = self.norms[li](x_res + F.dropout(h, p=self.dropout, training=self.training))
            if li < (self.num_layers - 1):
                x = self.act(x)
        return x


class FourierTime(nn.Module):
    def __init__(self, d_model, num_freq=8):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(num_freq))
        self.proj = nn.Linear(2 * num_freq, d_model)

    def forward(self, delta):  # delta: [N, W] (time gaps, e.g., days)
        D = delta.unsqueeze(-1) * self.freq  # [N,W,F]
        f = torch.cat([torch.sin(D), torch.cos(D)], dim=-1)  # [N,W,2F]
        return self.proj(f)  # [N,W,d_model]

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None

class TemporalMamba(nn.Module):
    def __init__(self, d_model, num_layers=2, max_len=64, dropout=0.1,
                 d_state=32, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is not installed. pip install mamba-ssm or set temporal_type='transformer'"
            )
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.recency_logit = nn.Parameter(torch.zeros(max_len))
        self.time_emb = FourierTime(d_model, num_freq=8)
        self.pool = nn.Linear(d_model, 1)

        self.blocks = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.drops  = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(d_model))
            self.blocks.append(Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            self.drops.append(nn.Dropout(dropout))

    # --- drop-in replacement for TemporalMamba.forward in model.py ---

    def forward(self, x_seq, delta_seq=None):  # x_seq: [N, W, d], delta_seq: [N, W]
        # 1) add time features
        x = x_seq
        if delta_seq is not None:
            x = x + self.time_emb(delta_seq)  # [N, W, d]

        # 2) add (learned) position embedding per step
        N, W, D = x.shape
        pos = torch.arange(W, device=x.device)
        x = x + self.pos_emb(pos)[None, :, :]  # [N, W, d]

        # 3) Mamba blocks (pre-norm + residual)
        z = x
        for ln, blk, dr in zip(self.norms, self.blocks, self.drops):
            z = z + dr(blk(ln(z)))  # residual path that can integrate over the whole window

        # 4) recency-aware pooling over time (learned softmax weights)
        w = torch.softmax(self.recency_logit[:W], dim=0)  # [W]
        out = (z * w.view(1, W, 1)).sum(dim=1)  # [N, d]
        return out


# ---------------- Main model ----------------
class STFormerGCN(nn.Module):
    def __init__(self,
                 in_dim, gcn_dim, d_model, proj_dim,
                 num_nodes, use_id_emb=True,
                 nhead=8, num_tlayers=2, gcn_layers=2,
                 dropout=0.1, max_len=64,
                 sigma_floor=1e-4,
                 temporal_type: str = "mamba",
                 spatial_type: str = "bisage",      # NEW: 'gcn' or 'bisage'
                 directed: bool = False):        # NEW: if True, use forward-only energy
        super().__init__()
        self.is_stformer = True
        self.in_dim = in_dim
        self.gcn_dim = gcn_dim
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.use_id_emb = use_id_emb
        self.sigma_floor = sigma_floor
        self.directed = directed
        self.var_reg = 0.0

        if self.use_id_emb:
            self.id_emb = nn.Embedding(num_nodes, in_dim)
            self.id_drop = nn.Dropout(p=0.2)

        # --- Spatial encoder choice ---
        if spatial_type.lower() == "bisage":
            self.spatial = SpatialBiSAGE(in_dim, gcn_dim, d_model, num_layers=gcn_layers, dropout=dropout)
        else:
            self.spatial = SpatialGCN(in_dim, gcn_dim, d_model, num_layers=gcn_layers, dropout=dropout)

        # --- Temporal encoder (unchanged) ---
        if temporal_type.lower() == "mamba":
            self.temporal = TemporalMamba(d_model, num_layers=num_tlayers, max_len=max_len,
                                          dropout=dropout, d_state=32, d_conv=4, expand=2)

        # Gaussian heads
        self.mu_src_head  = nn.Linear(d_model, d_model)
        self.rho_src_head = nn.Linear(d_model, d_model)
        self.mu_dst_head  = nn.Linear(d_model, d_model)
        self.rho_dst_head = nn.Linear(d_model, d_model)

        self.window_size = getattr(self, 'window_size', 10)
        self.logit_scale = nn.Parameter(torch.tensor(2.5))

    # ---- Encoders ----
    def encode_one_snapshot(self, snapshot):
        dev = next(self.parameters()).device
        ei = snapshot.edge_index.to(dev)
        if self.use_id_emb:
            x = self.id_emb.weight
            if self.training:
                x = self.id_drop(x)
        else:
            x = snapshot.node_feature.to(dev)
        h = self.spatial(x, ei)  # [N, d_model]
        return h

    def encode_window(self, snapshots, delta_seq=None):
        Hs = [self.encode_one_snapshot(s) for s in snapshots]  # [N,d] each
        H_seq = torch.stack(Hs, dim=1)  # [N,W,d]
        z = self.temporal(H_seq, delta_seq)  # [N,d]
        return z

    # ---- Gaussian params ----
    def gaussian_params(self, z):
        mu_src = self.mu_src_head(z)
        rho_src = self.rho_src_head(z)
        mu_dst = self.mu_dst_head(z)
        rho_dst = self.rho_dst_head(z)
        sigma_src = F.softplus(rho_src) + self.sigma_floor
        sigma_dst = F.softplus(rho_dst) + self.sigma_floor
        return mu_src, sigma_src, mu_dst, sigma_dst

    @staticmethod
    def kl_diag(mu1, var1, mu2, var2):
        d = mu1.size(-1)
        ratio = var1 / var2
        trace = ratio.sum(dim=-1)
        delta = mu2 - mu1
        quad  = (delta * delta / var2).sum(dim=-1)
        logdet= (torch.log(var2) - torch.log(var1)).sum(dim=-1)
        return 0.5 * (trace + quad - d + logdet)

    def energy(self, mu_src, var_src, mu_dst, var_dst, edge_index):
        u, v = edge_index
        e_fwd = self.kl_diag(mu_src[u], var_src[u], mu_dst[v], var_dst[v])
        if self.directed:
            return e_fwd
        # fallback to symmetric if graph is undirected
        e_bwd = self.kl_diag(mu_dst[v], var_dst[v], mu_src[u], var_src[u])
        return 0.5 * (e_fwd + e_bwd)

    def score_edges(self, z, edge_index):
        mu_src, var_src, mu_dst, var_dst = self.gaussian_params(z)
        energy = self.energy(mu_src, var_src, mu_dst, var_dst, edge_index)
        logits = -energy * self.logit_scale.clamp(min=0.05, max=50.0)
        return logits