import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

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
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(len(dims)-1)])
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, edge_index):
        for li, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if li < self.num_layers - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class FourierTime(nn.Module):
    def __init__(self, d_model, num_freq=16):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(num_freq) * 0.1)
        self.phase = nn.Parameter(torch.randn(num_freq) * 0.1)
        self.proj = nn.Sequential(
            nn.Linear(2 * num_freq, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, delta):  # delta: [N, W] (time gaps, e.g., days)
        D = delta.unsqueeze(-1) * self.freq + self.phase  # [N,W,F]
        f = torch.cat([torch.sin(D), torch.cos(D)], dim=-1)  # [N,W,2F]
        return self.proj(f)  # [N,W,d_model]

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None

class TemporalMamba(nn.Module):
    def __init__(self, d_model, num_layers=2, max_len=64, dropout=0.1,
                 d_state=64, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is not installed. pip install mamba-ssm or set temporal_type='transformer'"
            )
        self.d_model = d_model
        self.time_emb = FourierTime(d_model, num_freq=16)

        # Learnable recency weighting
        self.recency_weight = nn.Parameter(torch.ones(max_len))

        self.blocks = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.drops  = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(d_model))
            self.blocks.append(Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            self.drops.append(nn.Dropout(dropout))

        # Output projection with residual
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x_seq, delta_seq=None):  # x_seq: [N,W,d], delta_seq: [N,W]
        N, W, D = x_seq.shape

        # Apply time encoding
        x = x_seq
        if delta_seq is not None:
            time_enc = self.time_emb(delta_seq)
            x = x + time_enc

        # Apply learnable recency weights
        recency = torch.sigmoid(self.recency_weight[:W]).view(1, W, 1)
        x = x * recency

        # Mamba blocks with residual connections
        z = x
        for ln, blk, dr in zip(self.norms, self.blocks, self.drops):
            res = z
            z = ln(z)
            z = blk(z)
            z = dr(z) + res  # residual connection

        # Take last timestep output
        out = z[:, -1, :]  # [N,d]

        # Final projection with residual
        out = self.output_norm(out)
        out = out + self.output_proj(out)

        return out


# ---------------- Main model ----------------
class STFormerGCN(nn.Module):
    def __init__(self,
                 in_dim, gcn_dim, d_model, proj_dim,
                 num_nodes, use_id_emb=True,
                 nhead=8, num_tlayers=2, gcn_layers=2,
                 dropout=0.1, max_len=64,
                 sigma_floor=1e-4,
                 temporal_type: str = "mamba"):
        super().__init__()
        self.is_stformer = True
        self.in_dim = in_dim
        self.gcn_dim = gcn_dim
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.use_id_emb = use_id_emb
        self.sigma_floor = sigma_floor
        self.sym_kl = False  # Asymmetric is better for directed graphs
        self.var_reg = 0.0

        if self.use_id_emb:
            self.id_emb = nn.Embedding(num_nodes, in_dim)
            nn.init.xavier_uniform_(self.id_emb.weight)
            self.id_drop = nn.Dropout(p=0.1)

        self.spatial = SpatialGCN(in_dim, gcn_dim, d_model, num_layers=gcn_layers, dropout=dropout)

        if temporal_type.lower() == "mamba":
            self.temporal = TemporalMamba(d_model, num_layers=num_tlayers, max_len=max_len,
                                          dropout=dropout, d_state=64, d_conv=4, expand=2)
        else:
            self.temporal = TemporalTransformer(d_model, nhead=nhead, num_layers=num_tlayers,
                                                max_len=max_len, dropout=dropout)

        # Separate Gaussian heads for source and destination with better initialization
        self.mu_src_head  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.rho_src_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        self.mu_dst_head  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.rho_dst_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

        # Ensure training loop uses the intended window size
        self.window_size = max_len
        self.logit_scale = nn.Parameter(torch.tensor(3.0))

    # ---- Encoders ----
    def encode_one_snapshot(self, snapshot):
        dev = next(self.parameters()).device
        ei = snapshot.edge_index.to(dev)
        if self.use_id_emb:
            x = self.id_emb.weight
            if self.training:
                x = self.id_drop(x)
        else:
            x = snapshot.node_feature
        h = self.spatial(x.to(dev), ei)  # [N, d_model]
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

        # Better variance parameterization: rho in [-1, 1] -> sigma in [floor, ~e^1]
        sigma_src = F.softplus(rho_src * 2.0) + self.sigma_floor
        sigma_dst = F.softplus(rho_dst * 2.0) + self.sigma_floor
        return mu_src, sigma_src, mu_dst, sigma_dst

    @staticmethod
    def kl_diag(mu1, var1, mu2, var2):
        d = mu1.size(-1)
        ratio = var1 / var2
        trace = ratio.sum(dim=-1)
        delta = mu2 - mu1
        quad = (delta * delta / var2).sum(dim=-1)
        logdet = (torch.log(var2) - torch.log(var1)).sum(dim=-1)
        return 0.5 * (trace + quad - d + logdet)

    def energy_kl(self, mu_src, var_src, mu_dst, var_dst, edge_index):
        u, v = edge_index
        e_forward = self.kl_diag(mu_src[u], var_src[u], mu_dst[v], var_dst[v])
        if self.sym_kl:
            e_backward = self.kl_diag(mu_dst[v], var_dst[v], mu_src[u], var_src[u])
            return 0.5 * (e_forward + e_backward)
        return e_forward

    def score_edges(self, z, edge_index):
        mu_src, var_src, mu_dst, var_dst = self.gaussian_params(z)
        energy = self.energy_kl(mu_src, var_src, mu_dst, var_dst, edge_index)
        logits = -energy * self.logit_scale.clamp(min=0.1, max=20.0)
        return logits


# Transformer alternative (for CPU or comparison)
class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=2, max_len=64, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = FourierTime(d_model, num_freq=16)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                                   dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x_seq, delta_seq=None):
        N, W, D = x_seq.shape
        pos = torch.arange(W, device=x_seq.device).unsqueeze(0).expand(N, W)
        x = x_seq + self.pos_emb(pos)
        if delta_seq is not None:
            x = x + self.time_emb(delta_seq)
        z = self.transformer(x)
        out = z[:, -1, :]
        return self.output_proj(out)
