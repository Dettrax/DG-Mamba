import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan

torch.manual_seed(10)
torch.cuda.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(10)


@dataclass
class MambaConfig:
    d_model: int
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        attn_mat = None
        for layer in self.layers:
            x,attn_mat = layer(x)

        x = self.norm_f(x)

        return x, attn_mat

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        output,attn_mat = self.mixer(self.norm(x))
        output += x
        return output , attn_mat

    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        _, L, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = F.silu(x)
        y , AttnVecorOverCLS = self.ssm(x)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        return output , AttnVecorOverCLS

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        AttnVecorOverCLS = None
        if self.config.pscan:
            y , AttnVecorOverCLS = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y , AttnVecorOverCLS

    def compute_middle_attn_vector(self,dA,dB, C, x_shape):
        C = C.reshape(C.shape[0],1,C.shape[2],C.shape[1])
        AttnVecorOverCLS = torch.zeros(x_shape).to(dA.device)  # BHL: L vectors per batch and channel
        AttnVecorOverCLS = AttnVecorOverCLS.permute(0, 2, 1)
        cls_pos = x_shape[1] - 1
        for t in range(cls_pos):
            curr_C = C[:, :, :, cls_pos]
            currA = torch.ones(dA.shape[0], dA.shape[2], dA.shape[3]).to(dA.device)
            if t < (cls_pos - 1):
                for i in range(cls_pos - 1 - t):
                    currA = currA * dA[:, cls_pos - 1 - i, :, :]
            currB = dB[:, t, :, :]
            AttnVecorOverCLS[:, :, t] = torch.sum(curr_C * currA * currB, axis=-1)
        return AttnVecorOverCLS

    def compute_attn_matrix_fn(self,dA,dB, C, x_shape, dtype=torch.float16):
        C = C.reshape(C.shape[0],1,C.shape[2],C.shape[1])
        AttnMatrixOverCLS = torch.zeros((x_shape[0], x_shape[2], x_shape[1], x_shape[1]), requires_grad=True).to(
            dtype).to(dA.device)  # BHLL: L vectors per batch and channel

        for r in range(x_shape[1]):
            for c in range(r + 1):
                curr_C = C[:, :, :, r]
                currA = torch.ones((dA.shape[0], dA.shape[2], dA.shape[3]), requires_grad=True, dtype=dtype).to(
                    dA.device)
                if c < r:
                    for i in range(r - c):
                        currA = currA * dA[:, r - i, :, :]
                currB = dB[:, c, :, :]
                AttnMatrixOverCLS[:, :, r, c] = torch.sum(curr_C * currA * currB, axis=-1)
        return AttnMatrixOverCLS
    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x
        AttnVecorOverCLS = self.compute_attn_matrix_fn(deltaA,deltaB, C, y.shape)
        return y , AttnVecorOverCLS

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

    def step(self, x, cache):
        h, inputs = cache

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)

        BX = deltaB * (x.unsqueeze(-1))

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)

        h = deltaA * h + BX

        y = (h @ C.unsqueeze(-1)).squeeze(2)

        y = y + D * x

        return y, h.squeeze(1)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output