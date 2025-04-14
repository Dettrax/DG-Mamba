import torch
from mamba_ssm import Mamba2

# Original dimensions
batch, length, dim = 8, 4, 92
x = torch.randn(batch, length, dim).to("cuda")

# Model configuration with ALL dimensions divisible by 8
model = Mamba2(
    d_model=96,     # Rounded up from 92 to be divisible by 8
    d_state=64,     # Changed from 6 to 64 (must be divisible by 8)
    d_conv=4,       # Changed from 3 to 4 to be even (better for performance)
    expand=2,       # This is a multiplier
    headdim=16,     # This is already divisible by 8
).to("cuda")

# Pad input to match model dimension
pad = torch.zeros(batch, length, 96-dim, device="cuda")
padded_x = torch.cat([x, pad], dim=-1)

# Forward pass
y = model(padded_x)

# For output, trim back to original dimension
trimmed_y = y[:, :, :dim]
assert trimmed_y.shape == x.shape