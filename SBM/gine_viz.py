# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".
import torch_geometric.transforms as T
import os

try:
    os.chdir("../../TransformerG2G/SBM")
except:
    pass
from models import *
from utils import *
import pickle
import json
from eval_mod import get_MAP_avg

import warnings

warnings.filterwarnings("ignore")

import itertools

from torch.nn import ELU, Dropout

from mamba import MambaConfig, Mamba_q
from mamba_ssm.modules.mamba_simple import Mamba
from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

torch.backends.cudnn.deterministic = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
dim_out = 64
dim_in = 1000

dim_val = 256

f = open("config.json")
config = json.load(f)
lookback = config["lookback"]

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class SBMDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset, self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
        self.transform = T.AddRandomWalkPE(walk_length=16, attr_name='pe')

    def temp_process(self, data, lookback):

        dataset = {}
        for i in range(lookback, 50):
            B = np.zeros((1000, lookback + 1, 1000))
            for j in range(lookback + 1):
                B[:, j, :] = data[i - lookback + j][0].todense()
            dataset[i] = B

        # Construct dict of hops and scale terms
        hop_dict = {}
        scale_terms_dict = {}
        for i in range(lookback, 50):
            hops = get_hops(data[i][0], 2)
            scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                               hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
            hop_dict[i] = hops
            scale_terms_dict[i] = scale_terms

        triplet_dict = {}
        scale_dict = {}
        for i in range(lookback, 50):
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            triplet_dict[i] = triplet
            scale_dict[i] = scale

        return dataset, triplet_dict, scale_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = torch.tensor(data, dtype=torch.float32)
        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]
        return x, triplet, scale


# Get dataset and construct dict
data = get_data()


def Energy_KL(mu, sigma, pairs, L):
    ij_mu = mu[pairs]
    ij_sigma = sigma[pairs]
    sigma_ratio = ij_sigma[:, 1] / (ij_sigma[:, 0] + 1e-14)
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), 1)
    mu_diff_sq = torch.sum(torch.square(ij_mu[:, 0] - ij_mu[:, 1]) / (ij_sigma[:, 0] + 1e-14), 1)
    return 0.5 * (trace_fac + mu_diff_sq - L - log_det)


# Define loss function
def build_loss(triplets, scale_terms, mu, sigma, L, scale):
    hop_pos = torch.stack([torch.tensor(triplets[:, 0]), torch.tensor(triplets[:, 1])], 1).type(torch.int64)
    hop_neg = torch.stack([torch.tensor(triplets[:, 0]), torch.tensor(triplets[:, 2])], 1).type(torch.int64)
    eng_pos = Energy_KL(mu, sigma, hop_pos, L)
    eng_neg = Energy_KL(mu, sigma, hop_neg, L)
    energy = torch.square(eng_pos) + torch.exp(-eng_neg)
    if scale:
        loss = torch.mean(energy * torch.Tensor(scale_terms).cpu())
    else:
        loss = torch.mean(energy)
    return loss


import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GINEConv, global_add_pool
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from typing import Any, Dict, Optional
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch import Tensor


class ApplyConv(torch.nn.Module):
    def __init__(self, mamba_attn, dropout, channels: int, conv: Optional[MessagePassing],
                 norm: Optional[str] = 'batch_norm', norm_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.conv = conv
        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.dropout = dropout  # add this line to set dropout rate
        self.mamba = mamba_attn

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            edge_attr: Optional[Tensor] = None,
            **kwargs,
    ) -> Tensor:
        hs = []
        if self.conv is not None:
            h = self.conv(x, edge_index, edge_attr, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            h = self.norm1(h)
            hs.append(h)
        inp_mamba = x.reshape(1, x.size(0), x.size(1))

        h = self.mamba(inp_mamba)
        h = h.mean(dim=0)
        hs.append(h)

        out = sum(hs)
        return out


def get_graph_data(x):
    transform = T.AddRandomWalkPE(walk_length=16, attr_name='pe')
    x = torch.tensor(x, dtype=torch.float32)

    # Create placeholders for edge_index and edge_attr
    edge_index_list = []
    edge_attr_list = []

    adj_matrix = x
    edge_index, edge_attr = dense_to_sparse(adj_matrix)
    edge_index_list.append(edge_index)
    edge_attr_list.append(edge_attr)

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)

    # Create a pseudo graph data object for PE calculation
    graph_data = Data(x=torch.ones(x.size(0), 1), edge_index=edge_index, edge_attr=edge_attr)
    graph_data = transform(graph_data)

    pe = graph_data.pe  # Positional encodings

    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, pe, edge_index, edge_attr, batch


class MambaG2G(torch.nn.Module):
    def __init__(self, config, lin_dim, dim_out, dim_val, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.config = MambaConfig(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'],
                                  n_layers=1)

        self.mamba = Mamba(expand=1, d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])
        self.conv_mamba = Mamba_q(self.config)
        # Define a sequential model for GINEConv
        nn_model = Sequential(Linear(dim_in, dim_in), ReLU(), Linear(dim_in, dim_in))

        # Correctly instantiate GINEConv with the sequential model
        self.conv = ApplyConv(self.conv_mamba, dropout, dim_in, GINEConv(nn_model))

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)

        self.edge_emb = Embedding(2, dim_in)

    def forward(self, input):
        z = []
        for i in range(input.size(1)):
            x = input[:, i, :]
            x, pe, edge_index, edge_attr, batch = get_graph_data(x)
            edge_attr = self.edge_emb(edge_attr.int())
            x = self.conv(x, edge_index, batch=batch, edge_attr=edge_attr)
            z.append(x)
        z = torch.stack(z, 1)
        e = self.mamba(z)
        e = e.mean(dim=1)  # Average pooling to maintain the expected shape
        e = self.dropout(e)  # Apply dropout after average pooling
        x = torch.tanh(self.out_fc(e))
        x = self.elu(x)
        x = self.dropout(x)  # Apply dropout after the activation
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        sigma = self.elu(sigma) + 1 + 1e-14

        return x, mu, sigma


def optimise_mamba(data, lookback, lin_dim, d_conv, d_state, dropout, lr, weight_decay):
    dataset = SBMDataset(data, lookback)

    config = {
        'd_model': dim_in,
        'd_state': 12,
        'd_conv': 3
    }

    model = MambaG2G(config, lin_dim, dim_out, dim_val, dropout=dropout).to(device)
    # print total model parameters
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_losses = []
    train_loss = []
    test_loss = []
    best_MAP = 0
    for e in tqdm(range(50)):
        model.train()
        loss_step = []
        for i in range(lookback, 35):
            x, triplet, scale = dataset[i]
            x = x.clone().detach().requires_grad_(True).to(device)
            optimizer.zero_grad()
            _, mu, sigma = model(x)
            # loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
            loss = build_loss(triplet, scale, mu, sigma, dim_out, scale=False)
            loss_step.append(loss.cpu().detach().numpy())
            # print(f"Epoch: {e}, Timestamp: {i},Loss: {loss_list[-1]}")
            '''if i==lookback:
                print(triplet)'''
            loss.backward()
            optimizer.step()

    mu_timestamp = []
    sigma_timestamp = []
    with torch.no_grad():
        model.eval()
        for i in range(lookback, 50):
            x, triplet, scale = dataset[i]
            x = x.clone().detach().requires_grad_(False).to(device)
            _, mu, sigma = model(x)
            mu_timestamp.append(mu.cpu().detach().numpy())
            sigma_timestamp.append(sigma.cpu().detach().numpy())

    # Save mu and sigma matrices
    save_sigma_mu = True
    sigma_L_arr = []
    mu_L_arr = []
    if save_sigma_mu == True:
        sigma_L_arr.append(sigma_timestamp)
        mu_L_arr.append(mu_timestamp)
    curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback, data)
    if curr_MAP > best_MAP:
        best_MAP = curr_MAP
        print("Best MAP: ", e, best_MAP, sep=" ")
    return model, val_losses, train_loss, test_loss


lookback = 4
# {'lr': 0.0030654227230925636, 'lin_dim': 47, 'd_conv': 6, 'lookback': 4, 'd_state': 25, 'dropout': 0.3725448646977555, 'weight_decay': 1.02596357976919e-05}
model, val_losses, train_loss, test_loss = optimise_mamba(data, lookback, 47, 6, 25, 0.3725448646977555,
                                                          0.0030654227230925636, 1.02596357976919e-05)

dataset = SBMDataset(data, lookback)

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Assuming attn_mat is obtained as in your provided code snippet
# Here we loop through the timestamps 4 to 8

timestamps = range(5, 35)  # Timestamp range from 4 to 8
node = 6  # Node index 6 (0-based index is 5)

# Calculate the global vmin and vmax
vmin = float('inf')
vmax = float('-inf')

mu_timestamp = []
sigma_timestamp = []
attn_weights_per_timestamps = []
save_list = []
for i in tqdm(range(lookback, 50)):
    with torch.no_grad():
        model.eval()
        x, triplet, scale = dataset[i]
        x = x.clone().detach().requires_grad_(False).to(device)
        _, _, _, = model(x)

    # Extract and normalize attention matrices
    attn_matrix_a = model.mamba.attn_matrix_a.abs()
    attn_matrix_b = model.mamba.attn_matrix_b.abs()
    normalize_attn_mat = lambda attn_mat: (attn_mat.abs() - torch.min(attn_mat.abs())) / (
                torch.max(attn_mat.abs()) - torch.min(attn_mat.abs()))
    attn_matrix_a_normalize = normalize_attn_mat(attn_matrix_a)
    attn_matrix_b_normalize = normalize_attn_mat(attn_matrix_b)
    attn_weights_per_timestamps.append(attn_matrix_a_normalize[5].mean(dim=0).cpu().detach().numpy())
    save_list.append(attn_matrix_a_normalize[5].mean(dim=0).cpu().detach().numpy())

# Save the list to a file
with open('attn_mamba_gine.pkl', 'wb') as f:
    pickle.dump(save_list, f)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming attn_matrices is a list of 63 attention matrices, each of shape 5x5
attn_matrices = attn_weights_per_timestamps  # Replace with your actual attention matrices

# Calculate the global min and max values for consistent color range
vmin = min(attn_mat.min() for attn_mat in attn_matrices)
vmax = max(attn_mat.max() for attn_mat in attn_matrices)

# Create a figure with subplots, max 5 plots per row
n_rows = (len(attn_matrices) + 4) // 5  # Calculate the number of rows needed
fig, axes = plt.subplots(n_rows, 5, figsize=(30, n_rows * 6))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Visualize each attention matrix as a heatmap in its respective subplot
for i, attn_mat in enumerate(attn_matrices):
    sns.heatmap(attn_mat, annot=True, cmap='viridis', cbar=True, ax=axes[i], vmin=vmin, vmax=vmax)
    axes[i].set_title(f'Heatmap {lookback + i + 1}')

# Hide any unused subplots
for j in range(len(attn_matrices), len(axes)):
    fig.delaxes(axes[j])

# Display the figure
plt.tight_layout()
plt.show()
