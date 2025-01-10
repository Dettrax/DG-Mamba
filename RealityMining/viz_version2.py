#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:45:18 2024

@author: ap2934
"""

# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".
import torch_geometric.transforms as T
import os
try :
    os.chdir("RealityMining")
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

from torch.nn import ELU,Dropout

from mamba_ssm.modules.mamba_simple import Mamba
from tqdm import tqdm


from torch.nn.utils import clip_grad_norm_



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse



torch.backends.cudnn.deterministic=True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



f = open("config.json")
config = json.load(f)
lookback = config["lookback"]

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Get dataset and construct dict
data = dataset_mit('..')



class RMDataset(Dataset):
    def __init__(self, data, lookback,walk_length=20):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
        self.transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')


    def temp_process(self, data, lookback):
        dataset = {}
        for i in range(lookback, 90):
            B = np.zeros((96, lookback + 1, 96))
            for j in range(lookback + 1):
                adj_matr = data[i - lookback + j][0].todense()
                B[:adj_matr.shape[0], j, :adj_matr.shape[1]] = adj_matr
            dataset[i] = B

        # Construct dict of hops and scale terms
        hop_dict = {}
        scale_terms_dict = {}
        print("Constructing dict of hops and scale terms")
        for i in range(lookback, 90):
            hops = get_hops(data[i][0], 2)
            scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                               hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
            hop_dict[i] = hops
            scale_terms_dict[i] = scale_terms

        # Construct dict of triplets
        triplet_dict = {}
        scale_dict = {}
        print("Constructing dict of triplets")
        for i in range(lookback, 90):
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            triplet_dict[i] = triplet
            scale_dict[i] = scale
        return dataset, triplet_dict, scale_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = torch.tensor(data, dtype=torch.float32)

        # Create placeholders for edge_index and edge_attr
        edge_index_list = []
        edge_attr_list = []

        for i in range(x.size(1)):  # Iterate over lookback + 1 adjacency matrices
            adj_matrix = x[:, i, :]
            edge_index, edge_attr = dense_to_sparse(adj_matrix)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        # Create a pseudo graph data object for PE calculation
        graph_data = Data(x=torch.ones(x.size(0), 1), edge_index=edge_index, edge_attr=edge_attr)
        graph_data = self.transform(graph_data)

        pe = graph_data.pe  # Positional encodings

        batch = torch.zeros(x.size(0), dtype=torch.long)

        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]
        return x, pe, edge_index, edge_attr, batch, triplet, scale




# def val_loss(t):
#     l = []
#     for j in range(63, 72):
#         _, muval, sigmaval = t(val_data[j])
#         val_l = build_loss(triplet_dict[j], scale_dict[j], muval, sigmaval, 64, scale=False)
#         l.append(val_l.cpu().detach().numpy())
#     return np.mean(l)


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


class MambaG2G(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.mamba = Mamba(expand=1,d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)

    def forward(self, input,edge_index): # 96,5,96
        # e = self.enc_input_fc(input)
        e = self.mamba(input)
        e = e.mean(dim=1)  # Average pooling to maintain the expected shape
        e = self.dropout(e)  # Apply dropout after average pooling
        x = torch.tanh(self.out_fc(e))
        x = self.elu(x)
        x = self.dropout(x)  # Apply dropout after the activation
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        sigma = self.elu(sigma) + 1 + 1e-14

        return x, mu, sigma # 96,96,5,5


def optimise_mamba(lookback,dim_in,d_conv,d_state,dropout,lr,weight_decay,walk_length):


    # Create dataset
    dataset = RMDataset(data, lookback,walk_length)
    config = {
        'd_model':96,
        'd_state':d_state,
        'd_conv':d_conv
    }

    model = MambaG2G(config, dim_in, 64, dropout=dropout).to(device)
    #print total model parameters
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Define parameters
    epochs = 50

    A_matrices = []
    for e in tqdm(range(epochs)):
        model.train()
        loss_step = []
        for i in range(lookback, 63):
                x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
                optimizer.zero_grad()
                x = x.clone().detach().requires_grad_(True).to(device)
                edge_index = edge_index.clone().detach().to(device)
                _,mu, sigma = model(x,edge_index)
                loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)

                loss_step.append(loss.cpu().detach().numpy())
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

    return model,dataset


lookback = 4
walk = 16
model,dataset = optimise_mamba(lookback=lookback,dim_in=76,d_conv=3,d_state=12,dropout=0.4285,lr=0.000120,weight_decay=2.4530158734036414e-05,walk_length=walk)

mu_timestamp = []
sigma_timestamp = []
with torch.no_grad():
    model.eval()
    for i in range(lookback, 90):
        x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
        x = x.clone().detach().requires_grad_(True).to(device)
        edge_index = edge_index.clone().detach().to(device)
        _, mu, sigma = model(x, edge_index)
        mu_timestamp.append(mu.cpu().detach().numpy())
        sigma_timestamp.append(sigma.cpu().detach().numpy())

name = 'Results/RealityMining'
save_sigma_mu = True
sigma_L_arr = []
mu_L_arr = []
if save_sigma_mu == True:
    sigma_L_arr.append(sigma_timestamp)
    mu_L_arr.append(mu_timestamp)
curr_MAP ,_ = get_MAP_avg(mu_L_arr,lookback,data)
print(curr_MAP)


import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Assuming attn_mat is obtained as in your provided code snippet
# Here we loop through the timestamps 4 to 8

timestamps = range(5, 63)  # Timestamp range from 4 to 8
node = 6  # Node index 6 (0-based index is 5)

# Calculate the global vmin and vmax
vmin = float('inf')
vmax = float('-inf')

mu_timestamp = []
sigma_timestamp=[]
attn_weights_per_timestamps = []
for i in tqdm(range(lookback,90)):
    with torch.no_grad():
        model.eval()
        x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[5]
        x = x.clone().detach().requires_grad_(False).to(device)
        edge_index = edge_index.clone().detach().to(device)
        _, _, _, = model(x, edge_index)

    # Extract and normalize attention matrices
    attn_matrix_a = model.mamba.attn_matrix_a.abs()
    attn_matrix_b = model.mamba.attn_matrix_b.abs()
    normalize_attn_mat = lambda attn_mat : (attn_mat.abs() - torch.min(attn_mat.abs())) / (torch.max(attn_mat.abs()) - torch.min(attn_mat.abs()))
    attn_matrix_a_normalize = normalize_attn_mat(attn_matrix_a)
    attn_matrix_b_normalize = normalize_attn_mat(attn_matrix_b)
    attn_weights_per_timestamps.append(attn_matrix_a[5].mean(dim=0).cpu().detach().numpy())


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
    axes[i].set_title(f'Heatmap {lookback+i+1}')

# Hide any unused subplots
for j in range(len(attn_matrices), len(axes)):
    fig.delaxes(axes[j])

# Display the figure
plt.tight_layout()
plt.show()

