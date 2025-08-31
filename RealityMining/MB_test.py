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
from utils_mod import *
import pickle
import json
from exp_mod import get_MAP_avg


import warnings
warnings.filterwarnings("ignore")

import itertools

from torch.nn import ELU,Dropout

from mamba_ssm import Mamba
from tqdm import tqdm


from torch.nn.utils import clip_grad_norm_

from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

from torch_geometric.nn import GINEConv, global_add_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

# from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index

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



# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Get dataset and construct dict
data = dataset_mit('..',undirected=True)

def scan_dataset(dset, undirected=False, max_bins=5):
    total_sym_mismatch = 0
    for t in range(len(dset)):
        A, X = dset[t]
        N = A.shape[0]
        # shape & counts
        assert X._nnz() == A.nnz + N, f"[t={t}] X nnz {X._nnz()} != A nnz {A.nnz} + N {N}"
        assert A.diagonal().sum() == 0, f"[t={t}] A has self-loops"
        # symmetry?
        if undirected:
            mm = (A != A.T).nnz
            total_sym_mismatch += mm
            if t < max_bins:
                print(f"[t={t}] edges={A.nnz}, symmetry_mismatch={mm}")
        elif t < max_bins:
            print(f"[t={t}] edges={A.nnz} (directed OK)")
    if undirected:
        print(f"Total symmetry mismatches across all bins: {total_sym_mismatch}")

# example run:
scan_dataset(data, undirected=False)
scan_dataset(data, undirected=True)

class RMDataset(Dataset):
    def __init__(self, data, lookback, walk_length=20):
        self.data = data
        self.lookback = lookback
        self.dataset, self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
        self.transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')

    def temp_process(self, data, lookback, K: int = 2):
        """
        Build:
          - dataset[i]: dense stack of adjacencies for times [i-lookback, ..., i],
                        shape [N, lookback+1, N]
          - triplet_dict_data[i], scale_dict_data[i]: from utils.sample_hops + utils.to_triplets
        """
        dataset = {}

        # Global node count (fixed across bins)
        N = int(data[0][0].shape[0])
        T = len(data)  # <-- don't hard-code 90

        # 1) Dense windows for the sequence model
        for i in range(lookback, T):
            B = np.zeros((N, lookback + 1, N), dtype=np.float32)
            for j in range(lookback + 1):
                Aij = data[i - lookback + j][0]  # CSR
                # Aij is already N x N; convert to float32 numpy
                B[:, j, :] = Aij.toarray().astype(np.float32, copy=False)
            dataset[i] = B

        # 2) Triplets per time using utils’ sampler (no manual scale math)
        triplet_dict = {}
        scale_dict = {}
        for i in range(lookback, T):
            A_i = data[i][0]  # CSR at time i
            sampled_hops, scale_terms = sample_hops(A_i, K=K)  # from utils.py
            triplet, scale = to_triplets(sampled_hops, scale_terms)
            triplet_dict[i] = triplet
            scale_dict[i] = scale

        return dataset, triplet_dict, scale_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # idx is an absolute time index in [lookback, len(data)-1]
        x_np = self.dataset[idx]  # [N, lookback+1, N] dense (0/1)
        x = torch.tensor(x_np, dtype=torch.float32)

        # --- Build a SINGLE graph for positional encodings using the UNION of edges over the window ---
        # Union across the time dimension (axis=1) -> [N, N] boolean
        union_adj = (x > 0).any(dim=1).float()  # avoid parallel edges from concatenation
        edge_index, edge_attr = dense_to_sparse(union_adj)  # edge_attr are 1's

        graph_data = Data(x=torch.ones(x.size(0), 1),
                          edge_index=edge_index,
                          edge_attr=edge_attr)
        graph_data = self.transform(graph_data)
        pe = graph_data.pe

        batch = torch.zeros(x.size(0), dtype=torch.long)

        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]

        # Shapes returned:
        # x:        [N, lookback+1, N]
        # pe:       [N, d_pe]   (from AddRandomWalkPE)
        # edge_*:   union graph over the window
        # triplet:  [M, 3],   scale: [M]
        return x, triplet, scale

def get_graph_data(x):
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

    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, edge_index, edge_attr, batch


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


class MambaG2G(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super().__init__()
        self.D = dim_in
        self.elu = torch.nn.ELU()
        self.mamba = Mamba(d_model=config['d_model'],d_state=config['d_state'],d_conv=config['d_conv'] )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.out_fc = torch.nn.Linear(config['d_model'], self.D)
        self.sigma_fc = torch.nn.Linear(self.D, dim_out)
        self.mu_fc = torch.nn.Linear(self.D, dim_out)

    def forward(self, x):                       # x: [N, lookback+1, N]
        y = self.mamba(x)                   # [N, lookback+1, d_model]
        y = y.mean(dim=1)                      # pool over time -> [N, d_model]
        y = self.dropout(y)
        y = torch.tanh(self.out_fc(y))         # -> [N, D]
        y = self.elu(y)
        y = self.dropout(y)
        mu = self.mu_fc(y)                     # [N, dim_out]
        sigma = self.sigma_fc(y)               # [N, dim_out]
        sigma = F.softplus(sigma) + 1e-6       # strictly positive, numerically safe
        return y, mu, sigma


def optimise_mamba(data,lookback,dim_in,d_conv,d_state,dropout,lr,weight_decay):


    # Create dataset
    dataset = RMDataset(data, lookback)
    config = {
        'd_model':96,
        'd_state':d_state,
        'd_conv':d_conv
    }
    # hyperparams
    dim_out = 64
    dim_in = 96

    dim_val = 64
    dim_attn = 64
    lr = 0.0001

    n_heads = 1
    n_encoder_layers = 1
    model = MambaG2G(config, dim_in, 64, dropout=dropout).to(device)
    #print total model parameters
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in tqdm(range(50)):
        model.train()
        loss_step = []
        for i in range(lookback, 63):
                x, triplet, scale = dataset[i]
                optimizer.zero_grad()
                # x = x.clone().detach().requires_grad_(True).to(device)
                _,mu, sigma = model(x.to(device))
                loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)

                loss_step.append(loss.cpu().detach().numpy())
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        f_MAP = []
        if e % 5 == 0:
            for i in range(1):
                mu_timestamp = []
                sigma_timestamp = []
                with torch.no_grad():
                    model.eval()
                    for i in range(lookback, 90):
                        x, triplet, scale = dataset[i]
                        x = x.clone().detach().requires_grad_(False).to(device)
                        _, mu, sigma = model(x)
                        mu_timestamp.append(mu.cpu().detach().numpy())
                        sigma_timestamp.append(sigma.cpu().detach().numpy())

                # Save mu and sigma matrices
                name = 'Results/RealityMining'
                save_sigma_mu = True
                sigma_L_arr = []
                mu_L_arr = []
                if save_sigma_mu == True:
                    sigma_L_arr.append(sigma_timestamp)
                    mu_L_arr.append(mu_timestamp)

                MAP,_ = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback,data)
                print("Loss: ", np.mean(loss_step), "MAP: ", MAP)

    return model


#{'lr': 2.2307858381535968e-05, 'dim_in': 49, 'lookback': 4, 'd_conv': 3, 'd_state': 6, 'dropout': 0.17661562119283333, 'weight_decay': 1.466563344626497e-05}
lookback = 2
model = optimise_mamba(data,lookback=lookback,dim_in=96,d_conv=3,d_state=32,dropout=0.1766,lr=2.5e-05,weight_decay=1.4e-03)

dataset = RMDataset(data, lookback)
#read the best_model.pt
# model.load_state_dict(torch.load('best_model.pth'))
mu_timestamp = []
sigma_timestamp = []
with torch.no_grad():
    model.eval()
    for i in range(lookback, 90):
        x, triplet, scale = dataset[i]
        x = x.clone().detach().requires_grad_(True).to(device)
        _, mu, sigma = model(x)
        mu_timestamp.append(mu.cpu().detach().numpy())
        sigma_timestamp.append(sigma.cpu().detach().numpy())
name = 'Results/RealityMining'
save_sigma_mu = True
sigma_L_arr = []
mu_L_arr = []
if save_sigma_mu == True:
    sigma_L_arr.append(sigma_timestamp)
    mu_L_arr.append(mu_timestamp)

import time
start = time.time()
MAPS = []
MRR = []
for i in tqdm(range(1)):
    curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback,data)
    MAPS.append(curr_MAP)
    MRR.append(curr_MRR)
#print mean and std of map and mrr
print("Mean MAP: ", np.mean(MAPS))
print("Mean MRR: ", np.mean(MRR))
print("Std MAP: ", np.std(MAPS))
print("Std MRR: ", np.std(MRR))
print("Time taken: ", time.time() - start)

if save_sigma_mu == True:
    if not os.path.exists(name + '/Eval_Results/saved_array'):
        os.makedirs(name + '/Eval_Results/saved_array')
    with open(name + '/Eval_Results/saved_array/sigma_as', 'wb') as f:
        pickle.dump(sigma_L_arr, f)
    with open(name + '/Eval_Results/saved_array/mu_as', 'wb') as f:
        pickle.dump(mu_L_arr, f)