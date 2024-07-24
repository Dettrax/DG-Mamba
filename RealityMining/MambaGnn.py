# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".

import os

try:
    os.chdir("RealityMining")
except:
    pass
from models import *
from utils import *
import pickle
from exp_mod import get_MAP_avg
import json
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, precision_recall_curve

import warnings

warnings.filterwarnings("ignore")

import itertools

from torch.nn import ELU, Dropout

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

torch.backends.cudnn.deterministic = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# hyperparams
dim_out = 64
dim_in = 96

dim_val = 256



# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Get dataset and construct dict
data = dataset_mit('..')


class RMDataset(Dataset):
    def __init__(self, data, lookback, walk_length=20):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset, self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
        self.transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
        self.data_len = list(self.dataset.keys())[-1]

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


def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices


class MambaConv(torch.nn.Module):

    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            act: str = 'relu',
            att_type: str = 'transformer',
            order_by_degree: bool = False,
            shuffle_ind: int = 0,
            d_state: int = 16,
            d_conv: int = 4,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        assert (self.order_by_degree == True and self.shuffle_ind == 0) or (
                self.order_by_degree == False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'

        config = {
            'd_model': channels,
            'd_state': d_state,
            'd_conv': d_conv
        }
        if self.att_type == 'mamba':
            self.self_attn = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])

        self.mlp = Sequential(
            Linear(channels, channels),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        batch = batch.to(device)
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        if self.att_type == 'mamba':

            if self.order_by_degree:
                deg = degree(edge_index[0], x.shape[0]).to(torch.long)
                deg = deg.to(device)
                order_tensor = torch.stack([batch, deg], 1).T
                _, x = sort_edge_index(order_tensor, edge_attr=x)

            if self.shuffle_ind == 0:
                h, mask = to_dense_batch(x, batch)
                h = self.self_attn(h)
                h = h[mask]
            else:
                mamba_arr = []
                for _ in range(self.shuffle_ind):
                    h_ind_perm = permute_within_batch(x, batch)
                    h_i, mask = to_dense_batch(x[h_ind_perm], batch)
                    h_i = self.self_attn(h_i)[mask][h_ind_perm]
                    mamba_arr.append(h_i)
                h = sum(mamba_arr) / self.shuffle_ind

        ###

        h = F.dropout(h, p=self.dropout, training=self.training)
        h =  h +x   # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')


class GraphModel(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int,
                 d_conv: int, dropout: float, order_by_degree: False, window_size_emd: int, walk_length: int,
                 lookback: int):
        super().__init__()
        self.channels = channels
        self.process_dim = pe_dim + ((lookback + 1) * window_size_emd)
        self.node_emb = Embedding(window_size_emd
                                  , channels)
        self.pe_lin = Linear(walk_length, pe_dim)
        self.pe_norm = BatchNorm1d(walk_length)
        self.edge_emb = Embedding(2, self.channels)
        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        self.dropout = dropout
        self.dropout_ly = Dropout(dropout)
        self.convs = ModuleList()

        for _ in range(num_layers):
            nn = Sequential(
                Linear(self.channels, self.channels),
                ReLU(),
                Linear(self.channels, self.channels),
            )
            if self.model_type == 'gine':
                conv = GINEConv(nn)

            if self.model_type == 'mamba':
                conv = MambaConv(self.channels, GINEConv(nn), heads=4, attn_dropout=self.dropout,
                                 att_type='mamba',
                                 shuffle_ind=self.shuffle_ind,
                                 order_by_degree=self.order_by_degree,
                                 d_state=d_state, d_conv=d_conv)

            # conv = GINEConv(nn)
            self.convs.append(conv)

        self.mu = Linear(self.channels, 64)
        self.sigma = Linear(self.channels, 64)
        self.elu = ELU()
        self.convs_norm = ModuleList([BatchNorm1d(self.channels) for _ in range(num_layers)])
        self.node_emb_norm = BatchNorm1d((lookback + 1) * window_size_emd).to(device)
        self.pe_lin_norm = BatchNorm1d(pe_dim)
        self.x_linear = Linear(self.process_dim, self.channels).to(device)
        self.node_emb_norm_after = BatchNorm1d(self.channels).to(device)

    def forward(self, x, pe, edge_index, edge_attr, batch):

        x_pe_norm = self.pe_norm(pe)

        x_flat = x.view(x.size(0), -1)  # Flatten x to 2D

        x_after_norm = self.node_emb_norm(x_flat)

        x_pe_lin = self.pe_lin(x_pe_norm)
        x_pe_norm = self.pe_lin_norm(x_pe_lin)
        x_pe_drop = self.dropout_ly(x_pe_norm)

        x_cat = torch.cat((x_after_norm, x_pe_drop), dim=1)

        x_after_linear = self.x_linear(x_cat)

        x_after_norm = self.node_emb_norm_after(x_after_linear)
        x_after_drop = self.dropout_ly(x_after_norm)
        edge_attr = self.edge_emb(edge_attr.int())

        torch.cuda.empty_cache()
        for i, conv in enumerate(self.convs):
            if self.model_type == 'gine':
                x_after_drop = conv(x_after_drop, edge_index, edge_attr=edge_attr)
            else:
                x_after_drop = conv(x_after_drop, edge_index, batch, edge_attr=edge_attr)
            x_after_norm = self.convs_norm[i](x_after_drop)
            x_after_drop = self.dropout_ly(x_after_norm)

        mu = torch.sigmoid(self.mu(x_after_drop))
        sigma = self.sigma(x_after_drop)
        sigma = self.elu(sigma) + 1 + 1e-14
        return mu, sigma


def optimise_mamba(lookback, window_size, stride, channel, pe_dim, num_layers, d_conv, d_state, dropout, lr,
                   weight_decay, walk_length):
    # Create dataset
    dataset = RMDataset(data, lookback, walk_length)

    model = GraphModel(channels=channel, pe_dim=pe_dim, num_layers=num_layers,
                       model_type='mamba',
                       shuffle_ind=2, order_by_degree=False,
                       d_conv=d_conv, d_state=d_state, dropout=dropout, window_size_emd=window_size,
                       walk_length=walk_length, lookback=lookback
                       ).to(device)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    val_losses = []
    train_loss = []
    for e in tqdm(range(50)):
        model.train()
        loss_step = []
        for i in range(lookback, 63):
            x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
            optimizer.zero_grad()
            x = x.clone().detach().requires_grad_(True).to(device)
            pe = pe.clone().detach().requires_grad_(True).to(device)
            edge_index = edge_index.clone().detach().to(device)
            edge_attr = edge_attr.clone().detach().requires_grad_(True).to(device)
            batch = batch.clone().detach().to(device)
            mu, sigma = model(x, pe, edge_index, edge_attr,
                              batch)
            loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)

            loss_step.append(loss.cpu().detach().numpy())
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    mu_timestamp = []
    sigma_timestamp = []
    with torch.no_grad():
        model.eval()
        for i in range(lookback, 90):
            x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
            x = x.clone().detach().requires_grad_(False).to(device)
            pe = pe.clone().detach().requires_grad_(False).to(device)
            edge_index = edge_index.clone().detach().to(device)
            edge_attr = edge_attr.clone().detach().requires_grad_(False).to(device)
            batch = batch.clone().detach().to(device)
            mu, sigma = model(x, pe, edge_index, edge_attr, batch)
            mu_timestamp.append(mu.cpu().detach().numpy())
            sigma_timestamp.append(sigma.cpu().detach().numpy())

    name = 'Results/RealityMining'
    save_sigma_mu = True
    sigma_L_arr = []
    mu_L_arr = []
    if save_sigma_mu:
        sigma_L_arr.append(sigma_timestamp)
        mu_L_arr.append(mu_timestamp)
    MAPS = []
    MRRS = []
    for i in range(5):
        try:
            MAP, MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback, data)
        except:
            MAP = 0
            MRR = 0
        MAPS.append(MAP)
        MRRS.append(MRR)
    print(f"MAP: {np.mean(MAPS)} MRR: {np.mean(MRRS)}")


    return model , val_losses , train_loss


#{'lookback': 2, 'channel': 31, 'num_layers': 4, 'd_conv': 2, 'd_state': 15, 'dropout': 0.3213168331944855, 'lr': 2.0725755122766534e-05, 'weight_decay': 0.00039996048924527774}
# 'lookback': 1, 'channel': 4, 'num_layers': 4, 'd_conv': 3, 'd_state': 18, 'dropout': 0.22818672254820538, 'lr': 1.8856123947228383e-05, 'weight_decay': 0.0002582024187228026}
model , val_losses , loss_step  = optimise_mamba(2,96,1,31,4,4,2,15,0.3213168331944855,2.0725755122766534e-05,0.00039996048924527774,16)


#pplot loss
#add legend
# y title and x title for loss vs epoch
# from matplotlib import pyplot as plt
#
# plt.semilogy(val_losses)
# plt.semilogy(loss_step)
# plt.legend(['Validation Loss','Training Loss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

lookback = 2
dataset = RMDataset(data, lookback, 16)
# mu_timestamp = []
# sigma_timestamp = []
# with torch.no_grad():
#     model.eval()
#     for i in range(lookback, 90):
#         x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
#         x = x.clone().detach().requires_grad_(False).to(device)
#         pe = pe.clone().detach().requires_grad_(False).to(device)
#         edge_index = edge_index.clone().detach().to(device)
#         edge_attr = edge_attr.clone().detach().requires_grad_(False).to(device)
#         batch = batch.clone().detach().to(device)
#         mu, sigma = model(x, pe, edge_index, edge_attr, batch)
#         mu_timestamp.append(mu.cpu().detach().numpy())
#         sigma_timestamp.append(sigma.cpu().detach().numpy())
#
# name = 'Results/RealityMining'
# save_sigma_mu = True
# sigma_L_arr = []
# mu_L_arr = []
# if save_sigma_mu:
#     sigma_L_arr.append(sigma_timestamp)
#     mu_L_arr.append(mu_timestamp)
# for i in range(5):
#     MAP, MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback, data)
#     print(f"MAP: {MAP} MRR: {MRR}")

# if save_sigma_mu == True:
#     if not os.path.exists(name+'/Eval_Results/saved_array'):
#         os.makedirs(name+'/Eval_Results/saved_array')
#     with open(name+'/Eval_Results/saved_array/sigma_as','wb') as f: pickle.dump(sigma_L_arr, f)
#     with open(name+'/Eval_Results/saved_array/mu_as','wb') as f: pickle.dump(mu_L_arr, f)
