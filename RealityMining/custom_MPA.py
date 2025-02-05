# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".



import torch_geometric.transforms as T
import os
import sys
try :
    os.chdir("RealityMining")
    sys.path.append(os.getcwd())
except:
    pass
from utils import *
import pickle
import json
from eval_mod import get_MAP_avg
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, precision_recall_curve

import warnings
warnings.filterwarnings("ignore")

import itertools

from torch.nn import ELU,Dropout

from mamba_ssm import Mamba
from tqdm import tqdm


from torch.nn.utils import clip_grad_norm_



import torch
import torch.nn as nn
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
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from RealityMining.mamba import Mamba, MambaConfig

class MambaG2G(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.config = MambaConfig(d_model=config['d_model'], n_layers=1,d_state=config['d_state'], d_conv=config['d_conv'])
        self.mamba = Mamba(self.config)
        #self.mamba = MambaBlock(seq_len=lookback + 1, d_model=config['d_model'], state_size=config['d_state'], batch_size=96, device=device)
        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)

    def forward(self, input):
        # e = self.enc_input_fc(input)
        e = self.mamba(input)[0]
        e = e.mean(dim=1)  # Average pooling to maintain the expected shape
        e = self.dropout(e)  # Apply dropout after average pooling
        x = torch.tanh(self.out_fc(e))
        x = self.elu(x)
        x = self.dropout(x)  # Apply dropout after the activation
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        sigma = self.elu(sigma) + 1 + 1e-14

        return x, mu, sigma


lookback = 2
walk = 16

# Create dataset
dataset = RMDataset(data, lookback,16)
config = {
    'd_model':96,
    'd_state':6,
    'd_conv':9
}

model = MambaG2G(config, 76, 64, dropout=0.4).to(device)
#print total model parameters
print('Total parameters:', sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=2.4e-5)
# Define parameters
epochs = 50
# To store A matrices across timestamps and epochs
val_losses = []
train_loss = []
test_loss = []
best_MAP = 0
best_model = None
mu_epoch = []
for e in tqdm(range(epochs)):
    model.train()
    loss_step = []
    i = lookback
    for i in range(lookback, 63):
            x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
            optimizer.zero_grad()
            x = x.clone().detach().requires_grad_(True).to(device)
            _,mu, sigma = model(x)

            loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)

            loss_step.append(loss.cpu().detach().numpy())
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

mu_epoch = []
with torch.no_grad():
    model.eval()
    for i in range(lookback, 90):
        x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
        x = x.clone().detach().requires_grad_(True).to(device)
        _, mu, sigma = model(x)
        mu_epoch.append(mu[:5].cpu().detach().numpy())

import matplotlib.pyplot as plt
import networkx as nx
import umap
import matplotlib.animation as animation
import torch

# Assume that the following have already been defined and initialized:
# - data: your original data dictionary where each key t maps to (adjacency matrix, ...)
# - dataset: an instance of RMDataset (or similar) where dataset[t] returns (x, pe, edge_index, edge_attr, batch, triplet, scale)
# - model: your trained model which, given x, returns (_, mu, sigma)
# - device: the torch device (e.g., torch.device("cuda:0") or "cpu")

# Get a sorted list of timestamps for which you want to animate.
sorted_timestamps = sorted(dataset.dataset.keys())

# Create the figure for the animation.
fig, ax = plt.subplots(figsize=(8, 8))

# Fit UMAP once on the first timestamp for stable projections
x_init, _, _, _, _, _, _ = dataset[sorted_timestamps[0]]
x_init = x_init.clone().detach().requires_grad_(True).to(device)
_, mu_init, _ = model(x_init)
mu_np_init = mu_init.cpu().detach().numpy()
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_reducer.fit(mu_np_init)

# Store initial positions of nodes
initial_positions = {i: pos for i, pos in enumerate(umap_reducer.transform(mu_np_init))}


def update(frame):
    ax.clear()
    current_t = sorted_timestamps[frame]
    adj_matrix = data[current_t][0]

    try:
        G = nx.from_scipy_sparse_array(adj_matrix)
    except AttributeError:
        G = nx.from_scipy_sparse_matrix(adj_matrix)

    x, _, _, _, _, _, _ = dataset[current_t]
    x = x.clone().detach().requires_grad_(True).to(device)
    _, mu, _ = model(x)
    mu_np = mu.cpu().detach().numpy()
    embeddings_2d = umap_reducer.transform(mu_np)

    # Update positions only if they change, else keep the previous ones
    for i in range(len(embeddings_2d)):
        if not np.array_equal(initial_positions[i], embeddings_2d[i]):
            initial_positions[i] = embeddings_2d[i]

    pos = initial_positions

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='lightgreen', alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

    ax.set_title(f"Graph at Timestamp {current_t}")
    ax.axis("off")
    return ax,


ani = animation.FuncAnimation(fig, update, frames=len(sorted_timestamps), interval=1000, blit=False)
plt.show()
ani.save('graph_animation.gif', writer='imagemagick', fps=1)
