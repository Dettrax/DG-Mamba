# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".
import torch_geometric.transforms as T
import os
import sys

try :
    os.chdir("RealityMining")
except:
    pass
from models import *
from utils import *
import pickle
import json
import ray
from ray import tune
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

from exp_mod import get_MAP_avg
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



f = open("config.json")
config = json.load(f)
lookback = config["lookback"]

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)






class RMDataset(Dataset):
    def __init__(self, data, lookback,walk_length=20):
        self.data = data
        self.lookback = lookback

        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
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


class MambaG2G(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)

    def forward(self, input):
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

        return x, mu, sigma

data = dataset_mit('..')


def optimise_mamba(data,lookback, dim_in, d_conv, d_state, dropout, lr, weight_decay):
    walk_length = 16

    # Create dataset
    dataset = RMDataset(data, lookback, walk_length)
    config = {
        'd_model': 96,
        'd_state': d_state,
        'd_conv': d_conv
    }

    model = MambaG2G(config, dim_in, 64, dropout=dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-5)
    diff = dataset.data_len - lookback
    for e in range(10):
        model.train()
        loss_step = []
        for i in range(lookback, lookback + int(diff * 0.7)):
            x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
            optimizer.zero_grad()
            x = x.clone().detach().requires_grad_(True).to(device)

            _, mu, sigma = model(x)
            loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)

            loss_step.append(loss.cpu().detach().numpy())
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    f_MAP = []
    for i in range(1):
        mu_timestamp = []
        sigma_timestamp = []
        with torch.no_grad():
            model.eval()
            for i in range(lookback, 90):
                x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
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
        f_MAP.append(MAP)

    return np.mean(f_MAP)



lookback = 2
walk = 16
# model , val_losses , loss_step , test_loss = optimise_mamba(lookback=lookback,dim_in=24,d_conv=6,d_state=8,dropout=0.40,lr=0.00045,weight_decay=0.00037,walk_length=walk)
#

# print(optimise_mamba(data,lookback=lookback,dim_in=24,d_conv=6,d_state=8,dropout=0.40,lr=0.00045,weight_decay=0.00037))
#


def train_model(config):
    map_value = optimise_mamba(data,lookback=lookback, dim_in=config['dim_in'], d_conv=config['d_conv'],d_state=config['d_state'],dropout=config['dropout'],lr=config['lr'],weight_decay=config['weight_decay'])
    return map_value


def objective(config):  # ①
    while True:
        acc = train_model(config)
        train.report({"MAP": acc})  # Report to Tunevvvvvvv


ray.init(  runtime_env={
            "working_dir": str(os.getcwd()),
        })  # Initialize Ray


search_space = {"lr": tune.loguniform(1e-4, 1e-2)
    ,"dim_in": tune.randint(16, 100),
                "d_conv": tune.randint(2, 10),
                "d_state": tune.randint(2, 10),
                "dropout": tune.uniform(0.1, 0.5),
                "weight_decay": tune.loguniform(1e-5, 1e-3)}

# Create an Optuna search space
algo = OptunaSearch(
)


tuner = tune.Tuner(  # ③
    tune.with_resources(
        tune.with_parameters(objective),
        resources={"gpu": 1}
    ),
    tune_config=tune.TuneConfig(
        metric="MAP",
        mode="max",
        search_alg=algo,
        num_samples=12
    ),
    param_space=search_space,
    run_config=train.RunConfig(
        stop={"training_iteration": 1}  # Limit the training iterations to 1
    )
)

results = tuner.fit()
print("Best config is:", results.get_best_result().config)

#1 7min 53s
#0.5 4min 14s
#0.25 2min 26s , 2min 27s
#0.125 1min 55s