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
from exp_mod import get_MAP_avg
import ray
from ray import tune
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

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



# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Get dataset and construct dict
data = dataset_mit('..')



class RMDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)


    def temp_process(self, data, lookback):
        dataset = {}
        for i in range(lookback, 90):
            B = np.zeros((96, lookback + 1, 96))
            for j in range(lookback + 1):
                adj_matr = data[i - lookback + j][0].todense()
                B[:adj_matr.shape[0], j, :adj_matr.shape[1]] = adj_matr
            dataset[i] = torch.tensor(B).clone().detach().requires_grad_(True).to(device)

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

        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]
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


import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class ApplyConv(torch.nn.Module):
    def __init__(self,mamba_attn,dropout, channels: int, conv: Optional[MessagePassing], norm: Optional[str] = 'batch_norm', norm_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.conv = conv
        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.dropout = dropout # add this line to set dropout rate
        self.mamba = mamba_attn
        self.pe_lin = Linear(2, channels)

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
            h = h + x #96,96
            h = self.norm1(h)
            hs.append(h)

        inp_mamba = x.reshape(1,x.size(0), x.size(1)) #1,96,96  Batch , time stamp , features

        h = self.mamba(inp_mamba)
        h = h.mean(dim=0) #96,96
        hs.append(h)

        out = sum(hs) #96,96
        return out

class MambaG2G(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])
        self.conv_mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])
        # Define a sequential model for GINEConv
        nn_model = Sequential(Linear(96, 96), ReLU(), Linear(96, 96))

        # Correctly instantiate GINEConv with the sequential model
        self.conv = ApplyConv(self.conv_mamba,dropout,96, GINEConv(nn_model))

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)
        self.edge_emb = Embedding(2, 96)

    def forward(self, input):
        # e = self.enc_input_fc(input)
        z = []
        for i in range(input.size(1)):
            x = input[:, i, :]
            x, edge_index, edge_attr, batch = get_graph_data(x)
            edge_attr = self.edge_emb(edge_attr.int())
            x = self.conv(x,edge_index, batch=batch, edge_attr=edge_attr)
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


def optimise_mamba(data,lookback,dim_in,d_conv,d_state,dropout,lr,weight_decay):


    # Create dataset
    dataset = RMDataset(data, lookback)
    config = {
        'd_model':96,
        'd_state':d_state,
        'd_conv':d_conv
    }

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
                _,mu, sigma = model(x)
                loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)

                loss_step.append(loss.cpu().detach().numpy())
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
    # f_MAP = []
    # for i in range(3):
    #     mu_timestamp = []
    #     sigma_timestamp = []
    #     with torch.no_grad():
    #         model.eval()
    #         for i in range(lookback, 90):
    #             x, triplet, scale = dataset[i]
    #             x = x.clone().detach().requires_grad_(False).to(device)
    #             _, mu, sigma = model(x)
    #             mu_timestamp.append(mu.cpu().detach().numpy())
    #             sigma_timestamp.append(sigma.cpu().detach().numpy())
    #
    #     # Save mu and sigma matrices
    #     name = 'Results/RealityMining'
    #     save_sigma_mu = True
    #     sigma_L_arr = []
    #     mu_L_arr = []
    #     if save_sigma_mu == True:
    #         sigma_L_arr.append(sigma_timestamp)
    #         mu_L_arr.append(mu_timestamp)
    #
    #     MAP,_ = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback,data)
    #     f_MAP.append(MAP)

    return model


#{'lr': 2.2307858381535968e-05, 'dim_in': 49, 'lookback': 4, 'd_conv': 3, 'd_state': 6, 'dropout': 0.17661562119283333, 'weight_decay': 1.466563344626497e-05}
lookback = 2
model = optimise_mamba(data,lookback=lookback,dim_in=49,d_conv=3,d_state=6,dropout=0.1766,lr=2.2307858381535968e-05,weight_decay=1.466563344626497e-05)

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
for i in tqdm(range(5)):
    curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback,data)
    MAPS.append(curr_MAP)
    MRR.append(curr_MRR)
#print mean and std of map and mrr
print("Mean MAP: ", np.mean(MAPS))
print("Mean MRR: ", np.mean(MRR))
print("Std MAP: ", np.std(MAPS))
print("Std MRR: ", np.std(MRR))
print("Time taken: ", time.time() - start)




#
# def train_model(config):
#     map_value = optimise_mamba(data,lookback = config['lookback'], dim_in=config['dim_in'], d_conv=config['d_conv'],d_state=config['d_state'],dropout=config['dropout'],lr=config['lr'],weight_decay=config['weight_decay'],walk_length=walk)
#     return map_value
#
#
# def objective(config):  # ①
#     while True:
#         acc = train_model(config)
#         train.report({"MAP": acc})  # Report to Tunevvvvvvv
#
#
# ray.init(  runtime_env={
#             "working_dir": str(os.getcwd()),
#         })  # Initialize Ray
#
#
# search_space = {"lr": tune.loguniform(1e-5, 1e-2)
#         ,"dim_in": tune.randint(16, 100),
#         "lookback": tune.randint(1,5),
#                 "d_conv": tune.randint(2, 10),
#                 "d_state": tune.randint(2, 50),
#                 "dropout": tune.uniform(0.1, 0.5),
#                 "weight_decay": tune.loguniform(1e-5, 1e-3)}
#
# # Create an Optuna search space
# algo = OptunaSearch(
# )
#
#
# tuner = tune.Tuner(  # ③
#     tune.with_resources(
#         tune.with_parameters(objective),
#         resources={"gpu": 0.25}
#     ),
#     tune_config=tune.TuneConfig(
#         metric="MAP",
#         mode="max",
#         search_alg=algo,
#         num_samples=100
#     ),
#     param_space=search_space,
#     run_config=train.RunConfig(
#         stop={"training_iteration": 1}  # Limit the training iterations to 1
#     )
# )
#
# results = tuner.fit()
# print("Best config is:", results.get_best_result().config)

#{'lr': 2.6152417925517384e-05, 'dim_in': 71, 'lookback': 4, 'd_conv': 8, 'd_state': 6, 'dropout': 0.14669919601710057, 'weight_decay': 3.427957936022128e-05}
