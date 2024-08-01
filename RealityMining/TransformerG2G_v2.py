# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".

import os
import sys

try :
    os.chdir("RealityMining")
except:
    pass
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from models import *
from utils import *
import pickle
import json
import ray
from ray import tune
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from exp_mod import get_MAP_avg
# hyperparams
dim_out = 64
dim_in = 96

dim_val = 256


f = open("config.json")
config = json.load(f)


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# init network and optimizer

# Get dataset and construct dict
data = dataset_mit('..')



# val_data = {}
# for i in range(63, 72):
#     val = torch.tensor(dataset[i], dtype=torch.float32)
#     val_data[i] = val.to(device)
#
# test_data = {}
# for i in range(72, 90):
#     test = torch.tensor(dataset[i], dtype=torch.float32)
#     test_data[i] = test.to(device)
#

class RMDataset(Dataset):
    def __init__(self, data, lookback,walk_length=20):
        self.data = data
        self.lookback = lookback

        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
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

        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]
        return x, triplet, scale

Dataset = {}
for i in [1,2,3,4,5]:
    Dataset[i] = RMDataset(data,lookback=i)

def optimise_tr(data,dataset,lookback,dim_attn,n_encoder_layers,n_heads,lr,lin_dim,weight_decay,dropout):
    dataset = dataset[lookback]
    t = Graph2Gauss_Torch(dim_val, dim_attn, dim_in, dim_out,dropout, n_encoder_layers, n_heads, lookback,lin_dim)
    optimizer = torch.optim.Adam(t.parameters(), lr=lr, weight_decay=weight_decay)
    t.to(device)

    epochs = 139
    t.train()
    for e in range(epochs):
        for i in range(lookback, 63):
            x , triplet, scale = dataset[i]
            x = x.clone().detach().requires_grad_(True).to(device)
            optimizer.zero_grad()
            _, mu, sigma = t(x)

            loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)

            loss.backward()
            optimizer.step()

    f_MAP = []
    mu_timestamp = []
    sigma_timestamp = []
    with torch.no_grad():
        t.eval()
        for i in range(lookback, 90):
            x , _, _ = dataset[i]
            x = x.clone().detach().requires_grad_(True).to(device)
            _, mu, sigma = t(x)
            mu_timestamp.append(mu.cpu().detach().numpy())
            sigma_timestamp.append(sigma.cpu().detach().numpy())
    for i in range(3):
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


print(optimise_tr(data,Dataset,lookback=5,dim_attn=333,n_encoder_layers=8,n_heads=4,lr=3.523001333245195e-05,lin_dim=33,weight_decay=0.00048228370868892353,dropout=0.2572))
