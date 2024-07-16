# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".
import os
try :
    os.chdir("AS")
except:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from models import *
from utils import *
import pickle
import json
from mamba import Mamba, MambaConfig
# hyperparams
dim_out = 64
dim_in = 65535



n_heads = 1
n_encoder_layers = 1

f = open("config.json")
config = json.load(f)
lookback = config["lookback"]

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# init network and optimizer
# t = Graph2Gauss_Torch(dim_val, dim_attn, dim_in, dim_out, n_encoder_layers, n_heads, lookback)



# Get dataset and construct dict
data = dataset_as('../datasets/as_data')
"""dataset = {}
for i in range(lookback, 13):    #lookback + 1 because ignore timestamp 2
    B = np.zeros((50825,lookback+1,50825))
    for j in range(lookback+1):
        adj_matr = data[i-lookback+j][0].todense()
        B[:adj_matr.shape[0],j,:adj_matr.shape[1]] = adj_matr
    print(f"Timestamp: {i}, Adj matrix shape: {adj_matr.shape}")
    dataset[i] = B"""

# Construct dict of hops and scale terms
hop_dict = {}
scale_terms_dict = {}
print("Constructing dict of hops and scale terms")
for i in range(lookback, 100):
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
for i in range(lookback, 100):
    triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
    triplet_dict[i] = triplet
    scale_dict[i] = scale

# for i in range(lookback,13):
#    dataset[i] = torch.tensor(dataset[i], dtype = torch.float32)

# Train/Val/Test split
'''
train_data = {}
for i in range(lookback,95):
    train = torch.tensor(dataset[i], dtype = torch.float32)
    #train_data[i] = train.to(device)
    train_data[i] = train

val_data = {}
for i in range(95,109):
    val = torch.tensor(dataset[i], dtype = torch.float32)
    #val_data[i] = val.to(device)
    val_data[i] = val

test_data = {}
for i in range(109,137):
    test = torch.tensor(dataset[i], dtype = torch.float32)
    #test_data[i] = test.to(device)
    test_data[i] = test
'''


class MambaG2G(torch.nn.Module):
    def __init__(self, config: MambaConfig, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.mamba = Mamba(config)
        self.enc_input_fc = SparseLinear(dim_in, dim_val)
        self.red_input_fc = SparseLinear(dim_in, dim_val)

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config.d_model, self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(dim_val, dim_out)
        self.mu_fc = nn.Linear(dim_val, dim_out)

    def forward(self, input):
        # e = self.enc_input_fc(input)
        out_tuple = ()
        for inp_mat in input:
            out_mat = self.enc_input_fc(inp_mat)
            out_mat = self.red_input_fc(out_mat.T).T
            out_tuple += (out_mat,)
        z = torch.stack(out_tuple, 1)

        e = self.mamba(z)
        e = e.mean(dim=1)  # Average pooling to maintain the expected shape
        e = self.dropout(e)  # Apply dropout after average pooling
        x = torch.tanh(self.out_fc(e))
        x = self.elu(x)
        x = self.dropout(x).T  # Apply dropout after the activation

        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        sigma = self.elu(sigma) + 1 + 1e-14

        return x, mu, sigma


def val_loss(t):
    l = []
    for i in range(70, 80):
        # B = np.zeros((50825,lookback+1,50825))
        dataset = []
        for j in range(lookback + 1):
            dataset.append(data[i - lookback + j][1].to(device))
            # B[:adj_matr.shape[0],j,:adj_matr.shape[1]] = adj_matr
        # print(f"Timestamp: {i}")#, Adj matrix shape: {adj_matr.shape}")
        # dataset = B
        # dataset = torch.tensor(dataset,dtype = torch.float32)
        _, muval, sigmaval = t(dataset)
        val_l = build_loss(triplet_dict[i], scale_dict[i], muval, sigmaval, 64, scale=False)
        l.append(val_l.cpu().detach().numpy())
    return np.mean(l)


dim_val = 256
dim_attn = 256
lr = 0.001
config = MambaConfig(
    d_model=dim_val,
    n_layers=2,
    d_state=32,
    d_conv=4,
)


t = MambaG2G(config, dim_in, dim_out)

optimizer = torch.optim.Adam(t.parameters(), lr=lr)
# sched = ScheduledOptim(optimizer,lr_mul = 0.1, d_model = 256, n_warmup_steps = 130)
t.to(device)

epochs = 10
loss_mainlist = []
val_mainlist = []
for e in range(epochs):
    loss_list = []
    for i in range(lookback, 5):
        triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
        # sched.zero_grad()
        optimizer.zero_grad()
        # B = np.zeros((50825,lookback+1,50825))
        dataset = []
        for j in range(lookback + 1):
            dataset.append(data[i - lookback + j][1].to(device))
            # B[:adj_matr.shape[0],j,:adj_matr.shape[1]] = adj_matr
        # print(f"Timestamp: {i}")#, Adj matrix shape: {adj_matr.shape}")
        # dataset = B

        # dataset = torch.tensor(dataset, dtype = torch.float32)
        _, mu, sigma = t(dataset)
        # loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
        loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)
        loss_list.append(loss.cpu().detach().numpy())
        # print(f"Epoch: {e}, Timestamp: {i},Loss: {loss_list[-1]}")
        loss.backward()
        # sched.step_and_update_lr()
        optimizer.step()
    # val_mainlist.append(val_loss(t))
    loss_mainlist.append(np.mean(loss_list))
    print(f"Epoch: {e}, Average loss: {np.mean(loss_list)}, Val loss: 0")


# plt.figure()
# plt.semilogy(loss_mainlist)
# plt.semilogy(val_mainlist)
# plt.legend(['Train loss', 'Val loss'])
# Model eval
mu_timestamp = []
sigma_timestamp = []
for i in range(lookback, 100):
    dataset = []
    for j in range(lookback + 1):
        dataset.append(data[i - lookback + j][1].to(device))

    _, mu, sigma = t(dataset)
    mu_timestamp.append(mu.cpu().detach().numpy())
    sigma_timestamp.append(sigma.cpu().detach().numpy())

# Save mu and sigma matrices
name = 'Results/AS'
save_sigma_mu = True
sigma_L_arr = []
mu_L_arr = []
if save_sigma_mu == True:
    sigma_L_arr.append(sigma_timestamp)
    mu_L_arr.append(mu_timestamp)

from eval_mod import get_MAP_avg

print(get_MAP_avg(mu_L_arr, sigma_L_arr, 2))

