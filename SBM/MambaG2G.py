# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".
import os
try :
    os.chdir("SBM")
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
from tqdm import tqdm

# hyperparams
enc_seq_len = 6
dim_out = 64
dim_in = 1000

dim_val = 256
dim_attn = 256
lr = 0.0001
epochs = 10

n_heads = 1
n_encoder_layers = 1

f = open("config.json")
config = json.load(f)
lookback = config["lookback"]

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class MambaG2G(torch.nn.Module):
    def __init__(self, config: MambaConfig, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.mamba = Mamba(config)

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config.d_model, self.D)  # Adjusted to match output dimension
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

# init network and optimizer
config = MambaConfig(
    d_model=dim_in,
    n_layers=1,
    d_state=8,
    d_conv=4,
)

t  = MambaG2G(config, dim_in, 64, dropout=0.1).to(device)
optimizer = torch.optim.Adam(t.parameters(), lr=lr)
t.to(device)

# Get dataset and construct dict
data = get_data()
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

# Construct dict of triplets
triplet_dict = {}
scale_dict = {}
for i in range(lookback, 50):
    triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
    triplet_dict[i] = triplet
    scale_dict[i] = scale

# Train/Val/Test split

train_data = {}
for i in range(lookback, 35):
    train = torch.tensor(dataset[i], dtype=torch.float32)
    train_data[i] = train.to(device)

val_data = {}
for i in range(35, 40):
    val = torch.tensor(dataset[i], dtype=torch.float32)
    val_data[i] = val.to(device)

test_data = {}
for i in range(40, 50):
    test = torch.tensor(dataset[i], dtype=torch.float32)
    test_data[i] = test.to(device)


def val_loss(t):
    l = []
    for j in range(35, 40):
        _, muval, sigmaval = t(val_data[j])
        val_l = build_loss(triplet_dict[j], scale_dict[j], muval, sigmaval, 64, scale=False)
        l.append(val_l.cpu().detach().numpy())
    return np.mean(l)


epochs = 20
loss_mainlist = []
val_mainlist = []
for e in range(epochs):
    loss_list = []
    for i in range(lookback, 35):
        triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
        optimizer.zero_grad()
        _, mu, sigma = t(train_data[i])
        # loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
        loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)
        loss_list.append(loss.cpu().detach().numpy())
        # print(f"Epoch: {e}, Timestamp: {i},Loss: {loss_list[-1]}")
        loss.backward()
        optimizer.step()
    val_mainlist.append(val_loss(t))
    loss_mainlist.append(np.mean(loss_list))
    print(f"Epoch: {e}, Average loss: {np.mean(loss_list)}, Val loss: {val_mainlist[-1]}")



plt.figure()
plt.semilogy(loss_mainlist)
plt.semilogy(val_mainlist)
plt.legend(['Train loss', 'Val loss'])
plt.savefig("Loss.png")

# t.load_state_dict(torch.load('model20.pth'))

# Model eval
mu_timestamp = []
sigma_timestamp = []
for i in range(lookback, 50):
    _, mu, sigma = t(torch.tensor(dataset[i], dtype=torch.float32).to(device))
    mu_timestamp.append(mu.cpu().detach().numpy())
    sigma_timestamp.append(sigma.cpu().detach().numpy())

# Save mu and sigma matrices
name = 'Results/SBM'
save_sigma_mu = True
sigma_L_arr = []
mu_L_arr = []
if save_sigma_mu == True:
    sigma_L_arr.append(sigma_timestamp)
    mu_L_arr.append(mu_timestamp)

from eval_mod import get_MAP_avg

print(get_MAP_avg(mu_L_arr, sigma_L_arr, 2))




