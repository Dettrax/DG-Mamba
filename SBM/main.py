#This code creates and saves embedding with - transformer + G2G model.
#We have used some of the functionalities from Xu, M., Singh, A.V. &
#Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
#Method for Temporal Graphs".

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from models import *
from utils import *
import pickle
import json

#hyperparams
enc_seq_len = 6
dim_out = 64
dim_in  = 1000

dim_val = 256
dim_attn = 256
lr = 0.0001
epochs = 300

n_heads = 1
n_encoder_layers = 1

f = open("config.json")
config = json.load(f)
lookback = config["lookback"]

#Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#init network and optimizer
t = Graph2Gauss_Torch(dim_val, dim_attn, dim_in, dim_out, n_encoder_layers, n_heads, lookback)
optimizer = torch.optim.Adam(t.parameters(), lr=lr)
t.to(device)

#Get dataset and construct dict
data = get_data()
dataset = {}
for i in range(lookback, 50):
    B = np.zeros((1000,lookback+1,1000))
    for j in range(lookback+1):
        B[:,j,:] = data[i-lookback+j][0].todense()
    dataset[i] = B

#Construct dict of hops and scale terms
hop_dict = {}
scale_terms_dict = {}
for i in range(lookback, 50):
    hops = get_hops(data[i][0],2)
    scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                   hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
    hop_dict[i] = hops
    scale_terms_dict[i] = scale_terms

#Construct dict of triplets
triplet_dict = {}
scale_dict = {}
for i in range(lookback,50):
    triplet, scale = to_triplets(sample_all_hops(hop_dict[i]),scale_terms_dict[i])
    triplet_dict[i] = triplet
    scale_dict[i] = scale


#Train/Val/Test split

train_data = {}
for i in range(lookback,35):
    train = torch.tensor(dataset[i], dtype = torch.float32)
    train_data[i] = train.to(device)

val_data = {}
for i in range(35,40):
    val = torch.tensor(dataset[i], dtype = torch.float32)
    val_data[i] = val.to(device)

test_data = {}
for i in range(40,50):
    test = torch.tensor(dataset[i], dtype = torch.float32)
    test_data[i] = test.to(device)

def val_loss(t):
    l = []
    for j in range(35,40):
        _,muval,sigmaval = t(val_data[j])
        val_l = build_loss(triplet_dict[j], scale_dict[j], muval,sigmaval,64, scale = False)
        l.append(val_l.cpu().detach().numpy())
    return np.mean(l)

from tqdm import tqdm

epochs = 100
loss_mainlist = []
val_mainlist = []
for e in tqdm(range(epochs)):
    loss_list = []
    for i in range(lookback,35):
        triplet, scale = to_triplets(sample_all_hops(hop_dict[i]),scale_terms_dict[i])
        optimizer.zero_grad()
        _,mu,sigma,attn_weights = t(train_data[i])
        #loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
        loss = build_loss(triplet, scale, mu, sigma, 64, scale = False)
        loss_list.append(loss.cpu().detach().numpy())
        #print(f"Epoch: {e}, Timestamp: {i},Loss: {loss_list[-1]}")
        loss.backward()
        optimizer.step()
    # val_mainlist.append(val_loss(t))
    loss_mainlist.append(np.mean(loss_list))


#t.load_state_dict(torch.load('model20.pth'))

#Model eval
mu_timestamp = []
sigma_timestamp=[]
attn_weights_per_timestamps = []
for i in range(lookback,50):
    _,mu,sigma,attn_weights = t(torch.tensor(dataset[i],dtype = torch.float32).to(device))
    mu_timestamp.append(mu.cpu().detach().numpy())
    sigma_timestamp.append(sigma.cpu().detach().numpy())
    attn_weights_per_timestamp = attn_weights[0][5].cpu().detach().numpy()
    attn_weights_per_timestamps.append(attn_weights_per_timestamp)

#
# #Save mu and sigma matrices
# name = 'Results/SBM'
# save_sigma_mu = True
# sigma_L_arr = []
# mu_L_arr = []
# if save_sigma_mu == True:
#         sigma_L_arr.append(sigma_timestamp)
#         mu_L_arr.append(mu_timestamp)
#
# if save_sigma_mu == True:
#     if not os.path.exists(name+'/Eval_Results/saved_array'):
#         os.makedirs(name+'/Eval_Results/saved_array')
#     with open(name+'/Eval_Results/saved_array/sigma_as','wb') as f: pickle.dump(sigma_L_arr, f)
#     with open(name+'/Eval_Results/saved_array/mu_as','wb') as f: pickle.dump(mu_L_arr, f)
#
# from eval_mod import get_MAP_avg
# import time
# start = time.time()
# MAPS = []
# MRR = []
# for i in tqdm(range(2)):
#     curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr,sigma_L_arr, lookback,data)
#     MAPS.append(curr_MAP)
#     MRR.append(curr_MRR)
# #print mean and std of map and mrr
# print("Mean MAP: ", np.mean(MAPS))
# print("Mean MRR: ", np.mean(MRR))
# print("Std MAP: ", np.std(MAPS))
# print("Std MRR: ", np.std(MRR))
# print("Time taken: ", time.time() - start)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Load the list from the file
with open('attn_mamba.pkl', 'rb') as f:
    attn_mamba = pickle.load(f)

# Assuming attn_matrix is a list of attention matrices
attn_matrix = attn_weights_per_timestamps  # Replace with your actual attention matrices

# Ensure both lists have the same length
assert len(attn_mamba) == len(attn_matrix), "Both lists must have the same length"

# Calculate the global min and max values for Mamba and Transformer separately
vmin_mamba = min(attn_mat.min() for attn_mat in attn_mamba)
vmax_mamba = max(attn_mat.max() for attn_mat in attn_mamba)
vmin_transformer = min(attn_mat.min() for attn_mat in attn_matrix)
vmax_transformer = max(attn_mat.max() for attn_mat in attn_matrix)

# Filter odd timestamps
odd_indices = [i for i in range(len(attn_mamba)) if True]
n_odd_timestamps = len(odd_indices)
n_rows = (n_odd_timestamps + 2) // 3  # Calculate the number of rows needed
fig, axes = plt.subplots(n_rows, 6, figsize=(36, n_rows * 6), dpi=400)  # 6 plots per row (3 pairs)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Visualize each pair of attention matrices as heatmaps in their respective subplots
for idx, i in enumerate(odd_indices):
    if 2 * idx < len(axes):
        sns.heatmap(attn_mamba[i], annot=False, cmap='viridis', cbar=True, ax=axes[2 * idx], vmin=vmin_mamba, vmax=vmax_mamba)
        axes[2 * idx].set_title(f'Mamba Heatmap {lookback + i + 1}', fontsize=14)

        sns.heatmap(attn_matrix[i], annot=False, cmap='viridis', cbar=True, ax=axes[2 * idx + 1], vmin=vmin_transformer, vmax=vmax_transformer)
        axes[2 * idx + 1].set_title(f'Transformer Heatmap {lookback + i + 1}', fontsize=14)


# Hide any unused subplots
for j in range(2 * len(odd_indices), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()