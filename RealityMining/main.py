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
dim_out = 64
dim_in  = 96

dim_val = 256
dim_attn = 256
lr = 0.0001

n_heads = 1
n_encoder_layers = 1

f = open("config.json")
config = json.load(f)
lookback = 4

#Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#init network and optimizer
t = Graph2Gauss_Torch(dim_val, dim_attn, dim_in, dim_out, n_encoder_layers, n_heads, lookback=lookback)
optimizer = torch.optim.Adam(t.parameters(), lr=lr)
sched = ScheduledOptim(optimizer,lr_mul = 0.01, d_model = 256, n_warmup_steps = 100)
t.to(device)
print('Total parameters:', sum(p.numel() for p in t.parameters()))
#Get dataset and construct dict
data = dataset_mit('..')
dataset = {}
for i in range(lookback, 90):
    B = np.zeros((96,lookback+1,96))
    for j in range(lookback+1):
        adj_matr = data[i-lookback+j][0].todense()
        B[:adj_matr.shape[0],j,:adj_matr.shape[1]] = adj_matr
    dataset[i] = B

#Construct dict of hops and scale terms
hop_dict = {}
scale_terms_dict = {}
print("Constructing dict of hops and scale terms")
for i in range(lookback, 90):
    hops = get_hops(data[i][0],2)
    scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                   hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
    hop_dict[i] = hops
    scale_terms_dict[i] = scale_terms

#Construct dict of triplets
triplet_dict = {}
scale_dict = {}
print("Constructing dict of triplets")
for i in range(lookback,90):
    triplet, scale = to_triplets(sample_all_hops(hop_dict[i]),scale_terms_dict[i])
    triplet_dict[i] = triplet
    scale_dict[i] = scale


#Train/Val/Test split

train_data = {}
for i in range(lookback,63):
    train = torch.tensor(dataset[i], dtype = torch.float32)
    train_data[i] = train.to(device)

val_data = {}
for i in range(63,72):
    val = torch.tensor(dataset[i], dtype = torch.float32)
    val_data[i] = val.to(device)

test_data = {}
for i in range(72,90):
    test = torch.tensor(dataset[i], dtype = torch.float32)
    test_data[i] = test.to(device)

def val_loss(t):
    l = []
    for j in range(63,72):
        _,muval,sigmaval = t(val_data[j])
        val_l = build_loss(triplet_dict[j], scale_dict[j], muval,sigmaval,64, scale = False)
        l.append(val_l.cpu().detach().numpy())
    return np.mean(l)

from tqdm import tqdm
epochs = 200
loss_mainlist = []
val_mainlist = []
for e in tqdm(range(epochs)):
    loss_list = []
    for i in range(lookback,63):
        triplet, scale = to_triplets(sample_all_hops(hop_dict[i]),scale_terms_dict[i])
        sched.zero_grad()
        #optimizer.zero_grad()
        _,mu,sigma,attn_weights = t(train_data[i])
        #loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
        loss = build_loss(triplet, scale, mu, sigma, 64, scale = False)
        loss_list.append(loss.cpu().detach().numpy())
        #print(f"Epoch: {e}, Timestamp: {i},Loss: {loss_list[-1]}")
        loss.backward()
        sched.step_and_update_lr()
        #optimizer.step()
    # val_mainlist.append(val_loss(t))
    # loss_mainlist.append(np.mean(loss_list))
    # print(f"Epoch: {e}, Average loss: {np.mean(loss_list)}, Val loss: {val_mainlist[-1]}" )
    # if e%10==0:
    #     name = 'model' + str(e)+'.pth'
    #     torch.save(t.state_dict(), name)


#t.load_state_dict(torch.load('model20.pth'))

#Model eval
mu_timestamp = []
sigma_timestamp=[]
attn_weights_per_timestamps = []
for i in range(lookback,90):
    with torch.no_grad():
        t.eval()
        _,mu,sigma,attn_weights = t(torch.tensor(dataset[i],dtype = torch.float32).to(device))
        mu_timestamp.append(mu.cpu().detach().numpy())
        sigma_timestamp.append(sigma.cpu().detach().numpy())
        attn_weights_per_timestamp = attn_weights[0][5].cpu().detach().numpy()
        attn_weights_per_timestamps.append(attn_weights_per_timestamp)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# # Load the list from the file
# with open('attn_mamba.pkl', 'rb') as f:
#     attn_mamba = pickle.load(f)
#
# # Assuming attn_matrix is a list of attention matrices
# attn_matrix = attn_weights_per_timestamps  # Replace with your actual attention matrices
#
# # Ensure both lists have the same length
# assert len(attn_mamba) == len(attn_matrix), "Both lists must have the same length"
#
# # Calculate the global min and max values for consistent color range
# vmin = min(min(attn_mat.min() for attn_mat in attn_mamba), min(attn_mat.min() for attn_mat in attn_matrix))
# vmax = max(max(attn_mat.max() for attn_mat in attn_mamba), max(attn_mat.max() for attn_mat in attn_matrix))
#
# # Create a figure with subplots, 2 plots per row (one for each type of attention matrix)
# n_timestamps = len(attn_mamba)
# n_rows = (n_timestamps + 1) // 2  # Calculate the number of rows needed
# fig, axes = plt.subplots(n_rows, 4, figsize=(24, n_rows * 6))
#
# # Flatten the axes array for easy iteration
# axes = axes.flatten()
#
# # Visualize each pair of attention matrices as heatmaps in their respective subplots
# for i in range(n_timestamps):
#     sns.heatmap(attn_mamba[i], annot=True, cmap='viridis', cbar=True, ax=axes[2 * i], vmin=vmin, vmax=vmax)
#     axes[2 * i].set_title(f'Mamba Heatmap {lookback+i + 1}')
#
#     sns.heatmap(attn_matrix[i], annot=True, cmap='viridis', cbar=True, ax=axes[2 * i + 1], vmin=vmin, vmax=vmax)
#     axes[2 * i + 1].set_title(f'Transformer Heatmap {lookback+i + 1}')
#
# # Hide any unused subplots
# for j in range(2 * n_timestamps, len(axes)):
#     fig.delaxes(axes[j])
#
# # Display the figure
# plt.tight_layout()
# plt.show()


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

# select random 24 indices
import random
last_indices = sorted(random.sample(range(len(attn_mamba)), 24))

# Create a 4x6 plot
fig, axes = plt.subplots(4, 6, figsize=(36, 24), dpi=400)  # 4 rows, 6 plots per row

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Visualize each pair of attention matrices as heatmaps in their respective subplots
for idx, i in enumerate(last_indices):
    if 2 * idx + 1 < len(axes):
        sns.heatmap(attn_mamba[i], annot=False, cmap='viridis', cbar=True, ax=axes[2 * idx], vmin=vmin_mamba, vmax=vmax_mamba)
        axes[2 * idx].set_title(f'Mamba Heatmap {lookback + i + 1}', fontsize=24)

        sns.heatmap(attn_matrix[i], annot=False, cmap='viridis', cbar=True, ax=axes[2 * idx + 1], vmin=vmin_transformer, vmax=vmax_transformer)
        axes[2 * idx + 1].set_title(f'Transformer Heatmap {lookback + i + 1}', fontsize=24)

# Hide any unused subplots
for j in range(2 * len(last_indices), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure to a file
plt.savefig('last_24_heatmaps.png')

plt.show()
#
# # Calculate the global min and max values for consistent color range
# vmin = min(attn_mat.min() for attn_mat in attn_matrices)
# vmax = max(attn_mat.max() for attn_mat in attn_matrices)
#
# # Create a figure with subplots, max 5 plots per row
# n_rows = (len(attn_matrices) + 4) // 5  # Calculate the number of rows needed
# fig, axes = plt.subplots(n_rows, 5, figsize=(30, n_rows * 6))
#
# # Flatten the axes array for easy iteration
# axes = axes.flatten()

# # Visualize each attention matrix as a heatmap in its respective subplot
# for i, attn_mat in enumerate(attn_matrices):
#     sns.heatmap(attn_mat, annot=True, cmap='viridis', cbar=True, ax=axes[i], vmin=vmin, vmax=vmax)
#     axes[i].set_title(f'Heatmap {lookback+i+1}')
#
# # Hide any unused subplots
# for j in range(len(attn_matrices), len(axes)):
#     fig.delaxes(axes[j])
#
# # Display the figure
# plt.tight_layout()
# plt.show()

# #Save mu and sigma matrices
# name = 'Results/RealityMining'
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
#
# from eval_mod import get_MAP_avg
# import time
# start = time.time()
# MAPS = []
# MRR = []
# for i in tqdm(range(2)):
#     curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr, lookback,data)
#     MAPS.append(curr_MAP)
#     MRR.append(curr_MRR)
# #print mean and std of map and mrr
# print("Mean MAP: ", np.mean(MAPS))
# print("Mean MRR: ", np.mean(MRR))
# print("Std MAP: ", np.std(MAPS))
# print("Std MRR: ", np.std(MRR))
# print("Time taken: ", time.time() - start)
