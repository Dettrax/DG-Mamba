#This code creates and saves embedding with - transformer + G2G model.
#We have used some of the functionalities from Xu, M., Singh, A.V. &
#Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
#Method for Temporal Graphs".

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
import numpy as np
from matplotlib import pyplot as plt
from models_st_transformerg2g import *
from utils_st_transformerg2g import *
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

f = open("config_st_transformerg2g.json")
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
data = dataset_mit('../../..')
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
train_edge = {}
for i in range(lookback,63):
    train = torch.tensor(dataset[i], dtype = torch.float32)

    #Precompute edge_list
    edge_list = []
    for j in range(lookback+1):
        adj_matrix = train[:,j,:]
        edge_index,_ = dense_to_sparse(adj_matrix)
        edge_index = edge_index.to(device)
        edge_list.append(edge_index)
    train_edge[i] = edge_list

    train_data[i] = train.to(device)

val_data = {}
val_edge = {}
for i in range(63,72):
    val = torch.tensor(dataset[i], dtype = torch.float32)

    #Precompute edge_list
    edge_list = []
    for j in range(lookback+1):
        adj_matrix = val[:,j,:]
        edge_index,_ = dense_to_sparse(adj_matrix)
        edge_index = edge_index.to(device)
        edge_list.append(edge_index)
    val_edge[i] = edge_list

    val_data[i] = val.to(device)

test_data = {}
test_edge = {}
for i in range(72,90):
    test = torch.tensor(dataset[i], dtype = torch.float32)

    #Precompute edge_list
    edge_list = []
    for j in range(lookback+1):
        adj_matrix = val[:,j,:]
        edge_index,_ = dense_to_sparse(adj_matrix)
        edge_index = edge_index.to(device)
        edge_list.append(edge_index)
    test_edge[i] = edge_list

    test_data[i] = test.to(device)



def val_loss(t):
    l = []
    t.eval()
    for j in range(63,72):
        _,muval,sigmaval = t(val_data[j], val_edge[j])
        val_l = build_loss(triplet_dict[j], scale_dict[j], muval,sigmaval,64, scale = False)
        l.append(val_l.cpu().detach().numpy())
    return np.mean(l)


epochs = 100
loss_mainlist = []
val_mainlist = []
loss_savelist = []
for e in range(epochs):
    loss_list = []
    for i in range(lookback,63):
        t.train()
        triplet, scale = to_triplets(sample_all_hops(hop_dict[i]),scale_terms_dict[i])
        optimizer.zero_grad()
        _,mu,sigma = t(train_data[i], train_edge[i])
        #loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
        loss = build_loss(triplet, scale, mu, sigma, 64, scale = False)
        loss_list.append(loss.cpu().detach().numpy())
        #print(f"Epoch: {e}, Timestamp: {i},Loss: {loss_list[-1]}")
        loss.backward()
        optimizer.step()
    val_mainlist.append(val_loss(t))
    loss_mainlist.append(np.mean(loss_list))
    print(f"Epoch: {e}, Average loss: {np.mean(loss_list)}, Val loss: {val_mainlist[-1]}" )
    if e%10==0:
        name = 'model' + str(e)+'.pth'
        torch.save(t.state_dict(), name)
        loss_savelist.append(val_mainlist[-1])    


plt.figure()
plt.semilogy(loss_mainlist)
plt.semilogy(val_mainlist)
plt.legend(['Train loss', 'Val loss'])
plt.savefig("Loss.png")


#t.load_state_dict(torch.load('model20.pth'))
model_num = np.argmin(loss_savelist)*10
model_name= 'model'+str(int(model_num))+'.pth'
print("Best_model is ", model_name)
print("Loss of best model is ", np.min(loss_savelist))
t.load_state_dict(torch.load(model_name))

#Model eval
mu_timestamp = []
sigma_timestamp=[]
t.eval()
for i in range(lookback,90):
    inp = torch.tensor(dataset[i],dtype = torch.float32)
    edge_list = []
    for j in range(lookback+1):
        adj_matrix = inp[:,j,:]
        edge_index,_ = dense_to_sparse(adj_matrix)
        edge_index = edge_index.to(device)
        edge_list.append(edge_index)
    inp = inp.to(device)
    _,mu,sigma = t(inp,edge_list)
    mu_timestamp.append(mu.cpu().detach().numpy())
    sigma_timestamp.append(sigma.cpu().detach().numpy())
    


#Save mu and sigma matrices    
name = 'Results/RealityMining'
save_sigma_mu = True
sigma_L_arr = []
mu_L_arr = []
if save_sigma_mu == True:
        sigma_L_arr.append(sigma_timestamp)
        mu_L_arr.append(mu_timestamp)
        
if save_sigma_mu == True:
    if not os.path.exists(name+'/Eval_Results/saved_array'):
        os.makedirs(name+'/Eval_Results/saved_array')
    with open(name+'/Eval_Results/saved_array/sigma_as','wb') as f: pickle.dump(sigma_L_arr, f)
    with open(name+'/Eval_Results/saved_array/mu_as','wb') as f: pickle.dump(mu_L_arr, f)
