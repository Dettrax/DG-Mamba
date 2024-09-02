# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

import os
try :
    os.chdir("SBM")
except:
    pass
from tqdm import tqdm
from mamba_ssm import Mamba
from models import *
from utils import *
import pickle
import json
from eval_mod import get_MAP_avg
# hyperparams
dim_out = 64
dim_in  = 1000

dim_val = 256



# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SBMDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
        self.transform = T.AddRandomWalkPE(walk_length=16, attr_name='pe')


    def temp_process(self, data, lookback):

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

        triplet_dict = {}
        scale_dict = {}
        for i in range(lookback, 50):
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
        return  x, triplet, scale

# Get dataset and construct dict
data = get_data()


# Train/Val/Test split
#
# train_data = {}
# for i in range(lookback, 62):
#     train = torch.tensor(dataset[i], dtype=torch.float32)
#     train_data[i] = train.to(device)
#
# val_data = {}
# for i in range(62, 71):
#     val = torch.tensor(dataset[i], dtype=torch.float32)
#     val_data[i] = val.to(device)
#
# test_data = {}
# for i in range(71, 88):
#     test = torch.tensor(dataset[i], dtype=torch.float32)
#     test_data[i] = test.to(device)


def val_loss(t,val_data):
    l = []
    t.eval()
    for i in range(35,40):
        x ,triplet, scale = val_data[i]
        x = x.clone().detach().requires_grad_(True).to(device)
        _, muval, sigmaval = t(x)
        val_l = build_loss(triplet, scale, muval, sigmaval, dim_out, scale=False)
        l.append(val_l.cpu().detach().numpy())
    return np.mean(l)


class MambaG2G(torch.nn.Module):
    def __init__(self, config, lin_dim, dim_out,dim_val, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = lin_dim
        self.elu = nn.ELU()
        self.mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)

    def forward(self, input):

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


def optimise_mamba(data,lookback,lin_dim,d_conv,d_state,dropout,lr,weight_decay):
    dataset = SBMDataset(data, lookback)


    config = {
        'd_model':dim_in,
        'd_state':d_state,
        'd_conv':d_conv
    }

    model = MambaG2G(config, lin_dim, dim_out,dim_val, dropout=dropout).to(device)
    #print total model parameters
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_losses = []
    train_loss = []
    test_loss = []
    best_MAP = 0
    for e in tqdm(range(50)):
        model.train()
        loss_step = []
        for i in range(lookback, 35):
            x ,triplet, scale = dataset[i]
            x = x.clone().detach().requires_grad_(True).to(device)
            optimizer.zero_grad()
            _, mu, sigma = model(x)
            # loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
            loss = build_loss(triplet, scale, mu, sigma, dim_out, scale=False)
            loss_step.append(loss.cpu().detach().numpy())
            # print(f"Epoch: {e}, Timestamp: {i},Loss: {loss_list[-1]}")
            '''if i==lookback:
                print(triplet)'''
            loss.backward()
            optimizer.step()
        val_losses.append(val_loss(model,dataset))
        train_loss.append(np.mean(loss_step))

        if e%49 == 0:
                # if e %5 ==0:
            mu_timestamp = []
            sigma_timestamp = []
            with torch.no_grad():
                model.eval()
                for i in range(lookback, 50):
                    x,  triplet, scale = dataset[i]
                    x = x.clone().detach().requires_grad_(False).to(device)
                    _, mu, sigma = model(x)
                    mu_timestamp.append(mu.cpu().detach().numpy())
                    sigma_timestamp.append(sigma.cpu().detach().numpy())

            # Save mu and sigma matrices
            save_sigma_mu = True
            sigma_L_arr = []
            mu_L_arr = []
            if save_sigma_mu == True:
                sigma_L_arr.append(sigma_timestamp)
                mu_L_arr.append(mu_timestamp)
            curr_MAP ,curr_MRR = get_MAP_avg(mu_L_arr, sigma_L_arr,lookback,data)
            if curr_MAP > best_MAP:
                best_MAP = curr_MAP
                torch.save(model.state_dict(), 'best_model.pth')
                print("Best MAP: ",e, best_MAP,sep=" ")
            print(f"Epoch {e} Loss: {np.mean(np.stack(loss_step))} Val Loss: {np.mean(np.stack(val_losses))} Best MAP: {best_MAP}")
    return model , val_losses , train_loss , test_loss

lookback = 2
#{'lr': 0.0030654227230925636, 'lin_dim': 47, 'd_conv': 6, 'lookback': 4, 'd_state': 25, 'dropout': 0.3725448646977555, 'weight_decay': 1.02596357976919e-05}
model , val_losses , train_loss , test_loss = optimise_mamba(data,lookback,47,6,25,0.3725448646977555, 0.0030654227230925636, 1.02596357976919e-05)

# config = {
#     'd_model': dim_in,
#     'd_state': 13,
#     'd_conv': 6
# }
# model = MambaG2G(config, 23, dim_out, dim_val, dropout=0.495).to(device)
dataset = SBMDataset(data, lookback)
from eval_mod import get_MAP_avg
model.load_state_dict(torch.load('best_model.pth'))
mu_timestamp = []
sigma_timestamp = []
with torch.no_grad():
    model.eval()
    for i in range(lookback, 50):
        x, triplet, scale = dataset[i]
        x = x.clone().detach().requires_grad_(False).to(device)
        _, mu, sigma = model(x)
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
print("Time taken: ", time.time()-start)



# import time
# from exp_mod import get_MAP_avg
# start = time.time()
# MAPS = []
# MRR = []
# for i in tqdm(range(5)):
#     curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback,data)
#     MAPS.append(curr_MAP)
#     MRR.append(curr_MRR)
# #print mean and std of map and mrr
# print("Mean MAP: ", np.mean(MAPS))
# print("Mean MRR: ", np.mean(MRR))
# print("Std MAP: ", np.std(MAPS))
# print("Std MRR: ", np.std(MRR))
# print("Time taken: ", time.time()-start)
#
# if save_sigma_mu == True:
#     if not os.path.exists(name+'/Eval_Results/saved_array'):
#         os.makedirs(name+'/Eval_Results/saved_array')
#     with open(name+'/Eval_Results/saved_array/sigma_as','wb') as f: pickle.dump(sigma_L_arr, f)
#     with open(name+'/Eval_Results/saved_array/mu_as','wb') as f: pickle.dump(mu_L_arr, f)
#
#
