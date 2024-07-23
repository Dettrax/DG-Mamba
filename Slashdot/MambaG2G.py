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
    os.chdir("Slashdot")
except:
    pass
from tqdm import tqdm
from mamba_ssm import Mamba
from models import *
from utils import *
import pickle
import json

# hyperparams
dim_out = 64
dim_in = 50825

dim_val = 256

f = open("config.json")
config = json.load(f)
lookback = config["lookback"]

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SlashDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)


    def temp_process(self, data, lookback):

        # Construct dict of hops and scale terms
        hop_dict = {}
        scale_terms_dict = {}

        for i in range(lookback, 13):
            hops = get_hops(data[i][0], 2)
            scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                               hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
            hop_dict[i] = hops
            scale_terms_dict[i] = scale_terms

        # Construct dict of triplets
        triplet_dict = {}
        scale_dict = {}

        for i in range(lookback, 13):
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            triplet_dict[i] = triplet
            scale_dict[i] = scale

        return triplet_dict, scale_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):



        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]
        return  triplet, scale

# Get dataset and construct dict
data = dataset_slashdot('../')


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
    for i in range(9, 11):
        triplet, scale = val_data[i]
        dataset = []
        for j in range(lookback+1):
            dataset.append(data[i-lookback+j][1].to(device))
        _, muval, sigmaval = t(dataset)
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
        self.enc_input_fc = SparseLinear(dim_in, dim_val)


    def forward(self, input):
        out_tuple = ()
        for inp_mat in input:
            out_mat = self.enc_input_fc(inp_mat)
            out_tuple += (out_mat,)
        z = torch.stack(out_tuple,1)

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


def optimise_mamba(data,lookback,lin_dim,d_conv,d_state,dropout,lr,weight_decay):
    dataset = SlashDataset(data, lookback)


    config = {
        'd_model':dim_val,
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
        for i in range(lookback, 9):
            triplet, scale = dataset[i]
            temp_data = []
            for j in range(lookback + 1):
                temp_data.append(data[i - lookback + j][1].to(device))

            optimizer.zero_grad()
            _, mu, sigma = model(temp_data)
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

        print(f"Epoch {e} Loss: {np.mean(np.stack(loss_step))} Val Loss: {np.mean(np.stack(val_losses))}")

        # if e %5 ==0:
        #     mu_timestamp = []
        #     sigma_timestamp = []
        #     with torch.no_grad():
        #         model.eval()
        #         for i in range(lookback, 88):
        #             x,  triplet, scale = dataset[i]
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
        #     curr_MAP ,_ = get_MAP_avg(mu_L_arr, sigma_L_arr,lookback,data)
        #     if curr_MAP > best_MAP:
        #         best_MAP = curr_MAP
        #         torch.save(model.state_dict(), 'best_model.pth')
        #         print("Best MAP: ",e, best_MAP,sep=" ")

    return model , val_losses , train_loss , test_loss
lookback = 4
model , val_losses , train_loss , test_loss = optimise_mamba(data,lookback,lin_dim=28,d_conv=5,d_state=44,dropout=0.44539,lr=2.7608713668855892e-05,weight_decay=0.0001580)