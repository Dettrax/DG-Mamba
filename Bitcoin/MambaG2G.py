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
    os.chdir("Bitcoin")
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
dim_out = 256
dim_in  = 5881

dim_val = 512



# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class BITDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
        self.transform = T.AddRandomWalkPE(walk_length=16, attr_name='pe')


    def temp_process(self, data, lookback):

        dataset = {}
        for i in range(lookback, 10):  # lookback + 1 because ignore timestamp 2
            B = np.zeros((5881, lookback + 1, 5881))
            for j in range(lookback + 1):
                adj_matr = data[i - lookback + j][0].todense()
                B[:adj_matr.shape[0], j, :adj_matr.shape[1]] = adj_matr
            dataset[i] = B

        hop_dict = {}
        scale_terms_dict = {}
        for i in range(lookback, 10):
            hops = get_hops(data[i][0], 2)
            scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                               hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
            hop_dict[i] = hops
            scale_terms_dict[i] = scale_terms

        # Construct dict of triplets
        triplet_dict = {}
        scale_dict = {}

        for i in range(lookback, 10):
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
data = dataset_bit_alpha('..')


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
    for i in range(95,109):
        x ,triplet, scale = val_data[i]
        x = x.clone().detach().requires_grad_(True).to(device)
        _, muval, sigmaval = t(x)
        val_l = build_loss(triplet, scale, muval, sigmaval, dim_out, scale=False)
        l.append(val_l.cpu().detach().numpy())
    return np.mean(l)

datasets = {}
for i in [2]:
    datasets[i] = BITDataset(data, i)


class SparseLinear(torch.nn.Module):
    def __init__(self, size_i_1, size_i, bias=True):
        super().__init__()
        self.size_i_1 = size_i_1
        self.size_i = size_i
        self.bias = bias
        self.out_channels = size_i
        self.in_channels = size_i_1

        # self.weight = torch.nn.Parameter(torch.randn(size_i_1, size_i))
        self.weight = torch.nn.Parameter(torch.empty(size_i, size_i_1))
        # self.bias = torch.nn.Parameter(torch.randn(size_i))
        self.bias = torch.nn.Parameter(torch.empty(size_i))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # y = torch.sparse.mm(input, self.weight) + self.bias
        y = torch.sparse.mm(input, self.weight.T) + self.bias
        return y


class MambaG2G(torch.nn.Module):
    def __init__(self, config, lin_dim, dim_out, dim_val, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = lin_dim
        self.elu = nn.ELU()
        self.mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])
        self.enc_input_fc = SparseLinear(dim_in, 64)
        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)

    def forward(self, input):
        out_tuple = ()
        for i in range(input.size(1)):
            out_mat = self.enc_input_fc(input[:, i, :])
            out_tuple += (out_mat,)
        z = torch.stack(out_tuple, 1)
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


def optimise_mamba(data,dataset, lookback, lin_dim, d_conv, d_state, dropout, lr, weight_decay):

    config = {
        'd_model': 64,
        'd_state': d_state,
        'd_conv': d_conv
    }

    model = MambaG2G(config, lin_dim, dim_out, dim_val, dropout=dropout).to(device)
    # print total model parameters
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_losses = []
    train_loss = []
    test_loss = []
    best_MAP = -1
    best_model = None
    for e in tqdm(range(51)):
        model.train()
        loss_step = []
        for i in range(lookback, 10):
            x, triplet, scale = dataset[i]
            x = x.clone().detach().requires_grad_(True).to(device)
            optimizer.zero_grad()
            _, mu, sigma = model(x)
            # loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
            loss = build_loss(triplet, scale, mu, sigma, 256, scale=False)
            loss_step.append(loss.cpu().detach().numpy())
            # print(f"Epoch: {e}, Timestamp: {i},Loss: {loss_list[-1]}")
            '''if i==lookback:
                print(triplet)'''
            loss.backward()
            optimizer.step()

        mu_timestamp = []
        sigma_timestamp = []
        with torch.no_grad():
            model.eval()
            for i in range(lookback, 10):
                x, triplet, scale = dataset[i]
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

        curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback, data, device)

    return model, val_losses, train_loss, test_loss


lookback = 2
# {'lr': 0.0030654227230925636, 'lin_dim': 47, 'd_conv': 6, 'lookback': 4, 'd_state': 25, 'dropout': 0.3725448646977555, 'weight_decay': 1.02596357976919e-05}
model, val_losses, train_loss, test_loss = optimise_mamba(data,datasets[lookback], lookback, 15, 2, 25, 0.3725448646977555,
                                                          0.0030654227230925636, 1.02596357976919e-03)
