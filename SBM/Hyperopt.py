# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import ray
from ray import tune
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    f_MAP = []
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



    mu_timestamp = []
    sigma_timestamp = []
    with torch.no_grad():
        model.eval()
        for i in range(lookback, 50):
            x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
            x = x.clone().detach().requires_grad_(False).to(device)
            _, mu, sigma = model(x)
            mu_timestamp.append(mu.cpu().detach().numpy())
            sigma_timestamp.append(sigma.cpu().detach().numpy())

    # Save mu and sigma matrices
    name = 'Results/RealityMining'
    save_sigma_mu = True
    sigma_L_arr = []
    mu_L_arr = []
    if save_sigma_mu == True:
        sigma_L_arr.append(sigma_timestamp)
        mu_L_arr.append(mu_timestamp)
    for i in range(3):
        MAP, _ = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback, data)
        f_MAP.append(MAP)
    return sorted(f_MAP)[0]

# lookback = 2
# MAP = optimise_mamba(data,lookback,26,2,4,0.288,0.002141,9.9326e-05)
#

def train_model(config):
    map_value = optimise_mamba(data,lookback=config['lookback'], dim_in=config['dim_in'], d_conv=config['d_conv'],d_state=config['d_state'],dropout=config['dropout'],lr=config['lr'],weight_decay=config['weight_decay'])
    return map_value


def objective(config):  # ①
    while True:
        acc = train_model(config)
        train.report({"MAP": acc})  # Report to Tunevvvvvvv


ray.init(  runtime_env={
            os.getcwd(),
        })  # Initialize Ray


search_space = {"lr": tune.loguniform(1e-4, 1e-2)
    ,"dim_in": tune.randint(16, 100),
                "d_conv": tune.randint(2, 10),
                "lookback": tune.randint(1, 5),
                "d_state": tune.randint(2, 10),
                "dropout": tune.uniform(0.1, 0.5),
                "weight_decay": tune.loguniform(1e-5, 1e-3)}

# Create an Optuna search space
algo = OptunaSearch(
)


tuner = tune.Tuner(  # ③
    tune.with_resources(
        tune.with_parameters(objective),
        resources={"gpu": 0.25}
    ),
    tune_config=tune.TuneConfig(
        metric="MAP",
        mode="max",
        search_alg=algo,
        num_samples=5
    ),
    param_space=search_space,
    run_config=train.RunConfig(
        stop={"training_iteration": 1}  # Limit the training iterations to 1
    )
)

results = tuner.fit()
print("Best config is:", results.get_best_result().config)
