# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import os
try :
    os.chdir("UCI")
except:
    pass
from tqdm import tqdm
from mamba_ssm import Mamba
from models import *
from utils import *
import pickle
import json
from eval_mode import get_MAP_avg
# hyperparams
dim_out = 256
dim_in = 1899


f = open("config.json")
config = json.load(f)
lookback = config["lookback"]

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class UCIDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)


    def temp_process(self, data, lookback):
        dataset = {}
        for i in range(lookback, 88):
            B = np.zeros((1899, lookback + 1, 1899))
            for j in range(lookback + 1):
                adj_matr = data[i - lookback + j][0].todense()
                B[:adj_matr.shape[0], j, :adj_matr.shape[1]] = adj_matr

            dataset[i] = B

        # Construct dict of hops and scale terms
        hop_dict = {}
        scale_terms_dict = {}

        for i in range(lookback, 88):
            hops = get_hops(data[i][0], 2)
            scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                               hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
            hop_dict[i] = hops
            scale_terms_dict[i] = scale_terms

        # Construct dict of triplets
        triplet_dict = {}
        scale_dict = {}

        for i in range(lookback, 88):
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

# Get dataset and construct dict
data = dataset_UCI('../')


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
    for j in range(62, 71):
        x, triplet, scale = val_data[j]
        x = x.clone().detach().requires_grad_(False).to(device)
        _, muval, sigmaval = t(x)
        val_l = build_loss(triplet, scale, muval, sigmaval, 256, scale=False)
        l.append(val_l.cpu().detach().numpy())
    return np.mean(l)


class MambaG2G(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
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


def optimise_mamba(data,lookback,dim_in,d_conv,d_state,dropout,lr,weight_decay):
    dataset = UCIDataset(data, lookback)


    config = {
        'd_model':1899,
        'd_state':d_state,
        'd_conv':d_conv
    }

    model = MambaG2G(config, dim_in, 256, dropout=dropout).to(device)
    #print total model parameters
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)

    for e in tqdm(range(50)):
        model.train()
        loss_step = []
        for i in range(lookback, 62):
            x, triplet, scale = dataset[i]
            x = x.clone().detach().requires_grad_(True).to(device)
            optimizer.zero_grad()
            _, mu, sigma = model(x)
            # loss = build_loss(triplet_dict[i], scale_dict[i],mu,sigma,64, scale = False)
            loss = build_loss(triplet, scale, mu, sigma, 256, scale=False)
            loss_step.append(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()
    f_MAP = []
    for i in range(1):
        mu_timestamp = []
        sigma_timestamp = []
        with torch.no_grad():
            model.eval()
            for i in range(lookback, 88):
                x, triplet, scale = dataset[i]
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

        MAP,MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback,data)
        print('MAP:', MAP)
        print('MRR:', MRR)
        f_MAP.append(MAP)

    return np.mean(f_MAP)

print(optimise_mamba(data,lookback,26,2,4,0.288,0.001141,9.9326e-05))

# def train_model(config):
#     map_value = optimise_mamba(data,lookback=lookback, dim_in=config['dim_in'], d_conv=config['d_conv'],d_state=config['d_state'],dropout=config['dropout'],lr=config['lr'],weight_decay=config['weight_decay'])
#     return map_value
#
#
# def objective(config):  # ①
#     while True:
#         acc = train_model(config)
#         train.report({"MAP": acc})  # Report to Tunevvvvvvv
#
#
# ray.init(logging_level=logging.DEBUG,      runtime_env={
#             "working_dir": ,
#         })  # Initialize Ray
#
#
# search_space = {"lr": tune.loguniform(1e-4, 1e-2)
#     ,"dim_in": tune.randint(16, 100),
#                 "d_conv": tune.randint(2, 10),
#                 "d_state": tune.randint(2, 10),
#                 "dropout": tune.uniform(0.1, 0.5),
#                 "weight_decay": tune.loguniform(1e-5, 1e-3)}
#
# # Create an Optuna search space
# algo = OptunaSearch(
# )
#
#
# tuner = tune.Tuner(  # ③
#     tune.with_resources(
#         tune.with_parameters(objective),
#         resources={"gpu": 1}
#     ),
#     tune_config=tune.TuneConfig(
#         metric="MAP",
#         mode="max",
#         search_alg=algo,
#         num_samples=10
#     ),
#     param_space=search_space,
# )
#
# results = tuner.fit()
# print("Best config is:", results.get_best_result().config)
