from utils import *
from models import *
from scipy import sparse
import random
import pickle
import time
import json


import numpy as np
import random

f = open("config.json")
config = json.load(f)

K = config["K"]
p_val = config["p_val"]
p_nodes = config["p_nodes"]
n_hidden = config["n_hidden"]
max_iter = config["max_iter"]
tolerance_init = config["tolerance"]
time_list = config["time_list"]
L_list = config["L_list"]
save_time_complex = config["save_time_complex"]
save_MRR_MAP = config["save_MRR_MAP"]
save_sigma_mu = config["save_sigma_mu"]
scale = config["scale"]
seed = config["seed"]
verbose = config["verbose"]
name = 'Results/' + 'SBM'
# init_logging_handler(name)
# logging.info(str(config))
device = check_if_gpu()

name_loaded = 'Results/SBM'
with open(name_loaded + '/Eval_Results/saved_array/mu_as', 'rb') as f: mu_arr = pickle.load(f)
with open(name_loaded + '/Eval_Results/saved_array/sigma_as', 'rb') as f: sigma_arr = pickle.load(f)
data = get_data()

def get_all_edges_with_labels(A):
    # Extract positive edges
    positive_edges = list(zip(*A.nonzero()))

    # Generate all possible pairs of nodes
    all_pairs = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])]

    # Identify negative edges
    negative_edges = [pair for pair in all_pairs if A[pair[0], pair[1]] == 0]

    # Create labels
    labels = [1] * len(positive_edges) + [0] * len(negative_edges)

    # Combine edges and labels
    edges_with_labels = list(zip(positive_edges + negative_edges, labels))

    # Shuffle the combined list
    random.shuffle(edges_with_labels)

    return edges_with_labels
def unison_shuffled_copies(a, b, seed):
    assert len(a) == len(b)
    np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p]

from tqdm import tqdm

def get_inf(lookback, data,mu_64):
    return_dict = {}
    for ctr in tqdm(range(lookback, 35)):
        A = data[ctr][0].toarray()
        edges_with_labels = get_all_edges_with_labels(A)
        #sample 1000 edges
        edges_with_labels = random.sample(edges_with_labels, 203836)
        # Extract edges and labels
        edges = [edge_list[0] for edge_list in edges_with_labels]
        labels = torch.tensor([edge_list[1] for edge_list in edges_with_labels], dtype=torch.float32).to(device)

        # Create edges_matrix using list comprehension and vectorized operations
        edges_matrix = torch.tensor(
            [np.concatenate([mu_64[ctr - lookback][i], mu_64[ctr - lookback][j]]) for i, j in edges],
            dtype=torch.float32
        ).to(device)

        # Create a tensor of all the edges
        x = edges_matrix

        y = torch.tensor(labels, dtype=torch.float32).to(device)
        return_dict[ctr] = (x, y,labels)
    return return_dict


def get_MAP_avg(mu_arr,sigma_arr,lookback,data):
    mu_64 = mu_arr[0]
    sigma_64 = sigma_arr[0]

    class Classifier(torch.nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            activation = torch.nn.ReLU()

            self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=np.array(mu_64[0]).shape[1] * 2,
                                                           out_features=np.array(mu_64[0]).shape[1]),
                                           activation,
                                           torch.nn.Linear(in_features=np.array(mu_64[0]).shape[1],
                                                           out_features=1))

        def forward(self, x):
            return self.mlp(x)

    seed = 5
    torch.cuda.manual_seed_all(seed)
    classify = Classifier()
    classify.to(device)

    loss = torch.nn.BCEWithLogitsLoss(reduce=False)

    optim = torch.optim.Adam(classify.parameters(), lr=1e-3)

    num_epochs = 50
    return_dict = get_inf(lookback, data, mu_64)
    for epoch in range(num_epochs):
        for ctr in range(lookback, 35):

            x, y,labels = return_dict[ctr]
            weight = torch.tensor([0.1, 0.9]).to(device)
            y_hat = classify(x)
            #convert y_hat to 1,1
            y_hat = y_hat.reshape(y.shape)
            l = loss(y_hat, y)
            weight_ = weight[labels.data.view(-1).long()].view_as(labels)
            l = l * weight_
            l = l.mean()

            optim.zero_grad()
            l.backward()
            optim.step()
    num_epochs = 1
    MAP_time = []
    MRR_time = []
    time_ctr = 0
    for epoch in range(num_epochs):
        get_MAP_avg = []
        get_MRR_avg = []
        #     for i in range (70, len(val_timestep)):
        count = 0
        for ctr in range(40, 50):

            A_node = data[ctr][0].shape[0]
            A = data[ctr][0]

            if count >= 0:
                if A_node > A_prev_node:
                    A = A[:A_prev_node, :A_prev_node]

                if ctr >= 40:
                    # logging.debug('Testing')
                    # logging.debug(ctr)

                    ones_edj = A.nnz
                    zeroes_edj = A.shape[0] * 100
                    tot = ones_edj + zeroes_edj

                    val_ones = list(set(zip(*A.nonzero())))
                    val_ones = random.sample(val_ones, ones_edj)
                    val_ones = [list(ele) for ele in val_ones]
                    val_zeros = sample_zero_n(A, zeroes_edj)
                    val_zeros = [list(ele) for ele in val_zeros]
                    val_edges = np.row_stack((val_ones, val_zeros))

                    val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                    a, b = unison_shuffled_copies(val_edges, val_ground_truth, count)

                    if ctr > 0:

                        a_embed = np.array(mu_64[ctr - lookback])[a.astype(int)]
                        a_embed_sig = np.array(sigma_64[ctr - lookback])[a.astype(int)]

                        classify.eval()

                        inp_clf = []
                        for d_id in range(tot):
                            inp_clf.append(np.concatenate((a_embed[d_id][0], a_embed[d_id][1]), axis=0))

                        inp_clf = torch.tensor(np.asarray(inp_clf))

                        inp_clf = inp_clf.to(device)
                        with torch.no_grad():
                            out = classify(inp_clf).squeeze()

                        label = torch.tensor(np.asarray(b)).to(device)

                        weight = torch.tensor([0.1, 0.9]).to(device)

                        weight_ = weight[label.data.view(-1).long()].view_as(label)

                        l = loss(out, label)

                        l = l * weight_
                        l = l.mean()

                        MAP_val = get_MAP_e(out.cpu(), label.cpu(), None)
                        get_MAP_avg.append(MAP_val)

                        MRR = get_MRR(out.cpu(), label.cpu(), np.transpose(a))

                        get_MRR_avg.append(MRR)

                        '''try:
                            if ctr == time_list[time_ctr]:
                                MAP_time.append(MAP_val)
                                MRR_time.append(MRR)
                                time_ctr = time_ctr+1
                        except:
                            pass'''

                        # logging.debug(
                        #     'Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}, Running Mean MAP: {}, Running Mean MRR: {}'.format(
                        #         epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None), MRR,
                        #         np.asarray(get_MAP_avg).mean(), np.asarray(get_MRR_avg).mean()))

            A_prev_node = data[ctr][0].shape[0]
            count = count + 1


        return np.asarray(get_MAP_avg).mean() , np.asarray(get_MRR_avg).mean()



print(get_MAP_avg(mu_arr, sigma_arr, 2, data))


