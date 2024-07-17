from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse as sp
import itertools
import math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from models import *
from scipy import sparse
import random
from os import listdir
from os.path import isfile, join
import sklearn.metrics as metrics
import copy
from sklearn.metrics import precision_recall_curve
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import pickle
import time
import os
import logging
import json
import pandas as pd
from copy import deepcopy

print("Pid is", os.getpid())

# Read parameters from json file
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

lookback = config["lookback"]
# Load dataset
data = get_data()

name = 'Results/' + 'SBM'
init_logging_handler(name)
logging.info(str(config))
device = check_if_gpu()
logging.info('The code will be running on {}'.format(device))

# mu_64 = mu_64[2]
# sigma_64 = sigma_64[2]
name_loaded = 'Results/SBM'
with open(name_loaded + '/Eval_Results/saved_array/mu_as', 'rb') as f: mu_arr = pickle.load(f)
with open(name_loaded + '/Eval_Results/saved_array/sigma_as', 'rb') as f: sigma_arr = pickle.load(f)

MAP_l = []
MRR_l = []
time_list = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
print('This is', len(mu_arr))


def unison_shuffled_copies(a, b, seed):
    assert len(a) == len(b)
    np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def find_and_sample_zero_entries(sparse_matrix, num_samples=None):
    # Convert sparse matrix to dense format
    dense_matrix = sparse_matrix.toarray()

    # Create a boolean array: True where elements are zero
    zero_mask = dense_matrix == 0

    # Get the indices of zero elements
    zero_indices = np.argwhere(zero_mask)

    # Check if sampling is requested
    if num_samples is not None and num_samples > 0:
        # Ensure that there are enough zero entries to sample from
        if num_samples > len(zero_indices):
            raise ValueError("Requested more samples than available zeros.")
        # Sample indices randomly without replacement
        sampled_indices = zero_indices[np.random.choice(len(zero_indices), num_samples, replace=False)]
        return sampled_indices
    return zero_indices

def get_inf(data, mu_64, sigma_64, lookback):
    return_dict = {}
    #     for i in range (1, len(val_timestep) - 30):
    count = 0
    for ctr in range(lookback, 35):

        A_node = data[ctr][0].shape[0]
        A = data[ctr][0]

        if count > 0:
            if A_node > A_prev_node:
                A = A[:A_prev_node, :A_prev_node]

            if ctr < 35:

                ones_edj = A.nnz
                zeroes_edj = A.shape[0] * 100
                tot = ones_edj + zeroes_edj

                # Ensure A is in COO format
                A_coo = A.tocoo() if not isinstance(A, coo_matrix) else A

                # Get the pairs directly from the COO format properties
                val_ones = list(zip(A_coo.row, A_coo.col))

                val_ones = list(map(list, val_ones))

                val_zeros = find_and_sample_zero_entries(A, zeroes_edj)

                val_edges = np.row_stack((val_ones, val_zeros))

                val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                a, b = unison_shuffled_copies(val_edges, val_ground_truth, count)

                if ctr >= 0:
                    logging.debug('Training')
                    a_embed = np.array(mu_64[ctr - lookback])[a.astype(int)]

                    a_embed_stacked = np.vstack(a_embed)  # This stacks all [0] and [1] vertically

                    # Since we know every pair [0] and [1] are stacked sequentially, we can reshape:
                    n_features = a_embed.shape[2]  # Number of features in each sub-array
                    inp_clf_temp = a_embed_stacked.reshape(tot, 2 * n_features)

                    inp_clf = torch.tensor(inp_clf_temp)

                    inp_clf = inp_clf.to(device)
                    return_dict[ctr] = [inp_clf,b]
        A_prev_node = data[ctr][0].shape[0]
        count = count + 1
    return return_dict

for l_num in range(len(L_list)):

    mu_64 = mu_arr[l_num]
    sigma_64 = sigma_arr[l_num]




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
    return_dict = get_inf(data, mu_64, sigma_64, lookback)
    for epoch in range(num_epochs):
        #     for i in range (1, len(val_timestep) - 30):
        count = 0
        for ctr in range(lookback, 35):


            if count > 0:

                if ctr < 35:


                    if ctr >= 0:

                        classify.train()
                        decompose = return_dict[ctr]
                        inp_clf = decompose[0]
                        b = decompose[1]
                        out = classify(inp_clf).squeeze()

                        weight = torch.tensor([0.1, 0.9]).to(device)

                        label = torch.tensor(np.asarray(b)).to(device)

                        weight_ = weight[label.data.view(-1).long()].view_as(label)

                        l = loss(out, label)

                        l = l * weight_
                        l = l.mean()

                        optim.zero_grad()

                        l.backward()
                        optim.step()

                        MRR = 0

                        logging.debug('L:{}, Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}'.format(
                            np.array(mu_64[0]).shape[1], epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None),
                            MRR))

            A_prev_node = data[ctr][0].shape[0]
            count = count + 1

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
                    logging.debug('Testing')
                    logging.debug(ctr)

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

                        logging.debug(
                            'Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}, Running Mean MAP: {}, Running Mean MRR: {}'.format(
                                epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None), MRR,
                                np.asarray(get_MAP_avg).mean(), np.asarray(get_MRR_avg).mean()))

            A_prev_node = data[ctr][0].shape[0]
            count = count + 1

    MAP_l.append(MAP_time)
    MRR_l.append(MRR_time)






