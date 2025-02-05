# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".



import torch_geometric.transforms as T
import os
import sys
try :
    os.chdir("RealityMining")
    sys.path.append(os.getcwd())
except:
    pass
from utils import *
import pickle
import json
from eval_mod import get_MAP_avg
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, precision_recall_curve

import warnings
warnings.filterwarnings("ignore")

import itertools

from torch.nn import ELU,Dropout

from mamba_ssm import Mamba
from tqdm import tqdm


from torch.nn.utils import clip_grad_norm_



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

torch.backends.cudnn.deterministic=True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



f = open("config.json")
config = json.load(f)
lookback = 5

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Get dataset and construct dict
data = dataset_mit('..')

class RMDataset(Dataset):
    def __init__(self, data, lookback,walk_length=20):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)
        self.transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')


    def temp_process(self, data, lookback):
        dataset = {}
        for i in range(lookback, 90):
            B = np.zeros((96, lookback + 1, 96))
            for j in range(lookback + 1):
                adj_matr = data[i - lookback + j][0].todense()
                B[:adj_matr.shape[0], j, :adj_matr.shape[1]] = adj_matr
            dataset[i] = B

        # Construct dict of hops and scale terms
        hop_dict = {}
        scale_terms_dict = {}
        print("Constructing dict of hops and scale terms")
        for i in range(lookback, 90):
            hops = get_hops(data[i][0], 2)
            scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                               hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
            hop_dict[i] = hops
            scale_terms_dict[i] = scale_terms

        # Construct dict of triplets
        triplet_dict = {}
        scale_dict = {}
        print("Constructing dict of triplets")
        for i in range(lookback, 90):
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            triplet_dict[i] = triplet
            scale_dict[i] = scale
        return dataset, triplet_dict, scale_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = torch.tensor(data, dtype=torch.float32)

        # Create placeholders for edge_index and edge_attr
        edge_index_list = []
        edge_attr_list = []

        for i in range(x.size(1)):  # Iterate over lookback + 1 adjacency matrices
            adj_matrix = x[:, i, :]
            edge_index, edge_attr = dense_to_sparse(adj_matrix)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        # Create a pseudo graph data object for PE calculation
        graph_data = Data(x=torch.ones(x.size(0), 1), edge_index=edge_index, edge_attr=edge_attr)
        graph_data = self.transform(graph_data)

        pe = graph_data.pe  # Positional encodings

        batch = torch.zeros(x.size(0), dtype=torch.long)

        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]
        return x, pe, edge_index, edge_attr, batch, triplet, scale


import torch
import torch.nn as nn
import torch.nn.functional as F


def Energy_KL(mu, sigma, pairs, L):
    ij_mu = mu[pairs]
    ij_sigma = sigma[pairs]
    sigma_ratio = ij_sigma[:, 1] / (ij_sigma[:, 0] + 1e-14)
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), 1)
    mu_diff_sq = torch.sum(torch.square(ij_mu[:, 0] - ij_mu[:, 1]) / (ij_sigma[:, 0] + 1e-14), 1)
    return 0.5 * (trace_fac + mu_diff_sq - L - log_det)


# Define loss function
def build_loss(triplets, scale_terms, mu, sigma, L, scale):
    hop_pos = torch.stack([torch.tensor(triplets[:, 0]), torch.tensor(triplets[:, 1])], 1).type(torch.int64)
    hop_neg = torch.stack([torch.tensor(triplets[:, 0]), torch.tensor(triplets[:, 2])], 1).type(torch.int64)
    eng_pos = Energy_KL(mu, sigma, hop_pos, L)
    eng_neg = Energy_KL(mu, sigma, hop_neg, L)
    energy = torch.square(eng_pos) + torch.exp(-eng_neg)
    if scale:
        loss = torch.mean(energy * torch.Tensor(scale_terms).cpu())
    else:
        loss = torch.mean(energy)
    return loss


from RealityMining.mamba import Mamba, MambaConfig

class MambaG2G(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.config = MambaConfig(d_model=config['d_model'], n_layers=1,d_state=config['d_state'], d_conv=config['d_conv'])
        self.mamba = Mamba(self.config)
        #self.mamba = MambaBlock(seq_len=lookback + 1, d_model=config['d_model'], state_size=config['d_state'], batch_size=96, device=device)
        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.device = device
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)

    def forward(self, input):
        # e = self.enc_input_fc(input)
        e = self.mamba(input)[0]
        e = e.mean(dim=1)  # Average pooling to maintain the expected shape
        e = self.dropout(e)  # Apply dropout after average pooling
        x = torch.tanh(self.out_fc(e))
        x = self.elu(x)
        x = self.dropout(x)  # Apply dropout after the activation
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        sigma = self.elu(sigma) + 1 + 1e-14

        return mu, sigma


# **Triplet Loss Function**
def triplet_loss(triplet, mu, margin=1.0):
    pos_dist = torch.norm(mu[triplet[:, 0]] - mu[triplet[:, 1]], dim=1)
    neg_dist = torch.norm(mu[triplet[:, 0]] - mu[triplet[:, 2]], dim=1)
    return torch.relu(pos_dist - neg_dist + margin).mean()


# **Training Embedder**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    'd_model': 96,
    'd_state': 8,
    'd_conv': 3
}

embedder = MambaG2G(config, 96, 64, dropout=0.2).to(device)
optimizer_emb = torch.optim.Adam(embedder.parameters(), lr=0.001, weight_decay=1e-3)

# Training Loop (Only Embedding Learning)
epochs = 1
dataset = RMDataset(data, lookback=lookback, walk_length=16)

for epoch in range(epochs):
    embedder.train()
    loss_step = []

    for i in range(lookback, 63):
        x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
        x = x.to(device)

        optimizer_emb.zero_grad()
        embeddings = embedder(x)  # Generate embeddings
        loss_emb =build_loss(triplet, scale, embeddings[0], embeddings[1], 64, False)
        loss_emb.backward()
        optimizer_emb.step()

        loss_step.append(loss_emb.item())

    print(f"Epoch {epoch + 1}: Embedding Loss: {sum(loss_step) / len(loss_step):.4f}")

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve


class LinkPredictor(nn.Module):
    def __init__(self, embedding_dim=64):
        super(LinkPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x)


# Training Loop for Link Prediction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
link_predictor = LinkPredictor().to(device)
optimizer_link = torch.optim.Adam(link_predictor.parameters(), lr=0.0001,weight_decay=1e-3)
criterion = nn.BCEWithLogitsLoss()

def train_link_prediction(embedder, dataset, lookback=3):
    for i in range(lookback, 63):  # Same range as embedding training
        x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
        x = x.to(device)

        # Get embeddings from trained embedder
        embeddings , _ = embedder(x)

        # Get adjacency matrix for current timestep
        adj_matrix = x[:, -1, :]  # Use last timestep matrix
        edge_index, _ = dense_to_sparse(adj_matrix)

        # Prepare positive samples
        pos_edges = edge_index.t().cpu().numpy()
        num_pos = pos_edges.shape[0]

        # Sample negative edges
        num_nodes = adj_matrix.shape[0]
        neg_edges = []
        existing_edges = set(map(tuple, pos_edges))

        while len(neg_edges) < num_pos:
            i, j = np.random.randint(0, num_nodes, 2)
            if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
                neg_edges.append([i, j])
                existing_edges.add((i, j))

        neg_edges = np.array(neg_edges)

        # Combine positive and negative edges
        all_edges = np.vstack([pos_edges, neg_edges])
        all_labels = np.hstack([np.ones(num_pos), np.zeros(num_pos)])

        # Create edge features by concatenating node embeddings
        src_nodes = embeddings[torch.LongTensor(all_edges[:, 0]).to(device)]
        dst_nodes = embeddings[torch.LongTensor(all_edges[:, 1]).to(device)]
        edge_features = torch.cat([src_nodes, dst_nodes], dim=1)

        # Train step
        optimizer_link.zero_grad()
        pred = link_predictor(edge_features).squeeze()
        loss = criterion(pred, torch.FloatTensor(all_labels).to(device))
        loss.backward()
        optimizer_link.step()

        # Calculate metrics
        with torch.no_grad():
            prob = torch.sigmoid(pred).cpu().numpy()
            precision, recall, _ = precision_recall_curve(all_labels, prob)
            auc_score = auc(recall, precision)
            ap_score = average_precision_score(all_labels, prob)

        yield loss.item(), auc_score, ap_score


def evaluate_link_prediction(embedder, dataset, timestep):
    embedder.eval()
    link_predictor.eval()

    with torch.no_grad():
        x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[timestep]
        x = x.to(device)

        # Get embeddings
        embeddings = embedder(x)

        # Get adjacency matrix
        adj_matrix = x[:, -1, :]
        edge_index, _ = dense_to_sparse(adj_matrix)

        # Prepare positive and negative samples (same as training)
        pos_edges = edge_index.t().cpu().numpy()
        num_pos = pos_edges.shape[0]

        num_nodes = adj_matrix.shape[0]
        neg_edges = []
        existing_edges = set(map(tuple, pos_edges))

        while len(neg_edges) < num_pos:
            i, j = np.random.randint(0, num_nodes, 2)
            if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
                neg_edges.append([i, j])
                existing_edges.add((i, j))

        neg_edges = np.array(neg_edges)

        all_edges = np.vstack([pos_edges, neg_edges])
        all_labels = np.hstack([np.ones(num_pos), np.zeros(num_pos)])

        # Create edge features
        src_nodes = embeddings[torch.LongTensor(all_edges[:, 0]).to(device)]
        dst_nodes = embeddings[torch.LongTensor(all_edges[:, 1]).to(device)]
        edge_features = torch.cat([src_nodes, dst_nodes], dim=1)

        # Get predictions
        pred = link_predictor(edge_features).squeeze()
        prob = torch.sigmoid(pred).cpu().numpy()

        # Calculate metrics
        precision, recall, _ = precision_recall_curve(all_labels, prob)
        auc_score = auc(recall, precision)
        ap_score = average_precision_score(all_labels, prob)

        return auc_score, ap_score


# Example usage:
num_epochs = 50
for epoch in range(num_epochs):
    total_loss, total_auc, total_ap = 0, 0, 0
    num_steps = 0

    # Training
    embedder.eval()  # Keep embedder in eval mode since it's already trained
    link_predictor.train()

    for loss, auc_score, ap_score in train_link_prediction(embedder, dataset,lookback=lookback):
        total_loss += loss
        total_auc += auc_score
        total_ap += ap_score
        num_steps += 1

    avg_loss = total_loss / num_steps
    avg_auc = total_auc / num_steps
    avg_ap = total_ap / num_steps

    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg AUC: {avg_auc:.4f}")
    print(f"  Avg AP: {avg_ap:.4f}")

import torch
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_curve)
import random
from scipy.sparse import csr_matrix


def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal threshold using ROC curve and Youden's J statistic
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def calculate_metrics(embedder, link_predictor, dataset, test_timesteps=range(72, 90), mult_test=50):
    """
    Calculate metrics across test timesteps with dynamic thresholding
    """
    device = embedder.device
    MAP_scores = []
    F1_scores = []
    Precision_scores = []
    Recall_scores = []
    Thresholds = []

    embedder.eval()
    link_predictor.eval()

    with torch.no_grad():
        for ctr in test_timesteps:
            # Get data for current timestep
            x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[ctr]
            x = x.to(device)

            # Get embeddings
            embeddings,_ = embedder(x)

            # Get adjacency matrix for current timestep
            A = dataset.data[ctr][0]

            # Sample edges
            ones_edj = A.nnz
            zeroes_edj = min(A.shape[0] * mult_test,
                             (A.shape[0] - 1) * (A.shape[0] - 1) - A.nnz)

            # Get positive edges
            val_ones = list(set(zip(*A.nonzero())))
            val_ones = random.sample(val_ones, ones_edj)
            val_ones = [list(ele) for ele in val_ones]

            # Sample negative edges
            def sample_zero_n(mat, n):
                sampled = set()
                while len(sampled) < n:
                    i = random.randint(0, mat.shape[0] - 1)
                    j = random.randint(0, mat.shape[0] - 1)
                    if i != j and A[i, j] == 0:
                        sampled.add((i, j))
                return [list(x) for x in sampled]

            val_zeros = sample_zero_n(A, zeroes_edj)

            # Combine edges and create ground truth
            val_edges = np.row_stack((val_ones, val_zeros))
            val_ground_truth = np.concatenate([np.ones(len(val_ones)),
                                               np.zeros(len(val_zeros))])

            # Create edge features
            src_nodes = embeddings[torch.LongTensor(val_edges[:, 0]).to(device)]
            dst_nodes = embeddings[torch.LongTensor(val_edges[:, 1]).to(device)]
            edge_features = torch.cat([src_nodes, dst_nodes], dim=1)

            # Get predictions
            predictions = link_predictor(edge_features).squeeze()
            probs = torch.sigmoid(predictions).cpu().numpy()

            # Find optimal threshold
            threshold = find_optimal_threshold(val_ground_truth, probs)
            Thresholds.append(threshold)

            # Calculate MAP
            map_score = average_precision_score(val_ground_truth, probs)
            MAP_scores.append(map_score)

            # Calculate F1 using optimal threshold
            pred_labels = (probs > threshold).astype(int)
            f1 = f1_score(val_ground_truth, pred_labels)
            precision = precision_score(val_ground_truth, pred_labels)
            recall = recall_score(val_ground_truth, pred_labels)

            F1_scores.append(f1)
            Precision_scores.append(precision)
            Recall_scores.append(recall)

            print(f"Timestep {ctr}:")
            print(f"  Optimal Threshold: {threshold:.4f}")
            print(f"  MAP: {map_score:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")

    # Calculate final metrics
    final_metrics = {
        'MAP': np.mean(MAP_scores),
        'F1': np.mean(F1_scores),
        'Precision': np.mean(Precision_scores),
        'Recall': np.mean(Recall_scores),
        'Avg_Threshold': np.mean(Thresholds)
    }

    # Calculate scores by timestep
    scores_by_timestep = {
        timestep: {
            'Threshold': threshold,
            'MAP': map_score,
            'F1': f1_score,
            'Precision': prec_score,
            'Recall': rec_score
        }
        for timestep, threshold, map_score, f1_score, prec_score, rec_score
        in zip(test_timesteps, Thresholds, MAP_scores, F1_scores,
               Precision_scores, Recall_scores)
    }

    print("\nFinal Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    return final_metrics, scores_by_timestep


# Example usage:
def get_all_metrics(embedder, link_predictor, dataset):
    """
    Get final results with all metrics using dynamic thresholding
    """
    final_metrics, scores_by_timestep = calculate_metrics(
        embedder=embedder,
        link_predictor=link_predictor,
        dataset=dataset,
        test_timesteps=range(72, 90),
        mult_test=50
    )

    return {
        'final_metrics': final_metrics,
        'scores_by_timestep': scores_by_timestep
    }

results = get_all_metrics(embedder, link_predictor, dataset)

