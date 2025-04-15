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

from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
import numpy as np
from torch_geometric.nn import GINEConv, global_add_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

#from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index

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
lookback = config["lookback"]

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




# def val_loss(t):
#     l = []
#     for j in range(63, 72):
#         _, muval, sigmaval = t(val_data[j])
#         val_l = build_loss(triplet_dict[j], scale_dict[j], muval, sigmaval, 64, scale=False)
#         l.append(val_l.cpu().detach().numpy())
#     return np.mean(l)



def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute triplet loss: max(0, distance(anchor, positive) - distance(anchor, negative) + margin)

    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings
        negative: Negative embeddings
        margin: Margin value (default: 1.0)

    Returns:
        mean of triplet losses across the batch
    """
    distance_positive = torch.sum((anchor - positive) ** 2, dim=1)
    distance_negative = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.clamp(distance_positive - distance_negative + margin, min=0.0)
    return torch.mean(loss)


import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

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

        return x, mu, sigma


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2):
        super(LinkPredictor, self).__init__()
        self.fc1 = nn.Linear(2 * in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb_i, emb_j):
        # Concatenate the embeddings of the two nodes
        x = torch.cat([emb_i, emb_j], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


from sklearn.metrics import average_precision_score  # Built-in MAP computation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_by_threshold(scores, labels, threshold, smaller_scores_better=False, truth_label=1):
    """Evaluate predictions against ground truth using a threshold."""
    if smaller_scores_better:
        predictions = (scores <= threshold).int()
    else:
        predictions = (scores >= threshold).int()

    # Calculate metrics
    tp = ((predictions == truth_label) & (labels == truth_label)).sum().item()
    fp = ((predictions == truth_label) & (labels != truth_label)).sum().item()
    fn = ((predictions != truth_label) & (labels == truth_label)).sum().item()
    tn = ((predictions != truth_label) & (labels != truth_label)).sum().item()

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def search_best_threshold(scores, labels, threshold_granularity=100, smaller_scores_better=False, truth_label=1,
                          determined_metric="F1"):
    """Search for the threshold that maximizes a specific metric."""
    best_metric_value = -float('inf')
    best_results = None

    start = int(scores.min().item() * threshold_granularity)
    end = int(scores.max().item() * threshold_granularity) + 1

    for t in range(start, end):
        threshold = t / threshold_granularity
        results = evaluate_by_threshold(scores, labels, threshold, smaller_scores_better, truth_label)

        if results[determined_metric] > best_metric_value:
            best_metric_value = results[determined_metric]
            best_results = results

    return best_results

def optimise_mamba(lookback, dim_in, d_conv, d_state, dropout, lr, weight_decay, walk_length):
    # Create dataset
    dataset = RMDataset(data, lookback, walk_length)
    config = {
        'd_model': 96,
        'd_state': d_state,
        'd_conv': d_conv
    }

    # Instantiate your main model (MambaG2G) and move to device
    model = MambaG2G(config, dim_in, 64, dropout=dropout).to(device)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    # Define the link prediction head.
    # This head takes two node embeddings (e.g., mu) and outputs a probability.
    class LinkPredictor(nn.Module):
        def __init__(self, in_channels, hidden_channels, dropout=0.2):
            super(LinkPredictor, self).__init__()
            self.fc1 = nn.Linear(2 * in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, emb_i, emb_j):
            # Concatenate the embeddings for each node pair along the feature dimension.
            x = torch.cat([emb_i, emb_j], dim=-1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return torch.sigmoid(x)

    # Instantiate the link predictor (using 64 as the input node embedding dimension)
    link_predictor = LinkPredictor(in_channels=64, hidden_channels=32, dropout=dropout).to(device)

    # Merge parameters from both models into one optimizer.
    optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()),
                                 lr=lr, weight_decay=weight_decay)

    # Set weights for the loss components.
    alpha = 0.5  # weighting for the triplet loss
    beta = 0.5  # weighting for the link prediction (BCE) loss

    epochs = 50
    val_losses = []
    train_loss = []
    test_loss = []
    best_MAP = 0
    best_model = None

    for e in tqdm(range(epochs)):
        model.train()
        link_predictor.train()
        loss_step = []

        # Training loop over snapshots in the training period
        for i in range(lookback, 63):
            # Get one time snapshot from the dataset.
            x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
            optimizer.zero_grad()

            # Move input to device
            x = x.clone().detach().requires_grad_(True).to(device)

            # Forward pass: obtain node embeddings along with mu and sigma.
            _, mu, sigma = model(x)

            # 1. Compute triplet loss for embedding learning
            triplet_tensor = torch.tensor(triplet, dtype=torch.int64).to(device)
            # Replace the triplet loss calculation with:
            # Extract anchor, positive, and negative embeddings
            anchors = mu[triplet_tensor[:, 0]]
            positives = mu[triplet_tensor[:, 1]]
            negatives = mu[triplet_tensor[:, 2]]

            # Compute triplet loss
            loss_triplet = triplet_loss(anchors, positives, negatives, margin=1.0)

            # 2. Compute link prediction loss using the link predictor:
            # Convert triplets to a tensor (each row: [anchor, positive, negative])

            # Positive pairs: (anchor, positive)
            pos_pair = triplet_tensor[:, [0, 1]]
            # Negative pairs: (anchor, negative)
            neg_pair = triplet_tensor[:, [0, 2]]

            # Use the node embeddings (here, mu) for link prediction.
            pred_pos = link_predictor(mu[pos_pair[:, 0]], mu[pos_pair[:, 1]])
            pred_neg = link_predictor(mu[neg_pair[:, 0]], mu[neg_pair[:, 1]])

            # Create target labels: ones for positive pairs, zeros for negative.
            label_pos = torch.ones_like(pred_pos)
            label_neg = torch.zeros_like(pred_neg)

            # Compute BCE loss for positive and negative predictions.
            loss_bce_pos = F.binary_cross_entropy(pred_pos, label_pos)
            loss_bce_neg = F.binary_cross_entropy(pred_neg, label_neg)
            loss_bce = loss_bce_pos + loss_bce_neg

            # Combine the losses into one joint loss.
            joint_loss = alpha * loss_triplet + beta * loss_bce
            loss_step.append(joint_loss.cpu().detach().numpy())

            joint_loss.backward()
            clip_grad_norm_(list(model.parameters()) + list(link_predictor.parameters()), max_norm=1.0)
            optimizer.step()

        if (e+1) % 5 == 0:
            # Validation loop over snapshots 63 to 72 (using triplet loss as the evaluation metric)
            val_loss_value = 0.0
            val_samples = 0
            with torch.no_grad():
                model.eval()
                link_predictor.eval()
                for i in range(63, 72):
                    x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
                    x = x.clone().detach().to(device)
                    _, mu, sigma = model(x)
                    triplet_tensor = torch.tensor(triplet, dtype=torch.int64).to(device)
                    # Replace the triplet loss calculation with:
                    # Extract anchor, positive, and negative embeddings
                    anchors = mu[triplet_tensor[:, 0]]
                    positives = mu[triplet_tensor[:, 1]]
                    negatives = mu[triplet_tensor[:, 2]]

                    # Compute triplet loss
                    curr_val_loss = triplet_loss(anchors, positives, negatives, margin=1.0)

                    val_loss_value += curr_val_loss
                    val_samples += 1
                val_loss_value /= val_samples
            val_losses.append(val_loss_value)
            train_loss.append(np.mean(np.stack(loss_step)))

            # Testing loop over snapshots 72 to 90 (using triplet loss as the evaluation metric)
            test_loss_value = 0.0
            test_samples = 0
            with torch.no_grad():
                model.eval()
                link_predictor.eval()
                for i in range(72, 90):
                    x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
                    x = x.clone().detach().to(device)
                    _, mu, sigma = model(x)
                    triplet_tensor = torch.tensor(triplet, dtype=torch.int64).to(device)
                    # Replace the triplet loss calculation with:
                    # Extract anchor, positive, and negative embeddings
                    anchors = mu[triplet_tensor[:, 0]]
                    positives = mu[triplet_tensor[:, 1]]
                    negatives = mu[triplet_tensor[:, 2]]

                    # Compute triplet loss
                    curr_test_loss = triplet_loss(anchors, positives, negatives, margin=1.0)

                    test_loss_value += curr_test_loss
                    test_samples += 1
                test_loss_value /= test_samples
            test_loss.append(test_loss_value)

            # Calculate MAP using the built-in average_precision_score for snapshots lookback to 90.
            # Here we evaluate on both the positive and negative link predictions.
            # Add this to your optimise_mamba function where MAP is calculated
            ap_list = []
            f1_list = []  # Track F1 scores
            best_thresholds = []  # Track best thresholds

            with torch.no_grad():
                model.eval()
                link_predictor.eval()
                for i in range(lookback, 90):
                    x, pe, edge_index, edge_attr, batch, triplet, scale = dataset[i]
                    x = x.clone().detach().to(device)
                    _, mu, sigma = model(x)

                    # Convert triplets to tensor
                    triplet_tensor = torch.tensor(triplet, dtype=torch.long).to(device)
                    pos_pair = triplet_tensor[:, [0, 1]]
                    neg_pair = triplet_tensor[:, [0, 2]]

                    # Get predicted probabilities
                    pred_pos = link_predictor(mu[pos_pair[:, 0]], mu[pos_pair[:, 1]]).squeeze().cpu()
                    pred_neg = link_predictor(mu[neg_pair[:, 0]], mu[neg_pair[:, 1]]).squeeze().cpu()

                    # Construct predictions and ground-truth labels
                    preds = torch.cat([pred_pos, pred_neg])
                    labels = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)])

                    # Skip calculation if we have less than two unique labels
                    if len(torch.unique(labels)) < 2:
                        continue

                    # Calculate MAP
                    ap = average_precision_score(labels.numpy(), preds.numpy())
                    ap_list.append(ap)

                    # Find best threshold and calculate F1 score
                    results = search_best_threshold(
                        scores=preds,
                        labels=labels,
                        threshold_granularity=100,
                        smaller_scores_better=False,
                        determined_metric="F1"
                    )

                    if results:
                        f1_list.append(results["F1"])
                        best_thresholds.append(results["threshold"])

            # Compute mean average precision and mean F1 across evaluated snapshots
            if len(ap_list) > 0:
                curr_MAP = np.mean(ap_list)
                curr_F1 = np.mean(f1_list) if len(f1_list) > 0 else 0
                avg_threshold = np.mean(best_thresholds) if len(best_thresholds) > 0 else 0.5
            else:
                curr_MAP = 0
                curr_F1 = 0
                avg_threshold = 0.5

            # Print both MAP and F1 values
            print(f"Epoch {e + 1}/{epochs}")
            print(f"Training Loss: {np.mean(np.stack(loss_step)):.4f}")
            print(f"Validation Loss: {val_loss_value:.4f}")
            print(f"Test Loss: {test_loss_value:.4f}")
            print(f"Current MAP: {curr_MAP:.4f}")
            print(f"Current F1: {curr_F1:.4f} (threshold: {avg_threshold:.2f})")
            print("-" * 50)

            # Check if the current MAP is the best so far
            if curr_MAP > best_MAP:
                best_MAP = curr_MAP
                best_model = model
                print(f"New Best MAP: {best_MAP:.4f} (F1: {curr_F1:.4f})")
    return best_model, val_losses, train_loss, test_loss


# Train/Val/Test split

# train_data = {}
# for i in range(lookback, 63):
#     train = torch.tensor(dataset[i], dtype=torch.float32)
#     train_data[i] = train.to(device)
#
# val_data = {}
# for i in range(63, 72):
#     val = torch.tensor(dataset[i], dtype=torch.float32)
#     val_data[i] = val.to(device)
#
# test_data = {}
# for i in range(72, 90):
#     test = torch.tensor(dataset[i], dtype=torch.float32)
#     test_data[i] = test.to(device)
#

lookback = 4
walk = 16
model , val_losses , loss_step , test_loss = optimise_mamba(lookback=lookback,dim_in=64,d_conv=3,d_state=16,dropout=0.4285,lr=1e-4,weight_decay=1e-4,walk_length=walk)


