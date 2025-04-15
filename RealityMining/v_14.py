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


# ----- Link Predictor -----
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2):
        super(LinkPredictor, self).__init__()
        self.fc1 = nn.Linear(2 * in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb_i, emb_j):
        x = torch.cat([emb_i, emb_j], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


# ============================================================
# Stage 1: Pretrain the Node Embedder using triplet loss only.
def pretrain_node_embedder(dataset, model, optimizer, epochs=50):
    print("Stage 1: Pretraining Node Embedder with Triplet Loss")
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        # We iterate over a chosen time span.
        # Here, we use snapshots from "lookback" index up to 63 (adjust this as needed).
        for i in range(lookback, 63):
            # Get one snapshot sample
            x, _, _, _, _, triplet, _ = dataset[i]
            optimizer.zero_grad()
            x = x.clone().detach().to(device)
            # Forward pass through node embedder
            _, mu, _ = model(x)
            # Convert triplet list to tensor
            triplet_tensor = torch.tensor(triplet, dtype=torch.int64).to(device)
            anchors = mu[triplet_tensor[:, 0]]
            positives = mu[triplet_tensor[:, 1]]
            negatives = mu[triplet_tensor[:, 2]]
            loss = triplet_loss(anchors, positives, negatives, margin=1.0)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Triplet Loss: {np.mean(epoch_losses):.4f}")
    return model


# ============================================================
# Stage 2: Train the Link Predictor using the (now pretrained) node embeddings.
def train_link_predictor(dataset, node_model, link_predictor, optimizer, epochs=20):
    print("Stage 2: Training Link Predictor (with frozen node embedder)")
    # Freeze node embedder parameters so that only the link predictor is updated
    for param in node_model.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        epoch_losses = []
        link_predictor.train()
        node_model.eval()
        # You can reuse the same snapshot indices or adjust as needed.
        for i in range(lookback, 63):
            x, _, _, _, _, triplet, _ = dataset[i]
            optimizer.zero_grad()
            x = x.clone().detach().to(device)
            with torch.no_grad():
                # Get node embeddings; no gradient flow into node_model
                _, mu, _ = node_model(x)
            # Process triplets for link prediction: positives and negatives.
            triplet_tensor = torch.tensor(triplet, dtype=torch.long).to(device)
            pos_pair = triplet_tensor[:, [0, 1]]
            neg_pair = triplet_tensor[:, [0, 2]]
            pred_pos = link_predictor(mu[pos_pair[:, 0]], mu[pos_pair[:, 1]])
            pred_neg = link_predictor(mu[neg_pair[:, 0]], mu[neg_pair[:, 1]])
            label_pos = torch.ones_like(pred_pos)
            label_neg = torch.zeros_like(pred_neg)
            loss_bce_pos = F.binary_cross_entropy(pred_pos, label_pos)
            loss_bce_neg = F.binary_cross_entropy(pred_neg, label_neg)
            loss_bce = loss_bce_pos + loss_bce_neg
            loss_bce.backward()
            clip_grad_norm_(link_predictor.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss_bce.item())
        print(f"Epoch {epoch + 1}/{epochs}, Link Predictor BCE Loss: {np.mean(epoch_losses):.4f}")
    return link_predictor


# ===== Stage 3: Evaluate the trained models =====
def evaluate_model(dataset, node_model, link_predictor, snapshot_range, name="Validation"):
    """Evaluate the trained models on a specific range of snapshots."""
    print(f"Evaluating on {name} snapshots ({snapshot_range[0]}-{snapshot_range[1]})...")

    ap_list = []
    f1_list = []
    best_thresholds = []

    node_model.eval()
    link_predictor.eval()

    with torch.no_grad():
        for i in range(snapshot_range[0], snapshot_range[1]):
            x, _, _, _, _, triplet, _ = dataset[i]
            x = x.clone().detach().to(device)
            _, mu, _ = node_model(x)

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

    # Compute mean metrics across evaluated snapshots
    metrics = {}
    if len(ap_list) > 0:
        metrics["MAP"] = np.mean(ap_list)
        metrics["F1"] = np.mean(f1_list) if len(f1_list) > 0 else 0
        metrics["avg_threshold"] = np.mean(best_thresholds) if len(best_thresholds) > 0 else 0.5
    else:
        metrics["MAP"] = 0
        metrics["F1"] = 0
        metrics["avg_threshold"] = 0.5

    print(f"{name} MAP: {metrics['MAP']:.4f}")
    print(f"{name} F1: {metrics['F1']:.4f} (threshold: {metrics['avg_threshold']:.2f})")
    return metrics


# ============================================================
# Main training script
if __name__ == '__main__':
    # Load your configuration and dataset
    with open("config.json") as f:
        config = json.load(f)
    lookback = 4  # Adjust lookback as needed
    walk = 16  # Walk length for the random walk PE

    # Assuming dataset_mit is your data loader function
    data = dataset_mit('..')  # Replace with the appropriate data path or function
    dataset = RMDataset(data, lookback, walk_length=walk)

    # ===== Stage 1: Pretrain Node Embedder =====
    model_config = {
        'd_model': 96,
        'd_state': 16,  # Adjust as needed
        'd_conv': 3  # Adjust as needed
    }
    # Instantiate the node embedder (MambaG2G)
    node_model = MambaG2G(model_config, dim_in=64, dim_out=64, dropout=0.4285).to(device)
    optimizer_stage1 = torch.optim.Adam(node_model.parameters(), lr=2e-4, weight_decay=1e-5)
    pretrain_epochs = 50  # Number of epochs for stage 1 pretraining

    node_model = pretrain_node_embedder(dataset, node_model, optimizer_stage1, epochs=pretrain_epochs)

    # ===== Stage 2: Train Link Predictor =====
    # Instantiate the Link Predictor
    link_predictor = LinkPredictor(in_channels=64, hidden_channels=32, dropout=0.2).to(device)
    optimizer_stage2 = torch.optim.Adam(link_predictor.parameters(), lr=2e-4, weight_decay=1e-4)
    finetune_epochs = 50  # Adjust number of epochs

    link_predictor = train_link_predictor(dataset, node_model, link_predictor, optimizer_stage2, epochs=finetune_epochs)

    # Optionally, you can now evaluate your model on validation/test snapshots.
    # For example, by computing MAP, F1, etc., using the pretrained node_model and link_predictor.
    # Run evaluation on validation and test snapshots
    val_metrics = evaluate_model(dataset, node_model, link_predictor, (63, 72), "Validation")
    test_metrics = evaluate_model(dataset, node_model, link_predictor, (72, 90), "Test")

    # Print comparative results
    print("\n===== Final Evaluation Results =====")
    print(f"Validation MAP: {val_metrics['MAP']:.4f}, F1: {val_metrics['F1']:.4f}")
    print(f"Test MAP: {test_metrics['MAP']:.4f}, F1: {test_metrics['F1']:.4f}")
