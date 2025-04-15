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
import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset

def precompute_all_neighbors(A_sequence):
    """Precompute neighbors for all snapshots in the sequence."""
    return [A.tocsr().tolil().rows for A in A_sequence]



def temporal_sample_hops(neighbor_seq, node_idx, time_idx, lookback, num_samples=5):
    """
    Sample nodes from temporal neighborhoods for triplet formation.

    neighbor_seq: List of neighbor lists per snapshot.
                  Each element is a list/array where neighbor_seq[t][i] gives neighbors of node i.
    node_idx: The anchor node index.
    time_idx: The current time index (relative within the sequence).
    lookback: The number of previous time steps to consider.
    num_samples: The maximum number of samples to draw per category.
    """
    samples = {}

    # Gather recent neighbors from snapshots in the window [max(0, time_idx-lookback), time_idx]
    recent_neighbors = set()
    for t in range(max(0, time_idx - lookback), time_idx + 1):
        # neighbor_seq[t] is a list: neighbor_seq[t][node_idx] gives neighbors of node_idx at time t.
        rec_neighbors = neighbor_seq[t][node_idx]
        if rec_neighbors.size > 0:
            recent_neighbors.update(rec_neighbors.tolist())

    if recent_neighbors:
        recent_list = list(recent_neighbors)
        samples['positive'] = np.random.choice(
            recent_list,
            size=min(num_samples, len(recent_list)),
            replace=len(recent_list) < num_samples
        )
    else:
        samples['positive'] = np.array([], dtype=np.int64)

    # Gather historical neighbors from snapshots before the recent window
    historical_neighbors = set()
    if time_idx > lookback:
        for t in range(0, time_idx - lookback):
            hist_neighbors = neighbor_seq[t][node_idx]
            if hist_neighbors.size > 0:
                historical_neighbors.update(hist_neighbors.tolist())
        # Remove those already in recent_neighbors to avoid overlap.
        historical_neighbors = historical_neighbors - recent_neighbors
        if historical_neighbors:
            hist_list = list(historical_neighbors)
            samples['historical'] = np.random.choice(
                hist_list,
                size=min(num_samples, len(hist_list)),
                replace=len(hist_list) < num_samples
            )
        else:
            samples['historical'] = np.array([], dtype=np.int64)
    else:
        samples['historical'] = np.array([], dtype=np.int64)

    # Negative candidates: nodes that are not neighbors in the recent window (and not the anchor).
    num_nodes = len(neighbor_seq[0])
    all_nodes = set(range(num_nodes))
    all_nodes.discard(node_idx)
    non_neighbors = all_nodes - recent_neighbors
    if non_neighbors:
        non_list = list(non_neighbors)
        samples['negative'] = np.random.choice(
            non_list,
            size=min(num_samples, len(non_list)),
            replace=len(non_list) < num_samples
        )
    else:
        samples['negative'] = np.array([], dtype=np.int64)

    return samples


def generate_temporal_triplets(A_sequence, time_idx, lookback, num_triplets=1000):
    """
    Generate triplets for training using temporal information.

    A_sequence: List of sparse adjacency matrices.
                Precomputed neighbor lists for these matrices are assumed.
    time_idx: Current time index (within the sequence).
    lookback: Number of previous time steps to consider.
    num_triplets: Maximum number of triplets to generate.
    """
    triplets = []
    num_nodes = A_sequence[0].shape[0]

    # Precompute neighbor lists for each snapshot in the sequence
    neighbor_seq = [precompute_all_neighbors(A) for A in A_sequence]

    # Sample a set of anchor nodes (we sample extra to ensure enough triplets)
    anchor_nodes = np.random.choice(num_nodes, size=min(num_nodes, num_triplets * 2), replace=False)

    for anchor in anchor_nodes:
        samples = temporal_sample_hops(neighbor_seq, anchor, time_idx, lookback)
        # Use recent positives first
        if samples['positive'].size > 0 and samples['negative'].size > 0:
            for pos in samples['positive']:
                for neg in samples['negative']:
                    if anchor != pos and anchor != neg and pos != neg:
                        triplets.append([anchor, pos, neg])
                        if len(triplets) >= num_triplets:
                            return np.array(triplets)
        # If not enough, add historical positives
        if samples['historical'].size > 0 and samples['negative'].size > 0:
            for pos in samples['historical']:
                for neg in samples['negative']:
                    if anchor != pos and anchor != neg and pos != neg:
                        triplets.append([anchor, pos, neg])
                        if len(triplets) >= num_triplets:
                            return np.array(triplets)

    if len(triplets) == 0:
        # Fall back to a dummy triplet if nothing was generated.
        if num_nodes > 2:
            triplets.append([0, 1, 2])
    return np.array(triplets)




class RMDataset(Dataset):
    def __init__(self, data, lookback, walk_length=20):
        self.data = data
        self.lookback = lookback
        self.dataset, self.A_sequences = self.temp_process(data, lookback)
        self.transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')

        # Precompute neighbors for all snapshots in the sequence
        print("Precomputing neighbors for all snapshots...")
        self.precomputed_neighbors = {}
        for idx in range(len(self.dataset)):
            self.precomputed_neighbors[idx] = [A.tocsr().tolil().rows for A in self.A_sequences[idx]]

        # Precompute all triplets and weights
        print("Precomputing all triplets and weights...")
        self.precomputed_triplets = {}
        self.precomputed_weights = {}
        for idx in tqdm(range(len(self.dataset))):
            A_sequence = self.A_sequences[idx]
            time_idx = len(A_sequence) - 1
            neighbors = self.precomputed_neighbors[idx]
            self.precomputed_triplets[idx] = self.optimized_generate_triplets(
                A_sequence, neighbors, time_idx, self.lookback)
            self.precomputed_weights[idx] = self.calculate_recency_weights(
                self.precomputed_triplets[idx], A_sequence, time_idx, self.lookback)

    def calculate_recency_weights(self,triplets, A_sequence, time_idx, lookback, decay_factor=0.8):
        """
        Calculate weights for triplets based on recency of connections.
        (This part remains mostly similar.)
        """
        weights = []
        for anchor, pos, neg in triplets:
            connection_time = -1
            for t in range(time_idx, max(0, time_idx - lookback) - 1, -1):
                if t < len(A_sequence) and A_sequence[t][anchor, pos] > 0:
                    connection_time = t
                    break
            weight = (decay_factor ** (time_idx - connection_time)
                      if connection_time != -1 else 0.1)
            weights.append(weight)
        return np.array(weights)

    def temp_process(self, data, lookback):
        dataset = {}
        A_sequences = {}  # Will store lists of sparse matrices per snapshot
        print("Processing temporal data...")
        # Process snapshots from lookback to 90
        for i in tqdm(range(lookback, 90)):
            B = np.zeros((96, lookback + 1, 96))
            A_seq = []
            for j in range(lookback + 1):
                adj_matr = data[i - lookback + j][0]  # This is a scipy sparse matrix
                A_seq.append(adj_matr)
                B[:adj_matr.shape[0], j, :adj_matr.shape[1]] = adj_matr.todense()
            dataset[i - lookback] = B  # Re-index sequentially
            A_sequences[i - lookback] = A_seq
        return dataset, A_sequences


    def optimized_generate_triplets(self, A_sequence, precomputed_neighbors, time_idx, lookback, num_triplets=1000):
        """Optimized triplet generation using precomputed neighbors."""
        triplets = []
        num_nodes = A_sequence[0].shape[0]
        anchor_nodes = np.random.choice(num_nodes, size=min(num_nodes, num_triplets * 2), replace=False)

        for anchor in anchor_nodes:
            # Gather recent neighbors
            recent_neighbors = set()
            for t in range(max(0, time_idx - lookback), time_idx + 1):
                recent_neighbors.update(precomputed_neighbors[t][anchor])

            # Gather historical neighbors
            historical_neighbors = set()
            if time_idx > lookback:
                for t in range(0, time_idx - lookback):
                    historical_neighbors.update(precomputed_neighbors[t][anchor])
                historical_neighbors -= recent_neighbors

            # Identify negative samples
            non_neighbors = set(range(num_nodes)) - recent_neighbors - {anchor}

            # Sample nodes
            positive_samples = list(recent_neighbors)
            historical_samples = list(historical_neighbors)
            negative_samples = list(non_neighbors)

            # Use numpy's random choice for efficient sampling
            if positive_samples:
                positive_samples = np.random.choice(
                    positive_samples,
                    size=min(len(positive_samples), 5),
                    replace=False
                ).tolist()

            if historical_samples:
                historical_samples = np.random.choice(
                    historical_samples,
                    size=min(len(historical_samples), 5),
                    replace=False
                ).tolist()

            if negative_samples:
                negative_samples = np.random.choice(
                    negative_samples,
                    size=min(len(negative_samples), 5),
                    replace=False
                ).tolist()

            # Generate triplets
            for pos in positive_samples:
                for neg in negative_samples:
                    triplets.append([anchor, pos, neg])
                    if len(triplets) >= num_triplets:
                        return np.array(triplets)

            for pos in historical_samples:
                for neg in negative_samples:
                    triplets.append([anchor, pos, neg])
                    if len(triplets) >= num_triplets:
                        return np.array(triplets)

        return np.array(triplets) if triplets else np.array([[0, 1, 2]])

    def __getitem__(self, idx):
        data_tensor = self.dataset[idx]
        x = torch.tensor(data_tensor, dtype=torch.float32)

        # Create edge_index and edge_attr using the last snapshot
        adj_matrix = x[:, -1, :]
        edge_index, edge_attr = dense_to_sparse(adj_matrix)
        graph_data = Data(x=torch.ones(x.size(0), 1), edge_index=edge_index, edge_attr=edge_attr)
        graph_data = self.transform(graph_data)
        pe = graph_data.pe
        batch = torch.zeros(x.size(0), dtype=torch.long)

        # Use precomputed triplets and weights
        triplets = self.precomputed_triplets[idx]
        weights = self.precomputed_weights[idx]

        return x, pe, edge_index, edge_attr, batch, triplets, weights


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
        self.silu = nn.SiLU()
        self.config = MambaConfig(d_model=config['d_model'], n_layers=1,d_state=config['d_state'], d_conv=config['d_conv'])
        self.mamba = Mamba(self.config)
        #self.mamba = MambaBlock(seq_len=lookback + 1, d_model=config['d_model'], state_size=config['d_state'], batch_size=96, device=device)
        self.enc_input_fc = nn.Linear(config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)

    def forward(self, input):
        e = self.enc_input_fc(input)
        e = self.mamba(e)[0]
        e = e[:, -1, :]
        e = self.dropout(e)  # Apply dropout after average pooling
        x = self.out_fc(e)
        x = self.silu(x)
        x = self.dropout(x)  # Apply dropout after the activation
        mu = self.mu_fc(x)

        return x, mu, None


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
    print("Pretraining Node Embedder with Temporal Triplet Loss")
    model.train()

    for epoch in tqdm(range(epochs)):
        epoch_losses = []

        # Iterate over snapshots from "lookback" index up to 63
        for i in range(lookback, 63):
            try:
                # Get one snapshot sample
                x, _, _, _, _, triplets, weights = dataset[i]

                # Skip if no valid triplets
                if len(triplets) == 0:
                    continue

                # Transfer to device
                x = x.to(device)
                triplets_tensor = torch.tensor(triplets, dtype=torch.long).to(device)
                weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

                optimizer.zero_grad()

                # Forward pass through node embedder
                _, mu, _ = model(x)

                # Extract embeddings for triplets
                anchors = mu[triplets_tensor[:, 0]]
                positives = mu[triplets_tensor[:, 1]]
                negatives = mu[triplets_tensor[:, 2]]

                # Calculate weighted triplet loss
                distance_positive = torch.sum((anchors - positives) ** 2, dim=1)
                distance_negative = torch.sum((anchors - negatives) ** 2, dim=1)

                # Apply temporal weighting
                loss = weights_tensor * torch.clamp(distance_positive - distance_negative + 1.0, min=0.0)
                loss = torch.mean(loss)

                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
            except Exception as e:
                print(f"Error processing snapshot {i}: {e}")
                continue

        if epoch_losses:
            print(f"Epoch {epoch + 1}/{epochs}, Triplet Loss: {np.mean(epoch_losses):.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs}, No valid triplets found")

    # ===== Stage 2: Train Link Predictor =====
    # Instantiate the Link Predictor
    link_predictor = LinkPredictor(in_channels=64, hidden_channels=32, dropout=0.2).to(device)
    optimizer_stage2 = torch.optim.Adam(link_predictor.parameters(), lr=2e-4, weight_decay=1e-4)
    finetune_epochs = 50  # Adjust number of epochs

    link_predictor = train_link_predictor(dataset, node_model, link_predictor, optimizer_stage2, epochs=finetune_epochs)

    max_idx = 90 - lookback
    val_metrics = evaluate_model(dataset, node_model, link_predictor, (63, min(72, max_idx)), "Validation")
    test_metrics = evaluate_model(dataset, node_model, link_predictor, (72, min(90, max_idx)), "Test")
    return val_metrics, test_metrics


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
    pretrain_epochs = 60  # Number of epochs for stage 1 pretraining

    val_metrics, test_metrics = pretrain_node_embedder(dataset, node_model, optimizer_stage1, epochs=pretrain_epochs)


    # Print comparative results
    print("\n===== Final Evaluation Results =====")
    print(f"Validation MAP: {val_metrics['MAP']:.4f}, F1: {val_metrics['F1']:.4f}")
    print(f"Test MAP: {test_metrics['MAP']:.4f}, F1: {test_metrics['F1']:.4f}")
