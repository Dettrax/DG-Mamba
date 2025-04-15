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


import warnings
warnings.filterwarnings("ignore")

import itertools

from torch.nn import ELU,Dropout

from mamba_ssm import Mamba

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


from geoopt.manifolds import PoincareBall

c= 1.0/4
manifold = PoincareBall(c= c)

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



def hyperbolic_triplet_loss(anchor, positive, negative, margin=1.0):
    # manifold.dist computes the hyperbolic distance between two points
    d_pos = manifold.dist(anchor, positive)
    d_neg = manifold.dist(anchor, negative)
    loss = torch.clamp(d_pos - d_neg + margin, min=0.0)
    return torch.mean(loss)



import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from RealityMining.mamba import Mamba, MambaConfig


def exp_map_origin(x, k=-1.0):
    """
    Project from tangent space at origin to Poincaré ball with curvature k

    Args:
        x: Euclidean vectors (tangent vectors at origin)
        k: Curvature parameter (negative value)

    Returns:
        Points in the Poincaré ball
    """
    # Compute Euclidean norm
    norm = torch.norm(x, dim=-1, keepdim=True)
    # Avoid division by zero
    norm = torch.clamp(norm, min=1e-8)
    # Apply exponential map formula
    sqrt_k = torch.sqrt(torch.abs(k))
    exp_term = torch.tanh(sqrt_k * norm) / (sqrt_k * norm)
    return exp_term * x



class MambaG2G(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G, self).__init__()
        self.D = dim_in
        self.silu = nn.SiLU()
        self.config = MambaConfig(d_model=config['d_model'], n_layers=1,d_state=config['d_state'], d_conv=config['d_conv'])
        self.mamba = Mamba(self.config)
        #self.mamba = MambaBlock(seq_len=lookback + 1, d_model=config['d_model'], state_size=config['d_state'], batch_size=96, device=device)
        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension

        self.mu_fc = nn.Linear(self.D, dim_out)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)

    def forward(self, input):
        # e = self.enc_input_fc(input)
        e = self.mamba(input)[0]
        e = e.mean(dim=1)  # Average pooling to maintain the expected shape
        e = self.dropout(e)  # Apply dropout after average pooling
        x = self.out_fc(e)
        x = self.silu(x)
        x = self.dropout(x)  # Apply dropout after the activation
        mu = exp_map_origin(self.mu_fc(x)*self.scale, torch.tensor(- c))

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
class TangentSpaceLinkPredictor(nn.Module):
    def __init__(self, manifold, in_channels, hidden_channels, dropout=0.2):
        super(TangentSpaceLinkPredictor, self).__init__()
        self.manifold = manifold
        self.fc1 = nn.Linear(2 * in_channels+1, hidden_channels)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb_i, emb_j):
        # Calculate hyperbolic distance
        dist = self.manifold.dist(emb_i, emb_j)

        # Map embeddings to tangent space
        tan_i = self.manifold.logmap0(emb_i)
        tan_j = self.manifold.logmap0(emb_j)

        # Concatenate embeddings and distance
        x = torch.cat([tan_i, tan_j, dist.unsqueeze(1)], dim=-1)

        # Apply MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.silu(x)
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
            loss = hyperbolic_triplet_loss(anchors, positives, negatives, margin=1.0)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Triplet Loss: {np.mean(epoch_losses):.4f}")
    return model

def train_tangent_space_link_predictor(dataset, node_model, link_predictor, optimizer, epochs=20):
    print("Stage 2: Training Tangent Space Link Predictor (with frozen node embedder)")
    # Freeze node embedder parameters
    for param in node_model.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        epoch_losses = []
        link_predictor.train()
        node_model.eval()
        for i in range(lookback, 63):
            x, _, _, _, _, triplet, _ = dataset[i]
            optimizer.zero_grad()
            x = x.clone().detach().to(device)
            with torch.no_grad():
                # Get node embeddings
                _, mu, _ = node_model(x)
            # Process triplets for link prediction
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
        print(f"Epoch {epoch + 1}/{epochs}, Tangent Space Link Predictor BCE Loss: {np.mean(epoch_losses):.4f}")
    return link_predictor


# ===== Stage 3: Evaluate the trained models =====
def evaluate_tangent_space_model(dataset, node_model, link_predictor, snapshot_range, name="Validation"):
    """Evaluate the trained tangent space models on a specific range of snapshots."""
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

            # Get predicted probabilities using tangent space MLP
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


if __name__ == '__main__':
    # Load configuration and dataset
    with open("config.json") as f:
        config = json.load(f)
    lookback = 4
    walk = 16

    # Initialize dataset
    data = dataset_mit('..')
    dataset = RMDataset(data, lookback, walk_length=walk)

    # Initialize hyperbolic manifold
    from geoopt.manifolds import PoincareBall

    c = 1 / 4  # Curvature parameter can be tuned
    manifold = PoincareBall(c=c)

    # Model configuration
    model_config = {
        'd_model': 96,
        'd_state': 16,
        'd_conv': 3
    }

    # Initialize hyperbolic node embedder
    node_model = MambaG2G(model_config, dim_in=64, dim_out=64, dropout=0.4285).to(device)
    optimizer_stage1 = torch.optim.Adam(node_model.parameters(), lr=1e-4, weight_decay=1e-5)
    pretrain_epochs = 50

    # Pretrain node embedder
    node_model = pretrain_node_embedder(dataset, node_model, optimizer_stage1, epochs=pretrain_epochs)

    # Initialize tangent space link predictor
    link_predictor = TangentSpaceLinkPredictor(manifold, in_channels=64, hidden_channels=32, dropout=0.2).to(device)
    optimizer_stage2 = torch.optim.Adam(link_predictor.parameters(), lr=1e-4, weight_decay=1e-4)
    finetune_epochs = 50

    # Train link predictor
    link_predictor = train_tangent_space_link_predictor(dataset, node_model, link_predictor, optimizer_stage2,
                                                        epochs=finetune_epochs)

    # Evaluate model
    val_metrics = evaluate_tangent_space_model(dataset, node_model, link_predictor, (63, 72), "Validation")
    test_metrics = evaluate_tangent_space_model(dataset, node_model, link_predictor, (72, 90), "Test")

    # Print results
    print("\n===== Final Evaluation Results =====")
    print(f"Validation MAP: {val_metrics['MAP']:.4f}, F1: {val_metrics['F1']:.4f}")
    print(f"Test MAP: {test_metrics['MAP']:.4f}, F1: {test_metrics['F1']:.4f}")

