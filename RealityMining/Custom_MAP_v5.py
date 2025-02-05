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
import torch
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import networkx as nx


class NodeFeatureExtractor:
    def __init__(self, adj_matrix, max_nodes):
        """Initialize with an adjacency matrix"""
        self.max_nodes = max_nodes
        # Ensure adjacency matrix is padded to max_nodes
        if adj_matrix.shape[0] < max_nodes:
            padded_adj = np.zeros((max_nodes, max_nodes))
            padded_adj[:adj_matrix.shape[0], :adj_matrix.shape[0]] = adj_matrix.toarray()
            self.adj = sp.csr_matrix(padded_adj)
        else:
            self.adj = adj_matrix
        self.G = nx.from_scipy_sparse_array(self.adj)

    def get_structural_features(self):
        """Extract structural features for each node"""
        features = np.zeros((self.max_nodes, 4))

        # Degree features
        in_degrees = np.array(self.adj.sum(0)).flatten()
        out_degrees = np.array(self.adj.sum(1)).flatten()

        # Basic centrality (normalized degree)
        total_edges = max(1, self.adj.sum())
        centrality = (in_degrees + out_degrees) / total_edges

        # Activity level (self-loops removed)
        activity = np.array(self.adj.sum(1)).flatten() - np.diag(self.adj.toarray())

        features[:, 0] = in_degrees
        features[:, 1] = out_degrees
        features[:, 2] = centrality
        features[:, 3] = activity

        return features

    def get_temporal_features(self, previous_adjs, time_window=3):
        """Extract temporal features based on historical interactions"""
        features = np.zeros((self.max_nodes, 3))

        if not previous_adjs:  # If no historical data
            return features

        # Pad all previous adjacency matrices to max_nodes
        padded_prevs = []
        for prev_adj in previous_adjs:
            if prev_adj.shape[0] < self.max_nodes:
                padded = np.zeros((self.max_nodes, self.max_nodes))
                padded[:prev_adj.shape[0], :prev_adj.shape[0]] = prev_adj.toarray()
                padded_prevs.append(sp.csr_matrix(padded))
            else:
                padded_prevs.append(prev_adj)

        # Activity frequency
        activity = np.zeros(self.max_nodes)
        # Recent activity
        recent = np.zeros(self.max_nodes)

        for t, prev_adj in enumerate(padded_prevs):
            degree = np.array(prev_adj.sum(1)).flatten()
            if len(degree.shape) > 1:
                degree = degree.flatten()
            activity += degree

            # Weight recent interactions more
            time_weight = np.exp(-(time_window - t))
            recent += degree * time_weight

        # Stability (compare with most recent previous state)
        if padded_prevs:
            last_adj = padded_prevs[-1]
            stability = 1 - (np.array(np.abs(self.adj - last_adj).sum(1)).flatten() / self.max_nodes)
        else:
            stability = np.zeros(self.max_nodes)

        features[:, 0] = activity / max(1, len(padded_prevs))  # Average activity
        features[:, 1] = stability  # Stability score
        features[:, 2] = recent / (recent.max() if recent.max() > 0 else 1)  # Normalized recent activity

        return features


class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, time_window=3, predict_window=1):
        self.data = data
        self.time_window = time_window
        self.predict_window = predict_window
        self.num_timesteps = len(data)
        self.max_nodes = self._get_max_nodes()

    def _get_max_nodes(self):
        max_nodes = 0
        for t in range(self.num_timesteps):
            adj, _ = self.data[t]
            max_nodes = max(max_nodes, adj.shape[0])
        return max_nodes

    def _pad_adjacency(self, adj):
        if sp.issparse(adj):
            adj = adj.todense()
        adj = np.array(adj)

        current_size = adj.shape[0]
        if current_size < self.max_nodes:
            padded_adj = np.zeros((self.max_nodes, self.max_nodes))
            padded_adj[:current_size, :current_size] = adj
            return padded_adj
        return adj

    def _pad_features(self, features):
        current_size = features.shape[0]
        if current_size < self.max_nodes:
            padded_features = np.zeros((self.max_nodes, features.shape[1]))
            padded_features[:current_size, :] = features
            return torch.from_numpy(padded_features).float()
        return torch.from_numpy(features).float()

    def __len__(self):
        return self.num_timesteps - (self.time_window + self.predict_window) + 1

    def __getitem__(self, idx):
        history_graphs = []
        history_features = []

        for t in range(idx, idx + self.time_window):
            adj, _ = self.data[t]

            # Get previous adjacency matrices for temporal features
            prev_idx = max(0, t - self.time_window)
            previous_adjs = [self.data[i][0] for i in range(prev_idx, t)]

            # Create feature extractor
            feature_extractor = NodeFeatureExtractor(adj, self.max_nodes)

            # Get structural and temporal features
            structural_feats = feature_extractor.get_structural_features()
            temporal_feats = feature_extractor.get_temporal_features(previous_adjs)

            # Combine features
            enhanced_features = np.hstack([structural_feats, temporal_feats])

            # Pad adjacency matrix
            padded_adj = self._pad_adjacency(adj)

            history_graphs.append(torch.from_numpy(padded_adj).float())
            history_features.append(torch.from_numpy(enhanced_features).float())

        # Get target for prediction
        target_graphs = []
        for t in range(idx + self.time_window, idx + self.time_window + self.predict_window):
            target_adj, _ = self.data[t]
            padded_target = self._pad_adjacency(target_adj)
            target_graphs.append(torch.from_numpy(padded_target).float())

        # Stack temporal dimension
        history_tensor = torch.stack(history_graphs)
        target_tensor = torch.stack(target_graphs)
        feature_tensor = torch.stack(history_features)

        return {
            'history_graphs': history_tensor,
            'history_features': feature_tensor,
            'target': target_tensor
        }


def create_temporal_dataset(mit_dataset, time_window=3, predict_window=1):
    return TemporalGraphDataset(mit_dataset, time_window, predict_window)
# Create temporal dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from mamba_ssm import Mamba
from sklearn.metrics import average_precision_score


class NodeEmbedder(nn.Module):
    def __init__(self, num_nodes=96, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Graph structure processing
        self.graph_encoder = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Temporal attention using Mamba
        self.temporal_attention = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=3
        )

    def forward(self, history_graphs):
        batch_size = history_graphs.size(0)
        seq_len = history_graphs.size(1)
        num_nodes = history_graphs.size(2)

        # Process graph structure
        graph_embeddings = self.graph_encoder(history_graphs.view(-1, num_nodes))
        graph_embeddings = graph_embeddings.view(batch_size, seq_len, num_nodes, -1)

        # Reshape for temporal attention
        embeddings = graph_embeddings.permute(1, 0, 2, 3)
        embeddings = embeddings.reshape(seq_len, batch_size * num_nodes, self.hidden_dim)
        embeddings = embeddings.permute(1, 0, 2)

        # Apply temporal attention
        attended = self.temporal_attention(embeddings)

        # Get final embeddings
        node_embeddings = attended[:, -1, :].view(batch_size, num_nodes, self.hidden_dim)

        return node_embeddings


class LinkPredictor(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.edge_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings):
        batch_size, num_nodes, hidden_dim = node_embeddings.shape

        # Create all possible node pairs
        node_i = node_embeddings.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = node_embeddings.unsqueeze(1).expand(-1, num_nodes, -1, -1)

        # Concatenate node pairs
        edge_features = torch.cat([node_i, node_j], dim=-1)

        # Predict edge probabilities
        edge_probs = self.edge_predictor(edge_features.view(-1, hidden_dim * 2))
        edge_probs = edge_probs.view(batch_size, num_nodes, num_nodes)

        return edge_probs


def triplet_loss(anchor_embed, pos_embed, neg_embed, margin=1.0):
    pos_dist = torch.norm(anchor_embed - pos_embed, dim=-1)
    neg_dist = torch.norm(anchor_embed - neg_embed, dim=-1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()


import torch

def sample_triplets(adjacency, embeddings, num_samples=1000, k_hop=2):
    device = embeddings.device
    batch_size, num_nodes, _ = embeddings.shape

    triplets = []
    for b in range(batch_size):
        adj = adjacency[b]

        # Create k-hop adjacency matrix
        k_hop_adj = adj.clone().bool()
        curr_adj = adj.clone().bool()
        for _ in range(k_hop - 1):
            curr_adj = torch.matmul(curr_adj.float(), adj.float()).bool()
            k_hop_adj = k_hop_adj | curr_adj

        # For each node, find its k-hop neighbors and non-neighbors
        for i in range(num_nodes):
            # Get k-hop neighbors (excluding self)
            neighbors = torch.nonzero(k_hop_adj[i]).squeeze()
            neighbors = neighbors[neighbors != i]

            # Get non-neighbors
            non_neighbors = torch.nonzero(~k_hop_adj[i]).squeeze()

            if len(neighbors) == 0 or len(non_neighbors) == 0:
                continue

            # Sample positive and negative nodes
            pos_idx = torch.randint(0, len(neighbors), (num_samples // num_nodes,))
            neg_idx = torch.randint(0, len(non_neighbors), (num_samples // num_nodes,))

            pos_nodes = neighbors[pos_idx]
            neg_nodes = non_neighbors[neg_idx]

            # Create triplets
            anchor = embeddings[b, i].repeat(len(pos_idx), 1)
            positives = embeddings[b, pos_nodes]
            negatives = embeddings[b, neg_nodes]

            triplets.append((anchor, positives, negatives))

    # Combine triplets from all batches and nodes
    if not triplets:
        return None

    return [torch.cat(t) for t in zip(*triplets)]


def train_embedder(model, train_loader, epochs=50, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-3)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            history_graphs = batch['history_graphs'].to(device)
            target = batch['target'].squeeze(1).to(device)

            # Get node embeddings
            embeddings = model(history_graphs)

            # Sample triplets
            anchors, positives, negatives = sample_triplets(target, embeddings)

            # Calculate triplet loss
            loss = triplet_loss(anchors, positives, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

    return model


def train_link_predictor(embedder, link_predictor, train_loader, epochs=50, device='cuda'):
    optimizer = torch.optim.Adam(link_predictor.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    best_map = 0

    for epoch in range(epochs):
        link_predictor.train()
        total_loss = 0

        for batch in train_loader:
            history_graphs = batch['history_graphs'].to(device)
            target = batch['target'].squeeze(1).to(device)

            # Get embeddings from trained embedder
            with torch.no_grad():
                embeddings = embedder(history_graphs)

            # Predict links
            pred = link_predictor(embeddings)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate
        link_predictor.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in train_loader:
                history_graphs = batch['history_graphs'].to(device)
                target = batch['target'].squeeze(1).to(device)

                embeddings = embedder(history_graphs)
                pred = link_predictor(embeddings)

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
        all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
        map_score = average_precision_score(all_targets, all_preds)

        if map_score > best_map:
            best_map = map_score

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Average Loss: {total_loss / len(train_loader):.4f}')
            print(f'MAP Score: {map_score:.4f}')
            print(f'Best MAP Score: {best_map:.4f}')
            print('------------------------')

    return link_predictor, best_map


def main(temporal_data, device='cuda'):
    # Initialize models
    embedder = NodeEmbedder().to(device)
    link_predictor = LinkPredictor().to(device)

    # Create data loaders
    train_size = 63
    dataset_size = len(temporal_data)

    train_indices = list(range(train_size - temporal_data.time_window))
    test_indices = list(range(train_size - temporal_data.time_window, dataset_size - temporal_data.time_window))

    train_data = Subset(temporal_data, train_indices)
    test_data = Subset(temporal_data, test_indices)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Train embedder using triplet loss
    print("Training embedder...")
    #embedder = train_embedder(embedder, train_loader, epochs=100, device=device)

    # Train link predictor using trained embeddings
    print("\nTraining link predictor...")
    link_predictor, best_map = train_link_predictor(embedder, link_predictor, train_loader, epochs=50, device=device)

    return embedder, link_predictor, best_map


if __name__ == "__main__":
    # Assuming temporal_data is created using the TemporalGraphDataset class
    temporal_data = create_temporal_dataset(data, time_window=3, predict_window=1)
    embedder, link_predictor, best_map = main(temporal_data)
    print(f"Final Best MAP Score: {best_map:.4f}")

