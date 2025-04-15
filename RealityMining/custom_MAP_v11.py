# This code creates and saves embedding with - transformer + G2G model.
# We have used some of the functionalities from Xu, M., Singh, A.V. &
# Karniadakis G.K. "DynG2G: An efficient Stochastic Graph Embedding
# Method for Temporal Graphs".


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


import warnings
warnings.filterwarnings("ignore")

from mamba_ssm import Mamba
from tqdm import tqdm


from torch.nn.utils import clip_grad_norm_

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


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

        # Convert to numpy array with correct dtype and C-contiguous memory layout
        adj = np.ascontiguousarray(np.asarray(adj, dtype=np.float32))

        current_size = adj.shape[0]
        if current_size < self.max_nodes:
            padded_adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
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
            enhanced_features = np.ascontiguousarray(enhanced_features, dtype=np.float32)
            # Pad adjacency matrix
            padded_adj = self._pad_adjacency(adj)

            history_graphs.append(torch.FloatTensor(padded_adj))
            history_features.append(torch.FloatTensor(enhanced_features))

        # Get target for prediction
        target_graphs = []
        for t in range(idx + self.time_window, idx + self.time_window + self.predict_window):
            target_adj, _ = self.data[t]
            padded_target = self._pad_adjacency(target_adj)
            target_graphs.append(torch.FloatTensor(padded_target))

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

from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops


import geoopt
manifold = geoopt.manifolds.PoincareBall(c=1/100)

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout=0.2, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x)
        res = self.manifold.projx(mv)
        if self.use_bias:
            bias = self.bias.view(1, -1)
            hyp_bias = self.manifold.expmap0(bias)
            hyp_bias = self.manifold.projx(hyp_bias)
            res = self.manifold.mobius_add(res, hyp_bias)
            res = self.manifold.projx(res)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x))
        return self.manifold.projx(self.manifold.expmap0(xt))

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HypDropout(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, dropout=0.2):
        super(HypDropout, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout

    def forward(self, x):
        xt = F.dropout(self.manifold.logmap0(x), p=self.dropout, training=self.training)
        return self.manifold.projx(self.manifold.expmap0(xt))

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAttAgg(MessagePassing):
    def __init__(self, manifold, c, out_features, att_dropout=0.2, heads=4, concat=False):
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.dropout = att_dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.c = c
        self.concat = concat
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        glorot(self.att_i)
        glorot(self.att_j)

    def initHyperX(self, x, c=1/100):
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1/100):
        x_hyp = self.manifold.expmap0(x)
        x_hyp = self.manifold.projx(x_hyp)
        return x_hyp


    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_tangent0 = self.manifold.logmap0(x)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0)

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)
        support_t = self.manifold.projx(self.manifold.expmap0(support_t))

        return support_t


class NodeEmbedder(nn.Module):
    def __init__(self, num_nodes=96, hidden_dim=64, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Graph structure processing
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.c = torch.tensor(1/100)

        # Multihead attention for temporal modeling
        self.temporal_attention = HypAttAgg(
            manifold=manifold,
            c=self.c ,
            out_features=hidden_dim,
            att_dropout=dropout,
            heads=num_heads,
            concat=False
        )

        # Layer norm for attention
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.post_process = nn.Sequential(
            HypLinear(manifold=manifold, in_features=hidden_dim//4, out_features=hidden_dim, c=self.c),
            HypAct(manifold=manifold, c_in=self.c, c_out=self.c, act=nn.SiLU()),
            HypDropout(manifold=manifold, c_in=self.c, c_out=self.c, dropout=dropout),
            HypLinear(manifold=manifold, in_features=hidden_dim, out_features=hidden_dim, c=self.c)
        )

    def forward(self, history_graphs,edge_index):
        batch_size = history_graphs.size(0)
        seq_len = history_graphs.size(1)
        num_nodes = history_graphs.size(2)

        # Create a tensor of node indices: [0, 1, 2, ..., num_nodes-1]
        node_ids = torch.arange(num_nodes, device=history_graphs.device)
        # Expand dimensions to match (batch_size, seq_len, num_nodes)
        node_ids = node_ids.unsqueeze(0).unsqueeze(0)
        node_ids = node_ids.expand(batch_size, seq_len, num_nodes)

        # Lookup node embeddings (each of shape hidden_dim)
        # Output shape: (batch_size, seq_len, num_nodes, hidden_dim)
        x = self.temporal_attention.initHyperX(self.embedding(node_ids))
        x = x.view(batch_size * seq_len, num_nodes, self.hidden_dim)

        out_features = []
        # Process each graph with the hyper attention module, using the corresponding edge_index.
        # Here we assume edge_index is provided as a list (or similar indexed structure)
        for i in range(batch_size * seq_len):
            # Retrieve the edge_index for the i-th graph and ensure it is on the correct device
            ei = edge_index[i].to(x.device) if isinstance(edge_index, list) else edge_index[i]
            # Apply hyper attention message passing: expected input shape (num_nodes, hidden_dim)
            h = self.temporal_attention(x[i], ei)
            out_features.append(h)

        # Stack results and reshape to (batch_size, seq_len, num_nodes, hidden_dim)
        h = torch.stack(out_features, dim=0)

        h = h.view(batch_size, seq_len, num_nodes, self.hidden_dim//4)

        node_embeddings = h[:, -1, :].view(batch_size, num_nodes, self.hidden_dim//4)
        # Post-process embeddings
        node_embeddings = self.post_process(node_embeddings)

        return node_embeddings


class LinkPredictor(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.edge_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 3, embedding_dim),  # Accept 2*embedding_dim + 3 features
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings):
        node_embeddings = manifold.logmap0(node_embeddings)
        batch_size, num_nodes, hidden_dim = node_embeddings.shape

        # Create all possible node pairs
        node_i = node_embeddings.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = node_embeddings.unsqueeze(1).expand(-1, num_nodes, -1, -1)

        # Reshape for batch processing
        node_i_flat = node_i.reshape(batch_size * num_nodes * num_nodes, -1)
        node_j_flat = node_j.reshape(batch_size * num_nodes * num_nodes, -1)

        # Calculate hyperbolic distances
        pair_dists = manifold.dist(node_i_flat, node_j_flat)
        dist0_i = manifold.dist0(node_i_flat)
        dist0_j = manifold.dist0(node_j_flat)

        # Stack distance features
        distance_features = torch.stack([pair_dists, dist0_i, dist0_j], dim=1)

        # Concatenate node embeddings with distance features
        combined_features = torch.cat([node_i_flat, node_j_flat, distance_features], dim=1)

        # Predict edge probabilities
        edge_probs = self.edge_predictor(combined_features)
        edge_probs = edge_probs.view(batch_size, num_nodes, num_nodes)

        return edge_probs



def triplet_loss(anchor_embed, pos_embed, neg_embed, margin=1.0):
    pos_dist = manifold.dist(anchor_embed, pos_embed)
    neg_dist = manifold.dist(anchor_embed, neg_embed)
    cluster_triplet_loss = F.relu(pos_dist - neg_dist + margin)
    return cluster_triplet_loss.mean()


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


def create_edge_index_from_tensor(adj):
    """
    Given an adjacency matrix tensor `adj` of shape (num_nodes, num_nodes),
    this function returns the edge_index tensor of shape [2, num_edges].

    Assumes that a positive value in `adj` indicates an edge.
    """
    # Create a boolean mask where there is an edge (adjust threshold as needed)
    mask = adj > 0
    # Get the indices of the nonzero elements; result shape is (num_edges, 2)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index



def create_edge_index_list(history_graphs):
    """
    Given a history_graphs tensor of shape (batch_size, seq_len, num_nodes, num_nodes),
    returns a list of edge_index tensors, one for each graph in the batch and temporal sequence.
    """
    batch_size, seq_len, num_nodes, _ = history_graphs.size()
    edge_index_list = []
    for b in range(batch_size):
        for t in range(seq_len):
            adj = history_graphs[b, t]
            edge_index = create_edge_index_from_tensor(adj)
            edge_index_list.append(edge_index)
    return edge_index_list


def train_embedder(model, train_loader, epochs=50, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005,weight_decay=1e-4)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            history_graphs = batch['history_graphs'].to(device)
            target = batch['target'].squeeze(1).to(device)
            edge_index_list = create_edge_index_list(history_graphs)


            # Get node embeddings
            embeddings = model(history_graphs,edge_index_list)

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


def train_link_predictor(embedder, link_predictor, train_loader,test_loader, epochs=50, device='cuda'):
    optimizer = torch.optim.Adam(link_predictor.parameters(), lr=0.005,weight_decay=1e-4)
    criterion = nn.BCELoss()
    best_map = 0
    temp_auc  = 0

    for epoch in range(epochs):
        link_predictor.train()
        total_loss = 0

        for batch in train_loader:
            history_graphs = batch['history_graphs'].to(device)
            target = batch['target'].squeeze(1).to(device)
            edge_index_list = create_edge_index_list(history_graphs)

            # Get embeddings from trained embedder
            with torch.no_grad():
                embeddings = embedder(history_graphs,edge_index_list)

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
            for batch in test_loader:
                history_graphs = batch['history_graphs'].to(device)
                target = batch['target'].squeeze(1).to(device)
                edge_index_list = create_edge_index_list(history_graphs)

                embeddings = embedder(history_graphs,edge_index_list)
                pred = link_predictor(embeddings)

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
        all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
        map_score = average_precision_score(all_targets, all_preds)
        auc_score = roc_auc_score(all_targets, all_preds)
        if map_score > best_map:
            best_map = map_score
            temp_auc = auc_score


        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Average Loss: {total_loss / len(train_loader):.4f}')
            print(f'MAP Score: {map_score:.4f}')
            print(f'Best MAP Score: {best_map:.4f}')
            print(f'AUC Score: {temp_auc:.4f}')
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

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128)

    # Train embedder using triplet loss
    print("Training embedder...")
    embedder = train_embedder(embedder, train_loader, epochs=50, device=device)


    # Train link predictor using trained embeddings
    print("\nTraining link predictor...")
    link_predictor, best_map = train_link_predictor(embedder, link_predictor, train_loader, test_loader,epochs=50, device=device)

    return embedder, link_predictor, best_map

if __name__ == "__main__":
    # Assuming temporal_data is created using the TemporalGraphDataset class
    temporal_data = create_temporal_dataset(data, time_window=3, predict_window=1)
    embedder, link_predictor, best_map = main(temporal_data)
    print(f"Final Best MAP Score: {best_map:.4f}")

