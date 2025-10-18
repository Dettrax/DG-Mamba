import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg

from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import zeros
    
class LinkDecoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LinkDecoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2 * dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_out)
        )

    def forward(self, x, edge_label_index):
        x_i = x[edge_label_index[0]]
        x_j = x[edge_label_index[1]]
        prediction = self.mlp(torch.cat([x_i, x_j], dim=-1))
        return prediction

class Sampling(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Sampling, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp_edge_model = nn.Sequential(
          nn.Linear(self.input_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_feature=None):
        src, dst = edge_index[0], edge_index[1]
        emb_src = x[src]
        emb_dst = x[dst]
        
        if edge_feature is not None:
            edge_emb = torch.cat([emb_src, emb_dst, edge_feature], dim=-1)
        else:
            edge_emb = torch.cat([emb_src, emb_dst], dim=-1)
        edge_logits = self.mlp_edge_model(edge_emb)
        return edge_logits

class TimeEncode(torch.nn.Module):
    def __init__(self, dimension):
      super(TimeEncode, self).__init__()

      self.dimension = dimension
      self.w = torch.nn.Linear(1, dimension)

      self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                        .float().reshape(dimension, -1))
      self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

      self.w.weight.requires_grad = False
      self.w.bias.requires_grad = False

    def forward(self, t):
      t = t.float()
      t = t.unsqueeze(1)
      output = torch.cos(self.w(t))
      return output

class MPNN(MessagePassing):
    def __init__(self, node_dim):
        super().__init__(aggr='add', node_dim=node_dim)

    def forward(self, x, edge_index, norm=None, edge_feature=None, act=False):
        out = self.propagate(edge_index, x=x, norm=norm, edge_feature=edge_feature, act=act)
        return out

    def message(self, x_j, norm, edge_feature, act):
        if edge_feature is None:
            msg = x_j
        else:
            msg = x_j + edge_feature

        if act:
            msg = F.relu(msg)

        if norm is None:
            return msg
        else:
            return norm * msg

class MPNN2(MessagePassing):
    def __init__(self, node_dim):
        super().__init__(aggr='add', node_dim=node_dim)

    def forward(self, key, value, edge_index, norm=None, edge_feature=None):
        out = self.propagate(edge_index, key=key, value=value,
                             norm=norm, edge_feature=edge_feature)
        return out

    def message(self, key_j, value_j, norm, edge_feature):
        if edge_feature is None:
          key_j = F.relu(key_j)

        else:
          key_j = key_j + edge_feature
          key_j = F.relu(key_j)
          value_j = value_j + edge_feature

        M_j = torch.einsum('nhi,nhj->nhij', key_j, value_j)
        return norm * M_j