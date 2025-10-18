import random
import torch
import numpy as np

from torch_geometric.utils import structured_negative_sampling

def negative_sampling(edge_index):
    src, dst, neg_dst = structured_negative_sampling(edge_index)
    negative_edge_index = torch.stack((src, neg_dst), dim=0)
    return negative_edge_index

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.use_deterministic_algorithms(True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)