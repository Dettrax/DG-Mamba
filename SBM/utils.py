import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import itertools
import math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import normalize
import torch
import os
import time
import logging
import pandas as pd


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix

    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def sparse_feeder(M):
    """
    Prepares the input matrix into a format that is easy to feed into tensorflow's SparseTensor

    Parameters
    ----------
    M : scipy.sparse.spmatrix
        Matrix to be fed

    Returns
    -------
    indices : array-like, shape [n_edges, 2]
        Indices of the sparse elements
    values : array-like, shape [n_edges]
        Values of the sparse elements
    shape : array-like
        Shape of the matrix
    """
    M = sp.coo_matrix(M)
    return np.vstack((M.row, M.col)).T, M.data, M.shape


def score_link_prediction(labels, scores):
    """
    Calculates the area under the ROC curve and the average precision score.

    Parameters
    ----------
    labels : array-like, shape [N]
        The ground truth labels
    scores : array-like, shape [N]
        The (unnormalized) scores of how likely are the instances

    Returns
    -------
    roc_auc : float
        Area under the ROC curve score
    ap : float
        Average precision score
    """

    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def score_node_classification(features, z, p_labeled=0.1, n_repeat=10, norm=False):
    """
    Train a classifier using the node embeddings as features and reports the performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    z : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    n_repeat : int
        Number of times to repeat the experiment
    norm

    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    lrcv = LogisticRegressionCV(max_iter = 2000)

    if norm:
        features = normalize(features)

    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])

        f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')

        trace.append((f1_micro, f1_macro))

    return np.array(trace).mean(0)


def get_hops(A, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hopness to consider.

    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    """
    hops = {1: A.tolil(), -1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop
        hops[-1] += next_hop

    return hops


def sample_last_hop(A, nodes):
    """
    For each node in nodes samples a single node from their last (K-th) neighborhood.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix encoding which nodes belong to any of the 1, 2, ..., K-1, neighborhoods of every node
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N]
        The sampled nodes.
    """
    N = A.shape[0]

    sampled = np.random.randint(0, N, len(nodes))

    nnz = A[nodes, sampled].nonzero()[1]
    while len(nnz) != 0:
        new_sample = np.random.randint(0, N, len(nnz))
        sampled[nnz] = new_sample
        nnz = A[nnz, new_sample].nonzero()[1]

    return sampled


def sample_all_hops(hops, nodes=None):
    """
    For each node in nodes samples a single node from all of their neighborhoods.

    Parameters
    ----------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N, K]
        The sampled nodes.
    """

    N = hops[1].shape[0]

    if nodes is None:
        nodes = np.arange(N)

    return np.vstack((nodes,
                      np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
                                for h in hops.keys() if h != -1]),
                      sample_last_hop(hops[-1], nodes)
                      )).T


def to_triplets(sampled_hops, scale_terms):
    """
    Form all valid triplets (pairwise constraints) from a set of sampled nodes in triplets

    Parameters
    ----------
    sampled_hops : array-like, shape [N, K]
       The sampled nodes.
    scale_terms : dict
        The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood

    Returns
    -------
    triplets : array-like, shape [?, 3]
       The transformed triplets.
    """
    triplets = []
    triplet_scale_terms = []

    for i, j in itertools.combinations(np.arange(1, sampled_hops.shape[1]), 2):
        triplet = sampled_hops[:, [0] + [i, j]]
        triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
        triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
        triplets.append(triplet)

        triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])

    return np.row_stack(triplets), np.concatenate(triplet_scale_terms)


def sparse_feeder(M):
    
    M = sp.coo_matrix(M)
    return M


def spy_sparse2torch_sparse(data):
    """

    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples=data.shape[0]
    features=data.shape[1]
    values=data.data
    coo_data=data.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return t


def init_logging_handler(exp_name):
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(exp_name, current_time))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def check_if_gpu():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device


def sample_zero_forever(mat):
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    while True:
        t = tuple(np.random.randint(0, mat.shape[0], 2))
        if t not in nonzero_or_sampled:
            yield t
            nonzero_or_sampled.add(t)


def sample_zero_n(mat, n=1000):
    itr = sample_zero_forever(mat)
    return [next(itr) for _ in range(n)]


def get_row_MRR(probs,true_classes):
    existing_mask = true_classes == 1
        #descending in probability
    ordered_indices = np.flip(probs.argsort())

    ordered_existing_mask = existing_mask[ordered_indices]

    existing_ranks = np.arange(1,
                                   true_classes.shape[0]+1,
                                   dtype=np.float64)[ordered_existing_mask]

    MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
    return MRR


def get_MRR(predictions,true_classes, adj):
    probs = predictions

    probs = probs.detach().cpu().numpy()
    true_classes = true_classes
    adj = adj


    pred_matrix = sp.coo_matrix((probs,(adj[0],adj[1]))).toarray()
    true_matrix = sp.coo_matrix((true_classes,(adj[0],adj[1]))).toarray()

    row_MRRs = []
    for i, pred_row in enumerate(pred_matrix):
            #check if there are any existing edges
        if np.isin(1,true_matrix[i]):
            row_MRRs.append(get_row_MRR(pred_row,true_matrix[i]))

    avg_MRR = torch.tensor(row_MRRs).mean()
    return avg_MRR


def get_MAP_e(predictions,true_classes, adj):

    probs = predictions
    probs = probs.detach().cpu().numpy()
    true_classes = true_classes
    adj = adj

    var = average_precision_score(true_classes, probs)

    return var


def get_data():
    Adj_arr = []
    X_Sparse_arr = []
    count = 0
    max_size = 0

    df = pd.read_csv('sbm_50t_1000n_adj.csv')
    for i in range(50):
        # print(count)
        df1 = df[df['time']== i]
        arr = df1[['source', 'target']].to_numpy()
        
        A, X_Sparse, size = get_graph(arr, max_size)
        if size > max_size:
            max_size = size
            
        Adj_arr.append(A)
        X_Sparse_arr.append(X_Sparse)
        count = count + 1


    inpt = []
    for i in range(50):
        inpt.append((Adj_arr[i],X_Sparse_arr[i]))
    return inpt


def get_graph(arr, max_size):

        new_a = arr
        if max_size > int(new_a[-1][0])+1:
            new_max = max_size
            arr_zero = np.zeros((max_size, max_size))
        else:
            arr_zero = np.zeros((int(new_a[-1][0])+1, int(new_a[-1][0])+1))
            new_max =  int(new_a[-1][0])+1
        for i in new_a:
            arr_zero[int(i[0])][int(i[1])] = 1 
        arr_zero[range(len(arr_zero)), range(len(arr_zero))] = 0
        A = csr_matrix(arr_zero)
        X= A + sp.eye(A.shape[0])
        X_Sparse = sparse_feeder(X)
        X_Sparse = spy_sparse2torch_sparse(X_Sparse)
        
        return A, X_Sparse, new_max
