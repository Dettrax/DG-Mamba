import time
import torch
import copy
import numpy as np
from tqdm import tqdm

from utils.loss import compute_loss, compute_roc_auc, compute_ap
from utils.loss import gaussian_pair_distance, triplet_margin_loss, kl_prior_loss
from torch_geometric.utils import structured_negative_sampling


def train_step(model, optimizer, dataset, device, args):
    start, end = dataset.get_range_by_split('train')
    train_loss = 0
    count = 0
    train_ap = []
    train_auc = []

    model.train()
    model.reset_memory()

    init_state = torch.zeros(dataset.snapshots[0].node_feature.shape[0], model.hidden_dim)

    for t in tqdm(range(start, end - 1), leave=False):
        current_snapshot = dataset.snapshots[t]
        next_snapshot = dataset.snapshots[t + 1]

        x = current_snapshot.node_feature.to(device)
        init_state = init_state.to(device)
        edge_index = current_snapshot.edge_index.to(device)
        edge_feature = current_snapshot.edge_time.to(device) if hasattr(current_snapshot, 'edge_time') and current_snapshot.edge_time is not None else torch.zeros(edge_index.size(1), device=device)

        # Aligned structured negatives for triplet construction
        src, pos_dst, neg_dst = structured_negative_sampling(next_snapshot.edge_index)
        negative_edge_index = torch.stack((src, neg_dst), dim=0).to(device)

        pos_edge_index = next_snapshot.edge_index.to(device)
        edge_label_index = torch.cat([pos_edge_index, negative_edge_index], dim=-1)
        label = torch.cat([torch.ones(pos_edge_index.shape[1]),
                           torch.zeros(negative_edge_index.shape[1])]).to(device)

        prediction, new_state = model(x, edge_index, edge_label_index, edge_feature, init_state)

        # BCE loss on logits
        loss = compute_loss(prediction, label)

        # Optional Gaussian metric learning losses
        if getattr(model, 'use_gaussian', False):
            if args.triplet_weight > 0:
                # distances for pos/neg with current Gaussian params
                mu = model.last_mu
                logvar = model.last_logvar
                d_pos = gaussian_pair_distance(mu, logvar, pos_edge_index, mode=args.gauss_score)
                d_neg = gaussian_pair_distance(mu, logvar, negative_edge_index, mode=args.gauss_score)
                loss_triplet = triplet_margin_loss(d_pos, d_neg, margin=args.triplet_margin)
                loss = loss + args.triplet_weight * loss_triplet
            if args.prior_kl_weight > 0:
                loss = loss + args.prior_kl_weight * kl_prior_loss(model.last_mu, model.last_logvar)

        init_state = new_state.detach().cpu().clone()

        roc_auc = compute_roc_auc(prediction, label)
        ap = compute_ap(prediction, label)

        train_ap.append(ap)
        train_auc.append(roc_auc)

        train_loss += loss
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / count

    train_metric = {
        'loss': train_loss,
        'ap': np.mean(train_ap),
        'auc': np.mean(train_auc),
    }
    return train_metric, init_state, model.memory_edge_index, model.memory_edge_time


def evaluate_step(model, dataset, previous_state, previous_memory_edge_index, previous_memory_edge_time, spilt, device):
    start, end = dataset.get_range_by_split(spilt)
    test_loss = 0
    count = 0
    test_ap = []
    test_auc = []

    init_state = previous_state.clone()

    model.read_memory(copy.deepcopy(previous_memory_edge_index), copy.deepcopy(previous_memory_edge_time))

    model.eval()
    with torch.no_grad():
        for t in tqdm(range(start - 1, end - 1), leave=False):
            current_snapshot = dataset.snapshots[t]
            next_snapshot = dataset.snapshots[t + 1]

            x = current_snapshot.node_feature.to(device)
            init_state = init_state.to(device)
            edge_index = current_snapshot.edge_index.to(device)
            edge_feature = current_snapshot.edge_time.to(device) if hasattr(current_snapshot, 'edge_time') and current_snapshot.edge_time is not None else torch.zeros(edge_index.size(1), device=device)

            edge_label_index = next_snapshot.edge_label_index.to(device)
            label = next_snapshot.edge_label.to(device)

            prediction, new_state = model(x, edge_index, edge_label_index, edge_feature, init_state)

            loss = compute_loss(prediction, label)

            init_state = new_state.detach().cpu().clone()

            roc_auc = compute_roc_auc(prediction, label)

            ap = compute_ap(prediction, label)
            test_loss += loss
            count += 1

            test_ap.append(ap)
            test_auc.append(roc_auc)

    test_loss = test_loss / count

    test_metric = {
        'loss': test_loss,
        'ap': np.mean(test_ap),
        'auc': np.mean(test_auc),
    }

    if spilt == 'val':
        return test_metric, init_state, model.memory_edge_index, model.memory_edge_time
    else:
        return test_metric


def train(model, optimizer, dataset, n_epoch, patience, device, args=None):
    if args is None:
        class _A: pass
        args = _A()
        args.triplet_weight = 0.0
        args.triplet_margin = 1.0
        args.gauss_score = 'kl'
        args.prior_kl_weight = 0.0
        args.early_metric = 'auc'

    best_model = None
    best_model_unchanged = 0
    best_epoch = 0
    best_state = None
    best_val = -float('inf')
    best_mem_edge_index = None
    best_mem_edge_time = None

    # initialize to current (empty) memory; will be updated after first val pass
    val_memory_edge_index = model.memory_edge_index
    val_memory_edge_time = model.memory_edge_time

    train_auc = []
    valid_auc = []
    for epoch in range(n_epoch):

        start_time = time.time()

        train_metric, train_state, train_memory_edge_index, train_memory_edge_time = train_step(model, optimizer, dataset, device, args)

        val_metric, val_state, val_memory_edge_index, val_memory_edge_time = evaluate_step(model, dataset, train_state, train_memory_edge_index,
                                                                                           train_memory_edge_time, 'val', device)

        test_metric = evaluate_step(model, dataset, val_state, val_memory_edge_index, val_memory_edge_time, 'test', device)

        epoch_time = time.time() - start_time

        print('Epoch {}, Time: {}s'.format(epoch + 1, epoch_time))

        print('Train: loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(train_metric['loss'], train_metric['auc'], train_metric['ap']))

        print('Valid: loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(val_metric['loss'], val_metric['auc'], val_metric['ap']))

        print('Test : loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(test_metric['loss'], test_metric['auc'], test_metric['ap']))

        print('======' * 20)

        train_auc.append(train_metric['auc'])
        valid_auc.append(val_metric['auc'])

        # Early stopping based on configurable metric (auc or ap)
        val_score = val_metric['auc'] if getattr(args, 'early_metric', 'auc') == 'auc' else val_metric['ap']
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_state = val_state
            best_mem_edge_index = copy.deepcopy(val_memory_edge_index)
            best_mem_edge_time = copy.deepcopy(val_memory_edge_time)
            best_model_unchanged = 0
        else:
            best_model_unchanged += 1

    # Ensure we have validation memory to evaluate the final model
    if best_mem_edge_index is None or best_mem_edge_time is None:
        best_mem_edge_index = model.memory_edge_index
        best_mem_edge_time = model.memory_edge_time

    model.load_state_dict(best_model)

    final_metric = evaluate_step(model, dataset, best_state, best_mem_edge_index, best_mem_edge_time, 'test', device)

    print('Final Test Results: roc_auc: {:.4f}, ap: {:.4f}'.format(final_metric['auc'], final_metric['ap']))

    return final_metric['auc'], final_metric['ap']
