import time
import torch
import copy
import numpy as np
from utils.loss import compute_loss, compute_roc_auc, compute_ap
from dataloader.utils import negative_sampling

def evaluate_step_baseline(model, args, dataset, spilt, device, init_H, C, embeddings):
    start, end = dataset.get_range_by_split(spilt)
    test_ap = []
    test_auc = []
    history = [embeddings]

    H = init_H

    model.eval()
    with torch.no_grad():
        for t in range(start - 1, end - 1):
            current_snapshot = dataset.snapshots[t]
            next_snapshot = dataset.snapshots[t + 1]

            x = current_snapshot.node_feature.to(device)
            edge_index = current_snapshot.edge_index.to(device)

            edge_label_index = next_snapshot.edge_label_index.to(device)
            label = next_snapshot.edge_label.to(device)

            H = H.to(device)

            if args.model_name == 'EvolveGCN':
                prediction = model(x, edge_index, edge_label_index)
            
            elif args.model_name == 'TGCN':
                prediction, new_H = model(x, edge_index, edge_label_index, H)
                H = new_H.detach().cpu().clone()
            
            elif args.model_name == 'DySAT':
                prediction, temporal_input = model(x, edge_index, edge_label_index, history)
            else:
                prediction, new_H = model(x, edge_index, edge_label_index, H, C)
                H = new_H.detach().cpu().clone()

            if args.model_name == 'DySAT':
                test_loss = compute_loss(prediction[:, t], label)
                roc_auc = compute_roc_auc(prediction[:, t], label)
                ap = compute_ap(prediction[:, t], label)
            else:    
                test_loss = compute_loss(prediction, label)
                roc_auc = compute_roc_auc(prediction, label)
                ap = compute_ap(prediction, label)

            test_ap.append(ap)
            test_auc.append(roc_auc)

    test_metric = {
        'loss': test_loss,
        'ap': np.mean(test_ap),
        'auc': np.mean(test_auc),
    }
    if args.model_name == 'DySAT':
        return test_metric, H, C, temporal_input.detach()
    else: 
        return test_metric, H, C, None

def train_step_baseline(model, args, optimizer, dataset, device):
    start, end = dataset.get_range_by_split('train')
    train_ap = []
    train_auc = []
    count = 0
    train_loss = 0

    model.train()

    H = torch.zeros(dataset.snapshots[0].node_feature.shape[0], model.hidden_dim)
    C = None
    temporal_input = None
    history = []

    for t in range(start, end - 1):
        current_snapshot = dataset.snapshots[t]
        next_snapshot = dataset.snapshots[t + 1]

        x = current_snapshot.node_feature.to(device)
        edge_index = current_snapshot.edge_index.to(device)

        negative_edge_index = negative_sampling(next_snapshot.edge_index)
        edge_label_index = torch.cat([next_snapshot.edge_index, negative_edge_index], dim=-1).to(device)

        label = torch.cat([torch.ones(next_snapshot.edge_index.shape[1]),
                          torch.zeros(negative_edge_index.shape[1])]).to(device)
        H = H.to(device)

        if args.model_name == 'EvolveGCN':
            prediction = model(x, edge_index, edge_label_index)
            
        elif args.model_name == 'TGCN':
            prediction, new_H = model(x, edge_index, edge_label_index, H)
            H = new_H.detach().cpu().clone()

        elif args.model_name == 'DySAT':
            prediction, temporal_input = model(x, edge_index, edge_label_index, history)
        else:
            prediction, new_H = model(x, edge_index, edge_label_index, H, C)
            H = new_H.detach().cpu().clone()

        if args.model_name == 'DySAT':
            loss = compute_loss(prediction[:, t], label)
            roc_auc = compute_roc_auc(prediction[:, t], label)
            ap = compute_ap(prediction[:, t], label)
        else:    
            loss = compute_loss(prediction, label)
            roc_auc = compute_roc_auc(prediction, label)
            ap = compute_ap(prediction, label)

        train_ap.append(ap)
        train_auc.append(roc_auc)
        train_loss += loss
        count += 1

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    train_metric = {
        'loss': train_loss,
        'ap': np.mean(train_ap),
        'auc': np.mean(train_auc),
    }

    if args.model_name == 'DySAT':
        return train_metric, H, C, temporal_input.detach()
    else: 
        return train_metric, H, C, None

def train_baseline(model, args, optimizer, dataset, n_epoch, device):
    best_model = None
    best_model_unchanged = 0
    best_auc = 0
    best_epoch = 0
    best_H = None
    best_C = None
    best_history = None

    for epoch in range(n_epoch):
        start_time = time.time()

        train_metric, H_train, C_train, history_train = train_step_baseline(model, args, optimizer, dataset, device)
        val_metric, H_val, C_val, history_val = evaluate_step_baseline(model, args, dataset, 'val', device, H_train, C_train, history_train)
        test_metric, _, _, _ = evaluate_step_baseline(model, args, dataset, 'test', device, H_val, C_val, history_val)

        epoch_time = time.time() - start_time

        print('Epoch {}, Time: {}s'.format(epoch + 1, epoch_time))

        print('Train: loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(train_metric['loss'], train_metric['auc'], train_metric['ap']))

        print('Valid: loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(val_metric['loss'], val_metric['auc'], val_metric['ap']))

        print('Test : loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(test_metric['loss'], test_metric['auc'], test_metric['ap']))

        print('======' * 20)

        if val_metric['auc'] > best_auc:
            best_auc = val_metric['auc']
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_history = history_val
            best_H = H_val
            best_C = C_val
            best_model_unchanged = 0
        else:
            best_model_unchanged += 1

    model.load_state_dict(best_model)
    test_metric, _, _, _ = evaluate_step_baseline(model, args, dataset, 'test', device, best_H, best_C, best_history)

    print('Final Test Results: roc_auc: {:.4f}, ap: {:.4f}'.format(test_metric['auc'], test_metric['ap']))
    return test_metric['auc'], test_metric['ap']