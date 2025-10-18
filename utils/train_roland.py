import torch
import torch.optim as optim
import copy
import numpy as np

from utils.loss import compute_loss, compute_roc_auc, compute_ap
from dataloader.utils import negative_sampling

@torch.no_grad()
def evaluate_step_roland(model, dataset, task, hidden_state_list, device):
    today, tomorrow = task
    model.eval()

    current_snapshot = dataset.snapshots[today].clone()
    next_snapshot = dataset.snapshots[tomorrow].clone()

    x = current_snapshot.node_feature.to(device)
    edge_index = current_snapshot.edge_index.to(device)

    edge_label_index = next_snapshot.edge_label_index.to(device)
    label = next_snapshot.edge_label.to(device)

    if hidden_state_list is not None:
        previous_states = [x.detach().clone() for x in hidden_state_list]

        for i in range(len(previous_states)):
            if torch.is_tensor(previous_states[i]):
                previous_states[i] = previous_states[i].to(device)
    else:
        previous_states = [0 for _ in range(model.num_layer)]

    prediction = model(x, edge_index, edge_label_index, previous_states)

    loss = compute_loss(prediction, label)

    roc_auc = compute_roc_auc(prediction, label)
    ap = compute_ap(prediction, label)

    return {'loss': loss.item(), 'roc_auc': roc_auc, 'ap': ap}

@torch.no_grad()
def average_state_dict(dict1, dict2, weight):
    d1 = copy.deepcopy(dict1)
    d2 = copy.deepcopy(dict2)
    out = dict()
    for key in d1.keys():
        assert isinstance(d1[key], torch.Tensor)
        param1 = d1[key].detach().clone()
        assert isinstance(d2[key], torch.Tensor)
        param2 = d2[key].detach().clone()
        out[key] = (1 - weight) * param1 + weight * param2
    return out

def train_step_roland(model, optimizer, dataset, task, hidden_state_list, device):
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    today, tomorrow = task
    model.train()
    current_snapshot = dataset.snapshots[today].clone()
    next_snapshot = dataset.snapshots[tomorrow].clone()

    x = current_snapshot.node_feature.to(device)
    edge_index = current_snapshot.edge_index.to(device)

    # edge_label_index = next_snapshot.edge_label_index.to(device)
    # label = next_snapshot.edge_label.to(device)

    negative_edge_index = negative_sampling(next_snapshot.edge_index)
    edge_label_index = torch.cat([next_snapshot.edge_index, negative_edge_index], dim=-1).to(device)

    label = torch.cat([torch.ones(next_snapshot.edge_index.shape[1]),
                       torch.zeros(negative_edge_index.shape[1])]).to(device)

    if hidden_state_list is not None:
        previous_states = [x.detach().clone() for x in hidden_state_list]

        for i in range(len(previous_states)):
            if torch.is_tensor(previous_states[i]):
                previous_states[i] = previous_states[i].to(device)
    else:
        previous_states = [0 for _ in range(model.num_layer)]

    prediction = model(x, edge_index, edge_label_index, previous_states)

    loss = compute_loss(prediction, label)

    loss.backward()
    optimizer.step()

    return {'loss': loss}

@torch.no_grad()
def update_node_states(model, dataset, task, hidden_state_list, device):
    today, tomorrow = task
    model.eval()

    current_snapshot = dataset.snapshots[today].clone()

    x = current_snapshot.node_feature.to(device)
    edge_index = current_snapshot.edge_index.to(device)

    if hidden_state_list is not None:
        previous_states = [x.detach().clone() for x in hidden_state_list]

        for i in range(len(previous_states)):
            if torch.is_tensor(previous_states[i]):
                previous_states[i] = previous_states[i].to(device)

    else:
        previous_states = [0 for _ in range(model.num_layer)]

    new_hidden_state_list = model.get_hidden_state(x, edge_index, previous_states)

    new_hidden_state_list = [new_hidden_state.detach().clone() for new_hidden_state in new_hidden_state_list]

    return new_hidden_state_list

def train_roland(model, args, optimizer, dataset, n_epoch, device):
    task_range = range(len(dataset.snapshots) - 1)
    test_start, test_end = dataset.get_range_by_split('test')

    model_init = None

    auc_hist = list()
    ap_hist = list()
    mrr_hist = list()

    hidden_state_list = None

    for t in task_range:
        test_perf = evaluate_step_roland(model, dataset, (t, t + 1), hidden_state_list, device)

        print('Snapshot: {}, loss: {:.4f}, ap: {:.4f}, roc_auc: {:.4f}'.format(t + 1, test_perf['loss'], test_perf['ap'],
                                                                                            test_perf['roc_auc']))
        auc_hist.append(test_perf['roc_auc'])
        ap_hist.append(test_perf['ap'])

        del optimizer

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_model = {'val_loss': np.inf, 'train_epoch': 0, 'state': None}
        best_model_unchanged = 0

        if model_init is not None:
            model.load_state_dict(copy.deepcopy(model_init))

        for i in range(n_epoch):
            val_perf = evaluate_step_roland(model, dataset, (t, t + 1), hidden_state_list, device)

            if val_perf['loss'] < best_model['val_loss']:
                best_model = {'val_loss': val_perf['loss'], 'train_epoch': i, 'state': copy.deepcopy(model.state_dict())}
                best_model_unchanged = 0
            else:
                best_model_unchanged += 1

            # earyly stopping
            if best_model_unchanged >= 20:
                break
            else:
                train_perf = train_step_roland(model, optimizer,  dataset, (t, t + 1), hidden_state_list, device)

        model.load_state_dict(best_model['state'])

        # meta update
        if model_init is None:
            model_init = copy.deepcopy(best_model['state'])
        else:
            new_weight = 0.7
            model_init = average_state_dict(model_init, best_model['state'], new_weight)

        hidden_state_list = update_node_states(model, dataset, (t, t + 1), hidden_state_list, device)

    final_auc = np.mean(auc_hist[test_start - 1:])
    final_ap = np.mean(ap_hist[test_start - 1:])

    print('\nFinal Test Results: roc_auc: {}, ap: {}\n'.format(final_auc, final_ap))
    return final_auc, final_ap