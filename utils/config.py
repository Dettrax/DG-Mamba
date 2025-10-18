import argparse
import sys
import torch

from dataloader.dataset import UCIDataset, BitCoinDataset, SXDataset, TechDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='uci',
                        choices=['uci', 'bc-otc', 'tech', 'mathof', 'askubuntu', 'superuser'])

    parser.add_argument('--dataset_interval', help='time interval to spilt data', default='W',
                        choices=['W', 'D', 'M'])

    parser.add_argument('--train_ratio', type=float, default=0.7, help='ratio of validation set')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--input_dim', type=int, default=64, help='randomly generate initial embedding')

    parser.add_argument('--model_name', type=str, default='DyGSTA', help='name of the model',
                        choices=['DyGSTA', 'EvolveGCN', 'ROLAND', 'DySAT', 'GCRN', 'TGCN', 'WinGNN'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')

    # DyGSTA
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads used in attention layer')
    parser.add_argument('--window_size', type=int, default=5, help='merge window in subtree model')
    parser.add_argument('--sampling_ratio', type=float, default=0.8, help='sampling ratio')
    parser.add_argument('--num_hop', type=int, default=5, help='number of hop')
    parser.add_argument('--recurrent', action='store_true', default=True, help='use recurrent layer')
    parser.add_argument('--time_encode', action='store_true', default=True, help='use time encode')

    # Optional Gaussian embedding and metric learning
    parser.add_argument('--use_gaussian', action='store_true', default=True, help='use Gaussian node embeddings with KL scoring')
    parser.add_argument('--gauss_score', type=str, default='kl', choices=['kl', 'l2'], help='pairwise score type when using Gaussian embeddings')
    parser.add_argument('--gauss_scale', type=float, default=1.0, help='scale for converting distance to logits')
    parser.add_argument('--gauss_aux_only', action='store_true', default=True, help='use Gaussian as auxiliary (decoder scoring stays)')
    parser.add_argument('--triplet_weight', type=float, default=0.108, help='weight for triplet loss (Gaussian)')
    parser.add_argument('--triplet_margin', type=float, default=0.258, help='margin for triplet loss')
    parser.add_argument('--prior_kl_weight', type=float, default=0.000002, help='weight for KL to N(0,I) prior regularization')

    # Baselines
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--rnn', type=str, default='GRU', help='EvolveGCN RNN type')

    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')

    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')

    # Early stopping / selection metric
    parser.add_argument('--early_metric', type=str, default='ap', choices=['auc', 'ap'], help='metric to select best checkpoint')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    return args

def get_dataset(args):
    if args.dataset_name == 'uci':
        URL = 'http://snap.stanford.edu/data/CollegeMsg.txt.gz'
        FILE_NAME = 'CollegeMsg.txt.gz'
        dataset = UCIDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)

    elif args.dataset_name == 'bc-otc':
        URL = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
        FILE_NAME = 'soc-sign-bitcoinotc.csv.gz'
        dataset = BitCoinDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)

    elif args.dataset_name == 'superuser':
        URL = 'https://snap.stanford.edu/data/sx-superuser.txt.gz'
        FILE_NAME = 'sx-superuser.txt.gz'
        dataset = SXDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)

    elif args.dataset_name == 'mathof':
        URL = 'https://snap.stanford.edu/data/sx-mathoverflow.txt.gz'
        FILE_NAME = 'sx-mathoverflow.txt.gz'
        dataset = SXDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)

    elif args.dataset_name == 'askubuntu':
        URL = 'https://snap.stanford.edu/data/sx-askubuntu.txt.gz'
        FILE_NAME = 'sx-askubuntu.txt.gz'
        dataset = SXDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)

    elif args.dataset_name == 'tech':
        FILE_NAME = 'data/tech-as-topology.edges'
        dataset = TechDataset(FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)
    else:
        raise NotImplementedError('No such dataset: {}'.format(args.dataset_name))

    return dataset