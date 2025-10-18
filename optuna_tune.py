import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
import warnings
import optuna
from optuna.storages import RDBStorage
import argparse
import copy
from gine_v2 import DyGSTA

from utils.train import train
from dataloader.utils import fix_seed
from utils.config import get_args, get_dataset

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
import warnings
import optuna
from optuna.storages import RDBStorage
import argparse
import copy
from gine_v2 import DyGSTA

from utils.train import train
from dataloader.utils import fix_seed
from utils.config import get_args, get_dataset


def objective(trial, base_args, dataset):
    """Optuna objective function to optimize hyperparameters"""

    # Set the seed for reproducibility within each trial
    fix_seed(trial.number)  # Use trial number as seed for reproducibility

    # Suggest hyperparameters to tune
    triplet_weight = trial.suggest_float('triplet_weight', 0.001, 1.0, log=True)
    triplet_margin = trial.suggest_float('triplet_margin', 0.1, 2.0)
    prior_kl_weight = trial.suggest_float('prior_kl_weight', 1e-6, 1e-2, log=True)
    window_size = trial.suggest_categorical('window_size', [1, 3, 5, 7, 10])

    # Create a copy of args to avoid conflicts between parallel trials
    args = copy.deepcopy(base_args)

    # Update args with suggested hyperparameters
    args.triplet_weight = triplet_weight
    args.triplet_margin = triplet_margin
    args.prior_kl_weight = prior_kl_weight
    args.window_size = window_size

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create model
    model = DyGSTA(
        dim_in=args.input_dim,
        hidden_dim=args.hidden_dim,
        dim_out=1,
        num_heads=args.num_heads,
        num_hop=args.num_hop,
        window_size=args.window_size,
        recurrent=args.recurrent,
        time_encode=args.time_encode,
        device=args.device
    )

    model.to(args.device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Train and evaluate
    try:
        avg_auc, avg_ap = train(
            model,
            optimizer,
            dataset,
            args.num_epochs,
            args.patience,
            args.device,
            args
        )

        # Use AP as the optimization metric (you can change to avg_auc if preferred)
        return avg_ap

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()


def main():
    warnings.filterwarnings('ignore')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs within this worker')
    parser.add_argument('--study_name', type=str, default='dyg_mamba_tuning', help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_dyg_tuning.db', help='Optuna storage URL')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use for this worker')

    optuna_args = parser.parse_args()

    # Get base configuration
    import sys
    sys.argv = [sys.argv[0]]  # Reset argv to avoid conflicts with get_args()
    base_args = get_args()

    # Override GPU setting from command line
    base_args.gpu = optuna_args.gpu
    base_args.device = f'cuda:{optuna_args.gpu}' if torch.cuda.is_available() and optuna_args.gpu >= 0 else 'cpu'

    # Load dataset once (will be shared across trials)
    print(f'Worker on GPU {optuna_args.gpu}: Loading dataset...')
    dataset = get_dataset(base_args)
    print(f'Worker on GPU {optuna_args.gpu}: Dataset loaded successfully!')

    # Create or load study
    storage = RDBStorage(url=optuna_args.storage)

    # Create sampler (no seed for proper distributed coordination)
    sampler = optuna.samplers.TPESampler()
    try:
        study = optuna.create_study(
            study_name=optuna_args.study_name,
            storage=storage,
            direction='maximize',  # Maximize AP
            load_if_exists=True,
            sampler=sampler
        )
    except optuna.exceptions.DuplicatedStudyError:
        pass  # Study already exists

    print(f'Worker on GPU {optuna_args.gpu}: Starting Optuna optimization')
    print(f'Total trials to attempt: {optuna_args.n_trials}')
    print(f'Study name: {optuna_args.study_name}')
    print(f'Storage: {optuna_args.storage}')
    print(f'Device: {base_args.device}')
    print('=' * 80)

    # Run optimization with parallel jobs
    study.optimize(
        lambda trial: objective(trial, base_args, dataset),
        n_trials=optuna_args.n_trials,
        n_jobs=optuna_args.n_jobs,
        show_progress_bar=True
    )

    # Print results
    print('=' * 80)
    print(f'Worker on GPU {optuna_args.gpu}: Optimization completed!')

    # Only print best trial info if there are completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        print(f'Best trial: {study.best_trial.number}')
        print(f'Best AP: {study.best_trial.value:.4f}')
        print('Best hyperparameters:')
        for key, value in study.best_trial.params.items():
            print(f'  {key}: {value}')

        # Print top 5 trials
        print('\nTop 5 trials:')
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        for i, trial in enumerate(sorted_trials[:5], 1):
            print(f'{i}. Trial {trial.number}: AP={trial.value:.4f}, params={trial.params}')


if __name__ == "__main__":
    main()

def objective(trial, base_args, dataset):
    """Optuna objective function to optimize hyperparameters"""

    # Set the seed for reproducibility within each trial
    fix_seed(trial.number)  # Use trial number as seed for reproducibility

    # Suggest hyperparameters to tune
    triplet_weight = trial.suggest_float('triplet_weight', 0.001, 1.0, log=True)
    triplet_margin = trial.suggest_float('triplet_margin', 0.1, 2.0)
    prior_kl_weight = trial.suggest_float('prior_kl_weight', 1e-6, 1e-2, log=True)
    window_size = trial.suggest_categorical('window_size', [1, 3, 5, 7, 10])

    # Create a copy of args to avoid conflicts between parallel trials
    args = copy.deepcopy(base_args)

    # Update args with suggested hyperparameters
    args.triplet_weight = triplet_weight
    args.triplet_margin = triplet_margin
    args.prior_kl_weight = prior_kl_weight
    args.window_size = window_size

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create model
    model = DyGSTA(
        dim_in=args.input_dim,
        hidden_dim=args.hidden_dim,
        dim_out=1,
        num_heads=args.num_heads,
        num_hop=args.num_hop,
        window_size=args.window_size,
        recurrent=args.recurrent,
        time_encode=args.time_encode,
        device=args.device
    )

    model.to(args.device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Train and evaluate
    try:
        avg_auc, avg_ap = train(
            model,
            optimizer,
            dataset,
            args.num_epochs,
            args.patience,
            args.device,
            args
        )

        # Use AP as the optimization metric (you can change to avg_auc if preferred)
        return avg_ap

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()


def main():
    warnings.filterwarnings('ignore')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs within this worker')
    parser.add_argument('--study_name', type=str, default='dyg_mamba_tuning', help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_dyg_tuning.db', help='Optuna storage URL')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use for this worker')

    optuna_args = parser.parse_args()

    # Get base configuration
    import sys
    sys.argv = [sys.argv[0]]  # Reset argv to avoid conflicts with get_args()
    base_args = get_args()

    # Override GPU setting from command line
    base_args.gpu = optuna_args.gpu
    base_args.device = f'cuda:{optuna_args.gpu}' if torch.cuda.is_available() and optuna_args.gpu >= 0 else 'cpu'

    # Load dataset once (will be shared across trials)
    print(f'Worker on GPU {optuna_args.gpu}: Loading dataset...')
    dataset = get_dataset(base_args)
    print(f'Worker on GPU {optuna_args.gpu}: Dataset loaded successfully!')

    # Create or load study
    storage = RDBStorage(url=optuna_args.storage)

    # Create sampler (no seed for proper distributed coordination)
    sampler = optuna.samplers.TPESampler()
    try:
        study = optuna.create_study(
            study_name=optuna_args.study_name,
            storage=storage,
            direction='maximize',  # Maximize AP
            load_if_exists=True,
            sampler=sampler
        )
    except optuna.exceptions.DuplicatedStudyError:
        pass  # Study already exists

    print(f'Worker on GPU {optuna_args.gpu}: Starting Optuna optimization')
    print(f'Total trials to attempt: {optuna_args.n_trials}')
    print(f'Study name: {optuna_args.study_name}')
    print(f'Storage: {optuna_args.storage}')
    print(f'Device: {base_args.device}')
    print('=' * 80)

    # Run optimization with parallel jobs
    study.optimize(
        lambda trial: objective(trial, base_args, dataset),
        n_trials=optuna_args.n_trials,
        n_jobs=optuna_args.n_jobs,
        show_progress_bar=True
    )

    # Print results
    print('=' * 80)
    print(f'Worker on GPU {optuna_args.gpu}: Optimization completed!')

    # Only print best trial info if there are completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        print(f'Best trial: {study.best_trial.number}')
        print(f'Best AP: {study.best_trial.value:.4f}')
        print('Best hyperparameters:')
        for key, value in study.best_trial.params.items():
            print(f'  {key}: {value}')

        # Print top 5 trials
        print('\nTop 5 trials:')
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        for i, trial in enumerate(sorted_trials[:5], 1):
            print(f'{i}. Trial {trial.number}: AP={trial.value:.4f}, params={trial.params}')


if __name__ == "__main__":
    main()
