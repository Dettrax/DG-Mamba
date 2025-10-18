#!/usr/bin/env python
import optuna
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='View Optuna study results')
    parser.add_argument('--study_name', type=str, default='dyg_mamba_tuning', help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_dyg_tuning.db', help='Optuna storage URL')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top trials to show')
    
    args = parser.parse_args()
    
    # Load study
    print(f"Loading study '{args.study_name}' from {args.storage}...")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    
    print("=" * 80)
    print("OPTUNA STUDY RESULTS")
    print("=" * 80)
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
        print("\n" + "=" * 80)
        print("BEST TRIAL")
        print("=" * 80)
        print(f"Trial number: {study.best_trial.number}")
        print(f"AP (Average Precision): {study.best_trial.value:.6f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print(f"TOP {args.top_n} TRIALS")
        print("=" * 80)
        
        # Get completed trials and sort by value
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        
        # Create DataFrame for better visualization
        data = []
        for trial in sorted_trials[:args.top_n]:
            row = {
                'Trial': trial.number,
                'AP': f"{trial.value:.6f}",
                'triplet_weight': f"{trial.params['triplet_weight']:.6f}",
                'triplet_margin': f"{trial.params['triplet_margin']:.4f}",
                'prior_kl_weight': f"{trial.params['prior_kl_weight']:.6f}",
                'window_size': trial.params.get('window_size', 'N/A')
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_file = f"{args.study_name}_results.csv"
        full_data = []
        for trial in completed_trials:
            row = {
                'trial_number': trial.number,
                'ap': trial.value,
                'triplet_weight': trial.params['triplet_weight'],
                'triplet_margin': trial.params['triplet_margin'],
                'prior_kl_weight': trial.params['prior_kl_weight'],
                'window_size': trial.params.get('window_size', None)
            }
            full_data.append(row)
        
        full_df = pd.DataFrame(full_data)
        full_df.to_csv(csv_file, index=False)
        print(f"\nFull results saved to: {csv_file}")
        
        # Print parameter importance if enough trials
        if len(completed_trials) >= 10:
            print("\n" + "=" * 80)
            print("PARAMETER IMPORTANCE")
            print("=" * 80)
            try:
                importance = optuna.importance.get_param_importances(study)
                for param, imp in importance.items():
                    print(f"  {param}: {imp:.4f}")
            except Exception as e:
                print(f"Could not calculate parameter importance: {e}")
    else:
        print("\nNo completed trials yet.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
import subprocess
import time
import sys

def run_worker(worker_id, gpu_id=0, n_trials=50, study_name="dyg_mamba_tuning", storage="sqlite:///optuna_dyg_tuning.db"):
    """Run a single Optuna worker"""
    cmd = [
        sys.executable,
        "optuna_tune.py",
        "--n_trials", str(n_trials),
        "--n_jobs", "1",
        "--study_name", study_name,
        "--storage", storage,
        "--seed", str(worker_id)
    ]
    
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    
    log_file = f"optuna_worker{worker_id}.log"
    
    print(f"Starting Worker {worker_id}...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  GPU: {gpu_id}")
    print(f"  Log: {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**subprocess.os.environ, **env}
        )
    
    return process, log_file


def main():
    # Configuration
    GPU_ID = 0
    N_TRIALS = 50
    STUDY_NAME = "dyg_mamba_tuning"
    STORAGE = "sqlite:///optuna_dyg_tuning.db"
    N_WORKERS = 2
    
    print("=" * 80)
    print("Starting Optuna Distributed Optimization")
    print("=" * 80)
    print(f"Study name: {STUDY_NAME}")
    print(f"Storage: {STORAGE}")
    print(f"GPU: {GPU_ID}")
    print(f"Total trials: {N_TRIALS}")
    print(f"Workers: {N_WORKERS}")
    print("=" * 80)
    
    # Start workers
    workers = []
    log_files = []
    
    for i in range(1, N_WORKERS + 1):
        process, log_file = run_worker(i, GPU_ID, N_TRIALS, STUDY_NAME, STORAGE)
        workers.append(process)
        log_files.append(log_file)
        time.sleep(2)  # Small delay to avoid race conditions
    
    print("\n" + "=" * 80)
    print("Both workers are running!")
    print("=" * 80)
    print("Monitor progress with:")
    for log_file in log_files:
        print(f"  tail -f {log_file}")
    print("\nTo check GPU usage: watch -n 1 nvidia-smi")
    print("\nWaiting for workers to complete...")
    print("=" * 80)
    
    # Wait for all workers to finish
    for i, process in enumerate(workers, 1):
        return_code = process.wait()
        print(f"Worker {i} finished (exit code: {return_code})")
    
    print("\n" + "=" * 80)
    print("All workers completed!")
    print("=" * 80)
    print("\nTo view results, run:")
    print(f"  python view_optuna_results.py --study_name {STUDY_NAME} --storage {STORAGE}")


if __name__ == "__main__":
    main()
