#!/usr/bin/env python
import subprocess
import time
import sys
import os
import torch

def run_worker(worker_id, gpu_id=0, n_trials=25, study_name="dyg_mamba_tuning", storage="sqlite:///optuna_dyg_tuning.db"):
    """Run a single Optuna worker"""
    cmd = [
        sys.executable,
        "optuna_tune.py",
        "--n_trials", str(n_trials),
        "--n_jobs", "1",
        "--study_name", study_name,
        "--storage", storage,
        "--gpu", str(gpu_id)  # Pass GPU ID to the worker
    ]

    log_file = f"optuna_worker{worker_id}_gpu{gpu_id}.log"

    print(f"Starting Worker {worker_id}...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  GPU: {gpu_id}")
    print(f"  Trials for this worker: {n_trials}")
    print(f"  Log: {log_file}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    return process, log_file


def get_available_gpus():
    """Get list of available GPU IDs"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        return list(range(n_gpus))
    else:
        print("WARNING: No CUDA GPUs detected. Falling back to CPU.")
        return [-1]  # -1 indicates CPU


def main():
    # Configuration
    TOTAL_TRIALS = 50  # Total number of trials you want
    WORKERS_PER_GPU = 2  # Number of concurrent jobs per GPU
    STUDY_NAME = "dyg_mamba_tuning"
    STORAGE = "sqlite:///optuna_dyg_tuning.db"

    # Auto-detect available GPUs
    available_gpus = get_available_gpus()
    n_gpus = len(available_gpus)

    # Calculate total workers
    N_WORKERS = n_gpus * WORKERS_PER_GPU

    # Each worker will attempt up to TOTAL_TRIALS, but they share the study
    TRIALS_PER_WORKER = TOTAL_TRIALS

    print("=" * 80)
    print("Starting Optuna Distributed Optimization")
    print("=" * 80)
    print(f"Study name: {STUDY_NAME}")
    print(f"Storage: {STORAGE}")
    print(f"Available GPUs: {available_gpus if available_gpus != [-1] else 'CPU only'}")
    print(f"Number of GPUs: {n_gpus if available_gpus != [-1] else 0}")
    print(f"Workers per GPU: {WORKERS_PER_GPU}")
    print(f"Total workers: {N_WORKERS}")
    print(f"Total target trials: {TOTAL_TRIALS}")
    print(f"Strategy: {N_WORKERS} workers running trials in parallel")
    print("=" * 80)
    print("\nHow it works:")
    if available_gpus != [-1]:
        print(f"- {N_WORKERS} workers distributed across {n_gpus} GPU(s)")
        print(f"- {WORKERS_PER_GPU} workers per GPU running trials simultaneously")
    else:
        print(f"- {N_WORKERS} workers running on CPU")
    print(f"- All workers pick trials from the shared study pool")
    print(f"- Workers coordinate via the SQLite database")
    print(f"- They will collectively complete {TOTAL_TRIALS} unique trials")
    print(f"- Expected speedup: ~{N_WORKERS}x faster than sequential")
    print("=" * 80)

    # Start workers
    workers = []
    log_files = []
    worker_id = 1

    # Distribute workers across GPUs
    for gpu_id in available_gpus:
        for _ in range(WORKERS_PER_GPU):
            process, log_file = run_worker(worker_id, gpu_id, TRIALS_PER_WORKER, STUDY_NAME, STORAGE)
            workers.append(process)
            log_files.append(log_file)
            worker_id += 1
            time.sleep(1)  # Small delay to avoid race conditions

    print("\n" + "=" * 80)
    print(f"{N_WORKERS} workers are now running in parallel!")
    print("=" * 80)
    print("Monitor progress with:")
    for log_file in log_files:
        print(f"  tail -f {log_file}")

    if available_gpus != [-1]:
        print("\nTo check GPU usage:")
        print("  watch -n 1 nvidia-smi")
        print(f"\nYou should see {WORKERS_PER_GPU} processes per GPU")

    print("\nTo view current results:")
    print("  python view_optuna_results.py")
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
