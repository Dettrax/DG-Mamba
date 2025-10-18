#!/bin/bash

# Script to run Optuna workers in parallel across multiple GPUs
# Automatically detects available GPUs and runs 2 workers per GPU

STUDY_NAME="dyg_mamba_tuning"
STORAGE="sqlite:///optuna_dyg_tuning.db"
TOTAL_TRIALS=50
WORKERS_PER_GPU=2

# Detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $N_GPUS GPU(s)"
else
    echo "WARNING: nvidia-smi not found. Assuming 1 GPU."
    N_GPUS=1
fi

TOTAL_WORKERS=$((N_GPUS * WORKERS_PER_GPU))

echo "Starting Optuna distributed optimization"
echo "Study name: $STUDY_NAME"
echo "Storage: $STORAGE"
echo "Total target trials: $TOTAL_TRIALS"
echo "Number of GPUs: $N_GPUS"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Total workers: $TOTAL_WORKERS"
echo "========================================"
echo ""
echo "How it works:"
echo "- $TOTAL_WORKERS workers distributed across $N_GPUS GPU(s)"
echo "- $WORKERS_PER_GPU workers per GPU running trials simultaneously"
echo "- Workers coordinate via shared SQLite database"
echo "- Expected speedup: ~${TOTAL_WORKERS}x faster"
echo "========================================"

# Array to store process IDs
PIDS=()
WORKER_ID=1

# Start workers distributed across GPUs
for ((GPU_ID=0; GPU_ID<N_GPUS; GPU_ID++)); do
    for ((i=0; i<WORKERS_PER_GPU; i++)); do
        LOG_FILE="optuna_worker${WORKER_ID}_gpu${GPU_ID}.log"

        echo "Starting Worker $WORKER_ID on GPU $GPU_ID..."

        python optuna_tune.py \
            --n_trials $TOTAL_TRIALS \
            --n_jobs 1 \
            --study_name $STUDY_NAME \
            --storage $STORAGE \
            --gpu $GPU_ID > $LOG_FILE 2>&1 &

        PIDS+=($!)
        echo "  Worker $WORKER_ID started (PID: ${PIDS[-1]}, GPU: $GPU_ID, Log: $LOG_FILE)"

        WORKER_ID=$((WORKER_ID + 1))
        sleep 1  # Small delay to avoid race conditions
    done
done

echo "========================================"
echo "$TOTAL_WORKERS workers are running in parallel!"
echo "Monitor progress with:"
for ((i=1; i<=TOTAL_WORKERS; i++)); do
    GPU=$((($i - 1) / WORKERS_PER_GPU))
    echo "  tail -f optuna_worker${i}_gpu${GPU}.log"
done
echo ""
echo "To check GPU usage (you should see $WORKERS_PER_GPU processes per GPU):"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Waiting for workers to complete..."

# Wait for all workers to finish
WORKER_ID=1
for PID in "${PIDS[@]}"; do
    wait $PID
    EXIT_CODE=$?
    echo "Worker $WORKER_ID finished (exit code: $EXIT_CODE)"
    WORKER_ID=$((WORKER_ID + 1))
done

echo "========================================"
echo "All workers completed!"
echo "Check the results with:"
echo "  python view_optuna_results.py --study_name \"$STUDY_NAME\" --storage \"$STORAGE\""
