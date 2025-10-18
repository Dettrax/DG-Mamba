# Optuna Hyperparameter Tuning for DG-Mamba

This setup allows you to tune the following hyperparameters using Optuna:
- `triplet_weight`: Weight for triplet loss (range: 0.001 to 1.0, log scale)
- `triplet_margin`: Margin for triplet loss (range: 0.1 to 2.0)
- `prior_kl_weight`: Weight for KL divergence prior regularization (range: 1e-6 to 1e-2, log scale)

## Key Feature: Multi-GPU Parallel Optimization

**Automatically scales to use all available GPUs with 2 workers per GPU!**

- **1 GPU**: 2 workers → 2x speedup
- **2 GPUs**: 4 workers → 4x speedup
- **4 GPUs**: 8 workers → 8x speedup
- **8 GPUs**: 16 workers → 16x speedup

Instead of running 50 trials sequentially, this distributes trials across multiple GPUs and workers, completing optimization much faster.

## How It Works

- **Auto-detects available GPUs** using PyTorch
- Spawns **2 workers per GPU** (configurable)
- All workers share a **single Optuna study** via SQLite database
- Workers coordinate to avoid duplicate trials
- **Example with 4 GPUs and 50 trials**:
  - Sequential (1 worker): ~50 trial durations
  - Parallel (8 workers on 4 GPUs): ~6-7 trial durations
  - **Speedup: ~8x faster!**

### Distribution Pattern

```
GPU 0: Worker 1, Worker 2
GPU 1: Worker 3, Worker 4
GPU 2: Worker 5, Worker 6
GPU 3: Worker 7, Worker 8
```

All 8 workers pull from the same pool of 50 trials, completing them collectively.

## Installation

Install Optuna if you haven't already:

```bash
pip install optuna pandas
```

## Usage

### Option 1: Python Launcher (Recommended)

Simply run - it auto-detects your GPUs:

```bash
python run_optuna_multiprocess.py
```

This will:
- Auto-detect all available GPUs
- Start 2 workers per GPU automatically
- All workers share the same study (50 total trials)
- Each worker picks the next trial from the shared study
- Progress is logged to `optuna_worker{id}_gpu{gpu}.log`
- **Complete in ~(50 / (num_gpus * 2)) trial durations**

### Option 2: Bash Script

Make the script executable and run:

```bash
chmod +x run_optuna_parallel.sh
./run_optuna_parallel.sh
```

### Option 3: Manual Worker Launch (Advanced)

Launch workers manually specifying GPU IDs:

**Terminal 1 (GPU 0, Worker 1):**
```bash
python optuna_tune.py --n_trials 50 --n_jobs 1 --study_name dyg_mamba_tuning --storage sqlite:///optuna_dyg_tuning.db --gpu 0
```

**Terminal 2 (GPU 0, Worker 2):**
```bash
python optuna_tune.py --n_trials 50 --n_jobs 1 --study_name dyg_mamba_tuning --storage sqlite:///optuna_dyg_tuning.db --gpu 0
```

**Terminal 3 (GPU 1, Worker 3):**
```bash
python optuna_tune.py --n_trials 50 --n_jobs 1 --study_name dyg_mamba_tuning --storage sqlite:///optuna_dyg_tuning.db --gpu 1
```

And so on for additional GPUs...

## Monitoring Progress

### Check worker logs:
```bash
# View all worker logs
tail -f optuna_worker*_gpu*.log

# View specific GPU's workers
tail -f optuna_worker*_gpu0.log
```

### Monitor GPU usage (you should see 2 processes per GPU):
```bash
watch -n 1 nvidia-smi
```

### View current results (even while running):
```bash
python view_optuna_results.py
```

## Viewing Results

After optimization completes (or while it's running):

```bash
python view_optuna_results.py
```

This will show:
- Best trial and hyperparameters
- Top 10 trials
- Parameter importance analysis
- Save results to CSV file

## Customization

### Change total number of trials:

Edit `run_optuna_multiprocess.py`:
```python
TOTAL_TRIALS = 50  # Change this value
```

### Change number of workers per GPU:

```python
WORKERS_PER_GPU = 2  # Change to 1, 2, 3, etc.
```

**Note**: More workers per GPU requires more GPU memory.

### Use specific GPUs only:

If you want to use only specific GPUs (e.g., GPU 0 and GPU 2), modify `run_optuna_multiprocess.py`:

```python
def get_available_gpus():
    """Get list of available GPU IDs"""
    # Instead of auto-detection, manually specify:
    return [0, 2]  # Use only GPU 0 and GPU 2
```

### Change optimization metric:

By default, optimizes Average Precision (AP). To use AUC instead, edit `optuna_tune.py`:

```python
# In the objective function, change:
return avg_ap  # to:
return avg_auc
```

### Adjust hyperparameter search ranges:

Edit `optuna_tune.py`:

```python
triplet_weight = trial.suggest_float('triplet_weight', 0.001, 1.0, log=True)
triplet_margin = trial.suggest_float('triplet_margin', 0.1, 2.0)
prior_kl_weight = trial.suggest_float('prior_kl_weight', 1e-6, 1e-2, log=True)
```

## Files Created

- `optuna_tune.py` - Main Optuna optimization script (supports multi-GPU)
- `run_optuna_multiprocess.py` - Python launcher with auto GPU detection
- `run_optuna_parallel.sh` - Bash launcher with auto GPU detection
- `view_optuna_results.py` - Results visualization script
- `optuna_dyg_tuning.db` - SQLite database storing study results (shared across all workers)
- `optuna_worker{id}_gpu{gpu}.log` - Log files for each worker
- `dyg_mamba_tuning_results.csv` - Exported results

## Performance Notes

### GPU Memory Requirements

Running 2 trials in parallel per GPU requires ~2x GPU memory per GPU. 

**GPU Memory Guidelines:**
- 12GB GPU: 2 workers per GPU should work
- 24GB GPU: Can try 3-4 workers per GPU
- 40GB+ GPU: Can try 4-6 workers per GPU

If you encounter OOM errors:

**Option 1: Reduce workers per GPU**
```python
WORKERS_PER_GPU = 1  # One worker per GPU
```

**Option 2: Reduce model size** (in your main config)
```python
--hidden_dim 32  # Instead of 64
--num_heads 2    # Instead of 4
```

### Expected Speedup

| GPUs | Workers | Speedup | 50 Trials Time |
|------|---------|---------|----------------|
| 1    | 2       | ~2x     | ~25 durations  |
| 2    | 4       | ~4x     | ~12 durations  |
| 4    | 8       | ~8x     | ~6 durations   |
| 8    | 16      | ~16x    | ~3 durations   |

**Note**: Actual speedup may be slightly less due to:
- Database locking overhead
- I/O bottlenecks
- Slight trial duration variance

### Multi-GPU Behavior

- **All workers** pull from the same shared study
- **Optuna** handles trial assignment automatically
- **SQLite** coordinates access (brief locks are normal)
- **Each GPU** runs its assigned workers independently
- **Load balancing** is automatic - no manual intervention needed

## Troubleshooting

**Out of memory errors:**
- Reduce `WORKERS_PER_GPU` to 1
- Reduce model size or batch size
- Use GPUs with more memory

**Database locked errors:**
- Normal for brief periods with many workers
- Workers will automatically retry
- If persistent, reduce number of workers

**Not all GPUs being used:**
- Check GPU detection: `python -c "import torch; print(torch.cuda.device_count())"`
- Verify all GPUs are visible: `nvidia-smi`

**Workers finish too quickly:**
- Check worker logs for errors: `cat optuna_worker*_gpu*.log | grep -i error`
- Ensure workers are running: `ps aux | grep optuna_tune`

**No speedup observed:**
- Verify multiple processes are running: `nvidia-smi` should show multiple processes
- Check if GPU memory is sufficient for multiple workers
- Review worker logs to ensure all are actively training

## Example Output

```
================================================================================
Starting Optuna Distributed Optimization
================================================================================
Study name: dyg_mamba_tuning
Storage: sqlite:///optuna_dyg_tuning.db
Available GPUs: [0, 1, 2, 3]
Number of GPUs: 4
Workers per GPU: 2
Total workers: 8
Total target trials: 50
Strategy: 8 workers running trials in parallel
================================================================================

How it works:
- 8 workers distributed across 4 GPU(s)
- 2 workers per GPU running trials simultaneously
- All workers pick trials from the shared study pool
- Workers coordinate via the SQLite database
- They will collectively complete 50 unique trials
- Expected speedup: ~8x faster than sequential
================================================================================
```
