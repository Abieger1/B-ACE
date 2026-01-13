#!/usr/bin/env python3
"""
run_optuna_parallel.py

Launch multiple Optuna workers from a single terminal using multiprocessing.
Uses SQLite with extended timeouts and staggered starts to avoid DB contention.

Usage:
    python run_optuna_parallel.py --n-workers 2 --timeout-hours 72
    
    # With custom settings
    python run_optuna_parallel.py \
        --n-workers 2 \
        --study-name bace_ppo_hpo \
        --n-trials 100 \
        --timesteps-per-trial 3000000 \
        --timeout-hours 72 \
        --train-script /path/to/train_bace_clean.py
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Launch parallel Optuna workers")
    
    parser.add_argument(
        "--n-workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="bace_ppo_baseline_hpo",
        help="Optuna study name"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Total trials across all workers"
    )
    parser.add_argument(
        "--timesteps-per-trial",
        type=int,
        default=3_000_000,
        help="Training timesteps per trial"
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=72.0,
        help="Timeout in hours per worker"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Optuna_Output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default=None,
        help="Path to training script (required)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to B-ACE config"
    )
    parser.add_argument(
        "--optuna-script",
        type=str,
        default="rl_files/optuna_bace_hpo.py",
        help="Path to optuna HPO script"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--use-enriched-obs",
        action="store_true",
        default=False,
        help="Enable enriched observations"
    )
    parser.add_argument(
        "--stagger-seconds",
        type=int,
        default=10,
        help="Seconds between worker starts (reduces DB contention)"
    )
    
    return parser.parse_args()


def run_worker(
    worker_id: int,
    study_name: str,
    db_path: str,
    n_trials: int,
    timesteps: int,
    timeout_hours: float,
    output_dir: str,
    train_script: str,
    config: str,
    optuna_script: str,
    seed: int,
    use_enriched_obs: bool,
):
    """
    Function that runs in each worker process.
    Launches the optuna HPO script as a subprocess.
    """
    print(f"\n[Worker {worker_id}] Starting at {datetime.now().strftime('%H:%M:%S')}")
    print(f"[Worker {worker_id}] Study: {study_name}, Trials: {n_trials}, Timesteps: {timesteps:,}")
    
    # Build command
    cmd = [
        sys.executable,
        "-u",
        optuna_script,
        "--study-name", study_name,
        "--storage-url", f"sqlite:///{db_path}",
        "--worker-id", str(worker_id),
        "--n-trials", str(n_trials),
        "--timesteps-per-trial", str(timesteps),
        "--timeout-hours", str(timeout_hours),
        "--output-dir", output_dir,
        "--seed", str(seed),
    ]
    
    if train_script:
        cmd.extend(["--train-script", train_script])
    if config:
        cmd.extend(["--config", config])
    if use_enriched_obs:
        cmd.append("--use-enriched-obs")
    
    # Run and stream output with worker prefix
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
        )
        
        for line in process.stdout:
            # Prefix each line with worker ID for clarity
            print(f"[W{worker_id}] {line}", end="")
        
        process.wait()
        print(f"\n[Worker {worker_id}] Finished with return code {process.returncode}")
        return process.returncode
        
    except KeyboardInterrupt:
        print(f"\n[Worker {worker_id}] Interrupted by user")
        if process:
            process.terminate()
        return -1
    except Exception as e:
        print(f"\n[Worker {worker_id}] Error: {e}")
        return -1


def main():
    args = parse_args()
    
    # Validate required args
    if args.train_script is None:
        print("ERROR: --train-script is required")
        print("Example: --train-script /path/to/train_bace_clean.py")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Database path (shared by all workers)
    db_path = output_dir / f"{args.study_name}.db"
    
    print(f"\n{'='*70}")
    print(f"  PARALLEL OPTUNA HPO LAUNCHER")
    print(f"{'='*70}")
    print(f"  Workers:          {args.n_workers}")
    print(f"  Study:            {args.study_name}")
    print(f"  Database:         {db_path}")
    print(f"  Total trials:     {args.n_trials}")
    print(f"  Timesteps/trial:  {args.timesteps_per_trial:,}")
    print(f"  Timeout:          {args.timeout_hours} hours")
    print(f"  Train script:     {args.train_script}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Stagger:          {args.stagger_seconds}s between workers")
    print(f"{'='*70}")
    print(f"\n  Starting {args.n_workers} workers...")
    print(f"  Press Ctrl+C to stop all workers\n")
    
    # Use 'spawn' context for clean process isolation (important on macOS)
    ctx = mp.get_context('spawn')
    
    processes = []
    for i in range(args.n_workers):
        p = ctx.Process(
            target=run_worker,
            args=(
                i,                          # worker_id
                args.study_name,            # study_name
                str(db_path),               # db_path
                args.n_trials,              # n_trials (each worker can run up to this)
                args.timesteps_per_trial,   # timesteps
                args.timeout_hours,         # timeout_hours
                args.output_dir,            # output_dir
                args.train_script,          # train_script
                args.config,                # config
                args.optuna_script,         # optuna_script
                args.seed,                  # seed
                args.use_enriched_obs,      # use_enriched_obs
            )
        )
        processes.append(p)
        p.start()
        
        # Stagger worker starts to reduce SQLite contention
        if i < args.n_workers - 1:
            print(f"  [Main] Waiting {args.stagger_seconds}s before starting next worker...")
            time.sleep(args.stagger_seconds)
    
    # Wait for all workers
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\n[Main] Keyboard interrupt - terminating all workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
    
    print(f"\n{'='*70}")
    print(f"  ALL WORKERS COMPLETED")
    print(f"{'='*70}")
    print(f"  Results saved to: {output_dir}")
    print(f"  Database: {db_path}")
    print(f"\n  To view results:")
    print(f"    python -c \"import optuna; s = optuna.load_study('{args.study_name}', 'sqlite:///{db_path}'); print(s.best_params)\"")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
