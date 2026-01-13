#!/usr/bin/env python3
"""
optuna_bace_hpo.py

Optuna hyperparameter optimization for B-ACE PPO baseline.
Supports parallel workers via SQLite (with extended timeout) or PostgreSQL.

SCORING: Uses AlgScore - a weighted composite of lower confidence bounds:
    AlgScore = 0.6 * LCB(max_performance) + 0.4 * LCB(mean_performance)
    
    Where LCB = lower confidence bound (single-sided 95%) computed from
    evaluation episode rewards.

Time estimates (based on 2.5h for 10M steps):
- 2M steps/trial ≈ 30 min
- 3M steps/trial ≈ 45 min
- 5M steps/trial ≈ 75 min
- 48h × 2 workers ≈ 120+ trials at 3M steps

Usage:
    # With parallel launcher (recommended)
    python run_optuna_parallel.py --n-workers 2 --train-script /path/to/train_bace_clean.py
    
    # Single worker with SQLite
    python optuna_bace_hpo.py \
        --storage-url "sqlite:///Optuna_Output/bace_hpo.db" \
        --train-script /path/to/train_bace_clean.py
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# =============================================================================
# ALGORITHM SCORE COMPUTATION
# =============================================================================

def compute_lower_confidence_bound(values: List[float], confidence: float = 0.95) -> float:
    """
    Compute single-sided lower confidence bound for the mean.
    
    LCB = mean - t_{alpha, n-1} * (std / sqrt(n))
    
    Args:
        values: List of sample values
        confidence: Confidence level (default 0.95 for 95% one-sided)
    
    Returns:
        Lower confidence bound
    """
    from scipy import stats
    
    n = len(values)
    if n < 2:
        return float(np.mean(values)) if n == 1 else float('-inf')
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std (n-1 denominator)
    
    # t-value for one-sided confidence interval
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha, df=n-1)
    
    # Standard error of the mean
    sem = std / np.sqrt(n)
    
    # Lower confidence bound
    lcb = mean - t_value * sem
    
    return float(lcb)


def compute_alg_score(checkpoint_means: List[float], 
                      checkpoint_stds: List[float],
                      n_eval_episodes: int,
                      max_weight: float = 0.6, 
                      mean_weight: float = 0.4,
                      confidence: float = 0.95) -> Tuple[float, dict]:
    """
    Compute Algorithm Score (AlgScore) - a weighted composite indicating 
    reliable peak and average performance across ALL evaluation checkpoints.
    
    AlgScore = max_weight * LCB_max + mean_weight * LCB_mean
    
    Where:
        LCB_mean (40%): Lower bound of 95% CI for the average of all checkpoint means.
                        - Treats checkpoint means as samples
                        - Variance comes from variability ACROSS checkpoints
                        - Captures training stability (high variance = lower score)
        
        LCB_max (60%):  Lower bound of 95% CI for the single best checkpoint's mean.
                        - Uses the superlative checkpoint's own std from its eval episodes
                        - Captures worst-case superlative policy performance
    
    Args:
        checkpoint_means: List of mean rewards at each evaluation checkpoint
        checkpoint_stds: List of std rewards at each evaluation checkpoint
        n_eval_episodes: Number of evaluation episodes per checkpoint
        max_weight: Weight for LCB_max (default 0.6)
        mean_weight: Weight for LCB_mean (default 0.4)
        confidence: Confidence level for LCB computation (default 0.95)
    
    Returns:
        Tuple of (alg_score, details_dict)
    """
    from scipy import stats
    
    n_checkpoints = len(checkpoint_means)
    if n_checkpoints < 2:
        return float('-inf'), {"error": "Insufficient checkpoints"}
    
    means = np.array(checkpoint_means)
    stds = np.array(checkpoint_stds) if checkpoint_stds else np.zeros_like(means)
    
    # ==========================================================================
    # LCB_mean: Lower bound for AVERAGE of all checkpoint means
    # Treats each checkpoint mean as a sample, uses cross-checkpoint variance
    # ==========================================================================
    mean_of_means = float(np.mean(means))
    std_across_checkpoints = float(np.std(means, ddof=1))
    
    # t-value for one-sided CI (n_checkpoints - 1 degrees of freedom)
    alpha = 1 - confidence
    t_value_checkpoints = stats.t.ppf(1 - alpha, df=n_checkpoints - 1)
    sem_checkpoints = std_across_checkpoints / np.sqrt(n_checkpoints)
    
    lcb_mean = mean_of_means - t_value_checkpoints * sem_checkpoints
    
    # ==========================================================================
    # LCB_max: Lower bound for the SINGLE BEST checkpoint's mean
    # Uses that checkpoint's own std from its evaluation episodes
    # ==========================================================================
    best_idx = int(np.argmax(means))
    best_checkpoint_mean = float(means[best_idx])
    best_checkpoint_std = float(stds[best_idx]) if best_idx < len(stds) else 0.0
    
    # t-value for one-sided CI (n_eval_episodes - 1 degrees of freedom)
    t_value_episodes = None
    if n_eval_episodes > 1 and best_checkpoint_std > 0:
        t_value_episodes = stats.t.ppf(1 - alpha, df=n_eval_episodes - 1)
        sem_best = best_checkpoint_std / np.sqrt(n_eval_episodes)
        lcb_max = best_checkpoint_mean - t_value_episodes * sem_best
    else:
        # Fallback if no std available or only 1 episode
        lcb_max = best_checkpoint_mean
    
    # ==========================================================================
    # Compute weighted AlgScore
    # ==========================================================================
    alg_score = max_weight * lcb_max + mean_weight * lcb_mean
    
    # Detailed breakdown for logging/analysis
    details = {
        "alg_score": float(alg_score),
        # LCB components
        "lcb_max": float(lcb_max),
        "lcb_mean": float(lcb_mean),
        # Raw values
        "raw_max": float(best_checkpoint_mean),
        "raw_mean": float(mean_of_means),
        "raw_min": float(np.min(means)),
        # Superlative checkpoint info
        "superlative_checkpoint_idx": int(best_idx),
        "superlative_checkpoint_std": float(best_checkpoint_std),
        "superlative_n_eval_episodes": int(n_eval_episodes),
        # Cross-checkpoint variability (training stability)
        "std_across_checkpoints": float(std_across_checkpoints),
        "n_checkpoints": int(n_checkpoints),
        # Statistical details
        "t_value_for_mean": float(t_value_checkpoints),
        "t_value_for_max": float(t_value_episodes) if t_value_episodes is not None else None,
        "confidence": float(confidence),
        "max_weight": float(max_weight),
        "mean_weight": float(mean_weight),
    }
    
    return float(alg_score), details


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna HPO for B-ACE PPO with AlgScore")
    
    parser.add_argument(
        "--study-name",
        type=str,
        default="bace_ppo_baseline_hpo",
        help="Name of the Optuna study"
    )
    parser.add_argument(
        "--storage-url",
        type=str,
        default="sqlite:///Optuna_Output/bace_hpo.db",
        help="Database URL (postgresql://... or sqlite:///path.db)"
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker ID for parallel execution (affects seed offset)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Maximum number of trials (shared across all workers)"
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=72.0,
        help="Maximum time in hours for this worker"
    )
    parser.add_argument(
        "--timesteps-per-trial",
        type=int,
        default=3_000_000,
        help="Training timesteps per trial (~45 min at your speed)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=200_000,
        help="Evaluation frequency for pruning decisions"
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (recommend 20+ for AlgScore)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility"
    )
    parser.add_argument(
        "--use-enriched-obs",
        action="store_true",
        default=False,
        help="Enable enriched observations"
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="rl_files/train_bace_clean.py",
        help="Path to training script"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to B-ACE config"
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=15,
        help="Number of random trials before TPE kicks in"
    )
    parser.add_argument(
        "--pruning-warmup-steps",
        type=int,
        default=5,
        help="Number of evaluation steps before pruning can occur"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Optuna_Output",
        help="Directory for output files"
    )
    # AlgScore weights
    parser.add_argument(
        "--algscore-max-weight",
        type=float,
        default=0.6,
        help="Weight for LCB of max performance in AlgScore"
    )
    parser.add_argument(
        "--algscore-mean-weight",
        type=float,
        default=0.4,
        help="Weight for LCB of mean performance in AlgScore"
    )
    parser.add_argument(
        "--algscore-confidence",
        type=float,
        default=0.95,
        help="Confidence level for LCB computation (0.95 = 95%)"
    )
    
    return parser.parse_args()


# =============================================================================
# HYPERPARAMETER SPACE
# =============================================================================

def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    """
    Define the hyperparameter search space.
    
    Tuned parameters (8):
    1. initial_learning_rate - most impactful
    2. initial_entropy_coef - exploration
    3. n_steps - rollout length  
    4. batch_size - mini-batch size
    5. gae_lambda - advantage estimation
    6. clip_eps - PPO clipping
    7. n_epochs - PPO update epochs
    8. vf_coef - value function loss weight
    
    Fixed parameters:
    - gamma = 0.99
    - lr_decay_ratio = 0.1 (final_lr = initial_lr * 0.1)
    - entropy_decay_ratio = 0.5 (final_entropy = initial_entropy * 0.5)
    """
    
    # Learning rate: log-uniform
    initial_lr = trial.suggest_float("initial_learning_rate", 1e-5, 1e-3, log=True)
    
    # Entropy coefficient
    initial_entropy = trial.suggest_float("initial_entropy_coef", 0.001, 0.1, log=True)
    
    # Rollout and batch
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 6144, 8192])
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    
    # Core PPO params
    gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
    clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
    n_epochs = trial.suggest_int("n_epochs", 4, 12)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)
    
    return {
        "initial_learning_rate": initial_lr,
        "final_learning_rate": initial_lr * 0.1,        # Fixed decay ratio
        "initial_entropy_coef": initial_entropy,
        "final_entropy_coef": initial_entropy * 0.5,    # Fixed decay ratio
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": 0.99,                                   # Fixed
        "gae_lambda": gae_lambda,
        "clip_eps": clip_eps,
        "n_epochs": n_epochs,
        "vf_coef": vf_coef,
    }


# =============================================================================
# TRAINING AND RESULT PARSING
# =============================================================================

def run_training_trial(
    trial: optuna.Trial,
    params: dict,
    args,
) -> float:
    """Run a single training trial as subprocess."""
    
    # Unique seed per trial (offset by worker_id to avoid collisions)
    trial_seed = args.seed + trial.number * 10 + args.worker_id
    
    cmd = [
        sys.executable,
        args.train_script,
        "--total-timesteps", str(args.timesteps_per_trial),
        "--seed", str(trial_seed),
        "--experiment-name", f"optuna_{args.study_name}",
        "--run-name", f"trial_{trial.number:04d}_w{args.worker_id}",
        "--initial-learning-rate", str(params["initial_learning_rate"]),
        "--final-learning-rate", str(params["final_learning_rate"]),
        "--initial-entropy-coef", str(params["initial_entropy_coef"]),
        "--final-entropy-coef", str(params["final_entropy_coef"]),
        "--n-steps", str(params["n_steps"]),
        "--batch-size", str(params["batch_size"]),
        "--gamma", "0.99",
        "--gae-lambda", str(params["gae_lambda"]),
        "--clip-eps", str(params["clip_eps"]),
        "--epochs", str(params["n_epochs"]),
        "--vf-coef", str(params["vf_coef"]),
    ]
    
    if args.config:
        cmd.extend(["--config", args.config])
    if args.use_enriched_obs:
        cmd.append("--use-enriched-obs")
    
    print(f"\n[Worker {args.worker_id}] Trial {trial.number}: Starting")
    print(f"  Params: lr={params['initial_learning_rate']:.2e}, ent={params['initial_entropy_coef']:.3f}, "
          f"gamma={params['gamma']:.4f}, n_steps={params['n_steps']}")
    
    trial_start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=4 * 3600,  # 4 hour timeout (generous for 3M steps)
        )
        
        if result.returncode != 0:
            print(f"[Worker {args.worker_id}] Trial {trial.number} FAILED")
            print(f"STDERR (last 1000 chars): {result.stderr[-1000:]}")
            raise optuna.TrialPruned()
            
    except subprocess.TimeoutExpired:
        print(f"[Worker {args.worker_id}] Trial {trial.number} TIMED OUT")
        raise optuna.TrialPruned()
    
    trial_time = time.time() - trial_start
    print(f"[Worker {args.worker_id}] Trial {trial.number} completed in {trial_time/60:.1f} min")
    
    # Find and parse results, compute AlgScore
    alg_score, details = parse_results_algscore(trial, args, result.stdout)
    
    # Store additional metrics as user attributes for later analysis
    for key, value in details.items():
        trial.set_user_attr(key, value)
    
    print(f"[Worker {args.worker_id}] Trial {trial.number}: AlgScore = {alg_score:.2f}")
    print(f"  └─ LCB_max={details.get('lcb_max', 0):.2f} (from superlative @ checkpoint {details.get('superlative_checkpoint_idx', '?')})")
    print(f"  └─ LCB_mean={details.get('lcb_mean', 0):.2f} (from {details.get('n_checkpoints', '?')} checkpoints)")
    print(f"  └─ raw_max={details.get('raw_max', 0):.2f} ± {details.get('superlative_checkpoint_std', 0):.2f}, "
          f"raw_mean={details.get('raw_mean', 0):.2f} ± {details.get('std_across_checkpoints', 0):.2f}")
    
    return alg_score


def parse_results_algscore(trial: optuna.Trial, args, stdout: str) -> Tuple[float, dict]:
    """
    Parse results from JSON and compute AlgScore using ALL evaluation checkpoints.
    
    AlgScore uses:
    - LCB_mean (40%): Lower bound of 95% CI for average of all checkpoint means
                      (uses cross-checkpoint variance for training stability)
    - LCB_max (60%):  Lower bound of 95% CI for the best checkpoint's mean
                      (uses that checkpoint's own std from its eval episodes)
    
    Returns:
        Tuple of (alg_score, details_dict)
    """
    import glob
    
    # Try to find results JSON
    patterns = [
        f"runs_sb3/optuna_{args.study_name}/trial_{trial.number:04d}_w{args.worker_id}*/training_results_*.json",
        f"runs_sb3/optuna_{args.study_name}/*trial_{trial.number:04d}*/training_results_*.json",
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            with open(files[0]) as f:
                results = json.load(f)
            
            # Get ALL evaluation checkpoint data
            eval_summary = results.get("evaluation_summary", {})
            checkpoint_means = eval_summary.get("mean_rewards", [])
            checkpoint_stds = eval_summary.get("std_rewards", [])
            checkpoint_timesteps = eval_summary.get("timesteps", [])
            n_eval_episodes = eval_summary.get("n_eval_episodes", args.n_eval_episodes)
            
            # Report intermediate values for pruning
            # Wrapped in try/except to handle retried/resumed trials
            try:
                for i, reward in enumerate(checkpoint_means):
                    trial.report(reward, step=i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            except optuna.exceptions.UpdateFinishedTrialError:
                # Trial was already finished (from a previous run) - skip reporting
                pass
            
            if len(checkpoint_means) >= 2:
                # Compute AlgScore from ALL checkpoint data
                alg_score, details = compute_alg_score(
                    checkpoint_means=checkpoint_means,
                    checkpoint_stds=checkpoint_stds,
                    n_eval_episodes=n_eval_episodes,
                    max_weight=args.algscore_max_weight,
                    mean_weight=args.algscore_mean_weight,
                    confidence=args.algscore_confidence,
                )
                
                # Add source info
                details["source"] = "all_evaluation_checkpoints"
                details["json_file"] = files[0]
                
                # Add timestep info for superlative checkpoint
                best_idx = details.get("superlative_checkpoint_idx", 0)
                if checkpoint_timesteps and best_idx < len(checkpoint_timesteps):
                    details["superlative_checkpoint_timestep"] = checkpoint_timesteps[best_idx]
                
                # Also store final evaluation for reference
                final_eval = results.get("final_evaluation", {})
                details["final_mean_reward"] = final_eval.get("mean_reward", None)
                details["best_eval_mean_reward"] = results.get("best_eval_mean_reward", None)
                
                return alg_score, details
            
            elif len(checkpoint_means) == 1:
                # Only one checkpoint - use it directly (no LCB for mean possible)
                single_mean = checkpoint_means[0]
                single_std = checkpoint_stds[0] if checkpoint_stds else 0.0
                n_eval = eval_summary.get("n_eval_episodes", args.n_eval_episodes)
                
                # Can still compute LCB for max using that checkpoint's std
                from scipy import stats
                if n_eval > 1 and single_std > 0:
                    alpha = 1 - args.algscore_confidence
                    t_val = stats.t.ppf(1 - alpha, df=n_eval - 1)
                    lcb_max = single_mean - t_val * (single_std / np.sqrt(n_eval))
                else:
                    lcb_max = single_mean
                
                alg_score = args.algscore_max_weight * lcb_max + args.algscore_mean_weight * single_mean
                details = {
                    "alg_score": alg_score,
                    "lcb_max": lcb_max,
                    "lcb_mean": single_mean,  # No CI possible with 1 checkpoint
                    "raw_max": single_mean,
                    "raw_mean": single_mean,
                    "superlative_checkpoint_std": single_std,
                    "n_checkpoints": 1,
                    "source": "single_checkpoint",
                    "json_file": files[0],
                    "warning": "Only 1 checkpoint - LCB_mean uses raw mean (no CI)"
                }
                return alg_score, details
    
    # Fallback: parse stdout for any reward values (limited - no std available)
    import re
    matches = re.findall(r"eval.*?mean.*?reward.*?(-?\d+\.?\d*)", stdout, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"mean.*?reward.*?(-?\d+\.?\d*)", stdout, re.IGNORECASE)
    
    if matches:
        rewards = [float(m) for m in matches]
        if len(rewards) >= 2:
            # Fallback without std data - use cross-checkpoint variance for both
            mean_val = np.mean(rewards)
            std_val = np.std(rewards, ddof=1)
            max_val = np.max(rewards)
            
            from scipy import stats
            alpha = 1 - args.algscore_confidence
            t_val = stats.t.ppf(1 - alpha, df=len(rewards) - 1)
            sem = std_val / np.sqrt(len(rewards))
            
            lcb_mean = mean_val - t_val * sem
            lcb_max = max_val - t_val * sem  # Conservative fallback
            
            alg_score = args.algscore_max_weight * lcb_max + args.algscore_mean_weight * lcb_mean
            details = {
                "alg_score": alg_score,
                "lcb_max": lcb_max,
                "lcb_mean": lcb_mean,
                "raw_max": max_val,
                "raw_mean": mean_val,
                "std_across_checkpoints": std_val,
                "n_checkpoints": len(rewards),
                "source": "stdout_fallback",
                "warning": "No std data - LCB_max uses cross-checkpoint variance"
            }
            return alg_score, details
    
    print("WARNING: Could not parse results for AlgScore")
    return -1000.0, {"error": "parse_failed", "source": "none"}


# =============================================================================
# OPTUNA STUDY MANAGEMENT
# =============================================================================

def create_study(args) -> optuna.Study:
    """Create or load Optuna study with SQLite parallel support."""
    
    sampler = TPESampler(
        n_startup_trials=args.n_startup_trials,
        seed=args.seed,
        multivariate=True,
    )
    
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.pruning_warmup_steps,
        interval_steps=1,
    )
    
    # Create storage with retry logic
    if args.storage_url.startswith("postgresql"):
        storage = optuna.storages.RDBStorage(
            url=args.storage_url,
            heartbeat_interval=60,
            grace_period=120,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        )
    else:
        # SQLite with extended timeout for parallel workers
        db_path = args.storage_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        storage = optuna.storages.RDBStorage(
            url=args.storage_url,
            engine_kwargs={
                "connect_args": {
                    "timeout": 60,      # Wait up to 60s for locks
                    "check_same_thread": False,
                },
                "pool_pre_ping": True,
            },
            heartbeat_interval=30,
            grace_period=120,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        )
        print(f"[Worker {args.worker_id}] Using SQLite with extended timeout (60s)")
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",  # Maximize AlgScore
        load_if_exists=True,
    )
    
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"[Worker {args.worker_id}] Joined study with {n_existing} existing trials")
    
    return study


def save_best_params(study: optuna.Study, output_dir: Path, args):
    """Save best hyperparameters to JSON."""
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials to save")
        return
    
    best_trial = study.best_trial
    best_params = study.best_params
    
    full_config = {
        "initial_learning_rate": best_params["initial_learning_rate"],
        "final_learning_rate": best_params["initial_learning_rate"] * 0.1,
        "initial_entropy_coef": best_params["initial_entropy_coef"],
        "final_entropy_coef": best_params["initial_entropy_coef"] * 0.5,
        "n_steps": best_params["n_steps"],
        "batch_size": best_params["batch_size"],
        "gamma": 0.99,  # Fixed value
        "gae_lambda": best_params["gae_lambda"],
        "clip_eps": best_params["clip_eps"],
        "n_epochs": best_params["n_epochs"],
        "vf_coef": best_params["vf_coef"],
    }
    
    # Get AlgScore details from best trial
    algscore_details = {k: v for k, v in best_trial.user_attrs.items()}
    
    output = {
        "best_algscore": study.best_value,
        "best_params_raw": best_params,
        "best_params_for_training": full_config,
        "algscore_breakdown": algscore_details,
        "algscore_config": {
            "max_weight": args.algscore_max_weight,
            "mean_weight": args.algscore_mean_weight,
            "confidence": args.algscore_confidence,
        },
        "n_trials_completed": len(completed),
        "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_trials_total": len(study.trials),
        "study_name": study.study_name,
        "timestamp": datetime.now().isoformat(),
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"best_hyperparams_{study.study_name}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"BEST HYPERPARAMETERS (AlgScore: {study.best_value:.2f})")
    print(f"{'='*70}")
    print(f"\nAlgScore Breakdown:")
    
    # LCB_max (60%) - superlative policy
    lcb_max = algscore_details.get('lcb_max')
    raw_max = algscore_details.get('raw_max')
    sup_std = algscore_details.get('superlative_checkpoint_std')
    sup_idx = algscore_details.get('superlative_checkpoint_idx')
    sup_ts = algscore_details.get('superlative_checkpoint_timestep')
    
    print(f"  LCB_max (60%):        {lcb_max:.2f}" if isinstance(lcb_max, (int, float)) else f"  LCB_max (60%):        N/A")
    print(f"    └─ raw_max:         {raw_max:.2f} ± {sup_std:.2f}" if isinstance(raw_max, (int, float)) else f"    └─ raw_max:         N/A")
    if sup_ts:
        print(f"    └─ superlative at:  checkpoint {sup_idx} ({sup_ts:,} timesteps)")
    
    # LCB_mean (40%) - training stability
    lcb_mean = algscore_details.get('lcb_mean')
    raw_mean = algscore_details.get('raw_mean')
    std_cp = algscore_details.get('std_across_checkpoints')
    n_cp = algscore_details.get('n_checkpoints')
    
    print(f"  LCB_mean (40%):       {lcb_mean:.2f}" if isinstance(lcb_mean, (int, float)) else f"  LCB_mean (40%):       N/A")
    print(f"    └─ raw_mean:        {raw_mean:.2f} ± {std_cp:.2f} (across {n_cp} checkpoints)" if isinstance(raw_mean, (int, float)) else f"    └─ raw_mean:        N/A")
    
    print(f"\nHyperparameters:")
    for k, v in full_config.items():
        if isinstance(v, float):
            print(f"  --{k.replace('_', '-')} {v:.8f}")
        else:
            print(f"  --{k.replace('_', '-')} {v}")
    print(f"\nSaved to: {output_file}")


def print_summary(study: optuna.Study, args):
    """Print optimization summary."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"\n{'='*70}")
    print(f"[Worker {args.worker_id}] OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(study.trials)} | Completed: {len(completed)} | Pruned: {len(pruned)}")
    
    if completed:
        scores = [t.value for t in completed if t.value is not None]
        if scores:
            print(f"AlgScore - Best: {max(scores):.2f}, Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")
            
            # Show top 3 trials
            sorted_trials = sorted(completed, key=lambda t: t.value if t.value else float('-inf'), reverse=True)
            print(f"\nTop 3 Trials:")
            for i, t in enumerate(sorted_trials[:3]):
                print(f"  {i+1}. Trial {t.number}: AlgScore={t.value:.2f}, "
                      f"lr={t.params.get('initial_learning_rate', 0):.2e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"B-ACE PPO HPO - Worker {args.worker_id}")
    print(f"{'='*70}")
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage_url[:50]}...")
    print(f"Timesteps/trial: {args.timesteps_per_trial:,} (~{args.timesteps_per_trial/10_000_000 * 2.5 * 60:.0f} min)")
    print(f"Max trials: {args.n_trials}, Timeout: {args.timeout_hours}h")
    print(f"\nScoring: AlgScore = {args.algscore_max_weight}×LCB_max + {args.algscore_mean_weight}×LCB_mean")
    print(f"         Confidence: {args.algscore_confidence*100:.0f}% (single-sided)")
    print(f"{'='*70}")
    
    study = create_study(args)
    
    try:
        study.optimize(
            lambda trial: run_training_trial(trial, suggest_hyperparameters(trial), args),
            n_trials=args.n_trials,
            timeout=args.timeout_hours * 3600,
            catch=(Exception,),
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print(f"\n[Worker {args.worker_id}] Interrupted - progress saved!")
    
    print_summary(study, args)
    
    # Only worker 0 saves final results (avoid race conditions)
    if args.worker_id == 0:
        save_best_params(study, Path(args.output_dir), args)


if __name__ == "__main__":
    main()
