#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiment.py

Multi-seed experiment runner for B-ACE RL training.
Runs training script multiple times with different seeds for statistical rigor.

Usage:
    # Run with default seeds (42, 123, 456, 789, 1011)
    python run_experiment.py --experiment-name baseline_v1
    
    # Run with custom seeds
    python run_experiment.py --experiment-name baseline_v1 --seeds 42 123 456
    
    # Pass additional training arguments
    python run_experiment.py --experiment-name enriched_v1 --use-enriched-obs --total-timesteps 5000000
    
    # Continue from a failed run (skip completed seeds)
    python run_experiment.py --experiment-name baseline_v1 --continue-from-checkpoint

Author: Ad (Thesis Experiments)
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any


# Default seeds for reproducibility (same as used in thesis)
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]


def find_repo_root() -> Path:
    """Find the B-ACE repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "b_ace_py").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent


def get_completed_seeds(experiment_dir: Path) -> List[int]:
    """Check which seeds have already completed training."""
    completed = []
    
    if not experiment_dir.exists():
        return completed
    
    for seed_dir in experiment_dir.glob("*seed*"):
        if not seed_dir.is_dir():
            continue
        
        # Check for training_results JSON file (indicates completion)
        json_files = list(seed_dir.glob("training_results_seed*.json"))
        if json_files:
            # Extract seed number from directory name
            import re
            match = re.search(r"seed(\d+)", seed_dir.name)
            if match:
                completed.append(int(match.group(1)))
    
    return completed


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    if td.days > 0:
        return f"{td.days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def run_training(
    training_script: Path,
    seed: int,
    experiment_name: str,
    extra_args: List[str],
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a single training run with the given seed.
    
    Returns:
        Dict with status, duration, and any error info
    """
    cmd = [
        sys.executable,
        str(training_script),
        "--seed", str(seed),
        "--experiment-name", experiment_name,
    ] + extra_args
    
    print(f"\n{'='*60}")
    print(f"  Running seed {seed}")
    print(f"  Command: {' '.join(cmd[:6])}...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            result = subprocess.run(cmd)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return {
                "status": "success",
                "seed": seed,
                "duration": duration,
                "returncode": 0,
            }
        else:
            return {
                "status": "failed",
                "seed": seed,
                "duration": duration,
                "returncode": result.returncode,
            }
            
    except KeyboardInterrupt:
        duration = time.time() - start_time
        return {
            "status": "interrupted",
            "seed": seed,
            "duration": duration,
            "returncode": -1,
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "status": "error",
            "seed": seed,
            "duration": duration,
            "error": str(e),
            "returncode": -1,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run B-ACE training with multiple seeds for statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default seeds
    python run_experiment.py --experiment-name baseline_1v1
    
    # Custom seeds
    python run_experiment.py --experiment-name test --seeds 42 123
    
    # With enriched observations
    python run_experiment.py --experiment-name enriched_1v1 --use-enriched-obs
    
    # Continue interrupted experiment
    python run_experiment.py --experiment-name baseline_1v1 --continue-from-checkpoint
    
    # Custom timesteps and config
    python run_experiment.py --experiment-name long_run --total-timesteps 10000000
        """
    )
    
    # Runner-specific arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name for this experiment (used for output directory)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"List of seeds to run (default: {DEFAULT_SEEDS})"
    )
    parser.add_argument(
        "--training-script",
        type=str,
        default=None,
        help="Path to training script (default: auto-detect train_bace_clean.py)"
    )
    parser.add_argument(
        "--continue-from-checkpoint",
        action="store_true",
        help="Skip seeds that have already completed"
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Log each seed's output to a separate file instead of console"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    
    # Parse known args, pass rest to training script
    args, extra_args = parser.parse_known_args()
    
    # Find repository root and training script
    repo_root = find_repo_root()
    
    if args.training_script:
        training_script = Path(args.training_script).resolve()
    else:
        # Try common locations
        candidates = [
            repo_root / "rl_files" / "train_bace_clean.py",
            repo_root / "train_bace_clean.py",
            Path(__file__).parent / "train_bace_clean.py",
        ]
        training_script = None
        for candidate in candidates:
            if candidate.exists():
                training_script = candidate
                break
        
        if training_script is None:
            print("ERROR: Could not find train_bace_clean.py")
            print("Searched in:")
            for c in candidates:
                print(f"  - {c}")
            print("\nPlease specify with --training-script")
            sys.exit(1)
    
    if not training_script.exists():
        print(f"ERROR: Training script not found: {training_script}")
        sys.exit(1)
    
    # Setup experiment directory
    experiment_dir = repo_root / "runs_sb3" / args.experiment_name
    
    # Check for completed seeds if continuing
    seeds_to_run = list(args.seeds)
    if args.continue_from_checkpoint:
        completed = get_completed_seeds(experiment_dir)
        if completed:
            print(f"Found {len(completed)} completed seeds: {completed}")
            seeds_to_run = [s for s in seeds_to_run if s not in completed]
            if not seeds_to_run:
                print("All seeds already completed!")
                sys.exit(0)
            print(f"Remaining seeds to run: {seeds_to_run}")
    
    # Print experiment summary
    print("\n" + "="*60)
    print("  B-ACE MULTI-SEED EXPERIMENT RUNNER")
    print("="*60)
    print(f"  Experiment:      {args.experiment_name}")
    print(f"  Training script: {training_script}")
    print(f"  Seeds:           {seeds_to_run}")
    print(f"  Output dir:      {experiment_dir}")
    print(f"  Extra args:      {extra_args if extra_args else '(none)'}")
    print("="*60 + "\n")
    
    if args.dry_run:
        print("DRY RUN - Commands that would be executed:\n")
        for seed in seeds_to_run:
            cmd = [
                sys.executable,
                str(training_script),
                "--seed", str(seed),
                "--experiment-name", args.experiment_name,
            ] + extra_args
            print(f"  {' '.join(cmd)}\n")
        sys.exit(0)
    
    # Create experiment directory
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config = {
        "experiment_name": args.experiment_name,
        "seeds": seeds_to_run,
        "extra_args": extra_args,
        "training_script": str(training_script),
        "start_time": datetime.now().isoformat(),
    }
    config_path = experiment_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training for each seed
    results = []
    total_start = time.time()
    
    for i, seed in enumerate(seeds_to_run):
        print(f"\n{'#'*60}")
        print(f"  SEED {i+1}/{len(seeds_to_run)}: {seed}")
        print(f"{'#'*60}")
        
        # Optional: log to file
        log_file = None
        if args.log_to_file:
            log_file = experiment_dir / f"seed{seed}_training.log"
            print(f"  Logging to: {log_file}")
        
        result = run_training(
            training_script=training_script,
            seed=seed,
            experiment_name=args.experiment_name,
            extra_args=extra_args,
            log_file=log_file,
        )
        results.append(result)
        
        # Print result
        status = result["status"]
        duration = format_duration(result["duration"])
        
        if status == "success":
            print(f"\n  [OK] Seed {seed} completed in {duration}")
        elif status == "interrupted":
            print(f"\n  [INTERRUPTED] Seed {seed} after {duration}")
            print("  Stopping experiment. Use --continue-from-checkpoint to resume.")
            break
        else:
            print(f"\n  [FAILED] Seed {seed} after {duration}")
            print(f"  Return code: {result.get('returncode', 'N/A')}")
            if "error" in result:
                print(f"  Error: {result['error']}")
        
        # Estimate remaining time
        if i < len(seeds_to_run) - 1:
            avg_duration = sum(r["duration"] for r in results) / len(results)
            remaining = avg_duration * (len(seeds_to_run) - i - 1)
            print(f"  Estimated remaining: {format_duration(remaining)}")
    
    # Final summary
    total_duration = time.time() - total_start
    
    print("\n" + "="*60)
    print("  EXPERIMENT SUMMARY")
    print("="*60)
    print(f"  Experiment:     {args.experiment_name}")
    print(f"  Total duration: {format_duration(total_duration)}")
    print(f"  Results:")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    interrupted = [r for r in results if r["status"] == "interrupted"]
    
    print(f"    - Successful: {len(successful)}/{len(seeds_to_run)}")
    if successful:
        for r in successful:
            print(f"        Seed {r['seed']}: {format_duration(r['duration'])}")
    
    if failed:
        print(f"    - Failed: {len(failed)}")
        for r in failed:
            error_msg = r.get('error', f"exit code {r.get('returncode', '?')}")
            print(f"        Seed {r['seed']}: {error_msg}")
    
    if interrupted:
        print(f"    - Interrupted: {len(interrupted)}")
    
    print(f"\n  Output directory: {experiment_dir}")
    print("="*60 + "\n")
    
    # Save results summary
    summary = {
        "experiment_name": args.experiment_name,
        "total_duration_seconds": total_duration,
        "seeds_requested": args.seeds,
        "seeds_run": seeds_to_run,
        "results": results,
        "successful": len(successful),
        "failed": len(failed),
        "interrupted": len(interrupted),
        "end_time": datetime.now().isoformat(),
    }
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Exit code based on results
    if len(successful) == len(seeds_to_run):
        sys.exit(0)
    elif interrupted:
        sys.exit(130)  # Standard interrupt exit code
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
