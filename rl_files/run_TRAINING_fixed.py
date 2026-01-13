#!/usr/bin/env python3
"""
run_multiple_seeds.py

Python script to run training with multiple seeds sequentially.
FIXED: Better path resolution for finding training script

Usage:
    python run_multiple_seeds.py
    python run_multiple_seeds.py --seeds 42 123 456 789 1024
    python run_multiple_seeds.py --total-timesteps 200000
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os


def run_training_with_seed(seed, args):
    """
    Run training with a specific seed.
    
    Args:
        seed: Random seed to use
        args: Parsed command line arguments
    """
    print("=" * 60)
    print(f"Starting training with seed: {seed}")
    print("=" * 60)
    
    # Find the training script
    if args.training_script:
        training_script = Path(args.training_script).resolve()
    else:
        # Get the directory where this script is located
        script_dir = Path(__file__).resolve().parent
        
        # Try multiple locations
        candidates = [
            script_dir / "train_sb3_bace_with_decay_seeds.py",  # Same directory
            script_dir.parent / "train_sb3_bace_with_decay_seeds.py",  # Parent directory
            script_dir / "seed_experiments" / "train_sb3_bace_with_decay_seeds.py",  # Subdirectory
            Path.cwd() / "rl_files" / "train_sb3_bace_with_decay_seeds.py",  # From current working dir
            Path.cwd() / "train_sb3_bace_with_decay_seeds.py",  # Current working dir
        ]
        
        training_script = None
        for candidate in candidates:
            if candidate.exists():
                training_script = candidate
                print(f"✓ Found training script: {candidate}")
                break
        
        if training_script is None:
            print(f"ERROR: Cannot find train_sb3_bace_with_decay_seeds.py")
            print(f"Looked in:")
            for candidate in candidates:
                print(f"  - {candidate}")
            print(f"\nCurrent working directory: {Path.cwd()}")
            print(f"Script location: {script_dir}")
            print(f"\nTry one of these:")
            print(f"  1. Run from B-ACE directory: cd /Users/addison/Documents/GitHub/B-ACE")
            print(f"  2. Specify path: --training-script rl_files/train_sb3_bace_with_decay_seeds.py")
            print(f"  3. Check file exists: ls -la rl_files/train_sb3_bace_with_decay_seeds.py")
            return False
    
    if not training_script.exists():
        print(f"ERROR: Training script not found at: {training_script}")
        return False
    
    # Build the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(training_script),
        "--seed", str(seed),
        "--total-timesteps", str(args.total_timesteps),
        "--run-name", f"{args.experiment_name}_seed{seed}",
    ]
    
    # Add optional arguments
    if args.config:
        cmd.extend(["--config", args.config])
    
    if args.experiment_name:
        cmd.extend(["--experiment-name", args.experiment_name])

    if args.decay_clip_range:
        cmd.append("--decay-clip-range")
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    if args.initial_learning_rate:
        cmd.extend(["--initial-learning-rate", str(args.initial_learning_rate)])
    
    if args.final_learning_rate:
        cmd.extend(["--final-learning-rate", str(args.final_learning_rate)])
    
    if args.decay_schedule:
        cmd.extend(["--decay-schedule", args.decay_schedule])
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Seed {seed} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Seed {seed} failed with error code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠  Training interrupted by user\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run PPO training with multiple seeds sequentially"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1024],
        help="List of seeds to use (default: 42 123 456 789 1024)"
    )
    
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="Total timesteps per seed (default: 5,000,000)"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline_experiment",
        help="Base name for the experiment (default: baseline_experiment)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to B-ACE config file (optional)"
    )
    
    parser.add_argument(
        "--decay-clip-range",
        action="store_true",
        help="Enable clip range decay"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)"
    )
    
    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        default=None,
        help="Initial learning rate (optional)"
    )
    
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=None,
        help="Final learning rate (optional)"
    )
    
    parser.add_argument(
        "--decay-schedule",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="Decay schedule type (optional)"
    )
    
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop if any seed fails (default: continue with remaining seeds)"
    )
    
    parser.add_argument(
        "--training-script",
        type=str,
        default=None,
        help="Path to train_sb3_bace_with_decay_seeds.py (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("Running training with multiple seeds")
    print("=" * 60)
    print(f"Seeds: {args.seeds}")
    print(f"Total timesteps per seed: {args.total_timesteps:,}")
    print(f"Experiment name: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"Current directory: {Path.cwd()}")
    print("=" * 60)
    print()
    
    # Run training for each seed
    results = {}
    for seed in args.seeds:
        success = run_training_with_seed(seed, args)
        results[seed] = success
        
        if not success and args.stop_on_error:
            print("Stopping due to error (--stop-on-error flag set)")
            break
    
    # Print summary
    print("\n" + "=" * 60)
    print("All training runs complete!")
    print("=" * 60)
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nResults: {successful}/{total} seeds completed successfully")
    
    for seed, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} Seed {seed}")
    
    print("\n" + "=" * 60)
    print("To analyze results, run:")
    print(f"python plot_multi_seed_results.py --results-dir runs_sb3 --experiment-name {args.experiment_name}")
    print("=" * 60)
    print()
    
    # Exit with error code if any failed
    if successful < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
