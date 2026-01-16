#!/usr/bin/env python3
"""
optuna_utils.py

Utility scripts for working with Optuna optimization results.

Usage:
    # Analyze study results
    python optuna_utils.py analyze --study-name bace_ppo_optimization

    # Visualize results
    python optuna_utils.py visualize --study-name bace_ppo_optimization

    # Generate training command with best params
    python optuna_utils.py generate-command --study-name bace_ppo_optimization

    # Train with best parameters
    python optuna_utils.py train-best --study-name bace_ppo_optimization
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd
import matplotlib.pyplot as plt


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent


def analyze_study(study_name: str, storage: str):
    """Print detailed analysis of optimization study."""
    
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    print(f"\n{'='*70}")
    print(f"Optuna Study Analysis: {study_name}")
    print(f"{'='*70}\n")
    
    # Basic statistics
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    # Best trial
    print(f"\n{'='*70}")
    print(f"Best Trial")
    print(f"{'='*70}")
    print(f"Trial number: {study.best_trial.number}")
    print(f"Value (mean reward): {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.6f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # Top 10 trials
    print(f"\n{'='*70}")
    print(f"Top 10 Trials")
    print(f"{'='*70}")
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE'].sort_values('value', ascending=False)
    
    print(f"\n{'Trial':<8} {'Value':<12} {'Initial LR':<12} {'Batch Size':<12} {'Epochs':<8}")
    print(f"{'-'*70}")
    for idx, row in df_complete.head(10).iterrows():
        trial_num = int(row['number'])
        value = row['value']
        init_lr = row.get('params_initial_lr', 'N/A')
        batch_size = row.get('params_batch_size', 'N/A')
        epochs = row.get('params_n_epochs', 'N/A')
        
        if isinstance(init_lr, float):
            init_lr_str = f"{init_lr:.2e}"
        else:
            init_lr_str = str(init_lr)
        
        print(f"{trial_num:<8} {value:<12.4f} {init_lr_str:<12} {batch_size!s:<12} {epochs!s:<8}")
    
    # Parameter importance (if available)
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n{'='*70}")
        print(f"Hyperparameter Importances")
        print(f"{'='*70}")
        for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param:20s}: {importance:.4f}")
    except Exception as e:
        print(f"\nCould not compute parameter importances: {e}")
    
    # Statistics
    print(f"\n{'='*70}")
    print(f"Performance Statistics")
    print(f"{'='*70}")
    values = [t.value for t in study.trials if t.value is not None]
    if values:
        print(f"Mean reward: {sum(values)/len(values):.4f}")
        print(f"Std reward: {pd.Series(values).std():.4f}")
        print(f"Min reward: {min(values):.4f}")
        print(f"Max reward: {max(values):.4f}")
    
    print(f"\n{'='*70}\n")


def visualize_study(study_name: str, storage: str, output_dir: Path = None):
    """Create visualizations of optimization study."""
    
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    if output_dir is None:
        output_dir = REPO_ROOT / "optuna_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating visualizations in {output_dir}...")
    
    # Optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        print("  ✓ optimization_history.html")
    except Exception as e:
        print(f"  ✗ Could not create optimization history: {e}")
    
    # Parameter importances
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))
        print("  ✓ param_importances.html")
    except Exception as e:
        print(f"  ✗ Could not create parameter importances: {e}")
    
    # Parallel coordinate
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))
        print("  ✓ parallel_coordinate.html")
    except Exception as e:
        print(f"  ✗ Could not create parallel coordinate: {e}")
    
    # Slice plot
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(str(output_dir / "slice_plot.html"))
        print("  ✓ slice_plot.html")
    except Exception as e:
        print(f"  ✗ Could not create slice plot: {e}")
    
    # Contour plot (for 2D relationships)
    try:
        fig = optuna.visualization.plot_contour(study, params=['initial_lr', 'initial_entropy'])
        fig.write_html(str(output_dir / "contour_lr_entropy.html"))
        print("  ✓ contour_lr_entropy.html")
    except Exception as e:
        print(f"  ✗ Could not create contour plot: {e}")
    
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"Open the .html files in a browser to view them.\n")


def generate_command(study_name: str, storage: str, output_file: Path = None):
    """Generate training command with best hyperparameters."""
    
    study = optuna.load_study(study_name=study_name, storage=storage)
    params = study.best_params
    
    # Build command
    cmd = "python train_sb3_bace_with_decay.py"
    
    if 'initial_lr' in params:
        cmd += f" --initial-learning-rate {params['initial_lr']:.6f}"
    if 'final_lr' in params:
        cmd += f" --final-learning-rate {params['final_lr']:.6f}"
    if 'initial_entropy' in params:
        cmd += f" --initial-entropy-coef {params['initial_entropy']:.6f}"
    if 'final_entropy' in params:
        cmd += f" --final-entropy-coef {params['final_entropy']:.6f}"
    # Handle constant clip range (new format)
    if 'clip_range' in params:
        cmd += f" --clip-eps {params['clip_range']:.2f}"
    # Legacy support for old initial/final clip format
    elif 'initial_clip' in params:
        cmd += f" --initial-clip-range {params['initial_clip']:.6f}"
        if 'final_clip' in params:
            cmd += f" --final-clip-range {params['final_clip']:.6f}"
    if 'vf_coef' in params:
        cmd += f" --vf-coef {params['vf_coef']:.6f}"
    if 'n_epochs' in params:
        cmd += f" --epochs {params['n_epochs']}"
    if 'batch_size' in params:
        cmd += f" --batch-size {params['batch_size']}"
    if 'decay_schedule' in params:
        cmd += f" --decay-schedule {params['decay_schedule']}"
    
    cmd += " --total-timesteps 5000000"
    cmd += " --run-name optimized_model"
    
    print(f"\n{'='*70}")
    print(f"Training Command with Best Hyperparameters")
    print(f"{'='*70}\n")
    print(cmd)
    print(f"\n{'='*70}\n")
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(cmd + '\n')
        print(f"Command saved to: {output_file}\n")
    
    return cmd


def train_best(study_name: str, storage: str, total_timesteps: int = 5000000):
    """Train model with best hyperparameters from study."""
    
    print(f"\n{'='*70}")
    print(f"Training with Best Hyperparameters from {study_name}")
    print(f"{'='*70}\n")
    
    # Generate command
    cmd = generate_command(study_name, storage)
    
    # Update timesteps
    cmd = cmd.replace("--total-timesteps 5000000", f"--total-timesteps {total_timesteps}")
    
    # Execute
    print("Starting training...\n")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("\n✓ Training completed successfully!")
    else:
        print(f"\n✗ Training failed with return code {result.returncode}")
    
    return result.returncode


def compare_trials(study_name: str, storage: str, trial_numbers: list):
    """Compare specific trials side by side."""
    
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    print(f"\n{'='*70}")
    print(f"Trial Comparison")
    print(f"{'='*70}\n")
    
    trials = [study.trials[n] for n in trial_numbers if n < len(study.trials)]
    
    if not trials:
        print("No valid trials found.")
        return
    
    # Get all parameter keys
    all_keys = set()
    for trial in trials:
        all_keys.update(trial.params.keys())
    all_keys = sorted(all_keys)
    
    # Print comparison table
    print(f"{'Parameter':<20}", end='')
    for i, trial in enumerate(trials):
        print(f"Trial {trial.number:<10}", end='')
    print()
    print('-' * (20 + 15 * len(trials)))
    
    print(f"{'Value (reward)':<20}", end='')
    for trial in trials:
        if trial.value is not None:
            print(f"{trial.value:<15.4f}", end='')
        else:
            print(f"{'N/A':<15}", end='')
    print()
    
    print('-' * (20 + 15 * len(trials)))
    
    for key in all_keys:
        print(f"{key:<20}", end='')
        for trial in trials:
            val = trial.params.get(key, 'N/A')
            if isinstance(val, float):
                print(f"{val:<15.6f}", end='')
            else:
                print(f"{str(val):<15}", end='')
        print()
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Optuna utilities for B-ACE optimization")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze study results')
    analyze_parser.add_argument('--study-name', type=str, default='bace_ppo_optimization')
    analyze_parser.add_argument('--storage', type=str, default=None)
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--study-name', type=str, default='bace_ppo_optimization')
    viz_parser.add_argument('--storage', type=str, default=None)
    viz_parser.add_argument('--output-dir', type=str, default=None)
    
    # Generate command
    gen_parser = subparsers.add_parser('generate-command', help='Generate training command')
    gen_parser.add_argument('--study-name', type=str, default='bace_ppo_optimization')
    gen_parser.add_argument('--storage', type=str, default=None)
    gen_parser.add_argument('--output-file', type=str, default=None)
    
    # Train with best
    train_parser = subparsers.add_parser('train-best', help='Train with best hyperparameters')
    train_parser.add_argument('--study-name', type=str, default='bace_ppo_optimization')
    train_parser.add_argument('--storage', type=str, default=None)
    train_parser.add_argument('--total-timesteps', type=int, default=5000000)
    
    # Compare trials
    compare_parser = subparsers.add_parser('compare', help='Compare specific trials')
    compare_parser.add_argument('--study-name', type=str, default='bace_ppo_optimization')
    compare_parser.add_argument('--storage', type=str, default=None)
    compare_parser.add_argument('trials', type=int, nargs='+', help='Trial numbers to compare')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Set default storage
    if args.storage is None:
        args.storage = f"sqlite:///{REPO_ROOT}/optuna_studies.db"
    
    # Execute command
    if args.command == 'analyze':
        analyze_study(args.study_name, args.storage)
    
    elif args.command == 'visualize':
        output_dir = Path(args.output_dir) if args.output_dir else None
        visualize_study(args.study_name, args.storage, output_dir)
    
    elif args.command == 'generate-command':
        output_file = Path(args.output_file) if args.output_file else None
        generate_command(args.study_name, args.storage, output_file)
    
    elif args.command == 'train-best':
        train_best(args.study_name, args.storage, args.total_timesteps)
    
    elif args.command == 'compare':
        compare_trials(args.study_name, args.storage, args.trials)


if __name__ == "__main__":
    main()
