#!/usr/bin/env python3
"""
optuna_export_top_trials.py

Export and analyze the best N trials from an Optuna study.

Usage:
    # Export top 25 trials to CSV and JSON
    python optuna_export_top_trials.py --n-best 25
    
    # Export top 10 with detailed analysis
    python optuna_export_top_trials.py --n-best 10 --detailed
    
    # Export to specific directory
    python optuna_export_top_trials.py --n-best 25 --output-dir results/
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent.parent  # Go up to B-ACE root


def export_top_trials(
    study_name: str,
    storage: str,
    n_best: int = 50,
    output_dir: Path = None,
    detailed: bool = False
):
    """
    Export top N trials from Optuna study to CSV, JSON, and create visualizations.
    
    Args:
        study_name: Name of the Optuna study
        storage: Storage URL
        n_best: Number of top trials to export
        output_dir: Directory to save outputs
        detailed: Whether to create detailed analysis plots
    """
    
    # Load study
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Create output directory
    if output_dir is None:
        output_dir = REPO_ROOT / "optuna_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Exporting Top {n_best} Trials from Study: {study_name}")
    print(f"{'='*70}\n")
    
    # Get all trials as DataFrame
    df_all = study.trials_dataframe()
    
    # Filter completed trials and sort by value
    df_complete = df_all[df_all['state'] == 'COMPLETE'].copy()
    df_complete = df_complete.sort_values('value', ascending=False)
    
    if len(df_complete) < n_best:
        print(f"Warning: Only {len(df_complete)} completed trials available (requested {n_best})")
        n_best = len(df_complete)
    
    # Get top N trials
    df_top = df_complete.head(n_best)
    
    print(f"Found {len(df_complete)} completed trials")
    print(f"Exporting top {n_best} trials\n")
    
    # ==========================================
    # 1. Export to CSV
    # ==========================================
    csv_file = output_dir / f"top_{n_best}_trials.csv"
    df_top.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV: {csv_file}")
    
    # ==========================================
    # 2. Export to JSON (detailed)
    # ==========================================
    json_data = []
    for idx, row in df_top.iterrows():
        trial_num = int(row['number'])
        trial = study.trials[trial_num]
        
        trial_info = {
            'trial_number': trial_num,
            'mean_score': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
            'state': str(trial.state),
            'datetime_start': str(trial.datetime_start),
            'datetime_complete': str(trial.datetime_complete),
            'duration_seconds': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None,
        }
        json_data.append(trial_info)
    
    json_file = output_dir / f"top_{n_best}_trials.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved JSON: {json_file}")
    
    # ==========================================
    # 3. Create summary statistics
    # ==========================================
    summary = {}
    
    # Get parameter columns
    param_cols = [col for col in df_top.columns if col.startswith('params_')]
    
    # Statistics for each parameter
    summary['parameter_statistics'] = {}
    for col in param_cols:
        param_name = col.replace('params_', '')
        values = df_top[col].dropna()
        
        if len(values) == 0:
            continue
        
        # Check if numeric or categorical
        if pd.api.types.is_numeric_dtype(values):
            summary['parameter_statistics'][param_name] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75)),
            }
        else:
            # Categorical parameter
            value_counts = values.value_counts().to_dict()
            summary['parameter_statistics'][param_name] = {
                'mode': values.mode()[0] if len(values.mode()) > 0 else None,
                'value_counts': {str(k): int(v) for k, v in value_counts.items()},
            }
    
    # Overall performance stats
    summary['performance'] = {
        'mean_reward': float(df_top['value'].mean()),
        'std_reward': float(df_top['value'].std()),
        'min_reward': float(df_top['value'].min()),
        'max_reward': float(df_top['value'].max()),
        'median_reward': float(df_top['value'].median()),
    }
    
    # Best trial
    best_trial = df_top.iloc[0]
    summary['best_trial'] = {
        'trial_number': int(best_trial['number']),
        'value': float(best_trial['value']),
        'params': {col.replace('params_', ''): best_trial[col] 
                   for col in param_cols if pd.notna(best_trial[col])}
    }
    
    summary_file = output_dir / f"top_{n_best}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved Summary: {summary_file}")
    
    # ==========================================
    # 4. Create readable summary report
    # ==========================================
    report_file = output_dir / f"top_{n_best}_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"Top {n_best} Trials Analysis Report\n")
        f.write(f"Study: {study_name}\n")
        f.write(f"{'='*70}\n\n")
        
        # Performance summary
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Reward:   {summary['performance']['mean_reward']:.4f}\n")
        f.write(f"Std Reward:    {summary['performance']['std_reward']:.4f}\n")
        f.write(f"Min Reward:    {summary['performance']['min_reward']:.4f}\n")
        f.write(f"Max Reward:    {summary['performance']['max_reward']:.4f}\n")
        f.write(f"Median Reward: {summary['performance']['median_reward']:.4f}\n\n")
        
        # Best trial
        f.write("BEST TRIAL\n")
        f.write("-" * 70 + "\n")
        f.write(f"Trial Number: {summary['best_trial']['trial_number']}\n")
        f.write(f"Value: {summary['best_trial']['value']:.4f}\n")
        f.write("Parameters:\n")
        for param, value in summary['best_trial']['params'].items():
            if isinstance(value, float):
                f.write(f"  {param:20s}: {value:.6f}\n")
            else:
                f.write(f"  {param:20s}: {value}\n")
        f.write("\n")
        
        # Parameter statistics
        f.write("PARAMETER RANGES IN TOP TRIALS\n")
        f.write("-" * 70 + "\n")
        for param, stats in summary['parameter_statistics'].items():
            f.write(f"\n{param}:\n")
            if 'mean' in stats:
                # Numeric parameter
                f.write(f"  Range:  [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"  Mean:   {stats['mean']:.6f}\n")
                f.write(f"  Median: {stats['median']:.6f}\n")
                f.write(f"  Q1-Q3:  [{stats['q25']:.6f}, {stats['q75']:.6f}]\n")
                f.write(f"  Std:    {stats['std']:.6f}\n")
            else:
                # Categorical parameter
                f.write(f"  Mode: {stats['mode']}\n")
                f.write(f"  Distribution:\n")
                for val, count in stats['value_counts'].items():
                    pct = (count / n_best) * 100
                    f.write(f"    {val}: {count} ({pct:.1f}%)\n")
        
        # Top 10 trials
        f.write("\n" + "="*70 + "\n")
        f.write("TOP 10 TRIALS\n")
        f.write("="*70 + "\n\n")
        for idx, (_, row) in enumerate(df_top.head(10).iterrows(), 1):
            f.write(f"Rank {idx}: Trial {int(row['number'])} - Reward: {row['value']:.4f}\n")
            for col in param_cols:
                param_name = col.replace('params_', '')
                value = row[col]
                if pd.notna(value):
                    if isinstance(value, float):
                        f.write(f"  {param_name:20s}: {value:.6f}\n")
                    else:
                        f.write(f"  {param_name:20s}: {value}\n")
            f.write("\n")
    
    print(f"✓ Saved Report: {report_file}")
    
    # ==========================================
    # 5. Create visualizations
    # ==========================================
    if detailed:
        print(f"\nCreating detailed visualizations...")
        create_detailed_plots(df_top, param_cols, output_dir, n_best)
    
    print(f"\n{'='*70}")
    print(f"Export Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - top_{n_best}_trials.csv         (Spreadsheet data)")
    print(f"  - top_{n_best}_trials.json        (Detailed JSON)")
    print(f"  - top_{n_best}_summary.json       (Statistical summary)")
    print(f"  - top_{n_best}_report.txt         (Readable report)")
    if detailed:
        print(f"  - parameter_distributions.png   (Box plots)")
        print(f"  - parameter_vs_reward.png       (Scatter plots)")
        print(f"  - correlation_matrix.png        (Heatmap)")
    print(f"{'='*70}\n")


def create_detailed_plots(df_top, param_cols, output_dir, n_best):
    """Create detailed visualization plots."""
    
    # Extract numeric parameters only
    numeric_params = []
    for col in param_cols:
        if pd.api.types.is_numeric_dtype(df_top[col]):
            numeric_params.append(col)
    
    if len(numeric_params) == 0:
        print("  No numeric parameters to visualize")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # ==========================================
    # Plot 1: Parameter distributions (box plots)
    # ==========================================
    n_params = len(numeric_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_params):
        param_name = col.replace('params_', '')
        ax = axes[idx]
        
        values = df_top[col].dropna()
        ax.boxplot([values], labels=[param_name])
        ax.set_ylabel('Value')
        ax.set_title(f'{param_name} Distribution\n(Top {n_best} trials)')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numeric_params), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ parameter_distributions.png")
    
    # ==========================================
    # Plot 2: Parameter vs Reward (scatter plots)
    # ==========================================
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_params):
        param_name = col.replace('params_', '')
        ax = axes[idx]
        
        x = df_top[col].dropna()
        y = df_top.loc[x.index, 'value']
        
        ax.scatter(x, y, alpha=0.6, s=50)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Reward')
        ax.set_title(f'{param_name} vs Reward')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.5, linewidth=2)
    
    # Hide unused subplots
    for idx in range(len(numeric_params), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_vs_reward.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ parameter_vs_reward.png")
    
    # ==========================================
    # Plot 3: Correlation matrix
    # ==========================================
    if len(numeric_params) > 1:
        # Create correlation matrix
        corr_data = df_top[numeric_params + ['value']].copy()
        corr_data.columns = [col.replace('params_', '') for col in numeric_params] + ['reward']
        corr_matrix = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title(f'Parameter Correlations (Top {n_best} trials)')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ correlation_matrix.png")
    
    # ==========================================
    # Plot 4: Reward distribution
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    rewards = df_top['value']
    ax.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(rewards.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rewards.mean():.2f}')
    ax.axvline(rewards.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {rewards.median():.2f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Reward Distribution (Top {n_best} trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ reward_distribution.png")


def suggest_search_space_adjustments(study_name: str, storage: str, n_best: int = 25):
    """
    Analyze top trials and suggest adjustments to search space.
    """
    
    study = optuna.load_study(study_name=study_name, storage=storage)
    df_all = study.trials_dataframe()
    df_complete = df_all[df_all['state'] == 'COMPLETE'].copy()
    df_top = df_complete.sort_values('value', ascending=False).head(n_best)
    
    param_cols = [col for col in df_top.columns if col.startswith('params_')]
    
    print(f"\n{'='*70}")
    print(f"Search Space Adjustment Suggestions")
    print(f"Based on top {n_best} trials")
    print(f"{'='*70}\n")
    
    suggestions = []
    
    for col in param_cols:
        param_name = col.replace('params_', '')
        values = df_top[col].dropna()
        
        if len(values) == 0:
            continue
        
        if pd.api.types.is_numeric_dtype(values):
            # Numeric parameter
            min_val = values.min()
            max_val = values.max()
            mean_val = values.mean()
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            
            # Check if values cluster at boundaries
            full_range = max_val - min_val
            if full_range == 0:
                print(f"⚠️  {param_name}:")
                print(f"    All top trials use same value: {mean_val:.6f}")
                print(f"    → Consider fixing this parameter")
                suggestions.append(f"Fix {param_name} = {mean_val:.6f}")
            else:
                iqr = q75 - q25
                center = (q25 + q75) / 2
                
                print(f"📊 {param_name}:")
                print(f"    Range in top trials: [{min_val:.6f}, {max_val:.6f}]")
                print(f"    Mean: {mean_val:.6f}")
                print(f"    IQR: [{q25:.6f}, {q75:.6f}]")
                
                # Check if clustered
                concentration = iqr / full_range
                if concentration < 0.3:
                    print(f"    ✓ Values are concentrated (IQR = {concentration:.1%} of range)")
                    print(f"    → Consider narrowing search to [{q25:.6f}, {q75:.6f}]")
                    suggestions.append(f"Narrow {param_name} to [{q25:.6f}, {q75:.6f}]")
                else:
                    print(f"    ✓ Values are spread out")
                    print(f"    → Current range seems appropriate")
                
                print()
        else:
            # Categorical parameter
            value_counts = values.value_counts()
            mode = value_counts.index[0]
            mode_pct = (value_counts[mode] / len(values)) * 100
            
            print(f"📊 {param_name}:")
            print(f"    Distribution:")
            for val, count in value_counts.items():
                pct = (count / len(values)) * 100
                print(f"      {val}: {count} ({pct:.1f}%)")
            
            if mode_pct > 80:
                print(f"    ⚠️  {mode} appears in {mode_pct:.1f}% of top trials")
                print(f"    → Consider fixing to '{mode}' or removing other options")
                suggestions.append(f"Fix {param_name} = '{mode}'")
            elif mode_pct > 60:
                print(f"    ✓ {mode} is preferred ({mode_pct:.1f}%)")
            else:
                print(f"    ✓ No strong preference")
            
            print()
    
    if suggestions:
        print(f"\n{'='*70}")
        print(f"RECOMMENDED ADJUSTMENTS:")
        print(f"{'='*70}")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Export and analyze top Optuna trials")
    #parser.add_argument('--study-name', type=str, default='bace_ppo_optimization',
    #                   help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                       help='Storage URL (default: sqlite:///optuna_studies.db)')
    parser.add_argument('--n-best', type=int, default=25,
                       help='Number of top trials to export')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: optuna_analysis/)')
    parser.add_argument('--detailed', action='store_true',
                       help='Create detailed plots')
    parser.add_argument('--suggest-adjustments', action='store_true',
                       help='Suggest search space adjustments')
    parser.add_argument('--study-name', type=str, default='bace_ppo_optimization_v3',
                       help='Name of the Optuna study')
    
    args = parser.parse_args()
    
    # Set default storage
    if args.storage is None:
        args.storage = f"sqlite:///{REPO_ROOT}/optuna_studies.db"
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = None
    
    # Export trials
    export_top_trials(
        study_name=args.study_name,
        storage=args.storage,
        n_best=args.n_best,
        output_dir=output_dir,
        detailed=args.detailed
    )
    
    # Suggest adjustments if requested
    if args.suggest_adjustments:
        suggest_search_space_adjustments(
            study_name=args.study_name,
            storage=args.storage,
            n_best=args.n_best
        )


if __name__ == "__main__":
    main()
