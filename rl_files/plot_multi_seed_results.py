#!/usr/bin/env python3
"""
plot_multi_seed_results.py

Analyze and visualize training results across multiple seeds with 95% confidence intervals.

This script:
1. Loads training results from multiple seed runs
2. Computes 95% confidence interval halfwidths
3. Creates comparison plots with confidence bands
4. Generates summary statistics

Usage:
    python plot_multi_seed_results.py --results-dir runs_sb3 --experiment-name baseline
    
    Or specify individual result files:
    python plot_multi_seed_results.py --result-files runs_sb3/run1/training_results_seed42.json runs_sb3/run2/training_results_seed123.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def load_results_from_directory(results_dir: Path, pattern: str = "training_results_seed*.json") -> List[Dict[str, Any]]:
    """
    Load all training result files from a directory.
    
    Args:
        results_dir: Directory containing training results
        pattern: Glob pattern to match result files
        
    Returns:
        List of loaded result dictionaries
    """
    results = []
    
    # Search recursively for result files
    result_files = list(results_dir.rglob(pattern))
    
    if not result_files:
        print(f"Warning: No result files found matching pattern '{pattern}' in {results_dir}")
        return results
    
    print(f"Found {len(result_files)} result files:")
    for file_path in sorted(result_files):
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                results.append(result)
                seed = result['config'].get('seed', 'unknown')
                mean_reward = result['metrics'].get('mean_reward', 0)
                print(f"  - {file_path.name} (seed={seed}, mean_reward={mean_reward:.2f})")
        except Exception as e:
            print(f"  - Error loading {file_path}: {e}")
    
    return results


def load_results_from_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load training results from specific file paths.
    
    Args:
        file_paths: List of paths to result JSON files
        
    Returns:
        List of loaded result dictionaries
    """
    results = []
    
    print(f"Loading {len(file_paths)} result files:")
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                results.append(result)
                seed = result['config'].get('seed', 'unknown')
                mean_reward = result['metrics'].get('mean_reward', 0)
                print(f"  - {Path(file_path).name} (seed={seed}, mean_reward={mean_reward:.2f})")
        except Exception as e:
            print(f"  - Error loading {file_path}: {e}")
    
    return results


def compute_confidence_intervals(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Compute confidence interval statistics.
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, std, ci_lower, ci_upper, halfwidth)
    """
    n = len(data)
    if n < 2:
        return np.mean(data), 0.0, np.mean(data), np.mean(data), 0.0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(n)  # Standard error
    
    # t-distribution for small samples
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    halfwidth = t_critical * se
    ci_lower = mean - halfwidth
    ci_upper = mean + halfwidth
    
    return mean, std, ci_lower, ci_upper, halfwidth


def align_episode_data(results: List[Dict[str, Any]], window_size: int = 100) -> Dict[str, Any]:
    """
    Align episode data across seeds using a rolling window approach.
    
    Args:
        results: List of result dictionaries
        window_size: Size of rolling window for smoothing
        
    Returns:
        Dictionary with aligned data
    """
    all_rewards_by_timestep = {}
    
    for result in results:
        rewards = result['metrics']['episode_rewards']
        timesteps = result['metrics']['timesteps']
        
        # Create rolling average
        if len(rewards) >= window_size:
            rewards_smooth = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values
        else:
            rewards_smooth = rewards
        
        # Store by approximate timestep
        for ts, r in zip(timesteps, rewards_smooth):
            # Round to nearest 10k for alignment
            ts_key = (ts // 10000) * 10000
            if ts_key not in all_rewards_by_timestep:
                all_rewards_by_timestep[ts_key] = []
            all_rewards_by_timestep[ts_key].append(r)
    
    # Compute statistics at each timestep
    timesteps_sorted = sorted(all_rewards_by_timestep.keys())
    means = []
    ci_lowers = []
    ci_uppers = []
    halfwidths = []
    
    for ts in timesteps_sorted:
        rewards_at_ts = np.array(all_rewards_by_timestep[ts])
        mean, std, ci_lower, ci_upper, hw = compute_confidence_intervals(rewards_at_ts)
        means.append(mean)
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
        halfwidths.append(hw)
    
    return {
        'timesteps': timesteps_sorted,
        'mean': means,
        'ci_lower': ci_lowers,
        'ci_upper': ci_uppers,
        'halfwidth': halfwidths
    }


def plot_learning_curves_with_ci(results: List[Dict[str, Any]], save_path: Path, window_size: int = 100):
    """
    Plot learning curves with 95% confidence intervals.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the plot
        window_size: Rolling window size for smoothing
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual seed runs (transparent)
    for i, result in enumerate(results):
        seed = result['config']['seed']
        rewards = result['metrics']['episode_rewards']
        timesteps = result['metrics']['timesteps']
        
        # Smooth with rolling average
        if len(rewards) >= window_size:
            rewards_smooth = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values
        else:
            rewards_smooth = rewards
        
        ax.plot(timesteps, rewards_smooth, alpha=0.3, linewidth=1, label=f'Seed {seed}')
    
    # Plot mean with confidence interval
    aligned_data = align_episode_data(results, window_size)
    
    ax.plot(aligned_data['timesteps'], aligned_data['mean'], 
            color='darkblue', linewidth=2.5, label='Mean across seeds', zorder=10)
    
    ax.fill_between(aligned_data['timesteps'], 
                     aligned_data['ci_lower'], 
                     aligned_data['ci_upper'],
                     color='blue', alpha=0.2, label='95% CI', zorder=5)
    
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Episode Reward (smoothed)', fontsize=12)
    ax.set_title(f'Learning Curves Across {len(results)} Seeds (95% Confidence Interval)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curve plot to: {save_path}")
    plt.close()


def plot_final_performance_comparison(results: List[Dict[str, Any]], save_path: Path):
    """
    Plot final performance comparison across seeds with error bars.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the plot
    """
    seeds = [r['config']['seed'] for r in results]
    mean_rewards = [r['metrics']['mean_reward'] for r in results]
    
    # Compute overall statistics
    overall_mean, overall_std, ci_lower, ci_upper, halfwidth = compute_confidence_intervals(np.array(mean_rewards))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with individual seeds
    bars = ax.bar(range(len(seeds)), mean_rewards, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add mean line
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.2f}')
    
    # Add confidence interval band
    ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='red', label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title(f'Final Performance Across Seeds (Halfwidth: ±{halfwidth:.2f})', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'{s}' for s in seeds])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance comparison plot to: {save_path}")
    plt.close()


def plot_convergence_analysis(results: List[Dict[str, Any]], save_path: Path, window_size: int = 100):
    """
    Plot convergence analysis showing reward progression.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the plot
        window_size: Rolling window for computing statistics
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Mean reward over time
    ax1 = axes[0]
    aligned_data = align_episode_data(results, window_size)
    
    ax1.plot(aligned_data['timesteps'], aligned_data['mean'], 
             color='darkgreen', linewidth=2, label='Mean Reward')
    ax1.fill_between(aligned_data['timesteps'], 
                      aligned_data['ci_lower'], 
                      aligned_data['ci_upper'],
                      color='green', alpha=0.2, label='95% CI')
    
    ax1.set_xlabel('Timesteps', fontsize=11)
    ax1.set_ylabel('Mean Episode Reward', fontsize=11)
    ax1.set_title('Reward Progression', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Confidence interval halfwidth over time
    ax2 = axes[1]
    ax2.plot(aligned_data['timesteps'], aligned_data['halfwidth'], 
             color='darkorange', linewidth=2, label='95% CI Halfwidth')
    ax2.fill_between(aligned_data['timesteps'], 0, aligned_data['halfwidth'],
                      color='orange', alpha=0.2)
    
    ax2.set_xlabel('Timesteps', fontsize=11)
    ax2.set_ylabel('95% CI Halfwidth', fontsize=11)
    ax2.set_title('Confidence Interval Progression (Uncertainty Over Time)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence analysis plot to: {save_path}")
    plt.close()


def generate_summary_statistics(results: List[Dict[str, Any]], save_path: Path):
    """
    Generate and save summary statistics report.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the summary
    """
    if not results:
        print("No results to summarize.")
        return
    
    # Extract final performance metrics
    seeds = [r['config']['seed'] for r in results]
    mean_rewards = np.array([r['metrics']['mean_reward'] for r in results])
    std_rewards = np.array([r['metrics']['std_reward'] for r in results])
    median_rewards = np.array([r['metrics']['median_reward'] for r in results])
    total_episodes = np.array([r['metrics']['total_episodes'] for r in results])
    
    # Compute confidence intervals
    mean_ci = compute_confidence_intervals(mean_rewards)
    median_ci = compute_confidence_intervals(median_rewards)
    
    # Create summary
    summary = {
        'experiment_overview': {
            'num_seeds': len(results),
            'seeds': seeds,
            'total_timesteps': results[0]['config']['total_timesteps'],
        },
        'final_performance': {
            'mean_reward': {
                'across_seeds_mean': float(mean_ci[0]),
                'across_seeds_std': float(mean_ci[1]),
                '95_ci_lower': float(mean_ci[2]),
                '95_ci_upper': float(mean_ci[3]),
                '95_ci_halfwidth': float(mean_ci[4]),
                'individual_seeds': [float(x) for x in mean_rewards]
            },
            'median_reward': {
                'across_seeds_mean': float(median_ci[0]),
                'across_seeds_std': float(median_ci[1]),
                '95_ci_lower': float(median_ci[2]),
                '95_ci_upper': float(median_ci[3]),
                '95_ci_halfwidth': float(median_ci[4]),
                'individual_seeds': [float(x) for x in median_rewards]
            },
            'total_episodes': {
                'mean': float(np.mean(total_episodes)),
                'std': float(np.std(total_episodes)),
                'individual_seeds': [int(x) for x in total_episodes]
            }
        },
        'training_config': results[0]['config']
    }
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS ({len(results)} seeds)")
    print(f"{'='*60}")
    print(f"\nFinal Performance:")
    print(f"  Mean Reward: {mean_ci[0]:.2f} ± {mean_ci[4]:.2f} (95% CI halfwidth)")
    print(f"  95% CI: [{mean_ci[2]:.2f}, {mean_ci[3]:.2f}]")
    print(f"  Standard Deviation: {mean_ci[1]:.2f}")
    print(f"\n  Median Reward: {median_ci[0]:.2f} ± {median_ci[4]:.2f} (95% CI halfwidth)")
    print(f"  95% CI: [{median_ci[2]:.2f}, {median_ci[3]:.2f}]")
    print(f"\nIndividual Seed Results:")
    for seed, reward in zip(seeds, mean_rewards):
        print(f"  Seed {seed}: {reward:.2f}")
    print(f"\n{'='*60}")
    print(f"Summary saved to: {save_path}")
    print(f"{'='*60}\n")


def create_comparison_table(results: List[Dict[str, Any]], save_path: Path):
    """
    Create a comparison table of all seeds.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the CSV table
    """
    data = []
    
    for result in results:
        row = {
            'seed': result['config']['seed'],
            'mean_reward': result['metrics']['mean_reward'],
            'std_reward': result['metrics']['std_reward'],
            'median_reward': result['metrics']['median_reward'],
            'min_reward': result['metrics']['min_reward'],
            'max_reward': result['metrics']['max_reward'],
            'total_episodes': result['metrics']['total_episodes'],
            'mean_episode_length': result['metrics']['mean_episode_length'],
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('seed')
    
    # Add summary row
    summary_row = {
        'seed': 'MEAN',
        'mean_reward': df['mean_reward'].mean(),
        'std_reward': df['std_reward'].mean(),
        'median_reward': df['median_reward'].mean(),
        'min_reward': df['min_reward'].mean(),
        'max_reward': df['max_reward'].mean(),
        'total_episodes': df['total_episodes'].mean(),
        'mean_episode_length': df['mean_episode_length'].mean(),
    }
    
    # Compute 95% CI halfwidths
    _, _, _, _, hw_mean = compute_confidence_intervals(df['mean_reward'].values)
    _, _, _, _, hw_median = compute_confidence_intervals(df['median_reward'].values)
    
    ci_row = {
        'seed': '95% CI HW',
        'mean_reward': hw_mean,
        'std_reward': np.nan,
        'median_reward': hw_median,
        'min_reward': np.nan,
        'max_reward': np.nan,
        'total_episodes': np.nan,
        'mean_episode_length': np.nan,
    }
    
    df = pd.concat([df, pd.DataFrame([summary_row, ci_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False, float_format='%.2f')
    print(f"Saved comparison table to: {save_path}")
    
    # Print table
    print(f"\n{df.to_string(index=False, float_format='%.2f')}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize multi-seed training results with 95% confidence intervals."
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing training result files (searches recursively)'
    )
    
    parser.add_argument(
        '--result-files',
        type=str,
        nargs='+',
        default=None,
        help='Specific result JSON files to analyze'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='baseline',
        help='Name for this experiment (used in output filenames)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results',
        help='Directory to save analysis outputs'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='Rolling window size for smoothing learning curves'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load results
    results = []
    
    if args.result_files:
        results = load_results_from_files(args.result_files)
    elif args.results_dir:
        results_dir = Path(args.results_dir).expanduser().resolve()
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)
        results = load_results_from_directory(results_dir)
    else:
        print("Error: Must specify either --results-dir or --result-files")
        sys.exit(1)
    
    if len(results) < 2:
        print(f"Warning: Only {len(results)} result(s) found. Need at least 2 for meaningful confidence intervals.")
        if len(results) == 0:
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {len(results)} seed runs...")
    print(f"Output directory: {output_dir}")
    
    # Generate all analyses
    print("\nGenerating plots and statistics...")
    
    # 1. Learning curves with CI
    plot_learning_curves_with_ci(
        results, 
        output_dir / f'{args.experiment_name}_learning_curves.png',
        window_size=args.window_size
    )
    
    # 2. Final performance comparison
    plot_final_performance_comparison(
        results,
        output_dir / f'{args.experiment_name}_performance_comparison.png'
    )
    
    # 3. Convergence analysis
    plot_convergence_analysis(
        results,
        output_dir / f'{args.experiment_name}_convergence_analysis.png',
        window_size=args.window_size
    )
    
    # 4. Summary statistics
    generate_summary_statistics(
        results,
        output_dir / f'{args.experiment_name}_summary.json'
    )
    
    # 5. Comparison table
    create_comparison_table(
        results,
        output_dir / f'{args.experiment_name}_comparison.csv'
    )
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! All outputs saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
