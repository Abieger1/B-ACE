#!/usr/bin/env python3
"""
plot_multi_seed_results_simple.py

Simplified version that doesn't require scipy - uses normal approximation for confidence intervals.
This is appropriate when you have 5+ seeds.

Usage:
    python plot_multi_seed_results_simple.py --results-dir runs_sb3 --experiment-name baseline
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_results_from_directory(
    results_dir: Path,
    experiment_name: str,
    pattern: str = "training_results_seed*.json",
) -> List[Dict[str, Any]]:
    """Load all training result files from a directory for a specific experiment_name."""
    results: List[Dict[str, Any]] = []
    result_files = list(results_dir.rglob(pattern))

    if not result_files:
        print(f"Warning: No result files found matching pattern '{pattern}' in {results_dir}")
        return results

    print(f"Found {len(result_files)} result files (before filtering by experiment_name='{experiment_name}'):")

    for file_path in sorted(result_files):
        try:
            with open(file_path, "r") as f:
                result = json.load(f)

            config = result.get("config", {})
            exp_name = config.get("experiment_name", None)

            # Strictly filter by experiment_name
            if exp_name != experiment_name:
                # Uncomment this if you want to see what is being skipped:
                # print(f"  - Skipping {file_path.name}: experiment_name='{exp_name}' (wanted '{experiment_name}')")
                continue

            # Extract only what we need to reduce memory
            seed = config.get("seed", "unknown")
            mean_reward = result["metrics"].get("mean_reward", 0.0)

            print(f"  - {file_path.name} (seed={seed}, mean_reward={mean_reward:.2f})")

            # Downsample episode data if too large (keep every Nth point)
            episode_rewards = result["metrics"]["episode_rewards"]
            timesteps = result["metrics"]["timesteps"]

            max_points = 50000  # Increased from 5000 to preserve more detail
            if len(episode_rewards) > max_points:
                step = len(episode_rewards) // max_points
                episode_rewards = episode_rewards[::step]
                timesteps = timesteps[::step]
                print(
                    f"    Downsampled from {len(result['metrics']['episode_rewards'])} "
                    f"to {len(episode_rewards)} points"
                )

            # Create a lighter version with only needed data
            light_result = {
                "config": config,
                "metrics": {
                    "mean_reward": result["metrics"]["mean_reward"],
                    "std_reward": result["metrics"]["std_reward"],
                    "median_reward": result["metrics"]["median_reward"],
                    "min_reward": result["metrics"]["min_reward"],
                    "max_reward": result["metrics"]["max_reward"],
                    "total_episodes": result["metrics"]["total_episodes"],
                    "mean_episode_length": result["metrics"]["mean_episode_length"],
                    "episode_rewards": episode_rewards,
                    "timesteps": timesteps,
                },
            }
            results.append(light_result)

        except Exception as e:
            print(f"  - Error loading {file_path}: {e}")

    if not results:
        print(f"\nNo results matched experiment_name='{experiment_name}'.\n")

    return results

def compute_confidence_intervals_normal(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float, float, float]:
    """
    Compute confidence intervals using normal approximation.
    Good for n >= 5 samples.
    
    Returns: (mean, std, ci_lower, ci_upper, halfwidth)
    """
    n = len(data)
    if n < 2:
        return np.mean(data), 0.0, np.mean(data), np.mean(data), 0.0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    # Use z-score for normal distribution (1.96 for 95% CI)
    z_critical = 1.96
    
    halfwidth = z_critical * se
    ci_lower = mean - halfwidth
    ci_upper = mean + halfwidth
    
    return mean, std, ci_lower, ci_upper, halfwidth


def align_episode_data(results: List[Dict[str, Any]], window_size: int = 10) -> Dict[str, Any]:
    """Align episode data across seeds. window_size=1 means NO smoothing."""
    all_rewards_by_timestep = {}
    
    for result in results:
        rewards = result['metrics']['episode_rewards']
        timesteps = result['metrics']['timesteps']
        
        # CRITICAL FIX: Normalize timesteps to start at 0 for each seed
        # This handles cases where seeds were run sequentially with cumulative timesteps
        timesteps = np.array(timesteps)
        timesteps_normalized = timesteps - timesteps[0]  # Start each seed at timestep 0
        
        # Create rolling average
        if len(rewards) >= window_size:
            rewards_smooth = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values
        else:
            rewards_smooth = rewards
        
        # Store by approximate timestep (now normalized)
        for ts, r in zip(timesteps_normalized, rewards_smooth):
            #ts_key = (ts // 1000) * 1000  # Changed from 10000 to 1000 for finer granularity
            ts_key = ts
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
        mean, std, ci_lower, ci_upper, hw = compute_confidence_intervals_normal(rewards_at_ts)
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


def plot_learning_curves_with_ci(results: List[Dict[str, Any]], save_path: Path, window_size: int = 10):
    """Plot learning curves with 95% confidence intervals. window_size=1 means NO smoothing."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual seed runs (with normalized timesteps)
    for i, result in enumerate(results):
        seed = result['config']['seed']
        rewards = result['metrics']['episode_rewards']
        timesteps = np.array(result['metrics']['timesteps'])
        
        # Normalize timesteps to start at 0 for each seed
        timesteps_normalized = timesteps - timesteps[0]
        
        if len(rewards) >= window_size:
            rewards_smooth = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values
        else:
            rewards_smooth = rewards
        
        ax.plot(timesteps_normalized, rewards_smooth, alpha=0.3, linewidth=1, label=f'Seed {seed}')
    
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
    ax.set_title(f'Learning Curves Across {len(results)} Seeds (95% CI)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curve plot to: {save_path}")
    plt.close()


def plot_final_performance_comparison(results: List[Dict[str, Any]], save_path: Path):
    """Plot final performance comparison across seeds with error bars."""
    seeds = [r['config']['seed'] for r in results]
    mean_rewards = [r['metrics']['mean_reward'] for r in results]
    
    overall_mean, overall_std, ci_lower, ci_upper, halfwidth = compute_confidence_intervals_normal(np.array(mean_rewards))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(seeds)), mean_rewards, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.2f}')
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


def generate_summary_statistics(results: List[Dict[str, Any]], save_path: Path):
    """Generate and save summary statistics report."""
    if not results:
        print("No results to summarize.")
        return
    
    seeds = [r['config']['seed'] for r in results]
    mean_rewards = np.array([r['metrics']['mean_reward'] for r in results])
    std_rewards = np.array([r['metrics']['std_reward'] for r in results])
    median_rewards = np.array([r['metrics']['median_reward'] for r in results])
    total_episodes = np.array([r['metrics']['total_episodes'] for r in results])
    
    mean_ci = compute_confidence_intervals_normal(mean_rewards)
    median_ci = compute_confidence_intervals_normal(median_rewards)
    
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
        'training_config': results[0]['config'],
        'note': 'Confidence intervals computed using normal approximation (appropriate for n>=5 seeds)'
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS ({len(results)} seeds)")
    print(f"{'='*60}")
    print(f"\nFinal Performance:")
    print(f"  Mean Reward: {mean_ci[0]:.2f} ± {mean_ci[4]:.2f} (95% CI halfwidth)")
    print(f"  95% CI: [{mean_ci[2]:.2f}, {mean_ci[3]:.2f}]")
    print(f"\nIndividual Seed Results:")
    for seed, reward in zip(seeds, mean_rewards):
        print(f"  Seed {seed}: {reward:.2f}")
    print(f"\n{'='*60}")
    print(f"Summary saved to: {save_path}")
    print(f"{'='*60}\n")


def create_comparison_table(results: List[Dict[str, Any]], save_path: Path):
    """Create a comparison table of all seeds."""
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
    
    _, _, _, _, hw_mean = compute_confidence_intervals_normal(df['mean_reward'].values)
    _, _, _, _, hw_median = compute_confidence_intervals_normal(df['median_reward'].values)
    
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
    df.to_csv(save_path, index=False, float_format='%.2f')
    print(f"Saved comparison table to: {save_path}")
    print(f"\n{df.to_string(index=False, float_format='%.2f')}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze multi-seed training results (simplified - no scipy required)"
    )
    
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing training result files')
    parser.add_argument('--experiment-name', type=str, default='baseline',
                       help='Name for this experiment')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Directory to save analysis outputs')
    parser.add_argument('--window-size', type=int, default=50,
                       help='Rolling window size for smoothing')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    results = load_results_from_directory(results_dir, args.experiment_name)
    
    if len(results) < 1:
        print("Error: No results found.")
        sys.exit(1)
    
    if len(results) < 5:
        print(f"Warning: Only {len(results)} result(s) found. For best statistical confidence, use 5+ seeds.")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {len(results)} seed runs...")
    print(f"Output directory: {output_dir}")
    print("\nGenerating plots and statistics...")
    
    plot_learning_curves_with_ci(results, output_dir / f'{args.experiment_name}_learning_curves.png', 
                                 window_size=args.window_size)
    plot_final_performance_comparison(results, output_dir / f'{args.experiment_name}_performance_comparison.png')
    generate_summary_statistics(results, output_dir / f'{args.experiment_name}_summary.json')
    create_comparison_table(results, output_dir / f'{args.experiment_name}_comparison.csv')
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! All outputs saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
