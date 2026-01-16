#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_multiseed_learning_curve.py

Plot evaluation learning curves averaged across multiple seeds.
Shows mean performance with 95% confidence interval based on seed-to-seed variance.

Usage:
    python plot_multiseed_learning_curve.py \
        --experiment-dir runs_sb3/baseline_experiment_v1/ \
        --output-file baseline_learning_curve.png \
        --title "Baseline (1v1)"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Plus-minus symbol for display
PM = "\u00b1"


def load_from_json(json_path: Path) -> Tuple[List[int], List[float], Optional[List[float]]]:
    """Load evaluation data from training_results_seed*.json."""
    with open(json_path, "r") as f:
        data = json.load(f)

    eval_summary = data.get("evaluation_summary", {})
    timesteps = eval_summary.get("timesteps", [])
    mean_rewards = eval_summary.get("mean_rewards", [])
    std_rewards = eval_summary.get("std_rewards", None)

    if not timesteps or not mean_rewards:
        raise ValueError(f"{json_path} missing evaluation data")

    return (
        [int(t) for t in timesteps],
        [float(r) for r in mean_rewards],
        [float(s) for s in std_rewards] if std_rewards else None,
    )


def load_from_tensorboard(events_path: Path) -> Tuple[List[int], List[float], Optional[List[float]]]:
    """Load evaluation data from TensorBoard events file."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError("tensorboard not installed. Run: pip install tensorboard")
    
    ea = event_accumulator.EventAccumulator(str(events_path))
    ea.Reload()
    
    tags = ea.Tags().get('scalars', [])
    
    if 'eval/mean_reward' not in tags:
        raise ValueError(f"No eval/mean_reward in {events_path}")
    
    eval_mean = ea.Scalars('eval/mean_reward')
    timesteps = [e.step for e in eval_mean]
    mean_rewards = [e.value for e in eval_mean]
    
    std_rewards = None
    if 'eval/std_reward' in tags:
        eval_std = ea.Scalars('eval/std_reward')
        std_rewards = [e.value for e in eval_std]
    
    return (
        [int(t) for t in timesteps],
        [float(r) for r in mean_rewards],
        std_rewards,
    )


def find_data_sources(experiment_dir: Path) -> Dict[str, Tuple[str, Path]]:
    """Find data sources (JSON or TensorBoard) for each seed."""
    results: Dict[str, Tuple[str, Path]] = {}
    
    for seed_dir in sorted(experiment_dir.glob("*seed*")):
        if not seed_dir.is_dir():
            continue
        
        match = re.search(r"seed(\d+)", seed_dir.name)
        seed_name = f"seed{match.group(1)}" if match else seed_dir.name
        
        # Prefer JSON
        json_files = sorted(seed_dir.glob("training_results_seed*.json"))
        if json_files:
            results[seed_name] = ("json", json_files[-1])
            print(f"Found {seed_name}: {json_files[-1].name} (JSON)")
            continue
        
        # Fallback to TensorBoard
        tb_files = sorted(seed_dir.glob("events.out.tfevents.*"))
        if tb_files:
            results[seed_name] = ("tensorboard", tb_files[-1])
            print(f"Found {seed_name}: {tb_files[-1].name} (TensorBoard)")
            continue
        
        print(f"Warning: No data found for {seed_dir.name}")
    
    return results


def compute_alg_score_lcb(peak: float, auc: float, peak_hw: float, auc_hw: float, 
                          peak_weight: float = 0.6) -> float:
    """
    Compute AlgScore using Lower Confidence Bounds (LCB).
    
    AlgScore = peak_weight * Peak_LCB + (1 - peak_weight) * AUC_LCB
    """
    peak_lcb = peak - peak_hw
    auc_lcb = auc - auc_hw
    return peak_weight * peak_lcb + (1 - peak_weight) * auc_lcb


def plot_multiseed_learning_curve(
    experiment_dir: Path,
    output_file: Path,
    title: str,
    seeds_filter: Optional[List[str]] = None,
    start_at_origin: bool = True,
    show_individual_seeds: bool = True,
    interpolate_timesteps: bool = True,
) -> None:
    """
    Plot learning curve averaged across multiple seeds.
    
    Args:
        experiment_dir: Directory containing seed subdirectories
        output_file: Output path for the plot
        title: Plot title
        seeds_filter: Optional list of seeds to include (e.g., ['seed42', 'seed123'])
        start_at_origin: If True, prepend (0, 0) to the curve
        show_individual_seeds: If True, show faint lines for each seed
        interpolate_timesteps: If True, interpolate to common timesteps when seeds differ
    """
    # Find and load data
    data_sources = find_data_sources(experiment_dir)
    
    if seeds_filter:
        data_sources = {k: v for k, v in data_sources.items() if k in seeds_filter}
        if not data_sources:
            raise ValueError(f"None of requested seeds found: {seeds_filter}")
    
    if len(data_sources) < 2:
        raise ValueError(f"Need at least 2 seeds for multi-seed plot, found {len(data_sources)}")
    
    # Load all seed data (raw, before alignment)
    raw_seed_data: Dict[str, Tuple[List[int], List[float]]] = {}
    
    for seed_name, (source_type, path) in sorted(data_sources.items()):
        try:
            if source_type == "json":
                ts, means, _ = load_from_json(path)
            else:
                ts, means, _ = load_from_tensorboard(path)
            
            raw_seed_data[seed_name] = (ts, means)
            print(f"  Loaded {seed_name}: {len(means)} evals, range [{min(means):.2f}, {max(means):.2f}]")
            
        except Exception as e:
            print(f"  Warning - Error loading {seed_name}: {e}")
    
    if len(raw_seed_data) < 2:
        raise ValueError(f"Need at least 2 valid seeds, only loaded {len(raw_seed_data)}")
    
    # Check if timesteps match across all seeds
    all_timesteps = [ts for ts, _ in raw_seed_data.values()]
    timesteps_match = all(ts == all_timesteps[0] for ts in all_timesteps)
    
    if not timesteps_match:
        if interpolate_timesteps:
            # Find common timestep range and create uniform grid
            min_ts = max(ts[0] for ts, _ in raw_seed_data.values())
            max_ts = min(ts[-1] for ts, _ in raw_seed_data.values())
            n_points = min(len(ts) for ts, _ in raw_seed_data.values())
            
            # Use the first seed's timesteps as reference, or create uniform grid
            reference_ts = all_timesteps[0]
            # Filter to common range
            common_ts = [t for t in reference_ts if min_ts <= t <= max_ts]
            
            if len(common_ts) < 2:
                # Fall back to uniform grid
                common_ts = np.linspace(min_ts, max_ts, n_points).astype(int).tolist()
            
            print(f"\n  Interpolating to {len(common_ts)} common timesteps ({min_ts:,} to {max_ts:,})")
            
            # Interpolate each seed to common timesteps
            seed_data: Dict[str, Tuple[List[int], List[float]]] = {}
            for seed_name, (ts, means) in raw_seed_data.items():
                interp_means = np.interp(common_ts, ts, means).tolist()
                seed_data[seed_name] = (common_ts, interp_means)
        else:
            raise ValueError("Timestep mismatch across seeds. Use --interpolate to align them.")
    else:
        seed_data = raw_seed_data
        common_ts = all_timesteps[0]
    
    n_seeds = len(seed_data)
    print(f"\nPlotting {n_seeds} seeds")
    
    # Stack all seed curves into array
    all_means = np.array([means for _, (_, means) in sorted(seed_data.items())])
    x_vals = np.array(common_ts, dtype=int)
    
    # Prepend origin if requested
    if start_at_origin and x_vals[0] != 0:
        x_vals = np.concatenate([[0], x_vals])
        all_means = np.concatenate([np.zeros((n_seeds, 1)), all_means], axis=1)
    
    # Compute mean and 95% CI across seeds
    mean_curve = np.mean(all_means, axis=0)
    std_curve = np.std(all_means, axis=0, ddof=1)
    se_curve = std_curve / np.sqrt(n_seeds)
    
    # t-critical value for 95% CI with n_seeds-1 degrees of freedom
    t_crit = t.ppf(0.975, n_seeds - 1)
    ci_hw = t_crit * se_curve  # half-width of CI
    
    # Compute summary statistics (excluding origin point for metrics)
    metrics_start_idx = 1 if (start_at_origin and common_ts[0] != 0) else 0
    
    # Per-seed peak and AUC
    peak_values = [np.max(all_means[i, metrics_start_idx:]) for i in range(n_seeds)]
    auc_values = [np.mean(all_means[i, metrics_start_idx:]) for i in range(n_seeds)]
    
    mean_peak = np.mean(peak_values)
    std_peak = np.std(peak_values, ddof=1)
    peak_se = std_peak / np.sqrt(n_seeds)
    peak_ci = t_crit * peak_se
    
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values, ddof=1)
    auc_se = std_auc / np.sqrt(n_seeds)
    auc_ci = t_crit * auc_se
    
    # Compute AlgScore for each seed, then get mean and CI
    alg_scores = [0.6 * peak + 0.4 * auc for peak, auc in zip(peak_values, auc_values)]
    mean_alg_score = np.mean(alg_scores)
    std_alg_score = np.std(alg_scores, ddof=1)
    alg_score_se = std_alg_score / np.sqrt(n_seeds)
    alg_score_ci = t_crit * alg_score_se
    
    # Also compute LCB-based AlgScore
    alg_score_lcb = compute_alg_score_lcb(mean_peak, mean_auc, peak_ci, auc_ci)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual seeds (faint)
    if show_individual_seeds:
        for i, (seed_name, _) in enumerate(sorted(seed_data.items())):
            ax.plot(x_vals, all_means[i], '-', linewidth=1, alpha=0.3,
                    label=f"Seed {seed_name.replace('seed', '')}")
    
    # Plot mean curve
    ax.plot(x_vals, mean_curve, '-o', ms=4, linewidth=2, color='black',
            label='Mean across seeds', zorder=5)
    
    # Plot 95% CI shaded region
    ax.fill_between(x_vals, mean_curve - ci_hw, mean_curve + ci_hw,
                    alpha=0.25, color='blue',
                    label=f'95% CI (n={n_seeds} seeds)', zorder=2)
    
    # Formatting
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Mean Evaluation Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=55)
    
    # Add subtitle with statistics (two lines)
    stats_line1 = (f"Superlative Policy: {mean_peak:.2f} {PM} {peak_ci:.2f} | "
                   f"AUC: {mean_auc:.2f} {PM} {auc_ci:.2f} (95% CI, n={n_seeds} seeds)")
    stats_line2 = f"AlgScore: {mean_alg_score:.2f} {PM} {alg_score_ci:.2f} | AlgScore (LCB): {alg_score_lcb:.2f}"
    
    ax.text(0.5, 1.06, stats_line1, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=10, color='gray')
    ax.text(0.5, 1.02, stats_line2, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=10, color='gray')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(output_file, dpi=300)
    print(f"\nSaved: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  MULTI-SEED SUMMARY ({n_seeds} seeds)")
    print(f"{'='*60}")
    print(f"  Superlative Policy: {mean_peak:.2f} {PM} {peak_ci:.2f} (95% CI)")
    print(f"  AUC:                {mean_auc:.2f} {PM} {auc_ci:.2f} (95% CI)")
    print(f"  AlgScore:           {mean_alg_score:.2f} {PM} {alg_score_ci:.2f} (95% CI)")
    print(f"  AlgScore (LCB):     {alg_score_lcb:.2f}")
    print(f"\n  Per-seed peaks:     {[f'{p:.2f}' for p in peak_values]}")
    print(f"  Per-seed AUCs:      {[f'{a:.2f}' for a in auc_values]}")
    print(f"  Per-seed AlgScores: {[f'{s:.2f}' for s in alg_scores]}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learning curve averaged across multiple seeds with 95% CI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python plot_multiseed_learning_curve.py \\
        --experiment-dir runs_sb3/baseline_experiment_v1/ \\
        --output-file baseline_learning_curve.png \\
        --title "Baseline (1v1)"
    
    # Filter to specific seeds
    python plot_multiseed_learning_curve.py \\
        --experiment-dir runs_sb3/baseline_experiment_v1/ \\
        --seeds seed42,seed123,seed456 \\
        --output-file baseline_3seeds.png
    
    # Hide individual seed lines
    python plot_multiseed_learning_curve.py \\
        --experiment-dir runs_sb3/baseline_experiment_v1/ \\
        --no-show-seeds \\
        --output-file baseline_mean_only.png
        """
    )
    
    parser.add_argument("--experiment-dir", type=str, required=True,
                        help="Directory containing seed subdirectories")
    parser.add_argument("--output-file", type=str, default="multiseed_learning_curve.png",
                        help="Output filename for the plot")
    parser.add_argument("--title", type=str, default="Multi-Seed Learning Curve",
                        help="Plot title")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds to include (e.g., seed42,seed123)")
    parser.add_argument("--no-start-at-origin", action="store_true",
                        help="Don't prepend (0,0) origin point")
    parser.add_argument("--no-show-seeds", action="store_true",
                        help="Hide individual seed lines, show only mean + CI")
    parser.add_argument("--no-interpolate", action="store_true",
                        help="Don't interpolate timesteps (fail if seeds have different eval points)")
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.exists():
        raise SystemExit(f"Directory not found: {experiment_dir}")
    
    seeds_filter = None
    if args.seeds:
        seeds_filter = [s.strip() for s in args.seeds.split(',')]
    
    plot_multiseed_learning_curve(
        experiment_dir=experiment_dir,
        output_file=Path(args.output_file),
        title=args.title,
        seeds_filter=seeds_filter,
        start_at_origin=not args.no_start_at_origin,
        show_individual_seeds=not args.no_show_seeds,
        interpolate_timesteps=not args.no_interpolate,
    )


if __name__ == "__main__":
    main()
