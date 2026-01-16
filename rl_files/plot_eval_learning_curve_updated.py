#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_eval_learning_curves.py

Plot evaluation learning curves from either:
1. training_results_seed*.json files (preferred)
2. TensorBoard events files (fallback)

Includes AUC and AlgScore calculation for RL benchmarking.
AlgScore uses Lower Confidence Bounds (LCB) to penalize high variance.
"""

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Plus-minus symbol for display
PM = "\u00b1"  # ±


def load_from_json(json_path: Path) -> Tuple[List[int], List[float], Optional[List[float]], Optional[int]]:
    """Load evaluation data from training_results_seed*.json."""
    with open(json_path, "r") as f:
        data = json.load(f)

    eval_summary = data.get("evaluation_summary", {})
    timesteps = eval_summary.get("timesteps", [])
    mean_rewards = eval_summary.get("mean_rewards", [])
    std_rewards = eval_summary.get("std_rewards", None)
    n_eval_episodes = eval_summary.get("n_eval_episodes", None)

    if not timesteps or not mean_rewards:
        raise ValueError(f"{json_path} missing evaluation data")

    return (
        [int(t) for t in timesteps],
        [float(r) for r in mean_rewards],
        [float(s) for s in std_rewards] if std_rewards else None,
        int(n_eval_episodes) if n_eval_episodes else None,
    )


def load_from_tensorboard(events_path: Path) -> Tuple[List[int], List[float], Optional[List[float]], Optional[int]]:
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
    
    # Assume 15 episodes if not specified (common default)
    n_eval_episodes = 15
    
    return (
        [int(t) for t in timesteps],
        [float(r) for r in mean_rewards],
        std_rewards,
        n_eval_episodes,
    )


def find_data_sources(experiment_dir: Path) -> Dict[str, Tuple[str, Path]]:
    """
    Find data sources (JSON or TensorBoard) for each seed.
    
    Returns dict: seed_name -> (source_type, path)
    """
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
    
    Where:
        Peak_LCB = Peak - HalfWidth (lower bound of 95% CI)
        AUC_LCB = AUC - HalfWidth (lower bound of 95% CI)
    
    Using LCB penalizes high variance, favoring consistent performance.
    
    Default: 0.6 * Peak_LCB + 0.4 * AUC_LCB
    """
    peak_lcb = peak - peak_hw
    auc_lcb = auc - auc_hw
    return peak_weight * peak_lcb + (1 - peak_weight) * auc_lcb


def plot_eval_learning_curve(
    seed_data: Dict[str, Tuple[List[int], List[float], Optional[List[float]], Optional[int]]],
    output_file: Path,
    title: str,
    hyperparams: Optional[str] = None,
) -> None:
    """Plot learning curve with AUC and AlgScore statistics."""
    
    n_seeds = len(seed_data)
    if n_seeds == 0:
        raise ValueError("No seed data provided.")

    fig, ax = plt.subplots(figsize=(10, 6))
    avgHW: Optional[np.ndarray] = None
    auc_values: List[float] = []
    peak_values: List[float] = []

    if n_seeds > 1:
        common_ts: Optional[List[int]] = None
        all_means = []

        for seed_name, (ts, means, stds, n_eval_eps) in sorted(seed_data.items()):
            if common_ts is None:
                common_ts = ts
            elif ts != common_ts:
                raise ValueError(f"Timestep mismatch: {seed_name} differs from others")

            ax.plot(ts, means, "-o", ms=3, linewidth=1, alpha=0.25,
                    label=f"Seed {seed_name.replace('seed', '')}")
            all_means.append(np.array(means, dtype=float))
            
            # Compute metrics for this seed (skip origin point if present)
            means_for_metrics = means[1:] if ts[0] == 0 else means
            seed_peak = float(np.max(means_for_metrics))
            seed_auc = float(np.mean(means_for_metrics))  # For evenly spaced evals, AUC = mean
            
            peak_values.append(seed_peak)
            auc_values.append(seed_auc)

        Z = n_seeds
        TestEval = np.stack(all_means, axis=0)
        avgEval = np.mean(TestEval, axis=0)
        avgSE = np.std(TestEval, axis=0, ddof=1) / np.sqrt(Z)
        avgHW = t.ppf(0.975, Z - 1) * avgSE
        x_vals = np.array(common_ts, dtype=int)

        ax.plot(x_vals, avgEval, "-o", ms=5, mec="k", linewidth=2, color="black",
                label="Mean Evaluation Return", zorder=5)
        ax.fill_between(x_vals, avgEval - avgHW, avgEval + avgHW, alpha=0.2,
                        label="95% Confidence Interval (across seeds)", zorder=1)
        
        # Compute statistics across seeds
        mean_peak = float(np.mean(peak_values))
        std_peak = float(np.std(peak_values, ddof=1))
        peak_se = std_peak / np.sqrt(Z)
        peak_hw = float(t.ppf(0.975, Z - 1) * peak_se)
        
        mean_auc = float(np.mean(auc_values))
        std_auc = float(np.std(auc_values, ddof=1))
        auc_se = std_auc / np.sqrt(Z)
        auc_hw = float(t.ppf(0.975, Z - 1) * auc_se)
        
        # Compute AlgScore using LCB
        mean_alg_score = compute_alg_score_lcb(mean_peak, mean_auc, peak_hw, auc_hw)
        
    else:
        # Single seed case - use episode-level variance for CIs
        seed_name, (ts, means, stds, n_eval_eps) = next(iter(seed_data.items()))
        x_vals = np.array(ts, dtype=int)
        avgEval = np.array(means, dtype=float)

        ax.plot(x_vals, avgEval, "-o", ms=5, mec="k", linewidth=2, color="black",
                label="Mean Evaluation Return")

        # Compute metrics for single seed (skip origin point if present)
        eval_for_metrics = avgEval[1:] if ts[0] == 0 else avgEval
        stds_for_metrics = np.array(stds[1:], dtype=float) if (stds is not None and ts[0] == 0) else (np.array(stds, dtype=float) if stds is not None else None)
        
        mean_peak = float(np.max(eval_for_metrics))
        mean_auc = float(np.mean(eval_for_metrics))
        peak_idx = int(np.argmax(eval_for_metrics))
        
        # Compute CIs from episode-level variance if available
        if stds is not None and n_eval_eps and n_eval_eps > 1:
            stds_arr = np.array(stds, dtype=float)
            
            # CI for shaded region (per-timestep) - uses all points for plotting
            SE = stds_arr / np.sqrt(n_eval_eps)
            avgHW = t.ppf(0.975, n_eval_eps - 1) * SE
            ax.fill_between(x_vals, avgEval - avgHW, avgEval + avgHW, alpha=0.2,
                            label=f"95% Confidence Interval (n={n_eval_eps} episodes)", zorder=1)
            
            # CI for Superlative Policy: use episode variance at the peak checkpoint
            # (stds_for_metrics aligns with eval_for_metrics which excludes origin)
            if stds_for_metrics is not None:
                peak_se = stds_for_metrics[peak_idx] / np.sqrt(n_eval_eps)
                peak_hw = float(t.ppf(0.975, n_eval_eps - 1) * peak_se)
                
                # CI for AUC: use pooled variance across all checkpoints (excluding origin)
                mean_se = float(np.mean(stds_for_metrics)) / np.sqrt(n_eval_eps)
                auc_hw = float(t.ppf(0.975, n_eval_eps - 1) * mean_se)
            else:
                peak_hw = 0.0
                auc_hw = 0.0
        else:
            # No episode-level variance available
            peak_hw = 0.0
            auc_hw = 0.0
        
        # Compute AlgScore using LCB
        mean_alg_score = compute_alg_score_lcb(mean_peak, mean_auc, peak_hw, auc_hw)

    # Formatting
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Mean Evaluation Return", fontsize=11)
    
    # Build stats line with proper plus-minus symbol
    if peak_hw > 0 or auc_hw > 0:
        stats_line = (f"Superlative Policy: {mean_peak:.2f} {PM} {peak_hw:.2f} | "
                      f"AUC: {mean_auc:.2f} {PM} {auc_hw:.2f}")
        alg_score_line = f"AlgScore (0.6*Peak_LCB + 0.4*AUC_LCB): {mean_alg_score:.2f}"
    else:
        stats_line = f"Superlative Policy: {mean_peak:.2f} | AUC: {mean_auc:.2f}"
        alg_score_line = f"AlgScore (0.6*Peak + 0.4*AUC): {mean_alg_score:.2f}"
    
    subtitle_lines = []
    if hyperparams:
        subtitle_lines.append(textwrap.fill(hyperparams, width=100))
    subtitle_lines.append(stats_line)
    subtitle_lines.append(alg_score_line)
    subtitle = "\n".join(subtitle_lines)
    
    n_lines = subtitle.count("\n") + 1
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20 + (n_lines - 1) * 14)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="gray", linespacing=1.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85 - (n_lines - 1) * 0.04)
    fig.savefig(output_file, dpi=300)
    print(f"\nSaved: {output_file}")
    
    # Print summary
    print(f"\nMetrics Summary:")
    if peak_hw > 0 or auc_hw > 0:
        print(f"   Superlative Policy: {mean_peak:.2f} {PM} {peak_hw:.2f} (95% CI)")
        print(f"   AUC: {mean_auc:.2f} {PM} {auc_hw:.2f} (95% CI)")
        print(f"   Peak LCB: {mean_peak - peak_hw:.2f}")
        print(f"   AUC LCB: {mean_auc - auc_hw:.2f}")
        print(f"   AlgScore (0.6*Peak_LCB + 0.4*AUC_LCB): {mean_alg_score:.2f}")
    else:
        print(f"   Superlative Policy: {mean_peak:.2f}")
        print(f"   AUC: {mean_auc:.2f}")
        print(f"   AlgScore: {mean_alg_score:.2f}")
        print(f"   (No variance data available for CI)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation learning curves")
    parser.add_argument("--experiment-dir", type=str, required=True,
                        help="Directory with seed subdirs")
    parser.add_argument("--output-file", type=str, default="eval_learning_curve.png")
    parser.add_argument("--title", type=str, default="Evaluation Learning Curve")
    parser.add_argument("--hyperparams", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds to include (e.g., seed123,seed456). If not specified, all seeds are included.")
    parser.add_argument("--start-at-origin", action="store_true", default=True,
                        help="Start the plot at (0, 0) (default: True)")
    parser.add_argument("--no-start-at-origin", action="store_false", dest="start_at_origin",
                        help="Don't prepend origin point, start at first evaluation")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.exists():
        raise SystemExit(f"Directory not found: {experiment_dir}")

    data_sources = find_data_sources(experiment_dir)
    if not data_sources:
        raise SystemExit(f"No data found in {experiment_dir}")

    # Filter to requested seeds if specified
    if args.seeds:
        requested_seeds = [s.strip() for s in args.seeds.split(',')]
        filtered_sources = {k: v for k, v in data_sources.items() if k in requested_seeds}
        
        # Check for seeds that weren't found
        found_seeds = set(filtered_sources.keys())
        missing_seeds = set(requested_seeds) - found_seeds
        if missing_seeds:
            print(f"Warning: Seeds not found: {missing_seeds}")
        
        if not filtered_sources:
            raise SystemExit(f"None of the requested seeds found: {requested_seeds}")
        
        data_sources = filtered_sources
        print(f"Filtered to seeds: {list(data_sources.keys())}")

    seed_data = {}
    for seed_name, (source_type, path) in sorted(data_sources.items()):
        try:
            if source_type == "json":
                data = load_from_json(path)
            else:
                data = load_from_tensorboard(path)
            
            ts, means, stds, n_eval = data
            
            # Prepend origin point (0, 0) so the curve starts at the origin
            if args.start_at_origin and ts[0] != 0:
                ts = [0] + ts
                means = [0.0] + means
                if stds is not None:
                    stds = [0.0] + stds
            
            print(f"  {seed_name}: {len(means)} evals, {ts[0]:,}->{ts[-1]:,}, "
                  f"mean={np.mean(means):.2f} std={np.std(means):.2f}")
            seed_data[seed_name] = (ts, means, stds, n_eval)
        except Exception as e:
            print(f"  Warning - Error loading {seed_name}: {e}")

    if not seed_data:
        raise SystemExit("No valid data loaded")

    plot_eval_learning_curve(
        seed_data=seed_data,
        output_file=Path(args.output_file),
        title=args.title,
        hyperparams=args.hyperparams,
    )


if __name__ == "__main__":
    main()
