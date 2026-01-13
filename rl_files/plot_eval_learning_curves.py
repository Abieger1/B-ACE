#!/usr/bin/env python3
"""
plot_eval_learning_curves.py

Build a SARSA(lambda)-style learning curve for PPO using ONLY evaluation
results from training_results_seed*.json produced by
train_sb3_bace_with_eval_checkpoints_FIXED.py.

Mapping to SARSA(Lambda) code:
- Z (number of replications)  <->  number of seeds
- milestones (episodes)       <->  evaluation timesteps
- TestEETDR[z, m]             <->  eval mean reward of seed z at milestone m

We then compute, exactly as in SARSA_Lambda.py:

    avgEval    = mean over seeds at each milestone
    avgSE      = std over seeds / sqrt(Z)
    avgHW      = t_{0.975, Z-1} * avgSE

and plot avgEval with a 95% CI band.

If only ONE seed is available, we fall back to using the standard deviation
across the evaluation episodes at each milestone, using:

    std_rewards[j] / sqrt(n_eval_episodes)

for milestone j.
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


# ---------------------------------------------------------------------
# Utilities to read training_results_seed*.json
# ---------------------------------------------------------------------

def find_results_files(experiment_dir: Path) -> Dict[str, Path]:
    """
    Find training_results_seed*.json files for each seed subdirectory.

    Expected structure:
        experiment_dir/
            seed42/
                training_results_seed42.json
            seed123/
                training_results_seed123.json
            ...
    """
    results_files: Dict[str, Path] = {}

    for seed_dir in experiment_dir.glob("*seed*"):
        if not seed_dir.is_dir():
            continue

        match = re.search(r"seed(\d+)", seed_dir.name)
        seed_num = match.group(1) if match else seed_dir.name
        seed_name = f"seed{seed_num}"

        candidates = sorted(seed_dir.glob("training_results_seed*.json"))
        if not candidates:
            print(f" No training_results_seed*.json found in {seed_dir}")
            continue

        json_path = candidates[-1]
        results_files[seed_name] = json_path
        print(f"Found results for {seed_name}: {json_path}")

    return results_files


def load_eval_series(
    json_path: Path,
) -> Tuple[List[int], List[float], Optional[List[float]], Optional[int]]:
    """
    Load evaluation information from a training_results_seed*.json file.

    Returns
    -------
    timesteps : list[int]
        Training steps where evaluations occurred.
    mean_rewards : list[float]
        Mean reward over episodes at each evaluation.
    std_rewards : list[float] or None
        Std dev of reward across episodes at each evaluation (if present).
    n_eval_episodes : int or None
        Number of episodes per evaluation (if present).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    eval_summary = data.get("evaluation_summary", {})
    timesteps = eval_summary.get("timesteps", [])
    mean_rewards = eval_summary.get("mean_rewards", [])
    std_rewards = eval_summary.get("std_rewards", None)
    n_eval_episodes = eval_summary.get("n_eval_episodes", None)

    if not timesteps or not mean_rewards:
        raise ValueError(
            f"{json_path} does not contain a populated 'evaluation_summary'."
        )

    if len(timesteps) != len(mean_rewards):
        raise ValueError(
            f"{json_path}: len(timesteps)={len(timesteps)} vs "
            f"len(mean_rewards)={len(mean_rewards)}"
        )

    if std_rewards is not None and len(std_rewards) != len(timesteps):
        raise ValueError(
            f"{json_path}: len(std_rewards)={len(std_rewards)} "
            f"but len(timesteps)={len(timesteps)}"
        )

    return (
        [int(t) for t in timesteps],
        [float(r) for r in mean_rewards],
        None if std_rewards is None else [float(s) for s in std_rewards],
        None if n_eval_episodes is None else int(n_eval_episodes),
    )


# ---------------------------------------------------------------------
# Plotting logic (mirrors SARSA_Lambda learning curve)
# ---------------------------------------------------------------------

def plot_eval_learning_curve(
    seed_data: Dict[str, Tuple[List[int], List[float], Optional[List[float]], Optional[int]]],
    output_file: Path,
    title: str,
    hyperparams: Optional[str] = None,
) -> None:
    """
    Build a SARSA-style learning curve from evaluation data.

    seed_data maps each seed to:
      (timesteps, mean_rewards, std_rewards, n_eval_episodes)
    hyperparams: optional string of hyperparameter values to display in subtitle
    """

    n_seeds = len(seed_data)
    if n_seeds == 0:
        raise ValueError("No seed data provided.")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Will be populated by either branch
    avgHW: Optional[np.ndarray] = None

    # -------------------------------------------------------------
    # 1) If we have multiple seeds, build a (Z, milestones) matrix
    #    and compute CI across seeds at each milestone (SARSA style).
    # -------------------------------------------------------------
    if n_seeds > 1:
        common_ts: Optional[List[int]] = None
        all_means = []

        for seed_name, (ts, means, stds, n_eval_eps) in sorted(seed_data.items()):
            if common_ts is None:
                common_ts = ts
            else:
                if ts != common_ts:
                    raise ValueError(
                        "Mismatch in evaluation timesteps between seeds. "
                        "All seeds must evaluate at the same steps to "
                        "build a SARSA-style CI.\n"
                        f"Reference: {common_ts}\n"
                        f"{seed_name}: {ts}"
                    )

            # Light individual seed curves
            ax.plot(
                ts,
                means,
                "-o",
                ms=3,
                linewidth=1,
                alpha=0.25,
                label=f"Seed {seed_name.replace('seed', '')}",
            )

            all_means.append(np.array(means, dtype=float))

        Z = n_seeds
        TestEval = np.stack(all_means, axis=0)  # shape: (Z, milestones)

        avgEval = np.mean(TestEval, axis=0)
        avgSE = np.std(TestEval, axis=0, ddof=1) / np.sqrt(Z)
        avgHW = t.ppf(1 - 0.05 / 2, Z - 1) * avgSE

        x_vals = np.array(common_ts, dtype=int)

        ax.plot(
            x_vals,
            avgEval,
            "-o",
            ms=5,
            mec="k",
            linewidth=2,
            color="black",
            label="Mean Evaluation Return",
            zorder=5,
        )
        ax.fill_between(
            x_vals,
            avgEval - avgHW,
            avgEval + avgHW,
            alpha=0.2,
            label="95% Confidence Interval (across seeds)",
            zorder=1,
        )

    # -------------------------------------------------------------
    # 2) If only ONE seed, fall back to CI across episodes at each
    #    evaluation (using std_rewards & n_eval_episodes).
    #    This still uses only evaluation data, not training.
    # -------------------------------------------------------------
    else:
        seed_name, (ts, means, stds, n_eval_eps) = next(iter(seed_data.items()))

        x_vals = np.array(ts, dtype=int)
        avgEval = np.array(means, dtype=float)

        ax.plot(
            x_vals,
            avgEval,
            "-o",
            ms=5,
            mec="k",
            linewidth=2,
            color="black",
            label="Mean Evaluation Return",
        )

        if stds is not None and n_eval_eps is not None and n_eval_eps > 1:
            std_arr = np.array(stds, dtype=float)
            SE = std_arr / np.sqrt(n_eval_eps)
            avgHW = t.ppf(1 - 0.05 / 2, n_eval_eps - 1) * SE

            ax.fill_between(
                x_vals,
                avgEval - avgHW,
                avgEval + avgHW,
                alpha=0.2,
                label=f"95% Confidence Interval (episodes, n={n_eval_eps})",
                zorder=1,
            )
        else:
            avgHW = None
            print(
                "  std_rewards or n_eval_episodes missing in JSON; "
                "cannot compute episode-based CI for single seed."
            )

    # -------------------------------------------------------------
    # Final plot formatting
    # -------------------------------------------------------------
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Mean Evaluation Return", fontsize=11)
    
    # Compute summary statistics for subtitle
    peak_idx = np.argmax(avgEval)
    peak_score = avgEval[peak_idx]
    avg_score = np.mean(avgEval)
    
    # Format peak and average with CIs if available
    if avgHW is not None:
        peak_hw = avgHW[peak_idx]
        avg_hw = np.mean(avgHW)
        stats_line = f"Peak: {peak_score:.2f} ± {peak_hw:.2f} | Avg: {avg_score:.2f} ± {avg_hw:.2f}"
    else:
        stats_line = f"Peak: {peak_score:.2f} | Avg: {avg_score:.2f}"
    
    # Build subtitle: hyperparams (wrapped) on top, stats below
    subtitle_lines = []
    if hyperparams:
        # Wrap hyperparams to ~100 chars per line
        wrapped = textwrap.fill(hyperparams, width=100)
        subtitle_lines.append(wrapped)
    subtitle_lines.append(stats_line)
    subtitle = "\n".join(subtitle_lines)
    
    # Calculate padding based on number of subtitle lines
    n_subtitle_lines = subtitle.count("\n") + 1
    title_pad = 20 + (n_subtitle_lines - 1) * 14
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=title_pad)
    ax.text(
        0.5, 1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="gray",
        linespacing=1.5,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    # Adjust top margin to accommodate subtitle
    fig.subplots_adjust(top=0.85 - (n_subtitle_lines - 1) * 0.04)
    fig.savefig(output_file, dpi=300)
    print(f"\n Saved evaluation learning curve to: {output_file}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SARSA(lambda)-style evaluation learning curves from "
            "training_results_seed*.json."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing seed subdirs (seedXX) with training_results_seed*.json",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="eval_learning_curve.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Agent Evaluation Learning Curve",
        help="Plot title.",
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=None,
        help="Hyperparameter string to display in subtitle (e.g., 'lr=3e-4, γ=0.99').",
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.exists():
        raise SystemExit(f"Experiment directory does not exist: {experiment_dir}")

    results_files = find_results_files(experiment_dir)
    if not results_files:
        raise SystemExit(
            f"No training_results_seed*.json files found under {experiment_dir}"
        )

    seed_data: Dict[str, Tuple[List[int], List[float], Optional[List[float]], Optional[int]]] = {}
    for seed_name, json_path in sorted(results_files.items()):
        ts, means, stds, n_eval_eps = load_eval_series(json_path)
        print(
            f"{seed_name}: {len(means)} evaluations, "
            f"steps {ts[0]:,} → {ts[-1]:,}, "
            f"mean eval reward = {np.mean(means):.2f} ± {np.std(means):.2f}"
        )
        seed_data[seed_name] = (ts, means, stds, n_eval_eps)

    plot_eval_learning_curve(
        seed_data=seed_data,
        output_file=Path(args.output_file),
        title=args.title,
        hyperparams=args.hyperparams,
    )


if __name__ == "__main__":
    main()