#!/usr/bin/env python3
"""
plot_eval_only_curve.py

Plot ONLY evaluation results from SB3 progress.csv:

- Connects each evaluation mean reward with a line
- Adds a 95% confidence interval band using eval/std_reward
  and a user-specified n_eval_episodes (default: 20)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


def plot_eval_only(
    progress_csv: Path,
    output_file: Path,
    title: str,
    n_eval_episodes: int = 20,
) -> None:
    if not progress_csv.exists():
        raise FileNotFoundError(f"progress.csv not found at: {progress_csv}")

    df = pd.read_csv(progress_csv)

    if "eval/mean_reward" not in df.columns:
        raise ValueError(
            f"'eval/mean_reward' column not found in {progress_csv}. "
            "Make sure your evaluation callback logs eval metrics."
        )

    # Keep only rows where an evaluation actually happened
    mask = df["eval/mean_reward"].notna()
    eval_df = df[mask].copy()

    if eval_df.empty:
        raise ValueError(
            "No evaluation rows found (eval/mean_reward is NaN everywhere)."
        )

    # X-axis: use total timesteps if available, otherwise index within eval_df
    if "time/total_timesteps" in eval_df.columns:
        x = eval_df["time/total_timesteps"].to_numpy()
    elif "timesteps" in eval_df.columns:
        x = eval_df["timesteps"].to_numpy()
    else:
        x = np.arange(len(eval_df))

    mean = eval_df["eval/mean_reward"].to_numpy()

    # Start plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean eval reward
    ax.plot(
        x,
        mean,
        "-o",
        ms=5,
        linewidth=2,
        label="Eval: mean_reward",
    )

    # 95% CI if we have std_reward and n_eval_episodes > 1
    if "eval/std_reward" in eval_df.columns and n_eval_episodes > 1:
        std = eval_df["eval/std_reward"].to_numpy()
        se = std / np.sqrt(n_eval_episodes)
        hw = t.ppf(1 - 0.05 / 2, n_eval_episodes - 1) * se

        ax.fill_between(
            x,
            mean - hw,
            mean + hw,
            alpha=0.2,
            label=f"95% CI (n={n_eval_episodes} episodes per eval)",
        )
    else:
        print(
            "eval/std_reward not found or n_eval_episodes <= 1; "
            "plotting eval mean without CI."
        )

    ax.set_xlabel("Total timesteps")
    ax.set_ylabel("Evaluation mean reward")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    print(f"\nSaved evaluation curve to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation-only learning curve with 95% CI from SB3 progress.csv."
    )
    parser.add_argument(
        "--progress-csv",
        type=str,
        required=True,
        help="Path to SB3 progress.csv file.",
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
        default="Evaluation Learning Curve",
        help="Plot title.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=20,
        help="Number of episodes per evaluation (for CI).",
    )

    args = parser.parse_args()

    progress_csv = Path(args.progress_csv).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    plot_eval_only(
        progress_csv=progress_csv,
        output_file=output_file,
        title=args.title,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()