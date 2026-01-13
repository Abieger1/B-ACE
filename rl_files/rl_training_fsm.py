#!/usr/bin/env python3
"""
train_sb3_bace_with_eval_checkpoints.py

Enhanced Stable-Baselines3 PPO training script for B-ACE with:
- Learning rate and entropy decay
- Multiple seed support
- EVALUATION-BASED checkpointing (NEW!)
- Runs 20 evaluation episodes every 250k steps
- Saves best model based on deterministic evaluation performance
- Metrics collection for confidence interval analysis
- JSON results export for later comparison
- ENRICHED OBSERVATIONS BY DEFAULT (47 dims with domain knowledge)

Key Features:
- Every 250k steps: Pauses training, runs 10 deterministic evaluation episodes
- Saves best model based on evaluation performance (not training rewards)
- Prevents saving models during catastrophic forgetting
- Standard RL research practice for reliable model selection

DEFAULT CONFIGURATION:
- Enriched observations (47 dims): Apollonius, BEZ, DMC, Multi-threat features
- Use --no-enriched-obs flag to get baseline (22 dims)

Usage:
    # Enriched observations (DEFAULT - 47 dims)
    python rl_files/train_sb3_bace_with_eval_checkpoints_fixed.py --seed 42
    
    # Baseline observations (22 dims)
    python train_sb3_bace_with_eval_checkpoints.py --no-enriched-obs --seed 42
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Tuple, Callable, List
import math

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from hierarchical_action_wrapper import HierarchicalActionWrapper, CurriculumConfig


DEBUG_HVAA_OBS = False

############################################
# Make sure we can import b_ace_py
############################################
THIS_FILE = Path(__file__).resolve()

# Find the B-ACE repo root by looking for b_ace_py directory
current_path = THIS_FILE.parent
while current_path != current_path.parent:  # Stop at filesystem root
    if (current_path / "b_ace_py").exists():
        REPO_ROOT = current_path
        break
    current_path = current_path.parent
else:
    # Fallback: assume script is two levels deep from root
    REPO_ROOT = THIS_FILE.parent.parent

if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

print(f"Using REPO_ROOT: {REPO_ROOT}")

from b_ace_py.utils import load_b_ace_config
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
from b_ace_py.enriched_observation_wrapper import EnrichedObservationWrapper
from b_ace_py.reward_shaping_wrapper import RewardShapingWrapper, RewardShapingConfig
try:
    from reward_component_eval_callback import RewardComponentEvalCallback
    COMPONENT_TRACKING_AVAILABLE = True
except ImportError:
    COMPONENT_TRACKING_AVAILABLE = False
    print("Note: RewardComponentEvalCallback not found. Using standard evaluation.")

############################################
# Learning Rate and Entropy Decay Functions
############################################

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate or entropy schedule.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        
    Returns:
        A function that takes progress (0 to 1) and returns the interpolated value
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        We want to go from initial_value to final_value.
        """
        return final_value + progress_remaining * (initial_value - final_value)
    
    return schedule


def exponential_schedule(initial_value: float, final_value: float, decay_rate: float = 0.1) -> Callable[[float], float]:
    """
    Exponential decay schedule (alternative to linear).
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        decay_rate: Controls how fast the decay happens
        
    Returns:
        A function that takes progress (0 to 1) and returns the exponentially decayed value
    """
    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1 to 0
        progress_made = 1 - progress_remaining
        decay_factor = np.exp(-decay_rate * progress_made * 10)  # Scale for smoother decay
        return final_value + (initial_value - final_value) * decay_factor
    
    return schedule


############################################
# Enhanced Callback for Metrics Collection
############################################

class MetricsCollectionCallback(BaseCallback):
    """
    Enhanced callback to:
    1. Update entropy coefficient during training
    2. Collect episode metrics for CI / learning curves (memory-efficient)
    3. Log schedules (lr, clip, entropy coef) only every metrics_collection_interval steps
    """
    def __init__(
        self,
        entropy_schedule=None,
        metrics_collection_interval: int = 1000,
        max_stored_episodes: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.entropy_schedule = entropy_schedule
        self.metrics_collection_interval = metrics_collection_interval
        self.max_stored_episodes = max_stored_episodes

        # Episode data (downsampled)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_history = []
        self.all_episode_rewards = []  # full list for final stats

        # Rolling statistics for “mean_reward_100ep”
        self.rolling_window = 100
        self.recent_rewards = []

    def _on_step(self) -> bool:
        # ---- 1) Update schedules every step ----
        # progress_remaining is used by SB3 schedules (1 -> 0)
        progress_remaining = 1.0 - (self.num_timesteps / self.model._total_timesteps)

        # Entropy coefficient schedule (update model every step)
        if self.entropy_schedule is not None:
            new_ent_coef = self.entropy_schedule(progress_remaining)
            self.model.ent_coef = new_ent_coef

        # ---- 2) Log schedules only every metrics_collection_interval steps ----
        if self.num_timesteps % self.metrics_collection_interval == 0:
            # Entropy coef (whatever it currently is)
            if self.entropy_schedule is not None:
                self.logger.record("train/entropy_coef", float(self.model.ent_coef))

            # Learning rate (if scheduled)
            if hasattr(self.model, "learning_rate"):
                lr = self.model.learning_rate
                if callable(lr):
                    current_lr = lr(progress_remaining)
                else:
                    current_lr = lr
                self.logger.record("train/learning_rate", float(current_lr))

            # Clip range (if scheduled)
            if hasattr(self.model, "clip_range"):
                clip = self.model.clip_range
                if callable(clip):
                    current_clip = clip(progress_remaining)
                else:
                    current_clip = clip
                self.logger.record("train/clip_range", float(current_clip))

        # ---- 3) Episode metrics from this step only ----
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                ep_info = info.get("episode")
                if ep_info is None:
                    continue

                reward = ep_info.get("r", None)
                length = ep_info.get("l", None)
                if reward is None or length is None:
                    continue

                # Track ALL rewards for final statistics
                self.all_episode_rewards.append(reward)

                # Store for learning curves
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.timesteps_history.append(self.num_timesteps)

                # Downsample if we exceed max_stored_episodes
                if len(self.episode_rewards) > self.max_stored_episodes * 1.1:
                    current_size = len(self.episode_rewards)
                    indices = np.linspace(
                        0, current_size - 1, self.max_stored_episodes, dtype=int
                    )

                    self.episode_rewards = [self.episode_rewards[i] for i in indices]
                    self.episode_lengths = [self.episode_lengths[i] for i in indices]
                    self.timesteps_history = [self.timesteps_history[i] for i in indices]

                # Update rolling window for reward stats
                self.recent_rewards.append(reward)
                if len(self.recent_rewards) > self.rolling_window:
                    self.recent_rewards.pop(0)

                # Log rolling stats (per finished episode, not per step)
                if len(self.recent_rewards) >= 10:
                    self.logger.record(
                        "rollout/mean_reward_100ep", np.mean(self.recent_rewards)
                    )
                    self.logger.record(
                        "rollout/std_reward_100ep", np.std(self.recent_rewards)
                    )

        return True

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the entire training run.
        Uses all_episode_rewards for accurate statistics, but only saves limited episode data.
        """
        if len(self.all_episode_rewards) == 0:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_episode_length": 0.0,
                "total_episodes": 0,
                "episode_rewards": [],
                "episode_lengths": [],
                "timesteps": [],
            }

        all_rewards_array = np.array(self.all_episode_rewards)

        return {
            "mean_reward": float(np.mean(all_rewards_array)),
            "std_reward": float(np.std(all_rewards_array)),
            "median_reward": float(np.median(all_rewards_array)),
            "min_reward": float(np.min(all_rewards_array)),
            "max_reward": float(np.max(all_rewards_array)),
            "total_episodes": len(self.all_episode_rewards),
            "mean_episode_length": float(np.mean(self.episode_lengths))
            if self.episode_lengths
            else 0.0,
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_lengths": [int(l) for l in self.episode_lengths],
            "timesteps": [int(t) for t in self.timesteps_history],
            "note": (
                f"Learning curve data downsampled to {len(self.episode_rewards)} "
                f"points from {len(self.all_episode_rewards)} total episodes for file size"
            ),
        }

############################################
# Evaluation-Based Checkpoint Callback
############################################

class EvaluationBasedCheckpointCallback(BaseCallback):
    """
    Callback for saving the best model based on ACTUAL EVALUATION performance.
    
    Every eval_freq timesteps:
    1. Pauses training
    2. Runs n_eval_episodes in deterministic mode
    3. Computes mean reward
    4. Saves checkpoint if it's the best so far
    
    This is more reliable than using training episode rewards because:
    - Evaluation uses deterministic actions (no exploration noise)
    - Prevents catastrophic forgetting from being saved
    - Standard practice in RL research
    """
    def __init__(
        self,
        eval_env: DummyVecEnv,
        eval_freq: int = 50000,
        n_eval_episodes: int = 20,
        log_dir: Path = None,
        verbose: int = 1,
        deterministic: bool = True,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.deterministic = deterministic
        
        # Best model tracking
        self.best_mean_reward = -np.inf
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []
        self.evaluations_std = []
        
        # Create eval checkpoints directory
        self.eval_checkpoint_dir = self.log_dir / "eval_checkpoints"
        self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last evaluation
        self.last_eval_step = 0
        
    def _evaluate_policy(self) -> Tuple[float, float, List[float]]:
        """
        Run n_eval_episodes and return mean reward, std reward, and episode rewards.
        """
        episode_rewards = []
        episode_lengths = []
        
        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        episodes_completed = 0
        
        while episodes_completed < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, info = self.eval_env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0]:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episodes_completed += 1
                
                episode_reward = 0.0
                episode_length = 0
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        return mean_reward, std_reward, mean_length, episode_rewards

    def _on_training_start(self) -> None:
        """
        Run an initial evaluation at timestep 0 so the learning curve starts at 0.
        """
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔍 INITIAL EVALUATION at 0 steps")
            print(f"   Running {self.n_eval_episodes} episodes...")
            print(f"{'='*60}")

        eval_start_time = time.time()
        mean_reward, std_reward, mean_length, episode_rewards = self._evaluate_policy()
        eval_time = time.time() - eval_start_time

        # Store evaluation results (timestep 0)
        self.evaluations_timesteps.append(0)
        self.evaluations_results.append(mean_reward)
        self.evaluations_length.append(mean_length)
        self.evaluations_std.append(std_reward)

        # Log to tensorboard
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)

        if self.verbose > 0:
            print(f" Initial Evaluation Results:")
            print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   Mean Length: {mean_length:.1f}")
            print(f"   Time: {eval_time:.1f}s")

        # Save checkpoint for this evaluation (optional but consistent)
        checkpoint_path = self.eval_checkpoint_dir / "model_0_steps"
        self.model.save(checkpoint_path.as_posix())

        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            norm_path = self.eval_checkpoint_dir / "vecnormalize_0_steps.pkl"
            vec_env.save(norm_path.as_posix())

        if self.verbose > 0:
            print(" Initial checkpoint saved")

        # Best-model tracking: treat this as the current best if it beats -inf
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

            best_model_path = self.log_dir / "best_model"
            self.model.save(best_model_path.as_posix())

            if isinstance(vec_env, VecNormalize):
                best_norm_path = self.log_dir / "best_vecnormalize.pkl"
                vec_env.save(best_norm_path.as_posix())

            best_info = {
                "timestep": 0,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "mean_length": float(mean_length),
                "episode_rewards": [float(r) for r in episode_rewards],
            }
            best_info_path = self.log_dir / "best_model_info.json"
            with open(best_info_path, "w") as f:
                json.dump(best_info, f, indent=2)

            if self.verbose > 0:
                print(f" INITIAL BEST MODEL saved at timestep 0")
                print(f"{'='*60}\n")        


    def _on_step(self) -> bool:
        # Check if we should evaluate
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            # Run evaluation
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"🔍 EVALUATION at {self.num_timesteps:,} steps")
                print(f"   Running {self.n_eval_episodes} episodes...")
                print(f"{'='*60}")
            
            eval_start_time = time.time()
            mean_reward, std_reward, mean_length, episode_rewards = self._evaluate_policy()
            eval_time = time.time() - eval_start_time
            
            # Store evaluation results
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(mean_reward)
            self.evaluations_length.append(mean_length)
            self.evaluations_std.append(std_reward)
            
            # Log to tensorboard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            
            if self.verbose > 0:
                print(f" Evaluation Results:")
                print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"   Mean Length: {mean_length:.1f}")
                print(f"   Time: {eval_time:.1f}s")
            
            # Save checkpoint for this evaluation
            checkpoint_path = self.eval_checkpoint_dir / f"model_{self.num_timesteps}_steps"
            self.model.save(checkpoint_path.as_posix())
            
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                norm_path = self.eval_checkpoint_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
                vec_env.save(norm_path.as_posix())
            
            if self.verbose > 0:
                print(f"💾 Checkpoint saved")
            
            # Check if this is the best model
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    improvement = mean_reward - self.best_mean_reward
                    print(f"⭐ NEW BEST MODEL! (+{improvement:.2f} improvement)")
                
                self.best_mean_reward = mean_reward
                
                # Save as best model
                best_model_path = self.log_dir / "best_model"
                self.model.save(best_model_path.as_posix())
                
                if isinstance(vec_env, VecNormalize):
                    best_norm_path = self.log_dir / "best_vecnormalize.pkl"
                    vec_env.save(best_norm_path.as_posix())
                
                # Save metadata about best model
                best_info = {
                    'timestep': self.num_timesteps,
                    'mean_reward': float(mean_reward),
                    'std_reward': float(std_reward),
                    'mean_length': float(mean_length),
                    'episode_rewards': [float(r) for r in episode_rewards],
                }
                
                best_info_path = self.log_dir / "best_model_info.json"
                with open(best_info_path, 'w') as f:
                    json.dump(best_info, f, indent=2)
                
                if self.verbose > 0:
                    print(f"   Saved to: {best_model_path}")
            else:
                if self.verbose > 0:
                    deficit = self.best_mean_reward - mean_reward
                    print(f"   (Best: {self.best_mean_reward:.2f}, current is {deficit:.2f} below)")
            
            if self.verbose > 0:
                print(f"{'='*60}\n")
        
        return True
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed during training."""
        return {
            'timesteps': self.evaluations_timesteps,
            'mean_rewards': self.evaluations_results,
            'mean_lengths': self.evaluations_length,
            "std_rewards": [float(s) for s in self.evaluations_std],   
            "n_eval_episodes": int(self.n_eval_episodes), 
            'best_mean_reward': float(self.best_mean_reward),
            'n_evaluations': len(self.evaluations_timesteps),
        }

############################################
# Helpers copied/adapted from train_ppo_bace.py
############################################

def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

def _resolve_env_path() -> str:
    platform = sys.platform
    if platform == "darwin":
        candidates = [REPO_ROOT / "Godot_Air_Combat" / "B_ACE.app"]
    elif platform.startswith("linux"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.x86_64"]
    elif platform.startswith("win"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.exe"]
    else:
        raise RuntimeError(f"Unsupported platform '{platform}'.")

    for binary in candidates:
        if binary.exists():
            return binary.as_posix()

    searched = "\n  - ".join(c.as_posix() for c in candidates)
    raise FileNotFoundError(
        "Could not find a suitable Godot binary. Checked:\n"
        f"  - {searched}"
    )

def _parse_args():
    parser = argparse.ArgumentParser(description="Train SB3 PPO on B-ACE with decay schedules and seed support.")
    parser.add_argument(
        "--config",
        type=str,
        default=(REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json").as_posix(),
        help="Path to a B-ACE JSON scenario config.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total agent timesteps for SB3 PPO.learn()."
    )
    
    # Seed support
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    
    # Learning rate arguments
    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        default=0.000258221042696745,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=0.000034963,
        help="Final learning rate after decay."
    )
    
    # Entropy coefficient arguments
    parser.add_argument(
        "--initial-entropy-coef",
        type=float,
        default=0.0450525120114811,
        help="Initial entropy coefficient."
    )
    parser.add_argument(
        "--final-entropy-coef",
        type=float,
        default=0.0254305724354894,
        help="Final entropy coefficient after decay."
    )
    
    # Decay schedule type
    parser.add_argument(
        "--decay-schedule",
        type=str,
        default="linear",
        choices=["linear", "exponential"],
        help="Type of decay schedule to use."
    )
    
    parser.add_argument(
        '--n_layers',
        type=int,
        default=2,
        help='Number of hidden layers in the neural network (default: 2)'
    )

    parser.add_argument(
        '--layer_size',
        type=int,
        default=512,
        help='Size of each hidden layer (default: 256)'
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor."
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.911539975010891,
        help="GAE lambda."
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.25,
        help="PPO clip range."
    )


    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.772813953325389,
        help="Value loss coefficient."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=6144,
        help="Rollout length per PPO update."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size per PPO epoch."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs per PPO update."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run name for logs/checkpoints."
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Logical experiment name used to group runs under runs_sb3/EXPERIMENT_NAME"
    )
    
    # Metrics collection
    parser.add_argument(
        "--metrics-interval",
        type=int,
        default=1000,
        help="Interval (in timesteps) for collecting detailed metrics."
    )
    
    # Enriched observations toggle (DEFAULT: ENABLED)
    parser.add_argument(
        "--use-enriched-obs",
        action="store_true",
        help="Enable enriched observations with pursuit-evasion heuristics (47 dims)"
    )
    parser.add_argument(
        "--no-enriched-obs",
        dest="use_enriched_obs",
        action="store_false",
        help="Disable enriched observations (use baseline 22 dims)"
    )
    # Set default to True (enriched observations enabled by default)
    parser.set_defaults(use_enriched_obs=True)
    
    return parser.parse_args()


############################################
# Single-agent wrapper for SB3
############################################

class SingleAgentBACEEnv(gym.Env):
    """
    SB3-compatible single-agent wrapper around B_ACE_GodotPettingZooWrapper.
    
    Uses "reset first" approach: resets env immediately, builds spaces from actual data.
    This avoids issues with PettingZoo's observation_space() method API.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        bace_config: Dict[str, Any],
        controlled_agent_id: str = "agent_0",
        max_episode_steps: int = 36000,
    ):
        super().__init__()
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        # Create underlying multi-agent env
        self._ma_env = B_ACE_GodotPettingZooWrapper(device="cpu", **bace_config)
        # ---- Expose labels + HVAA helper passthrough for downstream wrappers ----
        self.observation_labels = getattr(self._ma_env, "observation_labels", None)

        if hasattr(self._ma_env, "get_hvaa_indices"):
            self.get_hvaa_indices = self._ma_env.get_hvaa_indices
        # ---------------------------------------------------------------------------
        
        # Reset FIRST to get actual observation data
        obs_dict, info_dict = self._ma_env.reset()
        self._last_obs_dict = obs_dict

        # Figure out which agent we control
        agents = getattr(self._ma_env, "agents", [])
        if agents:
            if controlled_agent_id in agents:
                self.controlled_agent_id = controlled_agent_id
            else:
                print(f"WARNING: '{controlled_agent_id}' not in env.agents={agents}, "
                      f"defaulting to '{agents[0]}'")
                self.controlled_agent_id = agents[0]
        else:
            self.controlled_agent_id = controlled_agent_id

        # Build observation_space from actual observation
        if self.controlled_agent_id in obs_dict:
            raw_obs = obs_dict[self.controlled_agent_id]
        else:
            raw_obs = next(iter(obs_dict.values()))
        
        obs_vec = self._to_array(raw_obs)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_vec.shape,
            dtype=np.float32,
        )
        print(f"[DEBUG] Built observation_space from reset: shape={obs_vec.shape}")

        # Build action_space: B-ACE uses 4-dim continuous
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        print(f"[DEBUG] Built action_space: shape=(4,)")

    def _to_array(self, obj) -> np.ndarray:
        """Recursively extract numeric vector from nested structures."""
        if isinstance(obj, np.ndarray):
            return obj.astype(np.float32)
        if isinstance(obj, (list, tuple)):
            return np.asarray(obj, dtype=np.float32)
        if isinstance(obj, (float, int)):
            return np.asarray([obj], dtype=np.float32)
        if isinstance(obj, dict):
            if "obs" in obj:
                return self._to_array(obj["obs"])
            for v in obj.values():
                try:
                    arr = self._to_array(v)
                    if arr.dtype.kind in ("f", "i"):
                        return arr
                except (TypeError, ValueError):
                    continue
            raise RuntimeError(f"Could not extract obs from dict: {list(obj.keys())}")
        raise RuntimeError(f"Unsupported observation type {type(obj)}")

    def _extract_single_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Extract observation for controlled agent."""
        if self.controlled_agent_id in obs_dict:
            raw_obs = obs_dict[self.controlled_agent_id]
        else:
            raw_obs = next(iter(obs_dict.values()))
        return self._to_array(raw_obs)

    def _build_action_payload(self, agent_action: np.ndarray) -> Any:
        """Build action dict for all agents."""
        clipped = np.clip(agent_action, -1.0, 1.0).astype(np.float32)
        
        if hasattr(self._ma_env, "agents"):
            return {agent: clipped for agent in self._ma_env.agents}
        else:
            return {self.controlled_agent_id: clipped}

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        obs_dict, info_dict = self._ma_env.reset()
        self._last_obs_dict = obs_dict
        
        obs = self._extract_single_obs(obs_dict)
        info = info_dict.get(self.controlled_agent_id, {}) if isinstance(info_dict, dict) else {}
        return obs, info

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1

        action_dict = self._build_action_payload(action)
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._ma_env.step(action_dict)
        self._last_obs_dict = obs_dict

        # Extract values for controlled agent
        obs = self._extract_single_obs(obs_dict)
        
        if isinstance(rew_dict, dict):
            reward = float(rew_dict.get(self.controlled_agent_id, 0.0))
        else:
            reward = float(rew_dict)

        if isinstance(term_dict, dict):
            terminated = bool(term_dict.get(self.controlled_agent_id, False)) or bool(term_dict.get("__all__", False))
        else:
            terminated = bool(term_dict)

        if isinstance(trunc_dict, dict):
            truncated = bool(trunc_dict.get(self.controlled_agent_id, False)) or bool(trunc_dict.get("__all__", False))
        else:
            truncated = bool(trunc_dict)

        if self._elapsed_steps >= self._max_episode_steps and not terminated:
            truncated = True

        info = info_dict.get(self.controlled_agent_id, {}) if isinstance(info_dict, dict) else {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if hasattr(self._ma_env, "close"):
            self._ma_env.close()
            

############################################
# Main training driver
############################################

def main():
    args = _parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    
    # Load + sanitize B-ACE config
    config_path = Path(args.config).expanduser().resolve()
    B_ACE_config = load_b_ace_config(config_path.as_posix())
    if B_ACE_config is None:
        raise RuntimeError(f"Failed to load config file at {config_path}")

    env_cfg = B_ACE_config.setdefault("EnvConfig", {})

    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = _resolve_env_path()

    env_path_value = Path(env_cfg["env_path"]).expanduser()
    if not env_path_value.is_absolute():
        env_path_value = (REPO_ROOT / env_path_value).resolve()
    else:
        env_path_value = env_path_value.resolve()

    if not env_path_value.exists():
        fallback_path = Path(_resolve_env_path()).resolve()
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"[B-ACE ERROR] Tried {env_path_value}, and fallback {fallback_path}, "
                "but neither exists. You probably need to export the Godot binary."
            )
        print(
            f"Warning: Godot binary not found at {env_path_value}. "
            f"Falling back to {fallback_path}."
        )
        env_path_value = fallback_path

    env_cfg["env_path"] = env_path_value.as_posix()

    agents_cfg = B_ACE_config.setdefault("AgentsConfig", {})
    blue_cfg = agents_cfg.setdefault("blue_agents", {})
    red_cfg = agents_cfg.setdefault("red_agents", {})
    blue_cfg.setdefault("base_behavior", "external")
    red_cfg.setdefault("base_behavior", "baseline1")

    env_cfg.setdefault("renderize", 1)
    env_cfg.setdefault("speed_up", 200)

    # Logging / run naming with seed
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.run_name is not None:
        run_name = args.run_name
    elif args.experiment_name is not None:
        # Default pattern when experiment name is given
        run_name = f"{args.experiment_name}_seed{args.seed}_{timestamp}"
    else:
        # Fallback legacy pattern
        run_name = f"sb3ppo_bace_decay_seed{args.seed}_{timestamp}"

    # Base directory for all runs
    base_runs_dir = REPO_ROOT / "runs_sb3"

    # If an experiment name is given, group all seeds under runs_sb3/EXPERIMENT_NAME/
    if args.experiment_name is not None:
        log_dir = (base_runs_dir / args.experiment_name / run_name).resolve()
    else:
        # Legacy behavior: runs_sb3/RUN_NAME
        log_dir = (base_runs_dir / run_name).resolve()

    log_dir.mkdir(parents=True, exist_ok=True)

    # Build single-agent Gym env
    def make_env(log_dir_str, seed):
        def _thunk():
            e = SingleAgentBACEEnv(
                bace_config=B_ACE_config,
                controlled_agent_id="agent_0",
                max_episode_steps=6000,
            )
            
            # DEFAULT: Enriched observations with pursuit-evasion heuristics (47 dims)
            # Use --no-enriched-obs flag to disable and get baseline (22 dims)
            if args.use_enriched_obs:
                print("\n" + "="*60)
                print("USING ENRICHED OBSERVATIONS (DEFAULT)")
                print("="*60 + "\n")
                
                e = EnrichedObservationWrapper(
                    e,
                    obs_labels=None,  # Not needed anymore - we use indices directly
                    feature_config={
                        'pursuer_speed': 1.0,       # Normalized speed
                        'evader_speed': 1.0,        # Normalized speed
                        'capture_radius': 0.01,     # Small in normalized space
                        'pursuer_range': 0.5,       # Will use Godot's missile ranges
                        'normalize_distance': 1.0,  # Already normalized
                        'enable_apollonius': True,
                        'enable_bez': True,
                        'enable_dmc': True,
                        'enable_multi_threat': True,
                        'enable_hvaa_escort': True,
                        'enable_offense_wez': True,
                        'enable_offense_ttc': True,
                        'enable_reward_shaping': False,  # This is the old shaping (B-ACE shaping)
                    },
                    debug=False  # Set True initially to verify features
                )
                obs, _ = e.reset()
                print(f"Obs shape: {obs.shape}")
                print(f"Indices 27-35: {obs[27:35]}")
                print(f"[5] offensive_factor: {obs[5]:.3f}")
                print(f"[7] track_detected: {obs[7]:.3f}")
                print(f"[14] missile_track: {obs[14]:.3f}")
                print(f"[29] bez_penetration: {obs[29]:.3f}")
                print(f"[30] inside_bez: {obs[30]:.3f}")
                print(f"[32] dmc_normalized: {obs[32]:.3f}")

                print(f"\n✓ Environment created with enriched observations")
                print(f"   Final observation space: {e.observation_space}")
                print(f"   Observation dimension: {e.observation_space.shape[0]}")
                
                shaping_config = RewardShapingConfig(
                    enable_bez_shaping=True,
                    enable_dmc_shaping=True,
                    enable_heading_shaping=False,     
                    enable_distance_shaping=False,    
                    enable_ttc_shaping=False,          
                    enable_multithreat_shaping=False, 
                    enable_speed_shaping=False,  
                    enable_hvaa_escort_shaping=True,
                    #enable_offense_wez=True,
                    #enable_offense_ttc=True,
                    use_potential_based=False, 
                    track_reward_components=True,  # Add this if missing
                    track_episode_stats=True,    
                    

                    #HVAA Escort parameters
                    #hvaa_escort_weight=0.3,
                    #hvaa_desired_dist=0.15,

                    #Offensive Parameters
                    #wez_dom_weight=0.2,
                    #offense_ttc_weight=0.1,

                    # BEZ parameters
                    #bez_weight=0.1,
                    #bez_penalty_inside=-0.002,
                    #bez_reward_outside=0.00002,

                    # DMC Parameters
                    #dmc_weight = 0.2,
                    #dmc_penalty_high  = -0.009,  # Penalty for high DMC (must turn hard)
                    #dmc_reward_low = 0.003,  

                    enable_logging=True, 
                    log_file=str(Path(log_dir_str) / f"reward_shaping_seed{seed}.csv")
                )
                e = RewardShapingWrapper(e, config=shaping_config)

                hier_config = CurriculumConfig(
                initial_greedy_prob=0.05,
                final_greedy_prob=1.0,
                curriculum_steps=args.total_timesteps
                )
                e = HierarchicalActionWrapper(e, curriculum_config=hier_config, debug=True)
                print(f"✓ HierarchicalActionWrapper added")
                
                 # ================= HVAA DEBUG BLOCK =================
                if DEBUG_HVAA_OBS:
                    try:
                        # Unwrap down to the SingleAgentBACEEnv and the multi-agent B_ACE wrapper
                        # e: RewardShapingWrapper -> EnrichedObservationWrapper -> SingleAgentBACEEnv
                        rs_wrapper = e
                        enriched_wrapper = rs_wrapper.env
                        single_env = enriched_wrapper.env
                        ma_env = single_env._ma_env  # B_ACE_GodotPettingZooWrapper

                        # Get HVAA indices for the controlled agent
                        if hasattr(ma_env, "get_hvaa_indices") and hasattr(ma_env, "possible_agents"):
                            agent_name = single_env.controlled_agent_id
                            hv_idx = ma_env.get_hvaa_indices(agent_name)
                            print("\n" + "="*70)
                            print("[HVAA DEBUG] HVAA index mapping for agent:", agent_name)
                            for label, idx in hv_idx.items():
                                print(f"  {label:15s} -> index {idx}")
                            
                            # Get one sample observation AFTER all wrappers (enriched obs)
                            # This is exactly what the policy sees.
                            sample_obs, _ = e.reset(seed=seed)
                            print("\n[HVAA DEBUG] Sample enriched observation shape:", sample_obs.shape)

                            # Print the values at the HVAA indices
                            print("[HVAA DEBUG] HVAA-related values from observation:")
                            for label, idx in hv_idx.items():
                                if 0 <= idx < len(sample_obs):
                                    print(f"  idx {idx:3d} | {label:15s} = {sample_obs[idx]: .6f}")
                                else:
                                    print(f"  idx {idx:3d} | {label:15s} = <OUT OF RANGE>")

                            print("="*70 + "\n")
                        else:
                            print("[HVAA DEBUG] ma_env has no get_hvaa_indices or possible_agents; type:", type(ma_env))
                    except Exception as ex:
                        print("[HVAA DEBUG] Exception while probing HVAA obs:", ex)
                # =============== END HVAA DEBUG BLOCK ===============

            else:
                print("\n" + "="*60)
                print("USING DMC ONLY")
                print("="*60 + "\n")

            '''
            # DEBUG: Print observation info - FINAL VERSION
            print("\n" + "="*60)
            print("OBSERVATION SPACE DEBUG INFO")
            print("="*60)
            print(f"Observation space shape: {e.observation_space}")
            
            # The labels are in the multi-agent wrapper
            ma_env = e._ma_env  # Get the B_ACE_GodotPettingZooWrapper
            
            print(f"\n--- Observation Labels from Godot ---")
            if hasattr(ma_env, 'observation_labels'):
                for agent_id, labels in ma_env.observation_labels.items():
                    print(f"\nAgent ID {agent_id}:")
                    if isinstance(labels, list):
                        for i, label in enumerate(labels):
                            print(f"  [{i:2d}] {label}")
                    else:
                        print(f"  {labels}")
            
            print(f"\n--- Obs Map ---")
            if hasattr(ma_env, 'obs_map'):
                for agent_name, mapping in ma_env.obs_map.items():
                    print(f"\n{agent_name}:")
                    # Sort by index for readability
                    sorted_mapping = sorted(mapping.items(), key=lambda x: x[1])
                    for label, idx in sorted_mapping:
                        print(f"  [{idx:2d}] {label}")
            
            # Get a sample observation with values
            test_obs, _ = e.reset()
            print(f"\n--- Sample Observation Values ---")
            print(f"Shape: {test_obs.shape}")
            
            # Try to match labels with values
            if hasattr(ma_env, 'obs_map') and 'agent_0' in ma_env.obs_map:
                mapping = ma_env.obs_map['agent_0']
                sorted_mapping = sorted(mapping.items(), key=lambda x: x[1])
                print(f"\nLabeled values:")
                for label, idx in sorted_mapping:
                    if idx < len(test_obs):
                        print(f"  [{idx:2d}] {label:30s} = {test_obs[idx]:12.6f}")
            else:
                # Just print indexed values
                for i, val in enumerate(test_obs):
                    print(f"  [{i:2d}] {val:12.6f}")
            
            print("="*60 + "\n")
            # END DEBUG
            '''


            e = Monitor(e, filename=str(Path(log_dir_str) / "monitor.csv"))
            # === DEBUG: confirm final observation shape ===
            obs, _ = e.reset(seed=seed)
            print(f"[DEBUG] Final observation shape: {obs.shape}")
            # Optional safety check if you expect exactly 47 features:
            # assert obs.shape[0] == 47, f"Expected 47-dim obs, got {obs.shape}"
            #e.reset(seed=seed)
            return e
        return _thunk

    vec_env = DummyVecEnv([make_env(log_dir.as_posix(), args.seed)])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    print(f"[NEW***DEBUG] vec_env.observation_space: {vec_env.observation_space}")

    logger = configure(log_dir.as_posix(), ["stdout", "tensorboard"])

    # Create decay schedules
    if args.decay_schedule == "linear":
        lr_schedule = linear_schedule(args.initial_learning_rate, args.final_learning_rate)
        ent_schedule = linear_schedule(args.initial_entropy_coef, args.final_entropy_coef)
    else:  # exponential
        lr_schedule = exponential_schedule(args.initial_learning_rate, args.final_learning_rate)
        ent_schedule = exponential_schedule(args.initial_entropy_coef, args.final_entropy_coef)


    clip_schedule = args.clip_eps

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Random Seed: {args.seed}")
    print(f"Decay Schedule Type: {args.decay_schedule}")
    print(f"Learning Rate: {args.initial_learning_rate:.2e} → {args.final_learning_rate:.2e}")
    print(f"Entropy Coef: {args.initial_entropy_coef:.4f} → {args.final_entropy_coef:.4f}")
    print(f"Clip Range: {args.clip_eps:.3f} (constant)")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"{'='*60}\n")

    # Create PPO model with decay schedules and seed
    layer_sizes = [args.layer_size] * args.n_layers
    policy_kwargs = dict(net_arch=[dict(pi=layer_sizes, vf=layer_sizes)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr_schedule,  # Callable schedule - SB3 handles this
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=clip_schedule,
        ent_coef=args.initial_entropy_coef,  # Start with initial value (float)
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir.as_posix(),
        device=args.device,
        policy_kwargs=policy_kwargs,
        seed=args.seed  # Set seed for PPO
    )

    model.set_logger(logger)
    
    # Create evaluation environment (separate from training environment)
    print(f"\n{'='*60}")
    print("Creating Evaluation Environment")
    print(f"{'='*60}")
    
    # Evaluation environment should match training environment setup
    # but without normalization initially (we'll sync it from training env)
    eval_env = DummyVecEnv([make_env(log_dir.as_posix(), args.seed + 10000)])  # Different seed for eval
    
    # Wrap with VecNormalize but set training=False
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during eval
        clip_obs=10.0,
        clip_reward=10.0,
        training=False,  # Important: don't update normalization stats during eval
    )
    
    # Sync normalization stats from training env to eval env
    # This will be updated automatically during training
    eval_env.obs_rms = vec_env.obs_rms
    eval_env.ret_rms = vec_env.ret_rms
    
    print(f"✓ Evaluation environment created")
    print(f"  Eval episodes: 10")
    print(f"  Eval frequency: every 250,000 steps")
    print(f"{'='*60}\n")
    
    # Create callbacks
    # 1. Metrics collection callback (tracks training progress)
    metrics_callback = MetricsCollectionCallback(
        entropy_schedule=ent_schedule,
        metrics_collection_interval=args.metrics_interval,
        max_stored_episodes=10000  # Memory efficient
    )
    
    # 2. Evaluation-based checkpoint callback (saves best model based on eval performance)
    '''
    eval_checkpoint_callback = EvaluationBasedCheckpointCallback(
        eval_env=eval_env,
        eval_freq=50000,  # Evaluate every 50k steps
        n_eval_episodes=20,  # Run 20 episodes per evaluation
        log_dir=log_dir,
        verbose=1,
        deterministic=True  # Use deterministic actions during evaluation
    )
    '''
    # 2. Evaluation-based checkpoint callback with component tracking
    if COMPONENT_TRACKING_AVAILABLE:
        print("✓ Using RewardComponentEvalCallback (with component tracking)")
        eval_checkpoint_callback = RewardComponentEvalCallback(
            eval_env=eval_env,
            eval_freq=50000,  # Evaluate every 50k steps
            n_eval_episodes=20,  # Run 20 episodes per evaluation
            log_dir=log_dir,
            verbose=1,
            deterministic=True  # Use deterministic actions during evaluation
        )
    else:
        print("✓ Using EvaluationBasedCheckpointCallback (standard)")
        eval_checkpoint_callback = EvaluationBasedCheckpointCallback(
            eval_env=eval_env,
            eval_freq=50000,
            n_eval_episodes=20,
            log_dir=log_dir,
            verbose=1,
            deterministic=True
    )


    # Combine callbacks
    callback_list = CallbackList([metrics_callback, eval_checkpoint_callback])

    # Train with callbacks
    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=callback_list  # Use combined callbacks
    )
    training_time = time.time() - start_time

    # ============================================================
    # FINAL EVALUATION (after training completes)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"🏁 FINAL EVALUATION (after training complete)")
    print(f"   Running 20 episodes with FINAL model...")
    print(f"{'='*60}")
    
    final_eval_start = time.time()
    
    # Evaluate the final model
    final_episode_rewards = []
    final_episode_lengths = []
    
    obs = eval_env.reset()
    episode_reward = 0.0
    episode_length = 0
    episodes_completed = 0
    
    while episodes_completed < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        
        episode_reward += reward[0]
        episode_length += 1
        
        if done[0]:
            final_episode_rewards.append(episode_reward)
            final_episode_lengths.append(episode_length)
            episodes_completed += 1
            
            episode_reward = 0.0
            episode_length = 0
    
    final_mean_reward = np.mean(final_episode_rewards)
    final_std_reward = np.std(final_episode_rewards)
    final_mean_length = np.mean(final_episode_lengths)
    final_eval_time = time.time() - final_eval_start
    
    print(f"📊 Final Model Evaluation Results:")
    print(f"   Mean Reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}")
    print(f"   Mean Length: {final_mean_length:.1f}")
    print(f"   Time: {final_eval_time:.1f}s")
    print(f"{'='*60}\n")

    # Save FINAL model and normalization stats
    model_path = log_dir / "ppo_bace_blue_lead"
    norm_path = log_dir / "vecnormalize.pkl"
    model.save(model_path.as_posix())
    vec_env.save(norm_path.as_posix())
    
    # Get evaluation summary
    eval_summary = eval_checkpoint_callback.get_evaluation_summary()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"")
    print(f"FINAL MODEL:")
    print(f"  Path: {model_path}")
    print(f"  Eval reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}")
    print(f"")
    print(f"BEST MODEL:")
    print(f"  Path: {log_dir / 'best_model'}")
    print(f"  Eval reward: {eval_checkpoint_callback.best_mean_reward:.2f}")
    print(f"  Found at: {eval_summary['timesteps'][eval_summary['mean_rewards'].index(eval_checkpoint_callback.best_mean_reward)]:,} steps")
    print(f"")
    print(f"Comparison: Best is {eval_checkpoint_callback.best_mean_reward - final_mean_reward:.2f} points better than final")
    print(f"Number of evaluations: {eval_summary['n_evaluations']}")
    print(f"{'='*60}\n")

    # Collect and save metrics
    metrics_summary = metrics_callback.get_metrics_summary()
    
    # Add training configuration to metrics
    training_config = {
        'experiment_name': args.experiment_name or "default_experiment",
        'seed': args.seed,
        'total_timesteps': args.total_timesteps,
        'training_time_seconds': training_time,
        'decay_schedule': args.decay_schedule,
        'initial_learning_rate': args.initial_learning_rate,
        'final_learning_rate': args.final_learning_rate,
        'initial_entropy_coef': args.initial_entropy_coef,
        'final_entropy_coef': args.final_entropy_coef,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.epochs,
        'n_layers': args.n_layers,
        'layer_size': args.layer_size,
        'vf_coef': args.vf_coef,
    }
    
    results = {
        'config': training_config,
        'metrics': metrics_summary,
        'evaluation_summary': eval_summary,
        'final_evaluation': {  # NEW: Final model evaluation
            'mean_reward': float(final_mean_reward),
            'std_reward': float(final_std_reward),
            'mean_length': float(final_mean_length),
            'episode_rewards': [float(r) for r in final_episode_rewards],
        },
        'model_path': model_path.as_posix(),
        'best_model_path': (log_dir / "best_model.zip").as_posix(),
        'best_eval_mean_reward': float(eval_checkpoint_callback.best_mean_reward),
        'final_vs_best_difference': float(eval_checkpoint_callback.best_mean_reward - final_mean_reward),
        'eval_checkpoints_dir': eval_checkpoint_callback.eval_checkpoint_dir.as_posix(),
        'log_dir': log_dir.as_posix()
    }
    
    # Save results to JSON
    results_path = log_dir / f"training_results_seed{args.seed}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[SB3 PPO] Training complete!")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Total episodes: {metrics_summary['total_episodes']}")
    print(f"Mean reward: {metrics_summary['mean_reward']:.2f} ± {metrics_summary['std_reward']:.2f}")
    print(f"Model saved to: {model_path}")
    print(f"VecNormalize stats saved to: {norm_path}")
    print(f"Results saved to: {results_path}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"{'='*60}\n")
    
    # Clean up
    eval_env.close()


if __name__ == "__main__":
    main()
