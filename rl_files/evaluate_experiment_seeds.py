#!/usr/bin/env python3
"""
evaluate_experiment_seeds.py

Evaluate the superlative (best) policy from each training seed in an experiment.
Produces per-seed metrics and aggregate statistics across seeds for comparing
experimental configurations.

Usage:
    python evaluate_experiment_seeds.py \
        --experiment-dir runs_sb3/my_experiment \
        --episodes 100
    
    # Or specify specific seeds:
    python evaluate_experiment_seeds.py \
        --experiment-dir runs_sb3/my_experiment \
        --seeds seed42,seed123,seed456 \
        --episodes 100
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Plus-minus symbol for display
PM = "\u00b1"

# -----------------------------
# Repo root discovery
# -----------------------------
THIS_FILE = Path(__file__).resolve()
current_path = THIS_FILE.parent
while current_path != current_path.parent:
    if (current_path / "b_ace_py").exists():
        REPO_ROOT = current_path
        break
    current_path = current_path.parent
else:
    REPO_ROOT = THIS_FILE.parent.parent

if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

print(f"Using REPO_ROOT: {REPO_ROOT}")

from b_ace_py.utils import load_b_ace_config
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
from b_ace_py.enriched_observation_wrapper import EnrichedObservationWrapper


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class ConfidenceInterval:
    """Stores a metric with its confidence interval."""
    mean: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    n: int
    method: str
    count: Optional[int] = None
    std: Optional[float] = None


@dataclass
class EpisodeData:
    """Per-episode data from evaluation."""
    reward: float
    length: int
    hvaa_survived: bool
    red_killed: bool
    blue_killed: bool
    blue_missiles_fired: int = 0
    red_missiles_fired: int = 0
    blue_pk: float = 0.0
    red_pk: float = 0.0
    blue_shot_first: bool = False
    red_shot_first: bool = False
    no_shots_fired: bool = False
    min_separation_nm: float = -1.0
    end_reason: str = ""
    training_seed: str = ""  # Which training seed this episode came from


@dataclass
class SeedResults:
    """Results for a single training seed."""
    seed_name: str
    model_path: str
    vecnorm_path: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    hvaa_survival_rate: float
    red_kill_rate: float
    blue_loss_rate: float
    mean_length: float
    mean_blue_missiles: float
    mean_red_missiles: float
    mean_blue_pk: float
    mean_red_pk: float
    blue_shot_first_rate: float
    red_shot_first_rate: float
    no_shots_rate: float
    episodes: List[EpisodeData]


# -----------------------------
# CI computation
# -----------------------------
def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Wilson score CI for binomial proportions."""
    if total == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, 0.0, 0, "wilson_score", count=0)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    return ConfidenceInterval(
        mean=float(p),
        ci_lower=float(max(0, center - margin)),
        ci_upper=float(min(1, center + margin)),
        ci_width=float(min(1, center + margin) - max(0, center - margin)),
        n=total,
        method="wilson_score",
        count=successes
    )


def t_distribution_ci(data: List[float], confidence: float = 0.95) -> ConfidenceInterval:
    """t-distribution CI for continuous data."""
    arr = np.array(data)
    n = len(arr)
    
    if n < 2:
        mean = float(np.mean(arr)) if n > 0 else 0.0
        return ConfidenceInterval(mean, mean, mean, 0.0, n, "t_distribution", std=0.0)
    
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    sem = stats.sem(arr)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return ConfidenceInterval(
        mean=mean,
        ci_lower=float(mean - h),
        ci_upper=float(mean + h),
        ci_width=float(2 * h),
        n=n,
        method="t_distribution",
        std=std
    )


# -----------------------------
# Wrappers (same as training)
# -----------------------------
class HVAASurvivalEveryNStepsBonus(gym.Wrapper):
    def __init__(self, env, every_n_steps: int = 50, bonus: float = 0.05):
        super().__init__(env)
        self.every_n_steps = int(every_n_steps)
        self.bonus = float(bonus)
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        hvaa_alive = bool(info.get("hvaa_alive", True))
        if hvaa_alive and (self._t % self.every_n_steps == 0):
            rew = float(rew) + self.bonus
        self._t += 1
        return obs, rew, terminated, truncated, info


class FireCooldownWrapper(gym.Wrapper):
    def __init__(self, env, cooldown_steps: int = 120, fire_idx: int = 3, threshold: float = 0.5):
        super().__init__(env)
        self.cooldown_steps = int(cooldown_steps)
        self.fire_idx = int(fire_idx)
        self.threshold = float(threshold)
        self._t = 0
        self._last_fire_t = -10**9
        self._prev_want_fire = False

    def reset(self, **kwargs):
        self._t = 0
        self._last_fire_t = -10**9
        self._prev_want_fire = False
        return self.env.reset(**kwargs)

    def step(self, action):
        a = np.array(action, dtype=np.float32, copy=True)
        want_fire = float(a[self.fire_idx]) > self.threshold
        rising_edge = want_fire and (not self._prev_want_fire)
        if rising_edge and (self._t - self._last_fire_t) >= self.cooldown_steps:
            a[self.fire_idx] = 1.0
            self._last_fire_t = self._t
        else:
            a[self.fire_idx] = 0.0
        self._prev_want_fire = want_fire
        obs, rew, terminated, truncated, info = self.env.step(a)
        self._t += 1
        return obs, rew, terminated, truncated, info


class SingleAgentBACEEnv(gym.Env):
    """SB3-compatible single-agent wrapper."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, bace_config: Dict[str, Any], controlled_agent_id: str = "agent_0", max_episode_steps: int = 36000):
        super().__init__()
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._ma_env = B_ACE_GodotPettingZooWrapper(device="cpu", **bace_config)
        
        obs_dict, _ = self._ma_env.reset()
        self._last_obs_dict = obs_dict

        agents = getattr(self._ma_env, "agents", [])
        self.controlled_agent_id = controlled_agent_id if controlled_agent_id in agents else (agents[0] if agents else controlled_agent_id)

        raw_obs = obs_dict.get(self.controlled_agent_id, next(iter(obs_dict.values())))
        obs_vec = self._to_array(raw_obs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_vec.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _to_array(self, obj) -> np.ndarray:
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
                except:
                    continue
        raise RuntimeError(f"Could not extract obs")

    def _extract_single_obs(self, obs_dict):
        raw = obs_dict.get(self.controlled_agent_id, next(iter(obs_dict.values())))
        return self._to_array(raw)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0
        obs_dict, info_dict = self._ma_env.reset()
        self._last_obs_dict = obs_dict
        return self._extract_single_obs(obs_dict), info_dict.get(self.controlled_agent_id, {})

    def step(self, action):
        self._elapsed_steps += 1
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        action_dict = {agent: clipped for agent in getattr(self._ma_env, "agents", [self.controlled_agent_id])}
        
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._ma_env.step(action_dict)
        self._last_obs_dict = obs_dict
        
        obs = self._extract_single_obs(obs_dict)
        reward = float(rew_dict.get(self.controlled_agent_id, 0.0)) if isinstance(rew_dict, dict) else float(rew_dict)
        terminated = bool(term_dict.get(self.controlled_agent_id, False) or term_dict.get("__all__", False)) if isinstance(term_dict, dict) else bool(term_dict)
        truncated = bool(trunc_dict.get(self.controlled_agent_id, False) or trunc_dict.get("__all__", False)) if isinstance(trunc_dict, dict) else bool(trunc_dict)
        
        if self._elapsed_steps >= self._max_episode_steps and not terminated:
            truncated = True
        
        return obs, reward, terminated, truncated, info_dict.get(self.controlled_agent_id, {})

    def close(self):
        if hasattr(self._ma_env, "close"):
            self._ma_env.close()


# -----------------------------
# Auto-detection utilities
# -----------------------------
def detect_obs_dim_from_vecnormalize(vecnorm_path: Path) -> int:
    """Load VecNormalize pickle and extract expected observation dimension."""
    with open(vecnorm_path, 'rb') as f:
        data = pickle.load(f)
    
    if hasattr(data, 'obs_rms') and data.obs_rms is not None:
        obs_dim = data.obs_rms.mean.shape[0]
        return obs_dim
    
    raise ValueError(f"Could not determine obs dimension from {vecnorm_path}")


def determine_obs_mode(obs_dim: int) -> str:
    """Map observation dimension to mode name."""
    if obs_dim <= 30:
        return "baseline"
    elif obs_dim <= 50:
        return "enriched"
    else:
        return "enriched_extended"


# -----------------------------
# Helpers
# -----------------------------
def _resolve_env_path() -> str:
    platform = sys.platform
    if platform == "darwin":
        candidates = [REPO_ROOT / "Godot_Air_Combat" / "B_ACE.app"]
    elif platform.startswith("linux"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.x86_64"]
    elif platform.startswith("win"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.exe"]
    else:
        raise RuntimeError(f"Unsupported platform '{platform}'")
    
    for binary in candidates:
        if binary.exists():
            return binary.as_posix()
    raise FileNotFoundError(f"Could not find Godot binary")


def _as_bool(v) -> bool:
    """Safe boolean conversion - handles various types from info dict."""
    return bool(v) if isinstance(v, bool) else (v > 0.5 if isinstance(v, (int, float)) else False)


def _as_int(v) -> int:
    """Safe integer conversion."""
    try:
        return int(v)
    except:
        return 0


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, '__dict__'):
        return _jsonify(obj.__dict__)
    return str(obj)


# -----------------------------
# Seed discovery (similar to plotting script)
# -----------------------------
def find_seed_models(experiment_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    Find best model and vecnormalize for each seed in experiment directory.
    
    Returns dict: seed_name -> (model_path, vecnorm_path)
    """
    results: Dict[str, Tuple[Path, Path]] = {}
    
    for seed_dir in sorted(experiment_dir.glob("*seed*")):
        if not seed_dir.is_dir():
            continue
        
        match = re.search(r"seed(\d+)", seed_dir.name)
        seed_name = f"seed{match.group(1)}" if match else seed_dir.name
        
        # Find best model
        model_candidates = [
            seed_dir / "best_model.zip",
            seed_dir / "eval_checkpoints" / "best_model.zip",
        ]
        model_path = None
        for c in model_candidates:
            if c.exists():
                model_path = c
                break
        
        if model_path is None:
            print(f"Warning: No best_model.zip found for {seed_dir.name}")
            continue
        
        # Find vecnormalize
        vecnorm_candidates = [
            seed_dir / "best_vecnormalize.pkl",
            seed_dir / "vecnormalize.pkl",
            seed_dir / "eval_checkpoints" / "best_vecnormalize.pkl",
        ]
        vecnorm_path = None
        for c in vecnorm_candidates:
            if c.exists():
                vecnorm_path = c
                break
        
        if vecnorm_path is None:
            print(f"Warning: No vecnormalize.pkl found for {seed_dir.name}")
            continue
        
        results[seed_name] = (model_path, vecnorm_path)
        print(f"Found {seed_name}: {model_path.name}, {vecnorm_path.name}")
    
    return results


# -----------------------------
# Main evaluation
# -----------------------------
def validate_obs_dimension(
    bace_cfg: Dict,
    obs_mode: str,
    feature_config: Dict[str, Any],
    expected_obs_dim: int,
    ablation_config: Optional[str] = None,
) -> Tuple[bool, int, str]:
    """
    Validate that the configured observation dimension matches vecnormalize.
    
    Creates a temporary environment to check the actual observation dimension
    that would result from the current configuration.
    
    Returns:
        Tuple of (is_valid, actual_dim, message)
    """
    temp_env = None
    try:
        # Create a temporary environment to check dimension
        temp_env = SingleAgentBACEEnv(bace_config=bace_cfg, controlled_agent_id="agent_0", max_episode_steps=14000)
        baseline_dim = temp_env.observation_space.shape[0]
        
        if obs_mode in ["enriched", "enriched_extended"]:
            # Build the wrapper kwargs
            wrapper_kwargs = {
                'obs_labels': None,
                'debug': False,
            }
            
            if ablation_config:
                wrapper_kwargs['ablation_config'] = ablation_config
            else:
                # Pass feature_config for the wrapper's internal use
                wrapper_kwargs['feature_config'] = feature_config
            
            temp_env = EnrichedObservationWrapper(temp_env, **wrapper_kwargs)
        
        actual_dim = temp_env.observation_space.shape[0]
        
        if actual_dim == expected_obs_dim:
            return True, actual_dim, f"OK: Configured dim ({actual_dim}) matches vecnormalize dim ({expected_obs_dim})"
        else:
            msg = (f"MISMATCH: Configured observation dim ({actual_dim}) != "
                   f"vecnormalize dim ({expected_obs_dim}). "
                   f"Baseline dim is {baseline_dim}. "
                   f"The model was trained with {expected_obs_dim}-dim observations but "
                   f"your current config produces {actual_dim}-dim observations. "
                   f"This will cause evaluation errors or incorrect results!")
            return False, actual_dim, msg
    finally:
        if temp_env is not None:
            try:
                temp_env.close()
            except Exception:
                pass  # Ignore close errors


def evaluate_model(
    model_path: Path,
    vecnorm_path: Path,
    bace_cfg: Dict,
    n_episodes: int,
    training_seed: str,
    eval_seed: int = 42,  # Fixed eval seed for determinism
    obs_mode_override: Optional[str] = None,  # "auto", "baseline", "enriched", "enriched_extended"
    use_enriched_obs: bool = False,  # Explicit flag to enable enriched obs
    feature_config: Optional[Dict[str, Any]] = None,  # Feature configuration for enriched obs
    ablation_config: Optional[str] = None,  # Ablation config name for wrapper
    verbose: bool = True,
    strict_validation: bool = True,  # If True, raise error on dim mismatch; if False, just warn
) -> List[EpisodeData]:
    """Run evaluation episodes for a single model."""
    
    # Auto-detect observation dimension from vecnormalize
    expected_obs_dim = detect_obs_dim_from_vecnormalize(vecnorm_path)
    
    # Determine observation mode
    if obs_mode_override and obs_mode_override != "auto":
        obs_mode = obs_mode_override
    elif use_enriched_obs:
        obs_mode = "enriched"
    else:
        obs_mode = determine_obs_mode(expected_obs_dim)
    
    # Use default feature config if not provided
    if feature_config is None:
        feature_config = {
            'pursuer_speed': 1.0, 'evader_speed': 1.0, 'capture_radius': 0.01,
            'pursuer_range': 0.5, 'normalize_distance': 1.0,
            'enable_apollonius': True, 'enable_bez': True, 'enable_dmc': True,
            'enable_multi_threat': False, 'enable_hvaa_escort': False,
            'enable_offense_wez': True, 'enable_offense_ttc': True,
            'enable_reward_shaping': False,
        }
    
    if verbose:
        print(f"   Obs mode: {obs_mode} (vecnormalize dim={expected_obs_dim})")
        if obs_mode in ["enriched", "enriched_extended"]:
            enabled_features = [k for k, v in feature_config.items() if k.startswith('enable_') and v]
            print(f"   Enabled features: {enabled_features}")
    
    # === VALIDATION: Check observation dimension BEFORE creating eval environment ===
    is_valid, actual_dim, validation_msg = validate_obs_dimension(
        bace_cfg=bace_cfg,
        obs_mode=obs_mode,
        feature_config=feature_config,
        expected_obs_dim=expected_obs_dim,
        ablation_config=ablation_config,
    )
    
    if not is_valid:
        print(f"\n   ⚠️  OBSERVATION DIMENSION VALIDATION FAILED ⚠️")
        print(f"   {validation_msg}")
        print(f"   ")
        print(f"   This typically means:")
        print(f"   - The model was trained with a DIFFERENT observation configuration")
        print(f"   - Your --use-enriched-obs, --ablation-config, or feature flags don't match training")
        print(f"   ")
        print(f"   Vecnormalize dimension: {expected_obs_dim}")
        print(f"   Your configured dimension: {actual_dim}")
        print(f"   ")
        
        if strict_validation:
            raise ValueError(
                f"Observation dimension mismatch: configured={actual_dim}, vecnormalize={expected_obs_dim}. "
                f"Use --no-strict-validation to override (NOT RECOMMENDED)."
            )
        else:
            print(f"   ⚠️  Continuing anyway because --no-strict-validation was set.")
            print(f"   ⚠️  RESULTS MAY BE INVALID!")
    else:
        if verbose:
            print(f"   ✓ Validation: {validation_msg}")
    
    def make_env():
        def _thunk():
            e = SingleAgentBACEEnv(bace_config=bace_cfg, controlled_agent_id="agent_0", max_episode_steps=14000)
            
            # Only add enriched wrapper if needed
            if obs_mode in ["enriched", "enriched_extended"]:
                wrapper_kwargs = {
                    'obs_labels': None,
                    'debug': False,
                }
                if ablation_config:
                    wrapper_kwargs['ablation_config'] = ablation_config
                else:
                    wrapper_kwargs['feature_config'] = feature_config
                
                e = EnrichedObservationWrapper(e, **wrapper_kwargs)
            
            e = FireCooldownWrapper(e, cooldown_steps=120, fire_idx=3, threshold=0.5)
            e = HVAASurvivalEveryNStepsBonus(e, every_n_steps=50, bonus=0.05)
            return e
        return _thunk

    venv = DummyVecEnv([make_env()])
    eval_env = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env = VecNormalize.load(vecnorm_path.as_posix(), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # Load model
    model = PPO.load(model_path.as_posix(), device="cpu")
    model.set_env(eval_env)

    episodes: List[EpisodeData] = []
    
    # Reset with fixed seed for reproducibility
    eval_env.seed(eval_seed)
    obs = eval_env.reset()
    
    cur_reward = 0.0
    cur_length = 0
    ep_hvaa_destroyed = False
    ep_red_killed = False
    ep_blue_killed = False
    finished = 0
    
    while finished < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        step_out = eval_env.step(action)
        
        if len(step_out) == 4:
            obs, reward, done, infos = step_out
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = np.logical_or(terminated, truncated)
        
        info = infos[0] if isinstance(infos, (list, tuple)) else (infos if isinstance(infos, dict) else {})
        
        # Track events
        if _as_bool(info.get("hvaa_destroyed", False)) or not _as_bool(info.get("hvaa_alive", True)):
            ep_hvaa_destroyed = True
        
        if (_as_int(info.get("red_kills_this_step", 0)) > 0 or 
            _as_bool(info.get("red_killed_event", False)) or 
            _as_int(info.get("red_killed_total", 0)) > 0):
            ep_red_killed = True
        
        if _as_bool(info.get("Blue_Killed", False)) or _as_bool(info.get("blue_killed", False)):
            ep_blue_killed = True
        
        cur_reward += float(reward[0])
        cur_length += 1
        
        if bool(done[0]):
            episode = EpisodeData(
                reward=cur_reward,
                length=cur_length,
                hvaa_survived=not ep_hvaa_destroyed,
                red_killed=ep_red_killed,
                blue_killed=ep_blue_killed,
                blue_missiles_fired=_as_int(info.get("blue_missiles_fired", 0)),
                red_missiles_fired=_as_int(info.get("red_missiles_fired", 0)),
                blue_pk=float(info.get("blue_pk", 0.0)) if info.get("blue_pk") is not None else 0.0,
                red_pk=float(info.get("red_pk", 0.0)) if info.get("red_pk") is not None else 0.0,
                blue_shot_first=_as_bool(info.get("blue_shot_first", False)),
                red_shot_first=_as_bool(info.get("red_shot_first", False)),
                no_shots_fired=_as_bool(info.get("no_shots_fired", False)),
                min_separation_nm=float(info.get("min_separation_nm", -1.0)) if info.get("min_separation_nm") is not None else -1.0,
                end_reason=str(info.get("termination_reason", info.get("done_reason", ""))),
                training_seed=training_seed,
            )
            episodes.append(episode)
            finished += 1
            
            if verbose and (finished % 25 == 0 or finished == n_episodes):
                print(f"      {finished}/{n_episodes} episodes")
            
            # Reset for next episode
            cur_reward = 0.0
            cur_length = 0
            ep_hvaa_destroyed = False
            ep_red_killed = False
            ep_blue_killed = False
    
    eval_env.close()
    return episodes


def compute_seed_results(episodes: List[EpisodeData], seed_name: str, 
                         model_path: Path, vecnorm_path: Path) -> SeedResults:
    """Compute summary statistics for a single seed."""
    n = len(episodes)
    
    return SeedResults(
        seed_name=seed_name,
        model_path=str(model_path),
        vecnorm_path=str(vecnorm_path),
        n_episodes=n,
        mean_reward=float(np.mean([ep.reward for ep in episodes])),
        std_reward=float(np.std([ep.reward for ep in episodes], ddof=1)) if n > 1 else 0.0,
        hvaa_survival_rate=float(np.mean([ep.hvaa_survived for ep in episodes])),
        red_kill_rate=float(np.mean([ep.red_killed for ep in episodes])),
        blue_loss_rate=float(np.mean([ep.blue_killed for ep in episodes])),
        mean_length=float(np.mean([ep.length for ep in episodes])),
        mean_blue_missiles=float(np.mean([ep.blue_missiles_fired for ep in episodes])),
        mean_red_missiles=float(np.mean([ep.red_missiles_fired for ep in episodes])),
        mean_blue_pk=float(np.mean([ep.blue_pk for ep in episodes])),
        mean_red_pk=float(np.mean([ep.red_pk for ep in episodes])),
        blue_shot_first_rate=float(np.mean([ep.blue_shot_first for ep in episodes])),
        red_shot_first_rate=float(np.mean([ep.red_shot_first for ep in episodes])),
        no_shots_rate=float(np.mean([ep.no_shots_fired for ep in episodes])),
        episodes=episodes,
    )


def compute_aggregate_results(seed_results: List[SeedResults]) -> Dict:
    """
    Compute aggregate statistics across training seeds.
    
    For superlative policy comparison, we aggregate the per-seed means
    to get cross-seed statistics with proper confidence intervals.
    """
    n_seeds = len(seed_results)
    
    # Extract per-seed means for aggregation
    seed_mean_rewards = [sr.mean_reward for sr in seed_results]
    seed_hvaa_rates = [sr.hvaa_survival_rate for sr in seed_results]
    seed_red_kill_rates = [sr.red_kill_rate for sr in seed_results]
    seed_blue_loss_rates = [sr.blue_loss_rate for sr in seed_results]
    seed_mean_lengths = [sr.mean_length for sr in seed_results]
    seed_blue_missiles = [sr.mean_blue_missiles for sr in seed_results]
    seed_red_missiles = [sr.mean_red_missiles for sr in seed_results]
    seed_blue_pk = [sr.mean_blue_pk for sr in seed_results]
    seed_red_pk = [sr.mean_red_pk for sr in seed_results]
    seed_blue_first = [sr.blue_shot_first_rate for sr in seed_results]
    seed_red_first = [sr.red_shot_first_rate for sr in seed_results]
    seed_no_shots = [sr.no_shots_rate for sr in seed_results]
    
    # Compute confidence intervals across seeds
    return {
        "n_seeds": n_seeds,
        "episodes_per_seed": seed_results[0].n_episodes if seed_results else 0,
        "total_episodes": sum(sr.n_episodes for sr in seed_results),
        "cross_seed_statistics": {
            "mean_reward": asdict(t_distribution_ci(seed_mean_rewards)),
            "hvaa_survival_rate": asdict(t_distribution_ci(seed_hvaa_rates)),
            "red_kill_rate": asdict(t_distribution_ci(seed_red_kill_rates)),
            "blue_loss_rate": asdict(t_distribution_ci(seed_blue_loss_rates)),
            "mean_length": asdict(t_distribution_ci(seed_mean_lengths)),
            "mean_blue_missiles": asdict(t_distribution_ci(seed_blue_missiles)),
            "mean_red_missiles": asdict(t_distribution_ci(seed_red_missiles)),
            "mean_blue_pk": asdict(t_distribution_ci(seed_blue_pk)),
            "mean_red_pk": asdict(t_distribution_ci(seed_red_pk)),
            "blue_shot_first_rate": asdict(t_distribution_ci(seed_blue_first)),
            "red_shot_first_rate": asdict(t_distribution_ci(seed_red_first)),
            "no_shots_rate": asdict(t_distribution_ci(seed_no_shots)),
        },
    }


def print_results(seed_results: List[SeedResults], aggregate: Dict):
    """Print formatted results table."""
    n_seeds = len(seed_results)
    
    print("\n" + "="*90)
    print(f"{'SUPERLATIVE POLICY EVALUATION RESULTS':^90}")
    print("="*90)
    print(f"Training seeds evaluated: {n_seeds}")
    print(f"Episodes per seed: {aggregate['episodes_per_seed']}")
    print(f"Total episodes: {aggregate['total_episodes']}")
    print("="*90)
    
    # Per-seed breakdown
    print(f"\n{'PER-SEED RESULTS':^90}")
    print("-"*90)
    print(f"{'Seed':<12} {'Mean Reward':>14} {'Std':>10} {'HVAA %':>10} {'Red Kill %':>12} {'Blue Loss %':>12}")
    print("-"*90)
    
    for sr in sorted(seed_results, key=lambda x: x.seed_name):
        print(f"{sr.seed_name:<12} {sr.mean_reward:>14.2f} {sr.std_reward:>10.2f} "
              f"{sr.hvaa_survival_rate*100:>9.1f}% {sr.red_kill_rate*100:>11.1f}% "
              f"{sr.blue_loss_rate*100:>11.1f}%")
    
    print("="*90)
    
    # Cross-seed aggregates
    print(f"\n{'CROSS-SEED AGGREGATE STATISTICS (95% CI)':^90}")
    print("-"*90)
    
    cs = aggregate["cross_seed_statistics"]
    
    def format_ci(metric_dict, is_rate=False):
        mean = metric_dict["mean"]
        ci_lower = metric_dict["ci_lower"]
        ci_upper = metric_dict["ci_upper"]
        hw = (ci_upper - ci_lower) / 2
        if is_rate:
            return f"{mean*100:.1f}% {PM} {hw*100:.1f}%  [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]"
        else:
            return f"{mean:.2f} {PM} {hw:.2f}  [{ci_lower:.2f}, {ci_upper:.2f}]"
    
    print(f"{'Metric':<30} {'Mean ± HW':>25} {'95% CI':>25}")
    print("-"*90)
    
    metrics = [
        ("Mean Reward", "mean_reward", False),
        ("HVAA Survival Rate", "hvaa_survival_rate", True),
        ("Red Kill Rate", "red_kill_rate", True),
        ("Blue Loss Rate", "blue_loss_rate", True),
        ("Mean Episode Length", "mean_length", False),
        ("Mean Blue Missiles", "mean_blue_missiles", False),
        ("Mean Red Missiles", "mean_red_missiles", False),
        ("Mean Blue PK", "mean_blue_pk", False),
        ("Mean Red PK", "mean_red_pk", False),
        ("Blue Shot First Rate", "blue_shot_first_rate", True),
        ("Red Shot First Rate", "red_shot_first_rate", True),
        ("No Shots Rate", "no_shots_rate", True),
    ]
    
    for name, key, is_rate in metrics:
        m = cs[key]
        mean = m["mean"]
        hw = (m["ci_upper"] - m["ci_lower"]) / 2
        if is_rate:
            mean_str = f"{mean*100:.1f}% {PM} {hw*100:.1f}%"
            ci_str = f"[{m['ci_lower']*100:.1f}%, {m['ci_upper']*100:.1f}%]"
        else:
            mean_str = f"{mean:.2f} {PM} {hw:.2f}"
            ci_str = f"[{m['ci_lower']:.2f}, {m['ci_upper']:.2f}]"
        print(f"{name:<30} {mean_str:>25} {ci_str:>25}")
    
    print("="*90 + "\n")


def build_feature_config(args) -> Dict[str, Any]:
    """
    Build feature configuration from args.
    
    Note: When --ablation-config is provided (e.g., 'geometry_only'), 
    it's passed directly to the EnrichedObservationWrapper which handles
    the feature category selection internally. The feature_config dict
    returned here is mainly for logging/documentation purposes.
    """
    # Default feature config (used when no ablation_config specified)
    feature_config = {
        'pursuer_speed': 1.0,
        'evader_speed': 1.0,
        'capture_radius': 0.01,
        'pursuer_range': 0.5,
        'normalize_distance': 1.0,
        # Feature enable flags (these map to wrapper's internal categories)
        'enable_apollonius': True,
        'enable_bez': True,
        'enable_dmc': True,
        'enable_multi_threat': False,
        'enable_hvaa_escort': False,
        'enable_offense_wez': True,
        'enable_offense_ttc': True,
        'enable_reward_shaping': False,
    }
    
    # If ablation_config is provided, document what features it enables
    # (actual feature selection is handled by the wrapper)
    if args.ablation_config:
        ablation_feature_map = {
            'all': {'enable_apollonius': True, 'enable_bez': True, 'enable_dmc': True, 
                    'enable_offense_wez': True, 'enable_atddg': True, 'enable_range_limited': True},
            'none': {'enable_apollonius': False, 'enable_bez': False, 'enable_dmc': False,
                     'enable_offense_wez': False, 'enable_atddg': False, 'enable_range_limited': False},
            'geometry_only': {'enable_apollonius': True, 'enable_atddg': True,
                              'enable_bez': False, 'enable_dmc': False, 'enable_offense_wez': False,
                              'enable_range_limited': False},
            'engagement_only': {'enable_apollonius': False, 'enable_atddg': False,
                                'enable_bez': True, 'enable_dmc': True, 'enable_offense_wez': True,
                                'enable_range_limited': False},
            'range_limited_only': {'enable_apollonius': False, 'enable_atddg': False,
                                   'enable_bez': False, 'enable_dmc': False, 'enable_offense_wez': False,
                                   'enable_range_limited': True},
            'geometry_engagement': {'enable_apollonius': True, 'enable_atddg': True,
                                    'enable_bez': True, 'enable_dmc': True, 'enable_offense_wez': True,
                                    'enable_range_limited': False},
            'geometry_range': {'enable_apollonius': True, 'enable_atddg': True,
                               'enable_bez': False, 'enable_dmc': False, 'enable_offense_wez': False,
                               'enable_range_limited': True},
            'engagement_range': {'enable_apollonius': False, 'enable_atddg': False,
                                 'enable_bez': True, 'enable_dmc': True, 'enable_offense_wez': True,
                                 'enable_range_limited': True},
        }
        if args.ablation_config in ablation_feature_map:
            feature_config.update(ablation_feature_map[args.ablation_config])
    
    return feature_config


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate superlative (best) policies from each training seed in an experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all seeds in an experiment directory (auto-detect obs mode):
  python evaluate_experiment_seeds.py --experiment-dir runs_sb3/my_experiment

  # Evaluate specific seeds:
  python evaluate_experiment_seeds.py --experiment-dir runs_sb3/my_experiment --seeds seed42,seed123

  # Baseline (27-dim) evaluation:
  python evaluate_experiment_seeds.py --experiment-dir runs_sb3/baseline_exp --obs-mode baseline

  # Enriched with ALL features (GEOMETRY + ENGAGEMENT + RANGE_LIMITED = 40-dim):
  python evaluate_experiment_seeds.py --experiment-dir runs_sb3/enriched_all_exp \\
      --use-enriched-obs --ablation-config all

  # Ablation: GEOMETRY only (Apollonius + ATDDG = 33-dim):
  python evaluate_experiment_seeds.py --experiment-dir runs_sb3/geometry_exp \\
      --use-enriched-obs --ablation-config geometry_only

  # Ablation: ENGAGEMENT only (BEZ + DMC + WEZ = 32-dim):
  python evaluate_experiment_seeds.py --experiment-dir runs_sb3/engagement_exp \\
      --use-enriched-obs --ablation-config engagement_only

  # Ablation: GEOMETRY + ENGAGEMENT (no range-limited):
  python evaluate_experiment_seeds.py --experiment-dir runs_sb3/geo_engage_exp \\
      --use-enriched-obs --ablation-config geometry_engagement

Available ablation configs:
  'all'                 - All features (GEOMETRY + ENGAGEMENT + RANGE_LIMITED)
  'none'                - No enriched features (same as baseline)
  'geometry_only'       - GEOMETRY only (Apollonius + ATDDG)
  'engagement_only'     - ENGAGEMENT only (BEZ + DMC + WEZ)
  'range_limited_only'  - RANGE_LIMITED only
  'geometry_engagement' - GEOMETRY + ENGAGEMENT
  'geometry_range'      - GEOMETRY + RANGE_LIMITED
  'engagement_range'    - ENGAGEMENT + RANGE_LIMITED
        """
    )
    
    # Experiment directory (required)
    p.add_argument("--experiment-dir", type=str, required=True,
                   help="Directory containing seed subdirectories (e.g., runs_sb3/my_experiment)")
    
    # Seed selection (optional)
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list of seeds to include (e.g., seed42,seed123). "
                        "If not specified, all seeds found are included.")
    
    # Episodes
    p.add_argument("--episodes", type=int, default=100,
                   help="Episodes to run per training seed (default: 100)")
    
    # Eval seed (fixed for reproducibility)
    p.add_argument("--eval-seed", type=int, default=42,
                   help="Random seed for evaluation environment (default: 42)")
    
    # Observation mode
    obs_group = p.add_argument_group("Observation Configuration")
    obs_group.add_argument("--use-enriched-obs", action="store_true", default=False,
                           help="Use enriched observations (default: auto-detect from vecnormalize)")
    obs_group.add_argument("--obs-mode", type=str, choices=["auto", "baseline", "enriched", "enriched_extended"],
                           default="auto",
                           help="Observation mode: 'auto' detects from vecnormalize, or force a specific mode")
    
    # Ablation configuration
    ablation_group = p.add_argument_group("Ablation Configuration")
    ablation_group.add_argument("--ablation-config", type=str, default=None,
                                help="Ablation config name from EnrichedObservationWrapper. "
                                     "Options: 'all', 'none', 'geometry_only', 'engagement_only', "
                                     "'range_limited_only', 'geometry_engagement', 'geometry_range', "
                                     "'engagement_range'. This takes precedence over individual feature flags.")
    
    # Individual feature flags (only used when --ablation-config is NOT specified)
    feature_group = p.add_argument_group("Feature Flags (only used when --ablation-config is NOT specified)")
    feature_group.add_argument("--enable-apollonius", action="store_true", dest="enable_apollonius", default=None,
                               help="Enable Apollonius circle features (part of GEOMETRY category)")
    feature_group.add_argument("--no-enable-apollonius", action="store_false", dest="enable_apollonius",
                               help="Disable Apollonius circle features")
    feature_group.add_argument("--enable-bez", action="store_true", dest="enable_bez", default=None,
                               help="Enable Basic Engagement Zone features (part of ENGAGEMENT category)")
    feature_group.add_argument("--no-enable-bez", action="store_false", dest="enable_bez",
                               help="Disable Basic Engagement Zone features")
    feature_group.add_argument("--enable-dmc", action="store_true", dest="enable_dmc", default=None,
                               help="Enable Dynamic Maneuvering Cue features (part of ENGAGEMENT category)")
    feature_group.add_argument("--no-enable-dmc", action="store_false", dest="enable_dmc",
                               help="Disable Dynamic Maneuvering Cue features")
    feature_group.add_argument("--enable-offense-wez", action="store_true", dest="enable_offense_wez", default=None,
                               help="Enable offensive WEZ features (part of ENGAGEMENT category)")
    feature_group.add_argument("--no-enable-offense-wez", action="store_false", dest="enable_offense_wez",
                               help="Disable offensive WEZ features")
    feature_group.add_argument("--enable-offense-ttc", action="store_true", dest="enable_offense_ttc", default=None,
                               help="Enable offensive TTC features")
    feature_group.add_argument("--no-enable-offense-ttc", action="store_false", dest="enable_offense_ttc",
                               help="Disable offensive TTC features")
    feature_group.add_argument("--enable-reward-shaping", action="store_true", dest="enable_reward_shaping", default=None,
                               help="Enable reward shaping in wrapper")
    feature_group.add_argument("--no-enable-reward-shaping", action="store_false", dest="enable_reward_shaping",
                               help="Disable reward shaping in wrapper")
    
    # Config
    p.add_argument("--config", type=str, default=None,
                   help="B-ACE config path (default: auto-detect from repo)")
    
    # Environment
    p.add_argument("--renderize", type=int, default=0)
    p.add_argument("--speed-up", type=int, default=200)
    
    # Output
    p.add_argument("--output-file", type=str, default=None,
                   help="Output JSON path (default: <experiment_dir>/superlative_eval_results.json)")
    
    # Validation
    p.add_argument("--no-strict-validation", action="store_true", default=False,
                   help="Continue even if observation dimension doesn't match vecnormalize (NOT RECOMMENDED)")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Resolve experiment directory
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.exists():
        raise SystemExit(f"Directory not found: {experiment_dir}")
    
    print("\n" + "="*90)
    print("SUPERLATIVE POLICY EVALUATION")
    print("="*90)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Episodes per seed: {args.episodes}")
    print(f"Evaluation seed: {args.eval_seed}")
    print("="*90 + "\n")
    
    # Find seed models
    seed_models = find_seed_models(experiment_dir)
    if not seed_models:
        raise SystemExit(f"No seed models found in {experiment_dir}")
    
    # Filter to requested seeds if specified
    if args.seeds:
        requested_seeds = [s.strip() for s in args.seeds.split(',')]
        filtered_models = {k: v for k, v in seed_models.items() if k in requested_seeds}
        
        # Check for seeds that weren't found
        found_seeds = set(filtered_models.keys())
        missing_seeds = set(requested_seeds) - found_seeds
        if missing_seeds:
            print(f"Warning: Seeds not found: {missing_seeds}")
        
        if not filtered_models:
            raise SystemExit(f"None of the requested seeds found: {requested_seeds}")
        
        seed_models = filtered_models
        print(f"\nFiltered to seeds: {list(seed_models.keys())}")
    
    print(f"\nWill evaluate {len(seed_models)} seeds: {sorted(seed_models.keys())}")
    
    # Load config
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        config_path = REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json"
    
    bace_cfg = load_b_ace_config(config_path.as_posix())
    if bace_cfg is None:
        raise RuntimeError(f"Failed to load config: {config_path}")
    
    env_cfg = bace_cfg.setdefault("EnvConfig", {})
    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = _resolve_env_path()
    
    env_path = Path(env_cfg["env_path"]).expanduser()
    if not env_path.is_absolute():
        env_path = (REPO_ROOT / env_path).resolve()
    if not env_path.exists():
        env_cfg["env_path"] = _resolve_env_path()
    else:
        env_cfg["env_path"] = env_path.as_posix()
    
    env_cfg["renderize"] = args.renderize
    env_cfg["speed_up"] = args.speed_up
    
    agents_cfg = bace_cfg.setdefault("AgentsConfig", {})
    agents_cfg.setdefault("blue_agents", {})["base_behavior"] = "external"
    agents_cfg.setdefault("red_agents", {})["base_behavior"] = "baseline1"
    
    # Build feature configuration from args
    feature_config = build_feature_config(args)
    
    # Print configuration summary
    print(f"\nObservation Configuration:")
    if args.obs_mode != "auto":
        print(f"   Mode: {args.obs_mode} (explicit)")
    elif args.use_enriched_obs:
        print(f"   Mode: enriched (--use-enriched-obs flag)")
    else:
        print(f"   Mode: auto-detect from vecnormalize")
    
    if args.use_enriched_obs or args.obs_mode in ["enriched", "enriched_extended"]:
        if args.ablation_config:
            print(f"   Ablation config: '{args.ablation_config}'")
            # Show expected feature categories
            category_map = {
                'all': ['GEOMETRY', 'ENGAGEMENT', 'RANGE_LIMITED'],
                'none': [],
                'geometry_only': ['GEOMETRY'],
                'engagement_only': ['ENGAGEMENT'],
                'range_limited_only': ['RANGE_LIMITED'],
                'geometry_engagement': ['GEOMETRY', 'ENGAGEMENT'],
                'geometry_range': ['GEOMETRY', 'RANGE_LIMITED'],
                'engagement_range': ['ENGAGEMENT', 'RANGE_LIMITED'],
            }
            if args.ablation_config in category_map:
                cats = category_map[args.ablation_config]
                print(f"   Expected categories: {cats if cats else ['(none - baseline equivalent)']}")
        else:
            print(f"   Ablation config: (default - all features)")
            enabled_features = [k for k, v in feature_config.items() if k.startswith('enable_') and v]
            print(f"   Enabled features: {enabled_features}")
    
    # Evaluate each seed
    all_seed_results: List[SeedResults] = []
    
    for i, (seed_name, (model_path, vecnorm_path)) in enumerate(sorted(seed_models.items())):
        print(f"\n[{i+1}/{len(seed_models)}] Evaluating {seed_name}...")
        print(f"   Model: {model_path}")
        print(f"   VecNormalize: {vecnorm_path}")
        
        episodes = evaluate_model(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            bace_cfg=bace_cfg,
            n_episodes=args.episodes,
            training_seed=seed_name,
            eval_seed=args.eval_seed,
            obs_mode_override=args.obs_mode,
            use_enriched_obs=args.use_enriched_obs,
            feature_config=feature_config,
            ablation_config=args.ablation_config,
            verbose=True,
            strict_validation=not args.no_strict_validation,
        )
        
        seed_result = compute_seed_results(episodes, seed_name, model_path, vecnorm_path)
        all_seed_results.append(seed_result)
        
        print(f"   {seed_name} summary: reward={seed_result.mean_reward:.2f} {PM} {seed_result.std_reward:.2f}, "
              f"HVAA={seed_result.hvaa_survival_rate*100:.1f}%, Red Kill={seed_result.red_kill_rate*100:.1f}%")
    
    # Compute aggregate results
    aggregate = compute_aggregate_results(all_seed_results)
    
    # Print results
    print_results(all_seed_results, aggregate)
    
    # Prepare output
    output_data = {
        "experiment_dir": str(experiment_dir),
        "evaluation_config": {
            "episodes_per_seed": args.episodes,
            "eval_seed": args.eval_seed,
            "config_path": str(config_path),
            "obs_mode": args.obs_mode,
            "use_enriched_obs": args.use_enriched_obs,
            "ablation_config": args.ablation_config,
            "feature_config": feature_config,
        },
        "aggregate": aggregate,
        "per_seed_results": {
            sr.seed_name: {
                "model_path": sr.model_path,
                "vecnorm_path": sr.vecnorm_path,
                "n_episodes": sr.n_episodes,
                "mean_reward": sr.mean_reward,
                "std_reward": sr.std_reward,
                "hvaa_survival_rate": sr.hvaa_survival_rate,
                "red_kill_rate": sr.red_kill_rate,
                "blue_loss_rate": sr.blue_loss_rate,
                "mean_length": sr.mean_length,
                "mean_blue_missiles": sr.mean_blue_missiles,
                "mean_red_missiles": sr.mean_red_missiles,
                "mean_blue_pk": sr.mean_blue_pk,
                "mean_red_pk": sr.mean_red_pk,
                "blue_shot_first_rate": sr.blue_shot_first_rate,
                "red_shot_first_rate": sr.red_shot_first_rate,
                "no_shots_rate": sr.no_shots_rate,
                # Include per-episode data for detailed analysis
                "episodes": [asdict(ep) for ep in sr.episodes],
            }
            for sr in all_seed_results
        },
    }
    
    # Save JSON
    if args.output_file:
        out_path = Path(args.output_file).expanduser().resolve()
    else:
        out_path = experiment_dir / "superlative_eval_results.json"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(_jsonify(output_data), f, indent=2)
    
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
