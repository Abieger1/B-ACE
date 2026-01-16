#!/usr/bin/env python3
"""
evaluate_single_model_multi_seed.py

Evaluate a SINGLE trained model across multiple evaluation seeds to compute
95% confidence intervals.

This is for when you have ONE trained model and want to assess its performance
variance by running it with different environment seeds.

Usage:
    # Basic: 3 eval seeds, 100 episodes each
    python evaluate_single_model_multi_seed.py \
        --model-path runs_sb3/my_experiment/best_model.zip \
        --eval-seeds 42 43 44 \
        --episodes 100

    # More seeds for tighter CIs
    python evaluate_single_model_multi_seed.py \
        --model-path runs_sb3/my_experiment/best_model.zip \
        --eval-seeds 42 43 44 45 46 \
        --episodes 100

    # Auto-generate 5 seeds starting from 100
    python evaluate_single_model_multi_seed.py \
        --model-path runs_sb3/my_experiment/best_model.zip \
        --num-eval-seeds 5 \
        --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import stats

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    eval_seed: int = 0


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
    return bool(v) if isinstance(v, bool) else (v > 0.5 if isinstance(v, (int, float)) else False)


def _as_int(v) -> int:
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
# Main evaluation
# -----------------------------
def evaluate_with_seed(
    model: PPO,
    vecnorm_path: Path,
    bace_cfg: Dict,
    eval_seed: int,
    n_episodes: int,
    use_enriched_obs: bool,
    verbose: bool = True,
) -> List[EpisodeData]:
    """Run evaluation episodes with a specific seed."""
    
    def make_env():
        def _thunk():
            e = SingleAgentBACEEnv(bace_config=bace_cfg, controlled_agent_id="agent_0", max_episode_steps=14000)
            
            if use_enriched_obs:
                e = EnrichedObservationWrapper(
                    e, obs_labels=None,
                    feature_config={
                        'pursuer_speed': 1.0, 'evader_speed': 1.0, 'capture_radius': 0.01,
                        'pursuer_range': 0.5, 'normalize_distance': 1.0,
                        'enable_apollonius': True, 'enable_bez': True, 'enable_dmc': True,
                        'enable_multi_threat': False, 'enable_hvaa_escort': False,
                        'enable_offense_wez': True, 'enable_offense_ttc': True,
                        'enable_reward_shaping': False,
                    },
                    debug=False
                )
            
            e = FireCooldownWrapper(e, cooldown_steps=120, fire_idx=3, threshold=0.5)
            e = HVAASurvivalEveryNStepsBonus(e, every_n_steps=50, bonus=0.05)
            return e
        return _thunk

    venv = DummyVecEnv([make_env()])
    eval_env = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env = VecNormalize.load(vecnorm_path.as_posix(), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # Set the model's environment
    model.set_env(eval_env)

    episodes: List[EpisodeData] = []
    
    # Reset with specific seed
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
        if _as_int(info.get("red_kills_this_step", 0)) > 0 or _as_bool(info.get("red_killed_event", False)) or _as_int(info.get("red_killed_total", 0)) > 0:
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
                eval_seed=eval_seed,
            )
            episodes.append(episode)
            finished += 1
            
            if verbose and (finished % 25 == 0 or finished == n_episodes):
                print(f"      Seed {eval_seed}: {finished}/{n_episodes} episodes")
            
            # Reset for next episode
            cur_reward = 0.0
            cur_length = 0
            ep_hvaa_destroyed = False
            ep_red_killed = False
            ep_blue_killed = False
    
    eval_env.close()
    return episodes


def compute_results(all_episodes: List[EpisodeData], eval_seeds: List[int], episodes_per_seed: int) -> Dict:
    """Compute all metrics with confidence intervals."""
    n = len(all_episodes)
    
    # Extract arrays
    rewards = [ep.reward for ep in all_episodes]
    lengths = [float(ep.length) for ep in all_episodes]
    hvaa = [ep.hvaa_survived for ep in all_episodes]
    red_killed = [ep.red_killed for ep in all_episodes]
    blue_killed = [ep.blue_killed for ep in all_episodes]
    blue_missiles = [float(ep.blue_missiles_fired) for ep in all_episodes]
    red_missiles = [float(ep.red_missiles_fired) for ep in all_episodes]
    blue_pk = [ep.blue_pk for ep in all_episodes]
    red_pk = [ep.red_pk for ep in all_episodes]
    blue_first = [ep.blue_shot_first for ep in all_episodes]
    red_first = [ep.red_shot_first for ep in all_episodes]
    no_shots = [ep.no_shots_fired for ep in all_episodes]
    
    # Per-seed stats
    per_seed = {}
    for seed in eval_seeds:
        seed_eps = [ep for ep in all_episodes if ep.eval_seed == seed]
        per_seed[seed] = {
            "mean_reward": float(np.mean([ep.reward for ep in seed_eps])),
            "hvaa_survival_rate": float(np.mean([ep.hvaa_survived for ep in seed_eps])),
            "red_kill_rate": float(np.mean([ep.red_killed for ep in seed_eps])),
            "blue_loss_rate": float(np.mean([ep.blue_killed for ep in seed_eps])),
        }
    
    return {
        "summary": {
            "n_eval_seeds": len(eval_seeds),
            "episodes_per_seed": episodes_per_seed,
            "total_episodes": n,
            "eval_seeds": eval_seeds,
        },
        "confidence_intervals": {
            "mean_reward": asdict(t_distribution_ci(rewards)),
            "mean_length": asdict(t_distribution_ci(lengths)),
            "hvaa_survival_rate": asdict(wilson_score_interval(sum(hvaa), n)),
            "red_kill_rate": asdict(wilson_score_interval(sum(red_killed), n)),
            "blue_loss_rate": asdict(wilson_score_interval(sum(blue_killed), n)),
            "mean_blue_missiles_fired": asdict(t_distribution_ci(blue_missiles)),
            "mean_red_missiles_fired": asdict(t_distribution_ci(red_missiles)),
            "mean_blue_pk": asdict(t_distribution_ci(blue_pk)),
            "mean_red_pk": asdict(t_distribution_ci(red_pk)),
            "blue_shot_first_rate": asdict(wilson_score_interval(sum(blue_first), n)),
            "red_shot_first_rate": asdict(wilson_score_interval(sum(red_first), n)),
            "no_shots_fired_rate": asdict(wilson_score_interval(sum(no_shots), n)),
        },
        "per_seed_summary": per_seed,
        "per_episode_data": [asdict(ep) for ep in all_episodes],
    }


def print_results(results: Dict):
    """Print formatted results table."""
    ci = results["confidence_intervals"]
    summary = results["summary"]
    
    print("\n" + "="*80)
    print(f"{'EVALUATION RESULTS WITH 95% CONFIDENCE INTERVALS':^80}")
    print("="*80)
    print(f"Model evaluated with {summary['n_eval_seeds']} evaluation seeds")
    print(f"Episodes per seed: {summary['episodes_per_seed']}")
    print(f"Total episodes: {summary['total_episodes']}")
    print(f"Eval seeds: {summary['eval_seeds']}")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Mean':>12} {'95% CI':>25} {'n':>8}")
    print("-"*80)
    
    rate_metrics = [
        ("HVAA Survival Rate", "hvaa_survival_rate"),
        ("Red Kill Rate", "red_kill_rate"),
        ("Blue Loss Rate", "blue_loss_rate"),
        ("Blue Shot First Rate", "blue_shot_first_rate"),
        ("Red Shot First Rate", "red_shot_first_rate"),
        ("No Shots Fired Rate", "no_shots_fired_rate"),
    ]
    
    for name, key in rate_metrics:
        m = ci[key]
        ci_str = "[{:.1f}%, {:.1f}%]".format(m['ci_lower']*100, m['ci_upper']*100)
        print(f"{name:<30} {m['mean']*100:>11.1f}% {ci_str:>25} {m['n']:>8}")
    
    print("-"*80)
    
    cont_metrics = [
        ("Mean Reward", "mean_reward"),
        ("Mean Episode Length", "mean_length"),
        ("Mean Blue Missiles Fired", "mean_blue_missiles_fired"),
        ("Mean Red Missiles Fired", "mean_red_missiles_fired"),
        ("Mean Blue PK", "mean_blue_pk"),
        ("Mean Red PK", "mean_red_pk"),
    ]
    
    for name, key in cont_metrics:
        m = ci[key]
        ci_str = "[{:.2f}, {:.2f}]".format(m['ci_lower'], m['ci_upper'])
        print(f"{name:<30} {m['mean']:>12.2f} {ci_str:>25} {m['n']:>8}")
    
    print("="*80)
    
    # Per-seed breakdown
    print(f"\n{'Per-Seed Breakdown':^80}")
    print("-"*80)
    print(f"{'Eval Seed':<12} {'Mean Reward':>14} {'HVAA Surv':>12} {'Red Kill':>12} {'Blue Loss':>12}")
    print("-"*80)
    for seed, data in results["per_seed_summary"].items():
        print(f"{seed:<12} {data['mean_reward']:>14.2f} {data['hvaa_survival_rate']*100:>11.1f}% {data['red_kill_rate']*100:>11.1f}% {data['blue_loss_rate']*100:>11.1f}%")
    print("="*80 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a single model with multiple evaluation seeds")
    
    # Model path (required)
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to best_model.zip")
    p.add_argument("--vecnormalize-path", type=str, default=None,
                   help="Path to vecnormalize.pkl (auto-detected if not provided)")
    
    # Evaluation seeds (choose one)
    seed_group = p.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--eval-seeds", type=int, nargs='+',
                           help="List of evaluation seeds (e.g., --eval-seeds 42 43 44)")
    seed_group.add_argument("--num-eval-seeds", type=int,
                           help="Auto-generate N seeds starting from 100")
    
    # Episodes
    p.add_argument("--episodes", type=int, default=100,
                   help="Episodes PER evaluation seed (default: 100)")
    
    # Config
    p.add_argument("--config", type=str,
                   default=(REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json").as_posix(),
                   help="B-ACE config path")
    p.add_argument("--no-enriched-obs", dest="use_enriched_obs", action="store_false")
    p.set_defaults(use_enriched_obs=True)
    
    # Environment
    p.add_argument("--renderize", type=int, default=0)
    p.add_argument("--speed-up", type=int, default=200)
    
    # Output
    p.add_argument("--out-json", type=str, default=None,
                   help="Output path (default: <model_dir>/eval_multi_seed_results.json)")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Resolve model path
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        # Try adding .zip
        if model_path.with_suffix('.zip').exists():
            model_path = model_path.with_suffix('.zip')
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    run_dir = model_path.parent
    
    # Find vecnormalize
    if args.vecnormalize_path:
        vecnorm_path = Path(args.vecnormalize_path).expanduser().resolve()
    else:
        candidates = [
            run_dir / "best_vecnormalize.pkl",
            run_dir / "vecnormalize.pkl",
            run_dir / "eval_checkpoints" / "best_vecnormalize.pkl",
        ]
        vecnorm_path = None
        for c in candidates:
            if c.exists():
                vecnorm_path = c
                break
        if vecnorm_path is None:
            raise FileNotFoundError(f"No vecnormalize.pkl found in {run_dir}")
    
    # Determine eval seeds
    if args.eval_seeds:
        eval_seeds = args.eval_seeds
    else:
        eval_seeds = list(range(100, 100 + args.num_eval_seeds))
    
    print("\n" + "="*80)
    print("SINGLE MODEL MULTI-SEED EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print(f"Evaluation seeds: {eval_seeds}")
    print(f"Episodes per seed: {args.episodes}")
    print(f"Total episodes: {len(eval_seeds) * args.episodes}")
    print("="*80 + "\n")
    
    # Load config
    config_path = Path(args.config).expanduser().resolve()
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
    
    # Load model once
    print("Loading model...")
    model = PPO.load(model_path.as_posix(), device="cpu")
    
    # Evaluate with each seed
    all_episodes: List[EpisodeData] = []
    
    for i, eval_seed in enumerate(eval_seeds):
        print(f"\n[{i+1}/{len(eval_seeds)}] Evaluating with seed {eval_seed}...")
        
        episodes = evaluate_with_seed(
            model=model,
            vecnorm_path=vecnorm_path,
            bace_cfg=bace_cfg,
            eval_seed=eval_seed,
            n_episodes=args.episodes,
            use_enriched_obs=args.use_enriched_obs,
            verbose=True,
        )
        
        mean_r = np.mean([ep.reward for ep in episodes])
        hvaa_rate = np.mean([ep.hvaa_survived for ep in episodes])
        print(f"      Seed {eval_seed} summary: reward={mean_r:.2f}, HVAA={hvaa_rate*100:.1f}%")
        
        all_episodes.extend(episodes)
    
    # Compute results
    print(f"\nComputing confidence intervals from {len(all_episodes)} episodes...")
    results = compute_results(all_episodes, eval_seeds, args.episodes)
    
    # Print results
    print_results(results)
    
    # Save JSON
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
    else:
        out_path = run_dir / "eval_multi_seed_results.json"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add model info
    results["model_info"] = {
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnorm_path),
    }
    
    with open(out_path, 'w') as f:
        json.dump(_jsonify(results), f, indent=2)
    
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
