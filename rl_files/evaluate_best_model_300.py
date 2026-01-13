#!/usr/bin/env python3
"""
evaluate_best_model_300_fixed_names.py

Same as your current evaluate_best_model_300.py (the one I generated),
but with the variable-name collision fixed:

- Per-episode arrays use *_arr suffix
- Per-episode boolean flags use *_flag suffix

This fixes:
    AttributeError: 'bool' object has no attribute 'append'

It also keeps the EpisodeOverTerminatorWrapper + per-step event tracking
so your episode boundaries + kill/survival rates are much more trustworthy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _jsonify(obj):
    """
    Recursively convert objects to JSON-serializable types.
    Handles numpy arrays/scalars and common containers.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    # fallback: stringify unknown objects
    return str(obj)

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# -----------------------------
# Repo root discovery (same style as training)
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

from b_ace_py.utils import load_b_ace_config
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
from b_ace_py.enriched_observation_wrapper import EnrichedObservationWrapper


# -----------------------------
# Wrappers copied from training (unchanged)
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

        bonus_now = 0.0
        if hvaa_alive and (self._t % self.every_n_steps == 0):
            bonus_now = self.bonus
            rew = float(rew) + bonus_now

        info["hvaa_survival_bonus"] = float(bonus_now)

        self._t += 1
        return obs, rew, terminated, truncated, info


class FireCooldownWrapper(gym.Wrapper):
    """
    Prevents firing more than once every `cooldown_steps`.
    Also uses rising-edge triggering so holding FIRE high does not repeatedly launch.
    """
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

        info["fire_blocked"] = bool(want_fire and not (rising_edge and (self._t - 1 - self._last_fire_t) == 0))
        info["fire_cooldown_remaining_steps"] = int(max(0, self.cooldown_steps - (self._t - 1 - self._last_fire_t)))
        return obs, rew, terminated, truncated, info


class NanGuardWrapper(gym.Wrapper):
    """
    Raises immediately if obs/reward becomes NaN/Inf.
    """
    def __init__(self, env, name="env", print_first_n=20):
        super().__init__(env)
        self.name = name
        self.step_i = 0
        self.print_first_n = int(print_first_n)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_i = 0
        self._check_obs(obs, where="reset", info=info)
        return obs, info

    def step(self, action):
        self.step_i += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._check_obs(obs, where="step", info=info)
        self._check_reward(reward, info=info)
        return obs, reward, terminated, truncated, info

    def _check_obs(self, obs, where, info):
        arr = np.asarray(obs, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            bad = np.where(~np.isfinite(arr))[0]
            print(f"\n[NAN-GUARD:{self.name}] Non-finite OBS at {where} step={self.step_i}")
            print(f"  bad indices (first {self.print_first_n}): {bad[:self.print_first_n].tolist()}")
            print(f"  bad values  (first {self.print_first_n}): {arr[bad[:self.print_first_n]]}")
            if isinstance(info, dict):
                print("  info keys:", list(info.keys())[:50])
            raise RuntimeError("Non-finite observation detected (NaN/Inf).")

    def _check_reward(self, reward, info):
        r = float(reward)
        if not np.isfinite(r):
            print(f"\n[NAN-GUARD:{self.name}] Non-finite REWARD at step={self.step_i}: {r}")
            if isinstance(info, dict):
                print("  info keys:", list(info.keys())[:50])
            raise RuntimeError("Non-finite reward detected (NaN/Inf).")


# -----------------------------
# Helpers
# -----------------------------
def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


class EpisodeOverTerminatorWrapper(gym.Wrapper):
    """Ends the Gymnasium episode when the underlying Godot sim reports episode_over."""
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if (not terminated) and (not truncated) and bool(info.get("episode_over", False)):
            terminated = True
            info = dict(info)
            info["terminated_by_episode_over"] = True
        return obs, rew, terminated, truncated, info


def _as_bool(x) -> bool:
    try:
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x) != 0.0
        return bool(x)
    except Exception:
        return False


def _as_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _extract_reason_str(info: dict) -> str:
    """Best-effort extraction of termination reasons from info.

    Different B-ACE/Godot versions expose episode-end reasons under different keys.
    We try a small set and normalize to a single string.
    """
    if not isinstance(info, dict):
        return ""

    for k in ("reasons", "reason", "termination_reason", "episode_end_reason", "end_reason"):
        if k in info:
            v = info.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, (list, tuple)):
                try:
                    return ";".join(str(x) for x in v)
                except Exception:
                    return str(v)
            return str(v)
    return ""


def _reason_implies_red_kill(reason: str) -> bool:
    r = (reason or "").lower()
    # include a few common variants
    return any(tok in r for tok in ("red_killed", "enemy_killed", "enemy_destroyed", "red_destroyed"))


def _reason_implies_hvaa_destroyed(reason: str) -> bool:
    r = (reason or "").lower()
    return any(tok in r for tok in ("hvaa_destroyed", "hvaa_killed", "hvaa_dead"))


def _pick_int_key(info: dict, keys: tuple[str, ...]) -> Optional[int]:
    """Return first parsable int among candidate keys, else None."""
    for k in keys:
        if k in info:
            try:
                return int(info.get(k))
            except Exception:
                continue
    return None


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


# -----------------------------
# Single-agent env copied from training (minimal, identical behavior)
# -----------------------------
class SingleAgentBACEEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        bace_config: Dict[str, Any],
        controlled_agent_id: str = "agent_0",
        max_episode_steps: int = 14000,
    ):
        super().__init__()
        self._max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

        self._ma_env = B_ACE_GodotPettingZooWrapper(device="cpu", **bace_config)

        self.observation_labels = getattr(self._ma_env, "observation_labels", None)
        if hasattr(self._ma_env, "get_hvaa_indices"):
            self.get_hvaa_indices = self._ma_env.get_hvaa_indices

        self.obs_map = getattr(self._ma_env, "obs_map", None)
        self.observation_labels_flat = getattr(self._ma_env, "observation_labels_flat", None)
        self.obs_labels = getattr(self._ma_env, "obs_labels", None)

        obs_dict, _info_dict = self._ma_env.reset()
        self.obs_map = getattr(self._ma_env, "obs_map", self.obs_map)
        self.observation_labels_flat = getattr(self._ma_env, "observation_labels_flat", self.observation_labels_flat)
        self._last_obs_dict = obs_dict

        agents = getattr(self._ma_env, "agents", [])
        if agents:
            self.controlled_agent_id = controlled_agent_id if controlled_agent_id in agents else agents[0]
        else:
            self.controlled_agent_id = controlled_agent_id

        raw_obs = obs_dict[self.controlled_agent_id] if self.controlled_agent_id in obs_dict else next(iter(obs_dict.values()))
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
                except (TypeError, ValueError):
                    continue
            raise RuntimeError(f"Could not extract obs from dict: {list(obj.keys())}")
        raise RuntimeError(f"Unsupported observation type {type(obj)}")

    def _extract_single_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        raw_obs = obs_dict[self.controlled_agent_id] if self.controlled_agent_id in obs_dict else next(iter(obs_dict.values()))
        return self._to_array(raw_obs)

    def _build_action_payload(self, agent_action: np.ndarray) -> Any:
        clipped = np.clip(agent_action, -1.0, 1.0).astype(np.float32)
        if hasattr(self._ma_env, "agents"):
            return {agent: clipped for agent in self._ma_env.agents}
        return {self.controlled_agent_id: clipped}

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._elapsed_steps = 0
        obs_dict, info_dict = self._ma_env.reset()
        self.obs_map = getattr(self._ma_env, "obs_map", self.obs_map)
        self.observation_labels_flat = getattr(self._ma_env, "observation_labels_flat", self.observation_labels_flat)
        self._last_obs_dict = obs_dict
        obs = self._extract_single_obs(obs_dict)
        info = info_dict.get(self.controlled_agent_id, {}) if isinstance(info_dict, dict) else {}
        return obs, info

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1
        action_dict = self._build_action_payload(action)
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._ma_env.step(action_dict)
        self._last_obs_dict = obs_dict

        obs = self._extract_single_obs(obs_dict)
        reward = float(rew_dict.get(self.controlled_agent_id, 0.0)) if isinstance(rew_dict, dict) else float(rew_dict)

        terminated = bool(term_dict.get(self.controlled_agent_id, False)) or bool(term_dict.get("__all__", False)) if isinstance(term_dict, dict) else bool(term_dict)
        truncated  = bool(trunc_dict.get(self.controlled_agent_id, False)) or bool(trunc_dict.get("__all__", False)) if isinstance(trunc_dict, dict) else bool(trunc_dict)

        if self._elapsed_steps >= self._max_episode_steps and not terminated:
            truncated = True

        info = info_dict.get(self.controlled_agent_id, {}) if isinstance(info_dict, dict) else {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if hasattr(self._ma_env, "close"):
            self._ma_env.close()


# -----------------------------
# CLI + evaluation
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True, help="Path to best_model.zip (or any SB3 .zip).")
    p.add_argument("--vecnormalize-path", "--vec-normalize-path", type=str, default=None,
                   help="Path to VecNormalize stats .pkl. If omitted, auto-detects in model directory.")
    p.add_argument("--config", type=str, default=(REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json").as_posix(),
                   help="Path to B-ACE JSON scenario config.")
    p.add_argument("--episodes", "--num-episodes", type=int, default=300, help="Number of evaluation episodes.")
    p.add_argument("--seed", type=int, default=12345, help="Eval seed (for env reset).")
    p.add_argument("--no-nanguard", action="store_true", help="Disable NanGuardWrapper.")
    p.add_argument("--no-enriched-obs", dest="use_enriched_obs", action="store_false", help="Disable enriched observations.")
    p.add_argument("--use-enriched-obs", dest="use_enriched_obs", action="store_true", help="Enable enriched observations.")
    p.set_defaults(use_enriched_obs=True)

    p.add_argument("--renderize", type=int, default=0, help="EnvConfig.renderize (0=headless, 1=render).")
    p.add_argument("--speed-up", type=int, default=200, help="EnvConfig.speed_up (Godot sim speed).")

    p.add_argument("--out-json", type=str, default=None,
                   help="Optional output JSON path. Default: <model_dir>/eval_300eps_<timestamp>.json")
    p.add_argument("--progress-every", type=int, default=10, help="Print progress every N finished episodes.")
    return p.parse_args()


def _ci95_mean(x: np.ndarray) -> float:
    n = len(x)
    if n <= 1:
        return 0.0
    return 1.96 * float(np.std(x, ddof=1)) / math.sqrt(n)


def main():
    args = parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_dir = model_path.parent

    # Auto-detect VecNormalize stats
    vecnorm_path: Optional[Path] = Path(args.vecnormalize_path).expanduser().resolve() if args.vecnormalize_path else None
    if vecnorm_path is None:
        cand1 = run_dir / "best_vecnormalize.pkl"
        cand2 = run_dir / "vecnormalize.pkl"
        cand3 = run_dir / "eval_checkpoints" / "best_vecnormalize.pkl"
        for c in (cand1, cand2, cand3):
            if c.exists():
                vecnorm_path = c
                break

    if vecnorm_path is None or not vecnorm_path.exists():
        raise FileNotFoundError(
            "Could not find VecNormalize stats. Provide --vecnormalize-path.\n"
            f"Looked in: {run_dir}"
        )

    # Load and sanitize config
    config_path = Path(args.config).expanduser().resolve()
    bace_cfg = load_b_ace_config(config_path.as_posix())
    if bace_cfg is None:
        raise RuntimeError(f"Failed to load config file at {config_path}")

    env_cfg = bace_cfg.setdefault("EnvConfig", {})
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
                f"[B-ACE ERROR] Tried {env_path_value}, and fallback {fallback_path}, but neither exists."
            )
        print(f"Warning: Godot binary not found at {env_path_value}. Falling back to {fallback_path}.")
        env_path_value = fallback_path

    env_cfg["env_path"] = env_path_value.as_posix()
    env_cfg["renderize"] = int(args.renderize)
    env_cfg["speed_up"] = int(args.speed_up)

    agents_cfg = bace_cfg.setdefault("AgentsConfig", {})
    blue_cfg = agents_cfg.setdefault("blue_agents", {})
    red_cfg = agents_cfg.setdefault("red_agents", {})
    blue_cfg.setdefault("base_behavior", "external")
    red_cfg.setdefault("base_behavior", "baseline1")

    def make_env(seed: int):
        def _thunk():
            e = SingleAgentBACEEnv(bace_config=bace_cfg, controlled_agent_id="agent_0", max_episode_steps=14000)

            if args.use_enriched_obs:
                e = EnrichedObservationWrapper(
                    e,
                    obs_labels=None,
                    feature_config={
                        "pursuer_speed": 1.0,
                        "evader_speed": 1.0,
                        "capture_radius": 0.01,
                        "pursuer_range": 0.5,
                        "normalize_distance": 1.0,
                        "enable_apollonius": True,
                        "enable_bez": True,
                        "enable_dmc": True,
                        "enable_multi_threat": False,
                        "enable_hvaa_escort": False,
                        "enable_offense_wez": True,
                        "enable_offense_ttc": True,
                        "enable_reward_shaping": False,
                    },
                    debug=False,
                )
                e = FireCooldownWrapper(e, cooldown_steps=120, fire_idx=3)
                e = HVAASurvivalEveryNStepsBonus(e, every_n_steps=50, bonus=0.05)

            if not args.no_nanguard:
                e = NanGuardWrapper(e, name="eval_env")

            e = EpisodeOverTerminatorWrapper(e)
            e = Monitor(e, filename=None)

            e.reset(seed=seed)
            return e
        return _thunk

    venv = DummyVecEnv([make_env(args.seed)])
    eval_env = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        training=False,
    )

    eval_env = VecNormalize.load(vecnorm_path.as_posix(), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path.as_posix(), env=eval_env, device="cpu")

    print("\n" + "=" * 80)
    print("EVALUATION CONFIG")
    print("=" * 80)
    print(f"Model:          {model_path}")
    print(f"VecNormalize:   {vecnorm_path}")
    print(f"Episodes:       {args.episodes}")
    print(f"Seed:           {args.seed}")
    print(f"Enriched obs:   {bool(args.use_enriched_obs)}")
    print(f"Renderize:      {args.renderize}")
    print(f"Speed up:       {args.speed_up}")
    print("=" * 80 + "\n")

    # Per-episode arrays (lists)
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_hvaa_survived_arr: list[int] = []
    ep_red_killed_arr: list[int] = []
    ep_blue_killed_arr: list[int] = []
    ep_red_killed_total_arr: list[int] = []
    ep_hvaa_destroyed_arr: list[int] = []
    ep_internal_sim_episodes_arr: list[int] = []
    ep_end_reason_arr: list[str] = []
    ep_debug_last_info_arr: list[dict] = []

    # --- Extra per-episode combat/geometry metrics (from SimManager info) ---
    ep_blue_shot_first_arr: list[int] = []
    ep_red_shot_first_arr: list[int] = []
    ep_no_shots_fired_arr: list[int] = []
    ep_min_separation_nm_arr: list[float] = []
    ep_blue_missiles_fired_arr: list[int] = []
    ep_red_missiles_fired_arr: list[int] = []
    ep_blue_pk_arr: list[float] = []
    ep_red_pk_arr: list[float] = []

    # Aggregate counters (rates)
    hvaa_survived = 0
    red_killed = 0
    blue_killed = 0

    start = time.time()
    obs = eval_env.reset()

    cur_r = 0.0
    cur_l = 0
    finished = 0

    # Per-episode flags (booleans)
    ep_hvaa_destroyed_flag = False
    ep_red_killed_flag = False
    ep_blue_killed_flag = False
    ep_sim_episodes_count = 0

    # Per-episode counters (best-effort, for detecting kill events via deltas)
    prev_red_killed_count: Optional[int] = None
    prev_blue_killed_count: Optional[int] = None
    prev_hvaa_destroyed_count: Optional[int] = None

    while finished < args.episodes:
        action, _ = model.predict(obs, deterministic=True)

        # Try gate_action if available
        try:
            if hasattr(eval_env, "get_original_obs"):
                obs_for_gate = eval_env.get_original_obs()
                if obs_for_gate is None:
                    obs_for_gate = obs
            else:
                obs_for_gate = obs

            gated = eval_env.env_method("gate_action", action[0], obs_for_gate[0], indices=[0])[0]
            action = np.asarray([gated], dtype=np.float32)
        except Exception:
            pass

        step_out = eval_env.step(action)
        if len(step_out) == 4:
            obs, reward, done, infos = step_out
            terminated = done
            truncated = np.array([False] * len(done))
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = np.logical_or(terminated, truncated)

        info_step = infos[0] if isinstance(infos, (list, tuple)) else infos
        if isinstance(info_step, dict):
            if _as_bool(info_step.get("episode_over", False)):
                ep_sim_episodes_count += 1

            # ---- Counter-based detection (preferred if available) ----
            # Some builds expose cumulative counters instead of one-step flags.
            # If these exist, delta>0 during the episode is a very reliable signal.
            red_c = _pick_int_key(info_step, (
                "red_killed_total", "Red_Killed_Total", "red_destroyed_total", "enemy_destroyed_total",
                "red_kills", "enemy_kills", "Red_Kills", "Enemy_Kills",
            ))
            if red_c is not None:
                if prev_red_killed_count is None:
                    prev_red_killed_count = red_c
                elif red_c > prev_red_killed_count:
                    ep_red_killed_flag = True
                    prev_red_killed_count = red_c

            blue_c = _pick_int_key(info_step, (
                "blue_killed_total", "Blue_Killed_Total", "blue_destroyed_total", "blue_losses",
                "blue_kills", "Blue_Kills",
            ))
            if blue_c is not None:
                if prev_blue_killed_count is None:
                    prev_blue_killed_count = blue_c
                elif blue_c > prev_blue_killed_count:
                    ep_blue_killed_flag = True
                    prev_blue_killed_count = blue_c

            hvaa_c = _pick_int_key(info_step, (
                "hvaa_destroyed_total", "HVAA_Destroyed_Total", "hvaa_losses", "HVAA_Losses",
            ))
            if hvaa_c is not None:
                if prev_hvaa_destroyed_count is None:
                    prev_hvaa_destroyed_count = hvaa_c
                elif hvaa_c > prev_hvaa_destroyed_count:
                    ep_hvaa_destroyed_flag = True
                    prev_hvaa_destroyed_count = hvaa_c

            # --- HVAA destroyed / alive (prefer explicit keys your Godot patch will send every step) ---
            if "hvaa_destroyed" in info_step:
                if _as_bool(info_step["hvaa_destroyed"]):
                    ep_hvaa_destroyed_flag = True
            elif "hvaa_alive" in info_step:
                if not _as_bool(info_step["hvaa_alive"]):
                    ep_hvaa_destroyed_flag = True

            # --- RED kill detection (use the new Godot per-step keys) ---
            # Preferred: per-step delta
            if _as_int(info_step.get("red_kills_this_step", 0)) > 0:
                ep_red_killed_flag = True

            # Also accept an event boolean
            if _as_bool(info_step.get("red_killed_event", False)):
                ep_red_killed_flag = True

            # Also accept a cumulative total (if provided)
            red_total = _pick_int_key(info_step, ("red_killed_total",))
            if red_total is not None and red_total > 0:
                ep_red_killed_flag = True

            # Blue loss signals
            if _as_bool(info_step.get("Blue_Killed", False)) or _as_bool(info_step.get("blue_killed", False)):
                ep_blue_killed_flag = True
            if _as_int(info_step.get("blue_kills_this_step", 0)) > 0:
                ep_blue_killed_flag = True

        cur_r += float(reward[0])
        cur_l += 1

        if bool(done[0]):
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            info0 = info0 if isinstance(info0, dict) else {}
            reason = _extract_reason_str(info0)

            # --- finalize flags from last info (authoritative) ---
            # HVAA
            if _as_bool(info0.get("hvaa_destroyed", False)) or ("hvaa_alive" in info0 and not _as_bool(info0.get("hvaa_alive", True))):
                ep_hvaa_destroyed_flag = True

            # Red kill
            if _as_int(info0.get("red_kills_this_step", 0)) > 0:
                ep_red_killed_flag = True
            if _as_bool(info0.get("red_killed_event", False)):
                ep_red_killed_flag = True
            if _as_int(info0.get("red_killed_total", 0)) > 0:
                ep_red_killed_flag = True

            # Optional: reason-based
            if reason:
                if _reason_implies_red_kill(reason):
                    ep_red_killed_flag = True
                if _reason_implies_hvaa_destroyed(reason):
                    ep_hvaa_destroyed_flag = True

            # --- append ---
            ep_rewards.append(cur_r)
            ep_lengths.append(cur_l)
            ep_hvaa_survived_arr.append(0 if ep_hvaa_destroyed_flag else 1)
            ep_red_killed_arr.append(1 if ep_red_killed_flag else 0)
            ep_blue_killed_arr.append(1 if ep_blue_killed_flag else 0)
            ep_internal_sim_episodes_arr.append(ep_sim_episodes_count)
            ep_end_reason_arr.append(reason)

            # --- Pull extra metrics from SimManager (best-effort, safe defaults) ---
            blue_shot_first = 1 if _as_bool(info0.get("blue_shot_first", False)) else 0
            red_shot_first  = 1 if _as_bool(info0.get("red_shot_first", False)) else 0
            no_shots_fired  = 1 if _as_bool(info0.get("no_shots_fired", False)) else 0

            # min separation in NM (SimManager emits -1.0 if not available)
            try:
                min_sep_nm = float(info0.get("min_separation_nm", -1.0))
            except Exception:
                min_sep_nm = -1.0

            blue_missiles_fired = _as_int(info0.get("blue_missiles_fired", 0))
            red_missiles_fired  = _as_int(info0.get("red_missiles_fired", 0))

            try:
                blue_pk = float(info0.get("blue_pk", 0.0))
            except Exception:
                blue_pk = 0.0
            try:
                red_pk = float(info0.get("red_pk", 0.0))
            except Exception:
                red_pk = 0.0

            ep_blue_shot_first_arr.append(int(blue_shot_first))
            ep_red_shot_first_arr.append(int(red_shot_first))
            ep_no_shots_fired_arr.append(int(no_shots_fired))
            ep_min_separation_nm_arr.append(float(min_sep_nm))
            ep_blue_missiles_fired_arr.append(int(blue_missiles_fired))
            ep_red_missiles_fired_arr.append(int(red_missiles_fired))
            ep_blue_pk_arr.append(float(blue_pk))
            ep_red_pk_arr.append(float(red_pk))
            ep_debug_last_info_arr.append({k: info0.get(k) for k in list(info0.keys())[:50]})
            ep_red_killed_total_arr.append(_as_int(info0.get("red_killed_total", 0)))
            ep_hvaa_destroyed_arr.append(1 if ep_hvaa_destroyed_flag else 0)

            # --- aggregate rates ---
            hvaa_survived += (0 if ep_hvaa_destroyed_flag else 1)
            red_killed += (1 if ep_red_killed_flag else 0)
            blue_killed += (1 if ep_blue_killed_flag else 0)

            finished += 1

            if (finished % int(args.progress_every)) == 0 or finished == 1 or finished == args.episodes:
                elapsed = time.time() - start
                print(f"[EVAL] {finished:4d}/{args.episodes}  "
                      f"meanR={np.mean(ep_rewards):8.2f}  stdR={np.std(ep_rewards):6.2f}  "
                      f"meanL={np.mean(ep_lengths):7.1f}  elapsed={elapsed:6.1f}s")

            # reset for next episode
            cur_r = 0.0
            cur_l = 0
            ep_hvaa_destroyed_flag = False
            ep_red_killed_flag = False
            ep_blue_killed_flag = False
            ep_sim_episodes_count = 0
            prev_red_killed_count = None
            prev_blue_killed_count = None
            prev_hvaa_destroyed_count = None

    ep_rewards_np = np.asarray(ep_rewards, dtype=np.float64)
    ep_lengths_np = np.asarray(ep_lengths, dtype=np.float64)

    mean_r = float(np.mean(ep_rewards_np))
    std_r = float(np.std(ep_rewards_np, ddof=0))
    ci95 = float(_ci95_mean(ep_rewards_np))
    mean_l = float(np.mean(ep_lengths_np))
    std_l = float(np.std(ep_lengths_np, ddof=0))

    summary = {
        "model_path": model_path.as_posix(),
        "vecnormalize_path": vecnorm_path.as_posix(),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "use_enriched_obs": bool(args.use_enriched_obs),
        "renderize": int(args.renderize),
        "speed_up": int(args.speed_up),
        "mean_reward": mean_r,
        "std_reward": std_r,
        "ci95_mean_reward": ci95,
        "mean_length": mean_l,
        "std_length": std_l,
        "episode_rewards": [float(x) for x in ep_rewards_np.tolist()],
        "episode_lengths": [int(x) for x in ep_lengths_np.astype(int).tolist()],
        "episode_hvaa_survived": [int(x) for x in ep_hvaa_survived_arr],
        "episode_red_killed": [int(x) for x in ep_red_killed_arr],
        "episode_blue_killed": [int(x) for x in ep_blue_killed_arr],
        "episode_red_killed_total": [int(x) for x in ep_red_killed_total_arr],
        "episode_hvaa_destroyed": [int(x) for x in ep_hvaa_destroyed_arr],

        "episode_blue_shot_first": [int(x) for x in ep_blue_shot_first_arr],
        "episode_red_shot_first": [int(x) for x in ep_red_shot_first_arr],
        "episode_no_shots_fired": [int(x) for x in ep_no_shots_fired_arr],
        "episode_min_separation_nm": [float(x) for x in ep_min_separation_nm_arr],
        "episode_blue_missiles_fired": [int(x) for x in ep_blue_missiles_fired_arr],
        "episode_red_missiles_fired": [int(x) for x in ep_red_missiles_fired_arr],
        "episode_blue_pk": [float(x) for x in ep_blue_pk_arr],
        "episode_red_pk": [float(x) for x in ep_red_pk_arr],
        "blue_shot_first_rate": float(np.mean(ep_blue_shot_first_arr)) if len(ep_blue_shot_first_arr) else 0.0,
        "red_shot_first_rate": float(np.mean(ep_red_shot_first_arr)) if len(ep_red_shot_first_arr) else 0.0,
        "no_shots_fired_rate": float(np.mean(ep_no_shots_fired_arr)) if len(ep_no_shots_fired_arr) else 0.0,
        "mean_min_separation_nm": float(np.mean([x for x in ep_min_separation_nm_arr if x >= 0.0])) if any(x >= 0.0 for x in ep_min_separation_nm_arr) else -1.0,
        "mean_blue_missiles_fired": float(np.mean(ep_blue_missiles_fired_arr)) if len(ep_blue_missiles_fired_arr) else 0.0,
        "mean_red_missiles_fired": float(np.mean(ep_red_missiles_fired_arr)) if len(ep_red_missiles_fired_arr) else 0.0,
        "mean_blue_pk": float(np.mean(ep_blue_pk_arr)) if len(ep_blue_pk_arr) else 0.0,
        "mean_red_pk": float(np.mean(ep_red_pk_arr)) if len(ep_red_pk_arr) else 0.0,
        "episode_internal_sim_episodes": [int(x) for x in ep_internal_sim_episodes_arr],
        "episode_end_reasons": ep_end_reason_arr,
        "episode_last_info_debug": ep_debug_last_info_arr,
        "hvaa_survival_rate": float(hvaa_survived) / float(args.episodes),
        "red_kill_rate": float(red_killed) / float(args.episodes),
        "blue_loss_rate": float(blue_killed) / float(args.episodes),
    }

    out_path = Path(args.out_json).expanduser().resolve() if args.out_json else (
        run_dir / f"eval_{args.episodes}eps_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonify(summary), indent=2))

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Mean reward:     {mean_r:.2f} ± {std_r:.2f}   (95% CI on mean: ±{ci95:.2f})")
    print(f"Mean length:     {mean_l:.1f} ± {std_l:.1f}")
    print(f"HVAA survival:   {summary['hvaa_survival_rate']*100:.1f}%")
    print(f"Red kill rate:   {summary['red_kill_rate']*100:.1f}%")
    print(f"Blue loss rate:  {summary['blue_loss_rate']*100:.1f}%")
    print(f"Blue shot first:  {summary['blue_shot_first_rate']*100:.1f}%")
    print(f"Red shot first:   {summary['red_shot_first_rate']*100:.1f}%")
    print(f"No shots fired:   {summary['no_shots_fired_rate']*100:.1f}%")
    print(f"Mean min sep (NM): {summary['mean_min_separation_nm']:.2f}")
    print(f"Mean missiles fired (B/R): {summary['mean_blue_missiles_fired']:.2f} / {summary['mean_red_missiles_fired']:.2f}")
    print(f"Mean Pk (B/R):     {summary['mean_blue_pk']:.3f} / {summary['mean_red_pk']:.3f}")
    print(f"Saved JSON:      {out_path}")
    print("=" * 80 + "\n")

    eval_env.close()


if __name__ == "__main__":
    main()
