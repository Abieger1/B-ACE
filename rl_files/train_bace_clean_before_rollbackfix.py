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
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.buffers import RolloutBuffer

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import RolloutReturn
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
# #from hierarchical_action_wrapper import HierarchicalActionWrapper, CurriculumConfig  # DISABLED (no hierarchical wrapper)
# from hierarchical_action_wrapper_fsm_FIXED import HierarchicalFSMActionWrapper, CurriculumConfig  # DISABLED (no hierarchical wrapper)

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
from expert_alignment_wrapper import ExpertAlignmentWrapper, create_alignment_decay_fn, get_default_enriched_obs_indices
from sustained_turn_penalty import SustainedTurnPenaltySimple
from hvaa_destruction_penalty import HVAADestructionPenalty
from rollback_eval_callback import RollbackEvaluationCallbackTopK as RollbackEvaluationCallback
# from b_ace_py.reward_shaping_wrapper import RewardShapingWrapper, RewardShapingConfig  # DISABLED (no reward shaping wrapper)
try:
    from reward_component_eval_callback import RewardComponentEvalCallback
    COMPONENT_TRACKING_AVAILABLE = True
except ImportError:
    COMPONENT_TRACKING_AVAILABLE = False
    print("Note: RewardComponentEvalCallback not found. Using standard evaluation.")

def linear_anneal(t: int, start: float, end: float, steps: int, hold: int = 0) -> float:
    """Hold `start` for `hold` steps, then linearly move to `end` over `steps`."""
    if t <= hold:
        return float(start)
    if steps <= 0:
        return float(end)
    x = min(1.0, max(0.0, (t - hold) / float(steps)))
    return float(start + x * (end - start))


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

        # optional bookkeeping for debugging/plotting
        info["hvaa_survival_bonus"] = float(bonus_now)

        self._t += 1
        return obs, rew, terminated, truncated, info

class KillRewardWrapper(gym.Wrapper):
    """Add reward when red_killed_event FIRST becomes True."""
    def __init__(self, env, kill_reward: float = 10.0):
        super().__init__(env)
        self.kill_reward = kill_reward
        self._already_awarded = False  # Track if we already gave the reward
    
    def reset(self, **kwargs):
        self._already_awarded = False  # Reset on new episode
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Only award ONCE per episode when red_killed_event first becomes True
        if info.get("red_killed_event", False) and not self._already_awarded:
            reward += self.kill_reward
            self._already_awarded = True
            print(f"[KILL REWARD] +{self.kill_reward} for killing red!")
        
        return obs, reward, term, trunc, info


class ActionSmoothnessPenalty(gym.Wrapper):
    """
    Scheduled smoothness penalty (jerk + magnitude), increases over training.
    """
    def __init__(
        self,
        env,
        hdg_idx=0,
        jerk_start=0.0,
        jerk_end=0.002,
        mag_start=0.0,
        mag_end=0.001,
        anneal_steps=2_000_000,
        hold_steps=0,
    ):
        super().__init__(env)
        self.hdg_idx = int(hdg_idx)

        self.jerk_start = float(jerk_start)
        self.jerk_end = float(jerk_end)
        self.mag_start = float(mag_start)
        self.mag_end = float(mag_end)
        self.anneal_steps = int(anneal_steps)
        self.hold_steps = int(hold_steps)

        self.jerk_coef = float(jerk_start)
        self.mag_coef = float(mag_start)

        self.prev = None
        self._global_step = 0

    def set_global_step(self, t: int):
        self._global_step = int(t)
        if hasattr(self.env, "set_global_step"):
            try:
                self.env.set_global_step(t)
            except Exception:
                pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = None
        return obs, info

    def step(self, action):
        # update coefs from schedule
        self.jerk_coef = linear_anneal(self._global_step, self.jerk_start, self.jerk_end, self.anneal_steps, hold=self.hold_steps)
        self.mag_coef  = linear_anneal(self._global_step, self.mag_start,  self.mag_end,  self.anneal_steps, hold=self.hold_steps)

        a = np.asarray(action, dtype=np.float32)
        jerk_pen = 0.0
        mag_pen = 0.0

        if self.prev is not None:
            d = float(a[self.hdg_idx] - self.prev[self.hdg_idx])
            jerk_pen = d * d
        m = float(a[self.hdg_idx])
        mag_pen = m * m

        obs, rew, terminated, truncated, info = self.env.step(action)

        penalty = self.jerk_coef * jerk_pen + self.mag_coef * mag_pen
        rew = float(rew) - penalty

        info["pen_turn_jerk"] = float(self.jerk_coef * jerk_pen)
        info["pen_turn_mag"]  = float(self.mag_coef * mag_pen)
        info["pen_turn_total"] = float(penalty)
        info["pen_turn_jerk_coef"] = float(self.jerk_coef)
        info["pen_turn_mag_coef"]  = float(self.mag_coef)

        self.prev = a.copy()
        return obs, rew, terminated, truncated, info
    
class RangeClosingShaping(gym.Wrapper):
    def __init__(self, env, range_idx: int, track_idx: int = 7, k: float = 0.02, clip_delta: float = 0.02):
        super().__init__(env)
        self.range_idx = int(range_idx)
        self.track_idx = int(track_idx)
        self.k = float(k)
        self.clip_delta = float(clip_delta)
        self.prev_range = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_range = float(obs[self.range_idx])
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        track = float(obs[self.track_idx]) > 0.5
        shaping = 0.0
        if track and self.prev_range is not None:
            cur_r = float(obs[self.range_idx])
            delta = float(self.prev_range - cur_r)  # positive if closing
            delta = float(np.clip(delta, -self.clip_delta, self.clip_delta))
            shaping = self.k * delta
            rew = float(rew) + shaping
            self.prev_range = cur_r
        else:
            # if no track, still update prev_range so it stays current
            self.prev_range = float(obs[self.range_idx])

        info["range_close_shaping"] = float(shaping)
        return obs, rew, terminated, truncated, info


class FireCooldownWrapper(gym.Wrapper):
    """
    Fire gating with a schedule:
      - early training: low cooldown, no rising-edge requirement (easier to learn kills)
      - later training: higher cooldown + rising-edge requirement (realistic discipline)
    """
    def __init__(
        self,
        env,
        fire_idx: int = 3,
        threshold: float = 0.5,
        cd_start: int = 20,
        cd_end: int = 240,
        cd_steps: int = 2_000_000,
        cd_hold: int = 0,
        rising_edge_after: int = 1_000_000,  # when to enable rising-edge realism
    ):
        super().__init__(env)
        self.fire_idx = int(fire_idx)
        self.threshold = float(threshold)

        self.cd_start = int(cd_start)
        self.cd_end = int(cd_end)
        self.cd_steps = int(cd_steps)
        self.cd_hold = int(cd_hold)
        self.rising_edge_after = int(rising_edge_after)

        self._t = 0
        self._last_fire_t = -10**9
        self._prev_want_fire = False
        self._global_step = 0

        self.cooldown_steps = int(cd_start)  # current value

    def set_global_step(self, t: int):
        self._global_step = int(t)
        # propagate if needed
        if hasattr(self.env, "set_global_step"):
            try:
                self.env.set_global_step(t)
            except Exception:
                pass

    def reset(self, **kwargs):
        self._t = 0
        self._last_fire_t = -10**9
        self._prev_want_fire = False
        return self.env.reset(**kwargs)

    def step(self, action):
        # update cooldown from schedule
        self.cooldown_steps = int(round(linear_anneal(
            self._global_step, self.cd_start, self.cd_end, self.cd_steps, hold=self.cd_hold
        )))
        self.cooldown_steps = max(1, self.cooldown_steps)

        a = np.array(action, dtype=np.float32, copy=True)
        want_fire = float(a[self.fire_idx]) > self.threshold

        # realism switch
        use_rising_edge = (self._global_step >= self.rising_edge_after)
        rising_edge = want_fire and (not self._prev_want_fire)

        allow = False
        if want_fire:
            if (self._t - self._last_fire_t) >= self.cooldown_steps:
                if (not use_rising_edge) or rising_edge:
                    allow = True

        if allow:
            a[self.fire_idx] = 1.0
            self._last_fire_t = self._t
        else:
            a[self.fire_idx] = 0.0

        self._prev_want_fire = want_fire

        obs, rew, terminated, truncated, info = self.env.step(a)
        self._t += 1

        # debug/metrics
        info["fire_cd_current"] = int(self.cooldown_steps)
        info["fire_use_rising_edge"] = bool(use_rising_edge)
        info["fire_want"] = bool(want_fire)
        info["fire_allowed"] = bool(allow)
        info["fire_cooldown_remaining_steps"] = int(max(0, self.cooldown_steps - (self._t - 1 - self._last_fire_t)))
        return obs, rew, terminated, truncated, info
    
class HeadingRateLimitWrapper(gym.Wrapper):
    def __init__(self, env, hdg_idx=0, max_delta=0.15):
        super().__init__(env)
        self.hdg_idx = int(hdg_idx)
        self.max_delta = float(max_delta)
        self.prev_hdg_cmd = 0.0

    def reset(self, **kwargs):
        self.prev_hdg_cmd = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        a = np.array(action, dtype=np.float32, copy=True)
        target = float(a[self.hdg_idx])
        delta = np.clip(target - self.prev_hdg_cmd, -self.max_delta, self.max_delta)
        a[self.hdg_idx] = self.prev_hdg_cmd + delta
        self.prev_hdg_cmd = float(a[self.hdg_idx])
        return self.env.step(a)

class NanGuardWrapper(gym.Wrapper):
    """
    Raises immediately if obs/reward becomes NaN/Inf.
    Also prints useful context to map the bad value back to a feature.
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
        #print("[STEP ACTION]", action) # DEBUG FOR STEP ACTION
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._check_obs(obs, where="step", info=info)
        self._check_reward(reward, info=info)

        return obs, reward, terminated, truncated, info

    def _check_obs(self, obs, where, info):
        x = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
        arr = np.asarray(x, dtype=np.float32)

        if not np.all(np.isfinite(arr)):
            bad = np.where(~np.isfinite(arr))[0]
            print(f"\n[NAN-GUARD:{self.name}] Non-finite OBS at {where} step={self.step_i}")
            print(f"  bad indices (first {self.print_first_n}): {bad[:self.print_first_n].tolist()}")
            print(f"  bad values  (first {self.print_first_n}): {arr[bad[:self.print_first_n]]}")

            # Try to print labels if available
            labels = getattr(self.env, "observation_labels_flat", getattr(self.env, "observation_labels", None))
            if labels is not None and isinstance(labels, (list, tuple)) and len(labels) == len(arr):
                for i in bad[:self.print_first_n]:
                    print(f"  label[{i}] = {labels[i]}")

            # Print some info keys for context
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

############################################
# Learning Rate and Entropy Decay Functions
############################################

def print_wrapper_stack(env, name: str = "env"):
    """
    Print wrapper chain for:
      - VecEnv (DummyVecEnv/VecNormalize)
      - underlying Gym env wrappers (Monitor, NanGuardWrapper, etc.)
    """
    print("\n" + "=" * 80)
    print(f"[WRAPPER STACK] {name}")
    print("=" * 80)

    # ---- VecEnv-level chain ----
    e = env
    i = 0
    while True:
        print(f"  VecLayer {i}: {type(e)}")
        # VecNormalize wraps another VecEnv at .venv
        if hasattr(e, "venv"):
            e = e.venv
            i += 1
            continue
        break

    # ---- underlying gym env chain (only if DummyVecEnv) ----
    base = None
    if hasattr(e, "envs") and len(e.envs) > 0:
        base = e.envs[0]
        print("\n  Underlying gym wrapper chain (envs[0]):")
        j = 0
        cur = base
        while True:
            print(f"    GymLayer {j}: {type(cur)}")
            if hasattr(cur, "env"):
                cur = cur.env
                j += 1
                continue
            break

    print("=" * 80 + "\n")


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

class ExpertRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer + storage for expert actions (and mask).
    We store expert actions per step so PPO's train() can add an imitation loss.
    """
    def reset(self) -> None:
        super().reset()
        # buffer_size x n_envs x action_dim
        self.expert_actions = np.zeros_like(self.actions, dtype=np.float32)
        # buffer_size x n_envs
        self.expert_mask = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

class ExpertMixPPO(PPO):
    """PPO with policy-side expert action mixing (Fix A).

    This keeps PPO on-policy by ensuring the *executed* action is what gets stored in the rollout buffer,
    along with a matching log-prob/value computed under the current policy.

    Requirements:
      - Training env (VecEnv) must wrap the base env with HierarchicalFSMActionWrapper (training only),
        which exposes:
            - expert_action(obs) -> action
            - gate_action(action, obs) -> action  (optional fire gating)
      - Evaluation env should NOT use expert mixing (pure policy).
    """
    #BEFORE IMITATION Changes
    #def __init__(self, *args, greedy_prob_fn=None, use_fire_gate: bool = True, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.greedy_prob_fn = greedy_prob_fn or (lambda t: 1.0)
    #    self.use_fire_gate = bool(use_fire_gate)

    def __init__(
        self,
        *args,
        greedy_prob_fn=None,
        use_fire_gate: bool = True,
        bc_coef_fn=None,                 # NEW
        bc_use_gated_expert: bool = True, # NEW (recommended)
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.greedy_prob_fn = greedy_prob_fn or (lambda t: 1.0)
        self.use_fire_gate = bool(use_fire_gate)
        self._src_counts = {"policy": 0, "expert": 0}
        self._src_counts_ep = {"policy": 0, "expert": 0}

        # Imitation regularizer schedule (decays over time)
        self.bc_coef_fn = bc_coef_fn or (lambda t: 0.0)
        self.bc_use_gated_expert = bool(bc_use_gated_expert)

    def _get_original_obs_for_expert(self, env):
        # If VecNormalize is used, expert should generally see original (unnormalized) obs.
        if hasattr(env, "get_original_obs"):
            try:
                o = env.get_original_obs()
                if o is not None:
                    return o
            except Exception:
                pass
        return self._last_obs
    
    def _nan_tripwire(self, name: str, x):
        """
        Crash immediately if x contains NaN/Inf (torch tensor or numpy array).
        """
        if isinstance(x, th.Tensor):
            ok = th.isfinite(x).all().item()
        else:
            x = np.asarray(x)
            ok = np.isfinite(x).all()
        if not ok:
            raise RuntimeError(f"[NaN TRIPWIRE] non-finite detected in {name}")

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> RolloutReturn:
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        n_steps = 0
        while n_steps < n_rollout_steps:
            # ---- policy forward pass (for exploration) ----
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                self._nan_tripwire("self._last_obs (raw)", self._last_obs)
                self._nan_tripwire("obs_tensor", obs_tensor)

                actions_pi, values_pi, log_probs_pi = self.policy(obs_tensor)

            actions_pi_np = actions_pi.cpu().numpy()
            self._nan_tripwire("actions_pi_np (policy sampled)", actions_pi_np)

            # ---- expert mixing (policy-side) ----
            greedy_prob = float(self.greedy_prob_fn(self.num_timesteps))
            use_policy_mask = (np.random.rand(env.num_envs) < greedy_prob)
            n_pol = int(np.sum(use_policy_mask))
            n_exp = int(np.sum(~use_policy_mask))

            self._src_counts["policy"] += n_pol
            self._src_counts["expert"] += n_exp
            self._src_counts_ep["policy"] += n_pol
            self._src_counts_ep["expert"] += n_exp
            if (self.num_timesteps % 5000) == 0:
                expert_used = int(np.sum(~use_policy_mask))
                print(f"[MIX DEBUG] t={self.num_timesteps} greedy_prob={greedy_prob:.3f} "
                    f"policy_used={int(np.sum(use_policy_mask))} expert_used={expert_used}")

            # Expert actions computed from ORIGINAL obs (important if VecNormalize is enabled)
            obs_for_expert = self._get_original_obs_for_expert(env)

            # ---- compute expert actions for BC (for ALL envs) ----
            expert_all = []
            for i in range(env.num_envs):
                obs_i = obs_for_expert[i]
                a_exp = env.env_method("expert_action", obs_i, indices=[i])[0]
                expert_all.append(a_exp)
            expert_all = np.asarray(expert_all, dtype=np.float32)

            # If the gate is part of your control interface, store GATED expert actions
            if self.use_fire_gate and self.bc_use_gated_expert:
                gated_exp = []
                for i in range(env.num_envs):
                    obs_i = obs_for_expert[i]
                    a_i = env.env_method("gate_action", expert_all[i], obs_i, indices=[i])[0]
                    gated_exp.append(a_i)
                expert_all = np.asarray(gated_exp, dtype=np.float32)

            mixed_actions = actions_pi_np.copy()
            if not np.all(use_policy_mask):
                expert_actions = []
                for i in range(env.num_envs):
                    if use_policy_mask[i]:
                        expert_actions.append(mixed_actions[i])
                    else:
                        obs_i = obs_for_expert[i]
                        a_i = env.env_method("expert_action", obs_i, indices=[i])[0]
                        expert_actions.append(a_i)
                mixed_actions = np.asarray(expert_actions, dtype=np.float32)

            # Optional: apply fire gating to ALL executed actions (policy + expert)
            if self.use_fire_gate:
                gated = []
                for i in range(env.num_envs):
                    obs_i = obs_for_expert[i]
                    a_i = env.env_method("gate_action", mixed_actions[i], obs_i, indices=[i])[0]
                    gated.append(a_i)
                mixed_actions = np.asarray(gated, dtype=np.float32)
            
            self._nan_tripwire("mixed_actions (executed)", mixed_actions)

            # ---- recompute log-prob/value for EXECUTED action ----
            with th.no_grad():
                mixed_actions_t = th.as_tensor(mixed_actions).to(self.device)
                values_mixed, log_probs_mixed, _ = self.policy.evaluate_actions(obs_tensor, mixed_actions_t)
            self._nan_tripwire("values_mixed", values_mixed)
            self._nan_tripwire("log_probs_mixed", log_probs_mixed)

            # ---- env step using executed action ----
            env.env_method("set_global_step", int(self.num_timesteps))
            low, high = self.action_space.low, self.action_space.high  # model knows the action space
            mixed_actions = np.clip(mixed_actions, low, high).astype(np.float32)
            new_obs, rewards, dones, infos = env.step(mixed_actions)
            if np.any(dones):
                # There can be multiple envs; DummyVecEnv can be num_envs=1, but handle general case
                done_ids = np.where(dones)[0].tolist()
                print(f"[ACTION-SOURCE EP-END] done_envs={done_ids} "
                    f"policy_steps={self._src_counts_ep['policy']} expert_steps={self._src_counts_ep['expert']} "
                    f"(greedy_prob={greedy_prob:.3f}, t={self.num_timesteps})")
                self._src_counts_ep = {"policy": 0, "expert": 0}

            self._nan_tripwire("new_obs", new_obs)
            self._nan_tripwire("rewards", rewards)

            self.num_timesteps += env.num_envs

            # ---- store executed action + matching logprob/value ----
            rollout_buffer.add(
                self._last_obs,
                mixed_actions,
                rewards,
                self._last_episode_starts,
                values_mixed,
                log_probs_mixed,
            )

            # ---- store expert actions for imitation loss ----
            if hasattr(rollout_buffer, "expert_actions"):
                j = rollout_buffer.pos - 1  # index just written
                rollout_buffer.expert_actions[j] = expert_all
                rollout_buffer.expert_mask[j] = 1.0  # all steps have expert available in your wrapper

            self._last_obs = new_obs
            self._last_episode_starts = dones

            callback.update_locals(locals())
            if not callback.on_step():
                callback.on_rollout_end()
                return False, n_steps * env.num_envs, None

            n_steps += 1

        with th.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(values, self._last_episode_starts)
        callback.on_rollout_end()
        return True, n_steps * env.num_envs, None
    
    def train(self) -> None:
        """
        Same as SB3 PPO.train(), but with an added imitation (BC) loss:
            L = L_ppo + bc_coef(t) * MSE(a_mean(obs), a_expert)
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # PPO update
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is None:
            clip_range_vf = None
        else:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, bc_losses = [], [], [], []
        clip_fractions = []

        bc_coef = float(self.bc_coef_fn(self.num_timesteps))

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Policy loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # ---- NEW: imitation (BC) loss on mean action ----
                bc_loss = th.tensor(0.0, device=self.device)
                if bc_coef > 0.0 and hasattr(self.rollout_buffer, "expert_actions"):
                    # Get expert actions corresponding to this minibatch indices
                    # rollout_data.indices exists in SB3>=2; if not, we fall back to no BC
                    if hasattr(rollout_data, "indices"):
                        # rollout_data.indices are flattened indices into (buffer_size * n_envs)
                        exp_np = self.rollout_buffer.expert_actions.reshape(-1, self.action_space.shape[0])[rollout_data.indices]
                        exp = th.as_tensor(exp_np, device=self.device, dtype=th.float32)

                        # Mean action from current policy distribution
                        dist = self.policy.get_distribution(rollout_data.observations)
                        if hasattr(dist, "distribution") and hasattr(dist.distribution, "mean"):
                            a_mean = dist.distribution.mean
                        elif hasattr(dist, "mean"):
                            a_mean = dist.mean
                        else:
                            # fallback: use sampled action (less ideal, but safe)
                            a_mean = actions.float()

                        bc_loss = F.mse_loss(a_mean, exp)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + bc_coef * bc_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                bc_losses.append(float(bc_loss.item()) if isinstance(bc_loss, th.Tensor) else float(bc_loss))
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                with th.no_grad():
                    approx_kl_div = th.mean(rollout_data.old_log_prob - log_prob).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                continue_training = False
                break

        self._n_updates += self.n_epochs
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/bc_coef", bc_coef)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))



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
        # ---- 0) UPDATE GLOBAL STEP FOR SCHEDULED WRAPPERS ----
        # This is CRITICAL for FireCooldownWrapper, ActionSmoothnessPenalty, etc.
        # Without this, _global_step stays at 0 and schedules don't progress!
        try:
            self.model.get_env().env_method("set_global_step", int(self.num_timesteps))
        except Exception:
            pass  # Silently ignore if method doesn't exist
        
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
        eval_freq: int = 75000,
        n_eval_episodes: int = 15, #EVALUATION EPISODES
        log_dir: Path = None,
        verbose: int = 1,
        deterministic: bool = True,
        skip_initial_eval: bool = False,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.deterministic = deterministic
        self.skip_initial_eval = skip_initial_eval
        
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
            self.eval_env.env_method("set_global_step", int(self.num_timesteps))
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            try:
                # VecNormalize provides original obs for gating if available
                if hasattr(self.eval_env, "get_original_obs"):
                    obs_for_gate = self.eval_env.get_original_obs()
                    if obs_for_gate is None:
                        obs_for_gate = obs
                else:
                    obs_for_gate = obs

                # DummyVecEnv with 1 env: action and obs are shape (1, dim)
                gated = self.eval_env.env_method("gate_action", action[0], obs_for_gate[0], indices=[0])[0]
                action = np.asarray([gated], dtype=np.float32)
            except Exception as ex:
                # If gate_action isn't available for some reason, fall back to raw action.
                # (But with Fix A it should be available.)
                pass
            #obs, reward, done, info = self.eval_env.step(action)
            step_out = self.eval_env.step(action)

            # VecEnv usually returns 4 items: (obs, rewards, dones, infos)
            # Some wrappers may return 5: (obs, rewards, terminateds, truncateds, infos)
            if len(step_out) == 4:
                obs, reward, done, infos = step_out
                terminated, truncated = done, np.array([False] * len(done))
            else:
                obs, reward, terminated, truncated, infos = step_out
                done = np.logical_or(terminated, truncated)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0]:
                # VecEnv -> infos is list of dicts (1 per env). With 1 env:
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

                # This is the single, standard "episode end" payload created by Monitor
                ep = info0.get("episode", None)

                print(f"[EVAL EP-END] ep={episodes_completed+1}/{self.n_eval_episodes} episode={ep}")

                episode_rewards.append(float(episode_reward))
                episode_lengths.append(int(episode_length))
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
        if self.skip_initial_eval:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"⏭️  SKIPPING INITIAL EVALUATION (skip_initial_eval=True)")
                print(f"{'='*60}\n")
            return
        
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
            try:
                train_env = self.model.get_env()
                if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                    self.eval_env.obs_rms = train_env.obs_rms
                    self.eval_env.ret_rms = train_env.ret_rms
            except Exception:
                pass
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
        default=0.000258,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=0.000001,
        help="Final learning rate after decay."
    )
    
    # Entropy coefficient arguments
    parser.add_argument(
        "--initial-entropy-coef",
        type=float,
        default=0.0450525,
        help="Initial entropy coefficient."
    )
    parser.add_argument(
        "--final-entropy-coef",
        type=float,
        default=0.0254,
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
        default=0.920664477208517,
        help="GAE lambda."
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.197991384633053,
        help="PPO clip range."
    )


    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.7728139,
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
        default=8,
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
        default=False,
        help="Enable enriched observations with pursuit-evasion heuristics"
    )
    parser.add_argument(
        "--ablation-config",
        type=str,
        default="all",
        choices=['all', 'none', 'geometry_only', 'engagement_only', 'range_limited_only',
                 'geometry_engagement', 'geometry_range', 'engagement_range'],
        help="""Ablation config for feature categories (only applies when --use-enriched-obs is set):
  all                 = A+B+C (GEOMETRY + ENGAGEMENT + RANGE_LIMITED)
  none                = No theoretical features (raw obs only)
  geometry_only       = A only (Apollonius + ATDDG)
  engagement_only     = B only (BEZ + DMC + WEZ)
  range_limited_only  = C only (Critical escape + capture probability)
  geometry_engagement = A+B (GEOMETRY + ENGAGEMENT)
  geometry_range      = A+C (GEOMETRY + RANGE_LIMITED)
  engagement_range    = B+C (ENGAGEMENT + RANGE_LIMITED)
"""
    )
    #parser.add_argument(
    #    "--no-enriched-obs",
    #    dest="use_enriched_obs",
    #    action="store_false",
    #    help="Disable enriched observations (use baseline 22 dims)"
    #)
    parser.add_argument("--expert-alignment", action="store_true", default=False)
    parser.add_argument("--alignment-coef", type=float, default=0.1)
    parser.add_argument("--alignment-decay-start", type=int, default=500_000)
    parser.add_argument("--alignment-decay-end", type=int, default=2_000_000)
    # Set default to False (baseline observations by default)
    parser.set_defaults(use_enriched_obs=False)
    
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

        # ---- Expose obs_map + flat labels for downstream wrappers (FSM, etc.) ----
        self.obs_map = getattr(self._ma_env, "obs_map", None)
        self.observation_labels_flat = getattr(self._ma_env, "observation_labels_flat", None)

        # Optional: for backwards compatibility if anything expects obs_labels
        self.obs_labels = getattr(self._ma_env, "obs_labels", None)
        # -------------------------------------------------------------------------
        
        # Reset FIRST to get actual observation data
        obs_dict, info_dict = self._ma_env.reset()
        self.obs_map = getattr(self._ma_env, "obs_map", self.obs_map)
        self.observation_labels_flat = getattr(self._ma_env, "observation_labels_flat", self.observation_labels_flat)
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
    def make_env(log_dir_str, seed, is_eval: bool = False):
        def _thunk():
            e = SingleAgentBACEEnv(
                bace_config=B_ACE_config,
                controlled_agent_id="agent_0",
                max_episode_steps=13_500,
            )

            # === ENRICHED OBSERVATIONS (OPTIONAL) ===
            if args.use_enriched_obs:
                # Map ablation config to human-readable category names
                config_to_categories = {
                    'all': 'A+B+C (all features)',
                    'none': 'No features (enriched wrapper active but 0 features)',
                    'geometry_only': 'A only (GEOMETRY)',
                    'engagement_only': 'B only (ENGAGEMENT)',
                    'range_limited_only': 'C only (RANGE_LIMITED)',
                    'geometry_engagement': 'A+B (GEOMETRY + ENGAGEMENT)',
                    'geometry_range': 'A+C (GEOMETRY + RANGE_LIMITED)',
                    'engagement_range': 'B+C (ENGAGEMENT + RANGE_LIMITED)',
                }
                
                print("\n" + "="*60)
                print("USING ENRICHED OBSERVATIONS")
                print(f"  Ablation Config: {args.ablation_config}")
                print(f"  Categories: {config_to_categories.get(args.ablation_config, 'unknown')}")
                print("  Legend:")
                print("    [A] GEOMETRY     = Apollonius circle + ATDDG (Weintraub 2020)")
                print("    [B] ENGAGEMENT   = BEZ + DMC + WEZ (Von Moll 2024)")
                print("    [C] RANGE_LIMITED = Escape heading + capture prob (Weintraub 2023)")
                print("="*60 + "\n")
                
                e = EnrichedObservationWrapper(
                    e,
                    obs_labels=None,
                    feature_config={
                        'pursuer_speed': 1.0,
                        'evader_speed': 1.0,
                        'capture_radius': 0.01,
                        'pursuer_range': 0.5,
                        'normalize_distance': 1.0,
                    },
                    ablation_config=args.ablation_config,
                    debug=False
                )
            else:
                print("\n" + "="*60)
                print("USING BASELINE OBSERVATIONS (NO THEORETICAL FEATURES)")
                print("="*60 + "\n")

            # === SHAPING WRAPPERS (ALWAYS APPLIED) ===
            e = HeadingRateLimitWrapper(e, hdg_idx=0, max_delta=0.15)

            e = KillRewardWrapper(e, kill_reward=8.0)

            e = ActionSmoothnessPenalty(
                e,
                hdg_idx=0,
                jerk_start=0.0, jerk_end=0.0025,
                mag_start=0.0003,  mag_end=0.012,
                hold_steps=800_000,
                anneal_steps=4_500_000
            )

            e = SustainedTurnPenaltySimple(
                e,
                hdg_idx=0,
                window_size=15,
                turn_threshold=0.5,
                penalty_coef=0.01,  # increase if still spinning
            )
            
            e = RangeClosingShaping(e, range_idx=17, track_idx=26, k=0.2, clip_delta=0.03)
            
            e = FireCooldownWrapper(
                e,
                fire_idx=3,
                cd_start=55,
                cd_end=120,
                cd_hold=10_000,
                cd_steps=5_000_000,
                rising_edge_after=1_500_000
            )
            
            e = HVAASurvivalEveryNStepsBonus(e, every_n_steps=50, bonus=0.03)
            # DISABLED: Large instant penalty creates high variance and credit assignment issues
            # The survival bonus provides dense positive signal instead
            # e = HVAADestructionPenalty(e, penalty=-15.0)

            # === EXPERT ALIGNMENT (ONLY WITH ENRICHED OBS) ===
            if args.expert_alignment:
                if not args.use_enriched_obs:
                    print("  WARNING: Expert alignment requires enriched observations!")
                    print("    Skipping ExpertAlignmentWrapper for baseline run.")
                else:
                    expert = AggressiveExpert(fire_threshold=0.50, aspect_limit_deg=30.0)
                    obs_indices = get_default_enriched_obs_indices()
                    alignment_decay = create_alignment_decay_fn(
                        decay_start=args.alignment_decay_start,
                        decay_end=args.alignment_decay_end,
                        final_multiplier=0.3,
                    )
                    e = ExpertAlignmentWrapper(
                        e, expert=expert, obs_indices=obs_indices,
                        alignment_coef=args.alignment_coef,
                        action_weights=np.array([1.0, 0.3, 0.6, 0.5]),
                        decay_fn=alignment_decay,
                    )
                    print(f"✓ Expert alignment wrapper added")

            # === DEBUG OUTPUT ===
            obs, _ = e.reset()
            print(f"\n✓ Environment created")
            print(f"   Observation dimension: {obs.shape[0]}")
            print(f"   obs[0:10] = {np.round(obs[:10], 3)}")
            
            if args.use_enriched_obs and obs.shape[0] > 27:
                print(f"   Enriched features [27:]: {np.round(obs[27:], 3)}")

            # === FINAL WRAPPERS ===
            e = NanGuardWrapper(e, name=("eval_env" if is_eval else "train_env"))
            e = Monitor(e, filename=str(Path(log_dir_str) / "monitor.csv"))
            
            obs, _ = e.reset(seed=seed)
            print(f"[DEBUG] Final observation shape: {obs.shape}")
            return e
        return _thunk

    vec_env = DummyVecEnv([make_env(log_dir.as_posix(), args.seed)])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    print_wrapper_stack(vec_env, name="TRAIN vec_env (after VecNormalize)")

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

    def greedy_prob_fn(t: int) -> float:
        p_start = 1.0
        p_end   = 1.0       # Cap at 70% - always keep 30% expert
        hold_steps   = 500_000    # Shorter hold
        anneal_steps = 5_000_000  # Very gradual
        
        if t < hold_steps:
            return p_start
        if t >= hold_steps + anneal_steps:
            return p_end
        
        # LINEAR - no exponential!
        frac = (t - hold_steps) / anneal_steps
        return p_start + frac * (p_end - p_start)

    # BC - keep stronger throughout
    bc_start = 1.0
    bc_end   = 0.5      # Was 0.3
    bc_steps = 6_000_000  # Was 5M

    def bc_coef_fn(t: int) -> float:
        return 0.0
        '''
        if t <= 0:
            return float(bc_start)
        if t >= bc_steps:
            return float(bc_end)
        frac = float(t) / float(bc_steps)
        return float(bc_start + frac * (bc_end - bc_start))
        '''
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
        target_kl=0.05,
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
    eval_env = DummyVecEnv([make_env(log_dir.as_posix(), args.seed + 10000, is_eval=True)])

    
    # Wrap with VecNormalize but set training=False
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during eval
        clip_obs=10.0,
        clip_reward=10.0,
        training=False,  # Important: don't update normalization stats during eval
    )

    print_wrapper_stack(eval_env, name="EVAL eval_env (after VecNormalize)")
    
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
    eval_checkpoint_callback = RollbackEvaluationCallback(
        eval_env=eval_env,
        eval_freq=200000,
        n_eval_episodes=15,
        log_dir=log_dir,
        rollback_patience=5,      # Rollback after 3 declines
        lr_decay_factor=0.7,      
        activation_threshold=12.0,  # Only activate rollback after reaching 15.0 or better
        min_lr=1e-6,
        max_rollbacks=10,
        skip_initial_eval=True,   # Skip evaluation at timestep 0
        verbose=1,
        deterministic=True,
    )
    '''
    eval_checkpoint_callback = EvaluationBasedCheckpointCallback(
        eval_env=eval_env,
        eval_freq=50000,  # Evaluate every 50k steps
        n_eval_episodes=20,  # Run 20 episodes per evaluation
        log_dir=log_dir,
        verbose=1,
        deterministic=True  # Use deterministic actions during evaluation
    )
    
    # 2. Evaluation-based checkpoint callback with component tracking
    if COMPONENT_TRACKING_AVAILABLE:
        print("✓ Using RewardComponentEvalCallback (with component tracking)")
        eval_checkpoint_callback = RewardComponentEvalCallback(
            eval_env=eval_env,
            eval_freq=200000,  # Evaluate every 50k steps
            n_eval_episodes=15,  # Run 20 episodes per evaluation
            log_dir=log_dir,
            verbose=1,
            deterministic=True  # Use deterministic actions during evaluation
        )
    else:
        print("✓ Using EvaluationBasedCheckpointCallback (standard)")
        eval_checkpoint_callback = EvaluationBasedCheckpointCallback(
            eval_env=eval_env,
            eval_freq=200000,
            n_eval_episodes=15,
            log_dir=log_dir,
            verbose=1,
            deterministic=True
    )
    '''

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
        # Apply action gate during final eval too
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
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = np.logical_or(terminated, truncated)
        
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