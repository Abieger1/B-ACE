#!/usr/bin/env python3
"""
reward_component_eval_callback_V2.py

CRITICAL FIX: Evaluation callback that properly matches training behavior.

ISSUE DISCOVERED:
- Training with disable_gate_action=True: NO gate_action calls
- Original eval callback: STILL calls gate_action (line 180)
- This mismatch causes circling because gate_action does heading smoothing

THIS VERSION:
- Adds disable_gate_action flag to match training
- Properly syncs VecNormalize stats
- Has debug options to trace observation/action flow
- Can use stochastic actions instead of deterministic
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class RewardComponentEvalCallbackV2(BaseCallback):
    """
    Evaluation callback that properly aligns with training behavior.
    
    Key differences from original:
    - disable_gate_action option to match training
    - Proper VecNormalize sync with verification
    - Option for stochastic evaluation
    - Debug output to trace issues
    """
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 50000,
        n_eval_episodes: int = 20,
        log_dir: Path = None,
        verbose: int = 1,
        deterministic: bool = True,
        # NEW OPTIONS
        disable_gate_action: bool = True,   # Match training! Set True if training doesn't use it
        use_stochastic_eval: bool = False,  # Use stochastic actions (can help with circling)
        debug_actions: bool = False,        # Print action statistics
        debug_observations: bool = False,   # Print observation statistics
        verify_sync: bool = True,           # Verify VecNormalize sync
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        
        # Action mode
        self.deterministic = deterministic and not use_stochastic_eval
        self.use_stochastic_eval = use_stochastic_eval
        
        # Gate action - CRITICAL: must match training!
        self.disable_gate_action = disable_gate_action
        
        # Debug options
        self.debug_actions = debug_actions
        self.debug_observations = debug_observations
        self.verify_sync = verify_sync
        
        # Best model tracking
        self.best_mean_reward = -np.inf
        self.best_score = -np.inf
        self.best_stats: Dict[str, Any] = {}
        
        # Evaluation histories
        self.evaluations_timesteps: List[int] = []
        self.evaluations_results: List[float] = []
        self.evaluations_length: List[float] = []
        self.evaluations_std: List[float] = []
        self.evaluations_hvaa_survival: List[float] = []
        self.evaluations_red_kill: List[float] = []
        self.evaluations_component_breakdown: List[Dict[str, Any]] = []
        
        # Directories
        self.eval_checkpoint_dir = self.log_dir / "eval_checkpoints"
        self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.component_logs_dir = self.log_dir / "reward_component_logs"
        self.component_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_eval_step = 0
        
        # Print configuration
        if self.verbose > 0:
            print(f"\n[EvalCallback V2] Configuration:")
            print(f"  disable_gate_action: {self.disable_gate_action}")
            print(f"  deterministic: {self.deterministic}")
            print(f"  use_stochastic_eval: {self.use_stochastic_eval}")
            print(f"  verify_sync: {self.verify_sync}")

    def _sync_normalization(self) -> bool:
        """Sync VecNormalize stats with verification."""
        try:
            train_env = self.model.get_env()
            if not isinstance(train_env, VecNormalize) or not isinstance(self.eval_env, VecNormalize):
                return True  # Not using VecNormalize
            
            # Deep copy stats
            self.eval_env.obs_rms.mean = train_env.obs_rms.mean.copy()
            self.eval_env.obs_rms.var = train_env.obs_rms.var.copy()
            self.eval_env.obs_rms.count = train_env.obs_rms.count
            
            if train_env.ret_rms is not None and self.eval_env.ret_rms is not None:
                self.eval_env.ret_rms.mean = train_env.ret_rms.mean
                self.eval_env.ret_rms.var = train_env.ret_rms.var
                self.eval_env.ret_rms.count = train_env.ret_rms.count
            
            # Verify if requested
            if self.verify_sync:
                mean_diff = np.abs(train_env.obs_rms.mean - self.eval_env.obs_rms.mean).max()
                if mean_diff > 1e-10:
                    print(f"  ⚠️ VecNormalize sync verification failed! max_diff={mean_diff}")
                    return False
            
            if self.verbose > 0:
                print(f"  ✓ VecNormalize synced (count={train_env.obs_rms.count})")
            return True
            
        except Exception as e:
            print(f"  ⚠️ VecNormalize sync failed: {e}")
            return False

    @staticmethod
    def compute_eval_score(mean_reward: float, hvaa_survival_rate: float, red_kill_rate: float) -> float:
        if hvaa_survival_rate < 0.8:
            return -1e9
        return mean_reward + 50.0 * red_kill_rate

    def _evaluate_policy(self) -> Tuple[float, float, float, List[float], Dict[str, Any], float, float]:
        """
        Run evaluation episodes WITHOUT gate_action interference.
        """
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        episode_components: List[Dict[str, float]] = []
        hvaa_alive_count = 0
        red_kill_count = 0
        
        # Sync normalization BEFORE evaluation
        self._sync_normalization()
        
        # Reset environment
        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        episodes_completed = 0
        
        # Component tracking
        current_episode_components: Dict[str, float] = {
            'base_reward': 0.0, 'bez_shaping': 0.0, 'dmc_shaping': 0.0,
            'heading_shaping': 0.0, 'distance_shaping': 0.0, 'ttc_shaping': 0.0,
            'multithreat_shaping': 0.0, 'speed_shaping': 0.0, 'hvaa_escort_shaping': 0.0,
            'offense_wez_shaping': 0.0, 'offense_ttc_shaping': 0.0, 'potential_diff': 0.0,
            'total_shaping': 0.0, 'total_reward': 0.0,
        }
        
        # Debug tracking
        all_actions = []
        all_obs = []
        
        while episodes_completed < self.n_eval_episodes:
            # Get action from policy
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            
            # Debug: track actions and observations
            if self.debug_actions:
                all_actions.append(action[0].copy())
            if self.debug_observations:
                all_obs.append(obs[0].copy())
            
            # CRITICAL: Only call gate_action if it was used in training!
            if not self.disable_gate_action:
                try:
                    # Get unnormalized obs for gate
                    if hasattr(self.eval_env, 'get_original_obs'):
                        obs_for_gate = self.eval_env.get_original_obs()
                        if obs_for_gate is None:
                            obs_for_gate = obs
                    else:
                        obs_for_gate = obs
                    
                    gated = self.eval_env.env_method("gate_action", action[0], obs_for_gate[0], indices=[0])[0]
                    action = np.asarray([gated], dtype=np.float32)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"  gate_action failed: {e}")
            
            # Step environment
            step_out = self.eval_env.step(action)
            if len(step_out) == 4:
                obs, reward, done, infos = step_out
            else:
                obs, reward, terminated, truncated, infos = step_out
                done = np.logical_or(terminated, truncated)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Extract reward components
            if len(infos) > 0 and isinstance(infos[0], dict):
                breakdown = infos[0].get('reward_breakdown', {})
                for key in current_episode_components:
                    if key in breakdown:
                        current_episode_components[key] += breakdown[key]
            
            if done[0]:
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                
                # Mission metrics
                red_killed = info0.get('Red_Killed', False)
                hvaa_destroyed = info0.get('hvaa_destroyed', False)
                
                if red_killed:
                    red_kill_count += 1
                if not hvaa_destroyed:
                    hvaa_alive_count += 1
                
                episode_rewards.append(float(episode_reward))
                episode_lengths.append(int(episode_length))
                episode_components.append(current_episode_components.copy())
                episodes_completed += 1
                
                if self.verbose > 0:
                    print(f"[EVAL EP-END] ep={episodes_completed}/{self.n_eval_episodes} "
                          f"reward={episode_reward:.1f} len={episode_length}")
                
                # Print action debug for this episode
                if self.debug_actions and len(all_actions) > 0:
                    actions_arr = np.array(all_actions)
                    print(f"  Action stats: hdg_mean={actions_arr[:, 0].mean():.3f} "
                          f"hdg_std={actions_arr[:, 0].std():.3f}")
                    all_actions = []
                
                # Reset tracking
                episode_reward = 0.0
                episode_length = 0
                current_episode_components = {k: 0.0 for k in current_episode_components}
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        hvaa_survival_rate = hvaa_alive_count / self.n_eval_episodes
        red_kill_rate = red_kill_count / self.n_eval_episodes
        
        # Aggregate components
        component_stats = self._aggregate_components(episode_components)
        
        return (mean_reward, std_reward, mean_length, episode_rewards,
                component_stats, hvaa_survival_rate, red_kill_rate)

    def _aggregate_components(self, episode_components: List[Dict[str, float]]) -> Dict[str, Any]:
        if not episode_components:
            return {'aggregate': {}, 'per_episode': []}
        
        aggregate = {}
        for key in episode_components[0].keys():
            values = [ep.get(key, 0.0) for ep in episode_components]
            aggregate[f'{key}_mean'] = float(np.mean(values))
            aggregate[f'{key}_std'] = float(np.std(values))
        
        return {'aggregate': aggregate, 'per_episode': episode_components}

    def _save_best_model(self, mean_reward, std_reward, mean_length,
                        hvaa_survival_rate, red_kill_rate, score, component_stats):
        self.best_stats = {
            'timestep': self.num_timesteps,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_length': float(mean_length),
            'hvaa_survival_rate': float(hvaa_survival_rate),
            'red_kill_rate': float(red_kill_rate),
            'score': float(score),
            'component_breakdown': component_stats.get('aggregate', {}),
        }
        
        best_model_path = self.log_dir / "best_model"
        self.model.save(best_model_path.as_posix())
        
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            best_norm_path = self.log_dir / "best_vecnormalize.pkl"
            vec_env.save(best_norm_path.as_posix())
        
        with open(self.log_dir / "best_model_info.json", 'w') as f:
            json.dump(self.best_stats, f, indent=2)

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔍 INITIAL EVALUATION at 0 steps")
            print(f"   disable_gate_action={self.disable_gate_action}")
            print(f"   deterministic={self.deterministic}")
            print(f"{'='*60}")
        
        eval_start = time.time()
        (mean_reward, std_reward, mean_length, episode_rewards,
         component_stats, hvaa_survival, red_kill) = self._evaluate_policy()
        eval_time = time.time() - eval_start
        
        self.evaluations_timesteps.append(0)
        self.evaluations_results.append(mean_reward)
        self.evaluations_length.append(mean_length)
        self.evaluations_std.append(std_reward)
        self.evaluations_hvaa_survival.append(hvaa_survival)
        self.evaluations_red_kill.append(red_kill)
        self.evaluations_component_breakdown.append(component_stats)
        
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        
        if self.verbose > 0:
            print(f" Results: reward={mean_reward:.2f}±{std_reward:.2f}, len={mean_length:.1f}")
            print(f" Time: {eval_time:.1f}s")
        
        score = self.compute_eval_score(mean_reward, hvaa_survival, red_kill)
        if score > self.best_score:
            self.best_score = score
            self.best_mean_reward = mean_reward
            self._save_best_model(mean_reward, std_reward, mean_length,
                                 hvaa_survival, red_kill, score, component_stats)
        
        print(f"{'='*60}\n")

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"🔍 EVALUATION at {self.num_timesteps:,} steps")
                print(f"   disable_gate_action={self.disable_gate_action}")
                print(f"{'='*60}")
            
            eval_start = time.time()
            (mean_reward, std_reward, mean_length, episode_rewards,
             component_stats, hvaa_survival, red_kill) = self._evaluate_policy()
            eval_time = time.time() - eval_start
            
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(mean_reward)
            self.evaluations_length.append(mean_length)
            self.evaluations_std.append(std_reward)
            self.evaluations_hvaa_survival.append(hvaa_survival)
            self.evaluations_red_kill.append(red_kill)
            self.evaluations_component_breakdown.append(component_stats)
            
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/hvaa_survival_rate", hvaa_survival)
            self.logger.record("eval/red_kill_rate", red_kill)
            
            if self.verbose > 0:
                print(f" Results: reward={mean_reward:.2f}±{std_reward:.2f}")
                print(f" Length: {mean_length:.1f}, HVAA: {hvaa_survival:.0%}, Red Kill: {red_kill:.0%}")
                print(f" Time: {eval_time:.1f}s")
            
            # Save checkpoint
            checkpoint_path = self.eval_checkpoint_dir / f"model_{self.num_timesteps}_steps"
            self.model.save(checkpoint_path.as_posix())
            
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                norm_path = self.eval_checkpoint_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
                vec_env.save(norm_path.as_posix())
            
            # Component logs
            with open(self.component_logs_dir / f"components_{self.num_timesteps}_steps.json", 'w') as f:
                json.dump(component_stats, f, indent=2)
            
            # Best model check
            score = self.compute_eval_score(mean_reward, hvaa_survival, red_kill)
            if score > self.best_score:
                print(f"⭐ NEW BEST MODEL! (+{score - self.best_score:.2f})")
                self.best_score = score
                self.best_mean_reward = mean_reward
                self._save_best_model(mean_reward, std_reward, mean_length,
                                     hvaa_survival, red_kill, score, component_stats)
            
            print(f"{'='*60}\n")
        
        return True

    def get_evaluation_summary(self) -> Dict[str, Any]:
        return {
            'timesteps': self.evaluations_timesteps,
            'mean_rewards': self.evaluations_results,
            'mean_lengths': self.evaluations_length,
            'std_rewards': [float(s) for s in self.evaluations_std],
            'hvaa_survival_rates': [float(x) for x in self.evaluations_hvaa_survival],
            'red_kill_rates': [float(x) for x in self.evaluations_red_kill],
            'n_eval_episodes': int(self.n_eval_episodes),
            'best_mean_reward': float(self.best_mean_reward),
            'best_score': float(self.best_score),
            'n_evaluations': len(self.evaluations_timesteps),
            'component_breakdowns': self.evaluations_component_breakdown,
        }


if __name__ == "__main__":
    print("="*60)
    print("RewardComponentEvalCallbackV2")
    print("="*60)
    print("""
CRITICAL DIFFERENCE from original:
  - disable_gate_action parameter MUST match training setting
  - If training uses disable_gate_action=True, eval MUST too

USAGE:
    eval_callback = RewardComponentEvalCallbackV2(
        eval_env=eval_env,
        eval_freq=50000,
        n_eval_episodes=3,
        log_dir=log_dir,
        verbose=1,
        # CRITICAL: Match these to your training settings!
        disable_gate_action=True,   # If training doesn't use gate_action
        deterministic=True,         # Or False if circling persists
        debug_actions=True,         # To see action statistics
    )
""")
