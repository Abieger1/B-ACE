#!/usr/bin/env python3
"""
reward_component_eval_callback_FIXED.py

CRITICAL FIX: Ensures gate_action() receives UNNORMALIZED observations during evaluation,
matching the behavior during training.

Key Changes:
1. Explicit VecNormalize unwrapping to get original observations
2. Robust fallback that doesn't silently use normalized obs for gating
3. Better sync of normalization stats before each evaluation
4. Debug output to verify observation pipeline is working correctly
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


class RewardComponentEvalCallbackFIXED(BaseCallback):
    """
    FIXED evaluation callback that properly handles observation normalization
    to ensure consistent behavior between training and evaluation.
    """
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 50000,
        n_eval_episodes: int = 20,
        log_dir: Path = None,
        verbose: int = 1,
        deterministic: bool = True,
        debug_obs: bool = False,  # NEW: Enable observation debugging
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.deterministic = deterministic
        self.debug_obs = debug_obs
        
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

    def _get_unnormalized_obs(self, normalized_obs: np.ndarray) -> np.ndarray:
        """
        FIXED: Reliably get the unnormalized observation for gate_action().
        
        This is the key fix - we need to properly reverse the VecNormalize
        transformation to get the original observation values that the
        gate_action() method expects.
        """
        if not isinstance(self.eval_env, VecNormalize):
            # No normalization, return as-is
            return normalized_obs
        
        # Method 1: Try get_original_obs() first
        try:
            original_obs = self.eval_env.get_original_obs()
            if original_obs is not None:
                if self.debug_obs:
                    print(f"[EVAL DEBUG] Using get_original_obs(): shape={original_obs.shape}")
                return original_obs
        except Exception as e:
            if self.debug_obs:
                print(f"[EVAL DEBUG] get_original_obs() failed: {e}")
        
        # Method 2: Manually unnormalize using obs_rms
        # This reverses: normalized = clip((obs - mean) / std, -clip, clip)
        try:
            obs_rms = self.eval_env.obs_rms
            if obs_rms is not None and obs_rms.count > 0:
                mean = obs_rms.mean
                var = obs_rms.var
                std = np.sqrt(var + 1e-8)
                
                # Reverse normalization: obs = normalized * std + mean
                # Note: This is approximate due to clipping, but much better than using normalized
                unnormalized = normalized_obs * std + mean
                
                if self.debug_obs:
                    print(f"[EVAL DEBUG] Manually unnormalized: normalized[0]={normalized_obs[0, :5]}, unnorm[0]={unnormalized[0, :5]}")
                
                return unnormalized
        except Exception as e:
            if self.debug_obs:
                print(f"[EVAL DEBUG] Manual unnormalization failed: {e}")
        
        # Method 3: Last resort - warn and return normalized (this should be avoided!)
        print("[EVAL WARNING] Could not get unnormalized obs! gate_action() may behave incorrectly!")
        return normalized_obs

    def _sync_normalization_stats(self):
        """
        FIXED: More robust normalization stats sync with error checking.
        """
        try:
            train_env = self.model.get_env()
            if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                # Deep copy to avoid reference issues
                self.eval_env.obs_rms.mean = train_env.obs_rms.mean.copy()
                self.eval_env.obs_rms.var = train_env.obs_rms.var.copy()
                self.eval_env.obs_rms.count = train_env.obs_rms.count
                
                if train_env.ret_rms is not None and self.eval_env.ret_rms is not None:
                    self.eval_env.ret_rms.mean = train_env.ret_rms.mean
                    self.eval_env.ret_rms.var = train_env.ret_rms.var
                    self.eval_env.ret_rms.count = train_env.ret_rms.count
                
                if self.verbose > 0:
                    print(f"[EVAL] Synced normalization stats (obs_rms.count={train_env.obs_rms.count})")
                return True
        except Exception as e:
            print(f"[EVAL WARNING] Failed to sync normalization stats: {e}")
            return False
        return False

    @staticmethod
    def compute_eval_score(
        mean_reward: float,
        hvaa_survival_rate: float,
        red_kill_rate: float,
    ) -> float:
        if hvaa_survival_rate < 0.8:
            return -1e9
        return mean_reward + 50.0 * red_kill_rate

    def _evaluate_policy_with_components(
        self,
    ) -> Tuple[float, float, float, List[float], Dict[str, Any], float, float]:
        """
        FIXED: Properly handles unnormalized observations for gate_action().
        """
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        episode_components: List[Dict[str, float]] = []
        hvaa_alive_count = 0
        red_kill_count = 0
        
        # CRITICAL FIX: Sync normalization stats BEFORE evaluation
        self._sync_normalization_stats()
        
        # Reset environment
        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        episodes_completed = 0
        
        current_episode_components: Dict[str, float] = {
            'base_reward': 0.0,
            'bez_shaping': 0.0,
            'dmc_shaping': 0.0,
            'heading_shaping': 0.0,
            'distance_shaping': 0.0,
            'ttc_shaping': 0.0,
            'multithreat_shaping': 0.0,
            'speed_shaping': 0.0,
            'hvaa_escort_shaping': 0.0,
            'offense_wez_shaping': 0.0,
            'offense_ttc_shaping': 0.0,
            'potential_diff': 0.0,
            'total_shaping': 0.0,
            'total_reward': 0.0,
        }
        
        # Debug: Track gate_action behavior
        gate_stats = {'fire_requests': 0, 'fire_passed': 0}
        
        while episodes_completed < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)

            # CRITICAL FIX: Get UNNORMALIZED observation for gate_action
            unnorm_obs = self._get_unnormalized_obs(obs)
            
            # Apply action gating with UNNORMALIZED observation
            try:
                gated_list = self.eval_env.env_method(
                    "gate_action", 
                    action[0], 
                    unnorm_obs[0],  # FIXED: Use unnormalized obs!
                    indices=[0]
                )
                
                # Track fire gating behavior for debugging
                if action[0, 3] > 0.5:  # Fire requested
                    gate_stats['fire_requests'] += 1
                    if gated_list[0][3] > 0.5:  # Fire passed gate
                        gate_stats['fire_passed'] += 1
                
                action = np.asarray([gated_list[0]], dtype=np.float32)
            except Exception as e:
                if self.debug_obs:
                    print(f"[EVAL DEBUG] gate_action failed: {e}")
            
            step_out = self.eval_env.step(action)
            
            if len(step_out) == 4:
                obs, reward, done, infos = step_out
                terminated, truncated = done, np.array([False] * len(done))
            else:
                obs, reward, terminated, truncated, infos = step_out
                done = np.logical_or(terminated, truncated)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Extract reward components from info
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            breakdown = info0.get('reward_breakdown', {})
            
            for key in current_episode_components:
                if key in breakdown:
                    current_episode_components[key] += float(breakdown.get(key, 0.0))
            
            if done[0]:
                # Check mission outcomes
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
                          f"reward={episode_reward:.1f} len={episode_length} "
                          f"red_killed={red_killed} hvaa_alive={not hvaa_destroyed}")
                
                # Reset tracking
                episode_reward = 0.0
                episode_length = 0
                current_episode_components = {k: 0.0 for k in current_episode_components}
        
        # Print gate stats
        if gate_stats['fire_requests'] > 0:
            pass_rate = 100 * gate_stats['fire_passed'] / gate_stats['fire_requests']
            print(f"[EVAL] Fire gate: {gate_stats['fire_passed']}/{gate_stats['fire_requests']} "
                  f"passed ({pass_rate:.1f}%)")
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        hvaa_survival_rate = hvaa_alive_count / self.n_eval_episodes
        red_kill_rate = red_kill_count / self.n_eval_episodes
        
        # Aggregate component stats
        component_stats = self._aggregate_component_stats(episode_components)
        
        return (mean_reward, std_reward, mean_length, episode_rewards, 
                component_stats, hvaa_survival_rate, red_kill_rate)

    def _aggregate_component_stats(self, episode_components: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate component statistics across episodes."""
        if not episode_components:
            return {'aggregate': {}, 'per_episode': []}
        
        aggregate = {}
        all_keys = set()
        for ep in episode_components:
            all_keys.update(ep.keys())
        
        for key in all_keys:
            values = [ep.get(key, 0.0) for ep in episode_components]
            aggregate[f'{key}_mean'] = float(np.mean(values))
            aggregate[f'{key}_std'] = float(np.std(values))
        
        return {
            'aggregate': aggregate,
            'per_episode': episode_components
        }

    def _on_training_start(self) -> None:
        """Run initial evaluation at timestep 0."""
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔍 INITIAL EVALUATION at 0 steps (FIXED obs handling)")
            print(f"   Running {self.n_eval_episodes} episodes...")
            print(f"{'='*60}")
        
        eval_start_time = time.time()
        (mean_reward, std_reward, mean_length, episode_rewards, 
         component_stats, hvaa_survival_rate, red_kill_rate) = self._evaluate_policy_with_components()
        eval_time = time.time() - eval_start_time
        
        self.evaluations_timesteps.append(0)
        self.evaluations_results.append(mean_reward)
        self.evaluations_length.append(mean_length)
        self.evaluations_std.append(std_reward)
        self.evaluations_hvaa_survival.append(hvaa_survival_rate)
        self.evaluations_red_kill.append(red_kill_rate)
        self.evaluations_component_breakdown.append(component_stats)
        
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/hvaa_survival_rate", hvaa_survival_rate)
        self.logger.record("eval/red_kill_rate", red_kill_rate)
        
        if self.verbose > 0:
            print(f" Initial Evaluation Results:")
            print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   Mean Length: {mean_length:.1f}")
            print(f"   HVAA Survival: {hvaa_survival_rate:.2%}")
            print(f"   Red Kill Rate: {red_kill_rate:.2%}")
            print(f"   Time: {eval_time:.1f}s")
        
        # Save initial checkpoint
        checkpoint_path = self.eval_checkpoint_dir / "model_0_steps"
        self.model.save(checkpoint_path.as_posix())
        
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            norm_path = self.eval_checkpoint_dir / "vecnormalize_0_steps.pkl"
            vec_env.save(norm_path.as_posix())
        
        # Track best
        score = self.compute_eval_score(mean_reward, hvaa_survival_rate, red_kill_rate)
        if score > self.best_score:
            self.best_score = score
            self.best_mean_reward = mean_reward
            self._save_best_model(mean_reward, std_reward, mean_length, 
                                 hvaa_survival_rate, red_kill_rate, score, component_stats)
            print(f" ⭐ INITIAL BEST MODEL saved at timestep 0")
        
        print(f"{'='*60}\n")

    def _save_best_model(self, mean_reward, std_reward, mean_length,
                        hvaa_survival_rate, red_kill_rate, score, component_stats):
        """Save best model and metadata."""
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
        
        best_info_path = self.log_dir / "best_model_info.json"
        with open(best_info_path, 'w') as f:
            json.dump(self.best_stats, f, indent=2)

    def _on_step(self) -> bool:
        """Evaluate periodically during training."""
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"🔍 EVALUATION at {self.num_timesteps:,} steps (FIXED obs handling)")
                print(f"   Running {self.n_eval_episodes} episodes...")
                print(f"{'='*60}")
            
            eval_start_time = time.time()
            (mean_reward, std_reward, mean_length, episode_rewards,
             component_stats, hvaa_survival_rate, red_kill_rate) = self._evaluate_policy_with_components()
            eval_time = time.time() - eval_start_time
            
            # Store results
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(mean_reward)
            self.evaluations_length.append(mean_length)
            self.evaluations_std.append(std_reward)
            self.evaluations_hvaa_survival.append(hvaa_survival_rate)
            self.evaluations_red_kill.append(red_kill_rate)
            self.evaluations_component_breakdown.append(component_stats)
            
            # Log to tensorboard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/hvaa_survival_rate", hvaa_survival_rate)
            self.logger.record("eval/red_kill_rate", red_kill_rate)
            
            if component_stats['aggregate']:
                for key, value in component_stats['aggregate'].items():
                    if '_mean' in key:
                        self.logger.record(f"eval_components/{key}", value)
            
            if self.verbose > 0:
                print(f" Evaluation Results:")
                print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"   Mean Length: {mean_length:.1f}")
                print(f"   HVAA Survival: {hvaa_survival_rate:.2%}")
                print(f"   Red Kill Rate: {red_kill_rate:.2%}")
                print(f"   Time: {eval_time:.1f}s")
            
            # Save checkpoint
            checkpoint_path = self.eval_checkpoint_dir / f"model_{self.num_timesteps}_steps"
            self.model.save(checkpoint_path.as_posix())
            
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                norm_path = self.eval_checkpoint_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
                vec_env.save(norm_path.as_posix())
            
            # Save component logs
            component_log_path = self.component_logs_dir / f"components_{self.num_timesteps}_steps.json"
            with open(component_log_path, 'w') as f:
                json.dump(component_stats, f, indent=2)
            
            # Check for best model
            score = self.compute_eval_score(mean_reward, hvaa_survival_rate, red_kill_rate)
            
            if score > self.best_score:
                if self.verbose > 0:
                    improvement = score - self.best_score
                    print(f"⭐ NEW BEST MODEL! (+{improvement:.2f} score improvement)")
                
                self.best_score = score
                self.best_mean_reward = mean_reward
                self._save_best_model(mean_reward, std_reward, mean_length,
                                     hvaa_survival_rate, red_kill_rate, score, component_stats)
                
                if self.verbose > 0:
                    print(f"   Saved BEST MODEL to: {self.log_dir / 'best_model'}")
            else:
                if self.verbose > 0:
                    deficit = self.best_score - score
                    print(f"   (Best score: {self.best_score:.2f}, current is {deficit:.2f} below)")
            
            if self.verbose > 0:
                print(f"{'='*60}\n")
        
        return True

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
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
    print("RewardComponentEvalCallbackFIXED")
    print("="*60)
    print("\nFIXES:")
    print("  ✓ Properly handles unnormalized observations for gate_action()")
    print("  ✓ Robust fallback with manual unnormalization if get_original_obs() fails")
    print("  ✓ Better normalization stats sync before each evaluation")
    print("  ✓ Debug output to verify observation pipeline")
    print("\nThis should fix the training vs evaluation behavioral discrepancy!")
