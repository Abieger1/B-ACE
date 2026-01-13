#!/usr/bin/env python3
"""
reward_component_eval_callback.py

Enhanced evaluation callback that tracks detailed reward component breakdowns
during evaluation episodes, PLUS mission-level metrics:

- Mean reward
- HVAA survival rate (from hvaa_destroyed flag)
- Red kill rate (from Red_Killed flag)
- Composite score for best-model selection

Usage in training:
    from reward_component_eval_callback import RewardComponentEvalCallback
    
    eval_callback = RewardComponentEvalCallback(
        eval_env=eval_env,
        eval_freq=50000,
        n_eval_episodes=20,
        log_dir=log_dir,
        verbose=1
    )
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class RewardComponentEvalCallback(BaseCallback):
    """
    Enhanced evaluation callback that tracks reward component breakdowns and
    mission-level metrics (HVAA survival, Red kill rate), and uses a composite
    score for best-model selection.
    """
    
    def __init__(
        self,
        eval_env,
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
        self.best_mean_reward = -np.inf     # for reporting
        self.best_score = -np.inf           # composite score (used for selection)
        self.best_stats: Dict[str, Any] = {}
        
        # Evaluation histories
        self.evaluations_timesteps: List[int] = []
        self.evaluations_results: List[float] = []           # mean reward
        self.evaluations_length: List[float] = []            # mean length
        self.evaluations_std: List[float] = []               # std reward
        self.evaluations_hvaa_survival: List[float] = []     # HVAA survival rate
        self.evaluations_red_kill: List[float] = []          # Red kill rate
        
        # Reward component tracking
        self.evaluations_component_breakdown: List[Dict[str, Any]] = []  # List of dicts per evaluation
        
        # Directories
        self.eval_checkpoint_dir = self.log_dir / "eval_checkpoints"
        self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.component_logs_dir = self.log_dir / "reward_component_logs"
        self.component_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last evaluation
        self.last_eval_step = 0

    # ------------------------------------------------------------------
    # Composite score for best-model selection
    # ------------------------------------------------------------------
    @staticmethod
    def compute_eval_score(
        mean_reward: float,
        hvaa_survival_rate: float,
        red_kill_rate: float,
    ) -> float:
        """
        Composite score used to compare policies.

        1) Enforce HVAA survival as a hard-ish constraint:
           - If survival < 0.8, reject this policy (very low score).
        2) Within that feasible set, prefer higher Red kill rate and mean reward.
        """
        if hvaa_survival_rate < 0.8:
            return -1e9  # reject models that don't protect the HVAA

        # Reward policies that kill Red more often and have higher mean reward.
        # The 50.0 weight is tunable.
        return mean_reward + 50.0 * red_kill_rate
        
    # ------------------------------------------------------------------
    # Evaluation with components + mission metrics
    # ------------------------------------------------------------------
    def _evaluate_policy_with_components(
        self,
    ) -> Tuple[float, float, float, List[float], Dict[str, Any], float, float]:
        """
        Run n_eval_episodes and collect:
            - mean_reward, std_reward, mean_length
            - episode_rewards
            - component_stats: detailed reward component breakdown
            - hvaa_survival_rate: fraction of episodes where HVAA survived
            - red_kill_rate: fraction of episodes where Red was killed

        Reads mission metrics from info[0] at episode termination, using:
            - "Red_Killed" (your env's key)
            - "hvaa_destroyed" (True if HVAA died)
        """
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        
        # Component tracking - per episode
        episode_components: List[Dict[str, float]] = []
        
        # Mission-level counters
        hvaa_alive_count = 0
        red_kill_count = 0
        
        # ---- KEEP EVAL NORMALIZATION IDENTICAL TO TRAINING NORMALIZATION ----
        try:
            train_env = self.model.get_env()
            if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = train_env.obs_rms
                self.eval_env.ret_rms = train_env.ret_rms
        except Exception:
            pass

        # Reset environment
        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        episodes_completed = 0
        
        # Track components for current episode
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
        
        while episodes_completed < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)

            # Apply training-equivalent action gating during eval
            try:
                if hasattr(self.eval_env, "get_original_obs"):
                    obs_for_gate = self.eval_env.get_original_obs()
                    if obs_for_gate is None:
                        obs_for_gate = obs
                else:
                    obs_for_gate = obs

                gated_list = self.eval_env.env_method("gate_action", action[0], obs_for_gate[0], indices=[0])
                action = np.asarray([gated_list[0]], dtype=np.float32)
            except Exception:
                pass

            step_out = self.eval_env.step(action)
            if len(step_out) == 4:
                obs, reward, done, infos = step_out
            else:
                obs, reward, terminated, truncated, infos = step_out
                done = np.logical_or(terminated, truncated)

            
            episode_reward += reward[0]
            episode_length += 1
            
            # Extract reward components if available in info
            # The RewardShapingWrapper puts this in info[0]['reward_breakdown']
            if len(infos) > 0 and isinstance(infos[0], dict):
                breakdown = infos[0].get('reward_breakdown', None)
                if breakdown is not None:
                    for key in current_episode_components.keys():
                        if key in breakdown:
                            current_episode_components[key] += breakdown[key]
            '''
            if done[0]:
                # Mission-level metrics from info[0] at termination
                info0 = infos[0] if (isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict)) else {}

                # Print the single "episode end key"
                ep = info0.get("episode", None)
                if self.verbose > 0:
                    print(f"[EVAL EP-END] ep={episodes_completed+1}/{self.n_eval_episodes} episode={ep}")
                '''
            if done[0]:
                info0 = infos[0] if (isinstance(infos, (list, tuple)) and len(infos) > 0) else {}

                ep = info0.get("episode", {})
                ep_reward = ep.get("r", float("nan"))
                true_return = float(episode_reward)
                ep_len = ep.get("l", -1)

                term = bool(info0.get("terminated", False))
                trunc = bool(info0.get("truncated", False))

                reason = (
                    info0.get("episode_end_reason")
                    or info0.get("termination_reason")
                    or info0.get("reasons")
                    or ("TIME_LIMIT" if ep_len == 6000 else None)
                    or "<?>"
                )

                print(
                    f"[EVAL EP-END] ep={episodes_completed+1}/{self.n_eval_episodes} "
                    f"return={ep_reward:.4f} len={ep_len} "
                    f"return={true_return:.4f} len={episode_length} "
                    f"terminated={term} truncated={trunc} reason={reason}"
                )
                # EXACT KEYS from your env:
                red_kill_flag = info0.get("Red_Killed", 0)
                hvaa_destroyed_flag = info0.get("hvaa_destroyed", 0)
                
                if red_kill_flag:
                    red_kill_count += 1
                
                # HVAA survives if it was NOT destroyed
                if not hvaa_destroyed_flag:
                    hvaa_alive_count += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Store this episode's component breakdown
                episode_components.append(current_episode_components.copy())
                
                episodes_completed += 1
                
                # Reset for next episode
                episode_reward = 0.0
                episode_length = 0
                current_episode_components = {k: 0.0 for k in current_episode_components.keys()}
        
        # Aggregate reward stats
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_length = float(np.mean(episode_lengths))
        
        # Mission-level rates
        hvaa_survival_rate = hvaa_alive_count / self.n_eval_episodes
        red_kill_rate = red_kill_count / self.n_eval_episodes
        
        # Compute mean component contributions across all episodes
        component_stats: Dict[str, Any] = {
            'per_episode': episode_components,
            'aggregate': {},
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_lengths': [int(l) for l in episode_lengths],
        }
        
        if episode_components:
            keys = episode_components[0].keys()
            for key in keys:
                values = [ep[key] for ep in episode_components]
                component_stats['aggregate'][f'{key}_mean'] = float(np.mean(values))
                component_stats['aggregate'][f'{key}_std'] = float(np.std(values))
                component_stats['aggregate'][f'{key}_min'] = float(np.min(values))
                component_stats['aggregate'][f'{key}_max'] = float(np.max(values))
        
        return (
            mean_reward,
            std_reward,
            mean_length,
            episode_rewards,
            component_stats,
            hvaa_survival_rate,
            red_kill_rate,
        )
    
    # ------------------------------------------------------------------
    # Initial evaluation at training start
    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        """Run initial evaluation at timestep 0."""
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔍 INITIAL EVALUATION at 0 steps (with reward components)")
            print(f"   Running {self.n_eval_episodes} episodes...")
            print(f"{'='*60}")
        
        eval_start_time = time.time()
        (
            mean_reward,
            std_reward,
            mean_length,
            episode_rewards,
            component_stats,
            hvaa_survival_rate,
            red_kill_rate,
        ) = self._evaluate_policy_with_components()
        eval_time = time.time() - eval_start_time
        
        # Store evaluation results
        self.evaluations_timesteps.append(0)
        self.evaluations_results.append(mean_reward)
        self.evaluations_length.append(mean_length)
        self.evaluations_std.append(std_reward)
        self.evaluations_component_breakdown.append(component_stats)
        self.evaluations_hvaa_survival.append(hvaa_survival_rate)
        self.evaluations_red_kill.append(red_kill_rate)
        
        # Log to tensorboard
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/hvaa_survival_rate", hvaa_survival_rate)
        self.logger.record("eval/red_kill_rate", red_kill_rate)
        
        # Log component breakdowns to tensorboard
        if component_stats['aggregate']:
            for key, value in component_stats['aggregate'].items():
                if '_mean' in key:
                    self.logger.record(f"eval_components/{key}", value)
        
        if self.verbose > 0:
            print(f" Initial Evaluation Results:")
            print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   Mean Length: {mean_length:.1f}")
            print(f"   HVAA Survival Rate: {hvaa_survival_rate:.2%}")
            print(f"   Red Kill Rate: {red_kill_rate:.2%}")
            print(f"   Time: {eval_time:.1f}s")
            
            # Print component breakdown summary
            if component_stats['aggregate']:
                print(f"\n   Reward Component Breakdown (mean per episode):")
                print(f"   {'Component':<25} {'Mean':>10} {'Std':>10}")
                print(f"   {'-'*47}")
                for key in sorted(component_stats['aggregate'].keys()):
                    if '_mean' in key:
                        component_name = key.replace('_mean', '')
                        mean_val = component_stats['aggregate'][key]
                        std_key = key.replace('_mean', '_std')
                        std_val = component_stats['aggregate'].get(std_key, 0.0)
                        print(f"   {component_name:<25} {mean_val:>10.4f} {std_val:>10.4f}")
        
        # Save detailed component breakdown to JSON
        component_log_path = self.component_logs_dir / f"components_0_steps.json"
        with open(component_log_path, 'w') as f:
            json.dump(component_stats, f, indent=2)
        
        # Save checkpoint for this initial eval
        checkpoint_path = self.eval_checkpoint_dir / "model_0_steps"
        self.model.save(checkpoint_path.as_posix())
        
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            norm_path = self.eval_checkpoint_dir / "vecnormalize_0_steps.pkl"
            vec_env.save(norm_path.as_posix())
        
        if self.verbose > 0:
            print(f" Checkpoint and component log saved")
            print(f"{'='*60}\n")
        
        # Initialize best model tracking using composite score
        score = self.compute_eval_score(
            mean_reward=mean_reward,
            hvaa_survival_rate=hvaa_survival_rate,
            red_kill_rate=red_kill_rate,
        )
        self.best_score = score
        self.best_mean_reward = mean_reward
        self.best_stats = {
            "timestep": 0,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "mean_length": float(mean_length),
            "hvaa_survival_rate": float(hvaa_survival_rate),
            "red_kill_rate": float(red_kill_rate),
            "score": float(score),
            "component_breakdown": component_stats['aggregate'],
        }
        
        best_model_path = self.log_dir / "best_model"
        self.model.save(best_model_path.as_posix())
        
        if isinstance(vec_env, VecNormalize):
            best_norm_path = self.log_dir / "best_vecnormalize.pkl"
            vec_env.save(best_norm_path.as_posix())
        
        best_info_path = self.log_dir / "best_model_info.json"
        with open(best_info_path, "w") as f:
            json.dump(self.best_stats, f, indent=2)
    
    # ------------------------------------------------------------------
    # Periodic evaluation during training
    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        """Evaluate periodically during training."""
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"🔍 EVALUATION at {self.num_timesteps:,} steps (with reward components)")
                print(f"   Running {self.n_eval_episodes} episodes...")
                print(f"{'='*60}")
            
            eval_start_time = time.time()
            (
                mean_reward,
                std_reward,
                mean_length,
                episode_rewards,
                component_stats,
                hvaa_survival_rate,
                red_kill_rate,
            ) = self._evaluate_policy_with_components()
            eval_time = time.time() - eval_start_time
            
            # Store evaluation results
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(mean_reward)
            self.evaluations_length.append(mean_length)
            self.evaluations_std.append(std_reward)
            self.evaluations_component_breakdown.append(component_stats)
            self.evaluations_hvaa_survival.append(hvaa_survival_rate)
            self.evaluations_red_kill.append(red_kill_rate)
            
            # Log to tensorboard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/hvaa_survival_rate", hvaa_survival_rate)
            self.logger.record("eval/red_kill_rate", red_kill_rate)
            
            # Log component breakdowns to tensorboard
            if component_stats['aggregate']:
                for key, value in component_stats['aggregate'].items():
                    if '_mean' in key:
                        self.logger.record(f"eval_components/{key}", value)
            
            if self.verbose > 0:
                print(f" Evaluation Results:")
                print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"   Mean Length: {mean_length:.1f}")
                print(f"   HVAA Survival Rate: {hvaa_survival_rate:.2%}")
                print(f"   Red Kill Rate: {red_kill_rate:.2%}")
                print(f"   Time: {eval_time:.1f}s")
                
                if component_stats['aggregate']:
                    print(f"\n   Reward Component Breakdown (mean per episode):")
                    print(f"   {'Component':<25} {'Mean':>10} {'Std':>10}")
                    print(f"   {'-'*47}")
                    for key in sorted(component_stats['aggregate'].keys()):
                        if '_mean' in key:
                            component_name = key.replace('_mean', '')
                            mean_val = component_stats['aggregate'][key]
                            std_key = key.replace('_mean', '_std')
                            std_val = component_stats['aggregate'].get(std_key, 0.0)
                            print(f"   {component_name:<25} {mean_val:>10.4f} {std_val:>10.4f}")
            
            # Save detailed component breakdown to JSON
            component_log_path = self.component_logs_dir / f"components_{self.num_timesteps}_steps.json"
            with open(component_log_path, 'w') as f:
                json.dump(component_stats, f, indent=2)
            
            # Save checkpoint for this evaluation
            checkpoint_path = self.eval_checkpoint_dir / f"model_{self.num_timesteps}_steps"
            self.model.save(checkpoint_path.as_posix())
            
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                norm_path = self.eval_checkpoint_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
                vec_env.save(norm_path.as_posix())
            
            if self.verbose > 0:
                print(f"💾 Checkpoint and component log saved")
            
            # Composite score-based best model selection
            score = self.compute_eval_score(
                mean_reward=mean_reward,
                hvaa_survival_rate=hvaa_survival_rate,
                red_kill_rate=red_kill_rate,
            )
            
            if score > self.best_score:
                if self.verbose > 0:
                    improvement = score - self.best_score
                    print(f"⭐ NEW BEST MODEL! (+{improvement:.2f} score improvement)")
                
                self.best_score = score
                self.best_mean_reward = mean_reward
                self.best_stats = {
                    'timestep': self.num_timesteps,
                    'mean_reward': float(mean_reward),
                    'std_reward': float(std_reward),
                    'mean_length': float(mean_length),
                    'hvaa_survival_rate': float(hvaa_survival_rate),
                    'red_kill_rate': float(red_kill_rate),
                    'score': float(score),
                    'component_breakdown': component_stats['aggregate'],
                }
                
                # Save as best model
                best_model_path = self.log_dir / "best_model"
                self.model.save(best_model_path.as_posix())
                
                if isinstance(vec_env, VecNormalize):
                    best_norm_path = self.log_dir / "best_vecnormalize.pkl"
                    vec_env.save(best_norm_path.as_posix())
                
                # Save metadata about best model (including components and mission metrics)
                best_info_path = self.log_dir / "best_model_info.json"
                with open(best_info_path, 'w') as f:
                    json.dump(self.best_stats, f, indent=2)
                
                if self.verbose > 0:
                    print(f"   Saved BEST MODEL to: {best_model_path}")
            else:
                if self.verbose > 0:
                    deficit = self.best_score - score
                    print(f"   (Best score: {self.best_score:.2f}, current is {deficit:.2f} below)")
            
            if self.verbose > 0:
                print(f"{'='*60}\n")
        
        return True
    
    # ------------------------------------------------------------------
    # Summary for downstream analysis
    # ------------------------------------------------------------------
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations including component breakdowns and mission metrics."""
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
    print("RewardComponentEvalCallback - Enhanced Evaluation with Components + Mission Metrics")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Tracks individual reward components during evaluation")
    print("  ✓ Tracks HVAA survival and Red kill rates")
    print("  ✓ Uses composite score (reward + mission metrics) for best-model selection")
    print("  ✓ Logs to TensorBoard under 'eval/' and 'eval_components/' namespaces")
    print("  ✓ Saves detailed JSON breakdowns per evaluation")
    print("  ✓ Compatible with RewardShapingWrapper")
    print("\nUsage:")
    print("  Use RewardComponentEvalCallback in place of EvaluationBasedCheckpointCallback")
    print("  Best model now respects mission objectives, not just mean reward.")
