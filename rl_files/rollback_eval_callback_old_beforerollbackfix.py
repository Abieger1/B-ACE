"""
rollback_eval_callback.py

Enhanced Evaluation Callback with Automatic Rollback and Learning Rate Decay

When performance degrades for N consecutive evaluations, this callback:
1. Rolls back model weights to the best checkpoint
2. Rolls back VecNormalize statistics 
3. Halves the learning rate
4. Continues training from that point

This helps capture learning in chaotic/non-stationary reward landscapes.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


class RollbackEvaluationCallback(BaseCallback):
    """
    Evaluation callback with automatic rollback when performance degrades.
    
    Features:
    - Periodic evaluation with deterministic policy
    - Best model tracking and checkpointing
    - Automatic rollback after N consecutive performance drops
    - Learning rate halving on rollback
    - Configurable minimum LR floor
    - Activation threshold: rollback only activates after reaching a minimum performance
    
    Args:
        eval_env: Evaluation environment (should be VecNormalize wrapped)
        eval_freq: Steps between evaluations
        n_eval_episodes: Episodes per evaluation
        log_dir: Directory for checkpoints and logs
        rollback_patience: Number of consecutive declines before rollback (default: 3)
        lr_decay_factor: Factor to multiply LR by on rollback (default: 0.5)
        min_lr: Minimum learning rate floor (default: 1e-6)
        min_lr_multiplier: Minimum LR multiplier floor - prevents compounding decay
                           from reducing LR below this fraction of base schedule.
                           Example: 0.1 means LR never drops below 10% of original.
                           (default: 0.1)
        max_rollbacks: Maximum number of rollbacks allowed (default: 10)
        activation_threshold: Minimum reward that must be achieved before rollback 
                              protection activates. Set to None to always be active.
                              Example: -15.0 means rollback only protects after 
                              achieving at least -15.0 mean eval reward. (default: None)
        skip_initial_eval: If True, skip the evaluation at timestep 0 (default: False)
        verbose: Verbosity level
        deterministic: Use deterministic actions during eval
    """
    
    def __init__(
        self,
        eval_env: DummyVecEnv,
        eval_freq: int = 200000,
        n_eval_episodes: int = 15,
        log_dir: Path = None,
        rollback_patience: int = 3,
        lr_decay_factor: float = 0.5,
        min_lr: float = 1e-6,
        min_lr_multiplier: float = 0.1,  # Floor: never reduce below 10% of base LR
        max_rollbacks: int = 10,
        activation_threshold: Optional[float] = None,
        skip_initial_eval: bool = False,
        verbose: int = 1,
        deterministic: bool = True,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.deterministic = deterministic
        self.skip_initial_eval = skip_initial_eval
        
        # Rollback configuration
        self.rollback_patience = rollback_patience
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.min_lr_multiplier = min_lr_multiplier
        self.max_rollbacks = max_rollbacks
        self.activation_threshold = activation_threshold
        
        # Tracking state
        self.best_mean_reward = -np.inf
        self.best_timestep = 0
        self.consecutive_declines = 0
        self.total_rollbacks = 0
        self.current_lr_multiplier = 1.0  # Tracks cumulative LR reductions
        
        # Activation tracking - rollback only active after threshold is reached
        self.rollback_activated = False if activation_threshold is not None else True
        self.activation_timestep = None  # When threshold was first reached
        
        # Evaluation history
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []
        self.evaluations_std = []
        
        # Rollback history (for thesis documentation)
        self.rollback_events = []
        
        # Create checkpoint directories
        self.eval_checkpoint_dir = self.log_dir / "eval_checkpoints"
        self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_checkpoint_dir = self.log_dir / "best_checkpoint"
        self.best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last evaluation step
        self.last_eval_step = 0
        
        # Store original LR schedule for modification
        self._original_lr_schedule = None
        self._base_lr_schedule = None

    def _get_current_lr(self) -> float:
        """Get the current learning rate value."""
        if hasattr(self.model, 'learning_rate'):
            lr = self.model.learning_rate
            if callable(lr):
                progress = 1.0 - (self.num_timesteps / self.model._total_timesteps)
                return lr(progress) * self.current_lr_multiplier
            else:
                return lr * self.current_lr_multiplier
        return 0.0

    def _create_scaled_lr_schedule(self, base_schedule: Callable, multiplier: float) -> Callable:
        """Create a new LR schedule that scales the base schedule by multiplier."""
        def scaled_schedule(progress_remaining: float) -> float:
            base_lr = base_schedule(progress_remaining)
            scaled_lr = base_lr * multiplier
            return max(scaled_lr, self.min_lr)
        return scaled_schedule

    def _update_learning_rate(self, new_multiplier: float):
        """Update the model's learning rate with a new multiplier."""
        self.current_lr_multiplier = new_multiplier
        
        if self._base_lr_schedule is None:
            # Store the original schedule on first call
            self._base_lr_schedule = self.model.learning_rate
        
        if callable(self._base_lr_schedule):
            # Create new scaled schedule
            new_schedule = self._create_scaled_lr_schedule(self._base_lr_schedule, new_multiplier)
            self.model.learning_rate = new_schedule
            self.model.lr_schedule = new_schedule
            
            # Update optimizer LR directly for immediate effect
            current_lr = new_schedule(1.0 - (self.num_timesteps / self.model._total_timesteps))
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            # Fixed LR - just scale it
            new_lr = max(self._base_lr_schedule * new_multiplier, self.min_lr)
            self.model.learning_rate = new_lr
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        if self.verbose > 0:
            actual_lr = self._get_current_lr()
            print(f"   📉 Learning rate updated: multiplier={new_multiplier:.4f}, current_lr={actual_lr:.2e}")

    def _safe_save_model(self, path: Path) -> bool:
        """
        Safely save the model, falling back to policy-only save if pickling fails.
        
        Returns True if save succeeded, False otherwise.
        """
        try:
            # Try full model save first
            self.model.save(path.as_posix())
            return True
        except Exception as e:
            if "pickle" in str(e).lower() or "tty" in str(e).lower():
                # Pickling error - fall back to saving just the policy state dict
                if self.verbose > 0:
                    print(f"   ⚠️  Full model save failed (pickling error), saving policy weights only...")
                try:
                    # Save just the PyTorch state dict
                    policy_path = path.parent / f"{path.stem}_policy.pt"
                    th.save({
                        'policy_state_dict': self.model.policy.state_dict(),
                        'timestep': self.num_timesteps,
                    }, policy_path.as_posix())
                    
                    if self.verbose > 0:
                        print(f"   ✓ Policy weights saved to {policy_path.name}")
                    return True
                except Exception as e2:
                    if self.verbose > 0:
                        print(f"   ❌ Policy save also failed: {e2}")
                    return False
            else:
                if self.verbose > 0:
                    print(f"   ❌ Model save failed: {e}")
                return False

    def _save_best_checkpoint(self, mean_reward: float, std_reward: float, 
                              mean_length: float, episode_rewards: List[float]):
        """Save the current model as the best checkpoint."""
        # Save model
        model_path = self.best_checkpoint_dir / "best_model"
        self._safe_save_model(model_path)
        
        # Always save policy state dict as backup (this never has pickling issues)
        policy_backup_path = self.best_checkpoint_dir / "best_policy.pt"
        th.save({
            'policy_state_dict': self.model.policy.state_dict(),
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
        }, policy_backup_path.as_posix())
        
        # Save VecNormalize statistics
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            norm_path = self.best_checkpoint_dir / "best_vecnormalize.pkl"
            vec_env.save(norm_path.as_posix())
        
        # Save metadata
        best_info = {
            'timestep': int(self.num_timesteps),
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_length': float(mean_length),
            'episode_rewards': [float(r) for r in episode_rewards],
            'lr_multiplier': float(self.current_lr_multiplier),
            'total_rollbacks_at_save': int(self.total_rollbacks),
        }
        
        info_path = self.best_checkpoint_dir / "best_model_info.json"
        with open(info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        # Also save to main log_dir for easy access
        self._safe_save_model(self.log_dir / "best_model")
        th.save({
            'policy_state_dict': self.model.policy.state_dict(),
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
        }, (self.log_dir / "best_policy.pt").as_posix())
        
        if isinstance(vec_env, VecNormalize):
            vec_env.save((self.log_dir / "best_vecnormalize.pkl").as_posix())

    def _perform_rollback(self):
        """Roll back to the best checkpoint and reduce learning rate."""
        if self.total_rollbacks >= self.max_rollbacks:
            if self.verbose > 0:
                print(f"\n⚠️  Maximum rollbacks ({self.max_rollbacks}) reached. Continuing without rollback.")
            return False
        
        model_path = self.best_checkpoint_dir / "best_model.zip"
        policy_path = self.best_checkpoint_dir / "best_policy.pt"
        
        # Check if we have any checkpoint to load
        if not model_path.exists() and not policy_path.exists():
            if self.verbose > 0:
                print(f"\n⚠️  No best checkpoint found. Cannot rollback.")
            return False
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔄 ROLLBACK TRIGGERED")
            print(f"   Consecutive declines: {self.consecutive_declines}")
            print(f"   Rolling back to timestep {self.best_timestep} (reward: {self.best_mean_reward:.2f})")
            print(f"{'='*60}")
        
        # Load best model weights into current model
        weights_loaded = False
        
        # Try loading full model first
        if model_path.exists():
            try:
                saved_model = self.model.__class__.load(
                    model_path.as_posix(),
                    env=None,
                    device=self.model.device,
                )
                self.model.policy.load_state_dict(saved_model.policy.state_dict())
                del saved_model
                weights_loaded = True
                
                if self.verbose > 0:
                    print(f"   ✓ Model weights restored (from full model)")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"   ⚠️  Full model load failed: {e}")
        
        # Fall back to policy-only checkpoint
        if not weights_loaded and policy_path.exists():
            try:
                checkpoint = th.load(policy_path.as_posix(), map_location=self.model.device)
                self.model.policy.load_state_dict(checkpoint['policy_state_dict'])
                weights_loaded = True
                
                if self.verbose > 0:
                    print(f"   ✓ Model weights restored (from policy checkpoint)")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"   ❌ Policy checkpoint load failed: {e}")
        
        if not weights_loaded:
            if self.verbose > 0:
                print(f"   ❌ Could not restore any weights. Rollback failed.")
            return False
        
        # Restore VecNormalize statistics
        vec_env = self.model.get_env()
        norm_path = self.best_checkpoint_dir / "best_vecnormalize.pkl"
        if isinstance(vec_env, VecNormalize) and norm_path.exists():
            try:
                saved_vec_env = VecNormalize.load(norm_path.as_posix(), vec_env.venv)
                vec_env.obs_rms = saved_vec_env.obs_rms
                vec_env.ret_rms = saved_vec_env.ret_rms
                
                # Also update eval env
                if isinstance(self.eval_env, VecNormalize):
                    self.eval_env.obs_rms = saved_vec_env.obs_rms
                    self.eval_env.ret_rms = saved_vec_env.ret_rms
                
                if self.verbose > 0:
                    print(f"   ✓ VecNormalize statistics restored")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"   ⚠️  Error restoring VecNormalize: {e}")
        
        # Reduce learning rate (with floor to prevent death spiral)
        new_multiplier = max(
            self.current_lr_multiplier * self.lr_decay_factor,
            self.min_lr_multiplier  # Never go below this fraction of base LR
        )
        old_lr = self._get_current_lr()
        self._update_learning_rate(new_multiplier)
        new_lr = self._get_current_lr()
        
        # Record rollback event
        rollback_event = {
            'timestep': int(self.num_timesteps),
            'rollback_to_timestep': int(self.best_timestep),
            'consecutive_declines': int(self.consecutive_declines),
            'old_lr_multiplier': float(self.current_lr_multiplier / self.lr_decay_factor),
            'new_lr_multiplier': float(self.current_lr_multiplier),
            'old_lr': float(old_lr),
            'new_lr': float(new_lr),
            'best_reward_at_rollback': float(self.best_mean_reward),
            'recent_rewards': self.evaluations_results[-self.rollback_patience:] if len(self.evaluations_results) >= self.rollback_patience else self.evaluations_results,
        }
        self.rollback_events.append(rollback_event)
        
        # Save rollback history
        rollback_history_path = self.log_dir / "rollback_history.json"
        with open(rollback_history_path, 'w') as f:
            json.dump(self.rollback_events, f, indent=2)
        
        # Reset decline counter and increment rollback count
        self.consecutive_declines = 0
        self.total_rollbacks += 1
        
        if self.verbose > 0:
            print(f"   ✓ Learning rate: {old_lr:.2e} → {new_lr:.2e}")
            print(f"   Total rollbacks: {self.total_rollbacks}/{self.max_rollbacks}")
            print(f"{'='*60}\n")
        
        return True

    def _evaluate_policy(self) -> Tuple[float, float, float, List[float]]:
        """Run evaluation episodes and return statistics."""
        episode_rewards = []
        episode_lengths = []
        
        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        episodes_completed = 0
        
        while episodes_completed < self.n_eval_episodes:
            # Update global step in env if method exists
            try:
                self.eval_env.env_method("set_global_step", int(self.num_timesteps))
            except:
                pass
            
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            
            # Try to apply action gating if available
            try:
                if hasattr(self.eval_env, "get_original_obs"):
                    obs_for_gate = self.eval_env.get_original_obs()
                    if obs_for_gate is None:
                        obs_for_gate = obs
                else:
                    obs_for_gate = obs
                gated = self.eval_env.env_method("gate_action", action[0], obs_for_gate[0], indices=[0])[0]
                action = np.asarray([gated], dtype=np.float32)
            except:
                pass
            
            step_out = self.eval_env.step(action)
            
            if len(step_out) == 4:
                obs, reward, done, infos = step_out
            else:
                obs, reward, terminated, truncated, infos = step_out
                done = np.logical_or(terminated, truncated)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0]:
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                ep = info0.get("episode", None)
                
                if self.verbose > 0:
                    print(f"[EVAL EP-END] ep={episodes_completed+1}/{self.n_eval_episodes} episode={ep}")
                
                episode_rewards.append(float(episode_reward))
                episode_lengths.append(int(episode_length))
                episodes_completed += 1
                
                episode_reward = 0.0
                episode_length = 0
        
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_length = float(np.mean(episode_lengths))
        
        return mean_reward, std_reward, mean_length, episode_rewards

    def _on_training_start(self) -> None:
        """Initial evaluation at timestep 0."""
        # Store original learning rate schedule
        self._base_lr_schedule = self.model.learning_rate
        
        if self.skip_initial_eval:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"⏭️  SKIPPING INITIAL EVALUATION (skip_initial_eval=True)")
                print(f"   Rollback patience: {self.rollback_patience} consecutive declines")
                print(f"   LR decay on rollback: {self.lr_decay_factor}x")
                if self.activation_threshold is not None:
                    print(f"   ⏳ Activation threshold: {self.activation_threshold:.2f}")
                print(f"{'='*60}\n")
            return
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔍 INITIAL EVALUATION at 0 steps")
            print(f"   Running {self.n_eval_episodes} episodes...")
            print(f"   Rollback patience: {self.rollback_patience} consecutive declines")
            print(f"   LR decay on rollback: {self.lr_decay_factor}x")
            if self.activation_threshold is not None:
                print(f"   ⏳ Activation threshold: {self.activation_threshold:.2f}")
                print(f"      (Rollback protection inactive until threshold reached)")
            else:
                print(f"   ✓ Rollback protection: ALWAYS ACTIVE")
            print(f"{'='*60}")
        
        eval_start_time = time.time()
        mean_reward, std_reward, mean_length, episode_rewards = self._evaluate_policy()
        eval_time = time.time() - eval_start_time
        
        # Store results
        self.evaluations_timesteps.append(0)
        self.evaluations_results.append(mean_reward)
        self.evaluations_length.append(mean_length)
        self.evaluations_std.append(std_reward)
        
        # Log to tensorboard
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("rollback/consecutive_declines", self.consecutive_declines)
        self.logger.record("rollback/total_rollbacks", self.total_rollbacks)
        self.logger.record("rollback/lr_multiplier", self.current_lr_multiplier)
        
        if self.verbose > 0:
            print(f" Initial Evaluation Results:")
            print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   Mean Length: {mean_length:.1f}")
            print(f"   Time: {eval_time:.1f}s")
        
        # This is the initial best
        self.best_mean_reward = mean_reward
        self.best_timestep = 0
        self._save_best_checkpoint(mean_reward, std_reward, mean_length, episode_rewards)
        
        if self.verbose > 0:
            print(f"⭐ INITIAL BEST MODEL saved")
            print(f"{'='*60}\n")

    def _on_step(self) -> bool:
        """Called at each step - check if evaluation is needed."""
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"🔍 EVALUATION at {self.num_timesteps:,} steps")
                print(f"   Running {self.n_eval_episodes} episodes...")
                print(f"{'='*60}")
            
            # Sync normalization stats
            eval_start_time = time.time()
            try:
                train_env = self.model.get_env()
                if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                    self.eval_env.obs_rms = train_env.obs_rms
                    self.eval_env.ret_rms = train_env.ret_rms
            except:
                pass
            
            # Run evaluation
            mean_reward, std_reward, mean_length, episode_rewards = self._evaluate_policy()
            eval_time = time.time() - eval_start_time
            
            # Store results
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
            
            # Save checkpoint for this evaluation (with safe fallback)
            checkpoint_path = self.eval_checkpoint_dir / f"model_{self.num_timesteps}_steps"
            self._safe_save_model(checkpoint_path)
            
            # Always save policy state dict backup (never fails)
            policy_path = self.eval_checkpoint_dir / f"policy_{self.num_timesteps}_steps.pt"
            th.save({
                'policy_state_dict': self.model.policy.state_dict(),
                'timestep': self.num_timesteps,
                'mean_reward': mean_reward,
            }, policy_path.as_posix())
            
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                norm_path = self.eval_checkpoint_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
                vec_env.save(norm_path.as_posix())
            
            if self.verbose > 0:
                print(f"💾 Checkpoint saved")
            
            # Check if activation threshold is reached (only matters if threshold is set)
            if not self.rollback_activated and self.activation_threshold is not None:
                if mean_reward >= self.activation_threshold:
                    self.rollback_activated = True
                    self.activation_timestep = self.num_timesteps
                    if self.verbose > 0:
                        print(f"🔓 ACTIVATION THRESHOLD REACHED!")
                        print(f"   Threshold: {self.activation_threshold:.2f}, Current: {mean_reward:.2f}")
                        print(f"   Rollback protection is now ACTIVE")
                else:
                    if self.verbose > 0:
                        print(f"⏳ Activation threshold not yet reached")
                        print(f"   Threshold: {self.activation_threshold:.2f}, Current: {mean_reward:.2f}")
                        print(f"   Rollback protection: INACTIVE")
            
            # Check if this is a new best
            if mean_reward > self.best_mean_reward:
                improvement = mean_reward - self.best_mean_reward
                
                if self.verbose > 0:
                    print(f"⭐ NEW BEST MODEL! (+{improvement:.2f} improvement)")
                
                self.best_mean_reward = mean_reward
                self.best_timestep = self.num_timesteps
                self.consecutive_declines = 0  # Reset decline counter
                
                self._save_best_checkpoint(mean_reward, std_reward, mean_length, episode_rewards)
                
                if self.verbose > 0:
                    print(f"   Consecutive declines reset to 0")
            else:
                # Performance did not improve
                deficit = self.best_mean_reward - mean_reward
                
                # Only count declines and potentially rollback if activated
                if self.rollback_activated:
                    self.consecutive_declines += 1
                    
                    if self.verbose > 0:
                        print(f"📉 No improvement (Best: {self.best_mean_reward:.2f}, deficit: {deficit:.2f})")
                        print(f"   Consecutive declines: {self.consecutive_declines}/{self.rollback_patience}")
                    
                    # Check if rollback is needed
                    if self.consecutive_declines >= self.rollback_patience:
                        self._perform_rollback()
                else:
                    # Rollback not yet activated - just report the decline without counting
                    if self.verbose > 0:
                        print(f"📉 No improvement (Best: {self.best_mean_reward:.2f}, deficit: {deficit:.2f})")
                        print(f"   ⏳ Rollback protection not yet active (threshold: {self.activation_threshold:.2f})")
            
            # Log rollback stats
            self.logger.record("rollback/consecutive_declines", self.consecutive_declines)
            self.logger.record("rollback/total_rollbacks", self.total_rollbacks)
            self.logger.record("rollback/lr_multiplier", self.current_lr_multiplier)
            self.logger.record("rollback/best_reward", self.best_mean_reward)
            self.logger.record("rollback/activated", float(self.rollback_activated))
            
            if self.verbose > 0:
                print(f"{'='*60}\n")
        
        return True

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of training with rollback info."""
        return {
            'timesteps': self.evaluations_timesteps,
            'mean_rewards': self.evaluations_results,
            'mean_lengths': self.evaluations_length,
            'std_rewards': [float(s) for s in self.evaluations_std],
            'n_eval_episodes': int(self.n_eval_episodes),
            'best_mean_reward': float(self.best_mean_reward),
            'best_timestep': int(self.best_timestep),
            'n_evaluations': len(self.evaluations_timesteps),
            # Rollback-specific info
            'total_rollbacks': int(self.total_rollbacks),
            'final_lr_multiplier': float(self.current_lr_multiplier),
            'rollback_events': self.rollback_events,
            'rollback_patience': int(self.rollback_patience),
            'lr_decay_factor': float(self.lr_decay_factor),
            'min_lr_multiplier': float(self.min_lr_multiplier),
            # Activation threshold info
            'activation_threshold': float(self.activation_threshold) if self.activation_threshold is not None else None,
            'rollback_activated': bool(self.rollback_activated),
            'activation_timestep': int(self.activation_timestep) if self.activation_timestep is not None else None,
        }
