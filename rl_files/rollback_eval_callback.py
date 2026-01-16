"""
rollback_eval_callback_topk.py

Enhanced Rollback Callback with Top-K Checkpoint Management

Only keeps the top K checkpoints by evaluation reward to save disk space.
Includes all fixes from rollback_eval_callback_fixed.py plus smart checkpoint pruning.

Storage savings example (10M steps, 200k eval freq, K=5):
- Without Top-K: 50 checkpoints × 6.7 MB = 335 MB
- With Top-K=5:   5 checkpoints × 6.7 MB =  34 MB (90% reduction)
"""

import json
import time
import heapq
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass, field

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


@dataclass(order=True)
class CheckpointInfo:
    """Stores info about a checkpoint for the priority queue."""
    reward: float  # Used for comparison (heap ordering)
    timestep: int = field(compare=False)
    policy_path: Path = field(compare=False)
    optimizer_path: Path = field(compare=False)
    vecnorm_path: Optional[Path] = field(compare=False, default=None)


class RollbackEvaluationCallbackTopK(BaseCallback):
    """
    Evaluation callback with automatic rollback and Top-K checkpoint management.
    
    Key features:
    - Only keeps the top K checkpoints by evaluation reward
    - Properly saves and restores optimizer state
    - Freezes VecNormalize after rollback
    - Validates rollback success
    
    Args:
        eval_env: Evaluation environment
        eval_freq: Steps between evaluations
        n_eval_episodes: Episodes per evaluation
        log_dir: Directory for checkpoints and logs
        max_checkpoints: Maximum number of checkpoints to keep (default: 5)
        rollback_patience: Consecutive declines before rollback (default: 3)
        lr_decay_factor: LR multiplier on rollback (default: 0.5)
        min_lr: Minimum learning rate floor (default: 1e-6)
        min_lr_multiplier: Minimum LR multiplier floor (default: 0.1)
        max_rollbacks: Maximum rollbacks allowed (default: 10)
        activation_threshold: Reward threshold to activate rollback protection
        freeze_normalize_steps: Steps to freeze VecNormalize after rollback
        validate_rollback: Run evaluation immediately after rollback
        verbose: Verbosity level
        deterministic: Use deterministic actions during eval
    """
    
    def __init__(
        self,
        eval_env: DummyVecEnv,
        eval_freq: int = 200000,
        n_eval_episodes: int = 15,
        log_dir: Path = None,
        max_checkpoints: int = 5,  # NEW: Only keep top K
        rollback_patience: int = 3,
        lr_decay_factor: float = 0.5,
        min_lr: float = 1e-6,
        min_lr_multiplier: float = 0.4,
        max_rollbacks: int = 10,
        activation_threshold: Optional[float] = None,
        skip_initial_eval: bool = False,
        freeze_normalize_steps: int = 50000,
        validate_rollback: bool = True,
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
        
        # Top-K checkpoint management
        self.max_checkpoints = max_checkpoints
        self._checkpoint_heap: List[CheckpointInfo] = []  # Min-heap by reward
        
        # Rollback configuration
        self.rollback_patience = rollback_patience
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.min_lr_multiplier = min_lr_multiplier
        self.max_rollbacks = max_rollbacks
        self.activation_threshold = activation_threshold
        
        # Post-rollback settings
        self.freeze_normalize_steps = freeze_normalize_steps
        self.validate_rollback = validate_rollback
        self._normalize_frozen_until = 0
        self._original_training_mode = True
        
        # Tracking state
        self.best_mean_reward = -np.inf
        self.best_timestep = 0
        self.consecutive_declines = 0
        self.total_rollbacks = 0
        self.current_lr_multiplier = 1.0
        
        # Activation tracking
        self.rollback_activated = False if activation_threshold is not None else True
        self.activation_timestep = None
        
        # Evaluation history
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []
        self.evaluations_std = []
        
        # Rollback history
        self.rollback_events = []
        
        # Create checkpoint directories
        self.eval_checkpoint_dir = self.log_dir / "eval_checkpoints"
        self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_checkpoint_dir = self.log_dir / "best_checkpoint"
        self.best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_eval_step = 0
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
            self._base_lr_schedule = self.model.learning_rate
        
        if callable(self._base_lr_schedule):
            new_schedule = self._create_scaled_lr_schedule(self._base_lr_schedule, new_multiplier)
            self.model.learning_rate = new_schedule
            self.model.lr_schedule = new_schedule
            
            current_lr = new_schedule(1.0 - (self.num_timesteps / self.model._total_timesteps))
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            new_lr = max(self._base_lr_schedule * new_multiplier, self.min_lr)
            self.model.learning_rate = new_lr
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        if self.verbose > 0:
            actual_lr = self._get_current_lr()
            print(f"   📉 Learning rate updated: multiplier={new_multiplier:.4f}, current_lr={actual_lr:.2e}")

    def _save_eval_checkpoint(self, mean_reward: float) -> CheckpointInfo:
        """
        Save a checkpoint and manage the top-K list.
        
        Returns the CheckpointInfo for the saved checkpoint.
        """
        timestamp = self.num_timesteps
        
        # Define paths
        policy_path = self.eval_checkpoint_dir / f"policy_{timestamp}_steps.pt"
        optimizer_path = self.eval_checkpoint_dir / f"optimizer_{timestamp}_steps.pt"
        vecnorm_path = self.eval_checkpoint_dir / f"vecnormalize_{timestamp}_steps.pkl"
        
        # Check if this checkpoint should be kept
        should_save = False
        checkpoint_to_remove = None
        
        if len(self._checkpoint_heap) < self.max_checkpoints:
            # Haven't reached max yet, always save
            should_save = True
        else:
            # Check if better than worst in top-K
            worst_checkpoint = self._checkpoint_heap[0]  # Min-heap, so index 0 is worst
            if mean_reward > worst_checkpoint.reward:
                should_save = True
                checkpoint_to_remove = worst_checkpoint
        
        if not should_save:
            if self.verbose > 0:
                print(f"   ⏭️  Checkpoint not in top-{self.max_checkpoints}, skipping save")
                print(f"      (reward {mean_reward:.2f} < worst kept {self._checkpoint_heap[0].reward:.2f})")
            return None
        
        # Save the checkpoint files
        th.save({
            'policy_state_dict': self.model.policy.state_dict(),
            'timestep': timestamp,
            'mean_reward': mean_reward,
        }, policy_path.as_posix())
        
        th.save({
            'optimizer_state_dict': self.model.policy.optimizer.state_dict(),
            'timestep': timestamp,
        }, optimizer_path.as_posix())
        
        vec_env = self.model.get_env()
        actual_vecnorm_path = None
        if isinstance(vec_env, VecNormalize):
            vec_env.save(vecnorm_path.as_posix())
            actual_vecnorm_path = vecnorm_path
        
        # Create checkpoint info
        new_checkpoint = CheckpointInfo(
            reward=mean_reward,
            timestep=timestamp,
            policy_path=policy_path,
            optimizer_path=optimizer_path,
            vecnorm_path=actual_vecnorm_path,
        )
        
        # Remove worst checkpoint if needed
        if checkpoint_to_remove is not None:
            heapq.heappop(self._checkpoint_heap)
            self._delete_checkpoint(checkpoint_to_remove)
            if self.verbose > 0:
                print(f"   🗑️  Removed checkpoint from step {checkpoint_to_remove.timestep:,} "
                      f"(reward {checkpoint_to_remove.reward:.2f})")
        
        # Add new checkpoint to heap
        heapq.heappush(self._checkpoint_heap, new_checkpoint)
        
        if self.verbose > 0:
            print(f"   💾 Checkpoint saved (top-{self.max_checkpoints}: "
                  f"[{', '.join(f'{c.reward:.1f}' for c in sorted(self._checkpoint_heap, reverse=True))}])")
        
        return new_checkpoint

    def _delete_checkpoint(self, checkpoint: CheckpointInfo):
        """Delete checkpoint files from disk."""
        try:
            if checkpoint.policy_path.exists():
                checkpoint.policy_path.unlink()
            if checkpoint.optimizer_path.exists():
                checkpoint.optimizer_path.unlink()
            if checkpoint.vecnorm_path and checkpoint.vecnorm_path.exists():
                checkpoint.vecnorm_path.unlink()
        except Exception as e:
            if self.verbose > 0:
                print(f"   ⚠️  Error deleting checkpoint files: {e}")

    def _save_best_checkpoint(self, mean_reward: float, std_reward: float, 
                              mean_length: float, episode_rewards: List[float]):
        """Save the current model as the best checkpoint (always kept, separate from top-K)."""
        timestamp = time.time()
        
        # Save policy weights
        policy_path = self.best_checkpoint_dir / "best_policy.pt"
        th.save({
            'policy_state_dict': self.model.policy.state_dict(),
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
            'timestamp': timestamp,
        }, policy_path.as_posix())
        
        # Save optimizer state
        optimizer_path = self.best_checkpoint_dir / "best_optimizer.pt"
        th.save({
            'optimizer_state_dict': self.model.policy.optimizer.state_dict(),
            'timestep': self.num_timesteps,
            'timestamp': timestamp,
        }, optimizer_path.as_posix())
        
        # Save VecNormalize statistics
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            norm_path = self.best_checkpoint_dir / "best_vecnormalize.pkl"
            vec_env.save(norm_path.as_posix())
        
        # Try full model save (optional)
        model_path = self.best_checkpoint_dir / "best_model"
        try:
            self.model.save(model_path.as_posix())
        except Exception as e:
            if self.verbose > 0:
                print(f"   ⚠️  Full model save failed: {str(e)[:50]}...")
        
        # Save metadata
        best_info = {
            'timestep': int(self.num_timesteps),
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_length': float(mean_length),
            'episode_rewards': [float(r) for r in episode_rewards],
            'lr_multiplier': float(self.current_lr_multiplier),
            'total_rollbacks_at_save': int(self.total_rollbacks),
            'timestamp': timestamp,
        }
        
        info_path = self.best_checkpoint_dir / "best_model_info.json"
        with open(info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        # Also save to main log_dir
        th.save({
            'policy_state_dict': self.model.policy.state_dict(),
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
            'timestamp': timestamp,
        }, (self.log_dir / "best_policy.pt").as_posix())
        
        th.save({
            'optimizer_state_dict': self.model.policy.optimizer.state_dict(),
            'timestep': self.num_timesteps,
            'timestamp': timestamp,
        }, (self.log_dir / "best_optimizer.pt").as_posix())
        
        if isinstance(vec_env, VecNormalize):
            vec_env.save((self.log_dir / "best_vecnormalize.pkl").as_posix())

    def _perform_rollback(self) -> bool:
        """Roll back to the best checkpoint and reduce learning rate."""
        if self.total_rollbacks >= self.max_rollbacks:
            if self.verbose > 0:
                print(f"\n⚠️  Maximum rollbacks ({self.max_rollbacks}) reached. Continuing without rollback.")
            return False
        
        policy_path = self.best_checkpoint_dir / "best_policy.pt"
        optimizer_path = self.best_checkpoint_dir / "best_optimizer.pt"
        norm_path = self.best_checkpoint_dir / "best_vecnormalize.pkl"
        
        if not policy_path.exists():
            if self.verbose > 0:
                print(f"\n⚠️  No best checkpoint found. Cannot rollback.")
            return False
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔄 ROLLBACK TRIGGERED")
            print(f"   Consecutive declines: {self.consecutive_declines}")
            print(f"   Rolling back to timestep {self.best_timestep} (reward: {self.best_mean_reward:.2f})")
            print(f"{'='*60}")
        
        # Restore policy weights
        try:
            checkpoint = th.load(policy_path.as_posix(), map_location=self.model.device)
            self.model.policy.load_state_dict(checkpoint['policy_state_dict'])
            if self.verbose > 0:
                print(f"   ✓ Policy weights restored")
        except Exception as e:
            if self.verbose > 0:
                print(f"   ❌ Policy restore failed: {e}")
            return False
        
        # Restore optimizer state
        if optimizer_path.exists():
            try:
                opt_checkpoint = th.load(optimizer_path.as_posix(), map_location=self.model.device)
                self.model.policy.optimizer.load_state_dict(opt_checkpoint['optimizer_state_dict'])
                if self.verbose > 0:
                    print(f"   ✓ Optimizer state restored (Adam momentum buffers)")
            except Exception as e:
                if self.verbose > 0:
                    print(f"   ⚠️  Optimizer restore failed: {e}")
                self._reset_optimizer()
        else:
            if self.verbose > 0:
                print(f"   ⚠️  No optimizer checkpoint found, resetting...")
            self._reset_optimizer()
        
        # Restore VecNormalize and freeze
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize) and norm_path.exists():
            try:
                saved_vec_env = VecNormalize.load(norm_path.as_posix(), vec_env.venv)
                vec_env.obs_rms = saved_vec_env.obs_rms
                vec_env.ret_rms = saved_vec_env.ret_rms
                
                if isinstance(self.eval_env, VecNormalize):
                    self.eval_env.obs_rms = saved_vec_env.obs_rms
                    self.eval_env.ret_rms = saved_vec_env.ret_rms
                
                # Freeze normalization
                self._original_training_mode = vec_env.training
                vec_env.training = False
                self._normalize_frozen_until = self.num_timesteps + self.freeze_normalize_steps
                
                if self.verbose > 0:
                    print(f"   ✓ VecNormalize restored and FROZEN for {self.freeze_normalize_steps:,} steps")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"   ⚠️  Error restoring VecNormalize: {e}")
        
        # Reduce learning rate
        new_multiplier = max(
            self.current_lr_multiplier * self.lr_decay_factor,
            self.min_lr_multiplier
        )
        old_lr = self._get_current_lr()
        self._update_learning_rate(new_multiplier)
        new_lr = self._get_current_lr()
        
        # Record rollback event
        rollback_event = {
            'timestep': int(self.num_timesteps),
            'rollback_to_timestep': int(self.best_timestep),
            'consecutive_declines': int(self.consecutive_declines),
            'old_lr': float(old_lr),
            'new_lr': float(new_lr),
            'best_reward_at_rollback': float(self.best_mean_reward),
        }
        self.rollback_events.append(rollback_event)
        
        # Save rollback history
        rollback_history_path = self.log_dir / "rollback_history.json"
        with open(rollback_history_path, 'w') as f:
            json.dump(self.rollback_events, f, indent=2)
        
        self.consecutive_declines = 0
        self.total_rollbacks += 1
        
        if self.verbose > 0:
            print(f"   ✓ Learning rate: {old_lr:.2e} → {new_lr:.2e}")
            print(f"   Total rollbacks: {self.total_rollbacks}/{self.max_rollbacks}")
        
        # Validate rollback
        if self.validate_rollback:
            print(f"\n   🔍 Validating rollback...")
            mean_reward, std_reward, _, _ = self._evaluate_policy()
            print(f"   📊 Post-rollback eval: {mean_reward:.2f} ± {std_reward:.2f}")
            
            if mean_reward < self.best_mean_reward * 0.5:
                print(f"   ⚠️  WARNING: Post-rollback performance degraded!")
            else:
                print(f"   ✓ Rollback validation passed")
        
        print(f"{'='*60}\n")
        return True

    def _reset_optimizer(self):
        """Reset optimizer to fresh state."""
        optimizer = self.model.policy.optimizer
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()
                if 'step' in state:
                    state['step'] = 0

    def _evaluate_policy(self) -> Tuple[float, float, float, List[float]]:
        """Run evaluation episodes."""
        episode_rewards = []
        episode_lengths = []
        
        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        episodes_completed = 0
        
        while episodes_completed < self.n_eval_episodes:
            try:
                self.eval_env.env_method("set_global_step", int(self.num_timesteps))
            except:
                pass
            
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            
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
        
        return (
            float(np.mean(episode_rewards)),
            float(np.std(episode_rewards)),
            float(np.mean(episode_lengths)),
            episode_rewards
        )

    def _on_training_start(self) -> None:
        """Initial evaluation at timestep 0."""
        self._base_lr_schedule = self.model.learning_rate
        
        if self.skip_initial_eval:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"⏭️  SKIPPING initial evaluation")
                print(f"{'='*60}\n")
            return
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔍 INITIAL EVALUATION at timestep 0")
            print(f"{'='*60}")
        
        eval_start_time = time.time()
        mean_reward, std_reward, mean_length, episode_rewards = self._evaluate_policy()
        eval_time = time.time() - eval_start_time
        
        self.evaluations_timesteps.append(0)
        self.evaluations_results.append(mean_reward)
        self.evaluations_length.append(mean_length)
        self.evaluations_std.append(std_reward)
        
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        
        if self.verbose > 0:
            print(f" Results: {mean_reward:.2f} ± {std_reward:.2f}, Time: {eval_time:.1f}s")
        
        self.best_mean_reward = mean_reward
        self.best_timestep = 0
        self._save_best_checkpoint(mean_reward, std_reward, mean_length, episode_rewards)
        self._save_eval_checkpoint(mean_reward)
        
        if self.verbose > 0:
            print(f"⭐ INITIAL BEST MODEL saved")
            print(f"{'='*60}\n")

    def _on_step(self) -> bool:
        """Called at each step."""
        
        # Check if we should unfreeze VecNormalize
        if self._normalize_frozen_until > 0 and self.num_timesteps >= self._normalize_frozen_until:
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                vec_env.training = self._original_training_mode
                if self.verbose > 0:
                    print(f"   ✓ VecNormalize UNFROZEN at step {self.num_timesteps:,}")
            self._normalize_frozen_until = 0
        
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"🔍 EVALUATION at {self.num_timesteps:,} steps")
                print(f"{'='*60}")
            
            # Sync normalization stats
            try:
                train_env = self.model.get_env()
                if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                    self.eval_env.obs_rms = train_env.obs_rms
                    self.eval_env.ret_rms = train_env.ret_rms
            except:
                pass
            
            # Run evaluation
            eval_start_time = time.time()
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
                print(f" Results: {mean_reward:.2f} ± {std_reward:.2f}, Time: {eval_time:.1f}s")
            
            # Save to top-K checkpoints (may skip if not good enough)
            self._save_eval_checkpoint(mean_reward)
            
            # Check activation threshold
            if not self.rollback_activated and self.activation_threshold is not None:
                if mean_reward >= self.activation_threshold:
                    self.rollback_activated = True
                    self.activation_timestep = self.num_timesteps
                    if self.verbose > 0:
                        print(f"🔓 ACTIVATION THRESHOLD REACHED! Rollback protection ACTIVE")
            
            # Check if this is a new best
            if mean_reward > self.best_mean_reward:
                improvement = mean_reward - self.best_mean_reward
                
                if self.verbose > 0:
                    print(f"⭐ NEW BEST MODEL! (+{improvement:.2f})")
                
                self.best_mean_reward = mean_reward
                self.best_timestep = self.num_timesteps
                self.consecutive_declines = 0
                
                self._save_best_checkpoint(mean_reward, std_reward, mean_length, episode_rewards)
            else:
                deficit = self.best_mean_reward - mean_reward
                
                if self.rollback_activated:
                    self.consecutive_declines += 1
                    
                    if self.verbose > 0:
                        print(f"📉 No improvement (Best: {self.best_mean_reward:.2f}, deficit: {deficit:.2f})")
                        print(f"   Consecutive declines: {self.consecutive_declines}/{self.rollback_patience}")
                    
                    if self.consecutive_declines >= self.rollback_patience:
                        self._perform_rollback()
                else:
                    if self.verbose > 0:
                        print(f"📉 No improvement, rollback not yet active")
            
            # Log rollback stats
            self.logger.record("rollback/consecutive_declines", self.consecutive_declines)
            self.logger.record("rollback/total_rollbacks", self.total_rollbacks)
            self.logger.record("rollback/lr_multiplier", self.current_lr_multiplier)
            self.logger.record("rollback/best_reward", self.best_mean_reward)
            self.logger.record("rollback/n_checkpoints", len(self._checkpoint_heap))
            
            if self.verbose > 0:
                print(f"{'='*60}\n")
        
        return True

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        return {
            'timesteps': self.evaluations_timesteps,
            'mean_rewards': self.evaluations_results,
            'mean_lengths': self.evaluations_length,
            'std_rewards': [float(s) for s in self.evaluations_std],
            'n_eval_episodes': int(self.n_eval_episodes),
            'best_mean_reward': float(self.best_mean_reward),
            'best_timestep': int(self.best_timestep),
            'n_evaluations': len(self.evaluations_timesteps),
            'total_rollbacks': int(self.total_rollbacks),
            'final_lr_multiplier': float(self.current_lr_multiplier),
            'rollback_events': self.rollback_events,
            'max_checkpoints': int(self.max_checkpoints),
            'kept_checkpoints': [
                {'timestep': c.timestep, 'reward': c.reward}
                for c in sorted(self._checkpoint_heap, reverse=True)
            ],
        }

    def get_top_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of top-K checkpoints, sorted by reward (highest first)."""
        return sorted(self._checkpoint_heap, reverse=True)
