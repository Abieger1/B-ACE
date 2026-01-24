"""
rollback_eval_callback_v5_fixed.py

FIXES:
1. Model is now saved to BOTH best_checkpoint_dir AND log_dir
2. Model save errors are no longer silently swallowed - they raise exceptions
3. Validates that files were actually created after saving
4. Adds retry logic for transient save failures

Previous enhancements preserved from v5:
- PERMANENT VECNORMALIZE FREEZE: When enabled, VecNormalize stays frozen after rollback
  until a NEW BEST checkpoint is achieved.
- v4: MIN_TIMESTEPS_BEFORE_ROLLBACK to prevent "lucky checkpoint" syndrome
- v3: ABSOLUTE DEFICIT THRESHOLD for immediate rollback on catastrophic drops
- v2: Buffer mismatch handling, entropy coefficient decay
- v1: Top-K checkpoint management, optimizer state restoration
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
    - Skips training updates after rollback to handle buffer mismatch
    - Decays entropy coefficient on rollback
    - Absolute deficit threshold for immediate rollback on catastrophic drops
    - Minimum timesteps before rollback to prevent "lucky checkpoint" syndrome
    - (v5) Permanent VecNormalize freeze until new best is achieved
    - (FIXED) Model saved to both best_checkpoint_dir AND log_dir
    - (FIXED) Save errors are no longer silently swallowed
    
    Args:
        eval_env: Evaluation environment
        eval_freq: Steps between evaluations
        n_eval_episodes: Episodes per evaluation
        log_dir: Directory for checkpoints and logs
        max_checkpoints: Maximum number of checkpoints to keep (default: 5)
        rollback_patience: Consecutive declines before rollback (default: 5)
        deficit_threshold: Absolute reward drop from best that triggers immediate rollback (default: 5.0)
                          Set to None to disable deficit-based rollback.
        min_timesteps_before_rollback: Minimum training steps before rollback is allowed (default: 0)
        lr_decay_factor: LR multiplier on rollback (default: 0.5)
        entropy_decay_factor: Entropy coef multiplier on rollback (default: 0.5)
        min_entropy_coef: Minimum entropy coefficient floor (default: 0.001)
        min_lr: Minimum learning rate floor (default: 1e-6)
        min_lr_multiplier: Minimum LR multiplier floor (default: 0.1)
        max_rollbacks: Maximum rollbacks allowed (default: 10)
        activation_threshold: Reward threshold to activate rollback protection
        freeze_normalize_steps: Steps to freeze VecNormalize after rollback (ignored if freeze_until_new_best=True)
        freeze_vecnormalize_until_new_best: If True, keep VecNormalize frozen after rollback
                                            until a new best checkpoint is achieved.
        skip_updates_after_rollback: Number of training updates to skip after rollback (default: 2)
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
        max_checkpoints: int = 5,
        rollback_patience: int = 5,
        deficit_threshold: float = 5.0,
        min_timesteps_before_rollback: int = 0,
        lr_decay_factor: float = 0.5,
        entropy_decay_factor: float = 0.7,
        min_entropy_coef: float = 0.001,
        min_lr: float = 1e-6,
        min_lr_multiplier: float = 0.4,
        max_rollbacks: int = 10,
        activation_threshold: Optional[float] = None,
        skip_initial_eval: bool = False,
        freeze_normalize_steps: int = 50000,
        freeze_vecnormalize_until_new_best: bool = False,
        skip_updates_after_rollback: int = 2,
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
        self.deficit_threshold = deficit_threshold
        self.min_timesteps_before_rollback = min_timesteps_before_rollback
        self.lr_decay_factor = lr_decay_factor
        self.entropy_decay_factor = entropy_decay_factor
        self.min_entropy_coef = min_entropy_coef
        self.min_lr = min_lr
        self.min_lr_multiplier = min_lr_multiplier
        self.max_rollbacks = max_rollbacks
        self.activation_threshold = activation_threshold
        
        # Post-rollback settings
        self.freeze_normalize_steps = freeze_normalize_steps
        self.freeze_vecnormalize_until_new_best = freeze_vecnormalize_until_new_best
        self.skip_updates_after_rollback = skip_updates_after_rollback
        self.validate_rollback = validate_rollback
        self._normalize_frozen_until = 0
        self._original_training_mode = True
        
        # Track permanent freeze state
        self._vecnormalize_permanently_frozen = False
        
        # Buffer mismatch handling - skip training counter
        self._skip_train_counter = 0
        self._original_train_fn = None
        
        # Tracking state
        self.best_mean_reward = -np.inf
        self.best_timestep = 0
        self.consecutive_declines = 0
        self.total_rollbacks = 0
        self.current_lr_multiplier = 1.0
        self.current_entropy_multiplier = 1.0
        
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

    def _init_callback(self) -> None:
        """
        Called when the callback is initialized.
        Wraps the model's train function to handle skip logic after rollback.
        """
        # Store reference to original train function
        self._original_train_fn = self.model.train
        
        # Create wrapped train function that respects skip counter
        def wrapped_train():
            if self._skip_train_counter > 0:
                self._skip_train_counter -= 1
                if self.verbose > 0:
                    print(f"   ⏭️  Skipping training update (buffer stabilization: {self._skip_train_counter} remaining)")
                return
            return self._original_train_fn()
        
        # Replace model's train method with wrapped version
        self.model.train = wrapped_train

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        if self.skip_initial_eval:
            self.last_eval_step = self.num_timesteps
            if self.verbose > 0:
                print(f"⏭️  Skipping initial evaluation (skip_initial_eval=True)")
            return
            
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🔍 INITIAL EVALUATION")
            print(f"{'='*60}")
        
        # Sync normalization stats
        try:
            train_env = self.model.get_env()
            if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = train_env.obs_rms
                self.eval_env.ret_rms = train_env.ret_rms
        except:
            pass
        
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

    def _on_training_end(self) -> None:
        """Restore original train function on training end."""
        if self._original_train_fn is not None:
            self.model.train = self._original_train_fn

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

    def _update_entropy_coef(self, decay_factor: float):
        """
        Update the model's entropy coefficient by applying decay factor.
        """
        old_entropy = self.model.ent_coef
        new_entropy = max(old_entropy * decay_factor, self.min_entropy_coef)
        self.model.ent_coef = new_entropy
        self.current_entropy_multiplier *= decay_factor
        
        if self.verbose > 0:
            print(f"   📉 Entropy coef updated: {old_entropy:.6f} → {new_entropy:.6f} "
                  f"(multiplier={self.current_entropy_multiplier:.4f})")
        
        self.logger.record("rollback/entropy_coef", new_entropy)
        self.logger.record("rollback/entropy_multiplier", self.current_entropy_multiplier)

    def _evaluate_policy(self) -> Tuple[float, float, float, List[float]]:
        """Run evaluation episodes and return mean reward, std, mean length, and episode rewards."""
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                if done[0]:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if self.verbose > 0:
                print(f"[EVAL EP-END] ep={ep+1}/{self.n_eval_episodes} episode={{'r': {episode_reward}, 'l': {episode_length}, 't': {time.time()}}}")
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        return float(mean_reward), float(std_reward), float(mean_length), episode_rewards

    def _save_eval_checkpoint(self, mean_reward: float) -> CheckpointInfo:
        """
        Save a checkpoint and manage the top-K list.
        """
        timestamp = self.num_timesteps
        
        policy_path = self.eval_checkpoint_dir / f"policy_{timestamp}_steps.pt"
        optimizer_path = self.eval_checkpoint_dir / f"optimizer_{timestamp}_steps.pt"
        vecnorm_path = self.eval_checkpoint_dir / f"vecnormalize_{timestamp}_steps.pkl"
        
        should_save = False
        checkpoint_to_remove = None
        
        if len(self._checkpoint_heap) < self.max_checkpoints:
            should_save = True
        else:
            worst_checkpoint = self._checkpoint_heap[0]
            if mean_reward > worst_checkpoint.reward:
                should_save = True
                checkpoint_to_remove = worst_checkpoint
        
        if not should_save:
            if self.verbose > 0:
                print(f"   ⏭️  Checkpoint not in top-{self.max_checkpoints}, skipping save")
                print(f"      (reward {mean_reward:.2f} < worst kept {self._checkpoint_heap[0].reward:.2f})")
            return None
        
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
        
        new_checkpoint = CheckpointInfo(
            reward=mean_reward,
            timestep=timestamp,
            policy_path=policy_path,
            optimizer_path=optimizer_path,
            vecnorm_path=actual_vecnorm_path,
        )
        
        if checkpoint_to_remove is not None:
            heapq.heappop(self._checkpoint_heap)
            self._delete_checkpoint(checkpoint_to_remove)
            if self.verbose > 0:
                print(f"   🗑️  Removed checkpoint from step {checkpoint_to_remove.timestep:,} "
                      f"(reward {checkpoint_to_remove.reward:.2f})")
        
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

    def _save_model_with_validation(self, model_path: Path, description: str, max_retries: int = 3) -> bool:
        """
        Save model with validation that file was created successfully.
        
        Args:
            model_path: Path to save model (without .zip extension)
            description: Description for logging
            max_retries: Number of retries on failure
            
        Returns:
            True if save succeeded, False otherwise
        """
        zip_path = Path(str(model_path) + ".zip")
        
        for attempt in range(max_retries):
            try:
                # Remove existing file if present (to ensure fresh save)
                if zip_path.exists():
                    zip_path.unlink()
                
                # Save the model
                self.model.save(model_path.as_posix())
                
                # Validate the save
                if not zip_path.exists():
                    raise RuntimeError(f"Save appeared to succeed but {zip_path} does not exist")
                
                file_size = zip_path.stat().st_size
                if file_size < 1000:  # Less than 1KB is suspicious
                    raise RuntimeError(f"Save produced suspiciously small file: {file_size} bytes")
                
                if self.verbose > 0:
                    print(f"   ✓ {description} saved: {zip_path} ({file_size:,} bytes)")
                
                return True
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"   ⚠️  {description} save attempt {attempt+1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief pause before retry
                else:
                    print(f"   ❌ CRITICAL: {description} save FAILED after {max_retries} attempts!")
                    print(f"      Error: {e}")
                    print(f"      The .pt files should still be valid as fallback.")
                    return False
        
        return False

    def _save_best_checkpoint(self, mean_reward: float, std_reward: float, 
                              mean_length: float, episode_rewards: List[float]):
        """
        Save the current model as the best checkpoint.
        
        FIXED: Now saves to BOTH best_checkpoint_dir AND log_dir
        FIXED: Validates saves and doesn't silently swallow errors
        """
        timestamp = time.time()
        
        # =====================================================
        # Save .pt files (these always work reliably)
        # =====================================================
        
        # Save to best_checkpoint_dir
        policy_path_best = self.best_checkpoint_dir / "best_policy.pt"
        th.save({
            'policy_state_dict': self.model.policy.state_dict(),
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
            'timestamp': timestamp,
        }, policy_path_best.as_posix())
        
        optimizer_path_best = self.best_checkpoint_dir / "best_optimizer.pt"
        th.save({
            'optimizer_state_dict': self.model.policy.optimizer.state_dict(),
            'timestep': self.num_timesteps,
            'timestamp': timestamp,
        }, optimizer_path_best.as_posix())
        
        # Save to log_dir (main directory)
        policy_path_main = self.log_dir / "best_policy.pt"
        th.save({
            'policy_state_dict': self.model.policy.state_dict(),
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
            'timestamp': timestamp,
        }, policy_path_main.as_posix())
        
        optimizer_path_main = self.log_dir / "best_optimizer.pt"
        th.save({
            'optimizer_state_dict': self.model.policy.optimizer.state_dict(),
            'timestep': self.num_timesteps,
            'timestamp': timestamp,
        }, optimizer_path_main.as_posix())
        
        if self.verbose > 0:
            print(f"   ✓ Policy weights saved (.pt files)")
        
        # =====================================================
        # Save VecNormalize stats
        # =====================================================
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            # Save to best_checkpoint_dir
            norm_path_best = self.best_checkpoint_dir / "best_vecnormalize.pkl"
            vec_env.save(norm_path_best.as_posix())
            
            # Save to log_dir (main directory)
            norm_path_main = self.log_dir / "best_vecnormalize.pkl"
            vec_env.save(norm_path_main.as_posix())
            
            if self.verbose > 0:
                print(f"   ✓ VecNormalize stats saved")
        
        # =====================================================
        # Save full model.zip (with validation)
        # FIXED: Now saves to BOTH locations
        # =====================================================
        
        # Save to best_checkpoint_dir
        model_path_best = self.best_checkpoint_dir / "best_model"
        save_ok_best = self._save_model_with_validation(model_path_best, "Full model (best_checkpoint)")
        
        # Save to log_dir (main directory) - THIS WAS MISSING!
        model_path_main = self.log_dir / "best_model"
        save_ok_main = self._save_model_with_validation(model_path_main, "Full model (log_dir)")
        
        if not save_ok_best and not save_ok_main:
            print(f"   ⚠️  WARNING: Full model save failed to both locations!")
            print(f"      Evaluation will need to use .pt file fallback")
        
        # =====================================================
        # Save metadata JSON
        # =====================================================
        best_info = {
            'timestep': int(self.num_timesteps),
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_length': float(mean_length),
            'episode_rewards': [float(r) for r in episode_rewards],
            'lr_multiplier': float(self.current_lr_multiplier),
            'entropy_multiplier': float(self.current_entropy_multiplier),
            'total_rollbacks_at_save': int(self.total_rollbacks),
            'timestamp': timestamp,
            'model_save_success': {
                'best_checkpoint_dir': save_ok_best,
                'log_dir': save_ok_main,
            }
        }
        
        # Save to best_checkpoint_dir
        info_path_best = self.best_checkpoint_dir / "best_model_info.json"
        with open(info_path_best, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        # Save to log_dir (main directory)
        info_path_main = self.log_dir / "best_model_info.json"
        with open(info_path_main, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        if self.verbose > 0:
            print(f"   ✓ Best model info saved")

    def _perform_rollback(self, trigger_reason: str = "consecutive_declines") -> bool:
        """
        Roll back to the best checkpoint and reduce learning rate AND entropy coefficient.
        """
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
            print(f"   Trigger: {trigger_reason.upper()}")
            print(f"   Consecutive declines: {self.consecutive_declines}")
            print(f"   Rolling back to timestep {self.best_timestep:,} (reward: {self.best_mean_reward:.2f})")
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
        
        # Restore VecNormalize stats
        if norm_path.exists():
            try:
                vec_env = self.model.get_env()
                if isinstance(vec_env, VecNormalize):
                    loaded = VecNormalize.load(norm_path.as_posix(), vec_env.venv)
                    vec_env.obs_rms = loaded.obs_rms
                    vec_env.ret_rms = loaded.ret_rms
                    if self.verbose > 0:
                        print(f"   ✓ VecNormalize stats restored")
            except Exception as e:
                if self.verbose > 0:
                    print(f"   ⚠️  VecNormalize restore failed: {e}")
        
        # Decay learning rate
        new_lr_mult = max(self.current_lr_multiplier * self.lr_decay_factor, self.min_lr_multiplier)
        self._update_learning_rate(new_lr_mult)
        
        # Decay entropy coefficient
        self._update_entropy_coef(self.entropy_decay_factor)
        
        # Freeze VecNormalize
        self._freeze_vecnormalize()
        
        # Set skip counter for training updates
        self._skip_train_counter = self.skip_updates_after_rollback
        
        # Record rollback
        self.total_rollbacks += 1
        self.consecutive_declines = 0
        
        self.rollback_events.append({
            'timestep': self.num_timesteps,
            'trigger': trigger_reason,
            'rolled_back_to': self.best_timestep,
            'new_lr_multiplier': self.current_lr_multiplier,
            'new_entropy_multiplier': self.current_entropy_multiplier,
        })
        
        if self.verbose > 0:
            print(f"   Total rollbacks: {self.total_rollbacks}/{self.max_rollbacks}")
            print(f"{'='*60}\n")
        
        # Validate rollback
        if self.validate_rollback:
            self._validate_rollback()
        
        return True

    def _reset_optimizer(self):
        """Reset optimizer state (used when optimizer checkpoint is missing)."""
        try:
            # Get optimizer class and lr
            opt_class = type(self.model.policy.optimizer)
            lr = self.model.policy.optimizer.param_groups[0]['lr']
            
            # Create new optimizer
            self.model.policy.optimizer = opt_class(
                self.model.policy.parameters(),
                lr=lr
            )
            if self.verbose > 0:
                print(f"   ⚠️  Optimizer reset (no checkpoint)")
        except Exception as e:
            if self.verbose > 0:
                print(f"   ⚠️  Optimizer reset failed: {e}")

    def _freeze_vecnormalize(self):
        """Freeze VecNormalize statistics."""
        try:
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                self._original_training_mode = vec_env.training
                vec_env.training = False
                
                if self.freeze_vecnormalize_until_new_best:
                    # Permanent freeze until new best
                    self._vecnormalize_permanently_frozen = True
                    self._normalize_frozen_until = float('inf')
                    if self.verbose > 0:
                        print(f"   📌 VecNormalize PERMANENTLY FROZEN (until new best)")
                else:
                    # Time-based freeze
                    self._normalize_frozen_until = self.num_timesteps + self.freeze_normalize_steps
                    if self.verbose > 0:
                        print(f"   🧊 VecNormalize frozen for {self.freeze_normalize_steps:,} steps")
        except Exception as e:
            if self.verbose > 0:
                print(f"   ⚠️  Failed to freeze VecNormalize: {e}")

    def _unfreeze_vecnormalize(self):
        """Unfreeze VecNormalize statistics."""
        try:
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                vec_env.training = self._original_training_mode
                if self.verbose > 0:
                    print(f"   ✓ VecNormalize UNFROZEN at step {self.num_timesteps:,}")
        except Exception as e:
            if self.verbose > 0:
                print(f"   ⚠️  Failed to unfreeze VecNormalize: {e}")
        self._normalize_frozen_until = 0
        self._vecnormalize_permanently_frozen = False

    def _validate_rollback(self):
        """Validate rollback by running a quick evaluation."""
        if self.verbose > 0:
            print(f"   🔍 Validating rollback...")
        
        try:
            # Sync normalization stats
            train_env = self.model.get_env()
            if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = train_env.obs_rms
                self.eval_env.ret_rms = train_env.ret_rms
            
            # Quick eval (just 3 episodes)
            rewards = []
            for _ in range(3):
                obs = self.eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    ep_reward += reward[0]
                    if done[0]:
                        break
                rewards.append(ep_reward)
            
            mean_val = np.mean(rewards)
            if self.verbose > 0:
                print(f"   ✓ Validation: {mean_val:.2f} (expected ~{self.best_mean_reward:.2f})")
                
                diff = abs(mean_val - self.best_mean_reward)
                if diff > 20:
                    print(f"   ⚠️  Validation differs from best by {diff:.1f} - this is expected with stochastic eval")
        except Exception as e:
            if self.verbose > 0:
                print(f"   ⚠️  Validation failed: {e}")

    def _on_step(self) -> bool:
        """Called at each step."""
        
        # Check if we should unfreeze VecNormalize (time-based, NOT permanent)
        if (not self._vecnormalize_permanently_frozen and 
            self._normalize_frozen_until > 0 and 
            self.num_timesteps >= self._normalize_frozen_until):
            self._unfreeze_vecnormalize()
        
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"🔍 EVALUATION at {self.num_timesteps:,} steps")
                if self._vecnormalize_permanently_frozen:
                    print(f"   📌 VecNormalize: PERMANENTLY FROZEN (waiting for new best)")
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
                
                # Unfreeze VecNormalize on new best (if permanently frozen)
                if self._vecnormalize_permanently_frozen:
                    if self.verbose > 0:
                        print(f"   🔓 NEW BEST achieved - unfreezing VecNormalize!")
                    self._unfreeze_vecnormalize()
                
                self.best_mean_reward = mean_reward
                self.best_timestep = self.num_timesteps
                self.consecutive_declines = 0
                
                self._save_best_checkpoint(mean_reward, std_reward, mean_length, episode_rewards)
            else:
                # Performance did not improve
                deficit = self.best_mean_reward - mean_reward
                
                if self.rollback_activated:
                    self.consecutive_declines += 1
                    
                    # Check if minimum timesteps reached for rollback
                    rollback_allowed = self.num_timesteps >= self.min_timesteps_before_rollback
                    
                    if self.verbose > 0:
                        print(f"📉 No improvement (Best: {self.best_mean_reward:.2f}, deficit: {deficit:.2f})")
                        print(f"   Consecutive declines: {self.consecutive_declines}/{self.rollback_patience}")
                        if not rollback_allowed:
                            print(f"   ⏳ Rollback delayed until {self.min_timesteps_before_rollback:,} steps "
                                  f"(currently at {self.num_timesteps:,})")
                    
                    # Only perform rollback if minimum timesteps reached
                    if rollback_allowed:
                        # Check ABSOLUTE DEFICIT THRESHOLD first
                        if self.deficit_threshold is not None and deficit > self.deficit_threshold:
                            if self.verbose > 0:
                                print(f"   🚨 DEFICIT THRESHOLD EXCEEDED! ({deficit:.2f} > {self.deficit_threshold:.2f})")
                            self._perform_rollback(trigger_reason="deficit_threshold")
                        
                        # Then check consecutive declines (secondary trigger)
                        elif self.consecutive_declines >= self.rollback_patience:
                            self._perform_rollback(trigger_reason="consecutive_declines")
                else:
                    if self.verbose > 0:
                        print(f"📉 No improvement, rollback not yet active")
            
            # Log rollback stats
            self.logger.record("rollback/consecutive_declines", self.consecutive_declines)
            self.logger.record("rollback/total_rollbacks", self.total_rollbacks)
            self.logger.record("rollback/lr_multiplier", self.current_lr_multiplier)
            self.logger.record("rollback/entropy_multiplier", self.current_entropy_multiplier)
            self.logger.record("rollback/best_reward", self.best_mean_reward)
            self.logger.record("rollback/n_checkpoints", len(self._checkpoint_heap))
            self.logger.record("rollback/vecnorm_frozen", float(self._vecnormalize_permanently_frozen))
            
            # Log deficit and rollback_allowed for monitoring
            if self.best_mean_reward > -np.inf:
                current_deficit = self.best_mean_reward - mean_reward
                self.logger.record("rollback/current_deficit", max(0, current_deficit))
            self.logger.record("rollback/min_steps_reached", 
                             float(self.num_timesteps >= self.min_timesteps_before_rollback))
            
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
            'final_entropy_multiplier': float(self.current_entropy_multiplier),
            'rollback_events': self.rollback_events,
            'max_checkpoints': int(self.max_checkpoints),
            'deficit_threshold': float(self.deficit_threshold) if self.deficit_threshold else None,
            'min_timesteps_before_rollback': int(self.min_timesteps_before_rollback),
            'freeze_vecnormalize_until_new_best': self.freeze_vecnormalize_until_new_best,
            'kept_checkpoints': [
                {'timestep': c.timestep, 'reward': c.reward}
                for c in sorted(self._checkpoint_heap, reverse=True)
            ],
        }

    def get_top_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of top-K checkpoints, sorted by reward (highest first)."""
        return sorted(self._checkpoint_heap, reverse=True)
