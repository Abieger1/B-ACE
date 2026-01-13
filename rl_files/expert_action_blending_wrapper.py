#!/usr/bin/env python3
"""
expert_action_blending_wrapper.py

Configuration 3: Expert Action Blending
Blends PPO agent action with expert action before execution.

Key insight: Unlike Config 2 (alignment reward), here we MODIFY the action
that gets executed. The agent experiences trajectories guided by the expert,
but progressively takes more control as alpha decays.

Blending formula:
    blended_action = alpha * expert_action + (1 - alpha) * ppo_action

Where alpha decays from high (expert-dominated) to low (PPO-dominated).

Key differences from Config 2:
- Config 2: Agent executes ITS OWN action, gets bonus reward for matching expert
- Config 3: Agent's action is BLENDED with expert before execution

Considerations:
- This creates a "behavioral cloning with gradual handoff" effect
- Early training: agent sees expert-quality trajectories
- Late training: agent takes full control
- The PPO still learns from the rewards of the blended trajectories

Usage:
    from expert_action_blending_wrapper import ExpertActionBlendingWrapper, create_alpha_decay_fn
    from simple_aggressive_expert import SimpleAggressiveExpert, get_default_obs_indices
    
    expert = SimpleAggressiveExpert()
    obs_indices = get_default_obs_indices()
    
    alpha_decay = create_alpha_decay_fn(
        initial_alpha=0.7,
        final_alpha=0.1,
        decay_start=500_000,
        decay_end=4_000_000,
    )
    
    env = ExpertActionBlendingWrapper(
        env,
        expert=expert,
        obs_indices=obs_indices,
        alpha_fn=alpha_decay,
        blend_dims=[0, 1, 2],  # Blend turn, altitude, g_force (not fire)
    )
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Callable, List


class ExpertActionBlendingWrapper(gym.Wrapper):
    """
    Blends agent action with expert action before execution.
    
    The BLENDED action is executed, not the agent's raw action.
    
    This approach:
    - Provides expert-quality trajectories early in training
    - Gradually hands control to PPO as alpha decays
    - Creates smooth curriculum from imitation to reinforcement learning
    
    Args:
        env: The environment to wrap
        expert: Expert policy with get_action(obs, obs_indices) method
        obs_indices: Dictionary mapping observation names to indices for the expert
        alpha_fn: Function(global_step) -> alpha coefficient [0, 1]
                  alpha=1 means 100% expert, alpha=0 means 100% PPO
        blend_dims: List of action dimensions to blend (default: [0, 1, 2] for turn, alt, g)
                    Fire (dim 3) is often handled separately
        fire_blend_mode: How to handle fire action:
                         'ppo' - always use PPO's fire decision
                         'expert' - always use expert's fire decision  
                         'blend' - blend like other dimensions
                         'gate' - use expert fire when alpha > threshold
        fire_gate_threshold: For 'gate' mode, use expert fire if alpha > this
        debug: Enable debug printing
    """
    
    def __init__(
        self,
        env: gym.Env,
        expert: Any,
        obs_indices: Dict[str, int],
        alpha_fn: Callable[[int], float],
        blend_dims: Optional[List[int]] = None,
        fire_blend_mode: str = 'gate',
        fire_gate_threshold: float = 0.3,
        debug: bool = False,
    ):
        super().__init__(env)
        self.expert = expert
        self.obs_indices = obs_indices
        self.alpha_fn = alpha_fn
        self.fire_blend_mode = fire_blend_mode
        self.fire_gate_threshold = fire_gate_threshold
        self.debug = debug
        
        # Default: blend turn (0), altitude (1), g_force (2), handle fire (3) separately
        if blend_dims is None:
            self.blend_dims = [0, 1, 2]
        else:
            self.blend_dims = list(blend_dims)
        
        # Current alpha value (updated each step)
        self._current_alpha = 1.0
        
        # Statistics tracking
        self._last_expert_action = None
        self._last_ppo_action = None
        self._last_blended_action = None
        self._episode_expert_contribution = 0.0
        self._episode_steps = 0
        self._total_steps = 0
        self._global_step = 0
        
        # Cache for original observation
        self._last_original_obs = None
        
    def set_global_step(self, t: int):
        """Update global training step (for alpha scheduling)."""
        self._global_step = int(t)
        self._current_alpha = self.alpha_fn(self._global_step)
        
        # Propagate to wrapped env
        if hasattr(self.env, "set_global_step"):
            self.env.set_global_step(t)
    
    def reset(self, **kwargs):
        """Reset environment and expert state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset expert if it has episode state
        if hasattr(self.expert, "reset"):
            self.expert.reset()
        
        # Reset episode stats
        self._episode_expert_contribution = 0.0
        self._episode_steps = 0
        
        # Cache the original observation
        self._cache_original_obs()
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """
        Blend agent's action with expert's action, then execute.
        
        The BLENDED action is what gets executed in the environment.
        """
        # Get current alpha
        alpha = self._current_alpha
        
        # Get observation for expert (base, unnormalized)
        obs_for_expert = self._get_obs_for_expert()
        
        # Get expert's recommended action
        expert_action = self.expert.get_action(obs_for_expert, self.obs_indices)
        self._last_expert_action = expert_action.copy()
        self._last_ppo_action = np.array(action).copy()
        
        # Compute blended action
        blended_action = self._blend_actions(action, expert_action, alpha)
        self._last_blended_action = blended_action.copy()
        
        # Execute BLENDED action
        obs, reward, terminated, truncated, info = self.env.step(blended_action)
        
        # Cache new observation for next step
        self._cache_original_obs()
        
        # Track statistics
        self._episode_expert_contribution += alpha
        self._episode_steps += 1
        self._total_steps += 1
        
        # Add info for logging
        info["blend_alpha"] = float(alpha)
        info["expert_action"] = expert_action.tolist()
        info["ppo_action"] = self._last_ppo_action.tolist()
        info["blended_action"] = blended_action.tolist()
        
        # Compute action divergence for monitoring
        divergence = np.mean(np.abs(self._last_ppo_action - expert_action))
        info["ppo_expert_divergence"] = float(divergence)
        
        if self.debug and self._total_steps % 500 == 0:
            print(f"[BLEND] step={self._total_steps} alpha={alpha:.3f} "
                  f"ppo=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.1f}] "
                  f"expert=[{expert_action[0]:.2f},{expert_action[1]:.2f},{expert_action[2]:.2f},{expert_action[3]:.1f}] "
                  f"blended=[{blended_action[0]:.2f},{blended_action[1]:.2f},{blended_action[2]:.2f},{blended_action[3]:.1f}]")
        
        return obs, reward, terminated, truncated, info
    
    def _blend_actions(
        self,
        ppo_action: np.ndarray,
        expert_action: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        Blend PPO and expert actions.
        
        blended = alpha * expert + (1 - alpha) * ppo
        
        Args:
            ppo_action: Action from PPO policy
            expert_action: Action from expert policy
            alpha: Blending coefficient (1 = all expert, 0 = all PPO)
        
        Returns:
            Blended action
        """
        ppo_a = np.asarray(ppo_action, dtype=np.float32).copy()
        expert_a = np.asarray(expert_action, dtype=np.float32).copy()
        
        # Start with PPO action
        blended = ppo_a.copy()
        
        # Blend specified dimensions
        for dim in self.blend_dims:
            if dim < len(blended) and dim < len(expert_a):
                blended[dim] = alpha * expert_a[dim] + (1 - alpha) * ppo_a[dim]
        
        # Handle fire action (dim 3) separately
        fire_dim = 3
        if fire_dim < len(blended):
            if self.fire_blend_mode == 'ppo':
                # Keep PPO's fire decision
                pass
            elif self.fire_blend_mode == 'expert':
                # Always use expert's fire decision
                blended[fire_dim] = expert_a[fire_dim]
            elif self.fire_blend_mode == 'blend':
                # Blend like other dimensions
                blended[fire_dim] = alpha * expert_a[fire_dim] + (1 - alpha) * ppo_a[fire_dim]
            elif self.fire_blend_mode == 'gate':
                # Use expert fire when alpha is high, otherwise PPO
                if alpha > self.fire_gate_threshold:
                    blended[fire_dim] = expert_a[fire_dim]
                # else keep PPO's fire decision
            elif self.fire_blend_mode == 'or':
                # Fire if either wants to fire
                if expert_a[fire_dim] > 0.5 or ppo_a[fire_dim] > 0.5:
                    blended[fire_dim] = 1.0
                else:
                    blended[fire_dim] = -1.0
        
        # Clip to action bounds
        blended = np.clip(blended, -1.0, 1.0)
        
        return blended
    
    def _flatten_obs_dict(self, obs_dict: dict) -> np.ndarray:
        """Recursively flatten observation dict to array."""
        flat = []
        for v in obs_dict.values():
            if isinstance(v, np.ndarray):
                flat.extend(v.flatten().tolist())
            elif isinstance(v, (list, tuple)):
                flat.extend([float(x) for x in v])
            elif isinstance(v, dict):
                flat.extend(self._flatten_obs_dict(v).tolist())
            elif isinstance(v, (int, float)):
                flat.append(float(v))
        return np.asarray(flat, dtype=np.float32)

    def _cache_original_obs(self):
        """Cache the original observation from the base environment."""
        e = self.env
        while hasattr(e, 'env'):
            if hasattr(e, '_last_obs') and e._last_obs is not None:
                if isinstance(e._last_obs, np.ndarray):
                    self._last_original_obs = np.asarray(e._last_obs, dtype=np.float32)
                    return
            if hasattr(e, '_last_obs_dict'):
                agent_id = getattr(e, 'controlled_agent_id', 'agent_0')
                if agent_id in e._last_obs_dict:
                    obs = e._last_obs_dict[agent_id]
                    if isinstance(obs, np.ndarray):
                        self._last_original_obs = np.asarray(obs, dtype=np.float32)
                        return
                    elif isinstance(obs, dict):
                        if 'observation' in obs and isinstance(obs['observation'], np.ndarray):
                            self._last_original_obs = np.asarray(obs['observation'], dtype=np.float32)
                        else:
                            self._last_original_obs = self._flatten_obs_dict(obs)
                        return
            e = e.env
        
        # Final check on innermost env
        if hasattr(e, '_last_obs_dict'):
            agent_id = getattr(e, 'controlled_agent_id', 'agent_0')
            if agent_id in e._last_obs_dict:
                obs = e._last_obs_dict[agent_id]
                if isinstance(obs, np.ndarray):
                    self._last_original_obs = np.asarray(obs, dtype=np.float32)
                elif isinstance(obs, dict):
                    if 'observation' in obs and isinstance(obs['observation'], np.ndarray):
                        self._last_original_obs = np.asarray(obs['observation'], dtype=np.float32)
                    else:
                        self._last_original_obs = self._flatten_obs_dict(obs)
    
    def _get_obs_for_expert(self) -> np.ndarray:
        """Get observation suitable for expert policy (base, unnormalized)."""
        if self._last_original_obs is not None:
            return self._last_original_obs
        
        if hasattr(self.env, "get_original_obs"):
            orig = self.env.get_original_obs()
            if orig is not None and isinstance(orig, np.ndarray):
                return np.asarray(orig, dtype=np.float32)
        
        # Walk up wrapper chain
        e = self.env
        while hasattr(e, "env"):
            if hasattr(e, "_last_obs") and e._last_obs is not None:
                if isinstance(e._last_obs, np.ndarray):
                    return np.asarray(e._last_obs, dtype=np.float32)
            if hasattr(e, "_last_original_obs") and e._last_original_obs is not None:
                if isinstance(e._last_original_obs, np.ndarray):
                    return np.asarray(e._last_original_obs, dtype=np.float32)
            e = e.env
        
        # Check innermost env
        if hasattr(e, '_last_obs_dict'):
            agent_id = getattr(e, 'controlled_agent_id', 'agent_0')
            if agent_id in e._last_obs_dict:
                obs = e._last_obs_dict[agent_id]
                if isinstance(obs, np.ndarray):
                    return np.asarray(obs, dtype=np.float32)
                elif isinstance(obs, dict) and 'observation' in obs:
                    return np.asarray(obs['observation'], dtype=np.float32)
        
        print("[WARN] ExpertActionBlendingWrapper: Could not find original observation")
        return np.zeros(27, dtype=np.float32)
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get blending statistics for current episode."""
        if self._episode_steps == 0:
            return {"mean_alpha": 0.0, "episode_steps": 0}
        
        return {
            "mean_alpha": self._episode_expert_contribution / self._episode_steps,
            "episode_steps": self._episode_steps,
        }
    
    def get_total_stats(self) -> Dict[str, float]:
        """Get blending statistics across all steps."""
        return {
            "total_steps": self._total_steps,
            "current_alpha": self._current_alpha,
        }
    
    def get_last_actions(self) -> Dict[str, Optional[np.ndarray]]:
        """Get the last PPO, expert, and blended actions."""
        return {
            "ppo": self._last_ppo_action,
            "expert": self._last_expert_action,
            "blended": self._last_blended_action,
        }


def create_alpha_decay_fn(
    initial_alpha: float = 0.7,
    final_alpha: float = 0.1,
    decay_start: int = 500_000,
    decay_end: int = 4_000_000,
    decay_type: str = 'linear',
) -> Callable[[int], float]:
    """
    Create a decay function for alpha coefficient.
    
    Args:
        initial_alpha: Starting alpha (expert influence) - typically high (0.5-0.8)
        final_alpha: Ending alpha - typically low but nonzero (0.05-0.2)
        decay_start: Step to begin decay (hold at initial_alpha before this)
        decay_end: Step to reach final_alpha
        decay_type: 'linear', 'exponential', or 'cosine'
    
    Returns:
        Function mapping global_step -> alpha
    """
    def decay_fn(step: int) -> float:
        if step < decay_start:
            return initial_alpha
        if step >= decay_end:
            return final_alpha
        
        progress = (step - decay_start) / (decay_end - decay_start)
        
        if decay_type == 'linear':
            return initial_alpha - progress * (initial_alpha - final_alpha)
        elif decay_type == 'exponential':
            # Exponential decay (faster early, slower late)
            import math
            return final_alpha + (initial_alpha - final_alpha) * math.exp(-3 * progress)
        elif decay_type == 'cosine':
            # Cosine annealing (smooth)
            import math
            return final_alpha + 0.5 * (initial_alpha - final_alpha) * (1 + math.cos(math.pi * progress))
        else:
            # Default to linear
            return initial_alpha - progress * (initial_alpha - final_alpha)
    
    return decay_fn


def create_staged_alpha_fn(
    stages: List[Dict[str, Any]],
) -> Callable[[int], float]:
    """
    Create a multi-stage alpha schedule.
    
    Example:
        stages = [
            {'end_step': 500_000, 'alpha': 0.8},      # Phase 1: High expert
            {'end_step': 2_000_000, 'alpha': 0.4},    # Phase 2: Medium blend
            {'end_step': 5_000_000, 'alpha': 0.1},    # Phase 3: Low expert
        ]
    
    Alpha transitions linearly between stages.
    """
    def staged_fn(step: int) -> float:
        prev_end = 0
        prev_alpha = stages[0]['alpha'] if stages else 0.5
        
        for stage in stages:
            if step < stage['end_step']:
                # Linear interpolation within this stage
                progress = (step - prev_end) / (stage['end_step'] - prev_end)
                return prev_alpha + progress * (stage['alpha'] - prev_alpha)
            prev_end = stage['end_step']
            prev_alpha = stage['alpha']
        
        # After all stages, return final alpha
        return stages[-1]['alpha'] if stages else 0.1
    
    return staged_fn
