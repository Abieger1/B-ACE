#!/usr/bin/env python3
"""
expert_alignment_wrapper.py

Configuration 2: Expert Alignment Reward
Rewards PPO agent based on how closely its actions match the expert's recommendations.

Key insight: Policy still executes ITS OWN actions, but receives bonus reward
when those actions align with expert suggestions. No distribution shift problem.

Usage:
    from expert_alignment_wrapper import ExpertAlignmentWrapper, create_alignment_decay_fn
    from simple_aggressive_expert import SimpleAggressiveExpert, get_default_obs_indices
    
    expert = SimpleAggressiveExpert()
    obs_indices = get_default_obs_indices()
    
    env = ExpertAlignmentWrapper(
        env,
        expert=expert,
        obs_indices=obs_indices,
        alignment_coef=0.1,
    )
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Callable


class ExpertAlignmentWrapper(gym.Wrapper):
    """
    Adds alignment bonus reward based on similarity between agent action and expert action.
    
    The agent's action is ALWAYS executed. The expert action is only used to compute
    a bonus reward for actions that align with expert recommendations.
    
    This approach:
    - Avoids distribution shift (policy learns from its own experience)
    - Provides soft guidance toward expert behavior
    - Allows policy to discover better strategies if they exist
    
    Args:
        env: The environment to wrap
        expert: Expert policy with get_action(obs, obs_indices) method
        obs_indices: Dictionary mapping observation names to indices for the expert
        alignment_coef: Coefficient for alignment bonus (default: 0.1)
        action_weights: Per-dimension weights for alignment calculation
        decay_fn: Optional function(global_step) -> multiplier for coefficient decay
        debug: Enable debug printing
    """
    
    def __init__(
        self,
        env: gym.Env,
        expert: Any,
        obs_indices: Dict[str, int],
        alignment_coef: float = 0.1,
        action_weights: Optional[np.ndarray] = None,
        decay_fn: Optional[Callable[[int], float]] = None,
        debug: bool = False,
    ):
        super().__init__(env)
        self.expert = expert
        self.obs_indices = obs_indices
        self.base_alignment_coef = float(alignment_coef)
        self.alignment_coef = float(alignment_coef)
        self.decay_fn = decay_fn
        self.debug = debug
        
        # Default weights: [turn, altitude, g_force, fire]
        if action_weights is None:
            # Turn is most important, fire is binary so lower weight
            self.action_weights = np.array([1.0, 0.3, 0.6, 0.5], dtype=np.float32)
        else:
            self.action_weights = np.asarray(action_weights, dtype=np.float32)
        
        # Normalize weights to sum to 1
        self.action_weights = self.action_weights / np.sum(self.action_weights)
        
        # Statistics tracking
        self._last_expert_action = None
        self._last_alignment_score = 0.0
        self._episode_alignment_sum = 0.0
        self._episode_steps = 0
        self._total_alignment_bonuses = 0.0
        self._total_steps = 0
        self._global_step = 0
        
        # Cache for original observation
        self._last_original_obs = None
        
    def set_global_step(self, t: int):
        """Update global training step (for decay scheduling)."""
        self._global_step = int(t)
        
        # Update coefficient if decay function provided
        if self.decay_fn is not None:
            multiplier = self.decay_fn(self._global_step)
            self.alignment_coef = self.base_alignment_coef * multiplier
        
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
        self._episode_alignment_sum = 0.0
        self._episode_steps = 0
        
        # Cache the original observation (before any normalization)
        self._cache_original_obs()
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """
        Execute agent's action and add alignment bonus.
        
        IMPORTANT: The agent's action is executed, NOT the expert's.
        The expert action is only used to compute an alignment bonus.
        """
        # Get observation BEFORE step (current state)
        # We need the original (unnormalized, base) obs for the expert
        obs_for_expert = self._get_obs_for_expert()
        
        # Get expert's recommended action
        expert_action = self.expert.get_action(obs_for_expert, self.obs_indices)
        self._last_expert_action = expert_action.copy()
        
        # Execute AGENT's action (not expert's!)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Cache new observation for next step
        self._cache_original_obs()
        
        # Compute alignment score
        alignment_score = self._compute_alignment(action, expert_action)
        self._last_alignment_score = alignment_score
        
        # Add alignment bonus to reward
        alignment_bonus = self.alignment_coef * alignment_score
        reward = float(reward) + alignment_bonus
        
        # Track statistics
        self._episode_alignment_sum += alignment_score
        self._episode_steps += 1
        self._total_alignment_bonuses += alignment_bonus
        self._total_steps += 1
        
        # Add info for logging
        info["expert_alignment_score"] = float(alignment_score)
        info["expert_alignment_bonus"] = float(alignment_bonus)
        info["expert_alignment_coef"] = float(self.alignment_coef)
        info["expert_action"] = expert_action.tolist()
        
        if self.debug and self._total_steps % 500 == 0:
            print(f"[ALIGN] step={self._total_steps} "
                  f"agent=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.1f}] "
                  f"expert=[{expert_action[0]:.2f},{expert_action[1]:.2f},{expert_action[2]:.2f},{expert_action[3]:.1f}] "
                  f"score={alignment_score:.3f} bonus={alignment_bonus:.4f}")
        
        return obs, reward, terminated, truncated, info
    
    def _flatten_obs_dict(self, obs_dict: dict) -> np.ndarray:
        """Recursively flatten observation dict to array."""
        flat = []
        for v in obs_dict.values():
            if isinstance(v, np.ndarray):
                flat.extend(v.flatten().tolist())
            elif isinstance(v, (list, tuple)):
                flat.extend([float(x) for x in v])
            elif isinstance(v, dict):
                # Recurse into nested dict
                flat.extend(self._flatten_obs_dict(v).tolist())
            elif isinstance(v, (int, float)):
                flat.append(float(v))
            # Skip non-numeric types
        return np.asarray(flat, dtype=np.float32)

    def _cache_original_obs(self):
        """Cache the original observation from the base environment."""
        e = self.env
        while hasattr(e, 'env'):
            # Check for _last_obs (numpy array)
            if hasattr(e, '_last_obs') and e._last_obs is not None:
                if isinstance(e._last_obs, np.ndarray):
                    self._last_original_obs = np.asarray(e._last_obs, dtype=np.float32)
                    return
            # Check for SingleAgentBACEEnv's _last_obs_dict
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
        """
        Get observation suitable for expert policy.
        
        The expert needs the original (non-enriched, non-normalized) observations.
        This method walks down the wrapper chain to find the base observation.
        """
        # Try cached observation first
        if self._last_original_obs is not None:
            return self._last_original_obs
        
        # Try to get from get_original_obs method
        if hasattr(self.env, "get_original_obs"):
            orig = self.env.get_original_obs()
            if orig is not None and isinstance(orig, np.ndarray):
                return np.asarray(orig, dtype=np.float32)
        
        # Walk up wrapper chain looking for original obs
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
                elif isinstance(obs, dict):
                    if 'observation' in obs:
                        return np.asarray(obs['observation'], dtype=np.float32)
        
        # Last resort: return zeros (will log warning)
        print("[WARN] ExpertAlignmentWrapper: Could not find original observation")
        return np.zeros(27, dtype=np.float32)
    
    def _compute_alignment(
        self,
        agent_action: np.ndarray,
        expert_action: np.ndarray,
    ) -> float:
        """
        Compute alignment score between agent and expert actions.
        
        Returns:
            Score in [0, 1] where 1 = perfect alignment
        """
        a_agent = np.asarray(agent_action, dtype=np.float32)
        a_expert = np.asarray(expert_action, dtype=np.float32)
        
        # Clip both to action space bounds
        a_agent = np.clip(a_agent, -1.0, 1.0)
        a_expert = np.clip(a_expert, -1.0, 1.0)
        
        # Compute per-dimension absolute difference (max diff is 2.0 for [-1,1] space)
        diff = np.abs(a_agent - a_expert)
        
        # Normalize to [0, 1] per dimension
        # diff=0 -> similarity=1, diff=2 -> similarity=0
        per_dim_similarity = 1.0 - (diff / 2.0)
        
        # Weighted mean
        alignment = float(np.sum(per_dim_similarity * self.action_weights))
        
        return alignment
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get alignment statistics for current episode."""
        if self._episode_steps == 0:
            return {"mean_alignment": 0.0, "episode_steps": 0}
        
        return {
            "mean_alignment": self._episode_alignment_sum / self._episode_steps,
            "episode_steps": self._episode_steps,
        }
    
    def get_total_stats(self) -> Dict[str, float]:
        """Get alignment statistics across all steps."""
        if self._total_steps == 0:
            return {"mean_alignment_bonus": 0.0, "total_steps": 0}
        
        return {
            "mean_alignment_bonus": self._total_alignment_bonuses / self._total_steps,
            "total_steps": self._total_steps,
            "current_coef": self.alignment_coef,
        }
    
    def get_last_expert_action(self) -> Optional[np.ndarray]:
        """Get the expert's last recommended action."""
        return self._last_expert_action
    
    def get_last_alignment_score(self) -> float:
        """Get the last computed alignment score."""
        return self._last_alignment_score


def create_alignment_decay_fn(
    decay_start: int =400_000,
    decay_end: int = 2_000_000,
    final_multiplier: float = 0.3,
) -> Callable[[int], float]:
    """
    Create a decay function for alignment coefficient.
    
    Starts at 1.0, holds until decay_start, then linearly decays to final_multiplier
    by decay_end.
    
    Args:
        decay_start: Step to begin decay
        decay_end: Step to reach final value
        final_multiplier: Final multiplier value (e.g., 0.3 means 30% of original)
    
    Returns:
        Function mapping global_step -> multiplier
    """
    def decay_fn(step: int) -> float:
        if step < decay_start:
            return 1.0
        if step >= decay_end:
            return final_multiplier
        
        # Linear decay
        progress = (step - decay_start) / (decay_end - decay_start)
        return 1.0 - progress * (1.0 - final_multiplier)
    
    return decay_fn


# For backwards compatibility with existing code that imports this
def get_default_enriched_obs_indices() -> Dict[str, int]:
    """
    Enriched observation indices (47 dims).
    Use get_default_obs_indices() from simple_aggressive_expert for base indices.
    """
    # This is a placeholder - the SimpleAggressiveExpert uses base indices
    from simple_aggressive_expert import get_default_obs_indices
    return get_default_obs_indices()
