#!/usr/bin/env python3
"""
option_specific_reward_wrapper.py

Wrapper that adds ADDITIONAL option-specific reward shaping on top of existing
RewardShapingWrapper. This is a LIGHTWEIGHT add-on that provides tactical guidance
specific to each option's learning objective.

Key Features:
- COMPLEMENTS your existing RewardShapingWrapper (doesn't replace it!)
- Adds small option-specific bonuses/penalties (typically 5-10% of base shaping)
- Layered architecture: Base shaping â†’ Option-specific guidance
- Each option gets focused tactical incentives

Architecture:
    Env â†’ EnrichedObservationWrapper â†’ RewardShapingWrapper â†’ OptionSpecificRewardWrapper
          â†‘ Your existing setup              â†‘ Your reward work      â†‘ NEW: Option focus

Usage:
    # Your existing setup already includes EnrichedObservationWrapper + RewardShapingWrapper
    # Just add OptionSpecificRewardWrapper on top for Phase 1 training
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional
from tactical_options import TacticalOption, Option, extract_tactical_state


class OptionSpecificRewardWrapper(gym.Wrapper):
    """
    Shapes rewards for training a specific low-level option policy.
    
    This wrapper is used during Phase 1 of hierarchical training to train
    specialized policies for each tactical option.
    """
    
    def __init__(self,
                 env: gym.Env,
                 option_id: TacticalOption,
                 option: Option,
                 obs_labels: Optional[Dict[str, int]] = None,
                 enriched_obs: bool = True,
                 shaping_strength: float = 1.0):
        """
        Initialize option-specific reward shaping.
        
        Args:
            env: Base environment
            option_id: Which option this policy is being trained for
            option: Option object with metadata
            obs_labels: Observation label mapping
            enriched_obs: Whether using enriched observations
            shaping_strength: Multiplier for all option-specific shaping
        """
        super().__init__(env)
        
        self.option_id = option_id
        self.option = option
        self.obs_labels = obs_labels or {}
        self.enriched_obs = enriched_obs
        self.shaping_strength = shaping_strength
        
        # Track previous state for potential-based shaping
        self.prev_state = None
    
    def _extract_state(self, obs: np.ndarray) -> Dict[str, Any]:
        """Extract tactical state from observation."""
        return extract_tactical_state(obs, self.obs_labels, self.enriched_obs)
    
    def _shape_defend_hvaa(self, 
                          base_reward: float, 
                          state: Dict[str, Any],
                          info: Dict[str, Any]) -> float:
        """
        LIGHTWEIGHT option-specific shaping for DEFEND_HVAA.
        
        Your RewardShapingWrapper already handles:
        - HVAA escort guardband (hvaa_escort_shaping)
        - BEZ avoidance (bez_shaping)
        - Defensive positioning (dmc_shaping)
        
        This adds ONLY:
        - Small bonus for optimal escort formation
        - Slight emphasis on interception posture
        
        All values are SMALL (0.0001-0.0005 range) to avoid overwhelming your base shaping.
        """
        shaped = base_reward
        
        # 1. Formation quality bonus (very small)
        # Your hvaa_escort_shaping already handles guardband
        # This just adds tiny bonus for ideal escort geometry
        hvaa_dist = state.get('distance_to_hvaa', 1.0)
        target_dist = 0.35  # Optimal normalized escort distance
        
        if 0.25 < hvaa_dist < 0.45:  # In sweet spot
            shaped += 0.0002 * self.shaping_strength  # Tiny bonus
        
        # 2. Interception readiness (facing threats while near HVAA)
        if state.get('enemy_detected', False) and hvaa_dist < 0.5:
            hvaa_angle = abs(state.get('hvaa_angle_off', 180.0))
            enemy_angle = abs(state.get('aspect_angle_to_enemy', 180.0))
            
            # Slight bonus for being between HVAA and threat
            if hvaa_angle > 90 and enemy_angle < 90:
                shaped += 0.0003 * self.shaping_strength
        
        return shaped
    
    def _shape_intercept_enemy(self,
                               base_reward: float,
                               state: Dict[str, Any],
                               info: Dict[str, Any]) -> float:
        """
        LIGHTWEIGHT shaping for INTERCEPT_ENEMY.
        
        Your RewardShapingWrapper already handles:
        - Offensive dominance (offense_shaping with wez_dom_weight)
        - Distance closing (implicit in offensive positioning)
        - Missile launch quality (missile_launch_shaping)
        
        This adds ONLY:
        - Small aggression bonus for closing quickly
        - Tiny emphasis on maintaining pursuit geometry
        """
        shaped = base_reward
        
        # 1. Aggressive closing bonus (very small)
        # Your offense_shaping already rewards good positioning
        # This adds tiny extra for rapid closing
        if self.prev_state is not None:
            prev_dist = self.prev_state.get('distance_to_enemy', 1.0)
            curr_dist = state.get('distance_to_enemy', 1.0)
            distance_delta = prev_dist - curr_dist  # Positive = closing
            
            if distance_delta > 0.01:  # Only if closing significantly
                shaped += 0.0002 * distance_delta * self.shaping_strength
        
        # 2. Stern chase bonus (pursuit geometry)
        aspect = abs(state.get('bez_aspect_angle', 180.0))
        if aspect < 30 and state.get('enemy_detected', False):
            shaped += 0.0001 * self.shaping_strength
        
        return shaped
    
    def _shape_evade_missile(self,
                            base_reward: float,
                            state: Dict[str, Any],
                            info: Dict[str, Any]) -> float:
        """
        LIGHTWEIGHT shaping for EVADE_MISSILE.
        
        Your RewardShapingWrapper already handles:
        - DMC-based evasion guidance (dmc_shaping)
        - Threat avoidance (dmc_bad_penalty)
        
        This adds ONLY:
        - Tiny bonus for optimal evasion execution
        - Small emphasis on maintaining evasion geometry
        """
        shaped = base_reward
        
        # 1. Evasion execution quality (very small)
        # Your dmc_shaping already guides heading
        # This adds tiny bonus for excellent geometry
        dmc_norm = state.get('dmc_normalized', 0.5)
        
        if dmc_norm < 0.2:  # Excellent evasion geometry
            shaped += 0.0002 * (0.2 - dmc_norm) * self.shaping_strength
        
        # 2. Successfully outside threat cone
        if not state.get('inside_threat_cone', False):
            shaped += 0.0001 * self.shaping_strength
        
        return shaped
    
    def _shape_offensive_positioning(self,
                                    base_reward: float,
                                    state: Dict[str, Any],
                                    info: Dict[str, Any]) -> float:
        """
        LIGHTWEIGHT shaping for OFFENSIVE_POSITIONING.
        
        Your RewardShapingWrapper already handles:
        - BEZ avoidance (bez_shaping with bez_penalty_inside)
        - Offensive factor rewards (offense_shaping)
        
        This adds ONLY:
        - Small bonus for optimal maneuvering geometry
        - Tiny emphasis on aspect angle optimization
        """
        shaped = base_reward
        
        # 1. Optimal positioning bonus (very small)
        # Your bez_shaping already penalizes BEZ entry
        # This rewards being in the "sweet spot" outside BEZ
        bez_penetration = state.get('bez_penetration', 0.0)
        if 0.1 < bez_penetration < 0.3:  # Outside BEZ but not too far
            shaped += 0.0002 * self.shaping_strength
        
        # 2. Aspect angle optimization
        aspect = abs(state.get('bez_aspect_angle', 180.0))
        if aspect < 45:  # Good stern aspect
            shaped += 0.0001 * self.shaping_strength
        
        return shaped
    
    def _shape_defensive_positioning(self,
                                    base_reward: float,
                                    state: Dict[str, Any],
                                    info: Dict[str, Any]) -> float:
        """
        LIGHTWEIGHT shaping for DEFENSIVE_POSITIONING.
        
        Your RewardShapingWrapper already handles:
        - BEZ escape (bez_shaping heavily penalizes being inside)
        - DMC-based escape guidance (dmc_shaping)
        
        This adds ONLY:
        - Small bonus for rapid BEZ escape
        - Tiny emphasis on optimal escape heading
        """
        shaped = base_reward
        
        # 1. Rapid escape bonus (very small)
        # Your bez_shaping already penalizes being inside
        # This adds tiny bonus for quickly reducing penetration
        if self.prev_state is not None:
            prev_bez = self.prev_state.get('bez_penetration', 1.0)
            bez_penetration = state.get('bez_penetration', 0.0)
            bez_delta = prev_bez - bez_penetration  # Positive = moving away
            
            if bez_delta > 0.02:  # Only if escaping significantly
                shaped += 0.0003 * bez_delta * self.shaping_strength
        
        # 2. Successfully escaped (outside BEZ)
        if state.get('bez_penetration', 1.0) < 0.1:
            shaped += 0.0001 * self.shaping_strength
        
        return shaped
    
    def reward(self, reward: float) -> float:
        """
        Shape the reward based on current option.
        
        Args:
            reward: Base reward from environment
            
        Returns:
            Shaped reward
        """
        # Get current observation and info from last step
        # Note: This is called by gym.Wrapper after step()
        
        # For now, return base reward (actual shaping happens in step())
        return reward
    
    def step(self, action):
        """Execute step and apply option-specific reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract current state
        state = self._extract_state(obs)
        
        # Apply option-specific shaping
        if self.option_id == TacticalOption.DEFEND_HVAA:
            shaped_reward = self._shape_defend_hvaa(reward, state, info)
        elif self.option_id == TacticalOption.INTERCEPT_ENEMY:
            shaped_reward = self._shape_intercept_enemy(reward, state, info)
        elif self.option_id == TacticalOption.EVADE_MISSILE:
            shaped_reward = self._shape_evade_missile(reward, state, info)
        elif self.option_id == TacticalOption.OFFENSIVE_POSITIONING:
            shaped_reward = self._shape_offensive_positioning(reward, state, info)
        elif self.option_id == TacticalOption.DEFENSIVE_POSITIONING:
            shaped_reward = self._shape_defensive_positioning(reward, state, info)
        else:
            shaped_reward = reward
        
        # Add intrinsic option bonus
        shaped_reward += self.option.reward_bonus
        
        # Store state for next step
        self.prev_state = state
        
        # Add shaping info for debugging
        info['option_shaping'] = shaped_reward - reward
        info['base_reward'] = reward
        info['shaped_reward'] = shaped_reward
        
        return obs, shaped_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and clear state."""
        self.prev_state = None
        return self.env.reset(**kwargs)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_option_env(base_env_fn,
                     option_id: TacticalOption,
                     option: Option,
                     obs_labels: Dict[str, int],
                     enriched_obs: bool = True,
                     shaping_strength: float = 1.0):
    """
    Factory function to create an environment for training a specific option.
    
    Args:
        base_env_fn: Function that creates base environment
        option_id: Which option to train
        option: Option object
        obs_labels: Observation labels
        enriched_obs: Whether using enriched observations
        shaping_strength: Reward shaping strength
        
    Returns:
        Wrapped environment ready for training this option
    """
    env = base_env_fn()
    
    wrapped = OptionSpecificRewardWrapper(
        env,
        option_id=option_id,
        option=option,
        obs_labels=obs_labels,
        enriched_obs=enriched_obs,
        shaping_strength=shaping_strength
    )
    
    return wrapped


if __name__ == "__main__":
    print("OptionSpecificRewardWrapper - Testing")
    print("="*60)
    
    from tactical_options import create_tactical_options
    
    options = create_tactical_options()
    
    print("\nOption-specific reward shaping configured for:")
    for opt_id, option in options.items():
        print(f"  - {option.name}")
    
    print("\nWrapper ready for hierarchical training!")
