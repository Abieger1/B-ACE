#!/usr/bin/env python3
"""
reward_shaping_wrapper.py - REVISED WITH RL METRICS TRACKING

Modular reward shaping wrapper for B-ACE using pursuit-evasion theory.
Designed for easy experimentation with different reward components.

NEW FEATURES (for thesis RL metrics):
- Detailed reward component breakdown per step
- Episode-level reward statistics
- Automatic logging to TensorBoard via info dict
- Shaping contribution percentage tracking
- Compatible with custom training callback

Key Features:
- Easily toggle individual reward components on/off
- Built-in logging to track contribution of each component
- Configurable weights for each reward term
- Potential-based shaping to preserve optimal policy
- Integration with EnrichedObservationWrapper

Usage:
    from reward_shaping_wrapper import RewardShapingWrapper, RewardShapingConfig
    
    # Configure which components to use
    config = RewardShapingConfig(
        enable_bez_shaping=True,
        enable_dmc_shaping=True,
        bez_weight=0.5,
        dmc_weight=0.3
    )
    
    # Wrap your environment
    env = RewardShapingWrapper(env, config=config)
    
    # In training callback, access reward breakdown:
    # info['reward_breakdown'] = per-step component breakdown
    # info['episode_reward_stats'] = cumulative stats when episode ends
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import csv
from pathlib import Path






@dataclass
class RewardShapingConfig:
    """
    Configuration for reward shaping components.
    
    Each component can be individually enabled/disabled and weighted.
    Weights determine the relative importance of each reward term.
    """
    
    # ===== BEZ (Basic Engagement Zone) Shaping =====
    enable_bez_shaping: bool = True
    bez_weight: float = 0.00025
    bez_penalty_inside: float = -0.05  # Penalty when inside BEZ
    bez_reward_outside: float = 0.001  # Reward for staying outside
    
    # ===== DMC (Dynamic Maneuvering Cue) Shaping =====
    enable_dmc_shaping: bool = True
    dmc_good_threshold: float = 0.1   # DMC below this = decent geometry
    dmc_bad_threshold: float  = 0.4   # above this = poor geometry
    dmc_good_reward: float    = 0.008  # per-step
    dmc_bad_penalty: float    = -0.01 # per-step
    dmc_weight: float         = 0.001
    
    # ===== Heading Alignment Shaping =====
    enable_heading_shaping: bool = True
    heading_weight: float = 0.2
    heading_reward_aligned: float = 0.05  # Reward for heading toward safe direction
    
    # ===== Distance Shaping =====
    enable_distance_shaping: bool = True
    distance_weight: float = 0.3
    distance_safe_threshold: float = 0.5  # Normalized distance considered "safe"
    
    # ===== Time-to-Capture Shaping =====
    enable_ttc_shaping: bool = True
    ttc_weight: float = 0.2
    ttc_safe_threshold: float = 50.0  # Time (normalized) considered safe

    # ===== HVAA Escort Shaping =====
    enable_hvaa_escort_shaping: bool = True
    hvaa_guardband_min_nm: float = 10.0      # slightly tighter inner guardband
    hvaa_guardband_max_nm: float = 40.0      # slightly tighter outer guardband
    hvaa_target_norm: float = 0.08           # desired normalized escort distance
    hvaa_max_penalty: float = -0.05          # per-step clip for escort penalty
    hvaa_escort_weight: float = 0.001     
    
    # ===== Multi-Threat Shaping =====
    enable_multithreat_shaping: bool = True
    multithreat_weight: float = 0.3
    surrounded_penalty: float = -0.2  # Penalty when fully surrounded
    
    # ===== Speed Ratio Shaping =====
    enable_speed_shaping: bool = True
    speed_weight: float = 0.1
    speed_advantage_bonus: float = 0.05  # Bonus when faster than threat

    # ===== Offensive Shaping (WEZ dominance + offensive TTC) =====
    enable_offense_shaping: bool = True
    wez_dom_weight: float = 0.002          # scales offensive_dominance
    offense_ttc_weight: float = 0.01   
    
    # ===== Global Settings =====
    use_potential_based: bool = True  # Use potential-based shaping (preserves optimality)
    gamma: float = 0.99  # Discount factor for potential-based shaping
    normalize_rewards: bool = True  # Normalize shaped rewards to [-1, 1]

    # ===== Missile Launch Shaping =====
    enable_missile_launch_shaping: bool = True
    good_launch_bonus: float = 8.0      # reward for a good shot
    bad_launch_penalty: float = -6.0    # penalty for a bad shot
    launch_off_dom_min: float = 0.5     # min offensive_dominance for “good” launch
    launch_ttc_min: float = 0.5         # min offensive_ttc_score for “good”
    base_launch_cost: float = -1.0 
    
    # ===== Logging =====
    enable_logging: bool = True
    log_frequency: int = 50000  # Log statistics every N steps (changed from 100 to 50000)
    log_file: Optional[str] = None  # Path to save reward statistics
    
    # ===== NEW: RL Metrics Tracking =====
    track_reward_components: bool = True  # Track detailed breakdown (for TensorBoard)
    track_episode_stats: bool = True  # Track cumulative episode statistics
    
    # ===== Decay Schedules =====
    enable_decay: bool = False  # Decay shaping influence over time
    decay_start_step: int = 100000
    decay_end_step: int = 500000
    decay_final_weight: float = 0.1  # Final weight multiplier after decay


class RewardShapingWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds theory-based reward shaping.
    
    This wrapper modifies the reward signal based on pursuit-evasion theory
    to guide learning. It can be used with or without the EnrichedObservationWrapper.
    
    The wrapper computes potential functions based on theoretical features
    and applies potential-based shaping: R' = R + γ*Φ(s') - Φ(s)
    
    This preserves the optimal policy while accelerating learning.
    
    NEW: Comprehensive reward component tracking for RL analysis and thesis metrics.
    """
    def _find_wrapper_in_chain(self, start_env: gym.Env, predicate, max_depth: int = 50):
        """
        Walk down .env chain and return the first wrapper/env that satisfies predicate.
        Works with Gymnasium wrappers that expose `.env`.
        """
        cur = start_env
        for _ in range(max_depth):
            if predicate(cur):
                return cur
            nxt = getattr(cur, "env", None)
            if nxt is None:
                return None
            cur = nxt
        return None


    def __init__(self,
                 env: gym.Env,
                 config: Optional[RewardShapingConfig] = None,
                 feature_computer: Optional[Any] = None,
                 obs_labels: Optional[Dict[str, int]] = None,
                 debug: bool = False):
        """
        Initialize reward shaping wrapper.
        
        Args:
            env: Base environment (can be EnrichedObservationWrapper or raw env)
            config: RewardShapingConfig with component settings
            feature_computer: PursuitEvasionFeatures instance (optional if using EnrichedWrapper)
            obs_labels: Observation label mapping (needed if not using EnrichedWrapper)
            debug: Enable debug output
        """
        super().__init__(env)
        
        self.config = config or RewardShapingConfig()
        self.feature_computer = feature_computer
        self.obs_labels = obs_labels or {}
        self.debug = debug
        
        # CRITICAL: Pass through obs_map from inner environment for FSM wrapper
        if hasattr(env, 'obs_map'):
            self.obs_map = env.obs_map
        if hasattr(env, 'observation_labels_flat'):
            self.observation_labels_flat = env.observation_labels_flat
        
        # Track previous state for potential-based shaping
        self.previous_potential = 0.0
        self.previous_features = None
        
        # Statistics tracking with fixed-size rolling window
        self.step_count = 0
        self.episode_count = 0
        self.rolling_window_size = 10000  # Keep only last 10k steps to prevent memory leak
        self.reward_components = defaultdict(lambda: deque(maxlen=self.rolling_window_size))
        self.prev_blue_missiles_fired = 0
        
        # NEW: Per-step reward breakdown (for TensorBoard logging)
        self.last_reward_breakdown = {
            'base_reward': 0.0,
            'bez_shaping': 0.0,
            'dmc_shaping': 0.0,
            'heading_shaping': 0.0,
            'distance_shaping': 0.0,
            'ttc_shaping': 0.0,
            'multithreat_shaping': 0.0,
            'speed_shaping': 0.0,
            'potential_diff': 0.0,
            'total_shaping': 0.0,
            'total_reward': 0.0,
            'decay_multiplier': 1.0,
            'hvaa_escort_shaping': 0.0,
            'offense_wez_shaping': 0.0,
            'offense_ttc_shaping': 0.0,
            'missile_launch_shaping': 0.0,
        }
        
        # NEW: Episode-level cumulative statistics
        self.episode_reward_stats = {
            'base_reward_sum': 0.0,
            'bez_shaping_sum': 0.0,
            'dmc_shaping_sum': 0.0,
            'heading_shaping_sum': 0.0,
            'distance_shaping_sum': 0.0,
            'ttc_shaping_sum': 0.0,
            'multithreat_shaping_sum': 0.0,
            'speed_shaping_sum': 0.0,
            'potential_diff_sum': 0.0,
            'total_shaping_sum': 0.0,
            'total_reward_sum': 0.0,
            'step_count': 0,
            'hvaa_escort_shaping_sum': 0.0,
            'offense_wez_shaping_sum': 0.0,
            'offense_ttc_shaping_sum': 0.0,
            'missile_launch_shaping_sum': 0.0,  
        }
        
        # Check if we're wrapping an EnrichedObservationWrapper
        self._enriched_env = self._find_wrapper_in_chain(
            env,
            lambda w: hasattr(w, "feature_computer") and hasattr(w, "num_added_features")
        )

        self._is_enriched_wrapper = self._enriched_env is not None
        if self._is_enriched_wrapper:
            self.feature_computer = self._enriched_env.feature_computer
            if self.debug:
                print("[RewardShaping] Found EnrichedObservationWrapper in chain; using its feature computer")
        else:
            if self.debug:
                print("[RewardShaping] EnrichedObservationWrapper NOT found in chain; reward shaping will use raw obs only")
        

                # ===== NEW: HVAA Features Extraction =====
        self._hvaa_indices = None
        self._hvaa_agent_name = None

        # Walk down through wrappers to find the base env
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        # If the base env is B_ACE_GodotPettingZooWrapper, it will
        # have get_hvaa_indices() and possible_agents
        if hasattr(base_env, "get_hvaa_indices"):
            try:
                # Your Godot wrapper’s get_hvaa_indices ignores agent_name and uses labels["101"]
                self._hvaa_agent_name = "agent_0"
                self._hvaa_indices = base_env.get_hvaa_indices(self._hvaa_agent_name)
                print("[RewardShaping] HVAA indices:", self._hvaa_indices)
            except Exception as e:
                print("[RewardShaping] Could not get HVAA indices:", e)
        else:
            print("[RewardShaping] WARNING: base_env has no HVAA helpers; type:", type(base_env))
        # ==== END HVAA Feature Hookup =====


        if self.debug:
            print(f"[RewardShaping] Initialized with config:")
            self._print_config()
            print(f"[RewardShaping] Reward component tracking: {self.config.track_reward_components}")
            print(f"[RewardShaping] Episode stats tracking: {self.config.track_episode_stats}")
    
    def _print_config(self):
        """Print current configuration."""
        print(f"  BEZ shaping: {self.config.enable_bez_shaping} (weight={self.config.bez_weight})")
        print(f"  DMC shaping: {self.config.enable_dmc_shaping} (weight={self.config.dmc_weight})")
        print(f"  Heading shaping: {self.config.enable_heading_shaping} (weight={self.config.heading_weight})")
        print(f"  Distance shaping: {self.config.enable_distance_shaping} (weight={self.config.distance_weight})")
        print(f"  TTC shaping: {self.config.enable_ttc_shaping} (weight={self.config.ttc_weight})")
        print(f"  Multi-threat shaping: {self.config.enable_multithreat_shaping} (weight={self.config.multithreat_weight})")
        print(f"  Speed shaping: {self.config.enable_speed_shaping} (weight={self.config.speed_weight})")
        print(f"  Offensive shaping: {self.config.enable_offense_shaping} "
              f"(wez={self.config.wez_dom_weight}, ttc={self.config.offense_ttc_weight})")
        print(f"  Potential-based: {self.config.use_potential_based}")
        print(f"  Logging: {self.config.enable_logging}")

    def _compute_missile_launch_shaping(self, f: Dict[str, float], info: Dict[str, Any]) -> float:
        """
        Penalize low-quality launches and mildly reward good-quality launches.

        We infer launches from the change in cumulative blue_missiles_fired:
            info["blue_missiles_fired_this_step"] = Δ missiles this step
        """
        if not self.config.enable_missile_launch_shaping:
            return 0.0

        missiles_this_step = int(info.get("blue_missiles_fired_this_step", 0))
        if missiles_this_step <= 0:
            return 0.0

        # Base cost for each missile fired
        reward = missiles_this_step * float(self.config.base_launch_cost)

        off_dom = float(f.get("offensive_dominance", 0.0))
        off_ttc = float(f.get("offensive_ttc_score", 0.0))

        # Decide if this was a "good" WEZ launch
        good_wez = (
            off_dom >= self.config.launch_off_dom_min
            and off_ttc >= self.config.launch_ttc_min
        )

        if good_wez:
            # Reward proportional to quality, capped in [0,1]
            quality = 0.5 * off_dom + 0.5 * off_ttc
            quality = float(np.clip(quality, 0.0, 1.0))
            reward += missiles_this_step * self.config.good_launch_bonus * quality
        else:
            reward += missiles_this_step * self.config.bad_launch_penalty

        return reward

    def _compute_offensive_ttc_reward(self, f):
        if not self.config.enable_offense_shaping:
            return 0.0
        off_ttc = float(f.get("offensive_ttc_score", 0.0))
        # Clip to sane range
        off_ttc = float(np.clip(off_ttc, -1.0, 1.0))
        if off_ttc > 0.0:
            # Reward helpful TTC (closing / favorable timing)
            return self.config.offense_ttc_weight * off_ttc
        # Negative: small penalty only
        return 0.5 * self.config.offense_ttc_weight * off_ttc
    
    def _compute_wez_dominance_reward(self, f: Dict[str, float]) -> float:
        """
        Compute reward based on offensive WEZ dominance.
        
        Rewards positive offensive_dominance (favorable weapon engagement zone position).
        """
        if not self.config.enable_offense_shaping:
            return 0.0
        
        off_dom = float(f.get("offensive_dominance", 0.0))
        # Clip to reasonable range [-1, 1]
        off_dom = float(np.clip(off_dom, -1.0, 1.0))
        
        if off_dom > 0.0:
            # Reward favorable WEZ position
            return self.config.wez_dom_weight * off_dom
        else:
            # Small penalty for unfavorable position
            return 0.5 * self.config.wez_dom_weight * off_dom
    
    def _extract_features_from_obs(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Extract relevant features from observation.
        
        Works with both enriched and raw observations.
        For EnrichedObservationWrapper, we assume the feature tail is ordered as:
          [ time_to_capture_norm, capture_feasible,
            bez_penetration, inside_bez, bez_aspect,
            dmc_normalized, inside_threat,
            evader_always_escapes, pursuer_always_captures,
            offensive_dominance, offensive_ttc_score ]
        (with some groups omitted if the corresponding flags are disabled).
        """
        features: Dict[str, float] = {}

        # Always grab some base geometry directly from the raw obs
        # (B-ACE layout: index 3 = own_dist_target, 4 = own_aspect_angle_target)
        if len(obs) > 3:
            features['dist_to_target'] = float(obs[3])
        if len(obs) > 4:
            features['aspect_to_target'] = float(obs[4])

        # If using EnrichedObservationWrapper, parse theoretical tail
        if self._is_enriched_wrapper:
            ew = self._enriched_env  # Enriched wrapper reference
            num_added = int(getattr(ew, "num_added_features", 0))
            if num_added <= 0:
                # Safety: if enriched exists but reports 0, treat as non-enriched
                return features

            feature_vec = obs[-num_added:]
            idx = 0

            # Apollonius-based features (defensive TTC)
            if getattr(ew, "enable_apollonius", False):
                if idx < len(feature_vec):
                    features['time_to_capture'] = float(feature_vec[idx])
                    idx += 1
                if idx < len(feature_vec):
                    features['capture_feasible'] = float(feature_vec[idx])
                    idx += 1

            # BEZ features
            if getattr(ew, "enable_bez", False):
                if idx < len(feature_vec):
                    features['bez_penetration'] = float(feature_vec[idx])
                    idx += 1
                if idx < len(feature_vec):
                    features['inside_bez'] = float(feature_vec[idx])
                    idx += 1
                if idx < len(feature_vec):
                    features['bez_aspect'] = float(feature_vec[idx])
                    idx += 1

            # DMC features
            if getattr(ew, "enable_dmc", False):
                if idx < len(feature_vec):
                    features['dmc_normalized'] = float(feature_vec[idx])
                    idx += 1
                if idx < len(feature_vec):
                    features['inside_threat'] = float(feature_vec[idx])
                    idx += 1

            # Escape feasibility (always computed by the enriched wrapper)
            if idx < len(feature_vec):
                features['evader_always_escapes'] = float(feature_vec[idx])
                idx += 1
            if idx < len(feature_vec):
                features['pursuer_always_captures'] = float(feature_vec[idx])
                idx += 1

            # Offensive features (WEZ dominance + offensive TTC)
            if getattr(ew, "enable_offense_wez", False):
                if idx < len(feature_vec):
                    features['offensive_dominance'] = float(feature_vec[idx])
                    idx += 1
            else:
                features.setdefault('offensive_dominance', 0.0)

            if getattr(ew, "enable_offense_ttc", False):
                if idx < len(feature_vec):
                    features['offensive_ttc_score'] = float(feature_vec[idx])
                    idx += 1
            else:
                features.setdefault('offensive_ttc_score', 0.0)

            # Placeholders to keep existing shaping code happy
            features.setdefault('speed_ratio', 1.0)
            features.setdefault('heading_error', 0.0)

        else:
            # Fall back to raw observation extraction (no enriched wrapper)
            if len(obs) > 3:
                features.setdefault('dist_to_target', float(obs[3]))
            if len(obs) > 4:
                features.setdefault('aspect_to_target', float(obs[4]))
            # sensible defaults
            features.setdefault('inside_bez', 0.0)
            features.setdefault('dmc_normalized', 0.0)
            features.setdefault('speed_ratio', 1.0)
            features.setdefault('heading_error', 0.0)
            features.setdefault('offensive_dominance', 0.0)
            features.setdefault('offensive_ttc_score', 0.0)

        # ===== inject HVAA features from main obs vector =====
        if self._hvaa_indices is not None:
            for label in ("hvaa_dist", "hvaa_alt_diff", "hvaa_angle_off", "hvaa_detected"):
                hidx = self._hvaa_indices.get(label)
                if hidx is not None and hidx < len(obs):
                    features[label] = float(obs[hidx])

        return features
    
    def _compute_bez_reward(self, features: Dict[str, float]) -> float:
        """Compute BEZ-based reward component."""
        if not self.config.enable_bez_shaping:
            return 0.0
        
        inside_bez = features.get('inside_bez', 0.0)
        bez_penetration = features.get('bez_penetration', 0.0)
        
        if inside_bez > 0.5:  # Inside BEZ
            reward = self.config.bez_penalty_inside * self.config.bez_weight
        else:  # Outside BEZ
            reward = self.config.bez_reward_outside * self.config.bez_weight
        
        return reward
    
    def _compute_dmc_reward(self, features: Dict[str, float]) -> float:
        """
        DMC shaping: small bonus for good geometry, small penalty for very bad,
        neutral band in the middle so DMC doesn't dominate behavior.
        """
        if not self.config.enable_dmc_shaping:
            return 0.0

        dmc = features.get("dmc_normalized", None)
        if dmc is None:
            return 0.0

        g = self.config.dmc_good_threshold
        b = self.config.dmc_bad_threshold

        if dmc <= g:
            shaped = self.config.dmc_good_reward
        elif dmc >= b:
            shaped = self.config.dmc_bad_penalty
        else:
            # Smooth interpolation between good and bad across the neutral band
            t = (dmc - g) / (b - g)  # t in (0,1)
            shaped = (1 - t) * self.config.dmc_good_reward + t * 0.0  # fade to ~0

        return self.config.dmc_weight * shaped
    '''
    def _compute_dmc_reward(self, features: Dict[str, float]) -> float:
        if not self.config.enable_dmc_shaping:
            return 0.0
        
        dmc = features.get('dmc_normalized', 0.0)
        
        # reward_low when dmc=0, penalty_high when dmc=1
        reward = (self.config.dmc_reward_low - 
              (self.config.dmc_reward_low - self.config.dmc_penalty_high) * dmc
             ) * self.config.dmc_weight
        

        return reward
    '''
    def _compute_heading_reward(self, features: Dict[str, float]) -> float:
        """Compute heading alignment reward."""
        if not self.config.enable_heading_shaping:
            return 0.0
        
        heading_error = abs(features.get('heading_error', 0.0))
        
        if heading_error < 0.3:  # Well aligned
            reward = self.config.heading_reward_aligned * self.config.heading_weight
        else:
            reward = 0.0
        
        return reward
    
    def _compute_distance_reward(self, features: Dict[str, float]) -> float:
        """Compute distance-based reward."""
        if not self.config.enable_distance_shaping:
            return 0.0
        
        dist = features.get('dist_to_target', 0.0)
        
        if dist > self.config.distance_safe_threshold:
            reward = 0.05 * self.config.distance_weight
        else:
            reward = -0.05 * self.config.distance_weight
        
        return reward
    
    def _compute_ttc_reward(self, features: Dict[str, float]) -> float:
        """Compute time-to-capture based reward."""
        if not self.config.enable_ttc_shaping:
            return 0.0
        
        ttc = features.get('time_to_capture', 100.0)
        
        if ttc > self.config.ttc_safe_threshold:
            reward = 0.05 * self.config.ttc_weight
        else:
            reward = -0.05 * self.config.ttc_weight
        
        return reward
    
    def _compute_multithreat_reward(self, features: Dict[str, float]) -> float:
        """Compute multi-threat based reward."""
        if not self.config.enable_multithreat_shaping:
            return 0.0
        
        # This would need safe_cone_width feature
        # Placeholder implementation
        return 0.0
    
    def _compute_speed_reward(self, features: Dict[str, float]) -> float:
        """Compute speed ratio based reward."""
        if not self.config.enable_speed_shaping:
            return 0.0
        
        speed_ratio = features.get('speed_ratio', 1.0)
        
        if speed_ratio > 1.0:  # Faster than threat
            reward = self.config.speed_advantage_bonus * self.config.speed_weight
        else:
            reward = 0.0
        
        return reward
    '''
    def _compute_hvaa_escort_reward(self, features: Dict[str, float]) -> float:
        """
        Example: encourage staying near the HVAA.
        Assumes 'hvaa_dist' is normalized distance ownship–HVAA.
        """
        if not self.config.enable_hvaa_escort_shaping:
            return 0.0

        hv_detected = features.get('hvaa_detected', 0.0)
        hv_dist = features.get('hvaa_dist', -1.0)

        if hv_detected < 0.5 or hv_dist < 0.0:
            return 0.0

        # Target a "sweet spot" distance (normalized); penalize being too far/too close
        desired = self.config.hvaa_desired_dist  # e.g., 0.1–0.2
        err = hv_dist - desired
        return -self.config.hvaa_escort_weight * (err ** 2)
    '''

    def _compute_hvaa_escort_reward(self, features: Dict[str, float]) -> float:
        if not self.config.enable_hvaa_escort_shaping:
            return 0.0
        
        hv_detected = features.get('hvaa_detected', 0.0)
        if hv_detected < 0.5:
            return 0.0
        
        hv_dist = features.get('hvaa_dist', -1.0)  # Use raw distance
        if hv_dist < 0:
            return 0.0
        
        # Normalize distance using B-ACE's 120nm scale
        # HVAA distance is in normalized units (0-1 range typically)
        # Target escort distance is ~0.35 (35% of max range, ~42nm)
        target = self.config.hvaa_target_norm
        
        # Compute error from ideal escort distance
        error = abs(hv_dist - target)
        
        # Small penalty when too far or too close
        if error > 0.1:  # More than 10% deviation
            reward = -self.config.hvaa_max_penalty * (error / 0.5)  # Scale by error
            reward = max(reward, self.config.hvaa_max_penalty)  # Clip
        else:
            reward = 0.01  # Small bonus for good positioning
        
        return self.config.hvaa_escort_weight * reward

    def _compute_potential(self, features: Dict[str, float]) -> float:
        """
        Compute potential function for potential-based shaping.
        
        Higher potential = better state
        """
        potential = 0.0
        
        # Distance from threats (farther = better)
        dist = features.get('dist_to_target', 0.0)
        potential += dist * 2.0
        
        # Outside BEZ is better
        inside_bez = features.get('inside_bez', 0.0)
        potential += (1.0 - inside_bez) * 1.5
        
        # Low DMC is better
        dmc = features.get('dmc_normalized', 0.0)
        potential += (1.0 - dmc) * 1.0
        
        return potential
    
    def _compute_shaped_reward(
        self,
        base_reward: float,
        features: Dict[str, float],
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward with all components.

        Returns:
            env_reward: scalar reward actually returned to the agent
            components: dictionary of all reward components for logging
        """
        components: Dict[str, float] = {}
        components['base_reward'] = float(base_reward)

        # ----- Individual shaping components -----
        bez_reward            = self._compute_bez_reward(features)
        dmc_reward            = self._compute_dmc_reward(features)
        heading_reward        = self._compute_heading_reward(features)
        distance_reward       = self._compute_distance_reward(features)
        ttc_reward            = self._compute_ttc_reward(features)
        multithreat_reward    = self._compute_multithreat_reward(features)
        speed_reward          = self._compute_speed_reward(features)
        hvaa_reward           = self._compute_hvaa_escort_reward(features)
        wez_offense_reward    = self._compute_wez_dominance_reward(features)
        offensive_ttc_reward  = self._compute_offensive_ttc_reward(features)
        missile_launch_reward = self._compute_missile_launch_shaping(features, info)

        components['bez']            = float(bez_reward)
        components['dmc']            = float(dmc_reward)
        components['heading']        = float(heading_reward)
        components['distance']       = float(distance_reward)
        components['ttc']            = float(ttc_reward)
        components['multithreat']    = float(multithreat_reward)
        components['speed']          = float(speed_reward)
        components['hvaa_escort']    = float(hvaa_reward)
        components['offense_wez']    = float(wez_offense_reward)
        components['offense_ttc']    = float(offensive_ttc_reward)
        components['missile_launch'] = float(missile_launch_reward)

        # ----- Optional decay on shaping -----
        decay_multiplier = 1.0
        if self.config.enable_decay and self.step_count > self.config.decay_start_step:
            progress = min(
                1.0,
                (self.step_count - self.config.decay_start_step)
                / (self.config.decay_end_step - self.config.decay_start_step),
            )
            decay_multiplier = 1.0 - progress * (1.0 - self.config.decay_final_weight)

        shaping_bonus = (
            bez_reward + dmc_reward + heading_reward + distance_reward +
            ttc_reward + multithreat_reward + speed_reward + hvaa_reward +
            wez_offense_reward + offensive_ttc_reward + missile_launch_reward
        )
        shaping_bonus *= decay_multiplier

        # ----- Potential-based shaping -----
        potential_diff = 0.0
        if self.config.use_potential_based:
            current_potential = self._compute_potential(features)
            if self.previous_potential is not None:
                potential_diff = self.config.gamma * current_potential - self.previous_potential
                shaping_bonus += potential_diff
            self.previous_potential = current_potential

        components['potential_diff']  = float(potential_diff)
        components['shaping_total']   = float(shaping_bonus)
        components['decay_multiplier'] = float(decay_multiplier)

        # ===== Apply terminal mission outcome bonuses/penalties =====
        mission_adjustment = 0.0

        # Check for HVAA destruction (catastrophic failure)
        #if info.get('hvaa_destroyed', False):
        #    mission_adjustment = -50.0
        #    components['mission_failure_penalty'] = -50.0
        #    if self.debug:
        #        print(f"[RewardShaping] HVAA DESTROYED - penalty -50.0")

        # Check for red kills (mission success)
        red_killed = info.get('Red_Killed', 0)
        if red_killed >= 2:
            mission_adjustment += 30.0
            components['mission_success_bonus'] = 30.0
            if self.debug:
                print(f"[RewardShaping] {red_killed} threats eliminated - bonus +30.0")

        # Apply adjustments to base reward
        base_reward += mission_adjustment
        components['base_reward'] = float(base_reward)


        # ----- RAW total reward (what you care about for the thesis) -----
        raw_total_reward = base_reward + shaping_bonus
        components['raw_total_reward'] = float(raw_total_reward)

        # ----- Reward actually returned to the agent -----
        env_reward = raw_total_reward
        is_terminal = info.get('episode_over', False) or info.get('terminated', False)
        if self.config.normalize_rewards and not is_terminal:
            env_reward = float(np.clip(env_reward, -1.0, 1.0))
        components['env_reward'] = float(env_reward)

        return env_reward, components
    
    def _update_reward_breakdown(self, base_reward: float, components: Dict[str, float]):
        raw_total = float(components.get('raw_total_reward', 0.0))
        env_total = float(components.get('env_reward', 0.0))
        
        self.last_reward_breakdown = {
            'base_reward': float(components.get('base_reward', base_reward)),
            'bez_shaping': float(components.get('bez', 0.0)),
            'dmc_shaping': float(components.get('dmc', 0.0)),
            'heading_shaping': float(components.get('heading', 0.0)),
            'distance_shaping': float(components.get('distance', 0.0)),
            'ttc_shaping': float(components.get('ttc', 0.0)),
            'multithreat_shaping': float(components.get('multithreat', 0.0)),
            'speed_shaping': float(components.get('speed', 0.0)),
            'potential_diff': float(components.get('potential_diff', 0.0)),
            'total_shaping': float(components.get('shaping_total', 0.0)),
            
            # CRITICAL FIX: Add all three total fields
            'total_reward': raw_total,          # For backwards compatibility
            'raw_total_reward': raw_total,      # Unclipped
            'env_reward': env_total,            # Clipped
            
            'decay_multiplier': float(components.get('decay_multiplier', 1.0)),
            'hvaa_escort_shaping': float(components.get('hvaa_escort', 0.0)),
            'offense_wez_shaping': float(components.get('offense_wez', 0.0)),
            'offense_ttc_shaping': float(components.get('offense_ttc', 0.0)),
            'missile_launch_shaping': float(components.get('missile_launch', 0.0)),
        }
    
    def _accumulate_episode_stats(self, components: Dict[str, float]):
        """
        Accumulate reward components for episode-level statistics.
        """
        self.episode_reward_stats['base_reward_sum']       += components.get('base_reward', 0.0)
        self.episode_reward_stats['bez_shaping_sum']       += components.get('bez', 0.0)
        self.episode_reward_stats['dmc_shaping_sum']       += components.get('dmc', 0.0)
        self.episode_reward_stats['heading_shaping_sum']   += components.get('heading', 0.0)
        self.episode_reward_stats['distance_shaping_sum']  += components.get('distance', 0.0)
        self.episode_reward_stats['ttc_shaping_sum']       += components.get('ttc', 0.0)
        self.episode_reward_stats['multithreat_shaping_sum'] += components.get('multithreat', 0.0)
        self.episode_reward_stats['speed_shaping_sum']     += components.get('speed', 0.0)
        self.episode_reward_stats['potential_diff_sum']    += components.get('potential_diff', 0.0)
        self.episode_reward_stats['total_shaping_sum']     += components.get('shaping_total', 0.0)
        self.episode_reward_stats['total_reward_sum']      += components.get('raw_total_reward', 0.0)
        self.episode_reward_stats['hvaa_escort_shaping_sum'] += components.get('hvaa_escort', 0.0)
        self.episode_reward_stats['offense_wez_shaping_sum'] += components.get('offense_wez', 0.0)
        self.episode_reward_stats['offense_ttc_shaping_sum'] += components.get('offense_ttc', 0.0)
        self.episode_reward_stats['missile_launch_shaping_sum'] += components.get('missile_launch', 0.0)
        self.episode_reward_stats['step_count'] += 1
    
    def _reset_episode_stats(self):
        """Reset episode statistics for new episode."""
        self.episode_reward_stats = {
            'base_reward_sum': 0.0,
            'bez_shaping_sum': 0.0,
            'dmc_shaping_sum': 0.0,
            'heading_shaping_sum': 0.0,
            'distance_shaping_sum': 0.0,
            'ttc_shaping_sum': 0.0,
            'multithreat_shaping_sum': 0.0,
            'speed_shaping_sum': 0.0,
            'potential_diff_sum': 0.0,
            'total_shaping_sum': 0.0,
            'total_reward_sum': 0.0,
            'step_count': 0,
            'hvaa_escort_shaping_sum': 0.0,
            'offense_wez_shaping_sum': 0.0,
            'offense_ttc_shaping_sum': 0.0,
            'missile_launch_shaping_sum': 0.0,
        }
    
    def _log_statistics(self):
        """Log reward shaping statistics."""
        if not self.config.enable_logging:
            return
        
        if self.step_count % self.config.log_frequency != 0:
            return
        
        print(f"\n[RewardShaping] Statistics at step {self.step_count}:")
        print(f"  Episodes: {self.episode_count}")
        
        # Compute averages over rolling window
        for component, values in self.reward_components.items():
            if values:
                mean_val = np.mean(values)
                print(f"  {component}: {mean_val:.4f}")
        
        # Save to file if specified
        if self.config.log_file:
            self._save_log()
    
    def _save_log(self):
        """Save statistics to CSV file."""
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        rows = []
        max_len = max(len(v) for v in self.reward_components.values()) if self.reward_components else 0
        
        for i in range(max_len):
            row = {'step': i}
            for component, values in self.reward_components.items():
                row[component] = values[i] if i < len(values) else 0.0
            rows.append(row)
        
        # Write CSV
        if rows:
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and tracking variables."""
        obs, info = self.env.reset(**kwargs)
        
        
        self.previous_potential = 0.0
        self.previous_features = None
        self.episode_count += 1
        self.prev_blue_missiles_fired = int(info.get("blue_missiles_fired", 0))
        
        # Reset episode stats
        self._reset_episode_stats()
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take step with reward shaping and comprehensive tracking.
        
        Returns observations, shaped reward, termination flags, and info dict.
        The info dict now contains:
        - 'reward_breakdown': Per-step component breakdown (if track_reward_components=True)
        - 'episode_reward_stats': Cumulative episode stats (if episode ended and track_episode_stats=True)
        """
        obs, base_reward, terminated, truncated, info = self.env.step(action)

         # ---- Missile usage tracking (Blue) ----
        # Env reports cumulative "blue_missiles_fired" (usually at episode end)
        current_blue_missiles = int(info.get("blue_missiles_fired", self.prev_blue_missiles_fired))
        missiles_fired_this_step = max(0, current_blue_missiles - self.prev_blue_missiles_fired)
        self.prev_blue_missiles_fired = current_blue_missiles

        # Expose per-step missile usage for shaping
        info["blue_missiles_fired_this_step"] = missiles_fired_this_step
        
        # ===== Standardize mission outcome keys =====
        if isinstance(info, dict):
            for key, value in info.items():
                if isinstance(value, dict):
                    # Standardize per-agent info
                    if 'hvaa_destroyed' not in value:
                        value['hvaa_destroyed'] = (
                            value.get('HVAA_Destroyed', False) or
                            value.get('hvaa_killed', False) or
                            False
                        )
                    if 'Red_Killed' not in value:
                        value['Red_Killed'] = (
                            value.get('red_killed', 0) or
                            value.get('enemies_killed', 0) or
                            0
                        )
            
            # Also set top-level keys
            if 'hvaa_destroyed' not in info:
                info['hvaa_destroyed'] = any(
                    v.get('hvaa_destroyed', False) 
                    for v in info.values() 
                    if isinstance(v, dict)
                )
            if 'Red_Killed' not in info:
                info['Red_Killed'] = max(
                    (v.get('Red_Killed', 0) for v in info.values() if isinstance(v, dict)),
                    default=0
                )

        # Debug logging on episode end
        if (terminated or truncated) and self.debug:
            print(f"\n[RewardShaping] Episode ended:")
            print(f"  hvaa_destroyed: {info.get('hvaa_destroyed', 'N/A')}")
            print(f"  Red_Killed: {info.get('Red_Killed', 'N/A')}")

        # Extract features
        features = self._extract_features_from_obs(obs)
        
        # Compute shaped reward with component breakdown
        shaped_reward, components = self._compute_shaped_reward(base_reward, features, info)
        
        # Track statistics for rolling window (only if logging enabled)
        self.step_count += 1
        if self.config.enable_logging:
            for component, value in components.items():
                self.reward_components[component].append(value)
        
        # NEW: Update per-step reward breakdown
        if self.config.track_reward_components:
            self._update_reward_breakdown(base_reward, components)
            info['reward_breakdown'] = self.last_reward_breakdown.copy()
        
        # NEW: Accumulate episode statistics
        if self.config.track_episode_stats:
            self._accumulate_episode_stats(components)
            
            # Add episode stats to info when episode ends
            if terminated or truncated:
                # Calculate averages
                steps = self.episode_reward_stats['step_count']
                if steps > 0:
                    episode_stats = self.episode_reward_stats.copy()
                    
                    # Add averages
                    episode_stats['avg_base_reward'] = episode_stats['base_reward_sum'] / steps
                    episode_stats['avg_bez_shaping'] = episode_stats['bez_shaping_sum'] / steps
                    episode_stats['avg_dmc_shaping'] = episode_stats['dmc_shaping_sum'] / steps
                    episode_stats['avg_total_shaping'] = episode_stats['total_shaping_sum'] / steps
                    episode_stats['avg_total_reward'] = episode_stats['total_reward_sum'] / steps
                    
                    # Calculate shaping contribution percentage
                    total_r = episode_stats['total_reward_sum']
                    if abs(total_r) > 1e-6:
                        episode_stats['shaping_contribution_pct'] = (
                            episode_stats['total_shaping_sum'] / total_r * 100.0
                        )
                    else:
                        episode_stats['shaping_contribution_pct'] = 0.0
                    
                    info['episode_reward_stats'] = episode_stats
        
        # Log periodically
        self._log_statistics()
        
        # Add debug info if enabled
        if self.debug:
            info['reward_components'] = components
            info['shaped_reward'] = shaped_reward
            info['base_reward'] = base_reward
        
        return obs, shaped_reward, terminated, truncated, info


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_config(**overrides) -> RewardShapingConfig:
    """
    Create a default configuration with optional overrides.
    
    Example:
        config = create_default_config(
            enable_bez_shaping=True,
            bez_weight=0.7,
            enable_dmc_shaping=False
        )
    """
    config = RewardShapingConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_minimal_config() -> RewardShapingConfig:
    """Create a minimal config with only essential shaping enabled."""
    return RewardShapingConfig(
        enable_bez_shaping=True,
        enable_dmc_shaping=True,
        enable_heading_shaping=False,
        enable_distance_shaping=False,
        enable_ttc_shaping=False,
        enable_multithreat_shaping=False,
        enable_speed_shaping=False,
        bez_weight=0.5,
        dmc_weight=0.5,
        track_reward_components=True,  # Enable for thesis metrics
        track_episode_stats=True,
    )


def create_aggressive_config() -> RewardShapingConfig:
    """Create config with all shaping components enabled and high weights."""
    return RewardShapingConfig(
        enable_bez_shaping=True,
        enable_dmc_shaping=True,
        enable_heading_shaping=True,
        enable_distance_shaping=True,
        enable_ttc_shaping=True,
        enable_multithreat_shaping=True,
        enable_speed_shaping=True,
        bez_weight=0.8,
        dmc_weight=0.7,
        heading_weight=0.5,
        distance_weight=0.6,
        ttc_weight=0.6,
        multithreat_weight=0.5,
        speed_weight=0.3,
        track_reward_components=True,
        track_episode_stats=True,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("RewardShapingWrapper - REVISED with RL Metrics Tracking")
    print("=" * 60)
    
    # Create a test config with tracking enabled
    config = create_minimal_config()
    config.enable_logging = True
    config.log_frequency = 10
    config.track_reward_components = True
    config.track_episode_stats = True
    
    print("\nConfiguration:")
    print(f"  BEZ shaping: {config.enable_bez_shaping}")
    print(f"  DMC shaping: {config.enable_dmc_shaping}")
    print(f"  Potential-based: {config.use_potential_based}")
    print(f"  Track reward components: {config.track_reward_components}")
    print(f"  Track episode stats: {config.track_episode_stats}")
    
    print("\nNEW FEATURES FOR RL METRICS:")
    print("  1. Per-step reward breakdown in info['reward_breakdown']")
    print("  2. Episode stats in info['episode_reward_stats'] (on episode end)")
    print("  3. Compatible with custom training callbacks")
    print("  4. Automatic TensorBoard logging via callbacks")
    
    print("\nTo use in training:")
    print("  1. Create config: config = create_minimal_config()")
    print("  2. Wrap env: env = RewardShapingWrapper(env, config)")
    print("  3. Add RewardComponentLogger callback to model.learn()")
    print("  4. View metrics in TensorBoard!")
    
    print("\nWrapper ready for thesis RL analysis!")
