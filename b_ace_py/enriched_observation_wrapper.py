#!/usr/bin/env python3
"""
enriched_observation_wrapper.py

Wrapper that enriches observations from the B-ACE Godot environment
with theoretical features from pursuit-evasion differential game theory.

This wrapper sits between your Godot environment and the RL agent,
adding computed features like:
- Apollonius circle geometry (time-to-capture estimates)
- Basic Engagement Zone (BEZ) penetration
- Dynamic Maneuvering Cue (DMC) 
- Active Target Defense (ATDDG) features for escort scenarios

Feature Categories (for ablation studies):
- APOLLONIUS: time_to_capture, capture_feasible
  Source: Weintraub et al. 2020 "An Introduction to Pursuit-Evasion Differential Games"
  
- BEZ_DMC: bez_penetration, inside_bez, dmc_normalized, inside_threat
  Source: Von Moll & Weintraub 2024 "Basic Engagement Zones"
          Von Moll & Weintraub (draft) "Dynamic Maneuvering Cue"
          
- ATDDG: defense_time_ratio, in_escape_region
  Source: Weintraub et al. 2020 "An Introduction to Pursuit-Evasion Differential Games" Section V
  
- WEZ: offensive_dominance
  Source: Von Moll & Weintraub 2024 "Basic Engagement Zones"

Usage:
    # In your training script, wrap the environment:
    from enriched_observation_wrapper import EnrichedObservationWrapper, FeatureCategory
    
    env = SingleAgentBACEEnv(...)
    env = EnrichedObservationWrapper(env, obs_labels=obs_map)
    
    # For ablation studies:
    env = EnrichedObservationWrapper(
        env, 
        enabled_categories=[FeatureCategory.APOLLONIUS, FeatureCategory.BEZ_DMC]
    )
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
try:
    from .pursuit_evasion_features import PursuitEvasionFeatures
except ImportError:
    from pursuit_evasion_features import PursuitEvasionFeatures


class FeatureCategory(Enum):
    """
    Feature categories for ablation studies.
    Each category corresponds to theoretical foundations from specific papers.
    
    Three independent categories:
    - GEOMETRY: Apollonius circle + ATDDG (Weintraub et al. 2020)
    - ENGAGEMENT: BEZ + DMC + WEZ (Von Moll & Weintraub 2024)
    - RANGE_LIMITED: Critical escape heading + capture probability (Weintraub et al. 2023)
    """
    GEOMETRY = auto()       # Weintraub et al. 2020 - Apollonius + ATDDG (intercept geometry)
    ENGAGEMENT = auto()     # Von Moll & Weintraub 2024 - BEZ + DMC + WEZ (engagement zones)
    RANGE_LIMITED = auto()  # Weintraub et al. 2023 - Critical escape heading, capture probability


# Predefined configurations for ablation studies
ABLATION_CONFIGS = {
    'all': {FeatureCategory.GEOMETRY, FeatureCategory.ENGAGEMENT, FeatureCategory.RANGE_LIMITED},
    'none': set(),
    'geometry_only': {FeatureCategory.GEOMETRY},
    'engagement_only': {FeatureCategory.ENGAGEMENT},
    'range_limited_only': {FeatureCategory.RANGE_LIMITED},
    'geometry_engagement': {FeatureCategory.GEOMETRY, FeatureCategory.ENGAGEMENT},
    'geometry_range': {FeatureCategory.GEOMETRY, FeatureCategory.RANGE_LIMITED},
    'engagement_range': {FeatureCategory.ENGAGEMENT, FeatureCategory.RANGE_LIMITED},
}


# Paper citations for thesis documentation
CATEGORY_CITATIONS = {
    FeatureCategory.GEOMETRY: {
        'paper': 'Weintraub, I.E., Pachter, M., & Garcia, E. (2020). "An Introduction to Pursuit-Evasion Differential Games." American Control Conference.',
        'features': ['time_to_capture', 'capture_feasible', 'defense_time_ratio', 'in_escape_region', 'barrier_value_normalized', 'heading_to_optimal_intercept'],
        'description': 'Apollonius circle geometry for intercept times/feasibility, Active Target Defense (ATDDG) for three-agent escort scenarios, barrier hyperbola from Game of Kind (Eq. 46-47), and optimal intercept heading from Game of Degree (Eq. 49).',
    },
    FeatureCategory.ENGAGEMENT: {
        'paper': 'Von Moll, A. & Weintraub, I.E. (2024). "Basic Engagement Zones." + "Dynamic Maneuvering Cue."',
        'features': ['bez_penetration', 'inside_bez', 'dmc_normalized', 'inside_threat', 'offensive_dominance'],
        'description': 'Engagement zone boundaries, evasive maneuver requirements (DMC), and mutual WEZ comparison.',
    },
    FeatureCategory.RANGE_LIMITED: {
        'paper': 'Weintraub, I.E., Von Moll, A., & Pachter, M. (2023). "Range-Limited Pursuit-Evasion."',
        'features': ['escape_cone_normalized', 'capture_probability_proxy'],
        'description': 'Critical escape heading ψ_E,crit from Eq. 34 defining safe heading boundaries, and continuous capture probability based on escape/capture thresholds from Eq. 25-27.',
    },
}


class EnrichedObservationWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds theoretical pursuit-evasion features
    to the observation space.
    
    This wrapper:
    1. Receives raw observations from the Godot environment
    2. Extracts agent/threat states from the observation
    3. Computes theoretical features (Apollonius, BEZ, DMC, ATDDG)
    4. Concatenates features to the observation
    5. Passes enriched observation to the RL agent
    
    The wrapper is designed to be:
    - Non-invasive: Doesn't modify the underlying environment
    - Configurable: Easy to enable/disable specific feature categories
    - Ablation-ready: Predefined configs for systematic experiments
    - Debuggable: Can log feature values for analysis
    """
    
    def __init__(self,
                 env: gym.Env,
                 obs_labels: Optional[Dict[str, int]] = None,
                 feature_config: Optional[Dict] = None,
                 enabled_categories: Optional[Set[FeatureCategory]] = None,
                 ablation_config: Optional[str] = None,
                 normalize_features: bool = True,
                 debug: bool = False):
        """
        Initialize the enriched observation wrapper.
        
        Args:
            env: The underlying gymnasium environment
            obs_labels: Dict mapping observation label names to indices
                       (from your B_ACE_GodotPettingZooWrapper.obs_map)
            feature_config: Configuration for feature computation parameters
            enabled_categories: Set of FeatureCategory enums to enable.
                               If None, all categories enabled.
            ablation_config: String key from ABLATION_CONFIGS (e.g., 'all', 'none', 
                            'apollonius_only'). Overrides enabled_categories if set.
            normalize_features: Whether to normalize features to [-1, 1]
            debug: Whether to print debug information
        """
        super().__init__(env)
        
        self.obs_labels = obs_labels or {}
        self.normalize_features = normalize_features
        self.debug = debug
        
        # CRITICAL: Pass through obs_map from inner environment so FSM wrapper can find it
        if hasattr(env, 'obs_map'):
            self.obs_map = env.obs_map
        if hasattr(env, 'observation_labels_flat'):
            self.observation_labels_flat = env.observation_labels_flat
        
        # Determine which feature categories to enable
        if ablation_config is not None:
            if ablation_config not in ABLATION_CONFIGS:
                raise ValueError(f"Unknown ablation_config '{ablation_config}'. "
                               f"Available: {list(ABLATION_CONFIGS.keys())}")
            self.enabled_categories = ABLATION_CONFIGS[ablation_config]
        elif enabled_categories is not None:
            self.enabled_categories = set(enabled_categories)
        else:
            # Default: all categories enabled
            self.enabled_categories = {FeatureCategory.GEOMETRY, FeatureCategory.ENGAGEMENT, FeatureCategory.RANGE_LIMITED}
        
        # Store config name for logging
        self.ablation_config_name = ablation_config or 'custom'
        
        # Initialize feature computer with config
        config = feature_config or {}
        self.feature_computer = PursuitEvasionFeatures(
            default_pursuer_speed=config.get('pursuer_speed', 1.0),
            default_evader_speed=config.get('evader_speed', 0.8),
            default_capture_radius=config.get('capture_radius', 0.1),
            default_pursuer_range=config.get('pursuer_range', float('inf')),
            normalize_distance=config.get('normalize_distance', 1000.0)
        )
        
        # Map categories to enable flags
        # GEOMETRY = Apollonius + ATDDG (Weintraub et al. 2020)
        # ENGAGEMENT = BEZ + DMC + WEZ (Von Moll & Weintraub 2024)
        # RANGE_LIMITED = Critical escape heading + capture probability (Weintraub et al. 2023)
        self.enable_apollonius = FeatureCategory.GEOMETRY in self.enabled_categories
        self.enable_atddg = FeatureCategory.GEOMETRY in self.enabled_categories
        self.enable_bez = FeatureCategory.ENGAGEMENT in self.enabled_categories
        self.enable_dmc = FeatureCategory.ENGAGEMENT in self.enabled_categories
        self.enable_offense_wez = FeatureCategory.ENGAGEMENT in self.enabled_categories
        self.enable_range_limited = FeatureCategory.RANGE_LIMITED in self.enabled_categories
        
        # Number of additional features we'll add
        self.num_added_features = self._calculate_num_features()
        
        # Modify observation space to include new features
        self._setup_observation_space()
        
        # For reward shaping (optional)
        self.previous_features = None
        self.enable_reward_shaping = config.get('enable_reward_shaping', False)
        self.shaping_coefficient = config.get('shaping_coefficient', 0.1)
        
        if self.debug:
            print(f"[EnrichedWrapper] Config: {self.ablation_config_name}")
            print(f"[EnrichedWrapper] Enabled categories: {[c.name for c in self.enabled_categories]}")
            print(f"[EnrichedWrapper] Features: {self.get_feature_names()}")
            print(f"[EnrichedWrapper] Num added features: {self.num_added_features}")
            print(f"[EnrichedWrapper] Original obs shape: {self.env.observation_space.shape}")
            print(f"[EnrichedWrapper] New obs shape: {self.observation_space.shape}")
    
    def _calculate_num_features(self) -> int:
        num = 0
        
        # Differential game theory COMPUTED features only
        if self.enable_apollonius:
            num += 2  # time_to_capture, capture_feasible
        
        if self.enable_bez:
            num += 2  # penetration, inside_bez (aspect_angle removed - already in raw obs)
        
        if self.enable_dmc:
            num += 2  # dmc_normalized, inside_threat
        
        # ATDDG features (Weintraub et al. 2020)
        if self.enable_atddg:
            num += 4  # defense_time_ratio, in_escape_region, barrier_value_normalized, heading_to_optimal_intercept
        
        if self.enable_offense_wez:
            num += 1
        
        # Range-Limited features (Weintraub et al. 2023)
        if self.enable_range_limited:
            num += 2  # escape_cone_normalized, capture_probability_proxy
        
        return num  # Returns 13 when all features enabled

    
    def _setup_observation_space(self):
        """Modify observation space to include additional features."""
        original_space = self.env.observation_space
        
        if isinstance(original_space, spaces.Box):
            # Extend the Box space
            original_shape = original_space.shape
            new_shape = (original_shape[0] + self.num_added_features,)
            
            # Determine bounds
            original_low = original_space.low
            original_high = original_space.high
            
            # New features are normalized to [-1, 1] or [0, 1]
            feature_low = np.full(self.num_added_features, -1.0, dtype=np.float32)
            feature_high = np.full(self.num_added_features, 1.0, dtype=np.float32)
            
            new_low = np.concatenate([original_low, feature_low])
            new_high = np.concatenate([original_high, feature_high])
            
            self.observation_space = spaces.Box(
                low=new_low,
                high=new_high,
                shape=new_shape,
                dtype=np.float32
            )
        else:
            # For Dict spaces or other types, store features separately
            # (would need more complex handling)
            self.observation_space = original_space
            print("[EnrichedWrapper] Warning: Non-Box observation space, features not concatenated")
    
    def _extract_state_from_obs(self, obs: np.ndarray) -> Dict[str, Any]:
        # ---- B-ACE observation indices (matches env_info["observation_labels"] for agent key 101) ----
        IDX_OWN_X = 0
        IDX_OWN_Z = 1
        IDX_OWN_ALT = 2
        IDX_OWN_DIST_TARGET = 3
        IDX_OWN_ASPECT = 4
        IDX_OWN_HDG = 5
        IDX_OWN_SPEED = 6
        IDX_OWN_MISSILES = 7
        IDX_OWN_MISSILE_INFLIGHT = 8

        # HVAA block
        IDX_HVAA_DIST = 9
        IDX_HVAA_ALT_DIFF = 10
        IDX_HVAA_ANGLE_OFF = 11
        IDX_HVAA_HDG = 12
        IDX_HVAA_DETECTED = 13

        # Track 201 block
        IDX_TRACK_ALT_DIFF = 14
        IDX_TRACK_ASPECT = 15
        IDX_TRACK_ANGLE_OFF = 16
        IDX_TRACK_DIST = 17
        IDX_TRACK_DIST2GO = 18
        IDX_TRACK_OWN_RMAX = 19
        IDX_TRACK_OWN_NEZ = 20
        IDX_TRACK_ENEMY_RMAX = 21
        IDX_TRACK_ENEMY_NEZ = 22
        IDX_TRACK_THREAT_FACTOR = 23
        IDX_TRACK_OFFENSIVE_FACTOR = 24
        IDX_TRACK_MISSILE_SUPPORT = 25
        IDX_TRACK_DETECTED = 26
        
        # Extract agent state
        # Note: Positions appear to be normalized. Heading is in normalized form too.
        # We'll convert heading from normalized [0,1] to radians [0, 2π] if needed
        own_hdg_normalized = obs[IDX_OWN_HDG]
        own_heading_rad = own_hdg_normalized * 2 * np.pi  # Convert if normalized to [0,1]
        
        own_speed_normalized = obs[IDX_OWN_SPEED]
        
        state = {
            'agent': {
                'position': np.array([obs[IDX_OWN_X], obs[IDX_OWN_Z]]),
                'altitude': obs[IDX_OWN_ALT],
                'velocity': np.array([
                    own_speed_normalized * np.cos(own_heading_rad),
                    own_speed_normalized * np.sin(own_heading_rad)
                ]),
                'heading': own_heading_rad,
                'heading_normalized': own_hdg_normalized,
                'speed': own_speed_normalized,
                'missiles': obs[IDX_OWN_MISSILES],
                'missile_in_flight': obs[IDX_OWN_MISSILE_INFLIGHT] > 0.5,
                'dist_to_target': obs[IDX_OWN_DIST_TARGET],
                'aspect_angle_to_target': obs[IDX_OWN_ASPECT],
            },
            'threats': [],
            # Store Godot's pre-computed features for comparison/use
            'godot_features': {
                'track_detected': obs[IDX_TRACK_DETECTED] > 0.5,
                'track_distance': obs[IDX_TRACK_DIST],
                'track_aspect_angle': obs[IDX_TRACK_ASPECT],
                'track_angle_off': obs[IDX_TRACK_ANGLE_OFF],
                'track_altitude_diff': obs[IDX_TRACK_ALT_DIFF],
                'own_missile_rmax': obs[IDX_TRACK_OWN_RMAX],
                'own_missile_nez': obs[IDX_TRACK_OWN_NEZ],
                'enemy_missile_rmax': obs[IDX_TRACK_ENEMY_RMAX],
                'enemy_missile_nez': obs[IDX_TRACK_ENEMY_NEZ],
                'threat_factor': obs[IDX_TRACK_THREAT_FACTOR],
                'offensive_factor': obs[IDX_TRACK_OFFENSIVE_FACTOR],
            },
            # NEW: Extract HVAA state for ATDDG calculations
            'hvaa': {
                'distance': obs[IDX_HVAA_DIST],
                'altitude_diff': obs[IDX_HVAA_ALT_DIFF],
                'angle_off': obs[IDX_HVAA_ANGLE_OFF],  # Angle from agent to HVAA
                'heading': obs[IDX_HVAA_HDG] * 2 * np.pi,  # Convert to radians
                'detected': obs[IDX_HVAA_DETECTED] > 0.5,
            }
        }

        # Extract threat (track 201) if detected
        # Note: track_dist_201 = -1 means no valid track
        if obs[IDX_TRACK_DETECTED] > 0.5 and obs[IDX_TRACK_DIST] > -0.99:
            # We don't have absolute threat position, but we can compute relative
            # Using agent position + distance + angle
            track_dist = obs[IDX_TRACK_DIST]
            track_aspect = obs[IDX_TRACK_ASPECT]
            
            # Estimate threat position relative to agent
            # Aspect angle is from agent's perspective
            threat_bearing = own_heading_rad + track_aspect * np.pi  # Convert if normalized
            
            # Since distances are normalized, we work in normalized space
            threat_relative_x = track_dist * np.cos(threat_bearing)
            threat_relative_z = track_dist * np.sin(threat_bearing)
            
            threat_pos = state['agent']['position'] + np.array([threat_relative_x, threat_relative_z])
            
            threat = {
                'position': threat_pos,
                'relative_position': np.array([threat_relative_x, threat_relative_z]),
                'distance': track_dist,
                'aspect_angle': track_aspect,
                'speed': 1.0,  # Assume similar speed (normalized)
                # Use Godot's missile range info for BEZ calculations
                'range': obs[IDX_TRACK_ENEMY_RMAX] if obs[IDX_TRACK_ENEMY_RMAX] > 0 else 0.5,
                'nez_range': obs[IDX_TRACK_ENEMY_NEZ],
                'capture_radius': 0.01,  # Small capture radius in normalized space
                'threat_factor': obs[IDX_TRACK_THREAT_FACTOR],
                'detected': True
            }
            state['threats'].append(threat)
        
        return state
    
    def _compute_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Compute all enabled theoretical features.
        
        Optimized for B-ACE BVR air combat:
        - Uses Godot's pre-computed features where available
        - Adds differential game theory features (Apollonius, BEZ, DMC, ATDDG)
        - All features normalized for neural network input
        
        Args:
            state: Extracted state dict with 'agent', 'threats', 'hvaa', and 'godot_features'
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        agent = state['agent']
        threats = state['threats']
        hvaa = state.get('hvaa', {})
        godot = state.get('godot_features', {})
        
        agent_pos = agent['position']
        agent_heading = agent['heading']
        agent_speed = agent['speed']
        agent_velocity = agent['velocity']  # Velocity vector [vx, vy]
        
        
        # =================================================================
        # DIFFERENTIAL GAME THEORY COMPUTED FEATURES
        # These add theoretical insights from the papers
        # =================================================================
        
        if threats:
            threat = threats[0]  # Primary threat
            threat_pos = np.array(threat['position'][:2])
            threat_speed = threat.get('speed', 1.0)
            
            # Use enemy missile RMax as the threat "range" for BEZ calculations
            # This is the key insight - the engagement zone IS the missile envelope
            threat_range = threat.get('range', 0.5)
            if threat_range <= 0:
                threat_range = 0.5  # Default if not available
            
            capture_radius = threat.get('capture_radius', 0.01)
            
            # Compute speed ratio for theoretical calculations
            # mu = evader/pursuer, here agent is evader, threat is pursuer
            mu = self.feature_computer.compute_speed_ratio(agent_speed, threat_speed)
            
            # Apollonius-based features
            if self.enable_apollonius:
                # For BVR, we compute from threat's perspective (threat pursuing agent)
                apollo = self.feature_computer.compute_apollonius_intercept(
                    threat_pos, agent_pos, agent_heading, mu
                )
                
                # Normalized time to capture (lower = more urgent)
                # Scale by reasonable max time
                t_capture = apollo['time_to_capture']
                features.append(np.clip(t_capture / 10.0, 0, 1) if t_capture < float('inf') else 1.0)
                
                # Capture feasibility (is threat able to intercept?)
                features.append(1.0 if apollo['capture_feasible'] else 0.0)
            
            # BEZ (Basic Engagement Zone) features
            if self.enable_bez:
                bez = self.feature_computer.compute_bez_penetration(
                    agent_pos, agent_heading, threat_pos, mu, threat_range, capture_radius
                )
                
                # BEZ penetration depth (positive = inside danger zone)
                features.append(np.clip(bez['normalized_penetration'], -1, 1))
                
                # Binary: inside engagement zone?
                features.append(1.0 if bez['inside_bez'] else 0.0)
                
                # NOTE: aspect_angle removed - already available in raw obs (IDX_TRACK_ASPECT)
            
            # DMC (Dynamic Maneuvering Cue) features
            if self.enable_dmc:
                dmc = self.feature_computer.compute_dmc(
                    agent_pos, agent_heading, threat_pos, mu, threat_range, capture_radius
                )
                
                # DMC value: how much turn needed to escape
                # This is the key "risk" indicator from the papers
                features.append(dmc['dmc_normalized'])
                
                # Inside threat zone requiring maneuver?
                features.append(1.0 if dmc['inside_threat'] else 0.0)
            
            # =================================================================
            # ATDDG (Active Target Defense) features
            # From: Weintraub et al. "An Introduction to P-E Differential Games"
            # Includes: defense_time_ratio, in_escape_region (existing)
            #           barrier_value_normalized, heading_to_optimal_intercept (new)
            # =================================================================
            if self.enable_atddg:
                # Estimate HVAA position from agent position + HVAA distance/angle
                if hvaa.get('detected', False) and hvaa.get('distance', 0) > 0:
                    # HVAA angle_off is relative to agent's heading
                    hvaa_bearing = agent_heading + hvaa['angle_off'] * np.pi
                    hvaa_pos = agent_pos + hvaa['distance'] * np.array([
                        np.cos(hvaa_bearing), 
                        np.sin(hvaa_bearing)
                    ])
                    hvaa_speed = 0.1  # HVAA is slow (normalized)
                    
                    atddg = self.feature_computer.compute_atddg_defense_features(
                        defender_pos=agent_pos,
                        defender_speed=agent_speed,
                        attacker_pos=threat_pos,
                        attacker_speed=threat_speed,
                        target_pos=hvaa_pos,
                        target_speed=hvaa_speed
                    )
                    
                    # defense_time_ratio: <0.5 means defender can intercept before attacker reaches HVAA
                    features.append(atddg['defense_time_ratio'])
                    
                    # in_escape_region: 1.0 if HVAA is geometrically defensible
                    features.append(atddg['in_escape_region'])
                    
                    # Enhanced ATDDG: Barrier hyperbola from Game of Kind (Eq. 46-47)
                    alpha = hvaa_speed / threat_speed if threat_speed > 1e-6 else 0.1
                    barrier = self.feature_computer.compute_barrier_hyperbola_value(
                        attacker_pos=threat_pos,
                        target_pos=hvaa_pos,
                        defender_pos=agent_pos,
                        alpha=alpha
                    )
                    features.append(barrier['barrier_value_normalized'])
                    
                    # Enhanced ATDDG: Optimal intercept heading from Game of Degree (Eq. 49)
                    opt_hdg = self.feature_computer.compute_optimal_intercept_heading(
                        defender_pos=agent_pos,
                        attacker_pos=threat_pos,
                        target_pos=hvaa_pos,
                        defender_heading=agent_heading
                    )
                    features.append(opt_hdg['heading_error_normalized'])
                else:
                    # No HVAA detected - use neutral/safe defaults
                    features.append(0.25)  # Slightly favorable defense ratio
                    features.append(1.0)   # Assume defensible
                    features.append(0.0)   # Neutral barrier value
                    features.append(0.0)   # No heading error
            
            # Offensive features (WEZ dominance only - offensive_ttc removed as redundant with defense_time_ratio)
            if self.enable_offense_wez:
                our_Rmax = godot.get('own_missile_rmax', threat_range)
                their_Rmax = threat_range

                # Compute threat velocity vector
                threat_relative = threat_pos - agent_pos
                threat_distance = np.linalg.norm(threat_relative)
                if threat_distance > 1e-6:
                    threat_direction = threat_relative / threat_distance
                else:
                    threat_direction = np.array([1.0, 0.0])
                
                threat_velocity = threat_direction * threat_speed

                offense = self.feature_computer.compute_offensive_metrics(
                    own_pos=agent_pos,
                    own_vel=agent_velocity,
                    tgt_pos=threat_pos,
                    tgt_vel=threat_velocity,
                    our_Rmax=our_Rmax,
                    their_Rmax=their_Rmax,
                )

                # offensive_dominance can be in [-1,1] naturally
                features.append(
                    np.clip(offense.get('offensive_dominance', 0.0), -1.0, 1.0)
                )
            
            # =================================================================
            # RANGE-LIMITED PURSUIT-EVASION features
            # From: Weintraub et al. "Range-Limited Pursuit-Evasion" (2023)
            # =================================================================
            if self.enable_range_limited:
                # Use threat's distance and range for Range-Limited analysis
                threat_dist = threat.get('distance', 0.5)
                
                # Critical escape heading and escape cone (Eq. 34)
                crit = self.feature_computer.compute_critical_escape_heading(
                    d=threat_dist,
                    mu=mu,
                    R=threat_range
                )
                # escape_cone_normalized: 1.0 = all headings safe, 0.0 = no safe headings
                features.append(crit['escape_cone_normalized'])
                
                # Continuous capture probability (Eq. 25-27)
                cap_prob = self.feature_computer.compute_capture_probability_proxy(
                    d=threat_dist,
                    mu=mu,
                    R=threat_range
                )
                features.append(cap_prob)
            
        else:
            # No threat detected - pad with safe values
            
            if self.enable_apollonius:
                features.extend([1.0, 0.0])  # time=max, not feasible
            
            if self.enable_bez:
                features.extend([0.0, 0.0])  # not penetrating, not inside
            
            if self.enable_dmc:
                features.extend([0.0, 0.0])  # no maneuver needed, not inside threat
            
            if self.enable_atddg:
                features.extend([0.25, 1.0, 0.0, 0.0])  # defense_ratio, in_escape, barrier, heading_error
            
            if self.enable_offense_wez:
                features.append(0.0)
            
            if self.enable_range_limited:
                features.extend([1.0, 0.0])  # all headings safe, no capture probability
        
        return np.array(features, dtype=np.float32)
    
    def _enrich_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Enrich observation with computed features.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            Enriched observation with theoretical features appended
        """
        # Handle dict observations (from Godot wrapper)
        if isinstance(obs, dict):
            if 'obs' in obs:
                raw_obs = np.array(obs['obs'], dtype=np.float32)
            else:
                # Try to extract observation array
                raw_obs = np.array(list(obs.values())[0], dtype=np.float32)
        else:
            raw_obs = np.array(obs, dtype=np.float32)
        
        # Extract state from observation
        state = self._extract_state_from_obs(raw_obs)
        
        # Compute features
        features = self._compute_features(state)
        
        # Pad/truncate features to match expected size
        if len(features) < self.num_added_features:
            features = np.pad(features, (0, self.num_added_features - len(features)))
        elif len(features) > self.num_added_features:
            features = features[:self.num_added_features]
        
        # Concatenate
        enriched_obs = np.concatenate([raw_obs, features])

        # One-time dimensional sanity check (prints regardless of self.debug)
        if not hasattr(self, "_printed_dims"):
            print(f"[EnrichedObs] base_dim={raw_obs.shape[0]}, "
                f"computed_len={len(features)}, "
                f"num_added_expected={self.num_added_features}, "
                f"final_dim={enriched_obs.shape[0]}")
            self._printed_dims = True
        
        if self.debug:
            print(f"[EnrichedWrapper] Raw obs shape: {raw_obs.shape}, Features: {features.shape}, Enriched: {enriched_obs.shape}")
        
        return enriched_obs
    
    def _compute_shaped_reward(self, 
                                base_reward: float,
                                current_features: np.ndarray) -> float:
        """
        Compute potential-based reward shaping.
        
        Uses theoretical features as potential function.
        Shaped reward = base_reward + γ*Φ(s') - Φ(s)
        
        Args:
            base_reward: Original reward from environment
            current_features: Current feature vector
            
        Returns:
            Shaped reward
        """
        if not self.enable_reward_shaping or self.previous_features is None:
            self.previous_features = current_features.copy()
            return base_reward
        
        # Use DMC as potential (lower DMC = better = higher potential)
        # Assuming DMC features are at specific indices
        # This is simplified - customize based on your feature order
        
        # Simple potential: negative of closest threat distance
        # (being farther from threats is better)
        current_potential = -current_features[1]  # closest_threat_distance (index 1)
        previous_potential = -self.previous_features[1]
        
        # Potential-based shaping (with discount factor = 1 for simplicity)
        shaping_bonus = self.shaping_coefficient * (current_potential - previous_potential)
        
        self.previous_features = current_features.copy()
        
        return base_reward + shaping_bonus
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return enriched initial observation."""
        obs, info = self.env.reset(**kwargs)
        enriched_obs = self._enrich_observation(obs)
        self.previous_features = None  # Reset shaping state
        return enriched_obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take step and return enriched observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        enriched_obs = self._enrich_observation(obs)
        
        # Optional reward shaping
        if self.enable_reward_shaping:
            features = enriched_obs[-self.num_added_features:]
            reward = self._compute_shaped_reward(reward, features)
        
        # Add features to info for logging/debugging
        if self.debug:
            info['theoretical_features'] = enriched_obs[-self.num_added_features:]
        
        return enriched_obs, reward, terminated, truncated, info
    
    def get_feature_names(self) -> List[str]:
        """Return names of enriched features for interpretability."""
        names = []
        if self.enable_apollonius:
            names.extend(['time_to_capture', 'capture_feasible'])
        if self.enable_bez:
            names.extend(['bez_penetration', 'inside_bez'])
        if self.enable_dmc:
            names.extend(['dmc_normalized', 'inside_threat'])
        if self.enable_atddg:
            names.extend(['defense_time_ratio', 'in_escape_region', 'barrier_value_normalized', 'heading_to_optimal_intercept'])
        if self.enable_offense_wez:
            names.append('offensive_dominance')
        if self.enable_range_limited:
            names.extend(['escape_cone_normalized', 'capture_probability_proxy'])
        return names
    
    def get_enabled_categories(self) -> List[str]:
        """Return list of enabled category names."""
        return [c.name for c in self.enabled_categories]
    
    def get_ablation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration for thesis documentation.
        
        Returns:
            Dict with config name, enabled categories, features, and citations
        """
        summary = {
            'config_name': self.ablation_config_name,
            'enabled_categories': self.get_enabled_categories(),
            'num_features': self.num_added_features,
            'feature_names': self.get_feature_names(),
            'citations': {}
        }
        
        for cat in self.enabled_categories:
            summary['citations'][cat.name] = CATEGORY_CITATIONS[cat]
        
        return summary
    
    @staticmethod
    def get_available_configs() -> Dict[str, Set[str]]:
        """Return all available ablation configurations."""
        return {name: {c.name for c in cats} for name, cats in ABLATION_CONFIGS.items()}
    
    @staticmethod
    def print_ablation_matrix():
        """Print ablation study matrix for thesis documentation."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY CONFIGURATION MATRIX")
        print("=" * 80)
        
        categories = list(FeatureCategory)
        header = f"{'Config':<20}" + "".join(f"{c.name:<15}" for c in categories) + "Features"
        print(header)
        print("-" * 80)
        
        for name, enabled in ABLATION_CONFIGS.items():
            row = f"{name:<20}"
            num_features = 0
            for cat in categories:
                if cat in enabled:
                    row += f"{'✓':<15}"
                    num_features += len(CATEGORY_CITATIONS[cat]['features'])
                else:
                    row += f"{'-':<15}"
            row += str(num_features)
            print(row)
        
        print("\n" + "=" * 80)
        print("FEATURE CATEGORIES (Paper Sources)")
        print("=" * 80)
        for cat, info in CATEGORY_CITATIONS.items():
            print(f"\n{cat.name}:")
            print(f"  Features: {info['features']}")
            print(f"  Paper: {info['paper']}")
            print(f"  Description: {info['description']}")


class FeatureOnlyWrapper(gym.Wrapper):
    """
    Alternative wrapper that REPLACES observations with just theoretical features.
    
    Use this if you want to train on pure theoretical features without
    raw Godot observations (for ablation studies or simplified scenarios).
    """
    
    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env)
        self.enriched_wrapper = EnrichedObservationWrapper(env, **kwargs)
        self.num_features = self.enriched_wrapper.num_added_features
        
        # Observation space is just the features
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_features,),
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        enriched_obs, info = self.enriched_wrapper.reset(**kwargs)
        features_only = enriched_obs[-self.num_features:]
        return features_only, info
    
    def step(self, action):
        enriched_obs, reward, term, trunc, info = self.enriched_wrapper.step(action)
        features_only = enriched_obs[-self.num_features:]
        return features_only, reward, term, trunc, info


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_enriched_env(base_env,
                        obs_labels: Dict = None,
                        enable_features: bool = True,
                        enable_reward_shaping: bool = False,
                        debug: bool = False,
                        **feature_kwargs) -> gym.Env:
    """
    Factory function to create an enriched environment.
    
    Args:
        base_env: Your base gymnasium environment
        obs_labels: Observation label mapping from Godot
        enable_features: Whether to add theoretical features
        enable_reward_shaping: Whether to apply reward shaping
        debug: Enable debug output
        **feature_kwargs: Additional configuration for features
        
    Returns:
        Wrapped environment
    """
    if not enable_features:
        return base_env
    
    feature_config = {
        'enable_reward_shaping': enable_reward_shaping,
        **feature_kwargs
    }
    
    return EnrichedObservationWrapper(
        base_env,
        obs_labels=obs_labels,
        feature_config=feature_config,
        debug=debug
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("EnrichedObservationWrapper - Ablation Study Demo")
    print("=" * 50)
    
    # Print the ablation matrix for thesis documentation
    EnrichedObservationWrapper.print_ablation_matrix()
    
    # Create a dummy environment for testing
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-10, high=10, shape=(27,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        def reset(self, seed=None, options=None):
            # Simulate B-ACE observation structure
            obs = np.zeros(27, dtype=np.float32)
            obs[0] = 0.5   # own_x
            obs[1] = 0.5   # own_z
            obs[5] = 0.25  # own_hdg (normalized)
            obs[6] = 0.8   # own_speed
            obs[9] = 0.3   # hvaa_dist
            obs[11] = 0.1  # hvaa_angle_off
            obs[13] = 1.0  # hvaa_detected
            obs[17] = 0.4  # track_dist
            obs[21] = 0.5  # enemy_rmax
            obs[26] = 1.0  # track_detected
            return obs, {}
        
        def step(self, action):
            obs = np.random.randn(27).astype(np.float32) * 0.1
            obs[13] = 1.0  # hvaa_detected
            obs[26] = 1.0  # track_detected
            obs[17] = 0.4  # track_dist
            obs[21] = 0.5  # enemy_rmax
            return obs, 0.0, False, False, {}
    
    # Test different ablation configurations
    print("\n" + "=" * 50)
    print("TESTING ABLATION CONFIGURATIONS")
    print("=" * 50)
    
    configs_to_test = ['all', 'none', 'geometry_only', 'engagement_only', 'range_limited_only', 'geometry_engagement']
    
    for config_name in configs_to_test:
        env = DummyEnv()
        wrapped_env = EnrichedObservationWrapper(
            env, 
            ablation_config=config_name,
            debug=False
        )
        
        obs, _ = wrapped_env.reset()
        summary = wrapped_env.get_ablation_summary()
        
        print(f"\nConfig: {config_name}")
        print(f"  Obs shape: {obs.shape}")
        print(f"  Categories: {summary['enabled_categories']}")
        print(f"  Features ({summary['num_features']}): {summary['feature_names']}")
    
    # Show available configs
    print("\n" + "=" * 50)
    print("AVAILABLE CONFIGURATIONS:")
    print("=" * 50)
    for name, cats in EnrichedObservationWrapper.get_available_configs().items():
        print(f"  {name}: {cats}")
    
    print("\nAblation demo complete!")
