#!/usr/bin/env python3
"""
enriched_observation_wrapper_ablation_fixed.py

FIXED VERSION of the ablation wrapper addressing critical issues:
1. Double-normalization with VecNormalize
2. Bimodal distributions from no-threat padding
3. Feature scale heterogeneity

Key changes from original ablation wrapper:
- normalize_features defaults to False (avoids double-norm with VecNormalize)
- Option to EXCLUDE enriched features from VecNormalize
- Smooth interpolation instead of hard padding when threat lost
- Separate feature groups with explicit scaling control
- Observation space bounds adjust based on normalization setting

Feature Categories (for ablation studies):
- GEOMETRY: Apollonius + ATDDG (Weintraub et al. 2020)
- ENGAGEMENT: BEZ + DMC + WEZ (Von Moll & Weintraub 2024)
- RANGE_LIMITED: Critical escape heading + capture probability (Weintraub et al. 2023)

Usage:
    from enriched_observation_wrapper_ablation_fixed import (
        EnrichedObservationWrapper, 
        FeatureCategory,
        SplitNormalizationWrapper
    )
    
    # Standard usage with split normalization (RECOMMENDED):
    env = SingleAgentBACEEnv(...)
    env = EnrichedObservationWrapper(
        env, 
        obs_labels=obs_map,
        normalize_features=False,  # Let VecNormalize handle base obs only
        smooth_threat_loss=True,   # Smooth transitions when threat lost
    )
    env = SplitNormalizationWrapper(env, base_obs_dim=env.original_obs_dim)
    
    # For ablation studies:
    env = EnrichedObservationWrapper(
        env, 
        ablation_config='geometry_only',
        normalize_features=False,
        smooth_threat_loss=True,
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
    FIXED Gymnasium wrapper that adds theoretical pursuit-evasion features
    to the observation space, with proper VecNormalize interaction.
    
    Key improvements over original:
    1. `normalize_features=False` default: Avoids double-normalization with VecNormalize
    2. `exclude_from_vecnorm=True`: Marks enriched features for exclusion from VecNormalize
    3. `smooth_threat_loss=True`: Smooth decay instead of hard padding when threat lost
    4. Provides `original_obs_dim` and `vecnorm_exclude_indices` for downstream use
    
    This wrapper:
    1. Receives raw observations from the Godot environment
    2. Extracts agent/threat states from the observation
    3. Computes theoretical features (Apollonius, BEZ, DMC, ATDDG, Range-Limited)
    4. Concatenates features to the observation
    5. Passes enriched observation to the RL agent
    """
    
    def __init__(self,
                 env: gym.Env,
                 obs_labels: Optional[Dict[str, int]] = None,
                 feature_config: Optional[Dict] = None,
                 enabled_categories: Optional[Set[FeatureCategory]] = None,
                 ablation_config: Optional[str] = None,
                 normalize_features: bool = False,  # CHANGED: Default False to avoid double-norm
                 exclude_from_vecnorm: bool = True,  # NEW: Tell VecNormalize to skip these
                 smooth_threat_loss: bool = True,    # NEW: Smooth interpolation on threat loss
                 threat_loss_decay: float = 0.9,     # NEW: Decay rate for smooth transitions
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
                            'geometry_only'). Overrides enabled_categories if set.
            normalize_features: If True, pre-normalize features to [-1,1]. 
                               Set False if using VecNormalize (avoids double-norm).
            exclude_from_vecnorm: If True, provides indices for VecNormalize to skip.
            smooth_threat_loss: If True, smoothly decay features when threat is lost
                               instead of jumping to padding values.
            threat_loss_decay: Decay factor per step when threat not detected (0-1).
            debug: Whether to print debug information
        """
        super().__init__(env)
        
        self.obs_labels = obs_labels or {}
        self.normalize_features = normalize_features
        self.exclude_from_vecnorm = exclude_from_vecnorm
        self.smooth_threat_loss = smooth_threat_loss
        self.threat_loss_decay = threat_loss_decay
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
        
        # Store original observation space info (CRITICAL for split normalization)
        self.original_obs_dim = env.observation_space.shape[0]
        
        # Build feature metadata for smooth decay targets
        self.feature_metadata = self._build_feature_metadata()
        
        # For smooth threat-loss transitions
        self._previous_features = None
        self._threat_was_detected = False
        
        # Modify observation space to include new features
        self._setup_observation_space()
        
        # Provide indices for VecNormalize exclusion
        if self.exclude_from_vecnorm:
            self.vecnorm_exclude_indices = list(range(
                self.original_obs_dim, 
                self.original_obs_dim + self.num_added_features
            ))
        
        # For reward shaping (optional)
        self.previous_features = None
        self.enable_reward_shaping = config.get('enable_reward_shaping', False)
        self.shaping_coefficient = config.get('shaping_coefficient', 0.1)
        
        if self.debug:
            print(f"[EnrichedWrapper] Config: {self.ablation_config_name}")
            print(f"[EnrichedWrapper] Enabled categories: {[c.name for c in self.enabled_categories]}")
            print(f"[EnrichedWrapper] Features: {self.get_feature_names()}")
            print(f"[EnrichedWrapper] Num added features: {self.num_added_features}")
            print(f"[EnrichedWrapper] Original obs dim: {self.original_obs_dim}")
            print(f"[EnrichedWrapper] New obs shape: {self.observation_space.shape}")
            print(f"[EnrichedWrapper] Pre-normalize: {self.normalize_features}")
            print(f"[EnrichedWrapper] Smooth threat loss: {self.smooth_threat_loss}")
            if self.exclude_from_vecnorm:
                print(f"[EnrichedWrapper] VecNorm exclude indices: {self.vecnorm_exclude_indices}")
    
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
    
    def _build_feature_metadata(self) -> List[Dict]:
        """
        Build metadata about each feature group for better control.
        
        Returns list of dicts with:
        - name: feature group name
        - count: number of features in group  
        - feature_type: 'continuous' or 'binary'
        - raw_range: expected range before normalization
        - neutral_value: value to use when no threat (for smooth decay target)
        """
        metadata = []
        
        if self.enable_apollonius:
            metadata.append({
                'name': 'time_to_capture',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (0, 10),
                'neutral_value': 1.0,  # max normalized = safe
            })
            metadata.append({
                'name': 'capture_feasible',
                'count': 1,
                'feature_type': 'binary',
                'raw_range': (0, 1),
                'neutral_value': 0.0,  # not feasible = safe
            })
        
        if self.enable_bez:
            metadata.append({
                'name': 'bez_penetration',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (-1, 1),
                'neutral_value': -0.5,  # slightly outside = safe
            })
            metadata.append({
                'name': 'inside_bez',
                'count': 1,
                'feature_type': 'binary',
                'raw_range': (0, 1),
                'neutral_value': 0.0,  # not inside = safe
            })
        
        if self.enable_dmc:
            metadata.append({
                'name': 'dmc_normalized',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (-1, 1),
                'neutral_value': 0.0,  # no maneuver needed
            })
            metadata.append({
                'name': 'inside_threat',
                'count': 1,
                'feature_type': 'binary',
                'raw_range': (0, 1),
                'neutral_value': 0.0,  # not inside threat
            })
        
        if self.enable_atddg:
            metadata.append({
                'name': 'defense_time_ratio',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (0, 1),
                'neutral_value': 0.25,  # slightly favorable defense ratio
            })
            metadata.append({
                'name': 'in_escape_region',
                'count': 1,
                'feature_type': 'binary',
                'raw_range': (0, 1),
                'neutral_value': 1.0,  # defensible
            })
            metadata.append({
                'name': 'barrier_value_normalized',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (-1, 1),
                'neutral_value': 0.0,  # neutral barrier value
            })
            metadata.append({
                'name': 'heading_to_optimal_intercept',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (-1, 1),
                'neutral_value': 0.0,  # no heading error
            })
        
        if self.enable_offense_wez:
            metadata.append({
                'name': 'offensive_dominance',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (-1, 1),
                'neutral_value': 0.0,  # neutral dominance
            })
        
        if self.enable_range_limited:
            metadata.append({
                'name': 'escape_cone_normalized',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (0, 1),
                'neutral_value': 1.0,  # all headings safe
            })
            metadata.append({
                'name': 'capture_probability_proxy',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (0, 1),
                'neutral_value': 0.0,  # no capture probability
            })
        
        return metadata
    
    def _get_neutral_features(self) -> np.ndarray:
        """Get neutral feature values for smooth decay target."""
        neutral = []
        for meta in self.feature_metadata:
            neutral.extend([meta['neutral_value']] * meta['count'])
        return np.array(neutral, dtype=np.float32)
    
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
            
            # Use wider bounds if NOT pre-normalizing (let VecNormalize/SplitNorm handle it)
            if self.normalize_features:
                # Pre-normalized features are in [-1, 1]
                feature_low = np.full(self.num_added_features, -1.0, dtype=np.float32)
                feature_high = np.full(self.num_added_features, 1.0, dtype=np.float32)
            else:
                # Raw features may have wider range - use conservative bounds
                feature_low = np.full(self.num_added_features, -10.0, dtype=np.float32)
                feature_high = np.full(self.num_added_features, 10.0, dtype=np.float32)
            
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
            self.observation_space = original_space
            print("[EnrichedWrapper] Warning: Non-Box observation space, features not concatenated")
    
    def _extract_state_from_obs(self, obs: np.ndarray) -> Dict[str, Any]:
        """Extract structured state from raw B-ACE observation array."""
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
            # Extract HVAA state for ATDDG calculations
            'hvaa': {
                'distance': obs[IDX_HVAA_DIST],
                'altitude_diff': obs[IDX_HVAA_ALT_DIFF],
                'angle_off': obs[IDX_HVAA_ANGLE_OFF],
                'heading': obs[IDX_HVAA_HDG] * 2 * np.pi,
                'detected': obs[IDX_HVAA_DETECTED] > 0.5,
            }
        }

        # Extract threat (track 201) if detected
        if obs[IDX_TRACK_DETECTED] > 0.5 and obs[IDX_TRACK_DIST] > -0.99:
            track_dist = obs[IDX_TRACK_DIST]
            track_aspect = obs[IDX_TRACK_ASPECT]
            
            threat_bearing = own_heading_rad + track_aspect * np.pi
            
            threat_relative_x = track_dist * np.cos(threat_bearing)
            threat_relative_z = track_dist * np.sin(threat_bearing)
            
            threat_pos = state['agent']['position'] + np.array([threat_relative_x, threat_relative_z])
            
            threat = {
                'position': threat_pos,
                'relative_position': np.array([threat_relative_x, threat_relative_z]),
                'distance': track_dist,
                'aspect_angle': track_aspect,
                'speed': 1.0,
                'range': obs[IDX_TRACK_ENEMY_RMAX] if obs[IDX_TRACK_ENEMY_RMAX] > 0 else 0.5,
                'nez_range': obs[IDX_TRACK_ENEMY_NEZ],
                'capture_radius': 0.01,
                'threat_factor': obs[IDX_TRACK_THREAT_FACTOR],
                'detected': True
            }
            state['threats'].append(threat)
        
        return state
    
    def _compute_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Compute all enabled theoretical features with optional normalization.
        
        If normalize_features=True, features are pre-normalized to [-1, 1].
        If normalize_features=False, features are returned in their natural range
        (to be normalized by SplitNormalizationWrapper or VecNormalize).
        
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
        agent_velocity = agent['velocity']
        
        if threats:
            threat = threats[0]  # Primary threat
            threat_pos = np.array(threat['position'][:2])
            threat_speed = threat.get('speed', 1.0)
            threat_range = threat.get('range', 0.5)
            if threat_range <= 0:
                threat_range = 0.5
            capture_radius = threat.get('capture_radius', 0.01)
            
            mu = self.feature_computer.compute_speed_ratio(agent_speed, threat_speed)
            
            # =================================================================
            # GEOMETRY CATEGORY: Apollonius + ATDDG (Weintraub et al. 2020)
            # =================================================================
            
            if self.enable_apollonius:
                apollo = self.feature_computer.compute_apollonius_intercept(
                    threat_pos, agent_pos, agent_heading, mu
                )
                
                t_capture = apollo['time_to_capture']
                if self.normalize_features:
                    features.append(np.clip(t_capture / 10.0, 0, 1) if t_capture < float('inf') else 1.0)
                else:
                    features.append(t_capture if t_capture < float('inf') else 10.0)
                
                features.append(1.0 if apollo['capture_feasible'] else 0.0)
            
            # =================================================================
            # ENGAGEMENT CATEGORY: BEZ + DMC + WEZ (Von Moll & Weintraub 2024)
            # =================================================================
            
            if self.enable_bez:
                bez = self.feature_computer.compute_bez_penetration(
                    agent_pos, agent_heading, threat_pos, mu, threat_range, capture_radius
                )
                
                if self.normalize_features:
                    features.append(np.clip(bez['normalized_penetration'], -1, 1))
                else:
                    features.append(bez['normalized_penetration'])
                
                features.append(1.0 if bez['inside_bez'] else 0.0)
            
            if self.enable_dmc:
                dmc = self.feature_computer.compute_dmc(
                    agent_pos, agent_heading, threat_pos, mu, threat_range, capture_radius
                )
                
                if self.normalize_features:
                    features.append(np.clip(dmc['dmc_normalized'], -1, 1))
                else:
                    features.append(dmc['dmc_normalized'])
                
                features.append(1.0 if dmc['inside_threat'] else 0.0)
            
            # =================================================================
            # GEOMETRY CATEGORY: ATDDG (Active Target Defense)
            # =================================================================
            
            if self.enable_atddg:
                if hvaa.get('detected', False) and hvaa.get('distance', 0) > 0:
                    hvaa_bearing = agent_heading + hvaa['angle_off'] * np.pi
                    hvaa_pos = agent_pos + hvaa['distance'] * np.array([
                        np.cos(hvaa_bearing), 
                        np.sin(hvaa_bearing)
                    ])
                    hvaa_speed = 0.1
                    
                    atddg = self.feature_computer.compute_atddg_defense_features(
                        defender_pos=agent_pos,
                        defender_speed=agent_speed,
                        attacker_pos=threat_pos,
                        attacker_speed=threat_speed,
                        target_pos=hvaa_pos,
                        target_speed=hvaa_speed
                    )
                    
                    features.append(atddg['defense_time_ratio'])
                    features.append(atddg['in_escape_region'])
                    
                    alpha = hvaa_speed / threat_speed if threat_speed > 1e-6 else 0.1
                    barrier = self.feature_computer.compute_barrier_hyperbola_value(
                        attacker_pos=threat_pos,
                        target_pos=hvaa_pos,
                        defender_pos=agent_pos,
                        alpha=alpha
                    )
                    features.append(barrier['barrier_value_normalized'])
                    
                    opt_hdg = self.feature_computer.compute_optimal_intercept_heading(
                        defender_pos=agent_pos,
                        attacker_pos=threat_pos,
                        target_pos=hvaa_pos,
                        defender_heading=agent_heading
                    )
                    features.append(opt_hdg['heading_error_normalized'])
                else:
                    # No HVAA detected - use neutral/safe defaults
                    features.extend([0.25, 1.0, 0.0, 0.0])
            
            # =================================================================
            # ENGAGEMENT CATEGORY: Offensive WEZ dominance
            # =================================================================
            
            if self.enable_offense_wez:
                our_Rmax = godot.get('own_missile_rmax', threat_range)
                their_Rmax = threat_range

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

                if self.normalize_features:
                    features.append(np.clip(offense.get('offensive_dominance', 0.0), -1.0, 1.0))
                else:
                    features.append(offense.get('offensive_dominance', 0.0))
            
            # =================================================================
            # RANGE-LIMITED CATEGORY (Weintraub et al. 2023)
            # =================================================================
            
            if self.enable_range_limited:
                threat_dist = threat.get('distance', 0.5)
                
                crit = self.feature_computer.compute_critical_escape_heading(
                    d=threat_dist,
                    mu=mu,
                    R=threat_range
                )
                features.append(crit['escape_cone_normalized'])
                
                cap_prob = self.feature_computer.compute_capture_probability_proxy(
                    d=threat_dist,
                    mu=mu,
                    R=threat_range
                )
                features.append(cap_prob)
            
            # Update threat tracking for smooth decay
            self._threat_was_detected = True
            
        else:
            # No threat detected - use smooth decay or neutral values
            pass  # Features will be set below based on smooth_threat_loss
        
        # Convert to numpy array
        computed_features = np.array(features, dtype=np.float32) if features else np.array([], dtype=np.float32)
        
        # Handle no-threat case with smooth decay
        if not threats:
            neutral = self._get_neutral_features()
            
            if self.smooth_threat_loss and self._previous_features is not None and self._threat_was_detected:
                # Smoothly decay previous features towards neutral
                computed_features = (
                    self.threat_loss_decay * self._previous_features + 
                    (1 - self.threat_loss_decay) * neutral
                )
                
                # Check if we've decayed enough to consider threat fully lost
                if np.allclose(computed_features, neutral, atol=0.05):
                    self._threat_was_detected = False
            else:
                computed_features = neutral
                self._threat_was_detected = False
        
        # Store for next step's smooth decay
        self._previous_features = computed_features.copy() if len(computed_features) > 0 else None
        
        return computed_features
    
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

        # One-time dimensional sanity check
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
        """
        if not self.enable_reward_shaping or self.previous_features is None:
            self.previous_features = current_features.copy()
            return base_reward
        
        current_potential = -current_features[1] if len(current_features) > 1 else 0
        previous_potential = -self.previous_features[1] if len(self.previous_features) > 1 else 0
        
        shaping_bonus = self.shaping_coefficient * (current_potential - previous_potential)
        
        self.previous_features = current_features.copy()
        
        return base_reward + shaping_bonus
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return enriched initial observation."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset smooth decay state
        self._previous_features = None
        self._threat_was_detected = False
        self.previous_features = None  # Reset shaping state
        
        enriched_obs = self._enrich_observation(obs)
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
            'original_obs_dim': self.original_obs_dim,
            'total_obs_dim': self.observation_space.shape[0],
            'normalize_features': self.normalize_features,
            'smooth_threat_loss': self.smooth_threat_loss,
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


# =============================================================================
# SPLIT NORMALIZATION WRAPPER
# =============================================================================

class SplitNormalizationWrapper(gym.Wrapper):
    """
    Apply different normalization to base vs enriched features.
    
    This wrapper:
    1. Applies running normalization ONLY to base observation features
    2. Passes enriched features through unchanged (they're already properly scaled)
    
    Use this INSTEAD of VecNormalize's norm_obs=True when using enriched observations.
    
    Usage:
        env = EnrichedObservationWrapper(env, normalize_features=False, ...)
        env = SplitNormalizationWrapper(env, base_obs_dim=env.original_obs_dim)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)  # CRITICAL
    """
    
    def __init__(self, env: gym.Env, 
                 base_obs_dim: int,
                 clip_obs: float = 10.0,
                 epsilon: float = 1e-8):
        """
        Args:
            env: Environment with enriched observations
            base_obs_dim: Number of dimensions in base observation (before enriched features)
            clip_obs: Clip normalized observations to [-clip_obs, clip_obs]
            epsilon: Small constant for numerical stability
        """
        super().__init__(env)
        
        self.base_obs_dim = base_obs_dim
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        
        # Running statistics for base features only
        self.obs_mean = np.zeros(base_obs_dim, dtype=np.float64)
        self.obs_var = np.ones(base_obs_dim, dtype=np.float64)
        self.count = 0
        
        self.training = True
    
    def _update_stats(self, obs: np.ndarray):
        """Update running mean/var for base features using Welford's algorithm."""
        if not self.training:
            return
        
        base_obs = obs[:self.base_obs_dim]
        
        self.count += 1
        delta = base_obs - self.obs_mean
        self.obs_mean += delta / self.count
        delta2 = base_obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.count
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize base features, pass through enriched features."""
        base_obs = obs[:self.base_obs_dim]
        enriched_obs = obs[self.base_obs_dim:]
        
        # Normalize base features
        normalized_base = (base_obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        normalized_base = np.clip(normalized_base, -self.clip_obs, self.clip_obs)
        
        # Concatenate with unchanged enriched features
        return np.concatenate([normalized_base, enriched_obs]).astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_stats(obs)
        return self._normalize_obs(obs), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._update_stats(obs)
        return self._normalize_obs(obs), reward, term, trunc, info
    
    def set_training(self, training: bool):
        """Set training mode (controls whether stats are updated)."""
        self.training = training
    
    def save(self, path: str):
        """Save normalization statistics."""
        np.savez(path, 
                 obs_mean=self.obs_mean, 
                 obs_var=self.obs_var,
                 count=self.count,
                 base_obs_dim=self.base_obs_dim)
    
    def load(self, path: str):
        """Load normalization statistics."""
        data = np.load(path)
        self.obs_mean = data['obs_mean']
        self.obs_var = data['obs_var']
        self.count = int(data['count'])


# =============================================================================
# FEATURE-ONLY WRAPPER (for ablation)
# =============================================================================

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
                        ablation_config: str = 'all',
                        use_split_normalization: bool = True,
                        enable_reward_shaping: bool = False,
                        debug: bool = False,
                        **feature_kwargs) -> gym.Env:
    """
    Factory function to create an enriched environment with proper normalization.
    
    Args:
        base_env: Your base gymnasium environment
        obs_labels: Observation label mapping from Godot
        enable_features: Whether to add theoretical features
        ablation_config: Which feature categories to enable ('all', 'none', 'geometry_only', etc.)
        use_split_normalization: Whether to use SplitNormalizationWrapper (recommended)
        enable_reward_shaping: Whether to apply reward shaping
        debug: Enable debug output
        **feature_kwargs: Additional configuration for features
        
    Returns:
        Wrapped environment ready for training
        
    Usage:
        env = create_enriched_env(
            base_env,
            ablation_config='geometry_engagement',
            use_split_normalization=True,
        )
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)  # CRITICAL
    """
    if not enable_features or ablation_config == 'none':
        return base_env
    
    feature_config = {
        'enable_reward_shaping': enable_reward_shaping,
        **feature_kwargs
    }
    
    env = EnrichedObservationWrapper(
        base_env,
        obs_labels=obs_labels,
        feature_config=feature_config,
        ablation_config=ablation_config,
        normalize_features=False,  # Let SplitNormalizationWrapper handle it
        smooth_threat_loss=True,
        debug=debug
    )
    
    if use_split_normalization:
        env = SplitNormalizationWrapper(env, base_obs_dim=env.original_obs_dim)
    
    return env


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("EnrichedObservationWrapper (FIXED) - Ablation Study Demo")
    print("=" * 60)
    
    # Print the ablation matrix for thesis documentation
    EnrichedObservationWrapper.print_ablation_matrix()
    
    # Create a dummy environment for testing
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-10, high=10, shape=(27,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        def reset(self, seed=None, options=None):
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
    print("\n" + "=" * 60)
    print("TESTING ABLATION CONFIGURATIONS (FIXED VERSION)")
    print("=" * 60)
    
    configs_to_test = ['all', 'none', 'geometry_only', 'engagement_only', 'range_limited_only', 'geometry_engagement']
    
    for config_name in configs_to_test:
        env = DummyEnv()
        wrapped_env = EnrichedObservationWrapper(
            env, 
            ablation_config=config_name,
            normalize_features=False,  # FIXED: Avoid double normalization
            smooth_threat_loss=True,   # FIXED: Smooth transitions
            debug=False
        )
        
        obs, _ = wrapped_env.reset()
        summary = wrapped_env.get_ablation_summary()
        
        print(f"\nConfig: {config_name}")
        print(f"  Obs shape: {obs.shape}")
        print(f"  Original dim: {summary['original_obs_dim']}")
        print(f"  Categories: {summary['enabled_categories']}")
        print(f"  Features ({summary['num_features']}): {summary['feature_names']}")
        print(f"  Normalize features: {summary['normalize_features']}")
        print(f"  Smooth threat loss: {summary['smooth_threat_loss']}")
    
    # Test with SplitNormalizationWrapper
    print("\n" + "=" * 60)
    print("TESTING WITH SPLIT NORMALIZATION")
    print("=" * 60)
    
    env = DummyEnv()
    env = EnrichedObservationWrapper(
        env,
        ablation_config='all',
        normalize_features=False,
        smooth_threat_loss=True,
        debug=False
    )
    print(f"After EnrichedWrapper: obs_dim={env.observation_space.shape[0]}, original_dim={env.original_obs_dim}")
    
    env = SplitNormalizationWrapper(env, base_obs_dim=env.original_obs_dim)
    obs, _ = env.reset()
    print(f"After SplitNorm: obs_shape={obs.shape}")
    
    # Simulate a few steps
    for i in range(5):
        obs, _, _, _, _ = env.step(np.zeros(4))
    print(f"After 5 steps: obs_mean[:5]={env.obs_mean[:5]}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED USAGE IN TRAINING SCRIPT")
    print("=" * 60)
    print("""
# In your make_env function:
def make_env(...):
    e = SingleAgentBACEEnv(...)
    
    if args.use_enriched_obs:
        e = EnrichedObservationWrapper(
            e,
            ablation_config=args.ablation_config,  # e.g., 'all', 'geometry_only'
            normalize_features=False,  # CRITICAL: Avoid double normalization
            smooth_threat_loss=True,   # CRITICAL: Smooth transitions
        )
        e = SplitNormalizationWrapper(e, base_obs_dim=e.original_obs_dim)
    
    # ... other wrappers ...
    return e

# After vectorizing:
vec_env = DummyVecEnv([make_env(...)])
vec_env = VecNormalize(
    vec_env,
    norm_obs=False,  # CRITICAL: Already handled by SplitNormalizationWrapper
    norm_reward=False,
)
""")
    
    print("\nFixed ablation wrapper demo complete!")
