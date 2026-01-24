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
- Multi-threat safe heading cones

Usage:
    # In your training script, wrap the environment:
    from enriched_observation_wrapper import EnrichedObservationWrapper
    
    env = SingleAgentBACEEnv(...)
    env = EnrichedObservationWrapper(env, obs_labels=obs_map)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple
from .pursuit_evasion_features import PursuitEvasionFeatures


class EnrichedObservationWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds theoretical pursuit-evasion features
    to the observation space.
    
    This wrapper:
    1. Receives raw observations from the Godot environment
    2. Extracts agent/threat states from the observation
    3. Computes theoretical features (Apollonius, BEZ, DMC, etc.)
    4. Concatenates features to the observation
    5. Passes enriched observation to the RL agent
    
    The wrapper is designed to be:
    - Non-invasive: Doesn't modify the underlying environment
    - Configurable: Easy to enable/disable specific features
    - Debuggable: Can log feature values for analysis
    """
    
    def __init__(self,
                 env: gym.Env,
                 obs_labels: Optional[Dict[str, int]] = None,
                 feature_config: Optional[Dict] = None,
                 normalize_features: bool = True,
                 debug: bool = False):
        """
        Initialize the enriched observation wrapper.
        
        Args:
            env: The underlying gymnasium environment
            obs_labels: Dict mapping observation label names to indices
                       (from your B_ACE_GodotPettingZooWrapper.obs_map)
            feature_config: Configuration for feature computation
            normalize_features: Whether to normalize features to [-1, 1]
            debug: Whether to print debug information
        """
        super().__init__(env)
        
        self.obs_labels = obs_labels or {}
        self.normalize_features = normalize_features
        self.debug = debug
        
        # CRITICAL: Pass through obs_map from inner environment so FSM wrapper can find it
        # This enables downstream wrappers to resolve observation indices correctly
        if hasattr(env, 'obs_map'):
            self.obs_map = env.obs_map
        if hasattr(env, 'observation_labels_flat'):
            self.observation_labels_flat = env.observation_labels_flat
        
        # Initialize feature computer with config
        config = feature_config or {}
        self.feature_computer = PursuitEvasionFeatures(
            default_pursuer_speed=config.get('pursuer_speed', 1.0),
            default_evader_speed=config.get('evader_speed', 0.8),
            default_capture_radius=config.get('capture_radius', 0.1),
            default_pursuer_range=config.get('pursuer_range', float('inf')),
            normalize_distance=config.get('normalize_distance', 1000.0)
        )
        
        # Configure which features to compute
        self.enable_apollonius = config.get('enable_apollonius', True)
        self.enable_bez = config.get('enable_bez', True)
        self.enable_dmc = config.get('enable_dmc', True)
        self.enable_multi_threat = config.get('enable_multi_threat', True)
        self.enable_offense_wez = config.get('enable_offense_wez', True)
        self.enable_offense_ttc = config.get('enable_offense_ttc', True)
        
        # Number of additional features we'll add
        self.num_added_features = self._calculate_num_features()
        
        # Modify observation space to include new features
        self._setup_observation_space()
        
        # For reward shaping (optional)
        self.previous_features = None
        self.enable_reward_shaping = config.get('enable_reward_shaping', False)
        self.shaping_coefficient = config.get('shaping_coefficient', 0.1)
        
        if self.debug:
            print(f"[EnrichedWrapper] Initialized with {self.num_added_features} additional features")
            print(f"[EnrichedWrapper] Original obs shape: {self.env.observation_space.shape}")
            print(f"[EnrichedWrapper] New obs shape: {self.observation_space.shape}")
    
    def _calculate_num_features(self) -> int:
        num = 0
        
        # Differential game theory COMPUTED features only
        if self.enable_apollonius:
            num += 2  # time_to_capture, capture_feasible (removed heading_error)
        
        if self.enable_bez:
            num += 3  # penetration, inside_bez, aspect_angle
        
        if self.enable_dmc:
            num += 2  # dmc_normalized, inside_threat
        
        num += 2  # escape feasibility (always computed)
        if self.enable_offense_wez:
            num += 1
        if self.enable_offense_ttc:
            num += 1
        # REMOVED: Multi-threat aggregate (2) - only used by disabled multithreat_shaping
        
        return num  # Returns 9 when all features enabled

    
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
            }
        }
        
        #DEBUG ==========================================
        #if self.debug or (np.random.random() < 0.01):
        #    print(f"[ExtractState] track_detected: {obs[IDX_TRACK_DETECTED]:.4f}")
        #    print(f"[ExtractState] track_distance: {obs[IDX_TRACK_DIST]:.4f}")
        #    print(f"[ExtractState] Will add threat: {obs[IDX_TRACK_DETECTED] > 0.5 and obs[IDX_TRACK_DIST] > -0.99}")

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
        - Adds differential game theory features (Apollonius, BEZ, DMC)
        - All features normalized for neural network input
        
        Args:
            state: Extracted state dict with 'agent', 'threats', and 'godot_features'
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        agent = state['agent']
        threats = state['threats']
    
        #if self.debug or (np.random.random() < 0.01):
        #    print(f"[EnrichedObs] threats detected: {len(threats)}")
        #    if threats:
        #        print(f"  Threat distance: {threats[0]['distance']:.4f}")

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
            
            # Compute speed ratio for theoretical calculations (but don't add to features - it's a duplicate)
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
                
                # Heading error to optimal escape
                # (how far is agent from optimal evasion heading?)
                #if apollo['capture_feasible'] and apollo['optimal_pursuer_heading'] != 0:
                #    # Optimal escape is perpendicular to intercept
                #    optimal_escape = apollo['optimal_pursuer_heading'] + np.pi/2
                #    heading_error = optimal_escape - agent_heading
                #    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                #    features.append(heading_error / np.pi)
                #else:
                #    features.append(0.0)
            
            # BEZ (Basic Engagement Zone) features
            if self.enable_bez:
                bez = self.feature_computer.compute_bez_penetration(
                    agent_pos, agent_heading, threat_pos, mu, threat_range, capture_radius
                )
                
                # BEZ penetration depth (positive = inside danger zone)
                features.append(np.clip(bez['normalized_penetration'], -1, 1))
                
                # Binary: inside engagement zone?
                features.append(1.0 if bez['inside_bez'] else 0.0)
                
                # Aspect angle (how agent is oriented relative to threat)
                features.append(bez['aspect_angle'] / np.pi)
            
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
            
            # Escape feasibility for range-limited scenario
            escape_info = self.feature_computer.compute_escape_feasibility(
                threat_pos, agent_pos, mu, threat_range
            )
            # ------------------------------------------------------------------
            # NEW: Offensive features (WEZ dominance + offensive Apollonius TTC)
            # ------------------------------------------------------------------
            if self.enable_offense_wez or self.enable_offense_ttc:
                # Approximate own/ enemy ranges:
                #   - our_Rmax: own missile vs track (Godot feature)
                #   - their_Rmax: enemy missile vs us (threat_range)
                our_Rmax = godot.get('own_missile_rmax', threat_range)
                their_Rmax = threat_range

                # Compute threat velocity vector
                # Since we don't have threat heading, estimate from relative position
                threat_relative = threat_pos - agent_pos
                threat_distance = np.linalg.norm(threat_relative)
                if threat_distance > 1e-6:
                    threat_direction = threat_relative / threat_distance
                else:
                    threat_direction = np.array([1.0, 0.0])
                
                # Threat velocity = speed * direction (approximate)
                threat_velocity = threat_direction * threat_speed

                offense = self.feature_computer.compute_offensive_metrics(
                    own_pos=agent_pos,
                    own_vel=agent_velocity,  # Use velocity vector, not speed
                    tgt_pos=threat_pos,
                    tgt_vel=threat_velocity,  # Use velocity vector, not speed
                    our_Rmax=our_Rmax,
                    their_Rmax=their_Rmax,
                )

                if self.enable_offense_wez:
                    # offensive_dominance can be in [-1,1] naturally
                    features.append(
                        np.clip(offense.get('offensive_dominance', 0.0), -1.0, 1.0)
                    )

                if self.enable_offense_ttc:
                    # offensive_ttc_score is defined in [0,1]
                    features.append(
                        np.clip(offense.get('offensive_ttc_score', 0.0), 0.0, 1.0)
                    )


            features.append(1.0 if escape_info['evader_always_escapes'] else 0.0)
            features.append(1.0 if escape_info['pursuer_always_captures'] else 0.0)
            
        else:
            # No threat detected - pad with safe values
            # Note: We don't add speed_ratio (mu) anymore - it was a duplicate
            
            if self.enable_apollonius:
                features.extend([1.0, 0.0])  # time=max, not feasible
            
            if self.enable_bez:
                features.extend([0.0, 0.0, 0.0])  # not penetrating, not inside, neutral aspect
            
            if self.enable_dmc:
                features.extend([0.0, 0.0])  # no maneuver needed, not inside threat
            
            features.extend([1.0, 0.0])  # can escape, won't be captured

            if self.enable_offense_wez:
                features.append(0.0)   # no WEZ dominance one way or another
            if self.enable_offense_ttc:
                features.append(0.0) 
        
        # =================================================================
        # REMOVED: MULTI-THREAT AGGREGATE (2 duplicates)
        # Only used by disabled multithreat_shaping
        # =================================================================
        
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
    print("EnrichedObservationWrapper - Example Usage")
    print("=" * 50)
    
    # Create a dummy environment for testing
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        def reset(self, seed=None, options=None):
            return np.zeros(10, dtype=np.float32), {}
        
        def step(self, action):
            obs = np.random.randn(10).astype(np.float32)
            return obs, 0.0, False, False, {}
    
    # Wrap it
    env = DummyEnv()
    wrapped_env = EnrichedObservationWrapper(env, debug=True)
    
    print(f"\nOriginal obs space: {env.observation_space}")
    print(f"Wrapped obs space: {wrapped_env.observation_space}")
    
    # Test reset
    obs, info = wrapped_env.reset()
    print(f"\nReset observation shape: {obs.shape}")
    
    # Test step
    obs, reward, term, trunc, info = wrapped_env.step(wrapped_env.action_space.sample())
    print(f"Step observation shape: {obs.shape}")
    
    print("\nWrapper test complete!")
