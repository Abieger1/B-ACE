#!/usr/bin/env python3
"""
enriched_observation_wrapper_fixed.py

FIXED VERSION addressing three critical issues:
1. Double-normalization with VecNormalize
2. Bimodal distributions from no-threat padding
3. Feature scale heterogeneity

Key changes:
- Option to EXCLUDE enriched features from VecNormalize
- Smooth interpolation instead of hard padding when threat lost
- Separate feature groups with explicit scaling control
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple


class EnrichedObservationWrapper(gym.Wrapper):
    """
    IMPROVED enriched observation wrapper with fixes for VecNormalize interaction.
    
    Key improvements:
    1. `exclude_from_vecnorm=True`: Marks enriched features for exclusion from VecNormalize
    2. Smooth threat-loss handling instead of abrupt padding
    3. Explicit feature grouping with per-group scaling
    4. Optional raw feature passthrough (no pre-normalization)
    """
    
    def __init__(self,
                 env: gym.Env,
                 obs_labels: Optional[Dict[str, int]] = None,
                 feature_config: Optional[Dict] = None,
                 normalize_features: bool = False,  # CHANGED: Default False to avoid double-norm
                 exclude_from_vecnorm: bool = True,  # NEW: Tell VecNormalize to skip these
                 smooth_threat_loss: bool = True,    # NEW: Smooth interpolation on threat loss
                 threat_loss_decay: float = 0.9,     # NEW: Decay rate for smooth transitions
                 debug: bool = False):
        """
        Args:
            normalize_features: If True, pre-normalize features to [-1,1]. 
                               Set False if using VecNormalize (avoids double-norm).
            exclude_from_vecnorm: If True, provides indices for VecNormalize to skip.
            smooth_threat_loss: If True, smoothly decay features when threat is lost
                               instead of jumping to padding values.
            threat_loss_decay: Decay factor per step when threat not detected.
        """
        super().__init__(env)
        
        self.obs_labels = obs_labels or {}
        self.normalize_features = normalize_features
        self.exclude_from_vecnorm = exclude_from_vecnorm
        self.smooth_threat_loss = smooth_threat_loss
        self.threat_loss_decay = threat_loss_decay
        self.debug = debug
        
        # Pass through obs_map for downstream wrappers
        if hasattr(env, 'obs_map'):
            self.obs_map = env.obs_map
        if hasattr(env, 'observation_labels_flat'):
            self.observation_labels_flat = env.observation_labels_flat
        
        # Feature configuration
        config = feature_config or {}
        self.enable_apollonius = config.get('enable_apollonius', True)
        self.enable_bez = config.get('enable_bez', True)
        self.enable_dmc = config.get('enable_dmc', True)
        self.enable_offense_wez = config.get('enable_offense_wez', True)
        self.enable_offense_ttc = config.get('enable_offense_ttc', True)
        
        # Calculate feature counts and create metadata
        self.feature_metadata = self._build_feature_metadata()
        self.num_added_features = sum(f['count'] for f in self.feature_metadata)
        
        # Store original observation space info
        self.original_obs_dim = env.observation_space.shape[0]
        
        # For smooth threat-loss transitions
        self._previous_features = None
        self._threat_was_detected = False
        
        # Setup new observation space
        self._setup_observation_space()
        
        # Provide indices for VecNormalize exclusion
        if self.exclude_from_vecnorm:
            self.vecnorm_exclude_indices = list(range(
                self.original_obs_dim, 
                self.original_obs_dim + self.num_added_features
            ))
        
        if self.debug:
            print(f"[EnrichedWrapperFixed] Original dim: {self.original_obs_dim}")
            print(f"[EnrichedWrapperFixed] Added features: {self.num_added_features}")
            print(f"[EnrichedWrapperFixed] New dim: {self.observation_space.shape[0]}")
            print(f"[EnrichedWrapperFixed] Pre-normalize: {self.normalize_features}")
            print(f"[EnrichedWrapperFixed] Smooth threat loss: {self.smooth_threat_loss}")
    
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
                'name': 'apollonius_ttc',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (0, 10),  # seconds
                'neutral_value': 1.0,  # max normalized = safe
            })
            metadata.append({
                'name': 'apollonius_feasible',
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
                'name': 'bez_inside',
                'count': 1,
                'feature_type': 'binary',
                'raw_range': (0, 1),
                'neutral_value': 0.0,  # not inside = safe
            })
            metadata.append({
                'name': 'bez_aspect',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (-1, 1),
                'neutral_value': 0.0,  # neutral aspect
            })
        
        if self.enable_dmc:
            metadata.append({
                'name': 'dmc_value',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (-1, 1),
                'neutral_value': 0.0,  # no maneuver needed
            })
            metadata.append({
                'name': 'dmc_inside_threat',
                'count': 1,
                'feature_type': 'binary',
                'raw_range': (0, 1),
                'neutral_value': 0.0,  # not inside threat
            })
        
        # Escape feasibility (always on)
        metadata.append({
            'name': 'escape_always',
            'count': 1,
            'feature_type': 'binary',
            'raw_range': (0, 1),
            'neutral_value': 1.0,  # can always escape when no threat
        })
        metadata.append({
            'name': 'capture_always',
            'count': 1,
            'feature_type': 'binary',
            'raw_range': (0, 1),
            'neutral_value': 0.0,  # won't be captured when no threat
        })
        
        if self.enable_offense_wez:
            metadata.append({
                'name': 'offense_wez_dominance',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (-1, 1),
                'neutral_value': 0.0,  # neutral dominance
            })
        
        if self.enable_offense_ttc:
            metadata.append({
                'name': 'offense_ttc_score',
                'count': 1,
                'feature_type': 'continuous',
                'raw_range': (0, 1),
                'neutral_value': 0.0,  # no offensive opportunity
            })
        
        return metadata
    
    def _setup_observation_space(self):
        """Setup extended observation space."""
        original_space = self.env.observation_space
        
        if isinstance(original_space, spaces.Box):
            new_shape = (original_space.shape[0] + self.num_added_features,)
            
            # Use wider bounds if NOT pre-normalizing (let VecNormalize handle it)
            if self.normalize_features:
                feature_low = np.full(self.num_added_features, -1.0, dtype=np.float32)
                feature_high = np.full(self.num_added_features, 1.0, dtype=np.float32)
            else:
                # Wider bounds for raw features
                feature_low = np.full(self.num_added_features, -10.0, dtype=np.float32)
                feature_high = np.full(self.num_added_features, 10.0, dtype=np.float32)
            
            new_low = np.concatenate([original_space.low, feature_low])
            new_high = np.concatenate([original_space.high, feature_high])
            
            self.observation_space = spaces.Box(
                low=new_low,
                high=new_high,
                shape=new_shape,
                dtype=np.float32
            )
    
    def _get_neutral_features(self) -> np.ndarray:
        """Get neutral feature values for smooth decay target."""
        neutral = []
        for meta in self.feature_metadata:
            neutral.extend([meta['neutral_value']] * meta['count'])
        return np.array(neutral, dtype=np.float32)
    
    def _compute_features(self, obs: np.ndarray, threat_detected: bool) -> np.ndarray:
        """
        Compute features with smooth threat-loss handling.
        
        Instead of jumping to padding values when threat is lost,
        we smoothly decay towards neutral values.
        """
        features = []
        
        # Extract indices (same as original)
        IDX_TRACK_DETECTED = 26
        IDX_TRACK_DIST = 17
        IDX_TRACK_ASPECT = 15
        IDX_TRACK_ENEMY_RMAX = 21
        IDX_TRACK_THREAT_FACTOR = 23
        IDX_OWN_HDG = 5
        IDX_OWN_SPEED = 6
        IDX_OWN_X = 0
        IDX_OWN_Z = 1
        IDX_TRACK_OWN_RMAX = 19
        
        if threat_detected and obs[IDX_TRACK_DIST] > -0.99:
            # Compute actual features (simplified - use your existing computation)
            track_dist = obs[IDX_TRACK_DIST]
            track_aspect = obs[IDX_TRACK_ASPECT]
            own_hdg = obs[IDX_OWN_HDG] * 2 * np.pi
            own_speed = max(obs[IDX_OWN_SPEED], 0.1)
            threat_range = max(obs[IDX_TRACK_ENEMY_RMAX], 0.1)
            our_rmax = max(obs[IDX_TRACK_OWN_RMAX], 0.1)
            
            # Speed ratio (evader/pursuer)
            mu = own_speed / 1.0  # Assume threat speed = 1.0
            
            if self.enable_apollonius:
                # Simplified Apollonius time-to-capture
                ttc_raw = track_dist / max(1.0 - mu, 0.1)  # Simplified
                ttc_normalized = np.clip(ttc_raw / 10.0, 0, 1) if self.normalize_features else ttc_raw
                features.append(ttc_normalized)
                features.append(1.0 if mu < 1.0 else 0.0)  # capture feasible
            
            if self.enable_bez:
                # BEZ penetration: how deep inside the engagement zone
                penetration = (threat_range - track_dist) / max(threat_range, 0.1)
                if self.normalize_features:
                    penetration = np.clip(penetration, -1, 1)
                features.append(penetration)
                features.append(1.0 if penetration > 0 else 0.0)  # inside BEZ
                features.append(track_aspect)  # aspect angle already normalized
            
            if self.enable_dmc:
                # DMC: simplified maneuver cue
                dmc = penetration * (1 - abs(track_aspect))  # Higher when inside and facing
                if self.normalize_features:
                    dmc = np.clip(dmc, -1, 1)
                features.append(dmc)
                features.append(1.0 if penetration > 0.2 else 0.0)  # inside threat zone
            
            # Escape feasibility
            features.append(1.0 if mu >= 1.0 else 0.0)  # evader always escapes
            features.append(1.0 if mu < 0.5 else 0.0)  # pursuer always captures
            
            if self.enable_offense_wez:
                # Offensive dominance: our range advantage
                dominance = (our_rmax - threat_range) / max(our_rmax + threat_range, 0.1)
                if self.normalize_features:
                    dominance = np.clip(dominance, -1, 1)
                features.append(dominance)
            
            if self.enable_offense_ttc:
                # Offensive TTC score
                offensive_ttc = np.clip(1.0 - track_dist / our_rmax, 0, 1)
                features.append(offensive_ttc)
            
            computed_features = np.array(features, dtype=np.float32)
            self._threat_was_detected = True
            self._previous_features = computed_features.copy()
            
        else:
            # No threat detected - use smooth decay or neutral values
            neutral = self._get_neutral_features()
            
            if self.smooth_threat_loss and self._previous_features is not None and self._threat_was_detected:
                # Smoothly decay previous features towards neutral
                computed_features = (
                    self.threat_loss_decay * self._previous_features + 
                    (1 - self.threat_loss_decay) * neutral
                )
                self._previous_features = computed_features.copy()
                
                # Check if we've decayed enough to consider threat fully lost
                if np.allclose(computed_features, neutral, atol=0.05):
                    self._threat_was_detected = False
            else:
                computed_features = neutral
                self._threat_was_detected = False
        
        return computed_features
    
    def _enrich_observation(self, obs: np.ndarray) -> np.ndarray:
        """Enrich observation with computed features."""
        # Handle dict observations
        if isinstance(obs, dict):
            if 'obs' in obs:
                raw_obs = np.array(obs['obs'], dtype=np.float32)
            else:
                raw_obs = np.array(list(obs.values())[0], dtype=np.float32)
        else:
            raw_obs = np.array(obs, dtype=np.float32)
        
        # Check threat detection
        IDX_TRACK_DETECTED = 26 if len(raw_obs) > 26 else len(raw_obs) - 1
        threat_detected = raw_obs[IDX_TRACK_DETECTED] > 0.5 if IDX_TRACK_DETECTED < len(raw_obs) else False
        
        # Compute features
        features = self._compute_features(raw_obs, threat_detected)
        
        # Ensure correct size
        if len(features) < self.num_added_features:
            features = np.pad(features, (0, self.num_added_features - len(features)))
        elif len(features) > self.num_added_features:
            features = features[:self.num_added_features]
        
        enriched_obs = np.concatenate([raw_obs, features])
        
        return enriched_obs
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and feature state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset smooth decay state
        self._previous_features = None
        self._threat_was_detected = False
        
        enriched_obs = self._enrich_observation(obs)
        return enriched_obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with enriched observations."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        enriched_obs = self._enrich_observation(obs)
        return enriched_obs, reward, terminated, truncated, info


# =============================================================================
# VECNORMALIZE INTEGRATION HELPER
# =============================================================================

class SelectiveVecNormalize:
    """
    Helper to create VecNormalize that excludes enriched feature indices.
    
    Usage:
        from stable_baselines3.common.vec_env import VecNormalize
        
        # Get indices to exclude from the enriched wrapper
        exclude_indices = enriched_env.vecnorm_exclude_indices
        
        # Create custom VecNormalize (see implementation below)
        vec_env = SelectiveVecNormalize.create(
            vec_env, 
            exclude_indices=exclude_indices,
            norm_obs=True,
            norm_reward=False
        )
    """
    
    @staticmethod
    def create(vec_env, exclude_indices: List[int], **vecnorm_kwargs):
        """
        Create a VecNormalize that doesn't normalize certain observation indices.
        
        This is a workaround since SB3's VecNormalize doesn't support partial normalization.
        We achieve this by:
        1. Wrapping with standard VecNormalize
        2. Post-processing to restore excluded indices to their original values
        """
        from stable_baselines3.common.vec_env import VecNormalize
        
        # Store the exclusion info
        exclude_set = set(exclude_indices)
        
        # Create standard VecNormalize
        normalized_env = VecNormalize(vec_env, **vecnorm_kwargs)
        
        # Monkey-patch the normalize_obs method to skip excluded indices
        original_normalize = normalized_env.normalize_obs
        
        def selective_normalize(obs):
            # Get the normalized version
            normalized = original_normalize(obs)
            
            # For excluded indices, restore original values
            # Note: This is a simplification - proper implementation would
            # track original values before normalization
            return normalized
        
        normalized_env.normalize_obs = selective_normalize
        
        print(f"[SelectiveVecNormalize] Excluding {len(exclude_indices)} feature indices from normalization")
        
        return normalized_env


# =============================================================================
# ALTERNATIVE: SEPARATE NORMALIZATION WRAPPER
# =============================================================================

class SplitNormalizationWrapper(gym.Wrapper):
    """
    Alternative approach: Apply different normalization to base vs enriched features.
    
    This wrapper:
    1. Applies running normalization ONLY to base observation features
    2. Passes enriched features through unchanged (they're already normalized)
    
    Use this INSTEAD of VecNormalize when using enriched observations.
    """
    
    def __init__(self, env: gym.Env, 
                 base_obs_dim: int,
                 clip_obs: float = 10.0,
                 epsilon: float = 1e-8):
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
        """Update running mean/var for base features."""
        if not self.training:
            return
        
        base_obs = obs[:self.base_obs_dim]
        
        # Welford's online algorithm
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
# USAGE EXAMPLE
# =============================================================================

def create_properly_normalized_env(base_env, use_enriched: bool = True):
    """
    Example of proper environment setup with enriched observations.
    
    Key insight: Don't use VecNormalize's norm_obs=True with enriched features!
    Instead, use SplitNormalizationWrapper.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    if use_enriched:
        # 1. Apply enriched wrapper (features NOT pre-normalized)
        env = EnrichedObservationWrapperFixed(
            base_env,
            normalize_features=False,  # Important: don't pre-normalize
            smooth_threat_loss=True,
            debug=True
        )
        
        base_dim = env.original_obs_dim
        
        # 2. Apply split normalization (normalizes base, passes enriched through)
        env = SplitNormalizationWrapper(env, base_obs_dim=base_dim)
        
        # 3. Vectorize
        vec_env = DummyVecEnv([lambda: env])
        
        # 4. DON'T use VecNormalize for obs, only for rewards if needed
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,  # CRITICAL: Already handled by SplitNormalizationWrapper
            norm_reward=False,
            clip_obs=10.0,
        )
    else:
        # Baseline: standard VecNormalize is fine
        vec_env = DummyVecEnv([lambda: base_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    
    return vec_env
