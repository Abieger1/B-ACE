"""
track_aware_shaping.py

Reward shaping wrappers that leverage track awareness to discourage fleeing:

1. TrackMaintenancePenalty: Penalizes losing track of the enemy (transitioning from tracked to untracked)
2. AsymmetricRangeShaping: Rewards closing range more than opening, or penalizes opening more heavily

These work together to discourage the agent from fleeing while it has a track, since:
- Fleeing often causes track loss (sensor cone limits)
- Opening range while tracked is penalized more heavily than closing range is rewarded
"""

import gymnasium as gym
import numpy as np


class TrackMaintenancePenalty(gym.Wrapper):
    """
    Penalizes the agent when it loses track of the enemy (while enemy is alive).
    
    The insight is that if the blue agent is fleeing from red, red will fall
    outside the sensor cone and track will be lost. This provides a direct
    signal against fleeing behavior when the agent currently has a track.
    
    IMPORTANT: Track loss is NOT penalized after red is killed, because losing
    track of a destroyed aircraft is expected behavior.
    
    Args:
        env: The environment to wrap
        track_idx: Index of the track_detected observation (0 or 1)
        loss_penalty: Penalty applied when track transitions from 1 to 0 (while red alive)
        maintain_bonus: Small bonus for maintaining track each step (optional)
        regain_bonus: Bonus when track is regained (optional, encourages reacquisition)
    """
    def __init__(
        self,
        env,
        track_idx: int = 26,
        loss_penalty: float = -0.5,
        maintain_bonus: float = 0.0,
        regain_bonus: float = 0.0,
    ):
        super().__init__(env)
        self.track_idx = int(track_idx)
        self.loss_penalty = float(loss_penalty)
        self.maintain_bonus = float(maintain_bonus)
        self.regain_bonus = float(regain_bonus)
        
        self._prev_track = None
        self._red_is_dead = False  # Track if red has been killed
        self._track_loss_count = 0
        self._track_regain_count = 0
        self._track_steps = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_track = float(obs[self.track_idx]) > 0.5
        self._red_is_dead = False
        self._track_loss_count = 0
        self._track_regain_count = 0
        self._track_steps = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if red was just killed (latching state)
        if info.get("red_killed_event", False):
            self._red_is_dead = True
        
        has_track = float(obs[self.track_idx]) > 0.5
        shaping = 0.0
        
        # Only apply track shaping while red is alive
        if self._prev_track is not None and not self._red_is_dead:
            # Track LOSS: had track, now lost it (and red is still alive!)
            if self._prev_track and not has_track:
                shaping = self.loss_penalty
                self._track_loss_count += 1
            
            # Track REGAIN: didn't have track, now have it
            elif not self._prev_track and has_track:
                shaping = self.regain_bonus
                self._track_regain_count += 1
            
            # Track MAINTAIN: had track, still have it
            elif self._prev_track and has_track:
                shaping = self.maintain_bonus
                self._track_steps += 1
        
        reward = float(reward) + shaping
        self._prev_track = has_track
        
        # Info for logging
        info["track_maintenance_shaping"] = float(shaping)
        info["track_loss_count"] = int(self._track_loss_count)
        info["track_regain_count"] = int(self._track_regain_count)
        info["track_maintained_steps"] = int(self._track_steps)
        info["has_track"] = bool(has_track)
        info["red_is_dead"] = bool(self._red_is_dead)
        
        return obs, reward, terminated, truncated, info


class AsymmetricRangeShaping(gym.Wrapper):
    """
    Asymmetric range shaping that penalizes opening range more heavily than
    it rewards closing range (or vice versa).
    
    The key insight is that if blue is fleeing, the range increases. By making
    the penalty for increasing range larger than the reward for closing, we
    create an asymmetric incentive that discourages fleeing when the agent
    has a track.
    
    This only applies when the agent has a track AND red is alive.
    Once red is destroyed, range shaping is disabled.
    
    Args:
        env: The environment to wrap
        range_idx: Index of the track_dist observation (normalized range)
        track_idx: Index of the track_detected observation
        k_close: Coefficient for closing range (positive = reward for closing)
        k_open: Coefficient for opening range (positive = penalty for opening)
        clip_delta: Maximum delta to consider (filters anomalies)
        asymmetry_ratio: If k_open not specified, k_open = k_close * asymmetry_ratio
        min_range_for_penalty: Don't penalize opening when very close (achieved kill position)
    """
    def __init__(
        self,
        env,
        range_idx: int = 17,
        track_idx: int = 26,
        k_close: float = 20.0,
        k_open: float = None,  # If None, uses asymmetry_ratio * k_close
        asymmetry_ratio: float = 2.0,  # k_open = 2x k_close by default
        clip_delta: float = 0.02,
        max_valid_delta: float = 0.05,
        min_range_for_penalty: float = 0.05,  # ~46 GDM if norm=926
    ):
        super().__init__(env)
        self.range_idx = int(range_idx)
        self.track_idx = int(track_idx)
        self.k_close = float(k_close)
        self.k_open = float(k_open) if k_open is not None else float(k_close * asymmetry_ratio)
        self.clip_delta = float(clip_delta)
        self.max_valid_delta = float(max_valid_delta)
        self.min_range_for_penalty = float(min_range_for_penalty)
        
        self._prev_range = None
        self._red_is_dead = False  # Track if red has been killed
        self._close_reward_total = 0.0
        self._open_penalty_total = 0.0
        self._step_count = 0
        self._anomaly_count = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_range = float(obs[self.range_idx])
        self._red_is_dead = False
        self._close_reward_total = 0.0
        self._open_penalty_total = 0.0
        self._step_count = 0
        self._anomaly_count = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if red was just killed (latching state)
        if info.get("red_killed_event", False):
            self._red_is_dead = True
        
        has_track = float(obs[self.track_idx]) > 0.5
        cur_range = float(obs[self.range_idx])
        shaping = 0.0
        
        self._step_count += 1
        
        # Only apply range shaping while red is alive and we have track
        if has_track and self._prev_range is not None and not self._red_is_dead:
            raw_delta = self._prev_range - cur_range  # positive if closing
            
            # Filter anomalous deltas (target destroyed, respawned, etc.)
            if abs(raw_delta) > self.max_valid_delta:
                self._anomaly_count += 1
                self._prev_range = cur_range
            else:
                # Clip the delta
                delta = float(np.clip(raw_delta, -self.clip_delta, self.clip_delta))
                
                if delta > 0:
                    # CLOSING: reward proportional to closing rate
                    shaping = self.k_close * delta
                    self._close_reward_total += shaping
                elif delta < 0:
                    # OPENING: penalty proportional to opening rate
                    # But don't penalize opening when already very close
                    if cur_range > self.min_range_for_penalty:
                        shaping = self.k_open * delta  # delta is negative, so shaping is negative
                        self._open_penalty_total += abs(shaping)
                
                reward = float(reward) + shaping
                self._prev_range = cur_range
        else:
            # No track or red is dead - just update prev_range for next step
            self._prev_range = cur_range
        
        # Info for logging
        info["asym_range_shaping"] = float(shaping)
        info["asym_close_reward_total"] = float(self._close_reward_total)
        info["asym_open_penalty_total"] = float(self._open_penalty_total)
        info["asym_anomaly_count"] = int(self._anomaly_count)
        info["red_is_dead"] = bool(self._red_is_dead)
        
        return obs, reward, terminated, truncated, info


class CombinedTrackAwareShaping(gym.Wrapper):
    """
    Convenience wrapper that combines TrackMaintenance and AsymmetricRangeShaping.
    
    This provides a unified interface for the two complementary mechanisms:
    1. Direct penalty for losing track (only while red is alive!)
    2. Asymmetric range shaping when track is maintained
    
    IMPORTANT: Track loss penalty is DISABLED once red is killed, because losing
    track of a destroyed aircraft is expected and should not be penalized.
    
    Args:
        env: The environment to wrap
        range_idx: Index of track_dist observation
        track_idx: Index of track_detected observation
        track_loss_penalty: Penalty when track is lost (while red is alive)
        k_close: Coefficient for closing range reward
        k_open: Coefficient for opening range penalty (or asymmetry_ratio * k_close)
        asymmetry_ratio: Ratio of k_open to k_close if k_open not specified
        clip_delta: Maximum range delta to consider
    """
    def __init__(
        self,
        env,
        range_idx: int = 17,
        track_idx: int = 26,
        track_loss_penalty: float = -0.5,
        track_maintain_bonus: float = 0.0,
        k_close: float = 20.0,
        k_open: float = None,
        asymmetry_ratio: float = 2.0,
        clip_delta: float = 0.02,
        max_valid_delta: float = 0.05,
        min_range_for_penalty: float = 0.05,
    ):
        super().__init__(env)
        self.range_idx = int(range_idx)
        self.track_idx = int(track_idx)
        self.track_loss_penalty = float(track_loss_penalty)
        self.track_maintain_bonus = float(track_maintain_bonus)
        self.k_close = float(k_close)
        self.k_open = float(k_open) if k_open is not None else float(k_close * asymmetry_ratio)
        self.clip_delta = float(clip_delta)
        self.max_valid_delta = float(max_valid_delta)
        self.min_range_for_penalty = float(min_range_for_penalty)
        
        # State tracking
        self._prev_track = None
        self._prev_range = None
        self._red_is_dead = False  # Track if red has been killed
        
        # Diagnostics
        self._track_loss_count = 0
        self._close_reward_total = 0.0
        self._open_penalty_total = 0.0
        self._track_shaping_total = 0.0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_track = float(obs[self.track_idx]) > 0.5
        self._prev_range = float(obs[self.range_idx])
        self._red_is_dead = False  # Reset on new episode
        self._track_loss_count = 0
        self._close_reward_total = 0.0
        self._open_penalty_total = 0.0
        self._track_shaping_total = 0.0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        has_track = float(obs[self.track_idx]) > 0.5
        cur_range = float(obs[self.range_idx])
        
        # Check if red was just killed (latching state)
        if info.get("red_killed_event", False):
            self._red_is_dead = True
        
        track_shaping = 0.0
        range_shaping = 0.0
        
        # === Track Maintenance (ONLY while red is alive) ===
        if self._prev_track is not None and not self._red_is_dead:
            if self._prev_track and not has_track:
                # Lost track (and red is still alive - this is bad!)
                track_shaping = self.track_loss_penalty
                self._track_loss_count += 1
            elif self._prev_track and has_track:
                # Maintained track
                track_shaping = self.track_maintain_bonus
        
        self._track_shaping_total += track_shaping
        
        # === Asymmetric Range Shaping (only when tracked AND red alive) ===
        if has_track and self._prev_range is not None and self._prev_track and not self._red_is_dead:
            raw_delta = self._prev_range - cur_range  # positive if closing
            
            if abs(raw_delta) <= self.max_valid_delta:
                delta = float(np.clip(raw_delta, -self.clip_delta, self.clip_delta))
                
                if delta > 0:
                    # CLOSING
                    range_shaping = self.k_close * delta
                    self._close_reward_total += range_shaping
                elif delta < 0 and cur_range > self.min_range_for_penalty:
                    # OPENING (with penalty)
                    range_shaping = self.k_open * delta  # negative
                    self._open_penalty_total += abs(range_shaping)
        
        # Update state
        self._prev_track = has_track
        self._prev_range = cur_range
        
        # Apply total shaping
        total_shaping = track_shaping + range_shaping
        reward = float(reward) + total_shaping
        
        # Info for logging
        info["track_aware_shaping"] = float(total_shaping)
        info["track_shaping"] = float(track_shaping)
        info["range_shaping"] = float(range_shaping)
        info["track_loss_count"] = int(self._track_loss_count)
        info["close_reward_total"] = float(self._close_reward_total)
        info["open_penalty_total"] = float(self._open_penalty_total)
        info["has_track"] = bool(has_track)
        info["red_is_dead"] = bool(self._red_is_dead)
        
        return obs, reward, terminated, truncated, info


# ============================================================================
# Example usage and integration notes
# ============================================================================
"""
Integration into train_bace_clean.py:

1. Import at the top:
   from track_aware_shaping import (
       TrackMaintenancePenalty, 
       AsymmetricRangeShaping,
       CombinedTrackAwareShaping
   )

2. Add in make_env() AFTER RangeClosingShaping (or REPLACE it):

   # Option A: Replace RangeClosingShaping with combined wrapper
   e = CombinedTrackAwareShaping(
       e,
       range_idx=17,
       track_idx=26,
       track_loss_penalty=-0.5,      # Penalize losing track
       k_close=30.0,                 # Same as current RangeClosingShaping
       asymmetry_ratio=2.0,          # Opening penalized 2x closing reward
       clip_delta=0.02,
   )

   # Option B: Use separate wrappers for more control
   e = TrackMaintenancePenalty(
       e,
       track_idx=26,
       loss_penalty=-0.5,
   )
   e = AsymmetricRangeShaping(
       e,
       range_idx=17,
       track_idx=26,
       k_close=30.0,
       asymmetry_ratio=2.0,
   )

Tuning suggestions:
- Start with track_loss_penalty=-0.5 (significant but not overwhelming)
- asymmetry_ratio=2.0 means opening range is penalized 2x the reward for closing
- Can increase asymmetry_ratio to 3.0 or 4.0 if agent still flees
- Monitor track_loss_count in evaluation to see if penalty is having effect
"""
