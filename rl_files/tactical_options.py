#!/usr/bin/env python3
"""
tactical_options.py

Defines high-level tactical options for hierarchical reinforcement learning in B-ACE.
Each option represents a distinct tactical behavior that agents can execute.

Options 0-4: Rule-based expert behaviors (deterministic)
Option 5: GREEDY - learned policy (trained via PPO)

The meta-controller selects among these 6 options. During training, a curriculum
shifts from mostly rule-based (95%) to mostly greedy (100%).

Based on pursuit-evasion differential game theory and BVR air combat tactics.
"""

from enum import IntEnum
import numpy as np
from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field


class TacticalOption(IntEnum):
    """
    High-level tactical options for B-ACE air combat.
    
    0-4: Rule-based expert behaviors (deterministic, domain knowledge)
    5: GREEDY - learned policy (trained via PPO)
    """
    DEFEND_HVAA = 0           # Defensive escort: Stay near HVAA, intercept threats
    INTERCEPT_ENEMY = 1       # Aggressive pursuit of enemy aircraft
    EVADE_MISSILE = 2         # Defensive evasion from incoming missile
    OFFENSIVE_POSITIONING = 3  # Maneuver for WEZ/BEZ advantage
    DEFENSIVE_POSITIONING = 4  # Maneuver to stay outside enemy BEZ
    GREEDY = 5                # Learned policy (PPO-trained)


# Constants for convenience
NUM_OPTIONS = len(TacticalOption)
NUM_RULE_BASED = 5  # Options 0-4
GREEDY_OPTION_ID = 5


def is_rule_based(option_id: int) -> bool:
    """Check if option is rule-based (vs learned)."""
    return 0 <= option_id <= 4


def is_greedy(option_id: int) -> bool:
    """Check if option is the learned greedy policy."""
    return option_id == GREEDY_OPTION_ID


@dataclass
class Option:
    """
    Represents a temporally extended tactical behavior.
    """
    name: str
    option_id: TacticalOption
    initiation_fn: Callable[[Dict[str, Any]], bool]
    termination_fn: Callable[[Dict[str, Any]], bool]
    reward_bonus: float = 0.0
    description: str = ""
    is_learned: bool = False
    
    def can_initiate(self, state: Dict[str, Any]) -> bool:
        return self.initiation_fn(state)
    
    def should_terminate(self, state: Dict[str, Any]) -> bool:
        return self.termination_fn(state)


# =============================================================================
# HARDCODED OBSERVATION INDICES
# =============================================================================
# These are fallback indices when obs_labels is empty/unavailable.
# Update these if your B-ACE observation structure changes.

# Base B-ACE observation indices (typical 22-27 dim observation)
BASE_OBS_INDICES = {
    # Agent state (indices 0-4)
    'own_x': 0,
    'own_y': 1,
    'own_heading': 2,
    'own_speed': 3,
    'own_altitude': 4,
    
    # Offensive/Defensive factors (indices 5-6)
    'offensive_factor': 5,
    'defensive_factor': 6,
    
    # Track info (indices 7-12)
    'track_detected': 7,
    'own_dist_target': 8,
    'own_aspect_angle_target': 9,
    'target_aspect_angle': 10,
    'target_heading': 11,
    'target_speed': 12,
    
    # Missile info (indices 13-17)
    'own_missile_count': 13,
    'incoming_missile_track': 14,
    'incoming_missile_dist': 15,
    'incoming_missile_aspect': 16,
    'own_missile_active': 17,
    
    # HVAA info (indices 18-21)
    'hvaa_dist': 18,
    'hvaa_angle_off': 19,
    'hvaa_heading': 20,
    'hvaa_detected': 21,
}

# Enriched feature indices (appended after base obs)
# These start at index 22 if base obs is 22 dims, or 27 if base is 27 dims
ENRICHED_OFFSETS = {
    # Apollonius features
    'time_to_capture': 0,
    'capture_feasible': 1,
    
    # BEZ features
    'bez_penetration': 2,
    'inside_bez': 3,
    'bez_aspect_angle': 4,
    
    # DMC features
    'dmc_normalized': 5,
    'inside_threat_cone': 6,
    
    # Escape feasibility
    'always_escapes': 7,
    'always_captured': 8,
    
    # Offensive metrics
    'offensive_wez_dominance': 9,
    'offensive_ttc_score': 10,
    
    # HVAA escort features
    'hvaa_escort_score': 11,
    'hvaa_threat_level': 12,
}


def extract_tactical_state(obs: np.ndarray, obs_labels: Dict[str, int] = None, 
                           enriched: bool = True) -> Dict[str, Any]:
    """
    Extract tactical state information from observation for option logic.
    
    Uses obs_labels if provided, otherwise falls back to hardcoded indices.
    
    Args:
        obs: Raw observation array
        obs_labels: Label to index mapping (optional, uses hardcoded if empty)
        enriched: Whether observation includes enriched features
        
    Returns:
        Dictionary with tactical state information
    """
    state = {}
    obs_labels = obs_labels or {}
    
    # Determine base observation size and enriched start index
    obs_len = len(obs)
    
    # Heuristic: if obs > 30, enriched features are present
    # Base obs is typically 22 or 27 dims
    if obs_len >= 35:
        base_size = 27
        enriched_start = 27
    elif obs_len > 25:
        base_size = 22
        enriched_start = 22 # No enriched features
    
    # --- Extract base observation features ---
    
    def get_idx(label: str, fallback: int) -> int:
        """Get index from obs_labels or fallback to hardcoded."""
        if label in obs_labels:
            return obs_labels[label]
        return fallback if fallback < obs_len else -1
    
    def safe_get(idx: int, default=0.0):
        """Safely get observation value."""
        if 0 <= idx < obs_len:
            return float(obs[idx])
        return default
    
    # Distance and aspect to enemy
    idx = get_idx('own_dist_target', BASE_OBS_INDICES.get('own_dist_target', 8))
    state['distance_to_enemy'] = safe_get(idx, 1.0)
    
    idx = get_idx('own_aspect_angle_target', BASE_OBS_INDICES.get('own_aspect_angle_target', 9))
    state['aspect_angle_to_enemy'] = safe_get(idx, 0.0)
    
    # Offensive/defensive factors
    idx = get_idx('offensive_factor', BASE_OBS_INDICES.get('offensive_factor', 5))
    state['offensive_factor'] = safe_get(idx, 0.0)
    
    idx = get_idx('defensive_factor', BASE_OBS_INDICES.get('defensive_factor', 6))
    state['defensive_factor'] = safe_get(idx, 0.0)
    
    # Track detection
    idx = get_idx('track_detected', BASE_OBS_INDICES.get('track_detected', 7))
    state['enemy_detected'] = safe_get(idx, 0.0) > 0.5
    
    # Missile tracking
    idx = get_idx('incoming_missile_track', BASE_OBS_INDICES.get('incoming_missile_track', 14))
    state['missile_tracking_us'] = safe_get(idx, 0.0) > 0.5
    
    # HVAA information
    idx = get_idx('hvaa_dist', BASE_OBS_INDICES.get('hvaa_dist', 18))
    state['distance_to_hvaa'] = safe_get(idx, 0.5)
    
    idx = get_idx('hvaa_angle_off', BASE_OBS_INDICES.get('hvaa_angle_off', 19))
    state['hvaa_angle_off'] = safe_get(idx, 0.0)
    
    idx = get_idx('hvaa_detected', BASE_OBS_INDICES.get('hvaa_detected', 21))
    state['hvaa_detected'] = safe_get(idx, 1.0) > 0.5
    
    # --- Extract enriched features (if present) ---
    
    if enriched and obs_len > enriched_start:
        def get_enriched(offset: int, default=0.0):
            idx = enriched_start + offset
            return safe_get(idx, default)
        
        # Apollonius features
        state['time_to_capture'] = get_enriched(ENRICHED_OFFSETS['time_to_capture'], 1.0)
        state['capture_feasible'] = get_enriched(ENRICHED_OFFSETS['capture_feasible'], 0.0) > 0.5
        
        # BEZ features - CRITICAL for expert behaviors
        state['bez_penetration'] = get_enriched(ENRICHED_OFFSETS['bez_penetration'], 0.0)
        state['inside_bez'] = get_enriched(ENRICHED_OFFSETS['inside_bez'], 0.0) > 0.5
        state['bez_aspect_angle'] = get_enriched(ENRICHED_OFFSETS['bez_aspect_angle'], 180.0)
        
        # DMC features - CRITICAL for evasion
        state['dmc_normalized'] = get_enriched(ENRICHED_OFFSETS['dmc_normalized'], 0.0)
        state['inside_threat_cone'] = get_enriched(ENRICHED_OFFSETS['inside_threat_cone'], 0.0) > 0.5
        
        # Escape feasibility
        state['always_escapes'] = get_enriched(ENRICHED_OFFSETS['always_escapes'], 0.0) > 0.5
        state['always_captured'] = get_enriched(ENRICHED_OFFSETS['always_captured'], 0.0) > 0.5
        
        # Offensive metrics (if available)
        if obs_len > enriched_start + ENRICHED_OFFSETS['offensive_wez_dominance']:
            state['offensive_wez_dominance'] = get_enriched(ENRICHED_OFFSETS['offensive_wez_dominance'], 0.0)
            state['offensive_ttc_score'] = get_enriched(ENRICHED_OFFSETS['offensive_ttc_score'], 0.0)
        
        # HVAA escort features (if available)
        if obs_len > enriched_start + ENRICHED_OFFSETS['hvaa_escort_score']:
            state['hvaa_escort_score'] = get_enriched(ENRICHED_OFFSETS['hvaa_escort_score'], 0.0)
            state['hvaa_threat_level'] = get_enriched(ENRICHED_OFFSETS['hvaa_threat_level'], 0.0)
    
    return state


# =============================================================================
# OPTION DEFINITIONS
# =============================================================================

def create_defend_hvaa_option() -> Option:
    """DEFEND_HVAA: Defensive escort mission (RULE-BASED)"""
    def initiation(state: Dict[str, Any]) -> bool:
        return True
    
    def termination(state: Dict[str, Any]) -> bool:
        missile_threat = state.get('missile_tracking_us', False)
        hvaa_lost = not state.get('hvaa_detected', True)
        return missile_threat or hvaa_lost
    
    return Option(
        name="Defend HVAA",
        option_id=TacticalOption.DEFEND_HVAA,
        initiation_fn=initiation,
        termination_fn=termination,
        reward_bonus=0.0,
        description="Stay near HVAA, intercept approaching threats",
        is_learned=False
    )


def create_intercept_enemy_option() -> Option:
    """INTERCEPT_ENEMY: Aggressive pursuit (RULE-BASED)"""
    def initiation(state: Dict[str, Any]) -> bool:
        enemy_detected = state.get('enemy_detected', False)
        no_missile = not state.get('missile_tracking_us', False)
        offensive_ok = state.get('offensive_factor', 0.0) > -0.5
        return enemy_detected and no_missile and offensive_ok
    
    def termination(state: Dict[str, Any]) -> bool:
        enemy_lost = not state.get('enemy_detected', False)
        missile_threat = state.get('missile_tracking_us', False)
        inside_bez = state.get('inside_bez', False)
        return enemy_lost or missile_threat or inside_bez
    
    return Option(
        name="Intercept Enemy",
        option_id=TacticalOption.INTERCEPT_ENEMY,
        initiation_fn=initiation,
        termination_fn=termination,
        reward_bonus=0.002,
        description="Aggressive pursuit and engagement",
        is_learned=False
    )


def create_evade_missile_option() -> Option:
    """EVADE_MISSILE: Emergency evasion (RULE-BASED)"""
    def initiation(state: Dict[str, Any]) -> bool:
        return state.get('missile_tracking_us', False)
    
    def termination(state: Dict[str, Any]) -> bool:
        no_missile = not state.get('missile_tracking_us', False)
        return no_missile
    
    return Option(
        name="Evade Missile",
        option_id=TacticalOption.EVADE_MISSILE,
        initiation_fn=initiation,
        termination_fn=termination,
        reward_bonus=0.0,
        description="Defensive evasion from incoming missile",
        is_learned=False
    )


def create_offensive_positioning_option() -> Option:
    """OFFENSIVE_POSITIONING: Maneuver for advantage (RULE-BASED)"""
    def initiation(state: Dict[str, Any]) -> bool:
        enemy_detected = state.get('enemy_detected', False)
        outside_bez = not state.get('inside_bez', False)
        no_missile = not state.get('missile_tracking_us', False)
        needs_positioning = state.get('offensive_factor', 0.0) < 0.5
        return enemy_detected and outside_bez and no_missile and needs_positioning
    
    def termination(state: Dict[str, Any]) -> bool:
        inside_bez = state.get('inside_bez', False)
        missile_threat = state.get('missile_tracking_us', False)
        good_position = state.get('offensive_factor', 0.0) > 0.6
        enemy_lost = not state.get('enemy_detected', False)
        return inside_bez or missile_threat or good_position or enemy_lost
    
    return Option(
        name="Offensive Positioning",
        option_id=TacticalOption.OFFENSIVE_POSITIONING,
        initiation_fn=initiation,
        termination_fn=termination,
        reward_bonus=0.001,
        description="Maneuver for WEZ advantage and BEZ avoidance",
        is_learned=False
    )


def create_defensive_positioning_option() -> Option:
    """DEFENSIVE_POSITIONING: Maintain defensive geometry (RULE-BASED)"""
    def initiation(state: Dict[str, Any]) -> bool:
        bez_penetration = state.get('bez_penetration', 0.0)
        near_or_in_bez = bez_penetration > 0.3
        no_missile = not state.get('missile_tracking_us', False)
        enemy_detected = state.get('enemy_detected', False)
        return near_or_in_bez and no_missile and enemy_detected
    
    def termination(state: Dict[str, Any]) -> bool:
        bez_penetration = state.get('bez_penetration', 0.0)
        safe_distance = bez_penetration < 0.2
        missile_threat = state.get('missile_tracking_us', False)
        enemy_lost = not state.get('enemy_detected', False)
        return safe_distance or missile_threat or enemy_lost
    
    return Option(
        name="Defensive Positioning",
        option_id=TacticalOption.DEFENSIVE_POSITIONING,
        initiation_fn=initiation,
        termination_fn=termination,
        reward_bonus=0.001,
        description="Maneuver to stay outside enemy BEZ",
        is_learned=False
    )


def create_greedy_option() -> Option:
    """GREEDY: Learned policy (PPO-TRAINED)"""
    def initiation(state: Dict[str, Any]) -> bool:
        return True
    
    def termination(state: Dict[str, Any]) -> bool:
        return False
    
    return Option(
        name="Greedy (Learned)",
        option_id=TacticalOption.GREEDY,
        initiation_fn=initiation,
        termination_fn=termination,
        reward_bonus=0.0,
        description="Learned policy trained via PPO",
        is_learned=True
    )


# =============================================================================
# OPTION FACTORY
# =============================================================================

def create_tactical_options() -> Dict[TacticalOption, Option]:
    """Factory function to create all tactical options."""
    return {
        TacticalOption.DEFEND_HVAA: create_defend_hvaa_option(),
        TacticalOption.INTERCEPT_ENEMY: create_intercept_enemy_option(),
        TacticalOption.EVADE_MISSILE: create_evade_missile_option(),
        TacticalOption.OFFENSIVE_POSITIONING: create_offensive_positioning_option(),
        TacticalOption.DEFENSIVE_POSITIONING: create_defensive_positioning_option(),
        TacticalOption.GREEDY: create_greedy_option(),
    }


def create_rule_based_options() -> Dict[TacticalOption, Option]:
    """Create only rule-based options (excludes GREEDY)."""
    all_options = create_tactical_options()
    return {k: v for k, v in all_options.items() if not v.is_learned}


def get_valid_options(state: Dict[str, Any], 
                      all_options: Dict[TacticalOption, Option]) -> List[TacticalOption]:
    """Get list of options that can be initiated in current state."""
    valid = []
    for opt_id, option in all_options.items():
        if option.can_initiate(state):
            valid.append(opt_id)
    
    if TacticalOption.DEFEND_HVAA not in valid and TacticalOption.DEFEND_HVAA in all_options:
        valid.append(TacticalOption.DEFEND_HVAA)
    
    if TacticalOption.GREEDY in all_options and TacticalOption.GREEDY not in valid:
        valid.append(TacticalOption.GREEDY)
    
    return valid


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Tactical Options - Self Test with Hardcoded Indices")
    print("=" * 70)
    
    # Test with a mock observation (47 dims - enriched)
    mock_obs = np.zeros(47, dtype=np.float32)
    
    # Set some base values
    mock_obs[5] = 0.3   # offensive_factor
    mock_obs[6] = -0.2  # defensive_factor
    mock_obs[7] = 1.0   # track_detected
    mock_obs[8] = 0.4   # distance_to_enemy
    mock_obs[14] = 0.0  # no incoming missile
    mock_obs[18] = 0.35 # hvaa_dist
    mock_obs[21] = 1.0  # hvaa_detected
    
    # Set enriched values (starting at index 22)
    mock_obs[24] = 0.25  # bez_penetration
    mock_obs[25] = 0.0   # inside_bez (False)
    mock_obs[27] = 0.3   # dmc_normalized
    mock_obs[28] = 0.0   # inside_threat_cone (False)
    
    print(f"\nMock observation shape: {mock_obs.shape}")
    print(f"Testing extract_tactical_state with empty obs_labels...")
    
    state = extract_tactical_state(mock_obs, obs_labels={}, enriched=True)
    
    print(f"\nExtracted state ({len(state)} keys):")
    for key, val in sorted(state.items()):
        print(f"  {key}: {val}")
    
    # Verify critical values (use approximate comparison for floats)
    print("\n" + "-" * 70)
    print("Verification:")
    assert abs(state['offensive_factor'] - 0.3) < 0.01, f"Expected ~0.3, got {state['offensive_factor']}"
    assert state['enemy_detected'] == True, f"Expected True, got {state['enemy_detected']}"
    assert abs(state['distance_to_enemy'] - 0.4) < 0.01, f"Expected ~0.4, got {state['distance_to_enemy']}"
    print("✓ Base observation values extracted correctly!")
    
    # Test with correct enriched indices (base=27 for 47-dim obs)
    # With enriched_start=27: bez_penetration at 29, dmc_normalized at 32
    mock_obs[29] = 0.25  # bez_penetration (27+2)
    mock_obs[30] = 0.0   # inside_bez (27+3)
    mock_obs[32] = 0.3   # dmc_normalized (27+5)
    
    state2 = extract_tactical_state(mock_obs, obs_labels={}, enriched=True)
    print(f"\nAfter setting enriched values at correct indices:")
    print(f"  bez_penetration: {state2['bez_penetration']}")
    print(f"  dmc_normalized: {state2['dmc_normalized']}")
    assert abs(state2['bez_penetration'] - 0.25) < 0.01, f"Expected ~0.25, got {state2['bez_penetration']}"
    assert abs(state2['dmc_normalized'] - 0.3) < 0.01, f"Expected ~0.3, got {state2['dmc_normalized']}"
    print("✓ Enriched observation values extracted correctly!")
    
    # Test option validity
    options = create_tactical_options()
    valid = get_valid_options(state, options)
    print(f"\nValid options: {[options[opt].name for opt in valid]}")
    
    print("\n" + "=" * 70)
    print("✓ Tactical options module ready!")
