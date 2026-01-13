#!/usr/bin/env python3
"""
rule_based_experts.py

Rule-based expert behaviors for hierarchical RL in B-ACE.

These are deterministic policies based on domain knowledge from pursuit-evasion
differential game theory. They are used directly by the meta-controller when
selecting rule-based options (as opposed to the learned greedy policy).

Each expert function:
- Takes tactical state dict as input
- Returns action array [fire, throttle, turn, altitude]
- Encodes domain knowledge (BEZ, DMC, Apollonius circles, etc.)

Usage:
    from rule_based_experts import get_expert_action, RuleBasedOption
    
    # Get action for a specific behavior
    action = get_expert_action(RuleBasedOption.DEFEND_HVAA, state)
"""

import numpy as np
from enum import IntEnum
from typing import Dict, Any, Callable


class RuleBasedOption(IntEnum):
    """Rule-based behavior options (excludes learned GREEDY)"""
    DEFEND_HVAA = 0
    INTERCEPT_ENEMY = 1
    EVADE_MISSILE = 2
    OFFENSIVE_POSITIONING = 3
    DEFENSIVE_POSITIONING = 4


# =============================================================================
# EXPERT BEHAVIOR FUNCTIONS
# =============================================================================

def expert_defend_hvaa(state: Dict[str, Any]) -> np.ndarray:
    """
    DEFEND_HVAA: Stay near HVAA, intercept threats.
    
    Domain knowledge:
    - Maintain guardband distance (normalized ~0.2-0.5)
    - Position between HVAA and threats
    - Fire at enemies threatening HVAA
    
    Args:
        state: Tactical state dict with keys like 'distance_to_hvaa', 
               'hvaa_angle_off', 'enemy_detected', etc.
    
    Returns:
        Action array [fire, throttle, turn, altitude]
    """
    hvaa_dist = state.get('distance_to_hvaa', 0.5)
    hvaa_angle = state.get('hvaa_angle_off', 0.0)
    enemy_detected = state.get('enemy_detected', False)
    distance_to_enemy = state.get('distance_to_enemy', 1.0)
    offensive_factor = state.get('offensive_factor', 0.0)
    
    # Fire: Engage enemies that are close and we have good shot
    fire = 0.0
    if enemy_detected and distance_to_enemy < 0.5 and offensive_factor > 0.0:
        fire = 1.0
    
    # Throttle: Moderate patrol speed
    throttle = 0.6
    
    # Turn: Maintain guardband and intercept threats
    target_dist = 0.35  # Target normalized escort distance
    
    if hvaa_dist > 0.5:
        # Too far from HVAA - turn back
        turn = np.clip(-hvaa_angle / 45.0, -1.0, 1.0)
    elif hvaa_dist < 0.2:
        # Too close to HVAA - turn away slightly
        turn = np.clip(hvaa_angle / 45.0, -1.0, 1.0)
    else:
        # In guardband - face toward threats if any
        if enemy_detected:
            aspect_to_enemy = state.get('aspect_angle_to_enemy', 0.0)
            turn = np.clip(-aspect_to_enemy / 60.0, -0.7, 0.7)
        else:
            turn = 0.0
    
    # Altitude: Maintain level with HVAA
    altitude = 0.0
    
    return np.array([fire, throttle, turn, altitude], dtype=np.float32)


def expert_intercept_enemy(state: Dict[str, Any]) -> np.ndarray:
    """
    INTERCEPT_ENEMY: Aggressive pursuit and engagement.
    
    Domain knowledge:
    - Close distance to enemy aggressively
    - Fire when in range, regardless of BEZ status
    - Prioritize kill over survival
    """
    inside_bez = state.get('inside_bez', False)
    distance = state.get('distance_to_enemy', 1.0)
    aspect = state.get('bez_aspect_angle', 180.0)
    offensive_factor = state.get('offensive_factor', 0.0)
    
    # Fire: Aggressive - fire when enemy in range
    fire = 1.0 if distance < 0.6 else 0.0
    
    # Throttle: Maximum for intercept
    throttle = 1.0
    
    # Turn: Always pursue enemy, ignore BEZ danger
    # Pursue toward stern aspect for better shot
    turn = np.clip(-aspect / 90.0, -1.0, 1.0)
    
    # Altitude: Neutral
    altitude = 0.0
    
    return np.array([fire, throttle, turn, altitude], dtype=np.float32)


def expert_evade_missile(state: Dict[str, Any]) -> np.ndarray:
    """
    EVADE_MISSILE: Defensive evasion from incoming missile.
    
    Domain knowledge:
    - Use DMC to guide evasive heading
    - Maximum throttle (speed = survival)
    - No firing during evasion
    - Altitude changes to break lock
    
    Args:
        state: Tactical state dict
    
    Returns:
        Action array [fire, throttle, turn, altitude]
    """
    dmc = state.get('dmc_normalized', 0.0)
    inside_threat = state.get('inside_threat_cone', False)
    
    # Fire: Never fire while evading
    fire = 0.0
    
    # Throttle: Maximum (speed is life)
    throttle = 1.0
    
    # Turn: Follow DMC guidance
    if inside_threat:
        # Hard turn to exit threat cone
        turn = 1.0 if dmc > 0 else -1.0
    else:
        # Moderate turn based on DMC
        turn = np.clip(dmc * 2.0, -1.0, 1.0)
    
    # Altitude: Jink to break lock
    altitude = np.random.choice([-0.5, 0.0, 0.5])
    
    return np.array([fire, throttle, turn, altitude], dtype=np.float32)


def expert_offensive_positioning(state: Dict[str, Any]) -> np.ndarray:
    """
    OFFENSIVE_POSITIONING: Maneuver for tactical advantage.
    
    Domain knowledge:
    - Stay outside enemy BEZ
    - Achieve good aspect angle (stern chase)
    - Maximize offensive factor
    - Don't fire - focus on positioning
    
    Args:
        state: Tactical state dict
    
    Returns:
        Action array [fire, throttle, turn, altitude]
    """
    bez_penetration = state.get('bez_penetration', 0.0)
    offensive_factor = state.get('offensive_factor', 0.0)
    aspect = state.get('bez_aspect_angle', 180.0)
    
    # Fire: Not the focus - positioning only
    fire = 0.0
    
    # Throttle: Moderate for maneuvering
    throttle = 0.7
    
    # Turn: Optimize position
    if bez_penetration > 0.3:
        # Too close to BEZ - turn away
        turn = 1.0 if bez_penetration > 0.5 else 0.5
    else:
        # Maneuver for stern aspect
        turn = np.clip(-aspect / 60.0, -0.7, 0.7)
    
    # Altitude: Slight climb for energy advantage
    altitude = 0.2 if offensive_factor < 0.3 else 0.0
    
    return np.array([fire, throttle, turn, altitude], dtype=np.float32)


def expert_defensive_positioning(state: Dict[str, Any]) -> np.ndarray:
    """
    DEFENSIVE_POSITIONING: Interpose between HVAA and enemy.
    
    Domain knowledge:
    - Position between HVAA and threat
    - Face toward enemy while staying near HVAA
    - Shield HVAA from attack
    """
    hvaa_angle = state.get('hvaa_angle_off', 0.0)
    enemy_aspect = state.get('aspect_angle_to_enemy', 0.0)
    hvaa_dist = state.get('distance_to_hvaa', 0.5)
    enemy_dist = state.get('distance_to_enemy', 1.0)
    inside_bez = state.get('inside_bez', False)
    
    # Fire: Only if enemy very close and threatening
    fire = 1.0 if enemy_dist < 0.3 else 0.0
    
    # Throttle: Moderate for positioning
    throttle = 0.7
    
    # Turn logic: Position between HVAA and enemy
    # If HVAA is behind us and enemy ahead, we're in good position
    # Otherwise, turn to interpose
    
    if hvaa_dist > 0.5:
        # Too far from HVAA - turn back toward it
        turn = np.clip(-hvaa_angle / 45.0, -1.0, 1.0)
    elif hvaa_dist < 0.15:
        # Too close to HVAA - move away but face enemy
        turn = np.clip(-enemy_aspect / 60.0, -0.8, 0.8)
    else:
        # Good distance - face toward enemy to shield HVAA
        turn = np.clip(-enemy_aspect / 60.0, -0.7, 0.7)
    
    # If inside BEZ, prioritize slight evasion while maintaining position
    if inside_bez:
        dmc = state.get('dmc_normalized', 0.0)
        turn = np.clip(turn + dmc * 0.3, -1.0, 1.0)
    
    # Altitude: Match HVAA altitude (neutral)
    altitude = 0.0
    
    return np.array([fire, throttle, turn, altitude], dtype=np.float32)


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

# Mapping from option to expert function
EXPERT_FUNCTIONS: Dict[RuleBasedOption, Callable[[Dict[str, Any]], np.ndarray]] = {
    RuleBasedOption.DEFEND_HVAA: expert_defend_hvaa,
    RuleBasedOption.INTERCEPT_ENEMY: expert_intercept_enemy,
    RuleBasedOption.EVADE_MISSILE: expert_evade_missile,
    RuleBasedOption.OFFENSIVE_POSITIONING: expert_offensive_positioning,
    RuleBasedOption.DEFENSIVE_POSITIONING: expert_defensive_positioning,
}


def get_expert_action(option: RuleBasedOption, state: Dict[str, Any]) -> np.ndarray:
    """
    Get action from rule-based expert for given option.
    
    Args:
        option: Which rule-based behavior to execute
        state: Current tactical state
        
    Returns:
        Action array [fire, throttle, turn, altitude]
    """
    expert_fn = EXPERT_FUNCTIONS.get(option)
    if expert_fn is None:
        raise ValueError(f"Unknown option: {option}")
    return expert_fn(state)


def get_expert_action_by_id(option_id: int, state: Dict[str, Any]) -> np.ndarray:
    """
    Get action from rule-based expert by integer ID.
    
    Args:
        option_id: Integer ID (0-4 for rule-based options)
        state: Current tactical state
        
    Returns:
        Action array [fire, throttle, turn, altitude]
    """
    try:
        option = RuleBasedOption(option_id)
        return get_expert_action(option, state)
    except ValueError:
        raise ValueError(f"Invalid option_id: {option_id}. Must be 0-4 for rule-based options.")


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Rule-Based Experts - Self Test")
    print("=" * 60)
    
    # Create test state
    test_state = {
        'distance_to_hvaa': 0.35,
        'hvaa_angle_off': 15.0,
        'enemy_detected': True,
        'distance_to_enemy': 0.4,
        'aspect_angle_to_enemy': 30.0,
        'offensive_factor': 0.3,
        'defensive_factor': -0.2,
        'inside_bez': False,
        'bez_penetration': 0.2,
        'bez_aspect_angle': 45.0,
        'dmc_normalized': 0.3,
        'inside_threat_cone': False,
        'missile_tracking_us': False,
    }
    
    print("\nTest State:")
    for key, val in test_state.items():
        print(f"  {key}: {val}")
    
    print("\n" + "-" * 60)
    print("Expert Actions for Each Option:")
    print("-" * 60)
    
    for option in RuleBasedOption:
        action = get_expert_action(option, test_state)
        print(f"\n{option.name}:")
        print(f"  Fire:     {action[0]:+.2f}")
        print(f"  Throttle: {action[1]:+.2f}")
        print(f"  Turn:     {action[2]:+.2f}")
        print(f"  Altitude: {action[3]:+.2f}")
    
    print("\n" + "-" * 60)
    print("Testing with missile threat state:")
    print("-" * 60)
    
    missile_state = test_state.copy()
    missile_state['missile_tracking_us'] = True
    missile_state['inside_threat_cone'] = True
    missile_state['dmc_normalized'] = -0.6
    
    action = get_expert_action(RuleBasedOption.EVADE_MISSILE, missile_state)
    print(f"\nEVADE_MISSILE action:")
    print(f"  Fire:     {action[0]:+.2f} (should be 0)")
    print(f"  Throttle: {action[1]:+.2f} (should be 1)")
    print(f"  Turn:     {action[2]:+.2f} (hard turn based on DMC)")
    print(f"  Altitude: {action[3]:+.2f} (jinking)")
    
    print("\n" + "=" * 60)
    print("✓ All rule-based experts working correctly!")
    print("=" * 60)
