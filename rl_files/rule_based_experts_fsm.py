#!/usr/bin/env python3
"""
rule_based_experts.py

Rule-based expert behaviors for hierarchical RL in B-ACE.
Based on baseline1 behavior from Fighter.gd (Godot).

KEY THRESHOLDS FROM BASELINE1:
- dShot  = 0.85  (fire when offensive_factor > 0.85)
- lCrank = 0.60  (crank when threat_factor > 0.6)
- lBreak = 0.95  (break/evade when threat_factor > 0.95)

FIRING RULES:
- offensive_factor > 0.85
- |aspect_angle| < 15° (60° for HVAA targets)
- No missile already in flight supporting this target

CRANK MANEUVER:
- When threat_factor > 0.6: turn 50° off target heading
- Maintains radar lock while increasing range

BREAK/EVADE:
- When threat_factor > 0.95: turn 180° from target, max G
- 80 second timeout before re-engaging

Usage:
    from rule_based_experts import get_expert_action, RuleBasedOption
    action = get_expert_action(RuleBasedOption.INTERCEPT_ENEMY, state)
"""

import numpy as np
from enum import IntEnum
from typing import Dict, Any, Callable


class RuleBasedOption(IntEnum):
    """Rule-based behavior options"""
    DEFEND_HVAA = 0
    INTERCEPT_ENEMY = 1
    EVADE_MISSILE = 2
    OFFENSIVE_POSITIONING = 3
    DEFENSIVE_POSITIONING = 4


# =============================================================================
# BASELINE1 THRESHOLDS (from Fighter.gd)
# =============================================================================

DSHOT = 0.85    # Fire threshold for offensive_factor
LCRANK = 0.60   # Crank threshold for threat_factor  
LBREAK = 0.95   # Break/evade threshold for threat_factor

CRANK_ANGLE = 50.0   # Degrees to turn off during crank (normalized: 50/180)
ASPECT_LIMIT = 15.0  # Max aspect angle for firing (degrees, normalized: 15/180)
HVAA_ASPECT_LIMIT = 60.0  # Wider aspect limit for HVAA targets


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def should_fire(state: Dict[str, Any], is_hvaa_target: bool = False) -> bool:
    """
    Determine if firing conditions are met (baseline1 logic).
    
    Conditions:
    - offensive_factor > DSHOT (0.85)
    - |aspect_angle| < 15° (or 60° for HVAA)
    - No missile already supporting this target
    - Have missiles remaining
    """
    offensive = state.get('offensive_factor', 0.0)
    aspect = abs(state.get('aspect_angle_to_enemy', 180.0))
    missile_in_flight = state.get('own_in_flight_missile', 0.0) > 0.5
    has_missiles = state.get('own_missiles', 0.0) > 0.0
    
    # Check offensive threshold
    if offensive <= DSHOT:
        return False
    
    # Check aspect angle (normalized values, so convert threshold)
    aspect_limit = HVAA_ASPECT_LIMIT if is_hvaa_target else ASPECT_LIMIT
    # Assuming aspect is normalized to [-1, 1] representing [-180, 180]
    # 15° / 180° ≈ 0.083, 60° / 180° ≈ 0.333
    aspect_limit_normalized = aspect_limit / 180.0
    if aspect > aspect_limit_normalized:
        return False
    
    # Don't fire if already supporting a missile
    if missile_in_flight:
        return False
    
    # Need missiles
    if not has_missiles:
        return False
    
    return True


def should_crank(state: Dict[str, Any]) -> bool:
    """Check if crank maneuver should be executed (threat > 0.6)."""
    threat = state.get('defensive_factor', 0.0)
    return threat > LCRANK


def should_break(state: Dict[str, Any], supporting_missile: bool = False) -> bool:
    """
    Check if break/evade should be executed (threat > 0.95).
    
    Note: baseline1 adds 0.5 to threshold if supporting a missile,
    making it harder to break while guiding.
    """
    threat = state.get('defensive_factor', 0.0)
    threshold = LBREAK + (0.5 if supporting_missile else 0.0)
    return threat > threshold


def get_crank_turn(state: Dict[str, Any], defense_side: int = 1) -> float:
    """
    Calculate turn value for crank maneuver.
    
    Crank = heading toward target + 50° offset
    defense_side: 1 or -1 to determine which way to crank
    """
    # In normalized action space [-1, 1], 50° ≈ 50/180 ≈ 0.28
    crank_offset = (CRANK_ANGLE / 180.0) * defense_side
    
    # Base turn toward target (using aspect angle as proxy)
    aspect = state.get('aspect_angle_to_enemy', 0.0)
    base_turn = np.clip(-aspect, -1.0, 1.0)
    
    # Add crank offset
    return np.clip(base_turn + crank_offset, -1.0, 1.0)


def get_break_turn(state: Dict[str, Any]) -> float:
    """
    Calculate turn value for break maneuver.
    Turn 180° from target (opposite of target heading).
    """
    aspect = state.get('aspect_angle_to_enemy', 0.0)
    # Turn away from target (opposite direction)
    return np.clip(aspect + np.sign(aspect) * 1.0, -1.0, 1.0)


# =============================================================================
# EXPERT BEHAVIOR FUNCTIONS
# =============================================================================

def expert_defend_hvaa(state: Dict[str, Any]) -> np.ndarray:
    """
    DEFEND_HVAA: Stay near HVAA, intercept threats.
    
    Logic:
    - Maintain escort distance (guardband)
    - Fire at enemies threatening HVAA when conditions met
    - Crank if threat > 0.6
    - Break if threat > 0.95
    """
    hvaa_dist = state.get('distance_to_hvaa', 0.5)
    hvaa_angle = state.get('hvaa_angle_off', 0.0)
    enemy_detected = state.get('enemy_detected', False)
    supporting_missile = state.get('own_in_flight_missile', 0.0) > 0.5
    
    # Fire decision (baseline1 rules)
    fire = 1.0 if (enemy_detected and should_fire(state)) else 0.0
    
    # Break check first (highest priority)
    if should_break(state, supporting_missile):
        return np.array([0.0, 1.0, get_break_turn(state), 0.0], dtype=np.float32)
    
    # Crank check
    if enemy_detected and should_crank(state):
        defense_side = 1 if hvaa_angle >= 0 else -1  # Crank toward HVAA
        turn = get_crank_turn(state, defense_side)
        return np.array([fire, 0.8, turn, 0.0], dtype=np.float32)
    
    # Normal patrol - maintain guardband
    throttle = 0.6
    
    if hvaa_dist > 0.5:
        # Too far - turn back toward HVAA
        turn = np.clip(-hvaa_angle * 2.0, -1.0, 1.0)
    elif hvaa_dist < 0.15:
        # Too close - turn away
        turn = np.clip(hvaa_angle * 2.0, -0.5, 0.5)
    else:
        # Good distance - face threats
        if enemy_detected:
            aspect = state.get('aspect_angle_to_enemy', 0.0)
            turn = np.clip(-aspect * 1.5, -0.7, 0.7)
        else:
            turn = 0.0
    
    return np.array([fire, throttle, turn, 0.0], dtype=np.float32)


def expert_intercept_enemy(state: Dict[str, Any]) -> np.ndarray:
    """
    INTERCEPT_ENEMY: Aggressive pursuit and engagement (baseline1 Engage state).
    
    Logic:
    - Head toward target
    - Fire when offensive_factor > 0.85 and aspect < 15°
    - Crank when threat > 0.6
    - Break when threat > 0.95
    """
    enemy_detected = state.get('enemy_detected', False)
    supporting_missile = state.get('own_in_flight_missile', 0.0) > 0.5
    aspect = state.get('aspect_angle_to_enemy', 0.0)
    
    # Fire decision (strict baseline1 rules)
    fire = 1.0 if (enemy_detected and should_fire(state)) else 0.0
    
    # Break check (highest priority)
    if should_break(state, supporting_missile):
        return np.array([0.0, 1.0, get_break_turn(state), 0.0], dtype=np.float32)
    
    # Crank check
    if should_crank(state):
        defense_side = np.random.choice([-1, 1])  # Random crank side
        turn = get_crank_turn(state, defense_side)
        throttle = 0.8
    else:
        # Pure pursuit - head straight for target
        turn = np.clip(-aspect * 2.0, -1.0, 1.0)
        throttle = 1.0
    
    return np.array([fire, throttle, turn, 0.0], dtype=np.float32)


def expert_evade_missile(state: Dict[str, Any]) -> np.ndarray:
    """
    EVADE_MISSILE: Defensive evasion (baseline1 Evade state).
    
    Logic:
    - Turn 180° from threat
    - Max throttle (G=6 in baseline1)
    - No firing during evasion
    - Use DMC for optimal escape heading
    """
    dmc = state.get('dmc_normalized', 0.0)
    inside_threat = state.get('inside_threat_cone', False)
    always_escapes = state.get('always_escapes', False)
    
    fire = 0.0  # Never fire while evading
    throttle = 1.0  # Max speed
    
    if always_escapes:
        # Already safe - moderate turn to maintain escape
        turn = np.clip(dmc * 1.0, -0.7, 0.7)
    elif inside_threat:
        # Hard turn to exit threat cone (baseline1: turn 180° from target)
        turn = 1.0 if dmc > 0 else -1.0
    else:
        # Follow DMC guidance
        turn = np.clip(dmc * 2.0, -1.0, 1.0)
    
    # Altitude jink for missile defense
    altitude = np.random.choice([-0.5, 0.0, 0.5])
    
    return np.array([fire, throttle, turn, altitude], dtype=np.float32)


def expert_offensive_positioning(state: Dict[str, Any]) -> np.ndarray:
    """
    OFFENSIVE_POSITIONING: Maneuver for shot opportunity.
    
    Logic:
    - Get into firing position (offensive_factor > 0.85)
    - Achieve good aspect angle (< 15°)
    - Stay outside enemy BEZ
    - Don't fire until properly positioned
    """
    bez_penetration = state.get('bez_penetration', 0.0)
    offensive = state.get('offensive_factor', 0.0)
    aspect = state.get('aspect_angle_to_enemy', 0.0)
    inside_bez = state.get('inside_bez', False)
    
    fire = 0.0  # Focus on positioning, not firing
    
    # If inside BEZ, priority is escape
    if inside_bez or bez_penetration > 0.3:
        dmc = state.get('dmc_normalized', 0.0)
        turn = 1.0 if dmc > 0 else -1.0
        throttle = 1.0
        altitude = 0.3
    else:
        # Maneuver toward stern aspect for good shot
        # Need offensive > 0.85 and aspect < 15° (normalized: 0.083)
        target_aspect = 0.0  # Stern chase
        aspect_error = aspect - target_aspect
        turn = np.clip(-aspect_error * 3.0, -0.7, 0.7)
        
        # Moderate throttle for positioning
        throttle = 0.7
        
        # Slight climb for energy advantage if not offensive enough
        altitude = 0.2 if offensive < DSHOT else 0.0
    
    return np.array([fire, throttle, turn, altitude], dtype=np.float32)


def expert_defensive_positioning(state: Dict[str, Any]) -> np.ndarray:
    """
    DEFENSIVE_POSITIONING: Interpose between HVAA and threat.
    
    Logic:
    - Position between HVAA and enemy
    - Face toward threat while staying near HVAA
    - Exit BEZ if inside
    - Fire only if enemy very close
    """
    hvaa_dist = state.get('distance_to_hvaa', 0.5)
    hvaa_angle = state.get('hvaa_angle_off', 0.0)
    enemy_aspect = state.get('aspect_angle_to_enemy', 0.0)
    enemy_dist = state.get('distance_to_enemy', 1.0)
    inside_bez = state.get('inside_bez', False)
    dmc = state.get('dmc_normalized', 0.0)
    
    # Fire only in emergency (enemy very close)
    fire = 1.0 if enemy_dist < 0.2 else 0.0
    
    # Priority: Exit BEZ
    if inside_bez:
        turn = 1.0 if dmc > 0 else -1.0
        throttle = 1.0
        altitude = 0.3
        return np.array([fire, throttle, turn, altitude], dtype=np.float32)
    
    throttle = 0.7
    
    # Position between HVAA and enemy
    if hvaa_dist > 0.5:
        # Too far from HVAA - return
        turn = np.clip(-hvaa_angle * 2.0, -1.0, 1.0)
    elif hvaa_dist < 0.1:
        # Too close - move out but face enemy
        turn = np.clip(-enemy_aspect * 1.5, -0.8, 0.8)
    else:
        # Good position - face toward enemy to shield HVAA
        turn = np.clip(-enemy_aspect * 1.5, -0.7, 0.7)
    
    altitude = 0.0
    return np.array([fire, throttle, turn, altitude], dtype=np.float32)


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

EXPERT_FUNCTIONS: Dict[RuleBasedOption, Callable[[Dict[str, Any]], np.ndarray]] = {
    RuleBasedOption.DEFEND_HVAA: expert_defend_hvaa,
    RuleBasedOption.INTERCEPT_ENEMY: expert_intercept_enemy,
    RuleBasedOption.EVADE_MISSILE: expert_evade_missile,
    RuleBasedOption.OFFENSIVE_POSITIONING: expert_offensive_positioning,
    RuleBasedOption.DEFENSIVE_POSITIONING: expert_defensive_positioning,
}


def get_expert_action(option: RuleBasedOption, state: Dict[str, Any]) -> np.ndarray:
    """Get action from rule-based expert for given option."""
    expert_fn = EXPERT_FUNCTIONS.get(option)
    if expert_fn is None:
        raise ValueError(f"Unknown option: {option}")
    return expert_fn(state)


def get_expert_action_by_id(option_id: int, state: Dict[str, Any]) -> np.ndarray:
    """Get action from rule-based expert by integer ID (0-4)."""
    try:
        option = RuleBasedOption(option_id)
        return get_expert_action(option, state)
    except ValueError:
        raise ValueError(f"Invalid option_id: {option_id}. Must be 0-4.")


# =============================================================================
# OPTION SELECTION (baseline1 state machine logic)
# =============================================================================

def select_expert_option(state: Dict[str, Any]) -> RuleBasedOption:
    """
    Select which expert to use based on state (baseline1 priority logic).
    
    Priority order:
    1. EVADE_MISSILE - if missile tracking us
    2. DEFENSIVE_POSITIONING - if inside enemy BEZ  
    3. DEFEND_HVAA - if too far from HVAA
    4. INTERCEPT_ENEMY - if good offensive position
    5. OFFENSIVE_POSITIONING - default
    """
    # Priority 1: Evade if missile tracking
    if state.get('missile_tracking_us', False):
        return RuleBasedOption.EVADE_MISSILE
    
    # Priority 2: Defensive if inside BEZ
    if state.get('inside_bez', False):
        return RuleBasedOption.DEFENSIVE_POSITIONING
    
    # Priority 3: Check HVAA distance
    hvaa_dist = state.get('distance_to_hvaa', 0.5)
    if hvaa_dist > 0.4:  # Too far from HVAA
        return RuleBasedOption.DEFEND_HVAA
    
    # Priority 4: Offensive engagement if good position
    offensive = state.get('offensive_factor', 0.0)
    if offensive > DSHOT * 0.8:  # Close to firing threshold
        return RuleBasedOption.INTERCEPT_ENEMY
    
    # Priority 5: Break if high threat
    threat = state.get('defensive_factor', 0.0)
    if threat > LBREAK:
        return RuleBasedOption.EVADE_MISSILE
    
    # Default: Offensive positioning
    return RuleBasedOption.OFFENSIVE_POSITIONING


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Rule-Based Experts (Baseline1 Logic) - Self Test")
    print("=" * 60)
    print(f"\nBaseline1 Thresholds:")
    print(f"  DSHOT  = {DSHOT}  (fire when offensive > this)")
    print(f"  LCRANK = {LCRANK}  (crank when threat > this)")
    print(f"  LBREAK = {LBREAK}  (break when threat > this)")
    
    # Test state with good firing position
    test_state_fire = {
        'distance_to_hvaa': 0.35,
        'hvaa_angle_off': 0.1,
        'enemy_detected': True,
        'distance_to_enemy': 0.4,
        'aspect_angle_to_enemy': 0.05,  # ~9° (normalized)
        'offensive_factor': 0.9,  # > DSHOT
        'defensive_factor': 0.3,  # < LCRANK
        'inside_bez': False,
        'bez_penetration': 0.1,
        'dmc_normalized': 0.2,
        'own_missiles': 6.0,
        'own_in_flight_missile': 0.0,
    }
    
    print("\n" + "-" * 60)
    print("Test: Good firing position (offensive=0.9, aspect=9°)")
    print("-" * 60)
    print(f"  should_fire: {should_fire(test_state_fire)}")  # Should be True
    print(f"  should_crank: {should_crank(test_state_fire)}")  # Should be False
    print(f"  should_break: {should_break(test_state_fire)}")  # Should be False
    
    action = get_expert_action(RuleBasedOption.INTERCEPT_ENEMY, test_state_fire)
    print(f"\n  INTERCEPT action: fire={action[0]:.1f}, throttle={action[1]:.1f}, "
          f"turn={action[2]:.2f}, alt={action[3]:.2f}")
    
    # Test state requiring crank
    test_state_crank = test_state_fire.copy()
    test_state_crank['defensive_factor'] = 0.7  # > LCRANK
    
    print("\n" + "-" * 60)
    print("Test: Crank required (threat=0.7)")
    print("-" * 60)
    print(f"  should_crank: {should_crank(test_state_crank)}")  # Should be True
    
    action = get_expert_action(RuleBasedOption.INTERCEPT_ENEMY, test_state_crank)
    print(f"  INTERCEPT action: fire={action[0]:.1f}, throttle={action[1]:.1f}, "
          f"turn={action[2]:.2f} (should be offset ~0.28)")
    
    # Test state requiring break
    test_state_break = test_state_fire.copy()
    test_state_break['defensive_factor'] = 0.98  # > LBREAK
    
    print("\n" + "-" * 60)
    print("Test: Break required (threat=0.98)")
    print("-" * 60)
    print(f"  should_break: {should_break(test_state_break)}")  # Should be True
    
    action = get_expert_action(RuleBasedOption.INTERCEPT_ENEMY, test_state_break)
    print(f"  INTERCEPT action: fire={action[0]:.1f} (0=no fire during break), "
          f"throttle={action[1]:.1f} (max), turn={action[2]:.2f} (turn away)")
    
    # Test aspect angle too wide
    test_state_wide_aspect = test_state_fire.copy()
    test_state_wide_aspect['aspect_angle_to_enemy'] = 0.2  # ~36° > 15°
    
    print("\n" + "-" * 60)
    print("Test: Aspect too wide (36°, limit is 15°)")
    print("-" * 60)
    print(f"  should_fire: {should_fire(test_state_wide_aspect)}")  # Should be False
    
    print("\n" + "=" * 60)
    print("✓ Baseline1 expert logic working correctly!")
