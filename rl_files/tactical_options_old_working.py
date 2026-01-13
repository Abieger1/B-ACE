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
    
    Attributes:
        name: Human-readable option name
        option_id: Enum ID for this option
        initiation_fn: Function to check if option can start
        termination_fn: Function to check if option should end
        reward_bonus: Intrinsic reward for executing this option (per step)
        description: Detailed description of tactical behavior
        is_learned: True if this is a learned policy (GREEDY), False for rule-based
    """
    name: str
    option_id: TacticalOption
    initiation_fn: Callable[[Dict[str, Any]], bool]
    termination_fn: Callable[[Dict[str, Any]], bool]
    reward_bonus: float = 0.0
    description: str = ""
    is_learned: bool = False
    
    def can_initiate(self, state: Dict[str, Any]) -> bool:
        """Check if option can be initiated given current state"""
        return self.initiation_fn(state)
    
    def should_terminate(self, state: Dict[str, Any]) -> bool:
        """Check if option should terminate given current state"""
        return self.termination_fn(state)


def extract_tactical_state(obs: np.ndarray, obs_labels: Dict[str, int], 
                           enriched: bool = True) -> Dict[str, Any]:
    """
    Extract tactical state information from observation for option logic.
    
    Args:
        obs: Raw observation array
        obs_labels: Label to index mapping
        enriched: Whether observation includes enriched features
        
    Returns:
        Dictionary with tactical state information
    """
    state = {}
    
    # Basic observations (indices 0-21 in base observation)
    if "own_dist_target" in obs_labels:
        state['distance_to_enemy'] = obs[obs_labels["own_dist_target"]]
    
    if "own_aspect_angle_target" in obs_labels:
        state['aspect_angle_to_enemy'] = obs[obs_labels["own_aspect_angle_target"]]
    
    # HVAA information (if escort mission)
    if "hvaa_dist" in obs_labels:
        state['distance_to_hvaa'] = obs[obs_labels["hvaa_dist"]]
    
    if "hvaa_angle_off" in obs_labels:
        state['hvaa_angle_off'] = obs[obs_labels["hvaa_angle_off"]]
    
    if "hvaa_detected" in obs_labels:
        state['hvaa_detected'] = obs[obs_labels["hvaa_detected"]] > 0.5
    
    # Missile tracking information
    if "incoming_missile_track" in obs_labels:
        state['missile_tracking_us'] = obs[obs_labels["incoming_missile_track"]] > 0.5
    
    # Track detection
    if "track_detected" in obs_labels:
        state['enemy_detected'] = obs[obs_labels["track_detected"]] > 0.5
    
    # Offensive/defensive factors from Godot
    if "offensive_factor" in obs_labels:
        state['offensive_factor'] = obs[obs_labels["offensive_factor"]]
    
    if "defensive_factor" in obs_labels:
        state['defensive_factor'] = obs[obs_labels["defensive_factor"]]
    
    # If using enriched observations, extract theoretical features
    if enriched and len(obs) > 22:
        enriched_start = 22
        
        # Apollonius features (indices 22-23)
        state['time_to_capture'] = obs[enriched_start + 0]
        state['capture_feasible'] = obs[enriched_start + 1] > 0.5
        
        # BEZ features (indices 24-26)
        state['bez_penetration'] = obs[enriched_start + 2]
        state['inside_bez'] = obs[enriched_start + 3] > 0.5
        state['bez_aspect_angle'] = obs[enriched_start + 4]
        
        # DMC features (indices 27-28)
        state['dmc_normalized'] = obs[enriched_start + 5]
        state['inside_threat_cone'] = obs[enriched_start + 6] > 0.5
        
        # Escape feasibility (indices 29-30)
        state['always_escapes'] = obs[enriched_start + 7] > 0.5
        state['always_captured'] = obs[enriched_start + 8] > 0.5
        
        # Offensive metrics (indices 31-32, if enabled)
        if len(obs) > enriched_start + 9:
            state['offensive_wez_dominance'] = obs[enriched_start + 9]
            state['offensive_ttc_score'] = obs[enriched_start + 10]
    
    return state


# =============================================================================
# OPTION DEFINITIONS
# =============================================================================

def create_defend_hvaa_option() -> Option:
    """
    DEFEND_HVAA: Defensive escort mission (RULE-BASED)
    
    Tactical Behavior:
    - Stay within guardband distance of HVAA
    - Position to intercept threats approaching HVAA
    - Prioritize HVAA survival over offensive actions
    """
    def initiation(state: Dict[str, Any]) -> bool:
        return True  # Can always defend HVAA
    
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
    """
    INTERCEPT_ENEMY: Aggressive pursuit (RULE-BASED)
    
    Tactical Behavior:
    - Close distance to enemy
    - Maneuver for offensive advantage (WEZ/BEZ dominance)
    - Prioritize enemy destruction
    """
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
        description="Aggressively pursue and engage enemy",
        is_learned=False
    )


def create_evade_missile_option() -> Option:
    """
    EVADE_MISSILE: Defensive evasion (RULE-BASED)
    
    Tactical Behavior:
    - Execute defensive maneuvers to break missile lock
    - Use DMC to guide evasive heading
    - Prioritize survival over offense
    """
    def initiation(state: Dict[str, Any]) -> bool:
        return state.get('missile_tracking_us', False)
    
    def termination(state: Dict[str, Any]) -> bool:
        return not state.get('missile_tracking_us', False)
    
    return Option(
        name="Evade Missile",
        option_id=TacticalOption.EVADE_MISSILE,
        initiation_fn=initiation,
        termination_fn=termination,
        reward_bonus=0.0,
        description="Defensive maneuvers to evade incoming missile",
        is_learned=False
    )


def create_offensive_positioning_option() -> Option:
    """
    OFFENSIVE_POSITIONING: Maneuver for tactical advantage (RULE-BASED)
    
    Tactical Behavior:
    - Position to maximize WEZ coverage
    - Stay outside enemy BEZ while threatening them
    - Use Apollonius circle geometry for optimal positioning
    """
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
    """
    DEFENSIVE_POSITIONING: Maintain defensive geometry (RULE-BASED)
    
    Tactical Behavior:
    - Stay outside enemy BEZ
    - Maintain safe distance based on speed ratio
    - Use DMC to guide defensive maneuvers
    """
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
    """
    GREEDY: Learned policy (PPO-TRAINED)
    
    This option uses the learned greedy policy instead of rule-based behavior.
    It can always be initiated and never self-terminates (meta-controller decides).
    
    The greedy policy is trained alongside the meta-controller and gradually
    takes over from rule-based options via curriculum.
    """
    def initiation(state: Dict[str, Any]) -> bool:
        return True  # Can always use greedy policy
    
    def termination(state: Dict[str, Any]) -> bool:
        return False  # Never self-terminates; meta-controller decides
    
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
    """
    Factory function to create all tactical options.
    
    Returns:
        Dictionary mapping option IDs to Option objects
    """
    options = {
        TacticalOption.DEFEND_HVAA: create_defend_hvaa_option(),
        TacticalOption.INTERCEPT_ENEMY: create_intercept_enemy_option(),
        TacticalOption.EVADE_MISSILE: create_evade_missile_option(),
        TacticalOption.OFFENSIVE_POSITIONING: create_offensive_positioning_option(),
        TacticalOption.DEFENSIVE_POSITIONING: create_defensive_positioning_option(),
        TacticalOption.GREEDY: create_greedy_option(),
    }
    return options


def create_rule_based_options() -> Dict[TacticalOption, Option]:
    """
    Create only rule-based options (excludes GREEDY).
    
    Returns:
        Dictionary mapping option IDs to Option objects (0-4 only)
    """
    all_options = create_tactical_options()
    return {k: v for k, v in all_options.items() if not v.is_learned}


def get_valid_options(state: Dict[str, Any], 
                      all_options: Dict[TacticalOption, Option]) -> List[TacticalOption]:
    """
    Get list of options that can be initiated in current state.
    
    Args:
        state: Current tactical state
        all_options: Dictionary of all available options
        
    Returns:
        List of valid option IDs
    """
    valid = []
    for opt_id, option in all_options.items():
        if option.can_initiate(state):
            valid.append(opt_id)
    
    # Always allow DEFEND_HVAA as fallback for rule-based
    if TacticalOption.DEFEND_HVAA not in valid and TacticalOption.DEFEND_HVAA in all_options:
        valid.append(TacticalOption.DEFEND_HVAA)
    
    # GREEDY is always valid if present
    if TacticalOption.GREEDY in all_options and TacticalOption.GREEDY not in valid:
        valid.append(TacticalOption.GREEDY)
    
    return valid


# =============================================================================
# OPTION SUMMARY
# =============================================================================

def print_option_summary(options: Dict[TacticalOption, Option]):
    """Print summary of all tactical options."""
    print("\n" + "="*70)
    print("TACTICAL OPTIONS FOR HIERARCHICAL RL")
    print("="*70)
    
    print("\nRULE-BASED OPTIONS (0-4):")
    print("-" * 70)
    for opt_id, option in options.items():
        if not option.is_learned:
            print(f"\n  {option.name} (ID: {int(opt_id)})")
            print(f"    Description: {option.description}")
            print(f"    Reward Bonus: {option.reward_bonus:+.4f} per step")
    
    print("\n" + "-" * 70)
    print("LEARNED OPTIONS:")
    print("-" * 70)
    for opt_id, option in options.items():
        if option.is_learned:
            print(f"\n  {option.name} (ID: {int(opt_id)})")
            print(f"    Description: {option.description}")
    
    print("\n" + "="*70)
    print(f"Total: {NUM_RULE_BASED} rule-based + 1 learned = {NUM_OPTIONS} options")
    print("="*70)


if __name__ == "__main__":
    # Test option creation
    options = create_tactical_options()
    print_option_summary(options)
    
    # Test helper functions
    print("\nTesting helper functions:")
    print(f"  is_rule_based(0): {is_rule_based(0)} (expected: True)")
    print(f"  is_rule_based(4): {is_rule_based(4)} (expected: True)")
    print(f"  is_rule_based(5): {is_rule_based(5)} (expected: False)")
    print(f"  is_greedy(5): {is_greedy(5)} (expected: True)")
    print(f"  is_greedy(0): {is_greedy(0)} (expected: False)")
    
    # Test state extraction and validation
    print("\nTesting option initiation logic...")
    
    dummy_state = {
        'enemy_detected': True,
        'missile_tracking_us': False,
        'hvaa_detected': True,
        'inside_bez': False,
        'offensive_factor': 0.3,
        'bez_penetration': 0.5,
    }
    
    valid = get_valid_options(dummy_state, options)
    print(f"\nValid options for test state: {[options[opt].name for opt in valid]}")
    
    # Test rule-based only factory
    rb_options = create_rule_based_options()
    print(f"\nRule-based options only: {[opt.name for opt in rb_options.values()]}")
    
    print("\n✓ Tactical options module ready!")
