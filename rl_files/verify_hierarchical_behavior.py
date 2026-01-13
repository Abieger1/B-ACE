#!/usr/bin/env python3
"""
verify_hierarchical_behavior.py

Verification script to ensure tactical state extraction and expert behaviors
are working correctly with your 38-dim observation space.

Run this after your environment is created to verify everything works.
"""

import numpy as np
from tactical_options import extract_tactical_state, BASE_OBS_INDICES, ENRICHED_OFFSETS
from rule_based_experts import get_expert_action_by_id, RuleBasedOption

def verify_indices():
    """Verify index mapping for 38-dim observation."""
    print("=" * 70)
    print("INDEX VERIFICATION FOR 38-DIM OBSERVATION")
    print("=" * 70)
    
    print("\n--- Base Observation Indices (0-26) ---")
    for name, idx in sorted(BASE_OBS_INDICES.items(), key=lambda x: x[1]):
        print(f"  [{idx:2d}] {name}")
    
    print("\n--- Enriched Feature Indices (27-37) ---")
    enriched_start = 27
    for name, offset in sorted(ENRICHED_OFFSETS.items(), key=lambda x: x[1]):
        actual_idx = enriched_start + offset
        if actual_idx <= 37:  # Only show features that fit in 38 dims
            print(f"  [{actual_idx:2d}] {name} (offset +{offset})")
    
    print("\n--- Your 38-dim observation should be: ---")
    print("  [0-26]  Base B-ACE features (27 dims)")
    print("  [27-28] Apollonius: time_to_capture, capture_feasible")
    print("  [29-31] BEZ: bez_penetration, inside_bez, bez_aspect_angle")
    print("  [32-33] DMC: dmc_normalized, inside_threat_cone")
    print("  [34-35] Escape: always_escapes, always_captured")
    print("  [36-37] Offense: offensive_wez_dominance, offensive_ttc_score")


def test_state_extraction():
    """Test extract_tactical_state with realistic mock observation."""
    print("\n" + "=" * 70)
    print("STATE EXTRACTION TEST")
    print("=" * 70)
    
    # Create 38-dim mock observation
    mock_obs = np.zeros(38, dtype=np.float32)
    
    # Set base observation values
    mock_obs[5] = 0.3     # offensive_factor
    mock_obs[6] = -0.2    # defensive_factor
    mock_obs[7] = 1.0     # track_detected (enemy detected)
    mock_obs[8] = 0.4     # own_dist_target (distance to enemy)
    mock_obs[9] = 30.0    # own_aspect_angle_target
    mock_obs[14] = 0.0    # incoming_missile_track (no missile)
    mock_obs[18] = 0.35   # hvaa_dist
    mock_obs[19] = 15.0   # hvaa_angle_off
    mock_obs[21] = 1.0    # hvaa_detected
    
    # Set enriched values (indices 27+)
    mock_obs[27] = 0.8    # time_to_capture
    mock_obs[28] = 1.0    # capture_feasible
    mock_obs[29] = 0.25   # bez_penetration
    mock_obs[30] = 0.0    # inside_bez (False)
    mock_obs[31] = 45.0   # bez_aspect_angle
    mock_obs[32] = 0.3    # dmc_normalized
    mock_obs[33] = 0.0    # inside_threat_cone (False)
    mock_obs[34] = 0.0    # always_escapes
    mock_obs[35] = 0.0    # always_captured
    mock_obs[36] = 0.2    # offensive_wez_dominance
    mock_obs[37] = 0.1    # offensive_ttc_score
    
    print(f"\nMock observation shape: {mock_obs.shape}")
    
    # Extract state
    state = extract_tactical_state(mock_obs, obs_labels={}, enriched=True)
    
    print(f"\n--- Extracted State ({len(state)} keys) ---")
    
    # Verify critical values
    checks = [
        ('offensive_factor', 0.3),
        ('enemy_detected', True),
        ('distance_to_enemy', 0.4),
        ('distance_to_hvaa', 0.35),
        ('missile_tracking_us', False),
        ('bez_penetration', 0.25),
        ('inside_bez', False),
        ('dmc_normalized', 0.3),
        ('inside_threat_cone', False),
    ]
    
    all_passed = True
    for key, expected in checks:
        actual = state.get(key)
        if isinstance(expected, bool):
            passed = actual == expected
        else:
            passed = abs(actual - expected) < 0.01 if actual is not None else False
        
        status = "✓" if passed else "✗"
        print(f"  {status} {key}: {actual} (expected: {expected})")
        if not passed:
            all_passed = False
    
    return all_passed, state


def test_expert_behaviors():
    """Test expert behaviors with different tactical situations."""
    print("\n" + "=" * 70)
    print("EXPERT BEHAVIOR TEST")
    print("=" * 70)
    
    scenarios = [
        {
            'name': 'Normal patrol (enemy detected, good position)',
            'state': {
                'enemy_detected': True,
                'missile_tracking_us': False,
                'distance_to_hvaa': 0.35,
                'distance_to_enemy': 0.4,
                'offensive_factor': 0.3,
                'inside_bez': False,
                'bez_penetration': 0.1,
                'dmc_normalized': 0.2,
                'inside_threat_cone': False,
                'hvaa_angle_off': 10.0,
                'aspect_angle_to_enemy': 30.0,
                'hvaa_detected': True,
            },
            'expected_behaviors': {
                0: 'DEFEND_HVAA: Should patrol near HVAA, moderate throttle',
                1: 'INTERCEPT: Should pursue, high throttle, may fire',
                3: 'OFFENSIVE_POS: Should maneuver for advantage',
            }
        },
        {
            'name': 'Inside enemy BEZ (danger!)',
            'state': {
                'enemy_detected': True,
                'missile_tracking_us': False,
                'distance_to_hvaa': 0.4,
                'distance_to_enemy': 0.2,
                'offensive_factor': -0.3,
                'inside_bez': True,
                'bez_penetration': 0.7,
                'dmc_normalized': -0.5,
                'inside_threat_cone': True,
                'hvaa_angle_off': 20.0,
                'aspect_angle_to_enemy': 150.0,
                'hvaa_detected': True,
            },
            'expected_behaviors': {
                1: 'INTERCEPT: Should escape BEZ first (hard turn)',
                4: 'DEFENSIVE_POS: Should escape with max throttle',
            }
        },
        {
            'name': 'Missile incoming!',
            'state': {
                'enemy_detected': True,
                'missile_tracking_us': True,
                'distance_to_hvaa': 0.5,
                'distance_to_enemy': 0.3,
                'offensive_factor': 0.0,
                'inside_bez': False,
                'bez_penetration': 0.0,
                'dmc_normalized': -0.6,
                'inside_threat_cone': True,
                'hvaa_angle_off': 30.0,
                'aspect_angle_to_enemy': 90.0,
                'hvaa_detected': True,
            },
            'expected_behaviors': {
                2: 'EVADE: fire=0, throttle=1, hard turn based on DMC',
            }
        },
        {
            'name': 'Too far from HVAA',
            'state': {
                'enemy_detected': True,
                'missile_tracking_us': False,
                'distance_to_hvaa': 0.8,  # Far from HVAA!
                'distance_to_enemy': 0.6,
                'offensive_factor': 0.2,
                'inside_bez': False,
                'bez_penetration': 0.0,
                'dmc_normalized': 0.0,
                'inside_threat_cone': False,
                'hvaa_angle_off': 45.0,
                'aspect_angle_to_enemy': 60.0,
                'hvaa_detected': True,
            },
            'expected_behaviors': {
                0: 'DEFEND_HVAA: Should turn back toward HVAA',
            }
        },
    ]
    
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        state = scenario['state']
        
        print(f"  Key state: enemy={state['enemy_detected']}, missile={state['missile_tracking_us']}, "
              f"inside_bez={state['inside_bez']}, hvaa_dist={state['distance_to_hvaa']:.2f}")
        
        for option_id in range(5):
            action = get_expert_action_by_id(option_id, state)
            option_name = RuleBasedOption(option_id).name
            print(f"\n  Option {option_id} ({option_name}):")
            print(f"    fire={action[0]:+.1f}, throttle={action[1]:.1f}, "
                  f"turn={action[2]:+.2f}, alt={action[3]:+.2f}")
            
            # Check expected behaviors
            if option_id in scenario['expected_behaviors']:
                print(f"    Expected: {scenario['expected_behaviors'][option_id]}")


def test_option_selection_logic():
    """Test which options would be valid in different states."""
    print("\n" + "=" * 70)
    print("OPTION SELECTION LOGIC TEST")
    print("=" * 70)
    
    from tactical_options import create_tactical_options, get_valid_options
    
    options = create_tactical_options()
    
    test_states = [
        ('Normal combat', {
            'enemy_detected': True,
            'missile_tracking_us': False,
            'inside_bez': False,
            'hvaa_detected': True,
            'offensive_factor': 0.3,
            'bez_penetration': 0.1,
        }),
        ('Missile threat', {
            'enemy_detected': True,
            'missile_tracking_us': True,
            'inside_bez': False,
            'hvaa_detected': True,
            'offensive_factor': 0.0,
            'bez_penetration': 0.0,
        }),
        ('Inside BEZ', {
            'enemy_detected': True,
            'missile_tracking_us': False,
            'inside_bez': True,
            'hvaa_detected': True,
            'offensive_factor': -0.3,
            'bez_penetration': 0.6,
        }),
    ]
    
    for name, state in test_states:
        valid = get_valid_options(state, options)
        valid_names = [options[opt].name for opt in valid]
        print(f"\n  {name}:")
        print(f"    Valid options: {valid_names}")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL BEHAVIOR VERIFICATION")
    print("=" * 70)
    
    verify_indices()
    
    passed, state = test_state_extraction()
    
    if passed:
        print("\n✓ State extraction working correctly!")
    else:
        print("\n✗ State extraction has issues - check indices!")
        return False
    
    test_expert_behaviors()
    test_option_selection_logic()
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ All verifications passed!")
    else:
        print("\n✗ Some verifications failed - review output above")
