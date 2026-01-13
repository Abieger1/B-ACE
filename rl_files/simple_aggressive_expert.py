"""
Simple Aggressive Expert - Mimics baseline1 but slightly more aggressive.

Based on Fighter.gd baseline1 behavior:
- dShot = 0.85 (fire when offensive_factor > this)
- aspect_angle < 15 degrees for fire
- Turn toward enemy using radial bearing
- No cranking (removed for aggression)
- No evading (removed for aggression)
"""

import numpy as np
from typing import Dict, Any, Optional


class SimpleAggressiveExpert:
    """
    Simplified expert that:
    1. Flies north when no enemy detected (toward red spawn)
    2. Turns toward enemy when detected
    3. Fires aggressively when in range
    """
    
    def __init__(self, 
                 fire_threshold: float = 0.50,  # Lower than baseline1's 0.85
                 aspect_limit_deg: float = 30.0,  # Wider than baseline1's 15
                 debug: bool = False):
        self.fire_threshold = fire_threshold
        self.aspect_limit_deg = aspect_limit_deg
        self.aspect_limit_norm = aspect_limit_deg / 180.0
        self.debug = debug
        
        self.step_count = 0
        self.fire_cooldown = 0
        self.fire_cooldown_steps = 100
        
        # Pursuit memory - keep chasing after detection lost
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 500  # Keep pursuing for 500 steps after last detection
        self.last_known_angle_off = 0.0
        
    def reset(self):
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0
        
    def get_action(self, 
                   obs: np.ndarray, 
                   obs_indices: Dict[str, int]) -> np.ndarray:
        """Get action from observation."""
        self.step_count += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        n = len(obs)
        
        def get_val(name: str, default: float = 0.0) -> float:
            idx = obs_indices.get(name)
            if idx is not None and 0 <= idx < n:
                return float(obs[idx])
            return default
        
        # Read key values
        enemy_detected = get_val('enemy_detected', 0.0) > 0.5
        distance = get_val('distance_to_enemy', -1.0)
        angle_off = get_val('angle_off_to_enemy', 0.0)
        aspect = get_val('aspect_angle_to_enemy', 0.0)
        offensive = get_val('offensive_factor', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        
        # HVAA info for patrol
        hvaa_dist = get_val('distance_to_hvaa', 0.3)
        hvaa_angle = get_val('hvaa_angle_off', 0.0)
        
        # Valid detection = detected AND distance is valid (>= 0)
        valid_enemy = enemy_detected and distance >= 0
        
        # Update pursuit memory when we have valid detection
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off
        
        # Check if we should be in pursuit mode (recent detection)
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        if self.debug and self.step_count % 200 == 0:
            mode = " PURSUIT" if (in_pursuit and not valid_enemy) else ""
            print(f"[EXPERT] step={self.step_count} det={enemy_detected} dist={distance:.2f} "
                  f"angle_off={angle_off:.2f} aspect={aspect:.2f} off={offensive:.2f}{mode}")
        
        if valid_enemy:
            # === ENGAGE MODE - Have active detection ===
            if missile_in_flight:
                # MISSILE SUPPORT - very gentle turn to maintain lock
                turn = np.clip(-angle_off * 0.2, -0.15, 0.15)
                g_force = 0.3
            else:
                # NO MISSILE - close distance
                if abs(angle_off) < 0.5:
                    # Enemy roughly ahead - turn toward them
                    turn = np.clip(-angle_off * 1.5, -0.8, 0.8)
                    g_force = 0.7
                else:
                    # Enemy far off-boresight (behind/side)
                    # Turn toward HVAA to stay between HVAA and threat
                    # This prevents turning further away from the fight
                    if abs(hvaa_angle) > 0.1:
                        turn = np.clip(hvaa_angle * 1.5, -0.8, 0.8)
                    else:
                        # HVAA ahead, turn toward enemy
                        turn = np.clip(-angle_off * 1.0, -0.8, 0.8)
                    g_force = 0.8
            altitude = 0.1
            
            if self.debug:
                print(f"[ENGAGE] angle_off={angle_off:.2f} hvaa_angle={hvaa_angle:.2f} -> turn={turn:.2f}")
            
            # Fire decision - tighter range for better hits
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                dist_ok = 0.15 < distance < 0.40
                
                if dist_ok:
                    fire = 1.0
                    self.fire_cooldown = self.fire_cooldown_steps
                    if self.debug:
                        print(f"[FIRE!] dist={distance:.2f}")
                
        elif in_pursuit:
            # === PURSUIT MODE - Lost detection but recently saw enemy ===
            if missile_in_flight:
                # Missile in flight - fly nearly straight to support
                turn = np.clip(-self.last_known_angle_off * 0.15, -0.1, 0.1)
                g_force = 0.3
            else:
                # No missile - stay between HVAA and last known enemy position
                if abs(self.last_known_angle_off) < 0.5:
                    # Enemy was roughly ahead - continue toward them
                    turn = np.clip(-self.last_known_angle_off * 0.8, -0.5, 0.5)
                    g_force = 0.6
                else:
                    # Enemy was far off - turn toward HVAA to stay in fight
                    if abs(hvaa_angle) > 0.1:
                        turn = np.clip(hvaa_angle * 1.0, -0.6, 0.6)
                    else:
                        turn = 0.0  # HVAA ahead, fly straight
                    g_force = 0.5
            altitude = 0.0
            fire = 0.0
            
            if self.debug and self.step_count % 200 == 0:
                print(f"[PURSUIT] last_angle={self.last_known_angle_off:.2f} hvaa={hvaa_angle:.2f} -> turn={turn:.2f}")
                
        else:
            # === PATROL MODE ===
            if hvaa_dist > 0.10:
                turn = np.clip(hvaa_angle * 2.0, -1.0, 1.0)
                g_force = 0.6
                if self.debug and self.step_count % 200 == 0:
                    print(f"[PATROL->HVAA] dist={hvaa_dist:.2f} angle={hvaa_angle:.2f} turn={turn:.2f}")
            else:
                turn = 0.0
                g_force = 0.4
                if self.debug and self.step_count % 500 == 0:
                    print(f"[PATROL-FWD] hvaa_dist={hvaa_dist:.2f}")
            
            altitude = 0.0
            fire = 0.0
        
        return np.array([turn, altitude, g_force, fire], dtype=np.float32)


def get_default_obs_indices() -> Dict[str, int]:
    """
    Default observation indices matching Godot B-ACE output (27 dims).
    """
    return {
        # Own ship info (indices 0-8)
        'own_x': 0,
        'own_z': 1,
        'own_altitude': 2,
        'dist_to_target': 3,
        'aspect_to_target': 4,
        'own_hdg': 5,
        'own_speed': 6,
        'own_missiles': 7,
        'own_in_flight_missile': 8,
        
        # HVAA info (indices 9-13)
        'distance_to_hvaa': 9,
        'hvaa_alt_diff': 10,
        'hvaa_angle_off': 11,
        'hvaa_heading': 12,
        'hvaa_detected': 13,
        
        # Enemy track info (track 201 = red enemy)
        'altitude_diff_enemy': 14,
        'aspect_angle_to_enemy': 15,
        'angle_off_to_enemy': 16,
        'distance_to_enemy': 17,
        'dist2go_enemy': 18,
        'own_missile_rmax': 19,
        'own_missile_nez': 20,
        'enemy_missile_rmax': 21,
        'enemy_missile_nez': 22,
        'defensive_factor': 23,  # threat_factor
        'offensive_factor': 24,
        'is_missile_support': 25,
        'enemy_detected': 26,
    }
