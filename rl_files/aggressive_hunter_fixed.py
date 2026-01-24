"""
Aggressive Hunter Expert - TRUE KILL PRIORITY VERSION

Design philosophy: DESTROY RED AT ALL COSTS - NO ORBITING
- Always point directly at enemy and fly toward them
- NO defensive maneuvers, NO wide turns, NO survivability logic
- Fire whenever in range and roughly pointed at enemy
- Simple pure pursuit: turn toward enemy, fly at them, shoot

Behavior: Identify -> Fire -> Fly STRAIGHT at red -> Fire again -> repeat until dead

Key changes from previous version:
- REMOVED the "enemy behind" wide-turn logic (this caused orbiting)
- Single pursuit mode: always turn toward enemy at same rate
- Higher g_force to maintain closure speed
- No turn commitment needed since we just always pursue
"""

import numpy as np
from typing import Dict, Any, Optional


class AggressiveHunterFixed:
    """
    TRUE KILL-PRIORITY aggressive hunter.
    
    Key fix: REMOVED all "enemy behind" special handling.
    Now it just always turns toward the enemy and flies at them.
    No more orbiting because there's no mode-switching.
    """
    
    def __init__(self,
                 # Fire parameters
                 fire_range_min: float = 0.08,
                 fire_range_max: float = 0.40,
                 fire_cooldown_steps: int = 100,
                 fire_cone: float = 0.55,  # ~100 degrees total
                 
                 # Pursuit - ONE mode, always the same
                 pursuit_gain: float = 2.0,  # How aggressively to turn toward enemy
                 max_turn_rate: float = 0.6,  # Single turn rate - no slow/wide turns
                 angle_deadband: float = 0.02,  # Fly straight when nearly aligned
                 
                 debug: bool = False):
        
        self.fire_range_min = fire_range_min
        self.fire_range_max = fire_range_max
        self.fire_cooldown_steps = fire_cooldown_steps
        self.fire_cone = fire_cone
        self.pursuit_gain = pursuit_gain
        self.max_turn_rate = max_turn_rate
        self.angle_deadband = angle_deadband
        self.debug = debug
        
        self.step_count = 0
        self.fire_cooldown = 0
        
        # Pursuit memory - continue toward last known position if lost
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 400
        self.last_known_angle_off = 0.0
        
        # Stats
        self.fire_count = 0
        self.engage_count = 0
        
    def reset(self):
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_steps': self.step_count,
            'fire_count': self.fire_count,
            'engage_count': self.engage_count,
        }
        
    def get_action(self, obs: np.ndarray, obs_indices: Dict[str, int]) -> np.ndarray:
        self.step_count += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        n = len(obs)
        
        def get_val(name: str, default: float = 0.0) -> float:
            idx = obs_indices.get(name)
            if idx is not None and 0 <= idx < n:
                return float(obs[idx])
            return default
        
        # Read observations
        enemy_detected = get_val('enemy_detected', 0.0) > 0.5
        distance = get_val('distance_to_enemy', -1.0)
        angle_off = get_val('angle_off_to_enemy', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        own_hdg = get_val('own_hdg', 0.0)
        
        valid_enemy = enemy_detected and distance >= 0
        
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off
        
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        # ================================================================
        # ENGAGE MODE - PURE PURSUIT - NO MODE SWITCHING
        # ================================================================
        if valid_enemy:
            self.engage_count += 1
            
            # SIMPLE: Always turn toward enemy, fly at them
            # No special cases for "behind" - just pursue
            if abs(angle_off) < self.angle_deadband:
                # Dead ahead - fly straight at them
                turn = 0.0
            else:
                # Turn toward enemy - same behavior regardless of where they are
                turn = np.clip(-angle_off * self.pursuit_gain, 
                              -self.max_turn_rate, self.max_turn_rate)
            
            # High g_force to maintain speed and closure
            g_force = 0.8
            altitude = 0.1
            
            if self.debug and self.step_count % 100 == 0:
                print(f"[ENGAGE] ang={angle_off:.2f} d={distance:.2f} turn={turn:.2f}")
            
            # ============================================================
            # FIRE DECISION
            # ============================================================
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                in_range = self.fire_range_min < distance < self.fire_range_max
                in_cone = abs(angle_off) < self.fire_cone
                
                if in_range and in_cone:
                    fire = 1.0
                    self.fire_cooldown = self.fire_cooldown_steps
                    self.fire_count += 1
                    
                    if self.debug:
                        print(f"[FIRE!] d={distance:.2f} ang={angle_off:.2f}")
                        
                elif self.debug and self.step_count % 100 == 0:
                    print(f"[NO FIRE] in_range={in_range}(d={distance:.2f}) in_cone={in_cone}(ang={angle_off:.2f})")
        
        # ================================================================
        # PURSUIT MODE - Lost contact, continue toward last known direction
        # ================================================================
        elif in_pursuit:
            turn = np.clip(-self.last_known_angle_off * 1.5, 
                          -self.max_turn_rate, self.max_turn_rate)
            g_force = 0.75
            altitude = 0.0
            fire = 0.0
            
            if self.debug and self.step_count % 200 == 0:
                print(f"[PURSUIT] last_ang={self.last_known_angle_off:.2f}")
        
        # ================================================================
        # HUNT MODE - No contact, fly north to find enemy
        # ================================================================
        else:
            heading_error = own_hdg
            
            if abs(heading_error) > 0.03:
                turn = np.clip(-heading_error * 2.0, -self.max_turn_rate, self.max_turn_rate)
                g_force = 0.6
            else:
                turn = 0.0
                g_force = 0.55
            
            altitude = 0.0
            fire = 0.0
            
            if self.debug and self.step_count % 300 == 0:
                print(f"[HUNT] hdg={own_hdg:.2f}")
        
        return np.array([turn, altitude, g_force, fire], dtype=np.float32)


class AggressiveHunterV3:
    """
    Version 3: Same pure pursuit logic, slightly tighter fire parameters.
    """
    
    def __init__(self, debug: bool = False):
        self.fire_range_min = 0.10
        self.fire_range_max = 0.38
        self.fire_cooldown_steps = 100
        self.fire_cone = 0.50
        self.pursuit_gain = 2.0
        self.max_turn_rate = 0.6
        self.angle_deadband = 0.02
        self.debug = debug
        
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 350
        self.last_known_angle_off = 0.0
        
        self.fire_count = 0
        self.engage_count = 0
        
    def reset(self):
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_steps': self.step_count,
            'fire_count': self.fire_count,
            'engage_count': self.engage_count,
        }
        
    def get_action(self, obs: np.ndarray, obs_indices: Dict[str, int]) -> np.ndarray:
        self.step_count += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        n = len(obs)
        
        def get_val(name: str, default: float = 0.0) -> float:
            idx = obs_indices.get(name)
            if idx is not None and 0 <= idx < n:
                return float(obs[idx])
            return default
        
        enemy_detected = get_val('enemy_detected', 0.0) > 0.5
        distance = get_val('distance_to_enemy', -1.0)
        angle_off = get_val('angle_off_to_enemy', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        own_hdg = get_val('own_hdg', 0.0)
        
        valid_enemy = enemy_detected and distance >= 0
        
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off
        
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        if valid_enemy:
            self.engage_count += 1
            
            if abs(angle_off) < self.angle_deadband:
                turn = 0.0
            else:
                turn = np.clip(-angle_off * self.pursuit_gain,
                              -self.max_turn_rate, self.max_turn_rate)
            
            g_force = 0.8
            altitude = 0.1
            
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                in_range = self.fire_range_min < distance < self.fire_range_max
                in_cone = abs(angle_off) < self.fire_cone
                
                if in_range and in_cone:
                    fire = 1.0
                    self.fire_cooldown = self.fire_cooldown_steps
                    self.fire_count += 1
                    if self.debug:
                        print(f"[FIRE] d={distance:.2f} ang={angle_off:.2f}")
                        
        elif in_pursuit:
            turn = np.clip(-self.last_known_angle_off * 1.5,
                          -self.max_turn_rate, self.max_turn_rate)
            g_force = 0.75
            altitude = 0.0
            fire = 0.0
        else:
            if abs(own_hdg) > 0.03:
                turn = np.clip(-own_hdg * 2.0, -self.max_turn_rate, self.max_turn_rate)
                g_force = 0.55
            else:
                turn = 0.0
                g_force = 0.5
            altitude = 0.0
            fire = 0.0
        
        return np.array([turn, altitude, g_force, fire], dtype=np.float32)


class AggressiveHunterTunable:
    """
    Tunable version for parameter sweeps.
    All key parameters exposed.
    """
    
    def __init__(self,
                 fire_range_min: float = 0.08,
                 fire_range_max: float = 0.40,
                 fire_cooldown_steps: int = 100,
                 fire_cone: float = 0.55,
                 pursuit_gain: float = 2.0,
                 max_turn_rate: float = 0.6,
                 angle_deadband: float = 0.02,
                 hunt_heading_gain: float = 2.0,
                 debug: bool = False):
        
        self.fire_range_min = fire_range_min
        self.fire_range_max = fire_range_max
        self.fire_cooldown_steps = fire_cooldown_steps
        self.fire_cone = fire_cone
        self.pursuit_gain = pursuit_gain
        self.max_turn_rate = max_turn_rate
        self.angle_deadband = angle_deadband
        self.hunt_heading_gain = hunt_heading_gain
        self.debug = debug
        
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 400
        self.last_known_angle_off = 0.0
        
        self.fire_count = 0
        self.engage_count = 0
        
    def reset(self):
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_steps': self.step_count,
            'fire_count': self.fire_count,
            'engage_count': self.engage_count,
        }
        
    def get_action(self, obs: np.ndarray, obs_indices: Dict[str, int]) -> np.ndarray:
        self.step_count += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        n = len(obs)
        
        def get_val(name: str, default: float = 0.0) -> float:
            idx = obs_indices.get(name)
            if idx is not None and 0 <= idx < n:
                return float(obs[idx])
            return default
        
        enemy_detected = get_val('enemy_detected', 0.0) > 0.5
        distance = get_val('distance_to_enemy', -1.0)
        angle_off = get_val('angle_off_to_enemy', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        own_hdg = get_val('own_hdg', 0.0)
        
        valid_enemy = enemy_detected and distance >= 0
        
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off
        
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        if valid_enemy:
            self.engage_count += 1
            
            if abs(angle_off) < self.angle_deadband:
                turn = 0.0
            else:
                turn = np.clip(-angle_off * self.pursuit_gain,
                              -self.max_turn_rate, self.max_turn_rate)
            
            g_force = 0.8
            altitude = 0.1
            
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                in_range = self.fire_range_min < distance < self.fire_range_max
                in_cone = abs(angle_off) < self.fire_cone
                
                if in_range and in_cone:
                    fire = 1.0
                    self.fire_cooldown = self.fire_cooldown_steps
                    self.fire_count += 1
                    if self.debug:
                        print(f"[FIRE] d={distance:.2f} ang={angle_off:.2f}")
                        
        elif in_pursuit:
            turn = np.clip(-self.last_known_angle_off * 1.5,
                          -self.max_turn_rate, self.max_turn_rate)
            g_force = 0.75
            altitude = 0.0
            fire = 0.0
        else:
            if abs(own_hdg) > 0.03:
                turn = np.clip(-own_hdg * self.hunt_heading_gain,
                              -self.max_turn_rate, self.max_turn_rate)
                g_force = 0.55
            else:
                turn = 0.0
                g_force = 0.5
            altitude = 0.0
            fire = 0.0
        
        return np.array([turn, altitude, g_force, fire], dtype=np.float32)


def get_default_obs_indices() -> Dict[str, int]:
    """Default observation indices matching Godot B-ACE output (27 dims)."""
    return {
        'own_x': 0,
        'own_z': 1,
        'own_altitude': 2,
        'dist_to_target': 3,
        'aspect_to_target': 4,
        'own_hdg': 5,
        'own_speed': 6,
        'own_missiles': 7,
        'own_in_flight_missile': 8,
        'distance_to_hvaa': 9,
        'hvaa_alt_diff': 10,
        'hvaa_angle_off': 11,
        'hvaa_heading': 12,
        'hvaa_detected': 13,
        'altitude_diff_enemy': 14,
        'aspect_angle_to_enemy': 15,
        'angle_off_to_enemy': 16,
        'distance_to_enemy': 17,
        'dist2go_enemy': 18,
        'own_missile_rmax': 19,
        'own_missile_nez': 20,
        'enemy_missile_rmax': 21,
        'enemy_missile_nez': 22,
        'defensive_factor': 23,
        'offensive_factor': 24,
        'is_missile_support': 25,
        'enemy_detected': 26,
    }


if __name__ == "__main__":
    print("="*60)
    print("AGGRESSIVE HUNTER - TRUE KILL PRIORITY (NO ORBIT)")
    print("="*60)
    print("""
    PROBLEM FIXED: Orbiting/spinning after firing
    
    ROOT CAUSE: Previous version had "enemy behind" logic that triggered
    wide sweeping turns at reduced turn rate. This caused orbiting because:
    1. Fire when enemy is ahead
    2. Geometry shifts, enemy crosses 0.6 angle threshold
    3. Enters "wide turn" mode -> starts orbiting
    
    FIX: REMOVED all "enemy behind" special handling.
    
    New behavior is PURE PURSUIT:
    - Always turn directly toward enemy
    - Same turn rate regardless of enemy position
    - No mode switching, no defensive maneuvers
    - Just point at them and fly
    
    Result: Identify -> Fire -> Fly STRAIGHT at red -> Fire again
    
    Key parameters:
    - pursuit_gain: 2.0 (how aggressively to turn toward enemy)
    - max_turn_rate: 0.6 (single rate, no slow/fast modes)
    - fire_range: 0.08-0.40
    - fire_cone: ~100 degrees  
    - cooldown: 100 steps
    - g_force: 0.8 in engage (high speed pursuit)
    
    Experts:
      AggressiveHunterFixed   - Main version, pure pursuit
      AggressiveHunterV3      - Same logic, tighter fire params
      AggressiveHunterTunable - All parameters exposed for tuning
      
    Test:
      python test_expert_standalone.py --expert hunter_fixed --episodes 10 --render --verbose
    """)
