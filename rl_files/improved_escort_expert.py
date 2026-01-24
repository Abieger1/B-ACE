"""
Improved Escort Expert - Better lethality while protecting HVAA.

Key insight: The mission is ESCORT (protect HVAA), not pure hunting!
Flying north to chase red ABANDONS the HVAA and loses the mission.

This expert improves on SimpleAggressiveExpert by:
1. KEEPING the HVAA-centric patrol and positioning (this was CORRECT)
2. IMPROVING engagement efficiency when enemy is ahead
3. IMPROVING fire decision (slightly wider envelope, better aspect check)
4. IMPROVING missile support (don't break lock with hard maneuvers)

Based on SWEEP WINNER parameters but with targeted improvements.
"""

import numpy as np
from typing import Dict, Any, Optional


class ImprovedEscortExpert:
    """
    Improved escort expert that:
    1. Stays near HVAA (doesn't abandon it)
    2. Positions between threat and HVAA
    3. More aggressive pursuit when enemy is AHEAD
    4. Better fire decisions
    5. Proper missile support
    """
    
    def __init__(self,
                 # Fire parameters - slightly wider than SWEEP WINNER but not too wide
                 fire_range_min: float = 0.08,   # Slightly closer (was 0.10)
                 fire_range_max: float = 0.45,   # Slightly wider (was 0.40)
                 fire_cooldown_steps: int = 80,  # Faster refire (was 100)
                 
                 # Pursuit parameters
                 pursuit_gain: float = 2.5,      # More aggressive (was 2.0)
                 engage_g: float = 0.95,         # Slightly higher (was 0.9)
                 
                 # Aspect limit for firing
                 max_fire_aspect: float = 0.7,   # Don't fire in pure tail chase
                 
                 debug: bool = False):
        
        self.fire_range_min = fire_range_min
        self.fire_range_max = fire_range_max
        self.fire_cooldown_steps = fire_cooldown_steps
        self.pursuit_gain = pursuit_gain
        self.engage_g = engage_g
        self.max_fire_aspect = max_fire_aspect
        self.debug = debug
        
        self.step_count = 0
        self.fire_cooldown = 0
        
        # Pursuit memory
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 500  # Same as original
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
        """Get escort-optimized action."""
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
        aspect = get_val('aspect_angle_to_enemy', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        
        # HVAA info - CRITICAL for escort mission
        hvaa_dist = get_val('distance_to_hvaa', 0.3)
        hvaa_angle = get_val('hvaa_angle_off', 0.0)
        
        # Weapon envelope from observation (if available)
        rmax = get_val('own_missile_rmax', self.fire_range_max)
        nez = get_val('own_missile_nez', self.fire_range_min)
        
        valid_enemy = enemy_detected and distance >= 0
        
        # Update pursuit memory
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off
        
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        # ================================================================
        # ENGAGE MODE - Active enemy contact
        # ================================================================
        if valid_enemy:
            self.engage_count += 1
            
            if missile_in_flight:
                # MISSILE SUPPORT - gentle maneuvering to maintain lock
                # DON'T do hard turns that break the shot!
                turn = np.clip(-angle_off * 0.3, -0.2, 0.2)
                g_force = 0.4
                
                if self.debug and self.step_count % 200 == 0:
                    print(f"[MISSILE SUPPORT] angle_off={angle_off:.2f} -> gentle turn={turn:.2f}")
                    
            else:
                # NO MISSILE IN FLIGHT - pursue or position
                
                if abs(angle_off) < 0.5:
                    # Enemy roughly AHEAD - aggressive pursuit!
                    # This is where we improve over simple expert
                    turn = np.clip(-angle_off * self.pursuit_gain, -0.9, 0.9)
                    g_force = self.engage_g
                    
                    if self.debug and self.step_count % 200 == 0:
                        print(f"[ENGAGE-AHEAD] angle_off={angle_off:.2f} -> turn={turn:.2f}")
                        
                else:
                    # Enemy to SIDE or BEHIND
                    # KEEP THE HVAA-CENTRIC BEHAVIOR - this is correct for escort!
                    # Turn toward HVAA to stay between threat and asset
                    if abs(hvaa_angle) > 0.1:
                        turn = np.clip(hvaa_angle * 2.0, -0.8, 0.8)
                    else:
                        # HVAA ahead, do a moderate turn toward enemy
                        turn = np.clip(-angle_off * 1.2, -0.7, 0.7)
                    g_force = self.engage_g
                    
                    if self.debug and self.step_count % 200 == 0:
                        print(f"[ENGAGE-SIDE] angle_off={angle_off:.2f} hvaa_angle={hvaa_angle:.2f} -> turn={turn:.2f}")
            
            altitude = 0.1
            
            # FIRE DECISION - improved logic
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                
                # Range check - use weapon envelope if available
                if rmax > 0.1 and nez > 0:
                    # Use actual weapon envelope with some margin
                    dist_ok = (nez * 0.9) < distance < (rmax * 0.85)
                else:
                    # Fallback to configured range
                    dist_ok = self.fire_range_min < distance < self.fire_range_max
                
                # Aspect check - don't fire in pure tail chase (low Pk)
                # aspect near 0 = head-on (good), near ±1 = tail (bad)
                aspect_ok = abs(aspect) < self.max_fire_aspect
                
                # Angle-off check - enemy should be roughly ahead
                angle_ok = abs(angle_off) < 0.4  # Within ~72 degrees of nose
                
                if dist_ok and aspect_ok and angle_ok:
                    fire = 1.0
                    self.fire_cooldown = self.fire_cooldown_steps
                    self.fire_count += 1
                    
                    if self.debug:
                        print(f"[FIRE!] dist={distance:.2f} aspect={aspect:.2f} angle_off={angle_off:.2f}")
                        
                elif self.debug and self.step_count % 300 == 0:
                    print(f"[NO FIRE] dist_ok={dist_ok} aspect_ok={aspect_ok} angle_ok={angle_ok} "
                          f"(d={distance:.2f} asp={aspect:.2f} ang={angle_off:.2f})")
        
        # ================================================================
        # PURSUIT MODE - Lost contact recently
        # ================================================================
        elif in_pursuit:
            if missile_in_flight:
                # Support the shot
                turn = np.clip(-self.last_known_angle_off * 0.2, -0.15, 0.15)
                g_force = 0.35
            else:
                # Continue toward last known, but bias toward HVAA
                if abs(self.last_known_angle_off) < 0.5:
                    turn = np.clip(-self.last_known_angle_off * 0.8, -0.5, 0.5)
                    g_force = 0.6
                else:
                    # Enemy was far off - turn toward HVAA
                    if abs(hvaa_angle) > 0.1:
                        turn = np.clip(hvaa_angle * 1.0, -0.6, 0.6)
                    else:
                        turn = 0.0
                    g_force = 0.5
            
            altitude = 0.0
            fire = 0.0
            
            if self.debug and self.step_count % 200 == 0:
                print(f"[PURSUIT] last_angle={self.last_known_angle_off:.2f} hvaa={hvaa_angle:.2f}")
        
        # ================================================================
        # PATROL MODE - Stay near HVAA (CORRECT for escort!)
        # ================================================================
        else:
            # Don't fly north! Stay near HVAA to protect it.
            if hvaa_dist > 0.10:
                # Too far from HVAA - close the distance
                turn = np.clip(hvaa_angle * 2.0, -1.0, 1.0)
                g_force = 0.6
                
                if self.debug and self.step_count % 500 == 0:
                    print(f"[PATROL->HVAA] dist={hvaa_dist:.2f} angle={hvaa_angle:.2f}")
            else:
                # Close to HVAA - orbit/patrol
                turn = 0.0
                g_force = 0.4
                
                if self.debug and self.step_count % 500 == 0:
                    print(f"[PATROL-ORBIT] hvaa_dist={hvaa_dist:.2f}")
            
            altitude = 0.0
            fire = 0.0
        
        return np.array([turn, altitude, g_force, fire], dtype=np.float32)


class TunedEscortExpert:
    """
    Escort expert with parameters tuned for maximum lethality.
    
    Key changes from SimpleAggressiveExpert:
    - More aggressive pursuit when enemy ahead (gain 2.5 vs 2.0)
    - Slightly wider fire envelope (0.08-0.45 vs 0.10-0.40)
    - Faster cooldown (80 vs 100)
    - Better aspect filtering (don't waste shots on tail chases)
    - Higher g-force in engagement (0.95 vs 0.9)
    
    KEEPS the HVAA-centric behavior that makes escort work.
    """
    
    def __init__(self, debug: bool = False):
        # These are the key tuning parameters
        self.fire_range_min = 0.08
        self.fire_range_max = 0.45
        self.fire_cooldown_steps = 80
        self.pursuit_gain = 2.5
        self.engage_g = 0.95
        self.max_fire_aspect = 0.65  # Tighter aspect requirement
        self.debug = debug
        
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 500
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
        aspect = get_val('aspect_angle_to_enemy', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        hvaa_dist = get_val('distance_to_hvaa', 0.3)
        hvaa_angle = get_val('hvaa_angle_off', 0.0)
        
        valid_enemy = enemy_detected and distance >= 0
        
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off
        
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        if valid_enemy:
            self.engage_count += 1
            
            if missile_in_flight:
                # Gentle missile support
                turn = np.clip(-angle_off * 0.25, -0.15, 0.15)
                g_force = 0.35
            else:
                if abs(angle_off) < 0.5:
                    # Enemy ahead - aggressive pursuit
                    turn = np.clip(-angle_off * self.pursuit_gain, -0.9, 0.9)
                    g_force = self.engage_g
                else:
                    # Enemy to side/behind - position toward HVAA
                    if abs(hvaa_angle) > 0.1:
                        turn = np.clip(hvaa_angle * 2.0, -0.8, 0.8)
                    else:
                        turn = np.clip(-angle_off * 1.2, -0.7, 0.7)
                    g_force = self.engage_g
            
            altitude = 0.1
            
            # Fire decision with aspect filter
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                dist_ok = self.fire_range_min < distance < self.fire_range_max
                aspect_ok = abs(aspect) < self.max_fire_aspect
                angle_ok = abs(angle_off) < 0.35
                
                if dist_ok and aspect_ok and angle_ok:
                    fire = 1.0
                    self.fire_cooldown = self.fire_cooldown_steps
                    self.fire_count += 1
                    if self.debug:
                        print(f"[FIRE] d={distance:.2f} asp={aspect:.2f} ang={angle_off:.2f}")
                        
        elif in_pursuit:
            if missile_in_flight:
                turn = np.clip(-self.last_known_angle_off * 0.15, -0.1, 0.1)
                g_force = 0.3
            else:
                if abs(self.last_known_angle_off) < 0.5:
                    turn = np.clip(-self.last_known_angle_off * 0.8, -0.5, 0.5)
                    g_force = 0.6
                else:
                    turn = np.clip(hvaa_angle * 1.0, -0.6, 0.6) if abs(hvaa_angle) > 0.1 else 0.0
                    g_force = 0.5
            altitude = 0.0
            fire = 0.0
            
        else:
            # PATROL - Stay near HVAA
            if hvaa_dist > 0.10:
                turn = np.clip(hvaa_angle * 2.0, -1.0, 1.0)
                g_force = 0.6
            else:
                turn = 0.0
                g_force = 0.4
            altitude = 0.0
            fire = 0.0
        
        return np.array([turn, altitude, g_force, fire], dtype=np.float32)


class ConservativeEscortExpert:
    """
    Most conservative escort expert - closest to proven SWEEP WINNER.
    Only changes: slightly faster cooldown and tighter aspect filter.
    """
    
    def __init__(self, debug: bool = False):
        # Almost identical to SWEEP WINNER
        self.fire_range_min = 0.10  # SWEEP WINNER
        self.fire_range_max = 0.40  # SWEEP WINNER
        self.fire_cooldown_steps = 90  # Slightly faster
        self.pursuit_gain = 2.0  # SWEEP WINNER
        self.engage_g = 0.9  # SWEEP WINNER
        self.max_fire_aspect = 0.6  # Added aspect filter
        self.debug = debug
        
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 500
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
        aspect = get_val('aspect_angle_to_enemy', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        hvaa_dist = get_val('distance_to_hvaa', 0.3)
        hvaa_angle = get_val('hvaa_angle_off', 0.0)
        
        valid_enemy = enemy_detected and distance >= 0
        
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off
        
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        if valid_enemy:
            self.engage_count += 1
            
            if missile_in_flight:
                turn = np.clip(-angle_off * 0.2, -0.15, 0.15)
                g_force = 0.3
            else:
                if abs(angle_off) < 0.5:
                    turn = np.clip(-angle_off * self.pursuit_gain, -0.8, 0.8)
                    g_force = self.engage_g
                else:
                    if abs(hvaa_angle) > 0.1:
                        turn = np.clip(hvaa_angle * 2.0, -0.8, 0.8)
                    else:
                        turn = np.clip(-angle_off * 1.5, -0.8, 0.8)
                    g_force = self.engage_g
            
            altitude = 0.1
            
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                dist_ok = self.fire_range_min < distance < self.fire_range_max
                aspect_ok = abs(aspect) < self.max_fire_aspect  # NEW: aspect filter
                
                if dist_ok and aspect_ok:
                    fire = 1.0
                    self.fire_cooldown = self.fire_cooldown_steps
                    self.fire_count += 1
                        
        elif in_pursuit:
            if missile_in_flight:
                turn = np.clip(-self.last_known_angle_off * 0.15, -0.1, 0.1)
                g_force = 0.3
            else:
                if abs(self.last_known_angle_off) < 0.5:
                    turn = np.clip(-self.last_known_angle_off * 0.8, -0.5, 0.5)
                    g_force = 0.6
                else:
                    turn = np.clip(hvaa_angle * 1.0, -0.6, 0.6) if abs(hvaa_angle) > 0.1 else 0.0
                    g_force = 0.5
            altitude = 0.0
            fire = 0.0
            
        else:
            if hvaa_dist > 0.10:
                turn = np.clip(hvaa_angle * 2.0, -1.0, 1.0)
                g_force = 0.6
            else:
                turn = 0.0
                g_force = 0.4
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
    print("IMPROVED ESCORT EXPERT")
    print("="*60)
    print("""
    Key insight: This is an ESCORT mission - protect the HVAA!
    
    The "fly north to hunt" approach FAILS because it abandons HVAA.
    
    This expert KEEPS the HVAA-centric behavior but improves:
    - More aggressive pursuit when enemy is AHEAD
    - Better fire decision (aspect filtering)
    - Faster cooldown
    - Higher engagement g-forces
    
    Use in training:
        from improved_escort_expert import ImprovedEscortExpert, get_default_obs_indices
        expert = ImprovedEscortExpert(debug=False)
        
    Or for most conservative (closest to SWEEP WINNER):
        from improved_escort_expert import ConservativeEscortExpert
        expert = ConservativeEscortExpert(debug=False)
    """)
