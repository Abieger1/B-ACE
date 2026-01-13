"""
Enriched Heuristic Expert - Uses ALL Available Domain Knowledge

This expert uses both:
1. Base observations (27 dims) - WEZ info, offensive/defensive factors
2. Enriched observations (13 dims) - Pursuit-evasion theory features

Key insights for LETHALITY:
- Use `heading_to_optimal_intercept` to fly the mathematically optimal path
- Use `offensive_dominance` to know when you have WEZ advantage
- Use `defense_time_ratio` to ensure HVAA protection
- Fire inside YOUR WEZ when `offensive_dominance` > 0
- Evade when `dmc_normalized` is high (need large turn to escape)

Observation space: 40 dimensions total
- Indices 0-26: Base B-ACE observations
- Indices 27-39: Enriched pursuit-evasion features
"""

import numpy as np
from typing import Dict, Optional


class EnrichedHeuristicExpert:
    """
    Expert policy leveraging full enriched observation space.
    
    Combines:
    - B-ACE computed WEZ geometry (Rmax, NEZ, offensive/defensive factors)
    - Differential game theory features (DMC, BEZ, optimal intercept heading)
    - Active Target Defense features (defense time ratio, barrier value)
    """
    
    # =========================================================================
    # OBSERVATION INDICES
    # =========================================================================
    
    # Base observations (0-26)
    IDX_OWN_X = 0
    IDX_OWN_Z = 1
    IDX_OWN_HDG = 5
    IDX_OWN_SPEED = 6
    IDX_OWN_MISSILES = 7
    IDX_MISSILE_IN_FLIGHT = 8
    
    IDX_HVAA_DIST = 9
    IDX_HVAA_ANGLE_OFF = 11
    
    IDX_ENEMY_ASPECT = 15
    IDX_ENEMY_ANGLE_OFF = 16
    IDX_ENEMY_DIST = 17
    IDX_OWN_RMAX = 19
    IDX_OWN_NEZ = 20
    IDX_ENEMY_RMAX = 21
    IDX_ENEMY_NEZ = 22
    IDX_DEFENSIVE_FACTOR = 23
    IDX_OFFENSIVE_FACTOR = 24
    IDX_ENEMY_DETECTED = 26
    
    # Enriched features (27-39) - when all categories enabled
    IDX_TIME_TO_CAPTURE = 27
    IDX_CAPTURE_FEASIBLE = 28
    IDX_BEZ_PENETRATION = 29
    IDX_INSIDE_BEZ = 30
    IDX_DMC_NORMALIZED = 31
    IDX_INSIDE_THREAT = 32
    IDX_DEFENSE_TIME_RATIO = 33
    IDX_IN_ESCAPE_REGION = 34
    IDX_BARRIER_VALUE = 35
    IDX_HEADING_TO_OPTIMAL = 36  # KEY FOR LETHALITY
    IDX_OFFENSIVE_DOMINANCE = 37  # KEY FOR LETHALITY
    IDX_ESCAPE_CONE = 38
    IDX_CAPTURE_PROB = 39
    
    def __init__(self,
                 # Firing thresholds
                 min_offensive_dominance: float = 0.0,  # Fire when WEZ advantage
                 nez_bonus: float = 0.2,                # Lower threshold inside NEZ
                 
                 # Defensive thresholds
                 dmc_evade_threshold: float = 0.5,      # DMC > 50% = evade
                 bez_penetration_evade: float = 0.3,    # Deep in enemy BEZ = evade
                 
                 # HVAA protection
                 defense_ratio_threshold: float = 0.6,  # Must be able to defend HVAA
                 
                 # Intercept optimization  
                 use_optimal_heading: bool = True,      # Use theory-optimal heading
                 heading_gain: float = 2.0,             # How aggressively to correct heading
                 
                 debug: bool = False):
        
        self.min_offensive_dominance = min_offensive_dominance
        self.nez_bonus = nez_bonus
        self.dmc_evade_threshold = dmc_evade_threshold
        self.bez_penetration_evade = bez_penetration_evade
        self.defense_ratio_threshold = defense_ratio_threshold
        self.use_optimal_heading = use_optimal_heading
        self.heading_gain = heading_gain
        self.debug = debug
        
        # State
        self.step_count = 0
        self.fire_cooldown = 0
        self.fire_cooldown_steps = 100
        
        # Pursuit memory
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 500
        self.last_known_angle_off = 0.0
        self.last_known_heading_error = 0.0
        
    def reset(self):
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0
        self.last_known_heading_error = 0.0
        
    def get_action(self, obs: np.ndarray, obs_indices: Dict[str, int] = None) -> np.ndarray:
        """
        Compute action using enriched observations.
        
        Args:
            obs: Full observation vector (40 dims with enriched features)
            obs_indices: Optional index mapping (ignored - we use fixed indices)
        """
        self.step_count += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        n = len(obs)
        
        def get(idx: int, default: float = 0.0) -> float:
            return float(obs[idx]) if idx < n else default
        
        # =====================================================================
        # READ BASE OBSERVATIONS
        # =====================================================================
        enemy_detected = get(self.IDX_ENEMY_DETECTED) > 0.5
        enemy_dist = get(self.IDX_ENEMY_DIST, -1.0)
        enemy_angle_off = get(self.IDX_ENEMY_ANGLE_OFF)
        enemy_aspect = get(self.IDX_ENEMY_ASPECT)
        
        missiles = get(self.IDX_OWN_MISSILES, 6.0)
        missile_in_flight = get(self.IDX_MISSILE_IN_FLIGHT) > 0.5
        
        # B-ACE computed WEZ info
        own_rmax = get(self.IDX_OWN_RMAX, 0.5)
        own_nez = get(self.IDX_OWN_NEZ, 0.2)
        enemy_rmax = get(self.IDX_ENEMY_RMAX, 0.5)
        enemy_nez = get(self.IDX_ENEMY_NEZ, 0.2)
        offensive_factor = get(self.IDX_OFFENSIVE_FACTOR)
        defensive_factor = get(self.IDX_DEFENSIVE_FACTOR)
        
        # HVAA info
        hvaa_dist = get(self.IDX_HVAA_DIST, 0.3)
        hvaa_angle = get(self.IDX_HVAA_ANGLE_OFF)
        
        # =====================================================================
        # READ ENRICHED FEATURES (the theoretical advantages!)
        # =====================================================================
        has_enriched = n > 27
        
        if has_enriched:
            # Pursuit-evasion theory features
            dmc_normalized = get(self.IDX_DMC_NORMALIZED)
            inside_bez = get(self.IDX_INSIDE_BEZ) > 0.5
            bez_penetration = get(self.IDX_BEZ_PENETRATION)
            inside_threat = get(self.IDX_INSIDE_THREAT) > 0.5
            
            # Active Target Defense features
            defense_time_ratio = get(self.IDX_DEFENSE_TIME_RATIO)
            in_escape_region = get(self.IDX_IN_ESCAPE_REGION) > 0.5
            barrier_value = get(self.IDX_BARRIER_VALUE)
            
            # OPTIMAL INTERCEPT - This is the key to lethality!
            heading_to_optimal = get(self.IDX_HEADING_TO_OPTIMAL)
            
            # WEZ dominance comparison
            offensive_dominance = get(self.IDX_OFFENSIVE_DOMINANCE)
            
            # Escape geometry
            escape_cone = get(self.IDX_ESCAPE_CONE)
            capture_prob = get(self.IDX_CAPTURE_PROB)
        else:
            # Fallback if enriched features not available
            dmc_normalized = 0.0
            inside_bez = False
            bez_penetration = 0.0
            inside_threat = False
            defense_time_ratio = 0.5
            in_escape_region = True
            barrier_value = 0.0
            heading_to_optimal = 0.0
            offensive_dominance = offensive_factor - 0.5  # Approximate
            escape_cone = 1.0
            capture_prob = 0.0
        
        # Valid detection
        valid_enemy = enemy_detected and enemy_dist >= 0
        
        # Update pursuit memory
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = enemy_angle_off
            self.last_known_heading_error = heading_to_optimal
            
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        # =====================================================================
        # TACTICAL SITUATION ASSESSMENT
        # =====================================================================
        
        # Am I in danger? (Multiple indicators)
        high_dmc = dmc_normalized > self.dmc_evade_threshold
        deep_in_enemy_bez = bez_penetration > self.bez_penetration_evade
        inside_enemy_nez = valid_enemy and enemy_dist < enemy_nez
        threatened = high_dmc or deep_in_enemy_bez or inside_enemy_nez
        
        # Do I have offensive advantage?
        has_wez_advantage = offensive_dominance > self.min_offensive_dominance
        inside_my_rmax = valid_enemy and enemy_dist < own_rmax
        inside_my_nez = valid_enemy and enemy_dist < own_nez
        
        # Can I defend HVAA?
        can_defend_hvaa = defense_time_ratio < self.defense_ratio_threshold
        
        if self.debug and self.step_count % 200 == 0:
            print(f"[EXPERT] step={self.step_count} det={valid_enemy} "
                  f"dist={enemy_dist:.2f} off_dom={offensive_dominance:.2f} "
                  f"dmc={dmc_normalized:.2f} hdg_err={heading_to_optimal:.2f} "
                  f"threatened={threatened} can_defend={can_defend_hvaa}")
        
        # =====================================================================
        # DECISION LOGIC
        # =====================================================================
        
        if valid_enemy:
            # --- ENGAGED ---
            
            if threatened and not has_wez_advantage:
                # EVADE: High DMC or deep in enemy BEZ without offensive advantage
                turn, g_force, altitude = self._evade(
                    enemy_angle_off, hvaa_angle, dmc_normalized, escape_cone
                )
                fire = 0.0
                
                if self.debug:
                    print(f"[EVADE] dmc={dmc_normalized:.2f} bez_pen={bez_penetration:.2f}")
                    
            elif missile_in_flight:
                # SUPPORT MISSILE: Gentle maneuvers to maintain guidance
                turn = np.clip(-enemy_angle_off * 0.2, -0.15, 0.15)
                g_force = 0.3
                altitude = 0.0
                fire = 0.0
                
            elif not can_defend_hvaa and hvaa_dist > 0.15:
                # HVAA IN DANGER: Prioritize returning to defensive position
                turn, g_force, altitude = self._defend_hvaa(
                    hvaa_angle, defense_time_ratio, heading_to_optimal
                )
                fire = 0.0
                
                if self.debug:
                    print(f"[DEFEND HVAA] def_ratio={defense_time_ratio:.2f}")
                    
            else:
                # ATTACK: Use optimal intercept heading for maximum lethality
                turn, g_force, altitude = self._attack(
                    enemy_angle_off, heading_to_optimal, enemy_dist, own_rmax
                )
                
                # FIRE DECISION
                fire = self._firing_decision(
                    missiles=missiles,
                    missile_in_flight=missile_in_flight,
                    inside_rmax=inside_my_rmax,
                    inside_nez=inside_my_nez,
                    offensive_dominance=offensive_dominance,
                    offensive_factor=offensive_factor,
                    enemy_aspect=enemy_aspect,
                    enemy_dist=enemy_dist
                )
                
        elif in_pursuit:
            # --- PURSUIT ---
            turn, g_force, altitude = self._pursuit(
                self.last_known_angle_off, 
                self.last_known_heading_error,
                hvaa_angle
            )
            fire = 0.0
            
        else:
            # --- PATROL ---
            turn, g_force, altitude = self._patrol(hvaa_dist, hvaa_angle)
            fire = 0.0
            
        return np.array([turn, altitude, g_force, fire], dtype=np.float32)
    
    # =========================================================================
    # MANEUVER METHODS
    # =========================================================================
    
    def _evade(self, enemy_angle_off: float, hvaa_angle: float, 
               dmc: float, escape_cone: float) -> tuple:
        """
        Evasive maneuver using DMC and escape cone information.
        
        If escape_cone is low (few safe headings), we need aggressive evasion.
        """
        if escape_cone < 0.3:
            # Very few safe headings - max effort escape
            # Turn perpendicular to threat (beam maneuver)
            turn_dir = 1.0 if enemy_angle_off < 0 else -1.0
            turn = turn_dir * 0.95
            g_force = 0.95
        elif dmc > 0.7:
            # High DMC - significant turn needed
            # Turn away from threat, toward safe region
            turn = np.clip(-enemy_angle_off * 0.5 + hvaa_angle * 0.5, -0.9, 0.9)
            g_force = 0.85
        else:
            # Moderate threat - defensive repositioning
            turn = np.clip(hvaa_angle * 1.5, -0.7, 0.7)
            g_force = 0.7
            
        altitude = 0.0
        return turn, g_force, altitude
    
    def _defend_hvaa(self, hvaa_angle: float, defense_ratio: float,
                     heading_to_optimal: float) -> tuple:
        """
        Return to defensive position to protect HVAA.
        
        Uses optimal intercept heading to position efficiently.
        """
        if self.use_optimal_heading and abs(heading_to_optimal) > 0.1:
            # Use theory-optimal heading to intercept attacker
            turn = np.clip(-heading_to_optimal * self.heading_gain, -0.8, 0.8)
        else:
            # Fall back to simple HVAA-oriented defense
            turn = np.clip(hvaa_angle * 2.0, -0.8, 0.8)
            
        g_force = 0.75
        altitude = 0.0
        return turn, g_force, altitude
    
    def _attack(self, enemy_angle_off: float, heading_to_optimal: float,
                enemy_dist: float, own_rmax: float) -> tuple:
        """
        Offensive maneuver using OPTIMAL INTERCEPT HEADING.
        
        This is the key to lethality - fly the mathematically optimal path!
        """
        if self.use_optimal_heading and abs(heading_to_optimal) > 0.05:
            # PRIMARY: Use differential game optimal heading
            # heading_to_optimal is already the error from optimal
            # Negative error = need to turn right, positive = turn left
            turn = np.clip(-heading_to_optimal * self.heading_gain, -0.8, 0.8)
            
            if self.debug and self.step_count % 200 == 0:
                print(f"[ATTACK-OPTIMAL] hdg_err={heading_to_optimal:.2f} -> turn={turn:.2f}")
        else:
            # FALLBACK: Pure pursuit (point nose at enemy)
            turn = np.clip(-enemy_angle_off * 1.5, -0.8, 0.8)
            
        # G-force based on distance
        if enemy_dist > own_rmax:
            g_force = 0.8  # Close range quickly
        else:
            g_force = 0.6  # Inside WEZ, smoother for shot
            
        altitude = 0.1
        return turn, g_force, altitude
    
    def _pursuit(self, last_angle_off: float, last_heading_error: float,
                 hvaa_angle: float) -> tuple:
        """Continue pursuit after losing track."""
        if self.use_optimal_heading and abs(last_heading_error) > 0.1:
            # Continue on last known optimal heading
            turn = np.clip(-last_heading_error * 1.5, -0.5, 0.5)
        else:
            turn = np.clip(-last_angle_off * 0.8, -0.5, 0.5)
            
        g_force = 0.6
        altitude = 0.0
        return turn, g_force, altitude
    
    def _patrol(self, hvaa_dist: float, hvaa_angle: float) -> tuple:
        """Return to HVAA and maintain station."""
        if hvaa_dist > 0.10:
            turn = np.clip(hvaa_angle * 2.0, -1.0, 1.0)
            g_force = 0.6
        else:
            turn = 0.0
            g_force = 0.4
        altitude = 0.0
        return turn, g_force, altitude
    
    # =========================================================================
    # FIRING DECISION
    # =========================================================================
    
    def _firing_decision(self, missiles: float, missile_in_flight: bool,
                         inside_rmax: bool, inside_nez: bool,
                         offensive_dominance: float, offensive_factor: float,
                         enemy_aspect: float, enemy_dist: float) -> float:
        """
        WEZ-aware firing using offensive_dominance from enriched features.
        
        Key insight: offensive_dominance > 0 means YOUR WEZ extends further
        than the enemy's WEZ at current geometry. This is THE indicator to fire!
        """
        if missiles <= 0 or missile_in_flight or self.fire_cooldown > 0:
            return 0.0
            
        # HARD REQUIREMENT: Must be inside your Rmax
        if not inside_rmax:
            return 0.0
        
        # Determine threshold based on position
        if inside_nez:
            # Inside NEZ = high Pk, lower threshold
            threshold = self.min_offensive_dominance - self.nez_bonus
        else:
            threshold = self.min_offensive_dominance
        
        # PRIMARY: Use offensive_dominance from enriched features
        if offensive_dominance > threshold:
            # Good aspect angle bonus
            good_aspect = abs(enemy_aspect) < 0.25  # ~45 degrees
            
            if good_aspect or offensive_dominance > 0.3:
                if self.debug:
                    print(f"[FIRE!] off_dom={offensive_dominance:.2f} "
                          f"nez={inside_nez} asp={enemy_aspect:.2f}")
                self.fire_cooldown = self.fire_cooldown_steps
                return 1.0
        
        # FALLBACK: Use B-ACE offensive_factor if dominance not available
        elif offensive_factor > 0.5 and inside_rmax:
            if self.debug:
                print(f"[FIRE-FALLBACK] off_fac={offensive_factor:.2f}")
            self.fire_cooldown = self.fire_cooldown_steps
            return 1.0
            
        return 0.0


# =============================================================================
# COMPARISON: What makes this expert more lethal?
# =============================================================================
#
# 1. OPTIMAL INTERCEPT HEADING (heading_to_optimal_intercept)
#    - Old: Turn toward enemy (pure pursuit)
#    - New: Fly mathematically optimal intercept course from differential game theory
#    - Result: Faster intercept, better geometry for shots
#
# 2. OFFENSIVE DOMINANCE (offensive_dominance) 
#    - Old: Fire based on arbitrary distance thresholds
#    - New: Fire when YOUR WEZ extends further than enemy's
#    - Result: Shots from positions of advantage
#
# 3. DMC-BASED EVASION (dmc_normalized)
#    - Old: No evasion, or simple defensive_factor check
#    - New: Evade when required heading change is large (quantified risk)
#    - Result: Survive to fight again
#
# 4. HVAA DEFENSE (defense_time_ratio)
#    - Old: Simple distance check to HVAA
#    - New: Can I intercept attacker before they reach HVAA?
#    - Result: Mission-focused protection
#
# 5. ESCAPE GEOMETRY (escape_cone_normalized)
#    - Old: No awareness of escape options
#    - New: Know how many safe headings exist
#    - Result: More informed evasion decisions
# =============================================================================


def get_enriched_obs_indices() -> Dict[str, int]:
    """
    Full observation indices for enriched 40-dim observation space.
    """
    return {
        # Base (0-26)
        'own_x': 0, 'own_z': 1, 'own_altitude': 2,
        'dist_to_target': 3, 'aspect_to_target': 4,
        'own_hdg': 5, 'own_speed': 6,
        'own_missiles': 7, 'own_in_flight_missile': 8,
        'distance_to_hvaa': 9, 'hvaa_alt_diff': 10,
        'hvaa_angle_off': 11, 'hvaa_heading': 12, 'hvaa_detected': 13,
        'altitude_diff_enemy': 14, 'aspect_angle_to_enemy': 15,
        'angle_off_to_enemy': 16, 'distance_to_enemy': 17,
        'dist2go_enemy': 18, 'own_missile_rmax': 19, 'own_missile_nez': 20,
        'enemy_missile_rmax': 21, 'enemy_missile_nez': 22,
        'defensive_factor': 23, 'offensive_factor': 24,
        'is_missile_support': 25, 'enemy_detected': 26,
        
        # Enriched (27-39)
        'time_to_capture': 27, 'capture_feasible': 28,
        'bez_penetration': 29, 'inside_bez': 30,
        'dmc_normalized': 31, 'inside_threat': 32,
        'defense_time_ratio': 33, 'in_escape_region': 34,
        'barrier_value_normalized': 35, 'heading_to_optimal_intercept': 36,
        'offensive_dominance': 37,
        'escape_cone_normalized': 38, 'capture_probability_proxy': 39,
    }
