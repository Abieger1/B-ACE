"""
Modular Heuristic Expert - Configurable Domain Knowledge Levels

Supports 4 levels of heuristic guidance for experimental design:
    - Level 0: None (don't use expert - handled externally by not wrapping)
    - Level 1: 'scripted' - Base observations only (27 dims), pure rule-based
    - Level 2: 'atddg' - Base + ATDDG features (defense_time_ratio, heading_to_optimal, etc.)
    - Level 3: 'full' - Base + ALL heuristics (ATDDG + WEZ/DMC)

This allows clean experimental separation to test:
    - Does expert guidance help at all? (scripted vs none)
    - Do ATDDG heuristics improve the expert? (atddg vs scripted)
    - Do WEZ/DMC heuristics add value? (full vs atddg)

Usage:
    # Level 1: Scripted (no heuristics)
    expert = ModularHeuristicExpert(heuristic_mode='scripted')
    
    # Level 2: ATDDG heuristics only
    expert = ModularHeuristicExpert(heuristic_mode='atddg')
    
    # Level 3: Full heuristics
    expert = ModularHeuristicExpert(heuristic_mode='full')
"""

import numpy as np
from typing import Dict, Optional, Literal


# Type alias for heuristic modes
HeuristicMode = Literal['scripted', 'atddg', 'wez_dmc', 'full']


class ModularHeuristicExpert:
    """
    Expert policy with configurable heuristic knowledge levels.
    
    Heuristic Modes:
        'scripted': Uses only base B-ACE observations (27 dims)
                    Pure rule-based behavior without theoretical features.
                    
        'atddg':    Uses base + Active Target Defense Differential Game features:
                    - defense_time_ratio: Can I intercept attacker before HVAA?
                    - heading_to_optimal_intercept: Differential game optimal heading
                    - offensive_dominance: WEZ advantage comparison
                    - in_escape_region, barrier_value: Escape geometry
                    
        'wez_dmc':  Uses base + WEZ/DMC threat assessment features:
                    - dmc_normalized: Dynamic Maneuvering Cue (evasion urgency)
                    - bez_penetration: How deep in enemy's Basic Engagement Zone
                    - inside_bez, inside_threat: Binary threat indicators
                    
        'full':     Uses base + ALL heuristic features (atddg + wez_dmc)
    
    Args:
        heuristic_mode: Which heuristics to enable ('scripted', 'atddg', 'wez_dmc', 'full')
        debug: Enable debug printing
        **kwargs: Additional tuning parameters (thresholds, gains, etc.)
    """
    
    # =========================================================================
    # OBSERVATION INDICES - Base (0-26)
    # =========================================================================
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
    
    # =========================================================================
    # OBSERVATION INDICES - Enriched (27-39)
    # When specific feature categories are enabled
    # =========================================================================
    
    # WEZ/DMC features (threat assessment)
    IDX_BEZ_PENETRATION = 29
    IDX_INSIDE_BEZ = 30
    IDX_DMC_NORMALIZED = 31
    IDX_INSIDE_THREAT = 32
    
    # ATDDG features (optimal pursuit/defense)
    IDX_TIME_TO_CAPTURE = 27
    IDX_CAPTURE_FEASIBLE = 28
    IDX_DEFENSE_TIME_RATIO = 33
    IDX_IN_ESCAPE_REGION = 34
    IDX_BARRIER_VALUE = 35
    IDX_HEADING_TO_OPTIMAL = 36
    IDX_OFFENSIVE_DOMINANCE = 37
    IDX_ESCAPE_CONE = 38
    IDX_CAPTURE_PROB = 39
    
    def __init__(
        self,
        heuristic_mode: HeuristicMode = 'full',
        # Firing thresholds
        min_offensive_factor: float = 0.5,  # For scripted mode
        min_offensive_dominance: float = 0.0,  # For heuristic modes
        nez_bonus: float = 0.2,
        # Defensive thresholds  
        dmc_evade_threshold: float = 0.5,
        bez_penetration_evade: float = 0.3,
        defensive_factor_evade: float = 0.7,  # For scripted mode
        # HVAA protection
        defense_ratio_threshold: float = 0.6,
        hvaa_dist_threshold: float = 0.25,  # For scripted mode
        # Intercept optimization
        heading_gain: float = 2.0,
        # Behavior
        debug: bool = False,
    ):
        # Validate mode
        valid_modes = ('scripted', 'atddg', 'wez_dmc', 'full')
        if heuristic_mode not in valid_modes:
            raise ValueError(f"heuristic_mode must be one of {valid_modes}, got '{heuristic_mode}'")
        
        self.heuristic_mode = heuristic_mode
        self.debug = debug
        
        # Feature flags based on mode
        self.use_atddg = heuristic_mode in ('atddg', 'full')
        self.use_wez_dmc = heuristic_mode in ('wez_dmc', 'full')
        
        # Thresholds
        self.min_offensive_factor = min_offensive_factor
        self.min_offensive_dominance = min_offensive_dominance
        self.nez_bonus = nez_bonus
        self.dmc_evade_threshold = dmc_evade_threshold
        self.bez_penetration_evade = bez_penetration_evade
        self.defensive_factor_evade = defensive_factor_evade
        self.defense_ratio_threshold = defense_ratio_threshold
        self.hvaa_dist_threshold = hvaa_dist_threshold
        self.heading_gain = heading_gain
        
        # State
        self.step_count = 0
        self.fire_cooldown = 0
        self.fire_cooldown_steps = 100
        
        # Pursuit memory
        self.last_detection_step = -1000
        self.pursuit_memory_steps = 500
        self.last_known_angle_off = 0.0
        self.last_known_heading_error = 0.0
        
        if self.debug:
            print(f"[ModularExpert] Initialized with mode='{heuristic_mode}' "
                  f"(use_atddg={self.use_atddg}, use_wez_dmc={self.use_wez_dmc})")
    
    def reset(self):
        """Reset episode state."""
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0
        self.last_known_heading_error = 0.0
    
    def get_action(self, obs: np.ndarray, obs_indices: Dict[str, int] = None) -> np.ndarray:
        """
        Compute action based on configured heuristic mode.
        
        Args:
            obs: Observation vector (27+ dims)
            obs_indices: Optional index mapping (ignored - we use fixed indices)
            
        Returns:
            Action array [turn, altitude, g_force, fire]
        """
        self.step_count += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        
        n = len(obs)
        
        def get(idx: int, default: float = 0.0) -> float:
            return float(obs[idx]) if idx < n else default
        
        # =====================================================================
        # READ BASE OBSERVATIONS (always available)
        # =====================================================================
        enemy_detected = get(self.IDX_ENEMY_DETECTED) > 0.5
        enemy_dist = get(self.IDX_ENEMY_DIST, -1.0)
        enemy_angle_off = get(self.IDX_ENEMY_ANGLE_OFF)
        enemy_aspect = get(self.IDX_ENEMY_ASPECT)
        
        missiles = get(self.IDX_OWN_MISSILES, 6.0)
        missile_in_flight = get(self.IDX_MISSILE_IN_FLIGHT) > 0.5
        
        # B-ACE WEZ info
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
        # READ HEURISTIC FEATURES (conditional on mode)
        # =====================================================================
        has_enriched = n > 27
        
        # WEZ/DMC features - only if mode includes them
        if self.use_wez_dmc and has_enriched:
            dmc_normalized = get(self.IDX_DMC_NORMALIZED)
            inside_bez = get(self.IDX_INSIDE_BEZ) > 0.5
            bez_penetration = get(self.IDX_BEZ_PENETRATION)
            inside_threat = get(self.IDX_INSIDE_THREAT) > 0.5
        else:
            # Scripted fallback: estimate from base observations
            dmc_normalized = 0.0
            inside_bez = False
            bez_penetration = 0.0
            inside_threat = defensive_factor > self.defensive_factor_evade
        
        # ATDDG features - only if mode includes them
        if self.use_atddg and has_enriched:
            defense_time_ratio = get(self.IDX_DEFENSE_TIME_RATIO)
            in_escape_region = get(self.IDX_IN_ESCAPE_REGION) > 0.5
            barrier_value = get(self.IDX_BARRIER_VALUE)
            heading_to_optimal = get(self.IDX_HEADING_TO_OPTIMAL)
            offensive_dominance = get(self.IDX_OFFENSIVE_DOMINANCE)
            escape_cone = get(self.IDX_ESCAPE_CONE)
        else:
            # Scripted fallback: use base observation approximations
            defense_time_ratio = 0.5  # Neutral
            in_escape_region = True
            barrier_value = 0.0
            heading_to_optimal = 0.0  # No optimal heading info -> pure pursuit
            offensive_dominance = offensive_factor - 0.5  # Crude approximation
            escape_cone = 1.0
        
        # Valid detection
        valid_enemy = enemy_detected and enemy_dist >= 0
        
        # Update pursuit memory
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = enemy_angle_off
            if self.use_atddg:
                self.last_known_heading_error = heading_to_optimal
        
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps
        
        # =====================================================================
        # TACTICAL SITUATION ASSESSMENT
        # =====================================================================
        
        # Threat assessment differs by mode
        if self.use_wez_dmc:
            # Use DMC/BEZ for threat assessment
            high_dmc = dmc_normalized > self.dmc_evade_threshold
            deep_in_enemy_bez = bez_penetration > self.bez_penetration_evade
            inside_enemy_nez = valid_enemy and enemy_dist < enemy_nez
            threatened = high_dmc or deep_in_enemy_bez or inside_enemy_nez
        else:
            # Scripted: use defensive_factor as proxy
            inside_enemy_nez = valid_enemy and enemy_dist < enemy_nez
            high_defensive_factor = defensive_factor > self.defensive_factor_evade
            threatened = high_defensive_factor or inside_enemy_nez
        
        # Offensive assessment differs by mode
        if self.use_atddg:
            has_wez_advantage = offensive_dominance > self.min_offensive_dominance
        else:
            # Scripted: use offensive_factor
            has_wez_advantage = offensive_factor > self.min_offensive_factor
        
        inside_my_rmax = valid_enemy and enemy_dist < own_rmax
        inside_my_nez = valid_enemy and enemy_dist < own_nez
        
        # HVAA defense assessment differs by mode
        if self.use_atddg:
            can_defend_hvaa = defense_time_ratio < self.defense_ratio_threshold
        else:
            # Scripted: simple distance check
            can_defend_hvaa = hvaa_dist < self.hvaa_dist_threshold
        
        if self.debug and self.step_count % 200 == 0:
            print(f"[EXPERT-{self.heuristic_mode.upper()}] step={self.step_count} "
                  f"det={valid_enemy} dist={enemy_dist:.2f} "
                  f"threatened={threatened} can_defend={can_defend_hvaa}")
        
        # =====================================================================
        # DECISION LOGIC
        # =====================================================================
        
        if valid_enemy:
            # --- ENGAGED ---
            
            if threatened and not has_wez_advantage:
                # EVADE
                turn, g_force, altitude = self._evade(
                    enemy_angle_off, hvaa_angle, dmc_normalized, escape_cone
                )
                fire = 0.0
                
            elif missile_in_flight:
                # SUPPORT MISSILE
                turn = np.clip(-enemy_angle_off * 0.2, -0.15, 0.15)
                g_force = 0.3
                altitude = 0.0
                fire = 0.0
                
            elif not can_defend_hvaa and hvaa_dist > 0.15:
                # DEFEND HVAA
                turn, g_force, altitude = self._defend_hvaa(
                    hvaa_angle, defense_time_ratio, heading_to_optimal
                )
                fire = 0.0
                
            else:
                # ATTACK
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
        Evasive maneuver.
        
        Uses DMC/escape_cone if wez_dmc mode enabled, otherwise simpler logic.
        """
        if self.use_wez_dmc and escape_cone < 0.3:
            # Very few safe headings - max effort escape (beam maneuver)
            turn_dir = 1.0 if enemy_angle_off < 0 else -1.0
            turn = turn_dir * 0.95
            g_force = 0.95
        elif self.use_wez_dmc and dmc > 0.7:
            # High DMC - significant turn needed
            turn = np.clip(-enemy_angle_off * 0.5 + hvaa_angle * 0.5, -0.9, 0.9)
            g_force = 0.85
        else:
            # Scripted/moderate: defensive repositioning toward HVAA
            turn = np.clip(hvaa_angle * 1.5, -0.7, 0.7)
            g_force = 0.7
        
        altitude = 0.0
        return turn, g_force, altitude
    
    def _defend_hvaa(self, hvaa_angle: float, defense_ratio: float,
                     heading_to_optimal: float) -> tuple:
        """
        Return to defensive position to protect HVAA.
        
        Uses optimal intercept heading if ATDDG mode enabled.
        """
        if self.use_atddg and abs(heading_to_optimal) > 0.1:
            # Use theory-optimal heading to intercept attacker
            turn = np.clip(-heading_to_optimal * self.heading_gain, -0.8, 0.8)
        else:
            # Scripted: simple HVAA-oriented defense
            turn = np.clip(hvaa_angle * 2.0, -0.8, 0.8)
        
        g_force = 0.75
        altitude = 0.0
        return turn, g_force, altitude
    
    def _attack(self, enemy_angle_off: float, heading_to_optimal: float,
                enemy_dist: float, own_rmax: float) -> tuple:
        """
        Offensive maneuver.
        
        Uses optimal intercept heading if ATDDG mode enabled.
        """
        if self.use_atddg and abs(heading_to_optimal) > 0.05:
            # Use differential game optimal heading
            turn = np.clip(-heading_to_optimal * self.heading_gain, -0.8, 0.8)
        else:
            # Scripted: pure pursuit (point nose at enemy)
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
        if self.use_atddg and abs(last_heading_error) > 0.1:
            # Continue on last known optimal heading
            turn = np.clip(-last_heading_error * 1.5, -0.5, 0.5)
        else:
            # Scripted: use last known angle
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
        Firing decision based on configured mode.
        
        ATDDG mode uses offensive_dominance, scripted uses offensive_factor.
        """
        if missiles <= 0 or missile_in_flight or self.fire_cooldown > 0:
            return 0.0
        
        # Must be inside Rmax
        if not inside_rmax:
            return 0.0
        
        if self.use_atddg:
            # Use offensive_dominance from enriched features
            threshold = self.min_offensive_dominance
            if inside_nez:
                threshold -= self.nez_bonus
            
            if offensive_dominance > threshold:
                good_aspect = abs(enemy_aspect) < 0.25
                if good_aspect or offensive_dominance > 0.3:
                    self.fire_cooldown = self.fire_cooldown_steps
                    return 1.0
        else:
            # Scripted: use offensive_factor threshold
            if offensive_factor > self.min_offensive_factor:
                self.fire_cooldown = self.fire_cooldown_steps
                return 1.0
        
        return 0.0


# =============================================================================
# FACTORY FUNCTION FOR EASY INSTANTIATION
# =============================================================================

def create_expert(level: int, debug: bool = False, **kwargs) -> Optional[ModularHeuristicExpert]:
    """
    Factory function to create expert for experimental design.
    
    Args:
        level: Experimental factor level (0-3)
            0 = None (returns None - don't use wrapper)
            1 = Scripted (base observations only)
            2 = Partial heuristic (ATDDG features)
            3 = Full heuristic (all features)
        debug: Enable debug printing
        **kwargs: Additional expert parameters
        
    Returns:
        ModularHeuristicExpert instance, or None for level 0
    """
    level_to_mode = {
        0: None,  # No expert
        1: 'scripted',
        2: 'atddg',  # Partial = ATDDG (contains optimal heading, key for lethality)
        3: 'full',
    }
    
    if level not in level_to_mode:
        raise ValueError(f"level must be 0-3, got {level}")
    
    mode = level_to_mode[level]
    
    if mode is None:
        return None
    
    return ModularHeuristicExpert(heuristic_mode=mode, debug=debug, **kwargs)


# =============================================================================
# OBSERVATION INDEX HELPERS
# =============================================================================

def get_base_obs_indices() -> Dict[str, int]:
    """Base observation indices (27 dims)."""
    return {
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
    }


def get_enriched_obs_indices() -> Dict[str, int]:
    """Full observation indices (40 dims with enriched features)."""
    base = get_base_obs_indices()
    enriched = {
        'time_to_capture': 27, 'capture_feasible': 28,
        'bez_penetration': 29, 'inside_bez': 30,
        'dmc_normalized': 31, 'inside_threat': 32,
        'defense_time_ratio': 33, 'in_escape_region': 34,
        'barrier_value_normalized': 35, 'heading_to_optimal_intercept': 36,
        'offensive_dominance': 37,
        'escape_cone_normalized': 38, 'capture_probability_proxy': 39,
    }
    return {**base, **enriched}
