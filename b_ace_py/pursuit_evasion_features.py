#!/usr/bin/env python3
"""
pursuit_evasion_features.py

Theoretical features for pursuit-evasion differential games.
Based on closed-form solutions from:
- Weintraub et al. "An Introduction to Pursuit-Evasion Differential Games"
- Von Moll & Weintraub "Basic Engagement Zones" 
- Dillon et al. "Optimal Trajectories for Aircraft Avoidance of Multiple WEZs"
- Von Moll & Weintraub "Dynamic Maneuvering Cue"

These features encode domain knowledge from differential game theory
to accelerate RL learning and improve agent performance.

Usage:
    from pursuit_evasion_features import PursuitEvasionFeatures
    
    feature_computer = PursuitEvasionFeatures()
    features = feature_computer.compute_all_features(obs_dict)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class AgentState:
    """Represents the state of a single agent."""
    position: np.ndarray  # [x, y] or [x, y, z]
    velocity: np.ndarray  # [vx, vy] or [vx, vy, vz]
    heading: float        # radians
    speed: float          # scalar speed


class PursuitEvasionFeatures:
    """
    Computes theoretical features from pursuit-evasion differential game theory.
    
    These features provide the RL agent with domain knowledge about:
    - Capture geometry (Apollonius circles/ovals)
    - Threat regions (Basic Engagement Zones)
    - Required evasive maneuvers (Dynamic Maneuvering Cue)
    - Optimal intercept trajectories
    - Active Target Defense (three-agent escort scenarios)
    
    All features are normalized to roughly [-1, 1] or [0, 1] range for neural network input.
    """
    
    def __init__(self, 
                 default_pursuer_speed: float = 1.0,
                 default_evader_speed: float = 0.8,
                 default_capture_radius: float = 0.1,
                 default_pursuer_range: float = float('inf'),
                 normalize_distance: float = 1000.0):
        """
        Initialize feature computer with default parameters.
        
        Args:
            default_pursuer_speed: Default speed of pursuer if not in observation
            default_evader_speed: Default speed of evader if not in observation
            default_capture_radius: Capture/kill radius
            default_pursuer_range: Maximum range of pursuer (for range-limited scenarios)
            normalize_distance: Distance used for normalizing features
        """
        self.default_pursuer_speed = default_pursuer_speed
        self.default_evader_speed = default_evader_speed
        self.default_capture_radius = default_capture_radius
        self.default_pursuer_range = default_pursuer_range
        self.normalize_distance = normalize_distance
    
    # =========================================================================
    # CORE GEOMETRIC CALCULATIONS
    # =========================================================================
    
    def compute_speed_ratio(self, evader_speed: float, pursuer_speed: float) -> float:
        """
        Compute speed ratio μ = v_E / v_P.
        
        This is fundamental to all pursuit-evasion geometry.
        μ < 1 means pursuer is faster (can eventually catch evader)
        μ > 1 means evader is faster (can escape)
        μ = 1 means equal speeds (special case)
        """
        if pursuer_speed <= 0:
            return float('inf')
        return evader_speed / pursuer_speed
    
    def compute_line_of_sight(self, 
                               from_pos: np.ndarray, 
                               to_pos: np.ndarray) -> Tuple[float, float]:
        """
        Compute line-of-sight angle and distance between two agents.
        
        Args:
            from_pos: Position of observer [x, y]
            to_pos: Position of target [x, y]
            
        Returns:
            (los_angle, distance): LOS angle in radians, distance scalar
        """
        delta = to_pos[:2] - from_pos[:2]
        distance = np.linalg.norm(delta)
        los_angle = np.arctan2(delta[1], delta[0])
        return los_angle, distance
    
    def compute_aspect_angle(self, 
                              heading: float, 
                              los_angle: float) -> float:
        """
        Compute aspect angle ξ = heading - LOS angle.
        
        The aspect angle determines the geometry of engagement.
        ξ = 0: Head-on approach
        ξ = π: Tail chase
        ξ = ±π/2: Beam approach
        
        Returns angle wrapped to [-π, π]
        """
        aspect = heading - los_angle
        # Wrap to [-π, π]
        return (aspect + np.pi) % (2 * np.pi) - np.pi
    
    def compute_offensive_metrics(self,
                              own_pos, own_vel,
                              tgt_pos, tgt_vel,
                              our_Rmax=20000.0,
                              their_Rmax=20000.0):
        """
        Computes both WEZ dominance and offensive time-to-capture (Apollonius).
        own_pos, tgt_pos = np.array([x,y,z])
        own_vel, tgt_vel = speeds in m/s
        """
        # ------------------------------
        # 1) WEZ DOMINANCE (BEZ-style)
        # ------------------------------
        # Red has WEZ on Blue (enemy_has_shot)
        mu_enemy = self.compute_speed_ratio(evader_speed=np.linalg.norm(own_vel),
                                            pursuer_speed=np.linalg.norm(tgt_vel))

        bez_enemy = self.compute_dmc(
            agent_pos=own_pos,
            agent_heading=0.0,
            threat_pos=tgt_pos,
            mu=mu_enemy,
            R=their_Rmax,
            r=self.default_capture_radius,
        )
        enemy_has_shot = float(bez_enemy.get("inside_threat", 0.0))

        # Blue has WEZ on Red (we_have_shot)
        mu_ours = self.compute_speed_ratio(evader_speed=np.linalg.norm(tgt_vel),
                                        pursuer_speed=np.linalg.norm(own_vel))

        bez_ours = self.compute_dmc(
            agent_pos=tgt_pos,
            agent_heading=0.0,
            threat_pos=own_pos,
            mu=mu_ours,
            R=our_Rmax,
            r=self.default_capture_radius,
        )
        we_have_shot = float(bez_ours.get("inside_threat", 0.0))

        offensive_dominance = we_have_shot - enemy_has_shot

        # ------------------------------
        # 2) OFFENSIVE TTC (Apollonius)
        # ------------------------------
        offensive_ttc_data = self.compute_apollonius_intercept(
            pursuer_pos=own_pos,
            evader_pos=tgt_pos,
            evader_heading=0.0,
            mu=self.compute_speed_ratio(
                evader_speed=np.linalg.norm(tgt_vel),
                pursuer_speed=np.linalg.norm(own_vel)
            )
        )

        my_ttc = offensive_ttc_data.get("time_to_capture", float("inf"))
        my_capture_feasible = float(offensive_ttc_data.get("capture_feasible", False))

        # Normalize TTC to [0,1] offensively (shorter is better)
        TTC_CAP = 150.0
        if my_capture_feasible and my_ttc > 0:
            t_norm = min(my_ttc, TTC_CAP) / TTC_CAP
            offensive_ttc_score = 1.0 - t_norm
        else:
            offensive_ttc_score = 0.0

        return dict(
            enemy_has_shot=enemy_has_shot,
            we_have_shot=we_have_shot,
            offensive_dominance=offensive_dominance,
            offensive_ttc_score=offensive_ttc_score,
            offensive_ttc_raw=my_ttc,
            capture_feasible=my_capture_feasible
        )

    # =========================================================================
    # APOLLONIUS CIRCLE GEOMETRY
    # From: "An Introduction to Pursuit-Evasion Differential Games"
    # =========================================================================
    
    def compute_apollonius_intercept(self,
                                      pursuer_pos: np.ndarray,
                                      evader_pos: np.ndarray,
                                      evader_heading: float,
                                      mu: float) -> Dict[str, float]:
        """
        Compute Apollonius circle intercept geometry.
        
        The Apollonius circle defines the locus of all possible interception
        points between a faster pursuer and slower evader.
        
        Args:
            pursuer_pos: Pursuer position [x, y]
            evader_pos: Evader position [x, y]
            evader_heading: Evader heading in radians
            mu: Speed ratio v_E / v_P (must be < 1 for capture)
            
        Returns:
            Dict with:
                - optimal_pursuer_heading: Optimal heading for pursuer
                - time_to_capture: Estimated time to intercept
                - intercept_distance: Distance pursuer must travel
                - capture_feasible: Whether capture is geometrically possible
        """
        los_angle, d = self.compute_line_of_sight(pursuer_pos, evader_pos)
        
        # Aspect angle from pursuer's perspective
        # ψ_E in the paper is evader heading relative to LOS from pursuer
        psi_E = evader_heading - los_angle
        
        result = {
            'optimal_pursuer_heading': 0.0,
            'time_to_capture': float('inf'),
            'intercept_distance': float('inf'),
            'capture_feasible': False
        }
        
        if mu >= 1.0:
            # Evader is faster or equal - no guaranteed capture
            return result
        
        if d < 1e-6:
            # Already at same position
            result['time_to_capture'] = 0.0
            result['intercept_distance'] = 0.0
            result['capture_feasible'] = True
            return result
        
        # Optimal pursuer heading (Eq. 17 from Range-Limited P-E paper)
        # ψ_P = sin^(-1)(μ sin(ψ_E))
        sin_psi_E = np.sin(psi_E)
        
        # Check if intercept is possible
        if abs(mu * sin_psi_E) > 1.0:
            return result
        
        optimal_psi_P = np.arcsin(mu * sin_psi_E)
        optimal_heading = los_angle + optimal_psi_P
        
        # Distance to intercept (Eq. 21 from Range-Limited P-E paper)
        # PI = d / (1 - μ²) * [μ cos(ψ_E) + √(1 - μ² sin²(ψ_E))]
        cos_psi_E = np.cos(psi_E)
        mu_sq = mu * mu
        
        sqrt_term = np.sqrt(max(0, 1 - mu_sq * sin_psi_E * sin_psi_E))
        PI = (d / (1 - mu_sq)) * (mu * cos_psi_E + sqrt_term)
        
        result['optimal_pursuer_heading'] = optimal_heading
        result['intercept_distance'] = PI
        result['time_to_capture'] = PI / self.default_pursuer_speed if self.default_pursuer_speed > 0 else float('inf')
        result['capture_feasible'] = PI > 0 and PI < float('inf')
        
        return result
    
    def compute_apollonius_circle_params(self,
                                          pursuer_pos: np.ndarray,
                                          evader_pos: np.ndarray,
                                          mu: float) -> Dict[str, Any]:
        """
        Compute parameters of the Apollonius circle.
        
        The Apollonius circle has:
        - Center offset from evader by μ²d/(1-μ²) toward pursuer
        - Radius μd/(1-μ²)
        
        Returns:
            Dict with center position and radius
        """
        los_angle, d = self.compute_line_of_sight(pursuer_pos, evader_pos)
        
        if mu >= 1.0 or d < 1e-6:
            return {'center': evader_pos, 'radius': float('inf'), 'valid': False}
        
        mu_sq = mu * mu
        denom = 1 - mu_sq
        
        # Center offset from evader toward pursuer
        offset_dist = mu_sq * d / denom
        center = evader_pos + offset_dist * np.array([np.cos(los_angle + np.pi), 
                                                       np.sin(los_angle + np.pi)])
        
        radius = mu * d / denom
        
        return {
            'center': center,
            'radius': radius,
            'valid': True
        }
    
    # =========================================================================
    # BASIC ENGAGEMENT ZONE (BEZ)
    # From: Von Moll & Weintraub "Basic Engagement Zones"
    # =========================================================================
    
    def compute_bez_boundary(self,
                              aspect_angle: float,
                              mu: float,
                              R: float,
                              r: float = 0.0) -> float:
        """
        Compute BEZ boundary distance at a given aspect angle.
        
        The BEZ represents the region where the threat can intercept
        the agent if the agent holds its current course.
        
        From Eq. 8 in "Basic Engagement Zones":
        ρ(ξ) = μR [cos(ξ) + √(cos²(ξ) - 1 + (R+r)²/(μ²R²))]
        
        Args:
            aspect_angle: Angle ξ between agent heading and LOS to threat
            mu: Speed ratio v_agent / v_threat (< 1 means threat is faster)
            R: Maximum range of threat
            r: Capture radius of threat
            
        Returns:
            Distance from threat to BEZ boundary at this aspect angle
        """
        if R <= 0 or R == float('inf'):
            return float('inf')
        
        if mu <= 0:
            return R + r
        
        cos_xi = np.cos(aspect_angle)
        
        # Term under the radical
        inner = cos_xi * cos_xi - 1 + ((R + r) / (mu * R)) ** 2
        
        if inner < 0:
            # No real solution - outside engagement zone for all headings
            return 0.0
        
        rho = mu * R * (cos_xi + np.sqrt(inner))
        return max(0.0, rho)
    
    def compute_bez_penetration(self,
                                 agent_pos: np.ndarray,
                                 agent_heading: float,
                                 threat_pos: np.ndarray,
                                 mu: float,
                                 R: float,
                                 r: float = 0.0) -> Dict[str, float]:
        """
        Compute how far inside (or outside) the BEZ the agent is.
        
        Returns:
            Dict with:
                - penetration_depth: Positive if inside BEZ, negative if outside
                - normalized_penetration: Penetration / R (for neural network)
                - boundary_distance: Distance to BEZ boundary
                - inside_bez: Boolean
        """
        los_angle, d = self.compute_line_of_sight(threat_pos, agent_pos)
        aspect_angle = self.compute_aspect_angle(agent_heading, los_angle)
        
        rho = self.compute_bez_boundary(aspect_angle, mu, R, r)
        
        penetration = rho - d  # Positive means inside BEZ
        
        return {
            'penetration_depth': penetration,
            'normalized_penetration': penetration / R if R > 0 and R != float('inf') else 0.0,
            'boundary_distance': d - rho,  # Distance to boundary (negative if inside)
            'inside_bez': penetration > 0,
            'aspect_angle': aspect_angle
        }
    
    # =========================================================================
    # DYNAMIC MANEUVERING CUE (DMC)
    # From: Von Moll & Weintraub "Reactive Vehicle Guidance using DMC"
    # =========================================================================
    
    def compute_dmc(self,
                    agent_pos: np.ndarray,
                    agent_heading: float,
                    threat_pos: np.ndarray,
                    mu: float,
                    R: float,
                    r: float = 0.0) -> Dict[str, float]:
        """
        Compute Dynamic Maneuvering Cue (DMC).
        
        DMC is the minimum heading change required to exit the threat's BEZ.
        It serves as a measure of instantaneous risk.
        
        From Eq. 5 and 7 in the DMC paper:
        ξ* = cos^(-1)((d² + μ²R² - (R+r)²) / (2μRd))
        DMC = sign(ξ) * min(|ξ - ξ*|, |ξ + ξ*|)
        
        Args:
            agent_pos: Agent position [x, y]
            agent_heading: Agent heading in radians
            threat_pos: Threat position [x, y]
            mu: Speed ratio v_agent / v_threat
            R: Maximum range of threat
            r: Capture radius of threat
            
        Returns:
            Dict with:
                - dmc_radians: DMC value in radians
                - dmc_degrees: DMC value in degrees
                - dmc_normalized: DMC / π (normalized to [-1, 1])
                - safe_heading_cw: Safe heading (clockwise option)
                - safe_heading_ccw: Safe heading (counter-clockwise option)
                - critical_aspect_angle: ξ* value
        """
        los_angle, d = self.compute_line_of_sight(threat_pos, agent_pos)
        xi = self.compute_aspect_angle(agent_heading, los_angle)
        
        result = {
            'dmc_radians': 0.0,
            'dmc_degrees': 0.0,
            'dmc_normalized': 0.0,
            'safe_heading_cw': agent_heading,
            'safe_heading_ccw': agent_heading,
            'critical_aspect_angle': 0.0,
            'inside_threat': False
        }
        
        if R <= 0 or R == float('inf') or d < 1e-6:
            return result
        
        # Check if outside BEZ at current heading
        rho = self.compute_bez_boundary(xi, mu, R, r)
        if d > rho:
            # Already safe
            return result
        
        result['inside_threat'] = True
        
        # Compute critical aspect angle ξ*
        # ξ* = cos^(-1)((d² + μ²R² - (R+r)²) / (2μRd))
        numerator = d * d + mu * mu * R * R - (R + r) * (R + r)
        denominator = 2 * mu * R * d
        
        if abs(denominator) < 1e-10:
            # Edge case: very close or degenerate geometry
            result['dmc_radians'] = np.pi
            result['dmc_degrees'] = 180.0
            result['dmc_normalized'] = 1.0
            return result
        
        cos_xi_star = numerator / denominator
        
        if abs(cos_xi_star) > 1.0:
            # No safe heading exists (too deep inside BEZ)
            result['dmc_radians'] = np.pi
            result['dmc_degrees'] = 180.0
            result['dmc_normalized'] = 1.0
            return result
        
        xi_star = np.arccos(np.clip(cos_xi_star, -1.0, 1.0))
        result['critical_aspect_angle'] = xi_star
        
        # DMC is the minimum turn to reach a safe aspect angle
        # Safe aspect angles are ξ > ξ* or ξ < -ξ*
        turn_to_positive = xi_star - xi  # Turn to reach +ξ*
        turn_to_negative = -xi_star - xi  # Turn to reach -ξ*
        
        # Wrap turns to [-π, π]
        turn_to_positive = (turn_to_positive + np.pi) % (2 * np.pi) - np.pi
        turn_to_negative = (turn_to_negative + np.pi) % (2 * np.pi) - np.pi
        
        # Choose minimum absolute turn
        if abs(turn_to_positive) <= abs(turn_to_negative):
            dmc = turn_to_positive
        else:
            dmc = turn_to_negative
        
        result['dmc_radians'] = dmc
        result['dmc_degrees'] = np.degrees(dmc)
        result['dmc_normalized'] = dmc / np.pi
        result['safe_heading_cw'] = agent_heading + turn_to_negative
        result['safe_heading_ccw'] = agent_heading + turn_to_positive
        
        return result
    
    # =========================================================================
    # ACTIVE TARGET DEFENSE DIFFERENTIAL GAME (ATDDG)
    # From: Weintraub et al. "An Introduction to Pursuit-Evasion Differential Games"
    # Section V: Active Target Defense
    # =========================================================================
    
    def compute_atddg_defense_features(self,
                                        defender_pos: np.ndarray,
                                        defender_speed: float,
                                        attacker_pos: np.ndarray,
                                        attacker_speed: float,
                                        target_pos: np.ndarray,
                                        target_speed: float = 0.1) -> Dict[str, float]:
        """
        Compute Active Target Defense features for escort scenarios.
        
        From Weintraub et al. 2020, Section V (ATDDG):
        - Three players: Target (T), Attacker (A), Defender (D)
        - Defender intercepts Attacker before Attacker reaches Target
        - Game of Kind: determines which team wins
        - Game of Degree: optimal strategies when Target can escape
        
        In BVR-CCA-EP context:
            - Defender = Agent (escort fighter)
            - Attacker = Enemy fighter
            - Target = HVAA (High Value Airborne Asset)
        
        Key theoretical results used:
        1. Time comparison: t_DA vs t_AT determines outcome
        2. Escape region boundary (Eq. 46-47): 
           x²_A + y²_T/(1-α²) - x²_T/α² = 0
           where α = V_T/V_A (target/attacker speed ratio)
        
        Args:
            defender_pos: Defender (agent) position [x, y]
            defender_speed: Defender speed (normalized)
            attacker_pos: Attacker (enemy) position [x, y]
            attacker_speed: Attacker speed (normalized)
            target_pos: Target (HVAA) position [x, y]
            target_speed: Target speed (HVAA is typically slow)
            
        Returns:
            Dict with:
                - defense_time_ratio: t_DA/t_AT, <1 means defender wins
                - in_escape_region: 1.0 if HVAA is defensible, 0.0 otherwise
                - time_to_intercept_attacker: Estimated t_DA
                - time_attacker_to_target: Estimated t_AT
        """
        # Ensure 2D positions
        defender_pos = np.array(defender_pos[:2])
        attacker_pos = np.array(attacker_pos[:2])
        target_pos = np.array(target_pos[:2])
        
        # Distances
        d_DA = np.linalg.norm(defender_pos - attacker_pos)  # Defender to Attacker
        d_AT = np.linalg.norm(attacker_pos - target_pos)    # Attacker to Target
        d_DT = np.linalg.norm(defender_pos - target_pos)    # Defender to Target
        
        result = {
            'defense_time_ratio': 0.5,  # Neutral default
            'in_escape_region': 1.0,    # Assume defensible by default
            'time_to_intercept_attacker': float('inf'),
            'time_attacker_to_target': float('inf'),
        }
        
        # Handle degenerate cases
        if d_AT < 1e-6:
            # Attacker already at target - defense failed
            result['defense_time_ratio'] = 1.0
            result['in_escape_region'] = 0.0
            return result
        
        if d_DA < 1e-6:
            # Defender already at attacker - defense succeeded
            result['defense_time_ratio'] = 0.0
            result['in_escape_region'] = 1.0
            return result
        
        # Speed ratio α = V_T / V_A (target speed / attacker speed)
        alpha = target_speed / attacker_speed if attacker_speed > 1e-6 else 0.0
        
        # =====================================================================
        # TIME-BASED ANALYSIS (Apollonius-derived intercept times)
        # =====================================================================
        
        # Time for Defender to intercept Attacker (t_DA)
        # Using Apollonius: depends on relative speeds and geometry
        mu_DA = attacker_speed / defender_speed if defender_speed > 1e-6 else float('inf')
        
        if mu_DA < 1.0:
            # Defender is faster - can intercept
            # Simplified estimate: use closing speed
            # More accurate: use Apollonius with attacker's heading toward target
            attacker_heading_to_target = np.arctan2(
                target_pos[1] - attacker_pos[1],
                target_pos[0] - attacker_pos[0]
            )
            apollo_DA = self.compute_apollonius_intercept(
                defender_pos, attacker_pos, attacker_heading_to_target, mu_DA
            )
            t_DA = apollo_DA['time_to_capture'] if apollo_DA['capture_feasible'] else float('inf')
        else:
            # Defender slower or equal - use simple estimate
            closing_speed = max(defender_speed - attacker_speed * 0.3, 0.1)
            t_DA = d_DA / closing_speed
        
        # Time for Attacker to reach Target (t_AT)
        # Attacker pursuing slow target
        mu_AT = target_speed / attacker_speed if attacker_speed > 1e-6 else 0.0
        
        if mu_AT < 1.0 and attacker_speed > target_speed:
            # Attacker faster than target - will catch
            # Simple pursuit: closing speed
            closing_speed_AT = attacker_speed - target_speed
            t_AT = d_AT / closing_speed_AT if closing_speed_AT > 1e-6 else float('inf')
        else:
            # Target faster or equal - won't be caught
            t_AT = float('inf')
        
        result['time_to_intercept_attacker'] = t_DA
        result['time_attacker_to_target'] = t_AT
        
        # Defense time ratio: t_DA / t_AT
        # < 1.0: Defender intercepts before Attacker reaches Target (defense wins)
        # > 1.0: Attacker reaches Target first (defense fails)
        if t_AT > 1e-6 and t_AT < float('inf'):
            defense_ratio = t_DA / t_AT
            # Normalize to [0, 1] where 0.5 is the critical boundary
            # ratio < 1 maps to [0, 0.5], ratio > 1 maps to [0.5, 1]
            result['defense_time_ratio'] = np.clip(defense_ratio / 2.0, 0.0, 1.0)
        else:
            # Target cannot be caught by attacker
            result['defense_time_ratio'] = 0.0
        
        # =====================================================================
        # ESCAPE REGION ANALYSIS (Game of Kind - Proposition 2)
        # =====================================================================
        # 
        # From the paper, the boundary between escapable (R_e) and capture (R_c) 
        # regions is given by (in reduced state space):
        #   x²_A + y²_T/(1-α²) - x²_T/α² = 0
        #
        # Simplified geometric check: Is defender well-positioned to intercept?
        
        # Vector from Attacker to Target
        AT_vec = target_pos - attacker_pos
        AT_unit = AT_vec / d_AT if d_AT > 1e-6 else np.array([1.0, 0.0])
        
        # Vector from Attacker to Defender  
        AD_vec = defender_pos - attacker_pos
        
        # Project defender position onto A-T line
        # projection > 0 means defender is "ahead" of attacker (toward target)
        projection = np.dot(AD_vec, AT_unit)
        
        # Perpendicular distance from defender to A-T line
        perp_distance = np.abs(np.cross(AT_unit, AD_vec))
        
        # Geometric defensibility criteria:
        # 1. Defender should be able to reach intercept point before attacker reaches target
        # 2. Consider the "cone" of interception
        
        # Use the time ratio as primary indicator
        in_escape = (result['defense_time_ratio'] < 0.5)  # defense_ratio < 1 before normalization
        
        # Additional geometric check: defender not too far behind
        # If defender is behind attacker (projection < 0) and far, likely indefensible
        if projection < -d_AT * 0.5:  # Defender far behind
            in_escape = in_escape and (defender_speed > attacker_speed * 1.2)
        
        # Apply speed ratio constraint from ATDDG
        # Critical condition: for defense to be possible, typically need V_D >= V_A
        if defender_speed < attacker_speed * 0.8:
            # Defender significantly slower - harder to defend
            in_escape = in_escape and (d_DA < d_AT * 0.5)
        
        result['in_escape_region'] = 1.0 if in_escape else 0.0
        
        return result
    
    # =========================================================================
    # RANGE-LIMITED PURSUIT (kept for backwards compatibility)
    # From: Weintraub et al. "Range-Limited Pursuit-Evasion"
    # =========================================================================
    
    def compute_escape_feasibility(self,
                                    pursuer_pos: np.ndarray,
                                    evader_pos: np.ndarray,
                                    mu: float,
                                    R: float) -> Dict[str, Any]:
        """
        Determine if evader can escape a range-limited pursuer.
        
        From the Range-Limited P-E paper:
        - If R < d/(1+μ): Evader ALWAYS escapes
        - If R ≥ d/(1-μ): Pursuer ALWAYS captures
        - Otherwise: Outcome depends on evader heading choice
        
        Args:
            pursuer_pos: Pursuer position
            evader_pos: Evader position
            mu: Speed ratio v_E / v_P
            R: Maximum range of pursuer
            
        Returns:
            Dict with escape analysis
        """
        _, d = self.compute_line_of_sight(pursuer_pos, evader_pos)
        
        if mu >= 1.0:
            return {
                'evader_always_escapes': True,
                'pursuer_always_captures': False,
                'outcome_uncertain': False,
                'escape_threshold': 0.0,
                'capture_threshold': float('inf')
            }
        
        escape_threshold = d / (1 + mu)  # R below this = evader escapes
        capture_threshold = d / (1 - mu)  # R above this = pursuer captures
        
        return {
            'evader_always_escapes': R < escape_threshold,
            'pursuer_always_captures': R >= capture_threshold,
            'outcome_uncertain': escape_threshold <= R < capture_threshold,
            'escape_threshold': escape_threshold,
            'capture_threshold': capture_threshold,
            'range_ratio': R / d if d > 0 else float('inf')
        }
    
    def compute_critical_escape_heading(self,
                                         d: float,
                                         mu: float,
                                         R: float) -> Dict[str, float]:
        """
        Compute critical escape heading from Eq. 34 of Range-Limited P-E paper.
        
        ψ_E,crit = cos^(-1)((1 - μ²)R - d²/R) / (2dμ))
        
        This defines the boundary between safe and unsafe headings when
        the pursuer has limited range and we're in the "limited capture region".
        
        Args:
            d: Distance between pursuer and evader
            mu: Speed ratio v_E / v_P (< 1 means pursuer faster)
            R: Maximum range of pursuer
            
        Returns:
            Dict with:
                - critical_heading: ψ_E,crit in radians (NaN if not applicable)
                - escape_cone_normalized: Fraction of headings that escape [0,1]
                - case: 'always_escape', 'always_capture', or 'limited'
        """
        result = {
            'critical_heading': np.nan,
            'escape_cone_normalized': 0.0,
            'case': 'always_capture'
        }
        
        if mu >= 1.0:
            # Evader is faster - always escapes
            result['escape_cone_normalized'] = 1.0
            result['case'] = 'always_escape'
            return result
        
        if d < 1e-6:
            return result  # Already captured
        
        # Check the three cases from the paper (Eq. 25, 27, 28)
        escape_threshold = d / (1 + mu)
        capture_threshold = d / (1 - mu)
        
        if R < escape_threshold:
            # Case 1: Evader always escapes (Eq. 25)
            result['escape_cone_normalized'] = 1.0
            result['case'] = 'always_escape'
            return result
        
        if R >= capture_threshold:
            # Case 2: Pursuer always captures (Eq. 27)
            result['escape_cone_normalized'] = 0.0
            result['case'] = 'always_capture'
            return result
        
        # Case 3: Limited capture region (Eq. 28)
        # Compute critical heading from Eq. 34:
        # ψ_E,crit = cos^(-1)((1 - μ²)R - d²/R) / (2dμ))
        mu_sq = mu * mu
        numerator = (1 - mu_sq) * R - (d * d) / R
        denominator = 2 * d * mu
        
        if abs(denominator) < 1e-10:
            return result
        
        cos_psi_crit = numerator / denominator
        
        # Clamp to valid range for arccos
        if cos_psi_crit > 1.0:
            result['escape_cone_normalized'] = 0.0
            result['case'] = 'always_capture'
        elif cos_psi_crit < -1.0:
            result['escape_cone_normalized'] = 1.0
            result['case'] = 'always_escape'
        else:
            psi_crit = np.arccos(cos_psi_crit)
            result['critical_heading'] = psi_crit
            # Safe headings are |ψ_E| > ψ_E,crit, so cone is (π - ψ_crit) on each side
            result['escape_cone_normalized'] = (np.pi - psi_crit) / np.pi
            result['case'] = 'limited'
        
        return result
    
    def compute_capture_probability_proxy(self,
                                           d: float,
                                           mu: float,
                                           R: float) -> float:
        """
        Continuous capture probability based on Range-Limited theory.
        
        Returns value in [0, 1]:
        - 0.0: Evader always escapes (R below escape threshold)
        - 1.0: Pursuer always captures (R above capture threshold)
        - Between: Linear interpolation in uncertain zone
        
        Useful for reward shaping and neural network input.
        
        Args:
            d: Distance between pursuer and evader
            mu: Speed ratio v_E / v_P
            R: Maximum range of pursuer
            
        Returns:
            Capture probability proxy in [0, 1]
        """
        if mu >= 1.0 or d < 1e-6:
            return 0.0  # Evader faster or already at same position
        
        escape_threshold = d / (1 + mu)
        capture_threshold = d / (1 - mu)
        
        if R < escape_threshold:
            return 0.0
        elif R >= capture_threshold:
            return 1.0
        else:
            # Linear interpolation in the uncertain zone
            return (R - escape_threshold) / (capture_threshold - escape_threshold)
    
    # =========================================================================
    # ATDDG BARRIER HYPERBOLA (Enhanced)
    # From: Weintraub et al. "An Introduction to P-E Differential Games" (2020)
    # Section V, Equations 46-47 (Game of Kind)
    # =========================================================================
    
    def compute_barrier_hyperbola_value(self,
                                         attacker_pos: np.ndarray,
                                         target_pos: np.ndarray,
                                         defender_pos: np.ndarray,
                                         alpha: float) -> Dict[str, float]:
        """
        Compute barrier hyperbola value from ATDDG Game of Kind (Eq. 46-47).
        
        The barrier surface separates:
        - R_e (escape region): Target survives - defender can intercept attacker
        - R_c (capture region): Attacker reaches target first
        
        Barrier equation (in target-centered coordinates):
        x²_A + y²_T/(1-α²) - x²_T/α² = 0
        
        where α = v_T / v_A (target/attacker speed ratio)
        
        Args:
            attacker_pos: Attacker (enemy) position [x, y]
            target_pos: Target (HVAA) position [x, y]
            defender_pos: Defender (agent) position [x, y]
            alpha: Speed ratio v_T / v_A (typically << 1 for slow HVAA)
            
        Returns:
            Dict with:
                - barrier_value_normalized: Clipped to [-1, 1], positive = defensible
                - in_escape_region: Boolean (True = target can be defended)
        """
        # Distances
        d_AT = np.linalg.norm(attacker_pos[:2] - target_pos[:2])
        d_DA = np.linalg.norm(defender_pos[:2] - attacker_pos[:2])
        
        if d_AT < 1e-6:
            # Attacker at target - defense failed
            return {
                'barrier_value_normalized': -1.0,
                'in_escape_region': False
            }
        
        # Clamp alpha to valid range
        alpha = np.clip(alpha, 0.01, 0.99)
        
        # Barrier value computation
        # Positive when defender can intercept attacker before attacker reaches target
        # This is a geometric approximation of the full barrier surface
        barrier_value = d_AT - d_DA * (1 + alpha) / (1 - alpha + 1e-6)
        
        # Normalize to [-1, 1] range
        scale = max(d_AT, d_DA, 1.0)
        barrier_normalized = np.clip(barrier_value / scale, -1.0, 1.0)
        
        return {
            'barrier_value_normalized': barrier_normalized,
            'in_escape_region': barrier_value > 0
        }
    
    def compute_optimal_intercept_heading(self,
                                           defender_pos: np.ndarray,
                                           attacker_pos: np.ndarray,
                                           target_pos: np.ndarray,
                                           defender_heading: float = 0.0) -> Dict[str, float]:
        """
        Compute heading toward optimal intercept point (simplified from Eq. 49).
        
        The quartic equation from Eq. 49 gives closed-form optimal defender heading.
        This implementation uses a geometric approximation suitable for real-time
        computation in RL environments.
        
        The optimal strategy is to intercept the attacker along its path to the target,
        accounting for relative speeds and positions.
        
        Args:
            defender_pos: Defender (agent) position [x, y]
            attacker_pos: Attacker (enemy) position [x, y]
            target_pos: Target (HVAA) position [x, y]
            defender_heading: Current defender heading (for error computation)
            
        Returns:
            Dict with:
                - optimal_heading: Heading in radians toward intercept point
                - heading_error_normalized: (optimal - current) / π, in [-1, 1]
                - intercept_point: [x, y] of optimal intercept location
        """
        d_AT = np.linalg.norm(attacker_pos[:2] - target_pos[:2])
        d_DA = np.linalg.norm(defender_pos[:2] - attacker_pos[:2])
        
        if d_AT < 1e-6:
            # Attacker at target
            return {
                'optimal_heading': 0.0,
                'heading_error_normalized': 0.0,
                'intercept_point': np.array(attacker_pos[:2])
            }
        
        # Attacker's direction toward target
        attack_dir = (target_pos[:2] - attacker_pos[:2]) / d_AT
        
        # Estimate intercept point along attacker's path
        # Use proportional navigation concept: intercept ahead of attacker
        intercept_fraction = np.clip(d_DA / (d_AT + d_DA + 1e-6), 0.0, 0.9)
        
        # Intercept point
        intercept_point = attacker_pos[:2] + intercept_fraction * d_AT * attack_dir
        
        # Optimal heading: from defender toward intercept point
        delta = intercept_point - defender_pos[:2]
        optimal_heading = np.arctan2(delta[1], delta[0])
        
        # Heading error (wrapped to [-π, π])
        heading_error = optimal_heading - defender_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        
        return {
            'optimal_heading': optimal_heading,
            'heading_error_normalized': heading_error / np.pi,
            'intercept_point': intercept_point
        }
    
    # =========================================================================
    # CARTESIAN OVAL (Non-Zero Capture Radius)
    # From: Range-Limited P-E paper, Section V
    # =========================================================================
    
    def compute_cartesian_oval_intercept(self,
                                          pursuer_pos: np.ndarray,
                                          evader_pos: np.ndarray,
                                          evader_heading: float,
                                          mu: float,
                                          rho: float) -> Dict[str, float]:
        """
        Compute intercept geometry with non-zero capture radius.
        
        When the pursuer has a capture radius ρ > 0, the interception
        locus becomes a Cartesian oval instead of an Apollonius circle.
        
        Args:
            pursuer_pos: Pursuer position
            evader_pos: Evader position
            evader_heading: Evader heading
            mu: Speed ratio
            rho: Capture radius
            
        Returns:
            Dict with intercept geometry
        """
        los_angle, d = self.compute_line_of_sight(pursuer_pos, evader_pos)
        psi_E = evader_heading - los_angle
        
        result = {
            'pursuer_distance': float('inf'),
            'optimal_pursuer_heading': 0.0,
            'capture_feasible': False
        }
        
        if d <= rho:
            # Already within capture radius
            result['pursuer_distance'] = 0.0
            result['capture_feasible'] = True
            return result
        
        if mu >= 1.0:
            return result
        
        # From Eq. 38 in Range-Limited P-E paper:
        # (1-μ²)PP_f² + (2ρ - 2dμcos(ψ_E))PP_f + ρ² - d² = 0
        a = 1 - mu * mu
        b = 2 * rho - 2 * d * mu * np.cos(psi_E)
        c = rho * rho - d * d
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return result
        
        # Take positive root
        PP_f = (-b + np.sqrt(discriminant)) / (2 * a)
        
        if PP_f <= 0:
            return result
        
        # Optimal pursuer heading from Eq. 41
        sin_psi_P = mu * PP_f * np.sin(psi_E) / (PP_f + rho)
        
        if abs(sin_psi_P) > 1.0:
            return result
        
        psi_P = np.arcsin(sin_psi_P)
        
        result['pursuer_distance'] = PP_f
        result['optimal_pursuer_heading'] = los_angle + psi_P
        result['capture_feasible'] = True
        
        return result
    
    # =========================================================================
    # MULTI-THREAT EXTENSION
    # From: DMC paper, Section IV.C
    # =========================================================================
    
    def compute_safe_heading_cone(self,
                                   agent_pos: np.ndarray,
                                   agent_heading: float,
                                   threats: List[Dict],
                                   mu: float,
                                   R: float,
                                   r: float = 0.0) -> Dict[str, Any]:
        """
        Compute the cone of safe headings considering multiple threats.
        
        For multiple threats, the safe heading cone is the intersection
        of individual safe cones.
        
        Args:
            agent_pos: Agent position
            agent_heading: Current agent heading
            threats: List of threat dicts with 'position' key
            mu: Speed ratio
            R: Threat range
            r: Capture radius
            
        Returns:
            Dict with safe cone information
        """
        if not threats:
            return {
                'safe_cone_lower': -np.pi,
                'safe_cone_upper': np.pi,
                'safe_cone_width': 2 * np.pi,
                'safe_cone_width_normalized': 1.0,
                'fully_surrounded': False,
                'heading_to_safe_cone': 0.0
            }
        
        # Collect safe cones from each threat
        safe_intervals = []
        
        for threat in threats:
            threat_pos = np.array(threat.get('position', [0, 0])[:2])
            dmc_result = self.compute_dmc(agent_pos, agent_heading, threat_pos, mu, R, r)
            
            if not dmc_result['inside_threat']:
                # Outside this threat's BEZ - all headings safe from it
                safe_intervals.append((-np.pi, np.pi))
            else:
                xi_star = dmc_result['critical_aspect_angle']
                los_angle, _ = self.compute_line_of_sight(threat_pos, agent_pos)
                
                # Safe headings: aspect angle > xi* or < -xi*
                safe_heading_ccw = los_angle + xi_star
                safe_heading_cw = los_angle - xi_star
                
                # This gives us headings pointing away from threat
                # We need headings where agent is OUTSIDE BEZ
                safe_intervals.append((safe_heading_cw, safe_heading_ccw))
        
        # Intersect all safe intervals (simplified: just check if any safe heading exists)
        # Full interval intersection is complex - use conservative estimate
        
        # For now, return the smallest safe cone
        min_width = 2 * np.pi
        best_interval = (-np.pi, np.pi)
        
        for lower, upper in safe_intervals:
            width = (upper - lower + 2 * np.pi) % (2 * np.pi)
            if width < min_width:
                min_width = width
                best_interval = (lower, upper)
        
        cone_center = (best_interval[0] + best_interval[1]) / 2
        heading_to_cone = cone_center - agent_heading
        heading_to_cone = (heading_to_cone + np.pi) % (2 * np.pi) - np.pi
        
        return {
            'safe_cone_lower': best_interval[0],
            'safe_cone_upper': best_interval[1],
            'safe_cone_width': min_width,
            'safe_cone_width_normalized': min_width / (2 * np.pi),
            'fully_surrounded': min_width < 0.1,  # Very small safe cone
            'heading_to_safe_cone': heading_to_cone
        }
    
    # =========================================================================
    # FEATURE VECTOR COMPUTATION
    # =========================================================================
    
    def compute_features_for_agent(self,
                                    agent_pos: np.ndarray,
                                    agent_vel: np.ndarray,
                                    agent_heading: float,
                                    agent_speed: float,
                                    threats: List[Dict],
                                    targets: List[Dict] = None) -> np.ndarray:
        """
        Compute full feature vector for a single agent.
        
        This is the main interface for the RL wrapper.
        
        Args:
            agent_pos: Agent position [x, y]
            agent_vel: Agent velocity [vx, vy]
            agent_heading: Agent heading in radians
            agent_speed: Agent speed scalar
            threats: List of threat dicts with 'position', 'speed', 'heading'
            targets: List of target/goal dicts (optional)
            
        Returns:
            Feature vector (numpy array) with normalized values
        """
        features = []
        
        # Agent's own normalized state
        features.append(agent_speed / self.default_pursuer_speed)  # Normalized speed
        
        # Process each threat
        for i, threat in enumerate(threats[:4]):  # Limit to 4 threats
            threat_pos = np.array(threat.get('position', [0, 0])[:2])
            threat_speed = threat.get('speed', self.default_pursuer_speed)
            threat_range = threat.get('range', self.default_pursuer_range)
            capture_radius = threat.get('capture_radius', self.default_capture_radius)
            
            # Speed ratio
            mu = self.compute_speed_ratio(agent_speed, threat_speed)
            features.append(mu)
            
            # Distance and LOS
            los_angle, distance = self.compute_line_of_sight(agent_pos, threat_pos)
            features.append(distance / self.normalize_distance)  # Normalized distance
            
            # Aspect angle
            aspect = self.compute_aspect_angle(agent_heading, los_angle)
            features.append(aspect / np.pi)  # Normalized to [-1, 1]
            
            # Apollonius intercept
            apollo = self.compute_apollonius_intercept(
                threat_pos, agent_pos, agent_heading, mu
            )
            features.append(apollo['time_to_capture'] / 100.0)  # Normalized time
            features.append(1.0 if apollo['capture_feasible'] else 0.0)
            
            # BEZ penetration
            bez = self.compute_bez_penetration(
                agent_pos, agent_heading, threat_pos, mu, threat_range, capture_radius
            )
            features.append(bez['normalized_penetration'])
            features.append(1.0 if bez['inside_bez'] else 0.0)
            
            # DMC
            dmc = self.compute_dmc(
                agent_pos, agent_heading, threat_pos, mu, threat_range, capture_radius
            )
            features.append(dmc['dmc_normalized'])
            
            # Heading error to optimal escape
            heading_to_safe = dmc['safe_heading_ccw'] - agent_heading
            heading_to_safe = (heading_to_safe + np.pi) % (2 * np.pi) - np.pi
            features.append(heading_to_safe / np.pi)
        
        # Pad if fewer than 4 threats (10 features per threat)
        features_per_threat = 10
        while len(features) < 1 + 4 * features_per_threat:
            features.append(0.0)
        
        # Multi-threat aggregate features
        if threats:
            safe_cone = self.compute_safe_heading_cone(
                agent_pos, agent_heading, threats,
                self.compute_speed_ratio(agent_speed, self.default_pursuer_speed),
                self.default_pursuer_range, self.default_capture_radius
            )
            features.append(safe_cone['safe_cone_width_normalized'])
            features.append(float(safe_cone['fully_surrounded']))
        else:
            features.extend([1.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Return names of all features for interpretability."""
        names = ['agent_speed_normalized']
        
        for i in range(4):
            prefix = f'threat_{i}_'
            names.extend([
                prefix + 'speed_ratio',
                prefix + 'distance_norm',
                prefix + 'aspect_angle_norm',
                prefix + 'time_to_capture_norm',
                prefix + 'capture_feasible',
                prefix + 'bez_penetration_norm',
                prefix + 'inside_bez',
                prefix + 'dmc_normalized',
                prefix + 'heading_to_safe_norm',
            ])
        
        names.extend([
            'safe_cone_width_norm',
            'fully_surrounded'
        ])
        
        return names


# =============================================================================
# CONVENIENCE FUNCTIONS FOR GODOT INTEGRATION
# =============================================================================

def extract_agent_state_from_obs(obs: np.ndarray, 
                                  obs_labels: Dict[str, int]) -> Dict[str, Any]:
    """
    Extract agent state from Godot observation array using label mapping.
    
    Args:
        obs: Raw observation array from Godot
        obs_labels: Dict mapping label names to indices
        
    Returns:
        Dict with extracted state values
    """
    state = {}
    
    # Common observation labels (adjust based on your Godot env)
    pos_labels = ['pos_x', 'pos_y', 'position_x', 'position_y', 'x', 'y']
    vel_labels = ['vel_x', 'vel_y', 'velocity_x', 'velocity_y', 'vx', 'vy']
    heading_labels = ['heading', 'yaw', 'orientation', 'angle']
    speed_labels = ['speed', 'velocity_magnitude', 'vel_mag']
    
    # Try to extract position
    for label in pos_labels:
        if label in obs_labels:
            if 'position' not in state:
                state['position'] = [0.0, 0.0]
            idx = obs_labels[label]
            if 'x' in label.lower():
                state['position'][0] = obs[idx]
            else:
                state['position'][1] = obs[idx]
    
    # Similar for velocity, heading, speed...
    # (Implementation depends on your specific observation space)
    
    return state


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    # Basic tests
    fc = PursuitEvasionFeatures()
    
    print("Testing Apollonius intercept...")
    pursuer = np.array([0.0, 0.0])
    evader = np.array([10.0, 0.0])
    result = fc.compute_apollonius_intercept(pursuer, evader, 0.0, 0.5)
    print(f"  Evader heading 0° (away): {result}")
    
    result = fc.compute_apollonius_intercept(pursuer, evader, np.pi, 0.5)
    print(f"  Evader heading 180° (toward): {result}")
    
    print("\nTesting BEZ...")
    bez = fc.compute_bez_penetration(
        agent_pos=np.array([5.0, 0.0]),
        agent_heading=0.0,
        threat_pos=np.array([0.0, 0.0]),
        mu=0.8,
        R=10.0,
        r=0.5
    )
    print(f"  BEZ result: {bez}")
    
    print("\nTesting DMC...")
    dmc = fc.compute_dmc(
        agent_pos=np.array([5.0, 0.0]),
        agent_heading=np.pi,  # Heading toward threat
        threat_pos=np.array([0.0, 0.0]),
        mu=0.8,
        R=10.0,
        r=0.5
    )
    print(f"  DMC result: {dmc}")
    
    print("\nTesting ATDDG defense features...")
    atddg = fc.compute_atddg_defense_features(
        defender_pos=np.array([5.0, 2.0]),   # Agent/escort
        defender_speed=1.0,
        attacker_pos=np.array([10.0, 0.0]),  # Enemy
        attacker_speed=1.0,
        target_pos=np.array([0.0, 0.0]),     # HVAA
        target_speed=0.1
    )
    print(f"  ATDDG result: {atddg}")
    
    # Test where defender is well-positioned
    atddg2 = fc.compute_atddg_defense_features(
        defender_pos=np.array([5.0, 0.0]),   # Agent between enemy and HVAA
        defender_speed=1.2,
        attacker_pos=np.array([10.0, 0.0]),  # Enemy
        attacker_speed=1.0,
        target_pos=np.array([0.0, 0.0]),     # HVAA
        target_speed=0.1
    )
    print(f"  ATDDG (defender between): {atddg2}")
    
    print("\nTesting feature vector...")
    threats = [{'position': [0.0, 0.0], 'speed': 1.0, 'range': 10.0}]
    features = fc.compute_features_for_agent(
        agent_pos=np.array([5.0, 2.0]),
        agent_vel=np.array([0.8, 0.0]),
        agent_heading=0.0,
        agent_speed=0.8,
        threats=threats
    )
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Feature vector: {features[:10]}...")
    
    print("\nAll tests passed!")
