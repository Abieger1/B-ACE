#!/usr/bin/env python3
"""
hierarchical_action_wrapper_fsm_FIXED.py

FIXES:
1. Dynamic observation index resolution from obs_map (handles enriched observations)
2. Better debug output to trace why fire is blocked
3. Fallback index detection with validation
4. Print actual observation values being read
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


# -----------------------------
# Action indices (single source of truth)
# -----------------------------
HDG  = 0  # heading change command
ALT  = 1  # altitude rate change command
G    = 2  # desired load factor / g command
FIRE = 3  # binary fire command


@dataclass
class CurriculumConfig:
    initial_greedy_prob: float = 0.5
    final_greedy_prob: float = 0.95
    curriculum_steps: int = 3_000_000

    def get_greedy_prob(self, total_steps: int) -> float:
        if self.curriculum_steps <= 0:
            return float(self.final_greedy_prob)
        frac = float(np.clip(total_steps / float(self.curriculum_steps), 0.0, 1.0))
        return float(self.initial_greedy_prob + frac * (self.final_greedy_prob - self.initial_greedy_prob))


class ExpertFSMState(str, Enum):
    DEFEND_HVAA = "DEFEND_HVAA"
    ENGAGE = "ENGAGE"
    MISSILE_SUPPORT = "MISSILE_SUPPORT"
    EVADE = "EVADE"


class HierarchicalFSMActionWrapper(gym.Wrapper):
    """
    Expert FSM wrapper with FIXED observation index handling.
    """
    def __init__(
        self,
        env: gym.Env,
        curriculum_config: Optional[CurriculumConfig] = None,
        debug: bool = False,
    ):
        super().__init__(env)
        self.debug = debug
        self.curriculum = curriculum_config or CurriculumConfig()
        
        # FIXED: Resolve indices dynamically from the environment
        self.obs_index = self._resolve_obs_indices()
        self._obs_index_validated = False

        if self.debug:
            print("[FSM] Resolved obs indices:", self.obs_index)

        # Episode-local step and global step
        self._episode_steps = 0
        self._global_step = 0
        self._last_obs: Optional[np.ndarray] = None

        # -----------------------------
        # Aggressive expert mode
        # -----------------------------
        self.aggressive_allow_multi_shot = True
        self.aggressive_disable_post_fire_calm = True
        self.aggressive_shooter_mode = True

        # -----------------------------
        # Gates
        # -----------------------------
        self.dShot = 0.60
        self.lCrank = 0.60
        self.lBreak = 0.95
        self.invert_hvaa_bearing = True
        self._hvaa_danger_count = 0
        self._hvaa_destroyed = False

        # Heading smoothing
        self._hdg_smooth = 0.0
        self.hdg_smooth_alpha = 0.25
        self._hdg_smooth_enemy = 0.0
        self.hdg_smooth_alpha_enemy = 0.20
        self.max_enemy_hdg_cmd = 0.65

        # Fire rising edge
        self.require_fire_rising_edge = False
        self._prev_fire_request = False

        # Emergency
        self.emergency_threat_gate = 1.75
        self.emergency_confirm_steps = 1
        self.emergency_min_steps = 40
        self._emergency_count = 0
        self._emergency_t = 0
        self._emergency_active = False
        self._prev_hdg_cmd = 0.0
        self.max_hdg_delta_per_step = 0.30 if self.aggressive_shooter_mode else 0.08

        # FSM timers
        self.fsm_state: ExpertFSMState = ExpertFSMState.DEFEND_HVAA
        self._fsm_t = 0
        self._defense_side = 1.0
        self._fired_in_this_engage = False

        self.evade_steps = 80
        self.missile_support_steps = 35
        self.post_fire_calm_steps = 60
        self.post_fire_max_hdg = 0.35
        self.post_fire_max_g = 0.35

        self.hvaa_protect_radius = 0.22
        self.hvaa_danger_enemy_dist = 0.80
        self.hvaa_danger_threat = 1.55
        self.hvaa_danger_confirm_steps = 2
        self.terminal_self_threat = 1.90

        # HVAA ring
        self.hvaa_ring_inner = 0.1
        self.hvaa_ring_outer = 0.5
        self.hvaa_hard_leash = 0.6
        self._prev_hvaa_dist = None
        self._hvaa_away_count = 0
        self.hvaa_away_break_steps = 8
        self.hvaa_away_eps = 1e-4
        self._last_valid_hvaa_bearing = 0.0
        self._last_valid_hvaa_dist = None
        self._returning_to_hvaa = False

        # -----------------------------
        # FIRE GATING - MORE AGGRESSIVE SETTINGS
        # -----------------------------
        self.fire_dist_gate = 0.2 if self.aggressive_shooter_mode else 0.2
        self.fire_aspect_gate = 1.00 if self.aggressive_shooter_mode else 0.75
        self.fire_cooldown_steps = 700 if self.aggressive_shooter_mode else 500
        self._last_fire_step = -10**9

        # Engage/disengage
        self.engage_confirm_steps = 3
        self.disengage_confirm_steps = 8
        self.force_engage_dist = 0.8
        self.force_engage_threat = 0.85
        self.min_offense_to_engage = 0.20
        self._engage_count = 0
        self._enemy_lost_count = 0

        # Stats
        self.stats = {
            "greedy_count": 0,
            "expert_count": 0,
            "expert_state_counts": {s.value: 0 for s in ExpertFSMState},
            "fire_requests": 0,
            "fire_accepted": 0,
            "fire_blocked_cooldown": 0,
            "fire_blocked_distance": 0,
            "fire_blocked_aspect": 0,
            "fire_blocked_no_enemy": 0,
        }

    def set_global_step(self, t: int) -> None:
        self._global_step = int(t)

    def _time_step(self) -> int:
        return int(self._global_step) if int(self._global_step) > 0 else int(self._episode_steps)

    def _find_attr_in_chain(self, name: str):
        e = self.env
        for _ in range(20):
            if hasattr(e, name):
                return getattr(e, name)
            if not hasattr(e, "env"):
                break
            e = e.env
        return None

    def _resolve_obs_indices(self) -> Dict[str, int]:
        """
        FIXED: Properly resolve observation indices from the environment's obs_map.
        This handles enriched observation spaces correctly.
        """
        # Default fallback indices (for ORIGINAL 22-dim space)
        # These will be overwritten if obs_map is found
        idx = dict(
            threat_raw=23,
            offense_raw=24,
            distance_to_hvaa=9,
            hvaa_angle_off=11,
            distance_to_enemy=17,
            aspect_angle_to_enemy=15,
            angle_off_to_enemy=16,
            missile_tracking_us=25,
            enemy_detected=26,
            hvaa_detected=13,
        )

        # Try to get obs_map from environment chain
        obs_map = self._find_attr_in_chain("obs_map")
        
        if isinstance(obs_map, dict):
            print("[FSM] Found obs_map in environment chain")
            # Pick agent_0's mapping
            agent_map = None
            for k in ("agent_0", "blue_0", "blue_agent_0"):
                if k in obs_map:
                    agent_map = obs_map[k]
                    print(f"[FSM] Using mapping for agent key: {k}")
                    break
            
            if agent_map is None and len(obs_map) > 0:
                # Use first available key
                first_key = list(obs_map.keys())[0]
                agent_map = obs_map[first_key]
                print(f"[FSM] Using mapping for first key: {first_key}")
            
            if isinstance(agent_map, dict):
                # Print all available labels for debugging
                print("[FSM] Available observation labels:")
                for label, index in sorted(agent_map.items(), key=lambda x: x[1]):
                    print(f"  {index:3d}: {label}")
                
                # Map our internal names to actual observation labels
                # ADJUST THESE MAPPINGS based on your actual Godot labels!
                label_mappings = {
                    "threat_raw": ["threat", "threat_raw", "threat_level"],
                    "offense_raw": ["offense", "offense_raw", "offense_level"],
                    "distance_to_hvaa": ["hvaa_dist", "dist_to_hvaa", "distance_to_hvaa"],
                    "hvaa_angle_off": ["hvaa_angle_off", "hvaa_bearing", "angle_off_hvaa"],
                    "distance_to_enemy": ["enemy_dist", "dist_to_enemy", "distance_to_enemy", "enemy_range"],
                    "aspect_angle_to_enemy": ["enemy_aspect", "aspect_angle", "aspect_to_enemy", "aspect_angle_to_enemy"],
                    "angle_off_to_enemy": ["enemy_angle_off", "angle_off", "angle_off_to_enemy", "ATA"],
                    "missile_tracking_us": ["missile_warning", "missile_tracking", "missile_tracking_us", "RWR"],
                    "enemy_detected": ["enemy_detected", "target_detected", "has_target", "enemy_visible"],
                    "hvaa_detected": ["hvaa_detected", "hvaa_seen", "hvaa_visible", "has_hvaa"],
                }
                
                for our_name, possible_labels in label_mappings.items():
                    for label in possible_labels:
                        # Case-insensitive search
                        for actual_label, actual_idx in agent_map.items():
                            if label.lower() == actual_label.lower():
                                idx[our_name] = int(actual_idx)
                                print(f"[FSM] Mapped '{our_name}' -> '{actual_label}' (index {actual_idx})")
                                break
                        else:
                            continue
                        break
        else:
            print("[FSM] WARNING: No obs_map found - using hardcoded fallback indices!")
            print("[FSM] This may cause issues with enriched observations!")
        
        # Also try observation_labels_flat
        labels_flat = self._find_attr_in_chain("observation_labels_flat")
        if labels_flat is not None:
            print(f"[FSM] Found observation_labels_flat with {len(labels_flat)} labels")
            label_to_idx = {label: i for i, label in enumerate(labels_flat)}
            
            # Same mapping logic
            label_mappings = {
                "threat_raw": ["threat", "threat_raw"],
                "offense_raw": ["offense", "offense_raw"],
                "distance_to_hvaa": ["hvaa_dist", "dist_to_hvaa"],
                "hvaa_angle_off": ["hvaa_angle_off"],
                "distance_to_enemy": ["enemy_dist", "dist_to_enemy"],
                "aspect_angle_to_enemy": ["enemy_aspect", "aspect_angle"],
                "angle_off_to_enemy": ["enemy_angle_off", "angle_off", "ATA"],
                "missile_tracking_us": ["missile_warning", "missile_tracking"],
                "enemy_detected": ["enemy_detected", "target_detected"],
                "hvaa_detected": ["hvaa_detected"],
            }
            
            for our_name, possible_labels in label_mappings.items():
                for label in possible_labels:
                    if label in label_to_idx:
                        idx[our_name] = label_to_idx[label]
                        print(f"[FSM] Mapped '{our_name}' -> '{label}' (index {idx[our_name]})")
                        break

        return idx

    def _validate_obs_indices(self, obs: np.ndarray) -> None:
        """Validate that observation indices are within bounds."""
        if self._obs_index_validated:
            return
            
        n = len(obs)
        #Observation DEBUG
        #print(f"\n[FSM] OBSERVATION INDEX VALIDATION (obs length = {n})")
        
        issues = []
        #for name, idx in self.obs_index.items():
            #if idx >= n:
            #    issues.append(f"  ERROR: {name} index {idx} >= obs length {n}")
            #else:
            #    print(f"  OK: {name} = index {idx} -> value {obs[idx]:.4f}")
        
        if issues:
            print("\n[FSM] INDEX ISSUES DETECTED:")
            for issue in issues:
                print(issue)
            print("\nThis likely means enriched observations shifted the indices!")
            print("Check that obs_map is being read correctly.\n")
        
        self._obs_index_validated = True

    def _clip_action(self, a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=np.float32, copy=True)
        if a.shape[0] != 4:
            return np.zeros((4,), dtype=np.float32)
        a[HDG] = float(np.clip(a[HDG], -1.0, 1.0))
        a[ALT] = float(np.clip(a[ALT], -1.0, 1.0))
        a[G]   = float(np.clip(a[G],   -1.0, 1.0))
        a[FIRE]= 1.0 if float(a[FIRE]) > 0.5 else 0.0
        return a

    def _deg_to_hdg_units(self, deg: float) -> float:
        max_turn_deg = 110.0
        return float(np.clip(deg / max_turn_deg, -1.0, 1.0))

    def _hdg_to_angle(self, bearing_norm: float, desired_offset_deg: float) -> float:
        desired_norm = self._deg_to_hdg_units(desired_offset_deg)
        err = float(bearing_norm - desired_norm)
        k = 0.7
        hdg_cmd = -k * err
        return float(np.clip(hdg_cmd, -1.0, 1.0))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._obs_index_validated = False  # Re-validate on reset

        self._episode_steps = 0
        self._last_fire_step = -10**9
        self._prev_hvaa_dist = None
        self._hvaa_away_count = 0
        self._last_valid_hvaa_bearing = 0.0
        self._last_valid_hvaa_dist = None
        self._returning_to_hvaa = False
        self._hvaa_destroyed = False
        self._prev_hdg_cmd = 0.0
        self._prev_alt_cmd = 0.0
        self._prev_g_cmd   = 0.0


        self._hdg_smooth = 0.0
        self._hdg_smooth_enemy = 0.0
        self._prev_hdg_cmd = 0.0
        self._emergency_count = 0
        self._emergency_t = 0
        self._emergency_active = False
        self._prev_fire_request = False

        self.fsm_state = ExpertFSMState.DEFEND_HVAA
        self._fsm_t = 0
        self._defense_side = 1.0
        self._engage_count = 0
        self._enemy_lost_count = 0
        self._fired_in_this_engage = False

        return obs, info

    def _extract_state(self, obs: np.ndarray) -> Dict[str, Any]:
        """Extract state features from observation with validation."""
        state: Dict[str, Any] = {}
        obs = np.asarray(obs, dtype=np.float32)
        n = int(obs.shape[0])
        idx = self.obs_index
        
        # Validate on first call
        self._validate_obs_indices(obs)

        def get(i: int, default: float = 0.0) -> float:
            if i is not None and 0 <= i < n:
                return float(obs[i])
            return float(default)

        def get_bool(i: int, default: bool = False) -> bool:
            if i is not None and 0 <= i < n:
                return bool(obs[i] > 0.5)
            return bool(default)

        # Extract values
        state["threat_raw"] = get(idx.get("threat_raw"), 0.0) + 1.0
        state["offense_raw"] = get(idx.get("offense_raw"), -1.0) + 1.0

        state["distance_to_hvaa"] = get(idx.get("distance_to_hvaa"), -1.0)
        b = get(idx.get("hvaa_angle_off"), 0.0)
        if getattr(self, "invert_hvaa_bearing", False):
            b = -b
        state["hvaa_angle_off"] = b
        state["hvaa_detected"] = get_bool(idx.get("hvaa_detected"), True)

        # CRITICAL: Enemy detection
        state["enemy_detected"] = get_bool(idx.get("enemy_detected"), False)
        state["distance_to_enemy"] = get(idx.get("distance_to_enemy"), 1.0)
        state["aspect_angle_to_enemy"] = get(idx.get("aspect_angle_to_enemy"), 0.0)
        state["angle_off_to_enemy"] = get(idx.get("angle_off_to_enemy"), 0.0)

        # DEBUG: Print raw values periodically
        if self.debug and (self._global_step % 500 == 0):
            enemy_idx = idx.get("enemy_detected")
            dist_idx = idx.get("distance_to_enemy")
            angoff_idx = idx.get("angle_off_to_enemy")
            print(f"\n[STATE DEBUG] step={self._global_step}")
            print(f"  enemy_detected: idx={enemy_idx}, raw={obs[enemy_idx] if enemy_idx and enemy_idx < n else 'N/A'}, parsed={state['enemy_detected']}")
            print(f"  distance_to_enemy: idx={dist_idx}, raw={obs[dist_idx] if dist_idx and dist_idx < n else 'N/A'}, parsed={state['distance_to_enemy']:.3f}")
            print(f"  angle_off_to_enemy: idx={angoff_idx}, raw={obs[angoff_idx] if angoff_idx and angoff_idx < n else 'N/A'}, parsed={state['angle_off_to_enemy']:.3f}")
            print(f"  fsm_state={self.fsm_state}")

        # HVAA danger
        hvaa_close = (state["distance_to_hvaa"] >= 0.0) and (state["distance_to_hvaa"] < self.hvaa_protect_radius)
        enemy_close = state["distance_to_enemy"] < self.hvaa_danger_enemy_dist
        threat_high = state["threat_raw"] > self.hvaa_danger_threat
        state["hvaa_in_danger"] = bool(state["enemy_detected"]) and bool(hvaa_close) and bool(enemy_close or threat_high)

        state["missile_tracking_us"] = get_bool(idx.get("missile_tracking_us"), False)

        return state

    def expert_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute the expert (FSM) action for a given observation."""
        obs = np.asarray(obs, dtype=np.float32)
        self._last_obs = obs

        state = self._extract_state(obs)
        self._update_fsm(state)

        self.stats["expert_count"] += 1
        self.stats["expert_state_counts"][self.fsm_state.value] += 1
        
        action = self._expert_action_from_fsm(state)
        action = self._clip_action(action)

        # DEBUG: Log action generation
        if self.debug and (self._global_step % 500 == 0):
            print(f"[EXPERT ACTION] fsm={self.fsm_state} enemy={state['enemy_detected']} "
                  f"dist={state['distance_to_enemy']:.3f} angoff={state['angle_off_to_enemy']:.3f}")
            print(f"  -> action: hdg={action[HDG]:.3f} alt={action[ALT]:.3f} g={action[G]:.3f} fire={action[FIRE]:.1f}")

        return action

    def gate_action(self, action: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Gate an action (primarily fire gating) given current observation."""
        obs = np.asarray(obs, dtype=np.float32)
        state = self._extract_state(obs)

        a = np.asarray(action, dtype=np.float32).copy()

        # Rate limit heading changes
        hdg = float(a[HDG])
        hdg = float(np.clip(
            hdg,
            self._prev_hdg_cmd - self.max_hdg_delta_per_step,
            self._prev_hdg_cmd + self.max_hdg_delta_per_step
        ))
        self._prev_hdg_cmd = hdg
        a[HDG] = hdg

        # Apply fire gate
        a = self._apply_fire_gate(a, state)

        # Post-fire calm
        if not getattr(self, "aggressive_disable_post_fire_calm", False):
            if (self._time_step() - self._last_fire_step) < self.post_fire_calm_steps:
                a[HDG] = float(np.clip(a[HDG], -self.post_fire_max_hdg, self.post_fire_max_hdg))
                a[G]   = float(np.clip(a[G],   -self.post_fire_max_g,   self.post_fire_max_g))

        return self._clip_action(a)

    def _apply_fire_gate(self, action: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        """Apply fire gating with detailed debug output."""
        a = self._clip_action(action)
        t = self._time_step()
        
        # CRITICAL FIX: If we already accepted fire THIS EXACT STEP, preserve it!
        # This prevents multiple calls per timestep from overwriting fire=1.0
        if t == self._last_fire_step:
            a[FIRE] = 1.0  # Keep fire active
            return a
        
        want_fire = float(a[FIRE]) > 0.5
        
        if want_fire:
            self.stats["fire_requests"] += 1

        # Check if enemy is detected FIRST
        if not bool(state.get("enemy_detected", False)):
            if want_fire and self.debug:
                print(f"[FIRE BLOCKED] No enemy detected! enemy_detected={state.get('enemy_detected')}")
                self.stats["fire_blocked_no_enemy"] += 1
            a[FIRE] = 0.0
            self._prev_fire_request = False
            return a

        # Rising edge (if enabled)
        if getattr(self, "require_fire_rising_edge", False):
            rising_edge = want_fire and (not self._prev_fire_request)
            self._prev_fire_request = want_fire
            if not rising_edge:
                if self.debug and want_fire:
                    print("[FIRE BLOCKED] Rising edge required (holding fire)")
                a[FIRE] = 0.0
                return a
        else:
            if not want_fire:
                a[FIRE] = 0.0
                self._prev_fire_request = False
                return a

        # Cooldown check
        t = self._time_step()
        cooldown = int(self.fire_cooldown_steps)
        if (t - self._last_fire_step) < cooldown:
            if self.debug:
                print(f"[FIRE BLOCKED] Cooldown: t={t}, last_fire={self._last_fire_step}, cd={cooldown}")
            self.stats["fire_blocked_cooldown"] += 1
            a[FIRE] = 0.0
            self._prev_fire_request = False
            return a

        # Distance check
        dist = float(state.get("distance_to_enemy", 1.0))
        dist_gate = self.fire_dist_gate
        if dist > dist_gate:
            if self.debug:
                print(f"[FIRE BLOCKED] Distance: {dist:.3f} > gate {dist_gate:.3f}")
            self.stats["fire_blocked_distance"] += 1
            a[FIRE] = 0.0
            self._prev_fire_request = False
            return a

        # Aspect check
        aspect = float(state.get("aspect_angle_to_enemy", 0.0))
        aspect_gate = self.fire_aspect_gate
        if abs(aspect) > aspect_gate:
            if self.debug:
                print(f"[FIRE BLOCKED] Aspect: |{aspect:.3f}| > gate {aspect_gate:.3f}")
            self.stats["fire_blocked_aspect"] += 1
            a[FIRE] = 0.0
            self._prev_fire_request = False
            return a

        # Multi-shot check (if disabled)
        if (self.fsm_state == ExpertFSMState.ENGAGE and self._fired_in_this_engage
            and (not getattr(self, "aggressive_allow_multi_shot", False))):
            #if self.debug:
                #print("[FIRE BLOCKED] Already fired in this ENGAGE (multi-shot disabled)")
            a[FIRE] = 0.0
            self._prev_fire_request = False
            return a

        # FIRE ACCEPTED!
        #print(f"[FIRE ACCEPTED!] t={t} dist={dist:.3f} aspect={aspect:.3f}")
        self.stats["fire_accepted"] += 1
        
        a[FIRE] = 1.0
        self._last_fire_step = int(t)
        self._fired_in_this_engage = True
        
        # Transition to missile support
        self._transition(ExpertFSMState.MISSILE_SUPPORT)
        return a

    def _update_fsm(self, state: Dict[str, Any]) -> None:
        """Update FSM state."""
        self._fsm_t += 1
        enemy = bool(state.get("enemy_detected", False))

        # If HVAA is destroyed / missing, stop DEFEND logic and keep fighting red
        if self._hvaa_destroyed or (not bool(state.get("hvaa_detected", True))):
            if enemy and self.fsm_state != ExpertFSMState.ENGAGE:
                self._transition(ExpertFSMState.ENGAGE)
            return

        # AGGRESSIVE MODE: Always fight when enemy detected
        if getattr(self, "aggressive_shooter_mode", False):
            if enemy:
                # Reset "no enemy" counter when enemy is detected
                self._enemy_lost_count = 0
                if self.fsm_state != ExpertFSMState.ENGAGE:
                    #print(f"[FSM] Transitioning to ENGAGE (enemy detected)")
                    self._transition(ExpertFSMState.ENGAGE)
            else:
                # Require multiple consecutive "no enemy" observations before leaving ENGAGE
                self._enemy_lost_count += 1
                if self._enemy_lost_count >= 5:  # Wait 5 steps before leaving ENGAGE
                    if self.fsm_state != ExpertFSMState.DEFEND_HVAA:
                        #print(f"[FSM] Transitioning to DEFEND_HVAA (no enemy for {self._enemy_lost_count} steps)")
                        self._transition(ExpertFSMState.DEFEND_HVAA)
            return


    def _transition(self, new_state: ExpertFSMState) -> None:
        if new_state != self.fsm_state:
            if new_state == ExpertFSMState.DEFEND_HVAA:
                self._fired_in_this_engage = False
                self._emergency_count = 0
                self._emergency_t = 0
                self._emergency_active = False

            if new_state == ExpertFSMState.ENGAGE:
                self._defense_side = -1.0 if self._defense_side > 0 else 1.0

            self.fsm_state = new_state
            self._fsm_t = 0

    def _expert_action_from_fsm(self, state: Dict[str, Any]) -> np.ndarray:
        """Generate expert action based on current FSM state."""
        if self.fsm_state == ExpertFSMState.DEFEND_HVAA:
            return self._expert_defend_hvaa_action(state)
        if self.fsm_state == ExpertFSMState.ENGAGE:
            return self._expert_engage_action(state)
        if self.fsm_state == ExpertFSMState.MISSILE_SUPPORT:
            return self._expert_missile_support(state)
        return self._expert_evade(state)

    def _expert_defend_hvaa_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Defend HVAA behavior."""
        d = float(state.get("distance_to_hvaa", 0.0))
        bearing = float(state.get("hvaa_angle_off", 0.0))

        if d <= -0.999 or (not bool(state.get("hvaa_detected", True))) or self._hvaa_destroyed:
            # If HVAA missing/dead, keep fighting red if possible
            if bool(state.get("enemy_detected", False)):
                self._transition(ExpertFSMState.ENGAGE)
                return self._expert_engage_action(state)

            # Otherwise: be stable (don’t spiral away)
            return self._clip_action(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        self._last_valid_hvaa_bearing = bearing
        self._last_valid_hvaa_dist = d

        # Returning to HVAA logic
        if d > self.hvaa_ring_outer:
            self._returning_to_hvaa = True
        if d < self.hvaa_ring_inner:
            self._returning_to_hvaa = False

        if self._returning_to_hvaa:
            raw_hdg = self._hdg_to_angle(bearing, 18.0 * self._defense_side)
            raw_hdg = float(np.clip(1.2 * raw_hdg, -1.0, 1.0))
            self._hdg_smooth = (1 - self.hdg_smooth_alpha) * self._hdg_smooth + self.hdg_smooth_alpha * raw_hdg
            hdg_cmd = float(np.clip(self._hdg_smooth, -1.0, 1.0))
            return self._clip_action(np.array([hdg_cmd, 0.0, 0.6, 0.0], dtype=np.float32))

        # Orbit behavior
        center = 0.5 * (self.hvaa_ring_inner + self.hvaa_ring_outer)
        radial_err = (d - center) / max(1e-6, (self.hvaa_ring_outer - self.hvaa_ring_inner))
        base_orbit_deg = 35.0
        corr_deg = float(np.clip(-radial_err * 20.0, -20.0, 20.0))
        desired_deg = (base_orbit_deg + corr_deg) * self._defense_side
        raw_hdg = self._hdg_to_angle(bearing, desired_deg)

        self._hdg_smooth = (1 - self.hdg_smooth_alpha) * self._hdg_smooth + self.hdg_smooth_alpha * raw_hdg
        hdg_cmd = float(np.clip(self._hdg_smooth, -1.0, 1.0))

        return self._clip_action(np.array([hdg_cmd, 0.0, 0.30, 0.0], dtype=np.float32))

    def _expert_engage_action(self, state: Dict[str, Any]) -> np.ndarray:
        """ENGAGE behavior - pure pursuit + always request fire."""
        enemy = bool(state.get("enemy_detected", False))
        if not enemy:
            return self._expert_defend_hvaa_action(state)

        angoff = float(state.get("angle_off_to_enemy", 0.0))
        
        # AGGRESSIVE MODE: Pure pursuit, always request fire
        if getattr(self, "aggressive_shooter_mode", False):
            # Pure pursuit: turn toward enemy
            hdg_cmd = float(np.clip(-angoff, -1.0, 1.0))
            g_cmd = 1.0  # Max turn rate
            fire = 1.0   # Always request fire (gate will decide)
            
            if self.debug and (self._global_step % 500 == 0):
                print(f"[ENGAGE] angoff={angoff:.3f} -> hdg_cmd={hdg_cmd:.3f}, fire=1.0")
            
            return self._clip_action(np.array([hdg_cmd, 0.0, g_cmd, fire], dtype=np.float32))

        # Non-aggressive engage (original logic)
        # ...
        return self._clip_action(np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

    def _expert_missile_support(self, state: Dict[str, Any]) -> np.ndarray:
        """Post-shot support behavior."""
        enemy = bool(state.get("enemy_detected", False))
        if not enemy:
            return self._expert_defend_hvaa_action(state)

        angoff = float(state.get("angle_off_to_enemy", 0.0))
        crank_deg = 45.0 * self._defense_side
        raw_hdg = self._hdg_to_angle(angoff, crank_deg)
        raw_hdg = float(np.clip(raw_hdg, -self.max_enemy_hdg_cmd, self.max_enemy_hdg_cmd))

        self._hdg_smooth_enemy = (1 - self.hdg_smooth_alpha_enemy) * self._hdg_smooth_enemy + \
                                self.hdg_smooth_alpha_enemy * raw_hdg
        hdg_cmd = float(np.clip(self._hdg_smooth_enemy, -self.max_enemy_hdg_cmd, self.max_enemy_hdg_cmd))
        g_cmd = 0.30

        return self._clip_action(np.array([hdg_cmd, 0.0, g_cmd, 0.0], dtype=np.float32))

    def _expert_evade(self, state: Dict[str, Any]) -> np.ndarray:
        """Evade behavior."""
        enemy = bool(state.get("enemy_detected", False))
        angoff = float(state.get("angle_off_to_enemy", 0.0))

        if enemy:
            hdg_cmd = float(np.clip(-np.sign(angoff) * 1.0, -1.0, 1.0))
        else:
            hdg_cmd = float(np.clip(0.8 * self._defense_side, -1.0, 1.0))

        return self._clip_action(np.array([hdg_cmd, 0.0, 1.0, 0.0], dtype=np.float32))

    def step(self, action: np.ndarray):
        """Pass-through step."""
        self._episode_steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        # Detect HVAA destruction from info (robust scan)
        info = dict(info)
        blob = " ".join([
            str(info.get("reason", "")),
            str(info.get("termination_reason", "")),
            str(info.get("reasons", "")),
            str(info.get("episode_end_reasons", "")),
        ])
        if "HVAA_Destroyed" in blob:
            self._hvaa_destroyed = True

        info = dict(info)
        info["hier_fsm_state"] = str(self.fsm_state.value)
        info["hier_used_expert"] = False
        info["hier_global_step"] = int(self._global_step)
        info["hier_fire_stats"] = {
            "requests": self.stats["fire_requests"],
            "accepted": self.stats["fire_accepted"],
            "blocked_no_enemy": self.stats["fire_blocked_no_enemy"],
            "blocked_cooldown": self.stats["fire_blocked_cooldown"],
            "blocked_distance": self.stats["fire_blocked_distance"],
            "blocked_aspect": self.stats["fire_blocked_aspect"],
        }
        return obs, reward, terminated, truncated, info

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["greedy_count"] + self.stats["expert_count"]
        return {
            **self.stats,
            "episode_steps": int(self._episode_steps),
            "global_step": int(self._global_step),
            "greedy_prob_ref": self.curriculum.get_greedy_prob(self._global_step),
            "greedy_ratio": self.stats["greedy_count"] / max(1, total),
        }


if __name__ == "__main__":
    print("HierarchicalFSMActionWrapper FIXED - with proper observation index handling")
    w = HierarchicalFSMActionWrapper(env=None)
    print("OK - constructed wrapper class")