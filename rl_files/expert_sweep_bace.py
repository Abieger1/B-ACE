#!/usr/bin/env python3
"""
expert_sweep_bace.py

Parameter sweep for tuning aggressive expert in B-ACE environment.
Adapted from test_expert_standalone.py interface.

Usage:
    python expert_sweep_bace.py --config configs/SimpleExample_B_ACE_config.json --episodes 20
    python expert_sweep_bace.py --config configs/SimpleExample_B_ACE_config.json --episodes 50 --full-sweep
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time
import json
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

# Find repo root
THIS_FILE = Path(__file__).resolve()
current_path = THIS_FILE.parent
while current_path != current_path.parent:
    if (current_path / "b_ace_py").exists():
        REPO_ROOT = current_path
        break
    current_path = current_path.parent
else:
    REPO_ROOT = THIS_FILE.parent.parent

if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

SCRIPT_DIR = THIS_FILE.parent
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

from b_ace_py.utils import load_b_ace_config
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

@dataclass
class SweepConfig:
    """Configuration for a single parameter combination."""
    # Firing parameters
    fire_threshold: float = 0.50
    distance_min: float = 0.15
    distance_max: float = 0.40
    aspect_limit_deg: float = 30.0
    use_aspect_check: bool = False
    use_offensive_factor: bool = False
    fire_cooldown_steps: int = 100
    
    # Maneuvering parameters
    engage_turn_gain: float = 1.5
    engage_g_force: float = 0.7
    pursuit_turn_gain: float = 0.8
    missile_support_turn_gain: float = 0.2
    
    # Pursuit memory
    pursuit_memory_steps: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fire_threshold': self.fire_threshold,
            'distance_min': self.distance_min,
            'distance_max': self.distance_max,
            'aspect_limit_deg': self.aspect_limit_deg,
            'use_aspect_check': self.use_aspect_check,
            'use_offensive_factor': self.use_offensive_factor,
            'fire_cooldown_steps': self.fire_cooldown_steps,
            'engage_turn_gain': self.engage_turn_gain,
            'engage_g_force': self.engage_g_force,
            'pursuit_turn_gain': self.pursuit_turn_gain,
            'missile_support_turn_gain': self.missile_support_turn_gain,
            'pursuit_memory_steps': self.pursuit_memory_steps,
        }
    
    def short_name(self) -> str:
        """Short identifier for this config."""
        parts = [f"d{self.distance_min:.2f}-{self.distance_max:.2f}"]
        if self.use_aspect_check:
            parts.append(f"asp{self.aspect_limit_deg:.0f}")
        if self.use_offensive_factor:
            parts.append(f"off{self.fire_threshold:.2f}")
        parts.append(f"cd{self.fire_cooldown_steps}")
        parts.append(f"tg{self.engage_turn_gain:.1f}")
        parts.append(f"g{self.engage_g_force:.1f}")
        return "_".join(parts)


@dataclass 
class SweepResult:
    """Results from evaluating a single configuration."""
    config: SweepConfig
    episodes: int = 0
    blue_wins: int = 0
    red_wins: int = 0
    draws: int = 0
    total_reward: float = 0.0
    total_steps: int = 0
    total_missiles_fired: int = 0
    rewards: List[float] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        return self.blue_wins / max(1, self.episodes)
    
    @property
    def loss_rate(self) -> float:
        return self.red_wins / max(1, self.episodes)
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.episodes)
    
    @property
    def avg_steps(self) -> float:
        return self.total_steps / max(1, self.episodes)
    
    @property
    def avg_missiles(self) -> float:
        return self.total_missiles_fired / max(1, self.episodes)
    
    @property
    def reward_std(self) -> float:
        if len(self.rewards) < 2:
            return 0.0
        return float(np.std(self.rewards))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'config_name': self.config.short_name(),
            'episodes': self.episodes,
            'blue_wins': self.blue_wins,
            'red_wins': self.red_wins,
            'draws': self.draws,
            'win_rate': self.win_rate,
            'loss_rate': self.loss_rate,
            'avg_reward': self.avg_reward,
            'reward_std': self.reward_std,
            'avg_steps': self.avg_steps,
            'avg_missiles': self.avg_missiles,
        }


# =============================================================================
# CONFIGURABLE EXPERT
# =============================================================================

class ConfigurableExpert:
    """Expert with all tunable parameters exposed via SweepConfig."""
    
    def __init__(self, config: SweepConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        self.aspect_limit_norm = config.aspect_limit_deg / 180.0
        
        # State
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0
        
        # Stats
        self.fire_count = 0
        self.engage_count = 0
        
    def reset(self):
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0
        
    def get_action(self, obs: np.ndarray, obs_indices: Dict[str, int]) -> np.ndarray:
        """Get action from observation using configured parameters."""
        self.step_count += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        n = len(obs)
        cfg = self.config
        
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
        offensive = get_val('offensive_factor', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        hvaa_dist = get_val('distance_to_hvaa', 0.3)
        hvaa_angle = get_val('hvaa_angle_off', 0.0)
        
        valid_enemy = enemy_detected and distance >= 0
        
        # Update pursuit memory
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off
        
        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < cfg.pursuit_memory_steps
        
        if valid_enemy:
            # === ENGAGE MODE ===
            self.engage_count += 1
            
            if missile_in_flight:
                turn = np.clip(-angle_off * cfg.missile_support_turn_gain, -0.15, 0.15)
                g_force = 0.3
            else:
                if abs(angle_off) < 0.5:
                    turn = np.clip(-angle_off * cfg.engage_turn_gain, -0.8, 0.8)
                    g_force = cfg.engage_g_force
                else:
                    if abs(hvaa_angle) > 0.1:
                        turn = np.clip(hvaa_angle * cfg.engage_turn_gain, -0.8, 0.8)
                    else:
                        turn = np.clip(-angle_off * cfg.engage_turn_gain * 0.7, -0.8, 0.8)
                    g_force = min(cfg.engage_g_force + 0.1, 1.0)
                    
            altitude = 0.1
            
            # === CONFIGURABLE FIRE DECISION ===
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                dist_ok = cfg.distance_min < distance < cfg.distance_max
                
                aspect_ok = True
                if cfg.use_aspect_check:
                    aspect_ok = abs(aspect) < self.aspect_limit_norm
                
                offensive_ok = True
                if cfg.use_offensive_factor:
                    offensive_ok = offensive > cfg.fire_threshold
                
                if dist_ok and aspect_ok and offensive_ok:
                    fire = 1.0
                    self.fire_cooldown = cfg.fire_cooldown_steps
                    self.fire_count += 1
                    
        elif in_pursuit:
            # === PURSUIT MODE ===
            if missile_in_flight:
                turn = np.clip(-self.last_known_angle_off * cfg.missile_support_turn_gain * 0.75, -0.1, 0.1)
                g_force = 0.3
            else:
                if abs(self.last_known_angle_off) < 0.5:
                    turn = np.clip(-self.last_known_angle_off * cfg.pursuit_turn_gain, -0.5, 0.5)
                    g_force = 0.6
                else:
                    if abs(hvaa_angle) > 0.1:
                        turn = np.clip(hvaa_angle * cfg.pursuit_turn_gain * 1.25, -0.6, 0.6)
                    else:
                        turn = 0.0
                    g_force = 0.5
            altitude = 0.0
            fire = 0.0
            
        else:
            # === PATROL MODE ===
            if hvaa_dist > 0.10:
                turn = np.clip(hvaa_angle * 2.0, -1.0, 1.0)
                g_force = 0.6
            else:
                turn = 0.0
                g_force = 0.4
            altitude = 0.0
            fire = 0.0
        
        return np.array([turn, altitude, g_force, fire], dtype=np.float32)


# =============================================================================
# SWEEP CONFIGURATIONS
# =============================================================================

def generate_focused_sweep_configs() -> List[SweepConfig]:
    """
    Generate a focused set of configs for initial exploration.
    Tests the most impactful parameter variations.
    """
    configs = []
    
    # 1. Baseline (current settings)
    configs.append(SweepConfig(
        distance_min=0.15, distance_max=0.40,
        use_aspect_check=False, use_offensive_factor=False,
        fire_cooldown_steps=100,
        engage_turn_gain=1.5, engage_g_force=0.7,
    ))
    
    # 2. Wider firing envelope
    configs.append(SweepConfig(
        distance_min=0.10, distance_max=0.50,
        use_aspect_check=False, use_offensive_factor=False,
        fire_cooldown_steps=100,
        engage_turn_gain=1.5, engage_g_force=0.7,
    ))
    
    # 3. Use offensive factor for fire decision
    configs.append(SweepConfig(
        distance_min=0.10, distance_max=0.50,
        use_aspect_check=False, use_offensive_factor=True,
        fire_threshold=0.50,
        fire_cooldown_steps=100,
        engage_turn_gain=1.5, engage_g_force=0.7,
    ))
    
    # 4. Add aspect angle check
    configs.append(SweepConfig(
        distance_min=0.10, distance_max=0.50,
        use_aspect_check=True, aspect_limit_deg=30.0,
        use_offensive_factor=True, fire_threshold=0.50,
        fire_cooldown_steps=100,
        engage_turn_gain=1.5, engage_g_force=0.7,
    ))
    
    # 5. More aggressive maneuvering
    configs.append(SweepConfig(
        distance_min=0.10, distance_max=0.50,
        use_aspect_check=True, aspect_limit_deg=30.0,
        use_offensive_factor=True, fire_threshold=0.50,
        fire_cooldown_steps=100,
        engage_turn_gain=2.0, engage_g_force=0.9,
    ))
    
    # 6. Faster follow-up shots
    configs.append(SweepConfig(
        distance_min=0.10, distance_max=0.50,
        use_aspect_check=True, aspect_limit_deg=30.0,
        use_offensive_factor=True, fire_threshold=0.50,
        fire_cooldown_steps=50,
        engage_turn_gain=2.0, engage_g_force=0.9,
    ))
    
    # 7. Lower fire threshold (more aggressive firing)
    configs.append(SweepConfig(
        distance_min=0.10, distance_max=0.50,
        use_aspect_check=True, aspect_limit_deg=45.0,
        use_offensive_factor=True, fire_threshold=0.30,
        fire_cooldown_steps=50,
        engage_turn_gain=2.0, engage_g_force=0.9,
    ))
    
    # 8. Very aggressive - fire based on distance only, high g
    configs.append(SweepConfig(
        distance_min=0.08, distance_max=0.55,
        use_aspect_check=False, use_offensive_factor=False,
        fire_cooldown_steps=30,
        engage_turn_gain=2.5, engage_g_force=1.0,
    ))
    
    # 9. Conservative aspect, aggressive maneuver
    configs.append(SweepConfig(
        distance_min=0.12, distance_max=0.45,
        use_aspect_check=True, aspect_limit_deg=20.0,
        use_offensive_factor=True, fire_threshold=0.60,
        fire_cooldown_steps=75,
        engage_turn_gain=2.0, engage_g_force=0.85,
    ))
    
    # 10. Wide aspect, moderate aggression
    configs.append(SweepConfig(
        distance_min=0.10, distance_max=0.50,
        use_aspect_check=True, aspect_limit_deg=60.0,
        use_offensive_factor=False,
        fire_cooldown_steps=60,
        engage_turn_gain=1.8, engage_g_force=0.8,
    ))
    
    return configs


def generate_full_sweep_configs() -> List[SweepConfig]:
    """
    Generate a comprehensive grid search.
    Warning: Can be many configs!
    """
    configs = []
    
    # Parameter ranges
    distance_ranges = [(0.10, 0.40), (0.10, 0.50), (0.15, 0.45), (0.08, 0.55)]
    aspect_configs = [(False, 30.0), (True, 20.0), (True, 30.0), (True, 45.0)]
    offensive_configs = [(False, 0.50), (True, 0.30), (True, 0.50), (True, 0.70)]
    fire_cooldowns = [30, 50, 75, 100]
    maneuver_configs = [(1.5, 0.7), (2.0, 0.8), (2.0, 0.9), (2.5, 1.0)]
    
    for (d_min, d_max) in distance_ranges:
        for (use_asp, asp_deg) in aspect_configs:
            for (use_off, off_thresh) in offensive_configs:
                for cd in fire_cooldowns:
                    for (tg, gf) in maneuver_configs:
                        configs.append(SweepConfig(
                            distance_min=d_min,
                            distance_max=d_max,
                            use_aspect_check=use_asp,
                            aspect_limit_deg=asp_deg,
                            use_offensive_factor=use_off,
                            fire_threshold=off_thresh,
                            fire_cooldown_steps=cd,
                            engage_turn_gain=tg,
                            engage_g_force=gf,
                        ))
    
    return configs


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================

def resolve_env_path() -> str:
    """Find the Godot binary for the current platform."""
    platform = sys.platform
    if platform == "darwin":
        candidates = [REPO_ROOT / "Godot_Air_Combat" / "B_ACE.app"]
    elif platform.startswith("linux"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.x86_64"]
    elif platform.startswith("win"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.exe"]
    else:
        raise RuntimeError(f"Unsupported platform '{platform}'.")
    
    for binary in candidates:
        if binary.exists():
            return binary.as_posix()
    raise FileNotFoundError(f"Could not find Godot binary in {candidates}")


def get_default_obs_indices() -> Dict[str, int]:
    """Default observation indices matching Godot B-ACE output."""
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


def get_agent_id(env) -> Any:
    """Determine the blue agent ID from environment."""
    agents = getattr(env, "possible_agents", None) or getattr(env, "agents", None)
    if agents:
        agent_id = agents[0]
        for a in agents:
            if "blue" in str(a).lower() or str(a) == "agent_0" or (isinstance(a, int) and a < 200):
                agent_id = a
                break
        return agent_id
    return "agent_0"


def extract_obs(obs_dict: dict, agent_id: Any) -> np.ndarray:
    """Extract observation array from obs_dict, handling nested structures."""
    raw_obs = None
    if agent_id in obs_dict:
        raw_obs = obs_dict[agent_id]
    else:
        for k, v in obs_dict.items():
            if str(k) == str(agent_id):
                raw_obs = v
                break
        if raw_obs is None:
            raw_obs = list(obs_dict.values())[0]
    
    if isinstance(raw_obs, dict):
        if 'obs' in raw_obs:
            inner = raw_obs['obs']
            if isinstance(inner, dict) and 'obs' in inner:
                obs = np.asarray(inner['obs'], dtype=np.float32)
            elif isinstance(inner, (list, np.ndarray)):
                obs = np.asarray(inner, dtype=np.float32)
            else:
                obs = np.asarray(list(inner.values()), dtype=np.float32)
        else:
            obs = np.asarray(list(raw_obs.values()), dtype=np.float32)
    else:
        obs = np.asarray(raw_obs, dtype=np.float32)
    
    return obs


def check_done(term_dict: dict, trunc_dict: dict, agent_id: Any) -> bool:
    """Check if episode is done."""
    done = False
    keys_to_check = [agent_id, str(agent_id), "__all__", "agent_0", 101, "101"]
    
    if isinstance(term_dict, dict):
        for key in keys_to_check:
            if key in term_dict and term_dict[key]:
                done = True
                break
    else:
        done = bool(term_dict)
    
    if isinstance(trunc_dict, dict):
        for key in keys_to_check:
            if key in trunc_dict and trunc_dict[key]:
                done = True
                break
    else:
        done = done or bool(trunc_dict)
    
    return done


# =============================================================================
# EPISODE RUNNER
# =============================================================================

def run_episode(
    env,
    expert: ConfigurableExpert,
    obs_indices: Dict[str, int],
    agent_id: Any,
    max_steps: int = 15000,
) -> Dict[str, Any]:
    """Run single episode with expert policy."""
    obs_dict, info_dict = env.reset()
    expert.reset()
    
    total_reward = 0.0
    steps = 0
    done = False
    missiles_fired = 0
    blue_kills = 0
    red_kills = 0
    
    while not done and steps < max_steps:
        obs = extract_obs(obs_dict, agent_id)
        action = expert.get_action(obs, obs_indices)
        
        if action[3] > 0.5:
            missiles_fired += 1
        
        action_dict = {agent_id: action}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
        
        # Get reward
        if isinstance(rew_dict, dict):
            reward = rew_dict.get(agent_id) or rew_dict.get(str(agent_id)) or list(rew_dict.values())[0]
            reward = float(reward)
        else:
            reward = float(rew_dict)
        
        total_reward += reward
        steps += 1
        
        # Check kills from info
        if isinstance(info_dict, dict):
            info = info_dict.get(agent_id) or info_dict.get(str(agent_id)) or {}
        else:
            info = {}
        if info.get("blue_kill", False) or info.get("enemy_killed", False):
            blue_kills += 1
        if info.get("red_kill", False) or info.get("blue_killed", False):
            red_kills += 1
        
        done = check_done(term_dict, trunc_dict, agent_id)
    
    # Infer outcome from reward if info didn't provide it
    if blue_kills == 0 and red_kills == 0:
        if total_reward > 10:
            blue_kills = 1
        elif total_reward < -10:
            red_kills = 1
    
    return {
        "reward": total_reward,
        "steps": steps,
        "blue_kills": blue_kills,
        "red_kills": red_kills,
        "missiles_fired": missiles_fired,
    }


# =============================================================================
# SWEEP RUNNER
# =============================================================================

def run_sweep(
    env,
    configs: List[SweepConfig],
    episodes_per_config: int,
    obs_indices: Dict[str, int],
    agent_id: Any,
    verbose: bool = False,
) -> List[SweepResult]:
    """Run the parameter sweep."""
    results = []
    
    print(f"\nRunning sweep: {len(configs)} configs × {episodes_per_config} episodes = "
          f"{len(configs) * episodes_per_config} total episodes")
    print("=" * 70)
    
    for cfg_idx, config in enumerate(configs):
        result = SweepResult(config=config)
        expert = ConfigurableExpert(config, debug=False)
        
        print(f"\n[{cfg_idx+1}/{len(configs)}] {config.short_name()}")
        
        ep_start = time.time()
        for ep in range(episodes_per_config):
            ep_result = run_episode(env, expert, obs_indices, agent_id)
            
            result.episodes += 1
            result.total_reward += ep_result["reward"]
            result.total_steps += ep_result["steps"]
            result.total_missiles_fired += ep_result["missiles_fired"]
            result.rewards.append(ep_result["reward"])
            
            if ep_result["blue_kills"] > 0:
                result.blue_wins += 1
            elif ep_result["red_kills"] > 0:
                result.red_wins += 1
            else:
                result.draws += 1
            
            if verbose:
                outcome = "W" if ep_result["blue_kills"] > 0 else "L" if ep_result["red_kills"] > 0 else "D"
                print(f"  Ep {ep+1}: {outcome} r={ep_result['reward']:.1f}")
        
        elapsed = time.time() - ep_start
        print(f"  Win: {result.win_rate:.0%} | Loss: {result.loss_rate:.0%} | "
              f"Avg reward: {result.avg_reward:.1f} ± {result.reward_std:.1f} | "
              f"{elapsed:.1f}s")
        
        results.append(result)
    
    return results


def print_summary(results: List[SweepResult]):
    """Print ranked summary of sweep results."""
    print("\n" + "=" * 70)
    print("SWEEP RESULTS - RANKED BY WIN RATE")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda r: (r.win_rate, r.avg_reward), reverse=True)
    
    print(f"\n{'Rank':<5} {'Win%':<7} {'Loss%':<7} {'Reward':<12} {'Config'}")
    print("-" * 70)
    
    for i, r in enumerate(sorted_results):
        reward_str = f"{r.avg_reward:.1f}±{r.reward_std:.1f}"
        print(f"{i+1:<5} {r.win_rate:>5.0%}   {r.loss_rate:>5.0%}   {reward_str:<12} {r.config.short_name()}")
    
    # Best config details
    best = sorted_results[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"  Win rate:     {best.win_rate:.1%}")
    print(f"  Loss rate:    {best.loss_rate:.1%}")
    print(f"  Avg reward:   {best.avg_reward:.2f} ± {best.reward_std:.2f}")
    print(f"  Avg steps:    {best.avg_steps:.0f}")
    print(f"  Avg missiles: {best.avg_missiles:.1f}")
    print("\n  Parameters:")
    for k, v in best.config.to_dict().items():
        print(f"    {k}: {v}")


def save_results(results: List[SweepResult], output_dir: Path):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"sweep_results_{timestamp}.json"
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'num_configs': len(results),
        'results': [r.to_dict() for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for B-ACE expert")
    parser.add_argument(
        "--config",
        type=str,
        default="b_ace_py/SimpleExample_B_ACE_config.json",
        help="B-ACE config JSON"
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per config")
    parser.add_argument("--full-sweep", action="store_true", help="Run full grid search (slow!)")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode results")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--output-dir", type=str, default="./sweep_results", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXPERT PARAMETER SWEEP")
    print("=" * 70)
    
    # Load config
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        cwd_path = Path.cwd() / config_path
        if cwd_path.exists():
            config_path = cwd_path.resolve()
        else:
            repo_path = REPO_ROOT / config_path
            if repo_path.exists():
                config_path = repo_path.resolve()
            else:
                config_path = config_path.resolve()
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_b_ace_config(config_path.as_posix())
    if config is None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Configure environment
    env_cfg = config.setdefault("EnvConfig", {})
    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = resolve_env_path()
    
    agents_cfg = config.setdefault("AgentsConfig", {})
    agents_cfg.setdefault("blue_agents", {})["base_behavior"] = "external"
    agents_cfg.setdefault("red_agents", {})["base_behavior"] = "baseline1"
    
    env_cfg["renderize"] = 1 if args.render else 0
    env_cfg["speed_up"] = 200
    
    print(f"Config: {config_path}")
    print(f"Episodes per config: {args.episodes}")
    
    # Generate sweep configs
    if args.full_sweep:
        sweep_configs = generate_full_sweep_configs()
        print(f"Mode: FULL SWEEP ({len(sweep_configs)} configurations)")
    else:
        sweep_configs = generate_focused_sweep_configs()
        print(f"Mode: FOCUSED SWEEP ({len(sweep_configs)} configurations)")
    
    # Create environment
    env = B_ACE_GodotPettingZooWrapper(device="cpu", **config)
    obs_indices = get_default_obs_indices()
    agent_id = get_agent_id(env)
    
    print(f"Agent ID: {agent_id}")
    
    # Run sweep
    results = run_sweep(
        env=env,
        configs=sweep_configs,
        episodes_per_config=args.episodes,
        obs_indices=obs_indices,
        agent_id=agent_id,
        verbose=args.verbose,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, output_dir)
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
