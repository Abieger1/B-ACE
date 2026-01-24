#!/usr/bin/env python3
"""
test_expert_standalone.py

Test aggressive expert in B-ACE environment to verify it achieves kills.
Run this BEFORE using the expert for alignment training.

Usage:
    # Test original simple expert
    python rl_files/test_expert_standalone.py --expert simple --episodes 20
    
    # Test new aggressive hunter expert
    python rl_files/test_expert_standalone.py --expert hunter --episodes 20
    
    # Test hunter V2 with lead pursuit
    python rl_files/test_expert_standalone.py --expert hunter_v2 --episodes 20
    
    # Compare all experts
    python rl_files/test_expert_standalone.py --expert all --episodes 20
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time
import json

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

# Import experts
from simple_aggressive_expert import SimpleAggressiveExpert, get_default_obs_indices
from aggressive_hunter_fixed import AggressiveHunterFixed

# Try to import the new aggressive hunter expert
try:
    from aggressive_hunter_expert import (
        AggressiveHunterExpert, 
        AggressiveHunterExpertV2,
        get_default_obs_indices as get_hunter_obs_indices
    )
    HUNTER_AVAILABLE = True
except ImportError:
    HUNTER_AVAILABLE = False
    print("Note: aggressive_hunter_expert.py not found in path.")

# Try to import improved escort experts
try:
    from improved_escort_expert import (
        ImprovedEscortExpert,
        TunedEscortExpert,
        ConservativeEscortExpert,
    )
    ESCORT_AVAILABLE = True
except ImportError:
    ESCORT_AVAILABLE = False
    print("Note: improved_escort_expert.py not found in path.")

# Try to import fixed hunter experts
try:
    from aggressive_hunter_fixed import (
        AggressiveHunterFixed,
        AggressiveHunterV3,
        AggressiveHunterTunable,
    )
    HUNTER_FIXED_AVAILABLE = True
except ImportError:
    HUNTER_FIXED_AVAILABLE = False
    print("Note: aggressive_hunter_fixed.py not found in path.")


def create_expert(expert_type: str, debug: bool = False):
    """Factory function to create the requested expert type."""
    
    if expert_type == "simple":
        return SimpleAggressiveExpert(
            fire_threshold=0.50,
            aspect_limit_deg=30.0,
            debug=debug
        ), "SimpleAggressiveExpert"
    
    # === IMPROVED ESCORT EXPERTS (recommended) ===
    elif expert_type == "escort":
        if not ESCORT_AVAILABLE:
            raise ImportError("ImprovedEscortExpert not available. Check import.")
        return ImprovedEscortExpert(debug=debug), "ImprovedEscortExpert"
    
    elif expert_type == "escort_tuned":
        if not ESCORT_AVAILABLE:
            raise ImportError("TunedEscortExpert not available. Check import.")
        return TunedEscortExpert(debug=debug), "TunedEscortExpert"
    
    elif expert_type == "escort_conservative":
        if not ESCORT_AVAILABLE:
            raise ImportError("ConservativeEscortExpert not available. Check import.")
        return ConservativeEscortExpert(debug=debug), "ConservativeEscortExpert"
    
    # === HUNTER EXPERTS (for non-escort missions) ===
    elif expert_type == "hunter":
        if not HUNTER_AVAILABLE:
            raise ImportError("AggressiveHunterExpert not available. Check import.")
        return AggressiveHunterExpert(
            fire_range_min=0.08,
            fire_range_max=0.55,
            fire_cooldown_steps=80,
            pursuit_gain=2.5,
            max_g=1.0,
            debug=debug
        ), "AggressiveHunterExpert"
    
    elif expert_type == "hunter_v2":
        if not HUNTER_AVAILABLE:
            raise ImportError("AggressiveHunterExpertV2 not available. Check import.")
        return AggressiveHunterExpertV2(
            fire_range_min=0.08,
            fire_range_max=0.55,
            fire_cooldown_steps=70,
            lead_factor=0.3,
            debug=debug
        ), "AggressiveHunterExpertV2"
    
    elif expert_type == "hunter_conservative":
        if not HUNTER_AVAILABLE:
            raise ImportError("AggressiveHunterExpert not available. Check import.")
        return AggressiveHunterExpert(
            fire_range_min=0.10,
            fire_range_max=0.45,
            fire_cooldown_steps=100,
            pursuit_gain=2.0,
            max_g=0.9,
            debug=debug
        ), "AggressiveHunterExpert (conservative)"
    
    elif expert_type == "hunter_aggressive":
        if not HUNTER_AVAILABLE:
            raise ImportError("AggressiveHunterExpert not available. Check import.")
        return AggressiveHunterExpert(
            fire_range_min=0.06,
            fire_range_max=0.60,
            fire_cooldown_steps=60,
            pursuit_gain=3.0,
            max_g=1.0,
            debug=debug
        ), "AggressiveHunterExpert (max aggression)"
    
    # === FIXED HUNTER EXPERTS (recommended for hunting) ===
    elif expert_type == "hunter_fixed":
        if not HUNTER_FIXED_AVAILABLE:
            raise ImportError("AggressiveHunterFixed not available. Check import.")
        return AggressiveHunterFixed(debug=debug), "AggressiveHunterFixed"
    
    elif expert_type == "hunter_v3":
        if not HUNTER_FIXED_AVAILABLE:
            raise ImportError("AggressiveHunterV3 not available. Check import.")
        return AggressiveHunterV3(debug=debug), "AggressiveHunterV3"
    
    elif expert_type == "hunter_tunable":
        if not HUNTER_FIXED_AVAILABLE:
            raise ImportError("AggressiveHunterTunable not available. Check import.")
        # Default tunable parameters - can be customized
        return AggressiveHunterTunable(
            fire_range_min=0.12,
            fire_range_max=0.35,
            fire_cooldown_steps=100,
            pursuit_gain=1.8,
            angle_deadband=0.05,
            debug=debug
        ), "AggressiveHunterTunable"
    
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")


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


def run_episode(env, expert, obs_indices, agent_id="agent_0", render=False, verbose=False):
    """Run single episode with expert policy."""
    obs_dict, info_dict = env.reset()
    expert.reset()
    
    total_reward = 0.0
    steps = 0
    done = False
    
    # Track outcomes
    blue_kills = 0
    red_kills = 0
    missiles_fired = 0
    
    # Track behavior for analysis
    engage_steps = 0
    patrol_steps = 0
    reversal_count = 0
    last_turn = 0.0
    
    while not done:
        # Get observation for controlled agent
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
        
        # Handle dict observations
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
        
        # Get expert action
        action = expert.get_action(obs, obs_indices)
        
        # Track behavior
        if action[3] > 0.5:
            missiles_fired += 1
        
        # Detect reversals (large turn changes)
        if abs(action[0] - last_turn) > 1.5:
            reversal_count += 1
        last_turn = action[0]
        
        # Track engage vs patrol (check if enemy detected)
        enemy_detected_idx = obs_indices.get('enemy_detected', 26)
        if enemy_detected_idx < len(obs) and obs[enemy_detected_idx] > 0.5:
            engage_steps += 1
        else:
            patrol_steps += 1
        
        # Build action dict
        action_dict = {agent_id: action}
        
        # Step environment
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
        
        # Get reward
        if isinstance(rew_dict, dict):
            reward = rew_dict.get(agent_id) or rew_dict.get(str(agent_id)) or list(rew_dict.values())[0]
            reward = float(reward)
        else:
            reward = float(rew_dict)
        
        total_reward += reward
        steps += 1
        
        # Check for kills from info
        if isinstance(info_dict, dict):
            info = info_dict.get(agent_id) or info_dict.get(str(agent_id)) or {}
        else:
            info = {}
        if info.get("blue_kill", False) or info.get("enemy_killed", False) or info.get("red_killed_event", False):
            blue_kills += 1
        if info.get("red_kill", False) or info.get("blue_killed", False) or info.get("hvaa_destroyed", False):
            red_kills += 1
        
        # Check termination
        done = False
        if isinstance(term_dict, dict):
            for key in [agent_id, str(agent_id), "__all__", "agent_0", 101, "101"]:
                if key in term_dict and term_dict[key]:
                    done = True
                    break
        else:
            done = bool(term_dict)
        
        if isinstance(trunc_dict, dict):
            for key in [agent_id, str(agent_id), "__all__", "agent_0", 101, "101"]:
                if key in trunc_dict and trunc_dict[key]:
                    done = True
                    break
        else:
            done = done or bool(trunc_dict)
        
        MAX_EPISODE_STEPS = 15000
        if steps >= MAX_EPISODE_STEPS:
            if verbose:
                print(f"  [WARN] Max steps ({MAX_EPISODE_STEPS}) reached")
            done = True
        
        if verbose and steps % 500 == 0:
            print(f"  Step {steps}: reward={reward:.2f}, total={total_reward:.2f}")
    
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
        "engage_steps": engage_steps,
        "patrol_steps": patrol_steps,
        "reversal_count": reversal_count,
    }


def test_expert(env, expert, expert_name, obs_indices, agent_id, num_episodes, render=False, verbose=False):
    """Test a single expert and return results."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {expert_name}")
    print(f"{'='*60}")
    
    results = []
    
    for ep in range(num_episodes):
        start = time.time()
        result = run_episode(env, expert, obs_indices, agent_id, render, verbose)
        elapsed = time.time() - start
        
        results.append(result)
        
        outcome = "BLUE_WIN" if result["blue_kills"] > 0 else "RED_WIN" if result["red_kills"] > 0 else "DRAW"
        print(f"Ep {ep+1:3d}: {outcome:8s} | reward={result['reward']:7.1f} | "
              f"steps={result['steps']:4d} | missiles={result['missiles_fired']} | {elapsed:.1f}s")
    
    # Compute statistics
    rewards = [r["reward"] for r in results]
    blue_wins = sum(1 for r in results if r["blue_kills"] > 0)
    red_wins = sum(1 for r in results if r["red_kills"] > 0)
    draws = num_episodes - blue_wins - red_wins
    total_missiles = sum(r["missiles_fired"] for r in results)
    total_engage = sum(r["engage_steps"] for r in results)
    total_patrol = sum(r["patrol_steps"] for r in results)
    total_reversals = sum(r["reversal_count"] for r in results)
    
    stats = {
        "expert_name": expert_name,
        "num_episodes": num_episodes,
        "blue_wins": blue_wins,
        "blue_win_rate": 100 * blue_wins / num_episodes,
        "red_wins": red_wins,
        "red_win_rate": 100 * red_wins / num_episodes,
        "draws": draws,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "total_missiles": total_missiles,
        "avg_missiles": total_missiles / num_episodes,
        "engage_ratio": total_engage / (total_engage + total_patrol + 1e-6),
        "avg_reversals": total_reversals / num_episodes,
    }
    
    print(f"\n{expert_name} Summary:")
    print(f"  Blue wins:    {blue_wins}/{num_episodes} ({stats['blue_win_rate']:.1f}%)")
    print(f"  Red wins:     {red_wins}/{num_episodes} ({stats['red_win_rate']:.1f}%)")
    print(f"  Mean reward:  {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Missiles/ep:  {stats['avg_missiles']:.1f}")
    print(f"  Engage ratio: {stats['engage_ratio']:.1%}")
    print(f"  Reversals/ep: {stats['avg_reversals']:.1f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Test aggressive expert(s) in B-ACE")
    parser.add_argument(
        "--config", 
        type=str, 
        default="b_ace_py/SimpleExample_B_ACE_config.json",
        help="B-ACE config JSON"
    )
    parser.add_argument(
        "--expert",
        type=str,
        default="simple",
        choices=["simple", 
                 "escort", "escort_tuned", "escort_conservative",
                 "hunter", "hunter_v2", "hunter_conservative", "hunter_aggressive",
                 "hunter_fixed", "hunter_v3", "hunter_tunable",  # NEW: Fixed hunters
                 "all", "all_escort", "all_hunter"],  # NEW: all_hunter
        help="Expert type to test. Use 'all_hunter' to compare hunter variants."
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of test episodes")
    parser.add_argument("--verbose", action="store_true", help="Print per-step info")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXPERT COMPARISON TEST")
    print("=" * 60)
    
    # Check expert availability
    if args.expert in ["escort", "escort_tuned", "escort_conservative", "all_escort"] and not ESCORT_AVAILABLE:
        print(f"\nERROR: Escort experts not available!")
        print(f"Make sure improved_escort_expert.py is in: {SCRIPT_DIR}")
        sys.exit(1)
    
    if args.expert in ["hunter", "hunter_v2", "hunter_conservative", "hunter_aggressive"] and not HUNTER_AVAILABLE:
        print(f"\nERROR: Hunter experts not available!")
        print(f"Make sure aggressive_hunter_expert.py is in: {SCRIPT_DIR}")
        sys.exit(1)
    
    if args.expert in ["hunter_fixed", "hunter_v3", "hunter_tunable", "all_hunter"] and not HUNTER_FIXED_AVAILABLE:
        print(f"\nERROR: Fixed hunter experts not available!")
        print(f"Make sure aggressive_hunter_fixed.py is in: {SCRIPT_DIR}")
        sys.exit(1)
    
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
    else:
        config_path = config_path.resolve()
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_b_ace_config(config_path.as_posix())
    if config is None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Set up env
    env_cfg = config.setdefault("EnvConfig", {})
    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = resolve_env_path()
    
    agents_cfg = config.setdefault("AgentsConfig", {})
    agents_cfg.setdefault("blue_agents", {})["base_behavior"] = "external"
    agents_cfg.setdefault("red_agents", {})["base_behavior"] = "baseline1"
    
    env_cfg["renderize"] = 1 if args.render else 0
    env_cfg["speed_up"] = 200
    
    print(f"Config: {config_path}")
    print(f"Episodes per expert: {args.episodes}")
    print(f"Red behavior: baseline1")
    
    # Create environment
    env = B_ACE_GodotPettingZooWrapper(device="cpu", **config)
    
    # Get observation indices
    obs_indices = get_default_obs_indices()
    
    # Determine agent ID
    agents = getattr(env, "possible_agents", None) or getattr(env, "agents", None)
    if agents:
        agent_id = agents[0]
        for a in agents:
            if "blue" in str(a).lower() or str(a) == "agent_0" or (isinstance(a, int) and a < 200):
                agent_id = a
                break
    else:
        agent_id = "agent_0"
    
    print(f"Controlling agent: {agent_id}")
    
    # Determine which experts to test
    if args.expert == "all":
        expert_types = ["simple"]
        if ESCORT_AVAILABLE:
            expert_types.extend(["escort", "escort_tuned", "escort_conservative"])
        if HUNTER_FIXED_AVAILABLE:
            expert_types.extend(["hunter_fixed", "hunter_v3"])
        if HUNTER_AVAILABLE:
            expert_types.extend(["hunter"])
    elif args.expert == "all_escort":
        # Compare only escort-style experts
        expert_types = ["simple"]
        if ESCORT_AVAILABLE:
            expert_types.extend(["escort", "escort_tuned", "escort_conservative"])
        else:
            print("\nNote: Escort experts not available, testing only 'simple'")
    elif args.expert == "all_hunter":
        # Compare hunter experts (recommended for aggressive hunting)
        expert_types = ["simple"]
        if HUNTER_FIXED_AVAILABLE:
            expert_types.extend(["hunter_fixed", "hunter_v3", "hunter_tunable"])
        if HUNTER_AVAILABLE:
            expert_types.extend(["hunter", "hunter_conservative"])
        if not HUNTER_FIXED_AVAILABLE and not HUNTER_AVAILABLE:
            print("\nNote: Hunter experts not available, testing only 'simple'")
    else:
        expert_types = [args.expert]
    
    # Test each expert
    all_stats = []
    
    for expert_type in expert_types:
        try:
            expert, expert_name = create_expert(expert_type, debug=args.verbose)
            stats = test_expert(
                env, expert, expert_name, obs_indices, agent_id,
                args.episodes, args.render, args.verbose
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"\nERROR testing {expert_type}: {e}")
            continue
    
    # Final comparison (if multiple experts)
    if len(all_stats) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        print(f"\n{'Expert':<35} {'Win Rate':>10} {'Mean Rew':>10} {'Missiles':>10}")
        print("-" * 70)
        
        for stats in sorted(all_stats, key=lambda x: -x['blue_win_rate']):
            print(f"{stats['expert_name']:<35} {stats['blue_win_rate']:>9.1f}% "
                  f"{stats['mean_reward']:>10.1f} {stats['avg_missiles']:>10.1f}")
        
        # Winner
        best = max(all_stats, key=lambda x: x['blue_win_rate'])
        print(f"\n🏆 BEST: {best['expert_name']} with {best['blue_win_rate']:.1f}% win rate")
    
    elif len(all_stats) == 1:
        stats = all_stats[0]
        print("\n" + "=" * 60)
        if stats['blue_win_rate'] >= 70:
            print(f"✓ EXCELLENT: {stats['blue_win_rate']:.1f}% win rate - ready for alignment training!")
        elif stats['blue_win_rate'] >= 50:
            print(f"✓ GOOD: {stats['blue_win_rate']:.1f}% win rate - usable for alignment")
        elif stats['blue_win_rate'] >= 30:
            print(f"⚠ MARGINAL: {stats['blue_win_rate']:.1f}% win rate - consider tuning")
        else:
            print(f"✗ POOR: {stats['blue_win_rate']:.1f}% win rate - needs improvement")
        print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
