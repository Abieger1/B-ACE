#!/usr/bin/env python3
"""
test_expert_standalone.py

Test aggressive expert in B-ACE environment to verify it achieves kills.
Run this BEFORE using the expert for alignment training.

Usage:
    python test_expert_standalone.py --config configs/Scen_1_config.json --episodes 20
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

from b_ace_py.utils import load_b_ace_config
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper

# Import expert
from simple_aggressive_expert import SimpleAggressiveExpert, get_default_obs_indices as get_simple_obs_indices


def resolve_env_path() -> str:
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


def extract_obs_indices(env) -> dict:
    """Extract observation indices from environment."""
    indices = get_simple_obs_indices()  # Start with defaults
    
    # Try to get from env
    if hasattr(env, "obs_map"):
        for agent_map in env.obs_map.values():
            if isinstance(agent_map, dict):
                indices.update(agent_map)
                break
    
    return indices


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
    
    while not done:
        # Get observation for controlled agent
        raw_obs = None
        if agent_id in obs_dict:
            raw_obs = obs_dict[agent_id]
        else:
            # Try to find by matching ID (handles int vs str mismatch)
            for k, v in obs_dict.items():
                if str(k) == str(agent_id):
                    raw_obs = v
                    break
            if raw_obs is None:
                raw_obs = list(obs_dict.values())[0]
        
        # Handle dict observations - may have nested 'obs' key
        if isinstance(raw_obs, dict):
            if 'obs' in raw_obs:
                inner = raw_obs['obs']
                # Could be another dict or array
                if isinstance(inner, dict) and 'obs' in inner:
                    obs = np.asarray(inner['obs'], dtype=np.float32)
                elif isinstance(inner, (list, np.ndarray)):
                    obs = np.asarray(inner, dtype=np.float32)
                else:
                    # Try to extract numeric values
                    obs = np.asarray(list(inner.values()), dtype=np.float32)
            else:
                # Try direct numeric extraction
                obs = np.asarray(list(raw_obs.values()), dtype=np.float32)
        else:
            obs = np.asarray(raw_obs, dtype=np.float32)
        
        # Get expert action
        action = expert.get_action(obs, obs_indices)
        
        # Track fire commands
        if action[3] > 0.5:
            missiles_fired += 1
        
        # Build action dict for all agents (expert controls blue, red is scripted)
        action_dict = {agent_id: action}
        
        # Step environment
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
        
        # Get reward (handle int/str key mismatch)
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
        if info.get("blue_kill", False) or info.get("enemy_killed", False):
            blue_kills += 1
        if info.get("red_kill", False) or info.get("blue_killed", False):
            red_kills += 1
        
        # Debug: print termination dicts on first few steps
        if verbose and steps <= 3:
            print(f"  [DEBUG] term_dict={term_dict}")
            print(f"  [DEBUG] trunc_dict={trunc_dict}")
        
        # Check termination - be more thorough about checking all keys
        done = False
        if isinstance(term_dict, dict):
            # Check all possible keys
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
        
        # Safety limit - episodes shouldn't exceed this
        MAX_EPISODE_STEPS = 15000
        if steps >= MAX_EPISODE_STEPS:
            if verbose:
                print(f"  [WARN] Max steps ({MAX_EPISODE_STEPS}) reached, forcing episode end")
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
    }


def main():
    parser = argparse.ArgumentParser(description="Test aggressive expert in B-ACE")
    parser.add_argument(
        "--config", 
        type=str, 
        default=(REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json").as_posix(),
        help="B-ACE config JSON (default: b_ace_py/SimpleExample_B_ACE_config.json)"
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of test episodes")
    parser.add_argument("--verbose", action="store_true", help="Print per-step info")
    parser.add_argument("--render", action="store_true", help="Render episodes (slower)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("AGGRESSIVE EXPERT TEST")
    print("=" * 60)
    
    # Load config
    config_path = Path(args.config).expanduser().resolve()
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_b_ace_config(config_path.as_posix())
    
    if config is None:
        # Try loading directly with json
        import json
        print(f"Warning: load_b_ace_config returned None, trying direct JSON load...")
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Set up env path
    env_cfg = config.setdefault("EnvConfig", {})
    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = resolve_env_path()
    
    # Configure agents
    agents_cfg = config.setdefault("AgentsConfig", {})
    agents_cfg.setdefault("blue_agents", {})["base_behavior"] = "external"
    agents_cfg.setdefault("red_agents", {})["base_behavior"] = "baseline1"
    
    # Rendering settings
    env_cfg["renderize"] = 1 if args.render else 0
    env_cfg["speed_up"] = 200 if args.render else 200
    
    print(f"Config: {config_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Red behavior: baseline1")
    print()
    
    # Create environment
    env = B_ACE_GodotPettingZooWrapper(device="cpu", **config)
    
    # Get observation indices
    obs_indices = extract_obs_indices(env)
    print(f"Observation indices found: {len(obs_indices)}")
    
    # Create expert (debug mode follows verbose flag)
    expert = SimpleAggressiveExpert(
        fire_threshold=0.50,
        aspect_limit_deg=30.0,
        debug=args.verbose
    )
    
    # Get default observation indices
    obs_indices = get_simple_obs_indices()
    
    # Determine agent ID - check possible_agents first, then agents
    agents = getattr(env, "possible_agents", None) or getattr(env, "agents", None)
    if agents:
        # Find blue agent (typically has lower ID or contains 'blue'/agent_0)
        agent_id = agents[0]
        for a in agents:
            if "blue" in str(a).lower() or str(a) == "agent_0" or (isinstance(a, int) and a < 200):
                agent_id = a
                break
    else:
        agent_id = "agent_0"
    print(f"Available agents: {agents}")
    print(f"Controlling agent: {agent_id}")
    print()
    
    # Run episodes
    results = []
    print("Running episodes...")
    print("-" * 60)
    
    for ep in range(args.episodes):
        start = time.time()
        result = run_episode(env, expert, obs_indices, agent_id, args.render, args.verbose)
        elapsed = time.time() - start
        
        results.append(result)
        
        outcome = "BLUE_WIN" if result["blue_kills"] > 0 else "RED_WIN" if result["red_kills"] > 0 else "DRAW"
        print(f"Ep {ep+1:3d}: {outcome:8s} | reward={result['reward']:7.1f} | "
              f"steps={result['steps']:4d} | missiles={result['missiles_fired']} | {elapsed:.1f}s")
    
    # Summary statistics
    print("-" * 60)
    print("\nSUMMARY")
    print("=" * 60)
    
    rewards = [r["reward"] for r in results]
    blue_wins = sum(1 for r in results if r["blue_kills"] > 0)
    red_wins = sum(1 for r in results if r["red_kills"] > 0)
    draws = args.episodes - blue_wins - red_wins
    total_missiles = sum(r["missiles_fired"] for r in results)
    
    print(f"Episodes:     {args.episodes}")
    print(f"Blue wins:    {blue_wins} ({100*blue_wins/args.episodes:.1f}%)")
    print(f"Red wins:     {red_wins} ({100*red_wins/args.episodes:.1f}%)")
    print(f"Draws:        {draws} ({100*draws/args.episodes:.1f}%)")
    print()
    print(f"Mean reward:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min reward:   {np.min(rewards):.2f}")
    print(f"Max reward:   {np.max(rewards):.2f}")
    print()
    print(f"Total missiles fired: {total_missiles}")
    print(f"Avg missiles/episode: {total_missiles/args.episodes:.1f}")
    print()
    
    # Expert internal stats
    expert_stats = expert.get_stats()
    print("Expert stats:")
    print(f"  Total steps:  {expert_stats['total_steps']}")
    print(f"  Fire count:   {expert_stats['fire_count']}")
    print(f"  Engage count: {expert_stats['engage_count']}")
    
    print("=" * 60)
    
    # Verdict
    if blue_wins >= args.episodes * 0.3:
        print("✓ Expert appears viable (>30% win rate)")
    else:
        print("⚠ Expert win rate is low - consider tuning thresholds")
    
    env.close()


if __name__ == "__main__":
    main()
