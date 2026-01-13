#!/usr/bin/env python3
"""
train_hierarchical_phase1.py

Phase 1 of hierarchical reinforcement learning: Train low-level option-specific policies.

Each tactical option gets its own specialized policy trained with option-specific
reward shaping. These policies learn focused behaviors:
- DEFEND_HVAA: Stay near HVAA, intercept threats
- INTERCEPT_ENEMY: Aggressively engage enemies
- EVADE_MISSILE: Execute defensive maneuvers
- OFFENSIVE_POSITIONING: Maneuver for WEZ advantage
- DEFENSIVE_POSITIONING: Escape from enemy BEZ

Usage:
    # Train all options with default settings
    python train_hierarchical_phase1.py --seed 42
    
    # Train specific option only
    python train_hierarchical_phase1.py --option DEFEND_HVAA --seed 42
    
    # Custom timesteps
    python train_hierarchical_phase1.py --timesteps 2000000 --seed 42
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

############################################
# Make sure we can import b_ace_py
############################################
THIS_FILE = Path(__file__).resolve()

# Find the B-ACE repo root by looking for b_ace_py directory
current_path = THIS_FILE.parent
while current_path != current_path.parent:  # Stop at filesystem root
    if (current_path / "b_ace_py").exists():
        REPO_ROOT = current_path
        break
    current_path = current_path.parent
else:
    # Fallback: assume script is two levels deep from root
    REPO_ROOT = THIS_FILE.parent.parent

if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

print(f"Using REPO_ROOT: {REPO_ROOT}")

from b_ace_py.utils import load_b_ace_config
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
from b_ace_py.enriched_observation_wrapper import EnrichedObservationWrapper
from b_ace_py.reward_shaping_wrapper import RewardShapingWrapper, RewardShapingConfig
from option_specific_reward_wrapper import OptionSpecificRewardWrapper

# Import hierarchical learning modules
from tactical_options import (
    TacticalOption, create_tactical_options, print_option_summary
)
from expert_action_injection import (
    ExpertActionInjector, ExpertActionInjectionWrapper, create_injector_for_option
)


class SingleAgentBACEEnv(gym.Env):
    """
    SB3-compatible single-agent wrapper around the multi-agent B_ACE_GodotPettingZooWrapper.

    - Controls a single blue agent (default: "agent_0")
    - Exposes a vector observation space and continuous Box action space (size 4)
    - Internally runs the full B-ACE multi-agent sim
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        bace_config: Dict[str, Any],
        controlled_agent_id: str = "agent_0",
        max_episode_steps: int = 6000,
    ):
        super().__init__()
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        # Underlying multi-agent PettingZoo-style env
        self._ma_env = B_ACE_GodotPettingZooWrapper(device="cpu", **bace_config)
        self._last_obs_dict: Optional[Dict[str, Any]] = None

        # Initial reset so we can infer observation shape
        obs_dict, info_dict = self._ma_env.reset()
        self._last_obs_dict = obs_dict

        # Figure out which agent we control
        agents = getattr(self._ma_env, "agents", [])
        if agents:
            if controlled_agent_id in agents:
                self.controlled_agent_id = controlled_agent_id
            else:
                print(
                    f"WARNING: controlled_agent_id '{controlled_agent_id}' not in env.agents={agents}, "
                    f"defaulting to '{agents[0]}'"
                )
                self.controlled_agent_id = agents[0]
        else:
            self.controlled_agent_id = controlled_agent_id

        # ----- Build observation_space from a real observation -----
        if self.controlled_agent_id in obs_dict:
            raw_obs = obs_dict[self.controlled_agent_id]
        else:
            raw_obs = next(iter(obs_dict.values()))

        obs_vec = self._to_array(raw_obs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_vec.shape,
            dtype=np.float32,
        )

        # ----- Build action_space: we KNOW it's 4-dim continuous from B-ACE -----
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    # ---------- Helper to convert arbitrary obs structures to a numeric array ----------

    def _to_array(self, obj) -> np.ndarray:
        """
        Recursively extract a numeric vector from nested dict/list/ndarray structures.

        Preference order:
        - If dict has 'obs' key, use that.
        - Otherwise, search values for something convertible to float array.
        """
        # Already an ndarray
        if isinstance(obj, np.ndarray):
            return obj.astype(np.float32)

        # Simple Python container of numbers
        if isinstance(obj, (list, tuple)):
            return np.asarray(obj, dtype=np.float32)

        # Scalar number
        if isinstance(obj, (float, int)):
            return np.asarray([obj], dtype=np.float32)

        # Nested dict: dig into it
        if isinstance(obj, dict):
            # Common pattern: {'obs': ...}
            if "obs" in obj:
                return self._to_array(obj["obs"])

            # Otherwise search values
            for v in obj.values():
                try:
                    arr = self._to_array(v)
                    # If we got here without exception and it's numeric, take it
                    if arr.dtype.kind in ("f", "i"):
                        return arr
                except (TypeError, ValueError):
                    continue

            raise RuntimeError(
                f"Could not extract numeric observation from dict with keys: {list(obj.keys())}"
            )

        # Anything else is unsupported
        raise RuntimeError(
            f"Unsupported observation type {type(obj)} when trying to build observation array"
        )

    # ---------- Helper to extract obs each step/reset ----------

    def _extract_single_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        if self.controlled_agent_id in obs_dict:
            raw_obs = obs_dict[self.controlled_agent_id]
        else:
            raw_obs = next(iter(obs_dict.values()))

        return self._to_array(raw_obs)

    # ---------- Gymnasium API ----------

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        obs_dict, info_dict = self._ma_env.reset(**kwargs)
        self._last_obs_dict = obs_dict

        obs = self._extract_single_obs(obs_dict)
        info = info_dict.get(self.controlled_agent_id, {}) if isinstance(info_dict, dict) else {}
        return obs, info

    def step(self, action):
        self._elapsed_steps += 1

        # Apply same action to all agents (only blue external actually uses it)
        if hasattr(self._ma_env, "agents"):
            action_dict = {agent: action for agent in self._ma_env.agents}
        else:
            action_dict = {self.controlled_agent_id: action}

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._ma_env.step(action_dict)
        self._last_obs_dict = obs_dict

        # Reward
        if isinstance(rew_dict, dict):
            reward = float(rew_dict.get(self.controlled_agent_id, 0.0))
        else:
            reward = float(rew_dict)

        # Termination + truncation
        if isinstance(term_dict, dict):
            terminated = bool(term_dict.get(self.controlled_agent_id, False))
        else:
            terminated = bool(term_dict)

        if isinstance(trunc_dict, dict):
            truncated = bool(trunc_dict.get(self.controlled_agent_id, False))
        else:
            truncated = bool(trunc_dict)

        # Enforce internal max_episode_steps as truncation
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        info = info_dict.get(self.controlled_agent_id, {}) if isinstance(info_dict, dict) else {}
        obs = self._extract_single_obs(obs_dict)

        return obs, reward, terminated, truncated, info

    def render(self):
        # Rendering handled by Godot
        pass

    def close(self):
        if hasattr(self._ma_env, "close"):
            self._ma_env.close()


def create_bace_env(
    config_path: str = None,
    render: bool = False,
    seed: int = 0,
    port: int = 11008,
) -> gym.Env:
    """
    Create B-ACE environment for hierarchical training.
    Mirrors the setup from train_sb3_bace_with_eval_checkpoints_FIXED.py,
    but returns a SingleAgentBACEEnv.
    """
    # Load + sanitize B-ACE config
    if config_path is None:
        config_path = REPO_ROOT / "b_ace_py" / "configs" / "train_config.json"

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    bace_config = load_b_ace_config(config_path.as_posix())
    if bace_config is None:
        raise RuntimeError(f"Failed to load config file at {config_path}")

    # ---- EnvConfig ----
    env_cfg = bace_config.setdefault("EnvConfig", {})

    # Resolve env_path if missing
    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = _resolve_env_path()

    env_path_value = Path(env_cfg["env_path"]).expanduser()
    if not env_path_value.is_absolute():
        env_path_value = (REPO_ROOT / env_path_value).resolve()
    else:
        env_path_value = env_path_value.resolve()

    if not env_path_value.exists():
        fallback_path = Path(_resolve_env_path()).resolve()
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"[B-ACE ERROR] Tried {env_path_value}, and fallback {fallback_path}, "
                "but neither exists. You probably need to export the Godot binary."
            )
        print(
            f"Warning: Godot binary not found at {env_path_value}. "
            f"Falling back to {fallback_path}."
        )
        env_path_value = fallback_path

    env_cfg["env_path"] = env_path_value.as_posix()

    # AgentsConfig defaults
    agents_cfg = bace_config.setdefault("AgentsConfig", {})
    blue_cfg = agents_cfg.setdefault("blue_agents", {})
    red_cfg = agents_cfg.setdefault("red_agents", {})
    blue_cfg.setdefault("base_behavior", "external")
    red_cfg.setdefault("base_behavior", "baseline1")

    # Training runtime config
    env_cfg["renderize"] = 1 if render else 0
    env_cfg["speed_up"] = 1000
    env_cfg["port"] = port
    env_cfg["seed"] = seed

    print("Creating B-ACE environment:")
    print(f"  Godot executable: {env_cfg['env_path']}")
    print(f"  Port: {port}")
    print(f"  Seed: {seed}")
    print(f"  Render: {render}")

    # Return SB3-compatible single-agent env
    env = SingleAgentBACEEnv(
        bace_config=bace_config,
        controlled_agent_id="agent_0",
        max_episode_steps=6000,  # adjust if you use a different horizon
    )
    return env


def _resolve_env_path() -> str:
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
            # just return the first that exists
            return binary.as_posix()

    searched = "\n  - ".join(c.as_posix() for c in candidates)
    raise FileNotFoundError(
        "Could not find a suitable Godot binary. Checked:\n"
        f"  - {searched}"
    )



def setup_option_training_env(
    option_id: TacticalOption,
    option: Any,
    config_path: str = None,
    seed: int = 0,
    port: int = 11008,
    use_enriched_obs: bool = True,
    use_base_reward_shaping: bool = True,
    option_shaping_strength: float = 1.0,
    use_expert_injection: bool = False,
    expert_injection_prob: float = 0.01,
    expert_use_curriculum: bool = True,
    expert_state_dependent: bool = True,
) -> gym.Env:
    """
    Set up environment for training a specific option.
    
    IMPORTANT: This creates a LAYERED wrapper architecture:
    1. EnrichedObservationWrapper: Add theoretical features (BEZ, DMC, etc.)
    2. RewardShapingWrapper: Your existing comprehensive reward shaping
    3. OptionSpecificRewardWrapper: Lightweight option-specific guidance
    4. ExpertActionInjectionWrapper: (OPTIONAL) Policy shaping via expert actions
    
    Args:
        option_id: Which option to train
        option: Option object
        config_path: Path to B-ACE config
        seed: Random seed
        port: Godot port
        use_enriched_obs: Whether to use enriched observations
        use_base_reward_shaping: Whether to use your RewardShapingWrapper
        option_shaping_strength: Strength of option-specific shaping (default: 1.0)
        use_expert_injection: Whether to inject expert actions during training
        expert_injection_prob: Base probability for expert injection
        expert_use_curriculum: Decay injection probability over time
        expert_state_dependent: Boost injection in critical states
        
    Returns:
        Fully wrapped environment ready for training
    """
    # Base single-agent env (already SB3-compatible)
    env = create_bace_env(config_path, render=False, seed=seed, port=port)

    # Get observation labels for feature extraction from underlying MA env
    obs_labels = {}
    ma_env = getattr(env, "_ma_env", None)
    if ma_env is not None and hasattr(ma_env, "obs_map") and len(ma_env.agents) > 0:
        obs_labels = ma_env.obs_map.get(ma_env.agents[0], {})
    
    # LAYER 1: Add enriched observations if enabled
    if use_enriched_obs:
        feature_config = {
            'enable_apollonius': True,
            'enable_bez': True,
            'enable_dmc': True,
            'enable_offense_wez': True,
            'enable_offense_ttc': True,
            'pursuer_speed': 1.0,
            'evader_speed': 0.8,
            'normalize_distance': 1000.0,
        }
        
        env = EnrichedObservationWrapper(
            env,
            obs_labels=obs_labels,
            feature_config=feature_config,
            debug=False
        )
        print(f"  ✓ Added EnrichedObservationWrapper (31-dim observations)")
    
    # LAYER 2: Add your comprehensive reward shaping
    if use_base_reward_shaping:
        # Create a balanced reward config (you can customize this)
        reward_config = RewardShapingConfig(
            # BEZ and DMC shaping
            enable_bez_shaping=True,
            bez_weight=0.00025,
            enable_dmc_shaping=True,
            dmc_weight=0.001,
            
            # HVAA escort shaping
            enable_hvaa_escort_shaping=True,
            hvaa_escort_weight=0.001,
            
            # Offensive shaping
            enable_offense_shaping=True,
            wez_dom_weight=0.02,
            offense_ttc_weight=0.01,
            
            # Missile launch shaping
            enable_missile_launch_shaping=True,
            
            # Disable some less critical components for simplicity
            enable_heading_shaping=False,
            enable_distance_shaping=False,
            enable_ttc_shaping=False,
            enable_multithreat_shaping=False,
            enable_speed_shaping=False,
            
            # Enable tracking for analysis
            track_reward_components=True,
            track_episode_stats=True,
            enable_logging=True,
            log_frequency=50000,
        )
        
        env = RewardShapingWrapper(
            env,
            config=reward_config,
            obs_labels=obs_labels,
            debug=False
        )
        print(f"  ✓ Added RewardShapingWrapper (your comprehensive reward shaping)")
    
    # LAYER 3: Add lightweight option-specific shaping
    env = OptionSpecificRewardWrapper(
        env,
        option_id=option_id,
        option=option,
        obs_labels=obs_labels,
        enriched_obs=use_enriched_obs,
        shaping_strength=option_shaping_strength
    )
    print(f"  ✓ Added OptionSpecificRewardWrapper (option-focused guidance)")
    
    # LAYER 4 (OPTIONAL): Add expert action injection for policy shaping
    if use_expert_injection:
        # Create injector for this option
        injector = create_injector_for_option(
            option_id=option_id,
            injection_prob=expert_injection_prob,
            use_curriculum=expert_use_curriculum,
            state_dependent=expert_state_dependent
        )
        
        env = ExpertActionInjectionWrapper(
            env,
            injector=injector,
            obs_labels=obs_labels,
            enriched_obs=use_enriched_obs,
            log_injections=False  # Set True for debugging
        )
        
        print(f"  ✓ Added ExpertActionInjectionWrapper (policy shaping)")
        print(f"    - Initial injection rate: {injector.get_injection_probability()*100:.2f}%")
        print(f"    - Curriculum: {expert_use_curriculum}")
        print(f"    - State-dependent boost: {expert_state_dependent}")
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    return env


def train_option_policy(option_id: TacticalOption,
                       option: Any,
                       total_timesteps: int = 1_000_000,
                       seed: int = 42,
                       log_dir: Path = None,
                       save_freq: int = 100_000,
                       config_path: str = None,
                       port: int = 11008,
                       use_enriched_obs: bool = True,
                       use_base_reward_shaping: bool = True,
                       option_shaping_strength: float = 1.0,
                       **ppo_kwargs) -> PPO:
    """
    Train a low-level policy for a specific option.
    
    Args:
        option_id: Which option to train
        option: Option object
        total_timesteps: Training timesteps
        seed: Random seed
        log_dir: Directory for logs and checkpoints
        save_freq: Save checkpoint every N steps
        config_path: Path to B-ACE config
        port: Godot port
        use_enriched_obs: Use enriched observations
        use_base_reward_shaping: Use your comprehensive RewardShapingWrapper
        option_shaping_strength: Strength multiplier for option-specific shaping
        **ppo_kwargs: Additional PPO arguments
        
    Returns:
        Trained PPO model
    """
    print(f"\n{'='*70}")
    print(f"TRAINING LOW-LEVEL POLICY: {option.name}")
    print(f"{'='*70}")
    print(f"Option ID: {option_id}")
    print(f"Description: {option.description}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Seed: {seed}")
    print(f"Enriched Observations: {use_enriched_obs}")
    print(f"Base Reward Shaping: {use_base_reward_shaping}")
    print(f"Option Shaping Strength: {option_shaping_strength}")
    print(f"{'='*70}\n")
    
    print("Wrapper Stack:")
    
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs") / "hierarchical" / "phase1" / f"option_{option_id}"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up environment
    def make_env():
        return setup_option_training_env(
            option_id=option_id,
            option=option,
            config_path=config_path,
            seed=seed,
            port=port,
            use_enriched_obs=use_enriched_obs,
            use_base_reward_shaping=use_base_reward_shaping,
            option_shaping_strength=option_shaping_strength
        )
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    

    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    # Set up logger
    logger = configure(str(log_dir / "logs"), ["stdout", "csv", "tensorboard"])
    
    # Default PPO hyperparameters optimized for B-ACE
    default_ppo_kwargs = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": str(log_dir / "tensorboard"),
        "seed": seed,
    }
    
    # Update with any user-provided kwargs
    default_ppo_kwargs.update(ppo_kwargs)
    
    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(env=env, **default_ppo_kwargs)
    model.set_logger(logger)
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix=f"option_{option_id}_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Train the model
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Average: {training_time/total_timesteps*1000:.3f} ms/step")
    
    # Save final model
    final_model_path = log_dir / f"option_{option_id}_final"
    model.save(str(final_model_path))
    env.save(str(log_dir / "vec_normalize.pkl"))
    
    print(f"\nModel saved to: {final_model_path}")
    
    # Save training metadata
    metadata = {
        "option_id": int(option_id),
        "option_name": option.name,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "training_time_seconds": training_time,
        "enriched_observations": use_enriched_obs,
        "base_reward_shaping": use_base_reward_shaping,
        "option_shaping_strength": option_shaping_strength,
        "observation_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "ppo_kwargs": default_ppo_kwargs,
    }
    
    with open(log_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return model
  


def train_all_options(timesteps_per_option: int = 1_000_000,
                     seed: int = 42,
                     base_log_dir: Path = None,
                     config_path: str = None,
                     base_port: int = 11008,
                     use_enriched_obs: bool = True,
                     use_base_reward_shaping: bool = True,
                     option_shaping_strength: float = 1.0,
                     **ppo_kwargs):
    """
    Train all low-level option policies sequentially.
    
    Args:
        timesteps_per_option: Timesteps to train each option
        seed: Random seed
        base_log_dir: Base directory for all logs
        config_path: Path to B-ACE config
        base_port: Starting port (each option gets port+1)
        use_enriched_obs: Use enriched observations
        use_base_reward_shaping: Use your comprehensive RewardShapingWrapper
        option_shaping_strength: Strength of option-specific shaping
        **ppo_kwargs: PPO hyperparameters
    """
    print("\n" + "="*70)
    print("HIERARCHICAL RL - PHASE 1: TRAINING LOW-LEVEL POLICIES")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Enriched Observations: {use_enriched_obs}")
    print(f"  - Base Reward Shaping: {use_base_reward_shaping}")
    print(f"  - Option Shaping Strength: {option_shaping_strength}")
    print(f"  - Timesteps per Option: {timesteps_per_option:,}")
    
    # Create tactical options
    options = create_tactical_options()
    print_option_summary(options)
    
    # Set base log directory
    if base_log_dir is None:
        base_log_dir = Path("logs") / "hierarchical" / "phase1"
    base_log_dir = Path(base_log_dir)
    
    # Train each option
    trained_models = {}
    
    for i, (opt_id, option) in enumerate(options.items()):
        # Use different port for each option to avoid conflicts
        port = base_port + i
        
        # Create option-specific log directory
        option_log_dir = base_log_dir / f"option_{opt_id}_{option.name.replace(' ', '_')}"
        
        # Train the option
        model = train_option_policy(
            option_id=opt_id,
            option=option,
            total_timesteps=timesteps_per_option,
            seed=seed,
            log_dir=option_log_dir,
            config_path=config_path,
            port=port,
            use_enriched_obs=use_enriched_obs,
            use_base_reward_shaping=use_base_reward_shaping,
            option_shaping_strength=option_shaping_strength,
            **ppo_kwargs
        )
        
        trained_models[opt_id] = {
            'model': model,
            'path': option_log_dir / f"option_{opt_id}_final.zip",
            'option': option
        }
        
        print(f"\n✓ Completed training for {option.name}")
        print(f"  Model saved at: {trained_models[opt_id]['path']}")
    
    # Save summary
    summary = {
        "phase": 1,
        "description": "Low-level option-specific policies",
        "timesteps_per_option": timesteps_per_option,
        "seed": seed,
        "enriched_observations": use_enriched_obs,
        "base_reward_shaping": use_base_reward_shaping,
        "option_shaping_strength": option_shaping_strength,
        "trained_options": {
            int(opt_id): {
                "name": info['option'].name,
                "model_path": str(info['path']),
                "description": info['option'].description,
            }
            for opt_id, info in trained_models.items()
        }
    }
    
    summary_path = base_log_dir / "phase1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("PHASE 1 TRAINING COMPLETE!")
    print("="*70)
    print(f"Summary saved to: {summary_path}")
    print(f"\nTrained {len(trained_models)} option policies:")
    for opt_id, info in trained_models.items():
        print(f"  - {info['option'].name} ({opt_id})")
    print("\nNext step: Evaluate individual option policies")
    print("Then proceed to Phase 2: Train high-level option selection policy")
    print("="*70 + "\n")
    
    return trained_models


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Train low-level option-specific policies"
    )
    
    # Training options
    parser.add_argument("--option", type=str, default=None,
                       choices=["DEFEND_HVAA", "INTERCEPT_ENEMY", "EVADE_MISSILE",
                               "OFFENSIVE_POSITIONING", "DEFENSIVE_POSITIONING", "ALL"],
                       help="Which option to train (default: ALL)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Timesteps per option (default: 1M)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to B-ACE config file")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Base log directory")
    parser.add_argument("--port", type=int, default=11008,
                       help="Starting Godot port (default: 11008)")
    
    # Observation options
    parser.add_argument("--no-enriched-obs", action="store_true",
                       help="Disable enriched observations (use base 22-dim obs)")
    parser.add_argument("--no-base-shaping", action="store_true",
                       help="Disable your comprehensive RewardShapingWrapper (for ablation)")
    parser.add_argument("--option-shaping-strength", type=float, default=1.0,
                       help="Multiplier for option-specific shaping (default: 1.0)")
    
    # Expert action injection (policy shaping)
    parser.add_argument("--expert-injection", action="store_true",
                       help="Enable expert action injection for policy shaping")
    parser.add_argument("--injection-prob", type=float, default=0.01,
                       help="Base expert injection probability (default: 0.01 = 1%%)")
    parser.add_argument("--no-injection-curriculum", action="store_true",
                       help="Disable curriculum (keep injection rate constant)")
    parser.add_argument("--no-injection-state-boost", action="store_true",
                       help="Disable state-dependent injection boost")
    
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    
    args = parser.parse_args()
    
    # PPO kwargs
    ppo_kwargs = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "ent_coef": args.ent_coef,
    }
    
    use_enriched_obs = not args.no_enriched_obs
    use_base_shaping = not args.no_base_shaping
    
    # Train specific option or all
    if args.option is None or args.option == "ALL":
        # Train all options
        train_all_options(
            timesteps_per_option=args.timesteps,
            seed=args.seed,
            base_log_dir=args.log_dir,
            config_path=args.config,
            base_port=args.port,
            use_enriched_obs=use_enriched_obs,
            use_base_reward_shaping=use_base_shaping,
            option_shaping_strength=args.option_shaping_strength,
            **ppo_kwargs
        )
    else:
        # Train single option
        options = create_tactical_options()
        option_id = TacticalOption[args.option]
        option = options[option_id]
        
        train_option_policy(
            option_id=option_id,
            option=option,
            total_timesteps=args.timesteps,
            seed=args.seed,
            log_dir=args.log_dir,
            config_path=args.config,
            port=args.port,
            use_enriched_obs=use_enriched_obs,
            use_base_reward_shaping=use_base_shaping,
            option_shaping_strength=args.option_shaping_strength,
            **ppo_kwargs
        )


if __name__ == "__main__":
    main()