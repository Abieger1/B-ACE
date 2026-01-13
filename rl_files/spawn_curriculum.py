"""
spawn_curriculum.py

Curriculum learning system for gradually increasing spawn stochasticity in B-ACE.

This module provides:
- SpawnCurriculumConfig: Configuration dataclass for curriculum parameters
- SpawnCurriculumWrapper: Gymnasium wrapper that modifies spawn configs per-reset
- SpawnCurriculumCallback: SB3 callback that tracks progress and coordinates updates

The curriculum starts with deterministic (or low-variance) spawns and gradually
increases randomization as training progresses, allowing the agent to first learn
the core task before handling environmental variability.

Usage:
    from spawn_curriculum import (
        SpawnCurriculumConfig, 
        SpawnCurriculumWrapper,
        SpawnCurriculumCallback
    )
    
    # Define curriculum
    curriculum_config = SpawnCurriculumConfig(
        start_step=1_000_000,
        end_step=7_000_000,
        blue_heading_start=(0.0, 0.0),
        blue_heading_end=(-15.0, 15.0),
        blue_radius_start=0.0,
        blue_radius_end=5.0,
        hvaa_heading_start=(165.0, 165.0),
        hvaa_heading_end=(150.0, 180.0),
    )
    
    # Wrap environment
    env = SpawnCurriculumWrapper(base_env, curriculum_config)
    
    # Add callback to training
    curriculum_callback = SpawnCurriculumCallback(env, verbose=1)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Callable
import copy
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class SpawnCurriculumConfig:
    """
    Configuration for spawn stochasticity curriculum.
    
    All ranges are interpolated linearly from start_step to end_step.
    Before start_step, values stay at *_start. After end_step, values stay at *_end.
    
    Attributes:
        start_step: Timestep when curriculum begins (before this, use start values)
        end_step: Timestep when curriculum ends (after this, use end values)
        
        blue_heading_start: Initial heading range [min_deg, max_deg] for blue agent
        blue_heading_end: Final heading range [min_deg, max_deg] for blue agent
        blue_radius_start: Initial spawn disk radius (NM) for blue agent
        blue_radius_end: Final spawn disk radius (NM) for blue agent
        
        hvaa_heading_start: Initial heading range for HVAA
        hvaa_heading_end: Final heading range for HVAA
        hvaa_rect_expand_start: Initial rect expansion factor (0 = no expansion)
        hvaa_rect_expand_end: Final rect expansion factor
        
        red_offset_start: Initial position offset range for red agent (x, y, z)
        red_offset_end: Final position offset range for red agent
        
        verbose: Print curriculum updates
    """
    # Timing
    start_step: int = 1_000_000
    end_step: int = 7_000_000
    
    # Blue agent spawn curriculum
    blue_heading_start: Tuple[float, float] = (0.0, 0.0)
    blue_heading_end: Tuple[float, float] = (-15.0, 15.0)
    blue_radius_start: float = 0.0
    blue_radius_end: float = 5.0
    
    # HVAA spawn curriculum
    hvaa_heading_start: Tuple[float, float] = (165.0, 165.0)
    hvaa_heading_end: Tuple[float, float] = (150.0, 180.0)
    hvaa_rect_expand_start: float = 0.0  # Multiplier for rect expansion
    hvaa_rect_expand_end: float = 3.0
    
    # Red agent spawn curriculum (uses rnd_offset_range)
    red_offset_start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    red_offset_end: Tuple[float, float, float] = (5.0, 0.0, 5.0)
    
    verbose: bool = True
    
    def get_progress(self, timestep: int) -> float:
        """
        Get curriculum progress as fraction [0, 1].
        
        Args:
            timestep: Current training timestep
            
        Returns:
            Progress fraction (0 before start, 1 after end, linear between)
        """
        if timestep <= self.start_step:
            return 0.0
        if timestep >= self.end_step:
            return 1.0
        return (timestep - self.start_step) / (self.end_step - self.start_step)
    
    def interpolate(self, start_val, end_val, progress: float):
        """
        Linearly interpolate between start and end values.
        
        Handles scalars, tuples, and dicts.
        """
        if isinstance(start_val, tuple):
            return tuple(
                s + progress * (e - s) 
                for s, e in zip(start_val, end_val)
            )
        if isinstance(start_val, dict):
            return {
                k: self.interpolate(start_val[k], end_val[k], progress)
                for k in start_val
            }
        return start_val + progress * (end_val - start_val)
    
    def get_spawn_params(self, timestep: int) -> Dict[str, Any]:
        """
        Get spawn parameters for current timestep.
        
        Args:
            timestep: Current training timestep
            
        Returns:
            Dictionary with spawn parameters for blue, HVAA, and red agents
        """
        progress = self.get_progress(timestep)
        
        blue_heading = self.interpolate(
            self.blue_heading_start, self.blue_heading_end, progress
        )
        blue_radius = self.interpolate(
            self.blue_radius_start, self.blue_radius_end, progress
        )
        hvaa_heading = self.interpolate(
            self.hvaa_heading_start, self.hvaa_heading_end, progress
        )
        hvaa_rect_expand = self.interpolate(
            self.hvaa_rect_expand_start, self.hvaa_rect_expand_end, progress
        )
        red_offset = self.interpolate(
            self.red_offset_start, self.red_offset_end, progress
        )
        
        return {
            "progress": progress,
            "blue_heading_range": blue_heading,
            "blue_radius": blue_radius,
            "hvaa_heading_range": hvaa_heading,
            "hvaa_rect_expand": hvaa_rect_expand,
            "red_offset": red_offset,
        }


class SpawnCurriculumWrapper(gym.Wrapper):
    """
    Wrapper that modifies spawn configuration based on curriculum progress.
    
    This wrapper intercepts reset() calls and modifies the underlying
    environment's spawn configuration before each reset, implementing
    a curriculum that gradually increases spawn stochasticity.
    
    The wrapper stores a reference to the original B-ACE config and creates
    modified versions based on the current curriculum progress.
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: SpawnCurriculumConfig,
        bace_config: Dict[str, Any],
    ):
        """
        Initialize the curriculum wrapper.
        
        Args:
            env: The environment to wrap (should be SingleAgentBACEEnv or similar)
            config: SpawnCurriculumConfig with curriculum parameters
            bace_config: The original B-ACE configuration dictionary
        """
        super().__init__(env)
        self.curriculum_config = config
        self.original_bace_config = copy.deepcopy(bace_config)
        
        # Track global timesteps (updated by callback)
        self._global_timestep = 0
        self._episode_count = 0
        self._last_logged_progress = -1.0
        
        # Cache for current spawn params
        self._current_spawn_params = config.get_spawn_params(0)
        
        # Try to find the underlying B-ACE wrapper for direct config updates
        self._ma_env = self._find_ma_env()
        
        if config.verbose:
            print(f"\n{'='*60}")
            print("SPAWN CURRICULUM WRAPPER INITIALIZED")
            print(f"{'='*60}")
            print(f"  Start step: {config.start_step:,}")
            print(f"  End step: {config.end_step:,}")
            print(f"  Blue heading: {config.blue_heading_start} → {config.blue_heading_end}")
            print(f"  Blue radius: {config.blue_radius_start} → {config.blue_radius_end}")
            print(f"  HVAA heading: {config.hvaa_heading_start} → {config.hvaa_heading_end}")
            print(f"  HVAA rect expand: {config.hvaa_rect_expand_start} → {config.hvaa_rect_expand_end}")
            print(f"  Red offset: {config.red_offset_start} → {config.red_offset_end}")
            print(f"{'='*60}\n")
    
    def _find_ma_env(self):
        """Find the underlying B_ACE_GodotPettingZooWrapper."""
        env = self.env
        while env is not None:
            if hasattr(env, '_ma_env'):
                return env._ma_env
            if hasattr(env, 'env'):
                env = env.env
            else:
                break
        return None
    
    def set_global_timestep(self, timestep: int):
        """
        Set the current global timestep (called by callback).
        
        Args:
            timestep: Current training timestep
        """
        self._global_timestep = timestep
        self._current_spawn_params = self.curriculum_config.get_spawn_params(timestep)
        
        # Propagate to inner wrappers if they have this method
        if hasattr(self.env, 'set_global_timestep'):
            self.env.set_global_timestep(timestep)
        if hasattr(self.env, 'set_global_step'):
            self.env.set_global_step(timestep)
    
    def _build_spawn_config_update(self) -> Dict[str, Any]:
        """
        Build the spawn configuration update based on current curriculum state.
        
        Returns:
            Dictionary with updated spawn parameters for Godot
        """
        params = self._current_spawn_params
        
        # Get original configs as base
        env_cfg = self.original_bace_config.get("EnvConfig", {})
        agents_cfg = self.original_bace_config.get("AgentsConfig", {})
        
        update = {}
        
        # Blue spawn update
        if "BlueSpawn" in env_cfg:
            blue_spawn = copy.deepcopy(env_cfg["BlueSpawn"])
            blue_spawn["heading_deg_range"] = list(params["blue_heading_range"])
            blue_spawn["outer_radius"] = params["blue_radius"]
            update["BlueSpawn"] = blue_spawn
        
        # HVAA spawn update
        if "HVAASpawn" in env_cfg:
            hvaa_spawn = copy.deepcopy(env_cfg["HVAASpawn"])
            hvaa_spawn["heading_deg_range"] = list(params["hvaa_heading_range"])
            
            # Expand rectangle if using random_rect mode
            if hvaa_spawn.get("mode") == "random_rect" and "rect" in hvaa_spawn:
                original_rect = env_cfg["HVAASpawn"]["rect"]
                expand = params["hvaa_rect_expand"]
                
                # Calculate rect center and expand outward
                x_center = (original_rect["xmin"] + original_rect["xmax"]) / 2
                z_center = (original_rect["zmin"] + original_rect["zmax"]) / 2
                x_half = (original_rect["xmax"] - original_rect["xmin"]) / 2
                z_half = (original_rect["zmax"] - original_rect["zmin"]) / 2
                
                hvaa_spawn["rect"] = {
                    "xmin": x_center - x_half - expand,
                    "xmax": x_center + x_half + expand,
                    "zmin": z_center - z_half - expand,
                    "zmax": z_center + z_half + expand,
                }
            update["HVAASpawn"] = hvaa_spawn
        
        # Red offset update (uses AgentsConfig.red_agents.rnd_offset_range)
        red_offset = params["red_offset"]
        update["RedOffset"] = {
            "x": red_offset[0],
            "y": red_offset[1],
            "z": red_offset[2],
        }
        
        return update
    
    def _apply_spawn_config(self):
        """
        Apply the current spawn configuration to the environment.
        
        This sends updated spawn parameters to Godot via the underlying
        B_ACE_GodotPettingZooWrapper.
        """
        if self._ma_env is None:
            return
        
        spawn_update = self._build_spawn_config_update()
        
        # Method 1: Direct config update if available
        if hasattr(self._ma_env, 'update_spawn_config'):
            try:
                self._ma_env.update_spawn_config(spawn_update)
            except Exception as e:
                if self.curriculum_config.verbose:
                    print(f"[CURRICULUM] Config update via update_spawn_config failed: {e}")
        
        # Method 2: Store in env_config for next reset
        # This modifies the config that Godot reads on reset
        if hasattr(self._ma_env, 'env_config'):
            if "BlueSpawn" in spawn_update:
                self._ma_env.env_config["BlueSpawn"] = spawn_update["BlueSpawn"]
            if "HVAASpawn" in spawn_update:
                self._ma_env.env_config["HVAASpawn"] = spawn_update["HVAASpawn"]
        
        if hasattr(self._ma_env, 'agents_config'):
            if "RedOffset" in spawn_update:
                red_cfg = self._ma_env.agents_config.get("red_agents", {})
                red_cfg["rnd_offset_range"] = spawn_update["RedOffset"]
    
    def reset(self, **kwargs):
        """
        Reset the environment with curriculum-adjusted spawn config.
        """
        self._episode_count += 1
        
        # Apply curriculum config before reset
        self._apply_spawn_config()
        
        # Log progress periodically
        progress = self._current_spawn_params["progress"]
        if self.curriculum_config.verbose:
            # Log every 10% progress or every 100 episodes
            if (progress - self._last_logged_progress >= 0.1 or 
                self._episode_count % 100 == 0):
                self._last_logged_progress = progress
                params = self._current_spawn_params
                print(f"[CURRICULUM] ep={self._episode_count} t={self._global_timestep:,} "
                      f"progress={progress:.1%} "
                      f"blue_hdg={params['blue_heading_range']} "
                      f"blue_r={params['blue_radius']:.1f} "
                      f"hvaa_hdg={params['hvaa_heading_range']}")
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Pass through step to underlying environment."""
        return self.env.step(action)
    
    def get_curriculum_state(self) -> Dict[str, Any]:
        """
        Get current curriculum state for logging/debugging.
        
        Returns:
            Dictionary with current curriculum parameters and progress
        """
        return {
            "timestep": self._global_timestep,
            "episode": self._episode_count,
            **self._current_spawn_params,
        }


class SpawnCurriculumCallback(BaseCallback):
    """
    Stable Baselines3 callback that coordinates spawn curriculum.
    
    This callback tracks training progress and updates the curriculum wrapper
    with the current timestep, allowing the wrapper to adjust spawn parameters.
    
    Usage:
        callback = SpawnCurriculumCallback(curriculum_wrapper, verbose=1)
        model.learn(total_timesteps=10_000_000, callback=callback)
    """
    
    def __init__(
        self,
        curriculum_wrapper: SpawnCurriculumWrapper = None,
        update_freq: int = 1000,
        verbose: int = 0,
    ):
        """
        Initialize the curriculum callback.
        
        Args:
            curriculum_wrapper: The SpawnCurriculumWrapper to coordinate with
            update_freq: How often (in timesteps) to update the wrapper
            verbose: Verbosity level (0=silent, 1=progress updates)
        """
        super().__init__(verbose)
        self.curriculum_wrapper = curriculum_wrapper
        self.update_freq = update_freq
        self._last_update = 0
    
    def set_curriculum_wrapper(self, wrapper: SpawnCurriculumWrapper):
        """Set the curriculum wrapper after initialization."""
        self.curriculum_wrapper = wrapper
    
    def _find_curriculum_wrapper(self):
        """Try to find the curriculum wrapper in the training env stack."""
        if self.curriculum_wrapper is not None:
            return self.curriculum_wrapper
        
        # Search through VecEnv stack
        env = self.training_env
        while env is not None:
            if hasattr(env, 'envs'):
                # DummyVecEnv
                for base_env in env.envs:
                    wrapper = self._search_wrapper_chain(base_env)
                    if wrapper is not None:
                        return wrapper
            if hasattr(env, 'venv'):
                env = env.venv
            else:
                break
        return None
    
    def _search_wrapper_chain(self, env):
        """Search a single env's wrapper chain for SpawnCurriculumWrapper."""
        while env is not None:
            if isinstance(env, SpawnCurriculumWrapper):
                return env
            if hasattr(env, 'env'):
                env = env.env
            else:
                break
        return None
    
    def _on_training_start(self):
        """Called at the start of training."""
        if self.curriculum_wrapper is None:
            self.curriculum_wrapper = self._find_curriculum_wrapper()
        
        if self.curriculum_wrapper is not None:
            self.curriculum_wrapper.set_global_timestep(self.num_timesteps)
            if self.verbose > 0:
                print("[SpawnCurriculumCallback] Found curriculum wrapper, tracking enabled")
        else:
            if self.verbose > 0:
                print("[SpawnCurriculumCallback] WARNING: No curriculum wrapper found!")
    
    def _on_step(self) -> bool:
        """Called after each step."""
        if self.curriculum_wrapper is None:
            return True
        
        # Update wrapper with current timestep
        if self.num_timesteps - self._last_update >= self.update_freq:
            self.curriculum_wrapper.set_global_timestep(self.num_timesteps)
            self._last_update = self.num_timesteps
            
            # Log to tensorboard if available
            if self.logger is not None:
                state = self.curriculum_wrapper.get_curriculum_state()
                self.logger.record("curriculum/progress", state["progress"])
                self.logger.record("curriculum/blue_radius", state["blue_radius"])
                self.logger.record("curriculum/hvaa_rect_expand", state["hvaa_rect_expand"])
        
        return True
    
    def _on_rollout_end(self):
        """Called at the end of each rollout."""
        if self.curriculum_wrapper is not None:
            self.curriculum_wrapper.set_global_timestep(self.num_timesteps)


def create_curriculum_env(
    base_env_fn: Callable[[], gym.Env],
    bace_config: Dict[str, Any],
    curriculum_config: SpawnCurriculumConfig,
) -> Tuple[SpawnCurriculumWrapper, SpawnCurriculumCallback]:
    """
    Factory function to create a curriculum-wrapped environment.
    
    Args:
        base_env_fn: Function that creates the base environment
        bace_config: B-ACE configuration dictionary
        curriculum_config: Curriculum configuration
        
    Returns:
        Tuple of (wrapped_env, callback)
    """
    base_env = base_env_fn()
    wrapped_env = SpawnCurriculumWrapper(base_env, curriculum_config, bace_config)
    callback = SpawnCurriculumCallback(wrapped_env, verbose=1)
    return wrapped_env, callback


# ============================================================================
# Example curriculum configurations for common scenarios
# ============================================================================

def get_default_curriculum_config() -> SpawnCurriculumConfig:
    """Get a sensible default curriculum for 10M timestep training."""
    return SpawnCurriculumConfig(
        start_step=1_000_000,      # Hold constant for first 1M steps
        end_step=7_000_000,        # Reach full randomization by 7M
        blue_heading_start=(0.0, 0.0),
        blue_heading_end=(-15.0, 15.0),
        blue_radius_start=0.0,
        blue_radius_end=5.0,
        hvaa_heading_start=(165.0, 165.0),
        hvaa_heading_end=(150.0, 180.0),
        hvaa_rect_expand_start=0.0,
        hvaa_rect_expand_end=3.0,
        red_offset_start=(0.0, 0.0, 0.0),
        red_offset_end=(5.0, 0.0, 5.0),
        verbose=True,
    )


def get_conservative_curriculum_config() -> SpawnCurriculumConfig:
    """More conservative curriculum with longer ramp and smaller final variance."""
    return SpawnCurriculumConfig(
        start_step=1_500_000,
        end_step=8_000_000,
        blue_heading_start=(0.0, 0.0),
        blue_heading_end=(-10.0, 10.0),
        blue_radius_start=0.0,
        blue_radius_end=3.0,
        hvaa_heading_start=(165.0, 165.0),
        hvaa_heading_end=(155.0, 175.0),
        hvaa_rect_expand_start=0.0,
        hvaa_rect_expand_end=2.0,
        red_offset_start=(0.0, 0.0, 0.0),
        red_offset_end=(3.0, 0.0, 3.0),
        verbose=True,
    )


def get_aggressive_curriculum_config() -> SpawnCurriculumConfig:
    """More aggressive curriculum with faster ramp and larger final variance."""
    return SpawnCurriculumConfig(
        start_step=500_000,
        end_step=5_000_000,
        blue_heading_start=(0.0, 0.0),
        blue_heading_end=(-20.0, 20.0),
        blue_radius_start=0.0,
        blue_radius_end=8.0,
        hvaa_heading_start=(165.0, 165.0),
        hvaa_heading_end=(140.0, 190.0),
        hvaa_rect_expand_start=0.0,
        hvaa_rect_expand_end=5.0,
        red_offset_start=(0.0, 0.0, 0.0),
        red_offset_end=(8.0, 0.0, 8.0),
        verbose=True,
    )


if __name__ == "__main__":
    # Test curriculum interpolation
    config = get_default_curriculum_config()
    
    print("Testing curriculum interpolation:")
    for timestep in [0, 500_000, 1_000_000, 4_000_000, 7_000_000, 10_000_000]:
        params = config.get_spawn_params(timestep)
        print(f"  t={timestep:>10,}: progress={params['progress']:.2f}, "
              f"blue_hdg={params['blue_heading_range']}, "
              f"blue_r={params['blue_radius']:.2f}")
