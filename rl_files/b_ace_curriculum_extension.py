"""
b_ace_curriculum_extension.py

Extension to B_ACE_GodotPettingZooWrapper that adds support for
dynamically updating spawn configurations during training.

This module provides:
- update_spawn_config(): Method to send updated spawn configs to Godot
- Proper message handling for curriculum updates

Usage:
    # Option 1: Monkey-patch the existing wrapper
    from b_ace_curriculum_extension import patch_bace_wrapper
    patch_bace_wrapper(ma_env)  # ma_env is B_ACE_GodotPettingZooWrapper instance
    
    # Option 2: Use the mixin directly
    # (requires modifying your wrapper imports)
"""

import json
from typing import Dict, Any, Optional


def update_spawn_config(self, spawn_updates: Dict[str, Any]) -> bool:
    """
    Send updated spawn configuration to Godot.
    
    This method modifies the spawn parameters that Godot will use
    on the next episode reset. It does NOT restart the current episode.
    
    Args:
        spawn_updates: Dictionary with spawn parameter updates:
            - BlueSpawn: dict with heading_deg_range, outer_radius, etc.
            - HVAASpawn: dict with heading_deg_range, rect, etc.
            - RedOffset: dict with x, y, z offset ranges
            
    Returns:
        True if successful, False otherwise
    
    Example:
        spawn_updates = {
            "BlueSpawn": {
                "heading_deg_range": [-10.0, 10.0],
                "outer_radius": 3.0
            },
            "HVAASpawn": {
                "heading_deg_range": [150.0, 180.0]
            }
        }
        env._ma_env.update_spawn_config(spawn_updates)
    """
    # Also update the local config cache
    if hasattr(self, 'env_config'):
        if "BlueSpawn" in spawn_updates:
            if "BlueSpawn" not in self.env_config:
                self.env_config["BlueSpawn"] = {}
            self.env_config["BlueSpawn"].update(spawn_updates["BlueSpawn"])
        
        if "HVAASpawn" in spawn_updates:
            if "HVAASpawn" not in self.env_config:
                self.env_config["HVAASpawn"] = {}
            self.env_config["HVAASpawn"].update(spawn_updates["HVAASpawn"])
    
    if hasattr(self, 'agents_config') and "RedOffset" in spawn_updates:
        if "red_agents" in self.agents_config:
            self.agents_config["red_agents"]["rnd_offset_range"] = spawn_updates["RedOffset"]
    
    # Try to send to Godot (fire-and-forget to avoid message sync issues)
    try:
        message = {
            "type": "update_spawn_config",
            "spawn_updates": spawn_updates
        }
        
        if hasattr(self, '_send_as_json') and hasattr(self, 'connection'):
            self._send_as_json(message)
            return True
        else:
            # Fallback: just update local config (will be used on next reset)
            return True
            
    except Exception as e:
        print(f"[B-ACE] Warning: Could not send spawn config to Godot: {e}")
        return False


def patch_bace_wrapper(wrapper) -> bool:
    """
    Patch an existing B_ACE_GodotPettingZooWrapper instance with
    the update_spawn_config method.
    
    Args:
        wrapper: B_ACE_GodotPettingZooWrapper instance
        
    Returns:
        True if patching was successful
    """
    import types
    
    if wrapper is None:
        return False
    
    # Check if this looks like a B-ACE wrapper
    if not (hasattr(wrapper, 'env_config') or hasattr(wrapper, 'connection')):
        print("[patch_bace_wrapper] Warning: wrapper doesn't look like B_ACE_GodotPettingZooWrapper")
        return False
    
    # Add the method
    wrapper.update_spawn_config = types.MethodType(update_spawn_config, wrapper)
    
    return True


def patch_wrapper_chain(env) -> bool:
    """
    Find and patch the B_ACE_GodotPettingZooWrapper in an environment chain.
    
    Args:
        env: Any environment (will traverse wrapper chain)
        
    Returns:
        True if wrapper was found and patched
    """
    current = env
    while current is not None:
        # Check if this is the multi-agent wrapper
        if hasattr(current, '_ma_env'):
            ma_env = current._ma_env
            if patch_bace_wrapper(ma_env):
                print("[patch_wrapper_chain] Successfully patched _ma_env")
                return True
        
        # Check if this is the B-ACE wrapper itself
        if hasattr(current, 'env_config') and hasattr(current, 'agents_config'):
            if patch_bace_wrapper(current):
                print("[patch_wrapper_chain] Successfully patched wrapper directly")
                return True
        
        # Move to next wrapper in chain
        if hasattr(current, 'env'):
            current = current.env
        elif hasattr(current, 'venv'):
            current = current.venv
        elif hasattr(current, 'envs'):
            # DummyVecEnv
            for e in current.envs:
                if patch_wrapper_chain(e):
                    return True
            break
        else:
            break
    
    return False


class CurriculumReadySingleAgentEnv:
    """
    Mixin class that adds curriculum support to SingleAgentBACEEnv.
    
    Usage:
        class MyEnv(CurriculumReadySingleAgentEnv, SingleAgentBACEEnv):
            pass
    """
    
    def update_spawn_config(self, spawn_updates: Dict[str, Any]) -> bool:
        """Forward spawn config updates to underlying multi-agent env."""
        if hasattr(self, '_ma_env'):
            if hasattr(self._ma_env, 'update_spawn_config'):
                return self._ma_env.update_spawn_config(spawn_updates)
            else:
                # Try to patch it first
                if patch_bace_wrapper(self._ma_env):
                    return self._ma_env.update_spawn_config(spawn_updates)
        return False


# ============================================================================
# Godot-side handler template (for reference)
# ============================================================================

GODOT_HANDLER_TEMPLATE = '''
# Add this to your Godot message handler (e.g., B_ACE_sync.gd)
# in the function that processes incoming messages from Python

func _handle_message(message: Dictionary):
    match message.get("type", ""):
        "update_spawn_config":
            _handle_spawn_config_update(message.get("spawn_updates", {}))
        # ... other message types

func _handle_spawn_config_update(spawn_updates: Dictionary):
    """Update spawn configuration for next reset."""
    
    # Update BlueSpawn
    if spawn_updates.has("BlueSpawn"):
        var blue_spawn = spawn_updates["BlueSpawn"]
        if simConfig["EnvConfig"].has("BlueSpawn"):
            for key in blue_spawn.keys():
                simConfig["EnvConfig"]["BlueSpawn"][key] = blue_spawn[key]
        else:
            simConfig["EnvConfig"]["BlueSpawn"] = blue_spawn
    
    # Update HVAASpawn
    if spawn_updates.has("HVAASpawn"):
        var hvaa_spawn = spawn_updates["HVAASpawn"]
        if simConfig["EnvConfig"].has("HVAASpawn"):
            for key in hvaa_spawn.keys():
                simConfig["EnvConfig"]["HVAASpawn"][key] = hvaa_spawn[key]
        else:
            simConfig["EnvConfig"]["HVAASpawn"] = hvaa_spawn
    
    # Update Red offset
    if spawn_updates.has("RedOffset"):
        var red_offset = spawn_updates["RedOffset"]
        if simConfig["AgentsConfig"].has("red_agents"):
            simConfig["AgentsConfig"]["red_agents"]["rnd_offset_range"] = red_offset
    
    print("[CURRICULUM] Updated spawn config: ", spawn_updates.keys())
'''


def print_godot_handler_template():
    """Print the Godot-side handler code for reference."""
    print(GODOT_HANDLER_TEMPLATE)


if __name__ == "__main__":
    print("B-ACE Curriculum Extension")
    print("=" * 60)
    print("\nTo use this extension, either:")
    print("1. Call patch_bace_wrapper(your_ma_env) after creating the env")
    print("2. Call patch_wrapper_chain(your_wrapped_env) to auto-find and patch")
    print("\nGodot-side handler template:")
    print_godot_handler_template()
