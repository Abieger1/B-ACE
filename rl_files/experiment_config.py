"""
Experimental Configuration Helper

Maps factor levels from the experimental design CSV to actual wrapper configurations.

Usage:
    from experiment_config import configure_expert_wrappers
    
    # From your CSV row
    D_level = 2  # Partial heuristic reward shaping
    E_level = 3  # Full heuristic IL
    
    # Get configured environment
    env = configure_expert_wrappers(
        base_env=env,
        D_reward_shaping_level=D_level,
        E_il_expert_level=E_level,
    )
    
Factor Levels:
    D (Reward Shaping) and E (IL Expert):
        0 = None - no expert guidance
        1 = Scripted - rule-based expert (no heuristics)
        2 = Partial Heuristic - ATDDG features only
        3 = Full Heuristic - all heuristics (ATDDG + WEZ/DMC)
"""

import gymnasium as gym
from typing import Optional, Dict, Any, Callable
import numpy as np

from modular_heuristic_expert import (
    ModularHeuristicExpert,
    create_expert,
    get_base_obs_indices,
    get_enriched_obs_indices,
)
from expert_alignment_wrapper import ExpertAlignmentWrapper, create_alignment_decay_fn
from expert_action_blending_wrapper import ExpertActionBlendingWrapper, create_alpha_decay_fn


# =============================================================================
# LEVEL DESCRIPTIONS (for logging/documentation)
# =============================================================================

LEVEL_DESCRIPTIONS = {
    0: "None - no expert guidance",
    1: "Scripted - rule-based expert (base observations only)",
    2: "Partial Heuristic - ATDDG features (optimal heading, offensive dominance)",
    3: "Full Heuristic - all features (ATDDG + WEZ/DMC)",
}


def describe_level(level: int) -> str:
    """Get human-readable description of a factor level."""
    return LEVEL_DESCRIPTIONS.get(level, f"Unknown level {level}")


# =============================================================================
# MAIN CONFIGURATION FUNCTION
# =============================================================================

def configure_expert_wrappers(
    base_env: gym.Env,
    D_reward_shaping_level: int,
    E_il_expert_level: int,
    # Reward shaping (D) parameters
    alignment_coef: float = 0.005,
    alignment_decay_fn: Optional[Callable[[int], float]] = None,
    # IL blending (E) parameters
    alpha_fn: Optional[Callable[[int], float]] = None,
    initial_alpha: float = 0.7,
    final_alpha: float = 0.1,
    decay_start: int = 500_000,
    decay_end: int = 4_000_000,
    # Shared parameters
    debug: bool = False,
) -> gym.Env:
    """
    Configure environment with expert wrappers based on experimental factor levels.
    
    Args:
        base_env: The base environment (possibly already wrapped with observation enrichment)
        D_reward_shaping_level: Factor D level (0-3)
        E_il_expert_level: Factor E level (0-3)
        alignment_coef: Base coefficient for reward shaping alignment bonus
        alignment_decay_fn: Optional decay function for alignment coefficient
        alpha_fn: Optional custom alpha decay function for IL blending
        initial_alpha: Starting alpha for IL (if alpha_fn not provided)
        final_alpha: Ending alpha for IL (if alpha_fn not provided)
        decay_start: Step to begin alpha decay
        decay_end: Step to reach final alpha
        debug: Enable debug printing
        
    Returns:
        Environment wrapped with appropriate expert wrappers
        
    Notes:
        - Level 0 for a factor means that wrapper is not applied
        - Wrappers are applied in order: base_env -> alignment (D) -> blending (E)
        - Both wrappers can be active simultaneously with different expert configs
    """
    env = base_env
    obs_indices = get_enriched_obs_indices()  # Use full indices, expert will use what it needs
    
    # Track what we're configuring for logging
    config_log = {
        'D_level': D_reward_shaping_level,
        'D_desc': describe_level(D_reward_shaping_level),
        'E_level': E_il_expert_level,
        'E_desc': describe_level(E_il_expert_level),
    }
    
    # =========================================================================
    # FACTOR D: Reward Shaping Wrapper
    # =========================================================================
    if D_reward_shaping_level > 0:
        # Create expert for reward shaping
        D_expert = create_expert(D_reward_shaping_level, debug=debug)
        
        if D_expert is not None:
            # Create decay function if not provided
            if alignment_decay_fn is None:
                alignment_decay_fn = create_alignment_decay_fn(
                    decay_start=decay_start,
                    decay_end=decay_end,
                    final_multiplier=0.3,
                )
            
            env = ExpertAlignmentWrapper(
                env=env,
                expert=D_expert,
                obs_indices=obs_indices,
                alignment_coef=alignment_coef,
                decay_fn=alignment_decay_fn,
                debug=debug,
            )
            
            config_log['D_expert_mode'] = D_expert.heuristic_mode
            
            if debug:
                print(f"[CONFIG] Added ExpertAlignmentWrapper with {D_expert.heuristic_mode} expert")
    
    # =========================================================================
    # FACTOR E: IL Blending Wrapper
    # =========================================================================
    if E_il_expert_level > 0:
        # Create expert for IL blending
        E_expert = create_expert(E_il_expert_level, debug=debug)
        
        if E_expert is not None:
            # Create alpha function if not provided
            if alpha_fn is None:
                alpha_fn = create_alpha_decay_fn(
                    initial_alpha=initial_alpha,
                    final_alpha=final_alpha,
                    decay_start=decay_start,
                    decay_end=decay_end,
                    decay_type='linear',
                )
            
            env = ExpertActionBlendingWrapper(
                env=env,
                expert=E_expert,
                obs_indices=obs_indices,
                alpha_fn=alpha_fn,
                debug=debug,
            )
            
            config_log['E_expert_mode'] = E_expert.heuristic_mode
            
            if debug:
                print(f"[CONFIG] Added ExpertActionBlendingWrapper with {E_expert.heuristic_mode} expert")
    
    # Store configuration in environment for later reference
    env._experiment_config = config_log
    
    if debug:
        print(f"[CONFIG] Final configuration: D={D_reward_shaping_level} ({describe_level(D_reward_shaping_level)}), "
              f"E={E_il_expert_level} ({describe_level(E_il_expert_level)})")
    
    return env


# =============================================================================
# BATCH CONFIGURATION FROM CSV
# =============================================================================

def configure_from_csv_row(
    base_env: gym.Env,
    row: Dict[str, Any],
    debug: bool = False,
    **kwargs,
) -> gym.Env:
    """
    Configure environment from a row of the experimental design CSV.
    
    Args:
        base_env: Base environment
        row: Dictionary with keys 'D_Reward_Shaping' and 'E_IL_Expert'
        debug: Enable debug printing
        **kwargs: Additional parameters passed to configure_expert_wrappers
        
    Returns:
        Configured environment
        
    Example:
        import pandas as pd
        design = pd.read_csv('thesis_splitplot_design.csv')
        
        for _, row in design.iterrows():
            env = configure_from_csv_row(base_env, row.to_dict())
            # ... run training ...
    """
    D_level = int(row.get('D_Reward_Shaping', row.get('D', 0)))
    E_level = int(row.get('E_IL_Expert', row.get('E', 0)))
    
    return configure_expert_wrappers(
        base_env=base_env,
        D_reward_shaping_level=D_level,
        E_il_expert_level=E_level,
        debug=debug,
        **kwargs,
    )


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_factor_levels(D_level: int, E_level: int) -> bool:
    """Check if factor levels are valid."""
    valid = D_level in (0, 1, 2, 3) and E_level in (0, 1, 2, 3)
    if not valid:
        raise ValueError(f"Invalid factor levels: D={D_level}, E={E_level}. Must be 0-3.")
    return True


def get_expected_observation_dims(A: int, B: int, C: int) -> int:
    """
    Get expected observation dimensions based on observation factors.
    
    Args:
        A: WEZ/DMC observations (0 or 1)
        B: Range-PE observations (0 or 1)
        C: ATDDG observations (0 or 1)
        
    Returns:
        Expected observation space dimensions
        
    Note: This depends on your enriched_observation_wrapper implementation.
    Adjust the counts based on how many features each category adds.
    """
    base_dims = 27
    
    # These counts should match your enriched_observation_wrapper
    wez_dmc_dims = 4 if A else 0  # bez_penetration, inside_bez, dmc_normalized, inside_threat
    range_pe_dims = 2 if B else 0  # time_to_capture, capture_feasible (example)
    atddg_dims = 7 if C else 0  # defense_time_ratio, in_escape_region, barrier_value, 
                                 # heading_to_optimal, offensive_dominance, escape_cone, capture_prob
    
    return base_dims + wez_dmc_dims + range_pe_dims + atddg_dims


# =============================================================================
# EXPERIMENT METADATA
# =============================================================================

def get_experiment_metadata(
    run_id: int,
    stage: int,
    A: int, B: int, C: int,
    D: int, E: int,
) -> Dict[str, Any]:
    """
    Generate metadata dictionary for logging/tracking an experimental run.
    
    Returns a dictionary suitable for logging to TensorBoard, W&B, or saving to JSON.
    """
    return {
        'run_id': run_id,
        'stage': stage,
        'factors': {
            'A_WEZ_DMC_Obs': A,
            'B_Range_PE_Obs': B,
            'C_ATDDG_Obs': C,
            'D_Reward_Shaping': D,
            'E_IL_Expert': E,
        },
        'descriptions': {
            'A': 'On' if A else 'Off',
            'B': 'On' if B else 'Off',
            'C': 'On' if C else 'Off',
            'D': describe_level(D),
            'E': describe_level(E),
        },
        'expected_obs_dims': get_expected_observation_dims(A, B, C),
    }


# =============================================================================
# TESTING / EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Print all configurations
    print("Experimental Factor Configurations\n" + "=" * 50)
    
    for D in range(4):
        for E in range(4):
            print(f"\nD={D}, E={E}")
            print(f"  D: {describe_level(D)}")
            print(f"  E: {describe_level(E)}")
            
            # Show what experts would be created
            D_expert = create_expert(D)
            E_expert = create_expert(E)
            
            D_mode = D_expert.heuristic_mode if D_expert else "None"
            E_mode = E_expert.heuristic_mode if E_expert else "None"
            
            print(f"  D expert mode: {D_mode}")
            print(f"  E expert mode: {E_mode}")
