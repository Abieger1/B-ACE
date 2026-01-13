"""
train_bace_with_spawn_curriculum.py

This is a PATCH file showing the exact modifications needed to add
spawn curriculum to train_bace_clean.py.

Instructions:
1. Copy spawn_curriculum.py and b_ace_curriculum_extension.py to your rl_files/ directory
2. Apply the changes shown below to your train_bace_clean.py

To use curriculum training:
    python train_bace_clean.py --use-curriculum --curriculum-start 1000000 --curriculum-end 7000000
"""

# ============================================================================
# CHANGES TO APPLY TO train_bace_clean.py
# ============================================================================

# ==== CHANGE 1: Add imports near the top of the file (after line ~45) ====

IMPORT_BLOCK = '''
# Spawn curriculum for gradually increasing environmental stochasticity
try:
    from spawn_curriculum import (
        SpawnCurriculumConfig,
        SpawnCurriculumWrapper,
        SpawnCurriculumCallback,
    )
    from b_ace_curriculum_extension import patch_wrapper_chain
    CURRICULUM_AVAILABLE = True
except ImportError:
    CURRICULUM_AVAILABLE = False
    print("Note: spawn_curriculum not found. Curriculum learning disabled.")
'''


# ==== CHANGE 2: Add arguments to _parse_args() (around line ~1440) ====

ARGS_BLOCK = '''
    # ================== SPAWN CURRICULUM ==================
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        default=False,
        help="Enable spawn stochasticity curriculum (gradually increase randomization)"
    )
    parser.add_argument(
        "--curriculum-start",
        type=int,
        default=1_000_000,
        help="Timestep when curriculum begins (hold constant before this)"
    )
    parser.add_argument(
        "--curriculum-end",
        type=int,
        default=7_000_000,
        help="Timestep when curriculum reaches full randomization"
    )
    parser.add_argument(
        "--curriculum-blue-heading-end",
        type=float,
        nargs=2,
        default=[-15.0, 15.0],
        metavar=("MIN", "MAX"),
        help="Final blue heading range [min, max] degrees"
    )
    parser.add_argument(
        "--curriculum-blue-radius-end",
        type=float,
        default=5.0,
        help="Final blue spawn radius (NM)"
    )
    parser.add_argument(
        "--curriculum-hvaa-heading-end",
        type=float,
        nargs=2,
        default=[150.0, 180.0],
        metavar=("MIN", "MAX"),
        help="Final HVAA heading range [min, max] degrees"
    )
'''


# ==== CHANGE 3: Add curriculum config creation in main() (after loading B_ACE_config) ====

CURRICULUM_CONFIG_BLOCK = '''
    # ================== SPAWN CURRICULUM CONFIG ==================
    curriculum_config = None
    if args.use_curriculum and CURRICULUM_AVAILABLE:
        # Get initial values from config to start deterministic
        env_cfg = B_ACE_config.get("EnvConfig", {})
        blue_spawn = env_cfg.get("BlueSpawn", {})
        hvaa_spawn = env_cfg.get("HVAASpawn", {})
        
        # Determine starting heading from config or use defaults
        blue_hdg_start = (0.0, 0.0)  # Deterministic
        hvaa_hdg_start = hvaa_spawn.get("heading_deg_range", [165.0, 165.0])
        if hvaa_hdg_start[0] != hvaa_hdg_start[1]:
            # Config already has range, start from midpoint
            mid = (hvaa_hdg_start[0] + hvaa_hdg_start[1]) / 2
            hvaa_hdg_start = (mid, mid)
        hvaa_hdg_start = tuple(hvaa_hdg_start)
        
        curriculum_config = SpawnCurriculumConfig(
            start_step=args.curriculum_start,
            end_step=args.curriculum_end,
            
            # Blue agent: start deterministic
            blue_heading_start=blue_hdg_start,
            blue_heading_end=tuple(args.curriculum_blue_heading_end),
            blue_radius_start=0.0,
            blue_radius_end=args.curriculum_blue_radius_end,
            
            # HVAA: start deterministic
            hvaa_heading_start=hvaa_hdg_start,
            hvaa_heading_end=tuple(args.curriculum_hvaa_heading_end),
            hvaa_rect_expand_start=0.0,
            hvaa_rect_expand_end=3.0,
            
            # Red agent offset
            red_offset_start=(0.0, 0.0, 0.0),
            red_offset_end=(5.0, 0.0, 5.0),
            
            verbose=True,
        )
        
        print(f"\\n{'='*60}")
        print("SPAWN CURRICULUM ENABLED")
        print(f"{'='*60}")
        print(f"  Start step: {curriculum_config.start_step:,}")
        print(f"  End step: {curriculum_config.end_step:,}")
        print(f"  Blue heading: {curriculum_config.blue_heading_start} → {curriculum_config.blue_heading_end}")
        print(f"  Blue radius: {curriculum_config.blue_radius_start} → {curriculum_config.blue_radius_end}")
        print(f"  HVAA heading: {curriculum_config.hvaa_heading_start} → {curriculum_config.hvaa_heading_end}")
        print(f"{'='*60}\\n")
'''


# ==== CHANGE 4: Modify make_env _thunk() to add curriculum wrapper ====
# This goes AFTER your existing wrappers but BEFORE NanGuardWrapper and Monitor

MAKE_ENV_CURRICULUM_BLOCK = '''
            # ================== SPAWN CURRICULUM WRAPPER ==================
            # (Add this after your shaping wrappers but before NanGuardWrapper)
            if curriculum_config is not None and not is_eval and CURRICULUM_AVAILABLE:
                # Patch the B-ACE wrapper to support config updates
                patch_wrapper_chain(e)
                
                # Add curriculum wrapper
                e = SpawnCurriculumWrapper(
                    env=e,
                    config=curriculum_config,
                    bace_config=B_ACE_config,
                )
                print(f"✓ Spawn curriculum wrapper added (training env only)")
'''


# ==== CHANGE 5: Add curriculum callback creation (after eval callback, before callback_list) ====

CALLBACK_BLOCK = '''
    # ================== SPAWN CURRICULUM CALLBACK ==================
    curriculum_callback = None
    if args.use_curriculum and CURRICULUM_AVAILABLE and curriculum_config is not None:
        # Create callback (it will auto-find the curriculum wrapper)
        curriculum_callback = SpawnCurriculumCallback(
            curriculum_wrapper=None,  # Will auto-find
            update_freq=1000,
            verbose=1,
        )
        print(f"✓ Spawn curriculum callback created")
'''


# ==== CHANGE 6: Add curriculum callback to callback_list ====

CALLBACK_LIST_CHANGE = '''
    # Change from:
    # callback_list = CallbackList([metrics_callback, eval_checkpoint_callback])
    
    # To:
    callbacks_list = [metrics_callback, eval_checkpoint_callback]
    if curriculum_callback is not None:
        callbacks_list.append(curriculum_callback)
    callback_list = CallbackList(callbacks_list)
'''


# ============================================================================
# COMPLETE MODIFIED make_env FUNCTION (for reference)
# ============================================================================

def example_make_env(log_dir_str, seed, is_eval, B_ACE_config, args, curriculum_config):
    """
    Complete example of make_env with curriculum support.
    
    This shows the full function for reference - compare with your existing
    make_env and integrate the curriculum wrapper addition.
    """
    def _thunk():
        # Assuming these imports are at the top of your file
        from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
        from b_ace_py.enriched_observation_wrapper import EnrichedObservationWrapper
        from stable_baselines3.common.monitor import Monitor
        import gymnasium as gym
        import numpy as np
        
        # ---- Your SingleAgentBACEEnv class (assuming it exists) ----
        # e = SingleAgentBACEEnv(bace_config=B_ACE_config, ...)
        
        # ---- Placeholder for the rest of your wrapper chain ----
        # e = EnrichedObservationWrapper(e, ...) if args.use_enriched_obs
        # e = HeadingRateLimitWrapper(e, ...)
        # e = ActionSmoothnessPenalty(e, ...)
        # ... etc.
        
        # ======== ADD CURRICULUM WRAPPER HERE ========
        # (After your shaping wrappers, before NanGuardWrapper/Monitor)
        
        CURRICULUM_AVAILABLE = True  # Set based on import success
        
        if curriculum_config is not None and not is_eval and CURRICULUM_AVAILABLE:
            # Import if not already done
            from spawn_curriculum import SpawnCurriculumWrapper
            from b_ace_curriculum_extension import patch_wrapper_chain
            
            # Patch the underlying B-ACE wrapper
            patch_wrapper_chain(e)
            
            # Add curriculum wrapper
            e = SpawnCurriculumWrapper(
                env=e,
                config=curriculum_config,
                bace_config=B_ACE_config,
            )
            print(f"✓ Spawn curriculum wrapper added")
        
        # ======== END CURRICULUM ADDITION ========
        
        # ---- Final wrappers (NanGuard, Monitor) ----
        # e = NanGuardWrapper(e, name=("eval_env" if is_eval else "train_env"))
        # e = Monitor(e, filename=str(Path(log_dir_str) / "monitor.csv"))
        
        # return e
        pass
    
    return _thunk


# ============================================================================
# TEST: Verify the curriculum module works
# ============================================================================

if __name__ == "__main__":
    print("Testing spawn curriculum module...")
    print("=" * 60)
    
    # Test 1: SpawnCurriculumConfig
    print("\n1. Testing SpawnCurriculumConfig:")
    try:
        from spawn_curriculum import SpawnCurriculumConfig
        config = SpawnCurriculumConfig(
            start_step=1_000_000,
            end_step=7_000_000,
        )
        
        # Test interpolation
        for t in [0, 1_000_000, 4_000_000, 7_000_000, 10_000_000]:
            params = config.get_spawn_params(t)
            print(f"   t={t:>10,}: progress={params['progress']:.2f}, "
                  f"blue_hdg={params['blue_heading_range']}, "
                  f"blue_r={params['blue_radius']:.1f}")
        print("   ✓ SpawnCurriculumConfig works!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Import extension
    print("\n2. Testing b_ace_curriculum_extension:")
    try:
        from b_ace_curriculum_extension import patch_bace_wrapper
        print("   ✓ Extension imports successfully!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("INTEGRATION CHECKLIST:")
    print("=" * 60)
    print("""
    [ ] Copy spawn_curriculum.py to your rl_files/ directory
    [ ] Copy b_ace_curriculum_extension.py to your rl_files/ directory
    [ ] Add imports (see IMPORT_BLOCK above)
    [ ] Add command-line arguments (see ARGS_BLOCK above)
    [ ] Add curriculum config creation in main() (see CURRICULUM_CONFIG_BLOCK)
    [ ] Add curriculum wrapper in make_env (see MAKE_ENV_CURRICULUM_BLOCK)
    [ ] Add curriculum callback (see CALLBACK_BLOCK)
    [ ] Update callback_list to include curriculum_callback
    
    To run with curriculum:
        python train_bace_clean.py --use-curriculum
    
    To customize curriculum:
        python train_bace_clean.py --use-curriculum \\
            --curriculum-start 1000000 \\
            --curriculum-end 7000000 \\
            --curriculum-blue-heading-end -20 20 \\
            --curriculum-blue-radius-end 6.0
    """)
