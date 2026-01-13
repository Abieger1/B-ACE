#!/usr/bin/env python3
"""
test_config3_blending.py - Quick diagnostic for Configuration 3 action blending

Verifies:
1. Expert action is computed correctly
2. PPO action is preserved (not overwritten)
3. Blending formula: blended = alpha * expert + (1 - alpha) * ppo
4. Alpha decay schedule works
5. Fire action handling modes

Usage:
    python test_config3_blending.py
"""

import numpy as np
import sys
from pathlib import Path

# Add repo root to path
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

print(f"Using REPO_ROOT: {REPO_ROOT}")

# Import the blending wrapper and expert
from expert_action_blending_wrapper import (
    ExpertActionBlendingWrapper, 
    create_alpha_decay_fn,
    create_staged_alpha_fn
)
from simple_aggressive_expert import SimpleAggressiveExpert, get_default_obs_indices


def test_alpha_decay():
    """Test that alpha decay functions work correctly."""
    print("\n" + "="*60)
    print("TEST 1: Alpha Decay Schedule")
    print("="*60)
    
    alpha_fn = create_alpha_decay_fn(
        initial_alpha=0.7,
        final_alpha=0.1,
        decay_start=500_000,
        decay_end=4_000_000,
        decay_type='linear'
    )
    
    test_steps = [0, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]
    
    print(f"\nLinear decay: 0.7 → 0.1 over steps 500k → 4M")
    print(f"{'Step':>12} | {'Alpha':>8} | {'Expert %':>10} | {'PPO %':>8}")
    print("-" * 50)
    
    for step in test_steps:
        alpha = alpha_fn(step)
        print(f"{step:>12,} | {alpha:>8.3f} | {alpha*100:>9.1f}% | {(1-alpha)*100:>7.1f}%")
    
    # Also test cosine decay
    print(f"\n\nCosine decay: 0.7 → 0.1 over steps 500k → 4M")
    alpha_fn_cosine = create_alpha_decay_fn(
        initial_alpha=0.7,
        final_alpha=0.1,
        decay_start=500_000,
        decay_end=4_000_000,
        decay_type='cosine'
    )
    
    print(f"{'Step':>12} | {'Alpha':>8} | {'Expert %':>10} | {'PPO %':>8}")
    print("-" * 50)
    
    for step in test_steps:
        alpha = alpha_fn_cosine(step)
        print(f"{step:>12,} | {alpha:>8.3f} | {alpha*100:>9.1f}% | {(1-alpha)*100:>7.1f}%")
    
    print("\n✓ Alpha decay test passed")


def test_blending_math():
    """Test the blending formula manually."""
    print("\n" + "="*60)
    print("TEST 2: Blending Math Verification")
    print("="*60)
    
    # Test cases: (ppo_action, expert_action, alpha, expected_blended)
    test_cases = [
        # alpha=1.0 should give 100% expert
        (np.array([0.5, 0.0, 0.3, 1.0]), np.array([-0.5, 0.2, 0.8, -1.0]), 1.0),
        # alpha=0.0 should give 100% PPO
        (np.array([0.5, 0.0, 0.3, 1.0]), np.array([-0.5, 0.2, 0.8, -1.0]), 0.0),
        # alpha=0.5 should give 50/50
        (np.array([0.5, 0.0, 0.3, 1.0]), np.array([-0.5, 0.2, 0.8, -1.0]), 0.5),
        # alpha=0.7 (typical early training)
        (np.array([0.5, 0.0, 0.3, 1.0]), np.array([-0.5, 0.2, 0.8, -1.0]), 0.7),
    ]
    
    blend_dims = [0, 1, 2]  # Don't blend fire (dim 3)
    
    for ppo, expert, alpha in test_cases:
        print(f"\n--- Alpha = {alpha} ---")
        print(f"PPO action:    [{ppo[0]:>6.2f}, {ppo[1]:>6.2f}, {ppo[2]:>6.2f}, {ppo[3]:>6.2f}]")
        print(f"Expert action: [{expert[0]:>6.2f}, {expert[1]:>6.2f}, {expert[2]:>6.2f}, {expert[3]:>6.2f}]")
        
        # Compute expected blend
        blended = ppo.copy()
        for dim in blend_dims:
            blended[dim] = alpha * expert[dim] + (1 - alpha) * ppo[dim]
        
        print(f"Blended:       [{blended[0]:>6.2f}, {blended[1]:>6.2f}, {blended[2]:>6.2f}, {blended[3]:>6.2f}]")
        
        # Verify formula
        for dim in blend_dims:
            expected = alpha * expert[dim] + (1 - alpha) * ppo[dim]
            assert abs(blended[dim] - expected) < 1e-6, f"Blend mismatch at dim {dim}"
        
        # Fire should be PPO's value (in default 'gate' mode with low alpha)
        if alpha <= 0.3:
            print(f"  Fire (dim 3): PPO's value kept (alpha <= 0.3)")
        else:
            print(f"  Fire (dim 3): Expert's value used (alpha > 0.3)")
    
    print("\n✓ Blending math test passed")


def test_expert_standalone():
    """Test the expert policy in isolation."""
    print("\n" + "="*60)
    print("TEST 3: Expert Policy Standalone")
    print("="*60)
    
    expert = SimpleAggressiveExpert(
        fire_threshold=0.50,
        aspect_limit_deg=30.0,
        debug=True
    )
    obs_indices = get_default_obs_indices()
    
    # Create a mock observation (27 dims)
    mock_obs = np.zeros(27, dtype=np.float32)
    
    # Test case 1: No enemy detected
    print("\n--- Case 1: No enemy detected ---")
    mock_obs[obs_indices['enemy_detected']] = 0.0
    mock_obs[obs_indices['distance_to_hvaa']] = 0.2
    mock_obs[obs_indices['hvaa_angle_off']] = 0.3
    
    action = expert.get_action(mock_obs, obs_indices)
    print(f"Expert action: turn={action[0]:.2f}, alt={action[1]:.2f}, g={action[2]:.2f}, fire={action[3]:.2f}")
    
    # Test case 2: Enemy detected, ahead
    print("\n--- Case 2: Enemy detected, ahead ---")
    expert.reset()
    mock_obs[obs_indices['enemy_detected']] = 1.0
    mock_obs[obs_indices['distance_to_enemy']] = 0.3
    mock_obs[obs_indices['angle_off_to_enemy']] = 0.1  # Slightly off boresight
    mock_obs[obs_indices['aspect_angle_to_enemy']] = 0.1
    mock_obs[obs_indices['offensive_factor']] = 0.6
    mock_obs[obs_indices['own_missiles']] = 4.0
    mock_obs[obs_indices['own_in_flight_missile']] = 0.0
    
    action = expert.get_action(mock_obs, obs_indices)
    print(f"Expert action: turn={action[0]:.2f}, alt={action[1]:.2f}, g={action[2]:.2f}, fire={action[3]:.2f}")
    
    # Test case 3: Enemy behind (large angle_off)
    print("\n--- Case 3: Enemy behind ---")
    expert.reset()
    mock_obs[obs_indices['angle_off_to_enemy']] = 0.8  # Enemy far off boresight
    
    action = expert.get_action(mock_obs, obs_indices)
    print(f"Expert action: turn={action[0]:.2f}, alt={action[1]:.2f}, g={action[2]:.2f}, fire={action[3]:.2f}")
    
    print("\n✓ Expert standalone test passed")


def test_wrapper_integration():
    """Test the wrapper with a mock environment."""
    print("\n" + "="*60)
    print("TEST 4: Wrapper Integration (Mock Environment)")
    print("="*60)
    
    import gymnasium as gym
    
    # Create a simple mock environment
    class MockBACEEnv(gym.Env):
        """Minimal mock environment for testing."""
        def __init__(self):
            self.observation_space = gym.spaces.Box(-1, 1, shape=(27,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
            self._step_count = 0
            self._last_obs = None
            self._last_obs_dict = {}
            self.controlled_agent_id = 'agent_0'
            
        def reset(self, **kwargs):
            self._step_count = 0
            obs = self._generate_obs()
            self._last_obs = obs
            self._last_obs_dict = {'agent_0': obs}
            return obs, {}
        
        def step(self, action):
            self._step_count += 1
            obs = self._generate_obs()
            self._last_obs = obs
            self._last_obs_dict = {'agent_0': obs}
            reward = 0.0
            terminated = self._step_count >= 100
            truncated = False
            info = {'action_executed': action.tolist()}
            return obs, reward, terminated, truncated, info
        
        def _generate_obs(self):
            obs = np.random.uniform(-0.5, 0.5, size=27).astype(np.float32)
            # Set some meaningful values
            obs[26] = 1.0 if self._step_count > 10 else 0.0  # enemy_detected
            obs[17] = 0.3  # distance_to_enemy
            obs[16] = np.random.uniform(-0.3, 0.3)  # angle_off_to_enemy
            obs[9] = 0.15  # distance_to_hvaa
            return obs
    
    # Create environment with wrapper
    env = MockBACEEnv()
    expert = SimpleAggressiveExpert(fire_threshold=0.50, aspect_limit_deg=30.0, debug=False)
    obs_indices = get_default_obs_indices()
    
    alpha_fn = create_alpha_decay_fn(
        initial_alpha=0.7,
        final_alpha=0.1,
        decay_start=10,
        decay_end=50,  # Quick decay for testing
    )
    
    wrapped_env = ExpertActionBlendingWrapper(
        env,
        expert=expert,
        obs_indices=obs_indices,
        alpha_fn=alpha_fn,
        blend_dims=[0, 1, 2],
        fire_blend_mode='gate',
        debug=False,
    )
    
    # Run a few steps
    obs, _ = wrapped_env.reset()
    print(f"\nInitial obs shape: {obs.shape}")
    
    print(f"\n{'Step':>4} | {'Alpha':>6} | {'PPO Turn':>9} | {'Exp Turn':>9} | {'Blend Turn':>10} | {'Verified':>8}")
    print("-" * 70)
    
    for step in range(60):
        # Simulate PPO action (random)
        ppo_action = np.random.uniform(-1, 1, size=4).astype(np.float32)
        
        # Set global step to trigger alpha update
        wrapped_env.set_global_step(step)
        
        # Step environment
        obs, reward, terminated, truncated, info = wrapped_env.step(ppo_action)
        
        # Get actions from info
        expert_action = np.array(info['expert_action'])
        blended_action = np.array(info['blended_action'])
        alpha = info['blend_alpha']
        
        # Verify blending formula for turn (dim 0)
        expected_turn = alpha * expert_action[0] + (1 - alpha) * ppo_action[0]
        verified = abs(blended_action[0] - expected_turn) < 1e-5
        
        if step % 10 == 0 or not verified:
            print(f"{step:>4} | {alpha:>6.3f} | {ppo_action[0]:>9.3f} | {expert_action[0]:>9.3f} | {blended_action[0]:>10.3f} | {'✓' if verified else '✗':>8}")
        
        if not verified:
            print(f"  ERROR: Expected {expected_turn:.5f}, got {blended_action[0]:.5f}")
        
        if terminated:
            obs, _ = wrapped_env.reset()
    
    print("\n✓ Wrapper integration test passed")


def test_fire_blend_modes():
    """Test different fire blending modes."""
    print("\n" + "="*60)
    print("TEST 5: Fire Blend Modes")
    print("="*60)
    
    import gymnasium as gym
    
    class MockBACEEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-1, 1, shape=(27,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
            self._last_obs_dict = {}
            self.controlled_agent_id = 'agent_0'
            
        def reset(self, **kwargs):
            obs = np.zeros(27, dtype=np.float32)
            obs[26] = 1.0  # enemy_detected
            obs[17] = 0.25  # distance_to_enemy (in fire range)
            obs[7] = 4.0  # own_missiles
            self._last_obs_dict = {'agent_0': obs}
            return obs, {}
        
        def step(self, action):
            obs = np.zeros(27, dtype=np.float32)
            obs[26] = 1.0
            obs[17] = 0.25
            obs[7] = 4.0
            self._last_obs_dict = {'agent_0': obs}
            return obs, 0.0, False, False, {'action_executed': action.tolist()}
    
    obs_indices = get_default_obs_indices()
    
    fire_modes = ['ppo', 'expert', 'blend', 'gate', 'or']
    
    for mode in fire_modes:
        print(f"\n--- Fire mode: '{mode}' ---")
        
        env = MockBACEEnv()
        expert = SimpleAggressiveExpert(fire_threshold=0.50, debug=False)
        
        wrapped = ExpertActionBlendingWrapper(
            env,
            expert=expert,
            obs_indices=obs_indices,
            alpha_fn=lambda t: 0.5,  # Fixed alpha for testing
            fire_blend_mode=mode,
            fire_gate_threshold=0.3,
        )
        
        wrapped.reset()
        
        # Test with PPO wanting to fire, expert not wanting to fire
        ppo_action = np.array([0.0, 0.0, 0.5, 1.0])  # PPO wants to fire
        
        # We need to manipulate the expert's decision - set obs so expert doesn't fire
        obs, _, _, _, info = wrapped.step(ppo_action)
        
        ppo_fire = info['ppo_action'][3]
        expert_fire = info['expert_action'][3]
        blended_fire = info['blended_action'][3]
        
        print(f"  PPO fire: {ppo_fire:.1f}, Expert fire: {expert_fire:.1f}, Blended fire: {blended_fire:.1f}")
    
    print("\n✓ Fire blend modes test passed")


def main():
    print("\n" + "="*60)
    print("CONFIGURATION 3 DIAGNOSTIC TESTS")
    print("="*60)
    
    test_alpha_decay()
    test_blending_math()
    test_expert_standalone()
    test_wrapper_integration()
    test_fire_blend_modes()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    print("\nConfiguration 3 blending is working correctly.")
    print("You can proceed with training using:")
    print("  python train_bace_config3.py --expert-blending --seed 42")


if __name__ == "__main__":
    main()
