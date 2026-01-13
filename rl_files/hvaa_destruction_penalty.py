"""
HVAADestructionPenalty - Applies immediate penalty when HVAA is destroyed.

The Godot environment reports hvaa_alive/hvaa_destroyed in info dict every step.
This wrapper detects the transition from alive→destroyed and applies a large
immediate penalty, enabling proper credit assignment for PPO.
"""

import numpy as np
import gymnasium as gym


class HVAADestructionPenalty(gym.Wrapper):
    """
    Detects HVAA destruction and applies immediate penalty.
    
    The key insight: PPO needs the penalty at the exact timestep of destruction,
    not as a delayed terminal reward hundreds/thousands of steps later.
    
    Parameters:
    -----------
    penalty : float
        Negative reward applied when HVAA transitions from alive to destroyed.
        Default -10.0 matches the Godot hvaa_loss_factor.
    only_once : bool
        If True, only applies penalty once per episode (first destruction).
        If False, applies if HVAA somehow respawns and dies again.
    """
    
    def __init__(
        self,
        env,
        penalty: float = -10.0,
        only_once: bool = True,
    ):
        super().__init__(env)
        self.penalty = float(penalty)
        self.only_once = only_once
        
        self._prev_hvaa_alive = True
        self._already_penalized = False
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_hvaa_alive = info.get("hvaa_alive", True)
        self._already_penalized = False
        return obs, info
    
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        # Check for HVAA destruction event
        hvaa_alive = info.get("hvaa_alive", True)
        hvaa_destroyed_now = (self._prev_hvaa_alive and not hvaa_alive)
        
        penalty_applied = 0.0
        if hvaa_destroyed_now:
            if not self.only_once or not self._already_penalized:
                penalty_applied = self.penalty
                rew = float(rew) + penalty_applied
                self._already_penalized = True
        
        self._prev_hvaa_alive = hvaa_alive
        
        # Logging
        info["hvaa_destruction_penalty"] = float(penalty_applied)
        info["hvaa_destruction_event"] = hvaa_destroyed_now
        
        return obs, rew, terminated, truncated, info


class HVAADestructionPenaltyWithDecay(gym.Wrapper):
    """
    Same as above but with optional penalty decay over training.
    
    Use case: Start with high penalty to establish behavior, then
    reduce it so the agent can focus on other objectives.
    
    Parameters:
    -----------
    penalty_start : float
        Initial penalty (e.g., -15.0)
    penalty_end : float
        Final penalty after annealing (e.g., -5.0)
    anneal_steps : int
        Steps over which to anneal
    hold_steps : int
        Steps to hold at initial value before annealing
    """
    
    def __init__(
        self,
        env,
        penalty_start: float = -10.0,
        penalty_end: float = -10.0,
        anneal_steps: int = 2_000_000,
        hold_steps: int = 0,
    ):
        super().__init__(env)
        self.penalty_start = float(penalty_start)
        self.penalty_end = float(penalty_end)
        self.anneal_steps = int(anneal_steps)
        self.hold_steps = int(hold_steps)
        
        self.penalty = float(penalty_start)
        self._prev_hvaa_alive = True
        self._already_penalized = False
        self._global_step = 0
        
    def set_global_step(self, t: int):
        self._global_step = int(t)
        # Update penalty from schedule
        self.penalty = self._compute_penalty(t)
        if hasattr(self.env, "set_global_step"):
            try:
                self.env.set_global_step(t)
            except Exception:
                pass
    
    def _compute_penalty(self, t: int) -> float:
        if t <= self.hold_steps:
            return self.penalty_start
        if self.anneal_steps <= 0:
            return self.penalty_end
        x = min(1.0, max(0.0, (t - self.hold_steps) / float(self.anneal_steps)))
        return self.penalty_start + x * (self.penalty_end - self.penalty_start)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_hvaa_alive = info.get("hvaa_alive", True)
        self._already_penalized = False
        return obs, info
    
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        hvaa_alive = info.get("hvaa_alive", True)
        hvaa_destroyed_now = (self._prev_hvaa_alive and not hvaa_alive)
        
        penalty_applied = 0.0
        if hvaa_destroyed_now and not self._already_penalized:
            penalty_applied = self.penalty
            rew = float(rew) + penalty_applied
            self._already_penalized = True
        
        self._prev_hvaa_alive = hvaa_alive
        
        info["hvaa_destruction_penalty"] = float(penalty_applied)
        info["hvaa_destruction_event"] = hvaa_destroyed_now
        info["hvaa_penalty_coef"] = float(self.penalty)
        
        return obs, rew, terminated, truncated, info
