"""
SustainedTurnPenalty - Penalizes constant/sustained spinning behavior.

The key insight is that jerk penalties (penalizing Δaction) become 0 once an agent 
commits to a constant spin. This wrapper tracks turn commands over a rolling window 
and penalizes sustained high turn rates.
"""

import numpy as np
import gymnasium as gym
from collections import deque


def linear_anneal(t: int, start: float, end: float, steps: int, hold: int = 0) -> float:
    """Hold `start` for `hold` steps, then linearly move to `end` over `steps`."""
    if t <= hold:
        return float(start)
    if steps <= 0:
        return float(end)
    x = min(1.0, max(0.0, (t - hold) / float(steps)))
    return float(start + x * (end - start))


class SustainedTurnPenalty(gym.Wrapper):
    """
    Penalizes sustained high turn rates by tracking turn commands over a rolling window.
    
    Two penalty components:
    1. Mean penalty: Penalizes when mean |turn| over the window is high
    2. Consistency penalty: Extra penalty when turning consistently in the same direction
    
    Parameters:
    -----------
    hdg_idx : int
        Index of heading command in action array (default 0)
    window_size : int
        Number of steps to track (default 20)
    turn_threshold : float
        Mean |turn| above this triggers penalty (default 0.6)
    consistency_threshold : float
        |mean(turn)| / mean(|turn|) above this indicates consistent direction (default 0.8)
    mean_coef_start/end : float
        Coefficient for mean turn penalty (annealed)
    consistency_coef_start/end : float
        Extra coefficient when turning same direction consistently
    anneal_steps : int
        Steps over which to anneal coefficients
    hold_steps : int
        Steps to hold at start values before annealing
    """
    
    def __init__(
        self,
        env,
        hdg_idx: int = 0,
        window_size: int = 20,
        turn_threshold: float = 0.6,
        consistency_threshold: float = 0.8,
        # Penalty coefficients (annealed)
        mean_coef_start: float = 0.0,
        mean_coef_end: float = 0.005,
        consistency_coef_start: float = 0.0,
        consistency_coef_end: float = 0.003,
        anneal_steps: int = 2_000_000,
        hold_steps: int = 500_000,
    ):
        super().__init__(env)
        self.hdg_idx = int(hdg_idx)
        self.window_size = int(window_size)
        self.turn_threshold = float(turn_threshold)
        self.consistency_threshold = float(consistency_threshold)
        
        self.mean_coef_start = float(mean_coef_start)
        self.mean_coef_end = float(mean_coef_end)
        self.consistency_coef_start = float(consistency_coef_start)
        self.consistency_coef_end = float(consistency_coef_end)
        self.anneal_steps = int(anneal_steps)
        self.hold_steps = int(hold_steps)
        
        # Current coefficient values
        self.mean_coef = float(mean_coef_start)
        self.consistency_coef = float(consistency_coef_start)
        
        # Rolling window of heading commands
        self._turn_history = deque(maxlen=window_size)
        self._global_step = 0
        
    def set_global_step(self, t: int):
        self._global_step = int(t)
        if hasattr(self.env, "set_global_step"):
            try:
                self.env.set_global_step(t)
            except Exception:
                pass
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._turn_history.clear()
        return obs, info
    
    def step(self, action):
        # Update coefficients from schedule
        self.mean_coef = linear_anneal(
            self._global_step, 
            self.mean_coef_start, 
            self.mean_coef_end, 
            self.anneal_steps, 
            hold=self.hold_steps
        )
        self.consistency_coef = linear_anneal(
            self._global_step,
            self.consistency_coef_start,
            self.consistency_coef_end,
            self.anneal_steps,
            hold=self.hold_steps
        )
        
        a = np.asarray(action, dtype=np.float32)
        turn_cmd = float(a[self.hdg_idx])
        
        # Add to history
        self._turn_history.append(turn_cmd)
        
        # Compute penalties only when window is full
        mean_penalty = 0.0
        consistency_penalty = 0.0
        
        if len(self._turn_history) >= self.window_size:
            turns = np.array(self._turn_history)
            
            # Mean absolute turn rate
            mean_abs_turn = np.mean(np.abs(turns))
            
            # Check if sustained high turn rate
            if mean_abs_turn > self.turn_threshold:
                # Penalty scales with how much above threshold
                excess = mean_abs_turn - self.turn_threshold
                mean_penalty = self.mean_coef * (excess ** 2)
                
                # Consistency check: are we turning in the same direction?
                mean_turn = np.mean(turns)
                # Ratio of |mean| to mean(|x|) is 1.0 for perfectly consistent direction
                if mean_abs_turn > 1e-6:
                    consistency_ratio = abs(mean_turn) / mean_abs_turn
                    if consistency_ratio > self.consistency_threshold:
                        # Extra penalty for always turning same direction
                        consistency_penalty = self.consistency_coef * (consistency_ratio ** 2)
        
        # Execute action in underlying env
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        # Apply penalties
        total_penalty = mean_penalty + consistency_penalty
        rew = float(rew) - total_penalty
        
        # Logging for debugging
        info["pen_sustained_turn_mean"] = float(mean_penalty)
        info["pen_sustained_turn_consistency"] = float(consistency_penalty)
        info["pen_sustained_turn_total"] = float(total_penalty)
        info["sustained_turn_mean_coef"] = float(self.mean_coef)
        info["sustained_turn_consistency_coef"] = float(self.consistency_coef)
        
        # Diagnostic: what's the current mean turn rate?
        if len(self._turn_history) >= self.window_size:
            info["diag_mean_abs_turn"] = float(np.mean(np.abs(list(self._turn_history))))
            info["diag_mean_turn"] = float(np.mean(list(self._turn_history)))
        
        return obs, rew, terminated, truncated, info


class SustainedTurnPenaltySimple(gym.Wrapper):
    """
    Simpler version: Just penalizes when mean |turn| over window exceeds threshold.
    No annealing - constant coefficient.
    
    Good for quick experiments to see if this helps at all.
    """
    
    def __init__(
        self,
        env,
        hdg_idx: int = 0,
        window_size: int = 15,
        turn_threshold: float = 0.5,
        penalty_coef: float = 0.01,
    ):
        super().__init__(env)
        self.hdg_idx = int(hdg_idx)
        self.window_size = int(window_size)
        self.turn_threshold = float(turn_threshold)
        self.penalty_coef = float(penalty_coef)
        self._turn_history = deque(maxlen=window_size)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._turn_history.clear()
        return obs, info
    
    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        turn_cmd = float(a[self.hdg_idx])
        self._turn_history.append(turn_cmd)
        
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        penalty = 0.0
        if len(self._turn_history) >= self.window_size:
            mean_abs_turn = np.mean(np.abs(list(self._turn_history)))
            if mean_abs_turn > self.turn_threshold:
                excess = mean_abs_turn - self.turn_threshold
                penalty = self.penalty_coef * (excess ** 2)
                rew = float(rew) - penalty
        
        info["pen_sustained_spin"] = float(penalty)
        return obs, rew, terminated, truncated, info
