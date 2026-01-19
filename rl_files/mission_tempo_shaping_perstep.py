"""
mission_tempo_shaping_perstep.py

Per-step mission tempo shaping (denser signal than every-N-steps version).

Rewards:
- While red is alive: penalty per step (creates urgency)
- After red is killed: bonus per step (rewards success)
"""

import gymnasium as gym


class MissionTempoShapingPerStep(gym.Wrapper):
    """
    Per-step mission tempo shaping.
    
    While red alive: -penalty_per_step each step
    After red dead:  +bonus_per_step each step (if HVAA alive)
    
    Example magnitudes (for 13,500 step episodes):
        penalty_per_step = -0.005  → -67.5 total if red never killed
        bonus_per_step   = +0.001  → +13.5 total if red killed at step 0
        
    If red killed at step 5000:
        penalty = 5000 * -0.005 = -25.0
        bonus   = 8500 * +0.001 = +8.5
        net tempo = -16.5
        
    If red killed at step 1000:
        penalty = 1000 * -0.005 = -5.0
        bonus   = 12500 * +0.001 = +12.5
        net tempo = +7.5  ← Faster kill = positive tempo!
    
    Args:
        env: The environment to wrap
        penalty_per_step: Penalty each step while red is alive (negative value)
        bonus_per_step: Bonus each step after red is killed (positive value)
        require_hvaa_alive: Only give bonus if HVAA is still alive
    """
    def __init__(
        self,
        env,
        penalty_per_step: float = -0.005,
        bonus_per_step: float = 0.001,
        require_hvaa_alive: bool = True,
    ):
        super().__init__(env)
        self.penalty_per_step = float(penalty_per_step)
        self.bonus_per_step = float(bonus_per_step)
        self.require_hvaa_alive = require_hvaa_alive
        
        self._red_killed = False
        self._kill_step = None
        self._step_count = 0
        self._total_penalty = 0.0
        self._total_bonus = 0.0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._red_killed = False
        self._kill_step = None
        self._step_count = 0
        self._total_penalty = 0.0
        self._total_bonus = 0.0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._step_count += 1
        
        # Check for kill event (latching)
        if not self._red_killed and info.get("red_killed_event", False):
            self._red_killed = True
            self._kill_step = self._step_count
        
        shaping = 0.0
        
        if not self._red_killed:
            # Red alive: apply penalty
            shaping = self.penalty_per_step
            self._total_penalty += abs(shaping)
        else:
            # Red dead: apply bonus (if HVAA ok)
            hvaa_ok = True
            if self.require_hvaa_alive:
                hvaa_ok = info.get("hvaa_alive", True)
            
            if hvaa_ok:
                shaping = self.bonus_per_step
                self._total_bonus += shaping
        
        reward = float(reward) + shaping
        
        # Info for logging
        info["tempo_shaping"] = float(shaping)
        info["tempo_penalty_total"] = float(self._total_penalty)
        info["tempo_bonus_total"] = float(self._total_bonus)
        info["tempo_net"] = float(self._total_bonus - self._total_penalty)
        info["red_killed"] = bool(self._red_killed)
        info["kill_step"] = self._kill_step if self._kill_step else 0
        info["steps_red_alive"] = self._kill_step if self._kill_step else self._step_count
        
        return obs, reward, terminated, truncated, info


class MissionTempoShapingScheduled(gym.Wrapper):
    """
    Per-step tempo shaping with optional scheduling.
    
    Can start with lower penalties and ramp up as training progresses.
    This allows early exploration without harsh penalties, then enforces
    discipline later in training.
    
    Args:
        env: The environment to wrap
        penalty_start: Initial penalty per step (e.g., -0.001)
        penalty_end: Final penalty per step (e.g., -0.005)
        bonus_per_step: Bonus per step after kill
        anneal_steps: Steps over which to anneal penalty
        hold_steps: Steps to hold at start value before annealing
    """
    def __init__(
        self,
        env,
        penalty_start: float = -0.001,
        penalty_end: float = -0.005,
        bonus_per_step: float = 0.001,
        anneal_steps: int = 3_000_000,
        hold_steps: int = 500_000,
        require_hvaa_alive: bool = True,
    ):
        super().__init__(env)
        self.penalty_start = float(penalty_start)
        self.penalty_end = float(penalty_end)
        self.bonus_per_step = float(bonus_per_step)
        self.anneal_steps = int(anneal_steps)
        self.hold_steps = int(hold_steps)
        self.require_hvaa_alive = require_hvaa_alive
        
        self._global_step = 0
        self._red_killed = False
        self._step_count = 0
        self._total_penalty = 0.0
        self._total_bonus = 0.0
        self._current_penalty = penalty_start
        
    def set_global_step(self, t: int):
        """Called by training loop to update global step."""
        self._global_step = int(t)
        # Propagate to wrapped env if it has this method
        if hasattr(self.env, "set_global_step"):
            self.env.set_global_step(t)
    
    def _get_current_penalty(self):
        """Calculate penalty based on schedule."""
        t = self._global_step
        if t <= self.hold_steps:
            return self.penalty_start
        if self.anneal_steps <= 0:
            return self.penalty_end
        progress = min(1.0, (t - self.hold_steps) / self.anneal_steps)
        return self.penalty_start + progress * (self.penalty_end - self.penalty_start)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._red_killed = False
        self._step_count = 0
        self._total_penalty = 0.0
        self._total_bonus = 0.0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._step_count += 1
        self._current_penalty = self._get_current_penalty()
        
        # Check for kill
        if not self._red_killed and info.get("red_killed_event", False):
            self._red_killed = True
        
        shaping = 0.0
        
        if not self._red_killed:
            shaping = self._current_penalty
            self._total_penalty += abs(shaping)
        else:
            hvaa_ok = info.get("hvaa_alive", True) if self.require_hvaa_alive else True
            if hvaa_ok:
                shaping = self.bonus_per_step
                self._total_bonus += shaping
        
        reward = float(reward) + shaping
        
        info["tempo_shaping"] = float(shaping)
        info["tempo_penalty_total"] = float(self._total_penalty)
        info["tempo_bonus_total"] = float(self._total_bonus)
        info["tempo_penalty_coef"] = float(self._current_penalty)
        info["red_killed"] = bool(self._red_killed)
        
        return obs, reward, terminated, truncated, info


# ============================================================================
# Quick reference
# ============================================================================
"""
REWARD ANALYSIS (13,500 step episode):

Scenario A: Kill at step 1,000
  Penalty: 1,000 × -0.005 = -5.0
  Bonus:   12,500 × 0.001 = +12.5
  Net tempo: +7.5 ✓ Fast kill rewarded!

Scenario B: Kill at step 6,750 (midpoint)
  Penalty: 6,750 × -0.005 = -33.75
  Bonus:   6,750 × 0.001 = +6.75
  Net tempo: -27.0

Scenario C: Kill at step 12,000 (late)
  Penalty: 12,000 × -0.005 = -60.0
  Bonus:   1,500 × 0.001 = +1.5
  Net tempo: -58.5 ✗ Slow kill penalized

Scenario D: No kill
  Penalty: 13,500 × -0.005 = -67.5
  Bonus:   0
  Net tempo: -67.5 ✗ Worst case

The crossover point where tempo becomes positive:
  penalty_steps × 0.005 = bonus_steps × 0.001
  Let k = kill_step, T = 13500
  k × 0.005 = (T - k) × 0.001
  0.005k = 0.001T - 0.001k
  0.006k = 0.001T
  k = T/6 = 2,250 steps

So kills before step 2,250 give positive tempo reward!

INTEGRATION:
```python
from mission_tempo_shaping_perstep import MissionTempoShapingPerStep

e = MissionTempoShapingPerStep(
    e,
    penalty_per_step=-0.005,  # -67.5 max penalty
    bonus_per_step=0.001,     # +13.5 max bonus
    require_hvaa_alive=True,
)
```

Or with scheduling (gentler early training):
```python
from mission_tempo_shaping_perstep import MissionTempoShapingScheduled

e = MissionTempoShapingScheduled(
    e,
    penalty_start=-0.001,     # Start gentle
    penalty_end=-0.005,       # End at full strength
    bonus_per_step=0.001,
    anneal_steps=3_000_000,   # Ramp up over 3M steps
    hold_steps=500_000,       # Stay gentle for first 500k
)
```
"""
