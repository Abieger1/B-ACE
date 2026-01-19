"""
reward_breakdown_logger.py

Diagnostic wrapper that tracks all reward components and prints a summary
at the end of each episode (or every N episodes).

Place this as the OUTERMOST wrapper (just before Monitor) to capture all shaping.
"""

import gymnasium as gym
import numpy as np
from collections import defaultdict


class RewardBreakdownLogger(gym.Wrapper):
    """
    Tracks reward components from info dict and prints episode summaries.
    
    This wrapper looks for specific keys in the info dict that other wrappers
    add, and accumulates them over the episode. At episode end, it prints
    a breakdown showing where the reward came from.
    
    Args:
        env: The environment to wrap
        print_every_n: Print breakdown every N episodes (1 = every episode)
        reward_keys: List of info keys to track. If None, uses default set.
        verbose: If True, prints every episode. If False, only prints every N.
    """
    
    # Default keys to look for (add your wrapper's keys here)
    DEFAULT_KEYS = [
        # Track-aware shaping
        "track_aware_shaping",
        "track_shaping", 
        "range_shaping",
        "track_loss_count",
        "close_reward_total",
        "open_penalty_total",
        
        # Mission tempo shaping
        "tempo_shaping",
        "tempo_penalty",
        "tempo_bonus",
        "tempo_penalty_total",
        "tempo_bonus_total",
        "steps_red_alive",
        
        # Original range closing (if still used)
        "range_close_shaping",
        
        # HVAA survival
        "hvaa_survival_bonus",
        
        # Kill reward
        "red_killed_event",
        
        # Turn penalties
        "pen_turn_total",
        "pen_turn_jerk",
        "pen_turn_mag",
        
        # Sustained turn
        "sustained_turn_penalty",
        
        # Expert alignment (if used)
        "expert_alignment_reward",
        
        # HVAA destruction (if used)
        "hvaa_destruction_penalty",
    ]
    
    def __init__(
        self,
        env,
        print_every_n: int = 1,
        reward_keys: list = None,
        verbose: bool = True,
        print_width: int = 60,
    ):
        super().__init__(env)
        self.print_every_n = max(1, int(print_every_n))
        self.reward_keys = reward_keys or self.DEFAULT_KEYS
        self.verbose = verbose
        self.print_width = print_width
        
        self._episode_count = 0
        self._reset_accumulators()
        
    def _reset_accumulators(self):
        """Reset all episode tracking."""
        self._step_count = 0
        self._total_reward = 0.0
        self._component_sums = defaultdict(float)
        self._component_counts = defaultdict(int)
        self._events = defaultdict(int)  # For counting events like kills
        self._red_killed = False
        self._kill_step = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_accumulators()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._step_count += 1
        self._total_reward += float(reward)
        
        # Track components from info
        for key in self.reward_keys:
            if key in info:
                val = info[key]
                if isinstance(val, bool):
                    if val:
                        self._events[key] += 1
                        if key == "red_killed_event" and not self._red_killed:
                            self._red_killed = True
                            self._kill_step = self._step_count
                elif isinstance(val, (int, float)):
                    self._component_sums[key] += float(val)
                    self._component_counts[key] += 1
        
        # Print breakdown at episode end
        if terminated or truncated:
            self._episode_count += 1
            if self.verbose and (self._episode_count % self.print_every_n == 0):
                self._print_breakdown(info)
        
        return obs, reward, terminated, truncated, info
    
    def _print_breakdown(self, final_info):
        """Print episode reward breakdown."""
        w = self.print_width
        
        print("\n" + "=" * w)
        print(f"📊 EPISODE {self._episode_count} REWARD BREAKDOWN")
        print("=" * w)
        print(f"Steps: {self._step_count:,}")
        print(f"Total Reward: {self._total_reward:.2f}")
        if self._red_killed:
            print(f"🎯 Red Killed at step {self._kill_step}")
        print("-" * w)
        
        # Group components by category
        categories = {
            "Track-Aware Shaping": [
                ("track_shaping", "Track maintenance"),
                ("range_shaping", "Range shaping"),
                ("track_aware_shaping", "Combined (per-step)"),
                ("close_reward_total", "Close rewards (cumulative)"),
                ("open_penalty_total", "Open penalties (cumulative)"),
                ("track_loss_count", "Track losses (count)"),
            ],
            "Mission Tempo": [
                ("tempo_penalty", "Red-alive penalty (per-step)"),
                ("tempo_bonus", "Post-kill bonus (per-step)"),
                ("tempo_penalty_total", "Penalty total"),
                ("tempo_bonus_total", "Bonus total"),
                ("steps_red_alive", "Steps with red alive"),
            ],
            "Movement Penalties": [
                ("pen_turn_total", "Turn penalty total"),
                ("pen_turn_jerk", "Jerk penalty"),
                ("pen_turn_mag", "Magnitude penalty"),
                ("sustained_turn_penalty", "Sustained turn"),
            ],
            "Mission Rewards": [
                ("hvaa_survival_bonus", "HVAA survival"),
                ("range_close_shaping", "Range closing (old)"),
            ],
            "Events": [
                ("red_killed_event", "Red kills"),
            ],
        }
        
        for category, keys in categories.items():
            has_data = False
            lines = []
            
            for key, label in keys:
                if key in self._component_sums and self._component_sums[key] != 0:
                    has_data = True
                    total = self._component_sums[key]
                    count = self._component_counts[key]
                    avg = total / count if count > 0 else 0
                    lines.append(f"  {label:30s} sum={total:+8.2f}  (avg={avg:+.4f}/step)")
                elif key in self._events and self._events[key] > 0:
                    has_data = True
                    lines.append(f"  {label:30s} count={self._events[key]}")
            
            if has_data:
                print(f"\n[{category}]")
                for line in lines:
                    print(line)
        
        # Summary
        print("\n" + "-" * w)
        print("COMPONENT TOTALS:")
        
        # Calculate major components
        track_aware = self._component_sums.get("close_reward_total", 0) - self._component_sums.get("open_penalty_total", 0)
        track_loss_penalty = self._component_sums.get("track_loss_count", 0) * -0.5  # Assuming -0.5 per loss
        tempo_net = self._component_sums.get("tempo_bonus_total", 0) - self._component_sums.get("tempo_penalty_total", 0)
        turn_penalty = -self._component_sums.get("pen_turn_total", 0)  # Usually tracked as positive penalty
        hvaa_bonus = self._component_sums.get("hvaa_survival_bonus", 0)
        kill_bonus = 12.0 if self._red_killed else 0.0  # Assuming +12 kill reward
        
        print(f"  Track-aware (close - open):  {track_aware:+.2f}")
        print(f"  Tempo (bonus - penalty):     {tempo_net:+.2f}")
        print(f"  Turn penalties:              {turn_penalty:+.2f}")
        print(f"  HVAA survival bonus:         {hvaa_bonus:+.2f}")
        print(f"  Kill reward:                 {kill_bonus:+.2f}")
        
        estimated = track_aware + tempo_net + turn_penalty + hvaa_bonus + kill_bonus
        print(f"\n  Estimated from components:   {estimated:+.2f}")
        print(f"  Actual total reward:         {self._total_reward:+.2f}")
        print(f"  Unaccounted:                 {self._total_reward - estimated:+.2f}")
        
        print("=" * w + "\n")


class CompactRewardLogger(gym.Wrapper):
    """
    Lighter-weight version that just prints one line per episode.
    """
    def __init__(self, env, print_every_n: int = 1):
        super().__init__(env)
        self.print_every_n = max(1, int(print_every_n))
        self._episode_count = 0
        self._reset()
        
    def _reset(self):
        self._steps = 0
        self._total = 0.0
        self._track_loss = 0
        self._tempo_penalty = 0.0
        self._tempo_bonus = 0.0
        self._close_rew = 0.0
        self._open_pen = 0.0
        self._killed = False
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._steps += 1
        self._total += float(reward)
        
        # Quick extraction
        self._track_loss = info.get("track_loss_count", self._track_loss)
        self._tempo_penalty = info.get("tempo_penalty_total", self._tempo_penalty)
        self._tempo_bonus = info.get("tempo_bonus_total", self._tempo_bonus)
        self._close_rew = info.get("close_reward_total", self._close_rew)
        self._open_pen = info.get("open_penalty_total", self._open_pen)
        
        if info.get("red_killed_event", False):
            self._killed = True
        
        if terminated or truncated:
            self._episode_count += 1
            if self._episode_count % self.print_every_n == 0:
                kill_str = "🎯" if self._killed else "  "
                print(f"[EP {self._episode_count:4d}] {kill_str} R={self._total:+7.1f} | "
                      f"tempo={-self._tempo_penalty + self._tempo_bonus:+6.1f} | "
                      f"range={self._close_rew - self._open_pen:+6.1f} | "
                      f"trk_loss={self._track_loss:2d} | "
                      f"steps={self._steps}")
        
        return obs, reward, terminated, truncated, info


# ============================================================================
# Integration
# ============================================================================
"""
Add as the LAST wrapper before Monitor:

```python
from reward_breakdown_logger import RewardBreakdownLogger, CompactRewardLogger

# ... all your other wrappers ...

# Option A: Detailed breakdown every episode
e = RewardBreakdownLogger(e, print_every_n=1, verbose=True)

# Option B: Compact one-liner every episode
e = CompactRewardLogger(e, print_every_n=1)

# Option C: Only every 10 episodes (less spam during training)
e = RewardBreakdownLogger(e, print_every_n=10)

e = Monitor(e, ...)  # Monitor should be outermost
```

The wrapper order matters - RewardBreakdownLogger should see the FINAL reward
after all shaping, so it goes just inside Monitor.
"""
