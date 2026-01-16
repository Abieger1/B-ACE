#!/usr/bin/env python3
"""
optuna_optimize_bace.py

Optuna hyperparameter optimization for B-ACE PPO training.

Optimizes:
- Initial and final learning rate
- Initial and final entropy coefficient  
- Initial and final clip range
- Value loss coefficient
- Number of epochs
- Mini-batch size
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Callable
import warnings

import numpy as np
import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Suppress Optuna warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

############################################
# Make sure we can import b_ace_py
############################################
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent.parent  
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from b_ace_py.utils import load_b_ace_config
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper


############################################
# Decay schedule functions
############################################

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


def exponential_schedule(initial_value: float, final_value: float, decay_rate: float = 0.1) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        progress_made = 1 - progress_remaining
        decay_factor = np.exp(-decay_rate * progress_made * 10)
        return final_value + (initial_value - final_value) * decay_factor
    return schedule


############################################
# Callbacks
############################################

class DecayMonitorCallback(BaseCallback):
    """Update entropy coefficient during training."""
    def __init__(self, entropy_schedule=None, verbose=0):
        super().__init__(verbose)
        self.entropy_schedule = entropy_schedule
        
    def _on_step(self) -> bool:
        if self.entropy_schedule is not None:
            progress_remaining = 1 - (self.num_timesteps / self.model._total_timesteps)
            new_ent_coef = self.entropy_schedule(progress_remaining)
            self.model.ent_coef = new_ent_coef
        return True


class OptunaPruningCallback(BaseCallback):
    """
    Callback for Optuna pruning based on episode reward.
    Reports intermediate values to Optuna and handles pruning.
    """
    def __init__(self, trial: optuna.Trial, eval_freq: int = 10000, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_freq = eval_freq
        self.last_mean_reward = -np.inf
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Collect episode rewards from info
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
        
        # Report to Optuna every eval_freq steps
        if self.n_calls % self.eval_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
            self.last_mean_reward = mean_reward
            
            # Report intermediate value
            self.trial.report(mean_reward, self.n_calls)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                raise optuna.TrialPruned()
                
        return True


############################################
# Helper functions
############################################

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
            return binary.as_posix()

    raise FileNotFoundError("Could not find Godot binary.")


############################################
# Single-agent wrapper (simplified)
############################################

class SingleAgentBACEEnv(gym.Env):
    """Simplified wrapper - same as in your training script."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, bace_config: Dict[str, Any], controlled_agent_id: str = "agent_0", max_episode_steps: int = 36000):
        super().__init__()
        self.controlled_agent_id = controlled_agent_id
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._ma_env = B_ACE_GodotPettingZooWrapper(device="cpu", **bace_config)
        
        if hasattr(self._ma_env, "agents") and self._ma_env.agents:
            if self.controlled_agent_id not in self._ma_env.agents:
                self.controlled_agent_id = self._ma_env.agents[0]
        
        self._lazy_obs_space = False
        ma_obs_space = self._ma_env.observation_space
        self.observation_space = None
        ma_act_space = self._ma_env.action_space
        self.action_space = None
        
        if isinstance(ma_obs_space, gym.spaces.Dict):
            if "obs" in ma_obs_space.spaces:
                self.observation_space = ma_obs_space.spaces["obs"]
            elif self.controlled_agent_id in ma_obs_space.spaces:
                candidate = ma_obs_space.spaces[self.controlled_agent_id]
                if isinstance(candidate, gym.spaces.Dict) and "obs" in candidate.spaces:
                    self.observation_space = candidate.spaces["obs"]
                else:
                    self.observation_space = candidate
            else:
                self._lazy_obs_space = True
        elif isinstance(ma_obs_space, gym.spaces.Box):
            self.observation_space = ma_obs_space
        
        if isinstance(ma_act_space, gym.spaces.Dict):
            if "input" in ma_act_space.spaces:
                self.action_space = ma_act_space.spaces["input"]
            else:
                raise RuntimeError("Expected 'input' key in Dict action_space")
        elif isinstance(ma_act_space, gym.spaces.Box):
            self.action_space = ma_act_space
        
        self._last_obs_dict = None

    def _unwrap_obs_payload(self, payload: Any) -> Any:
        current = payload
        visited = 0
        while isinstance(current, dict):
            visited += 1
            if visited > 10:
                raise RuntimeError(f"Observation payload nesting too deep")
            current = {k: v for k, v in current.items() if k not in {"mask", "info"}}
            if not current:
                break
            if "obs" in current:
                current = current["obs"]
                continue
            array_like_key = next((k for k, v in current.items() if isinstance(v, (list, tuple, np.ndarray))), None)
            if array_like_key is not None:
                current = current[array_like_key]
                continue
            if all(np.isscalar(v) for v in current.values()):
                return [current[k] for k in sorted(current.keys())]
            raise RuntimeError(f"Unable to unwrap observation payload")
        return current

    def _extract_single_obs(self, obs_src: Any) -> np.ndarray:
        if isinstance(obs_src, dict):
            if self.controlled_agent_id in obs_src:
                obs_raw = self._unwrap_obs_payload(obs_src[self.controlled_agent_id])
                return np.asarray(obs_raw, dtype=np.float32)
            if "obs" in obs_src:
                obs_payload = self._unwrap_obs_payload(obs_src["obs"])
                return np.asarray(obs_payload, dtype=np.float32)
            if len(obs_src) == 1:
                only_key = next(iter(obs_src))
                maybe = self._unwrap_obs_payload(obs_src[only_key])
                return np.asarray(maybe, dtype=np.float32)
            if self.observation_space is not None:
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            raise RuntimeError(f"Unexpected obs structure")
        return np.asarray(obs_src, dtype=np.float32)

    def _build_action_payload(self, agent_action: np.ndarray) -> Any:
        if hasattr(self._ma_env, "agents"):
            act_dict = {}
            agents_iterable = list(self._ma_env.agents)
            if not agents_iterable and hasattr(self._ma_env, "possible_agents"):
                agents_iterable = list(self._ma_env.possible_agents)
            for aid in agents_iterable:
                if aid == self.controlled_agent_id:
                    clipped = np.clip(agent_action, self.action_space.low, self.action_space.high).astype(np.float32)
                    act_dict[aid] = clipped
                else:
                    act_dict[aid] = np.zeros(self.action_space.shape, dtype=np.float32)
            return act_dict
        else:
            return np.clip(agent_action, self.action_space.low, self.action_space.high).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._elapsed_steps = 0
        reset_out = self._ma_env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs_like, info = reset_out
        else:
            obs_like = reset_out
            info = {}
        self._last_obs_dict = obs_like
        single_obs = self._extract_single_obs(obs_like)
        if self._lazy_obs_space or self.observation_space is None:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=single_obs.shape, dtype=np.float32)
            self._lazy_obs_space = False
        return single_obs, info

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1
        act_payload = self._build_action_payload(action)
        step_out = self._ma_env.step(act_payload)
        o0, o1, o2, o3, o4 = step_out
        if isinstance(o1, dict):
            obs = self._extract_single_obs(o0)
            reward = float(o1.get(self.controlled_agent_id, 0.0))
            terminated = bool(o2.get(self.controlled_agent_id, False)) or bool(o2.get("__all__", False))
            truncated = bool(o3.get(self.controlled_agent_id, False)) or bool(o3.get("__all__", False))
            info = o4.get(self.controlled_agent_id, {})
        else:
            obs = self._extract_single_obs(o0)
            reward = float(o1)
            terminated = bool(o2)
            truncated = bool(o3)
            info = o4
        if self._elapsed_steps >= self._max_episode_steps and not terminated:
            truncated = True
        self._last_obs_dict = o0
        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self._ma_env, "close"):
            self._ma_env.close()


############################################
# Optuna objective function
############################################

def objective(trial: optuna.Trial, config_path: Path, total_timesteps: int, n_eval_episodes: int = 10) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Returns:
        Mean episode reward over evaluation episodes.
    """
    
    # Sample hyperparameters
    initial_lr = trial.suggest_float("initial_lr", 1e-4, 1e-2, log=True)
    final_lr = trial.suggest_float("final_lr", 1e-5, initial_lr, log=True)
    
    initial_entropy = trial.suggest_float("initial_entropy", 1e-3, 5e-2)
    final_entropy = trial.suggest_float("final_entropy", 0.0, initial_entropy)
    
    # Constant clip range (not decaying - PPO best practice)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    
    vf_coef = trial.suggest_float("vf_coef", 0.3, 0.9)
    n_epochs = trial.suggest_categorical("n_epochs", [4, 6, 8, 10, 12])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    
    # Suggest decay schedule type
    decay_schedule = trial.suggest_categorical("decay_schedule", ["linear", "exponential"])
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number} - Testing hyperparameters:")
    print(f"{'='*70}")
    print(f"Initial LR: {initial_lr:.2e} → Final LR: {final_lr:.2e}")
    print(f"Initial Entropy: {initial_entropy:.4f} → Final Entropy: {final_entropy:.4f}")
    print(f"Clip Range: {clip_range:.1f} (constant)")
    print(f"VF Coef: {vf_coef:.3f}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Decay Schedule: {decay_schedule}")
    print(f"{'='*70}\n")
    
    # Load B-ACE config
    B_ACE_config = load_b_ace_config(config_path.as_posix())
    if B_ACE_config is None:
        raise RuntimeError(f"Failed to load config file at {config_path}")
    
    env_cfg = B_ACE_config.setdefault("EnvConfig", {})
    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = _resolve_env_path()
    
    env_path_value = Path(env_cfg["env_path"]).expanduser().resolve()
    if not env_path_value.exists():
        env_path_value = Path(_resolve_env_path()).resolve()
    env_cfg["env_path"] = env_path_value.as_posix()
    
    agents_cfg = B_ACE_config.setdefault("AgentsConfig", {})
    blue_cfg = agents_cfg.setdefault("blue_agents", {})
    red_cfg = agents_cfg.setdefault("red_agents", {})
    blue_cfg.setdefault("base_behavior", "external")
    red_cfg.setdefault("base_behavior", "baseline1")
    
    env_cfg.setdefault("renderize", 0)  # No rendering during optimization
    env_cfg.setdefault("speed_up", 200)
    
    # Create log directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "optuna_trials" / f"trial_{trial.number}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    def make_env():
        e = SingleAgentBACEEnv(
            bace_config=B_ACE_config,
            controlled_agent_id="agent_0",
            max_episode_steps=8000,
        )
        e = Monitor(e, filename=str(log_dir / "monitor.csv"))
        return e
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Create decay schedules
    if decay_schedule == "linear":
        lr_schedule = linear_schedule(initial_lr, final_lr)
        ent_schedule = linear_schedule(initial_entropy, final_entropy)
    else:
        lr_schedule = exponential_schedule(initial_lr, final_lr)
        ent_schedule = exponential_schedule(initial_entropy, final_entropy)
    
    # Note: clip_range is constant (not decaying), so no schedule needed
    
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=2048,  # Fixed for stability
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,  # Constant value (not decaying)
        ent_coef=initial_entropy,
        vf_coef=vf_coef,
        max_grad_norm=0.5,
        verbose=0,
        device="cpu",
        policy_kwargs=policy_kwargs
    )
    
    # Create callbacks
    decay_callback = DecayMonitorCallback(entropy_schedule=ent_schedule)
    pruning_callback = OptunaPruningCallback(trial, eval_freq=10000)
    
    try:
        # Train model
        model.learn(
            total_timesteps=total_timesteps,
            callback=[decay_callback, pruning_callback],
            progress_bar=False,
        )
        
        # Evaluate model
        episode_rewards = []
        obs = vec_env.reset()
        for _ in range(n_eval_episodes):
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                # Get unnormalized reward if available
                if hasattr(vec_env, 'get_original_reward'):
                    true_reward = vec_env.get_original_reward()
                else:
                    true_reward = reward[0]
                    

                episode_reward += reward[0]
                if done:
                    episode_rewards.append(episode_reward)
                    obs = vec_env.reset()
                    break
        
        mean_reward = np.mean(episode_rewards)
        
        print(f"\nTrial {trial.number} completed: Mean Reward = {mean_reward:.2f}\n")
        
    except optuna.TrialPruned:
        print(f"\nTrial {trial.number} pruned.\n")
        raise
    
    finally:
        vec_env.close()
    
    return mean_reward


############################################
# Main optimization
############################################

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for B-ACE PPO")
    parser.add_argument("--config", type=str, 
                       default=(REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json").as_posix(),
                       help="Path to B-ACE config JSON")
    parser.add_argument("--n-trials", type=int, default=150,
                       help="Number of Optuna trials to run")
    parser.add_argument("--timesteps-per-trial", type=int, default=2_000_000,
                       help="Timesteps per trial (shorter for faster optimization)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                       help="Number of episodes for final evaluation")
    parser.add_argument("--study-name", type=str, default="bace_ppo_optimization",
                       help="Name for Optuna study")
    parser.add_argument("--storage", type=str, default=None,
                       help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel jobs (careful with simulation resources!)")
    
    args = parser.parse_args()
    
    config_path = Path(args.config).expanduser().resolve()
    
    # Create Optuna study
    if args.storage:
        storage = args.storage
    else:
        storage = f"sqlite:///{REPO_ROOT}/optuna_studies.db"
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50000),
    )
    
    print(f"\n{'='*70}")
    print(f"Starting Optuna Optimization")
    print(f"{'='*70}")
    print(f"Study Name: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Timesteps per Trial: {args.timesteps_per_trial:,}")
    print(f"Evaluation Episodes: {args.n_eval_episodes}")
    print(f"Parallel Jobs: {args.n_jobs}")
    print(f"{'='*70}\n")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config_path, args.timesteps_per_trial, args.n_eval_episodes),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Optimization Complete!")
    print(f"{'='*70}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (mean reward): {study.best_value:.2f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    # Save results
    results_file = REPO_ROOT / "optuna_best_params.json"
    with open(results_file, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best parameters saved to: {results_file}")


if __name__ == "__main__":
    main()
