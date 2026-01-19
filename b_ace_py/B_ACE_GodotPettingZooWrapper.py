from .godot_env import GodotEnv
from pettingzoo.utils import ParallelEnv

from .utils import ActionSpaceProcessor, convert_macos_path
import numpy as np
import atexit
from sys import platform
from gymnasium import spaces
import random


class B_ACE_GodotPettingZooWrapper(GodotEnv, ParallelEnv):
    metadata = {'render.modes': [], 'name': "godot_rl_multi_agent"}

    def __init__(self,
                env_path: str = None,                
                seed: int = 0,         
                convert_action_space: bool = False,
                device: str = "cpu",
                **config_kwargs):                               
        
        self.device = device
        
        self.env_config = config_kwargs.get("EnvConfig", "")     
        #Godot Line Parameters Commands
        self.env_path       = self.env_config.get("env_path", "./bin/BVR.exe")  
        self.show_window    = int(self.env_config.get("renderize", 1))  
        self._seed          = int(self.env_config.get("seed", 1))  
        self.action_repeat  = int(self.env_config.get("action_repeat", 20))  
        self.action_type    = self.env_config.get("action_type", "Low_Level_Continuous")  
        self.speedup        = int(self.env_config.get("speed_up", 1000))                          
        self.parallel_envs  = int(self.env_config.get("parallel_envs", 1))   
        
        self.agents_config = config_kwargs.get("AgentsConfig", "")
        self._num_agents = int(self.agents_config["blue_agents"].get("num_agents", 1))
        
        self.share_states  = int(self.agents_config["blue_agents"].get("share_states", 1))
        self.share_tracks =  int(self.agents_config["blue_agents"].get("share_tracks", 1))       
        
        self.additional_config = self.env_config.get("additional_config", "") 
        
        # FIX 1: Use port from config, don't randomize it!
        self.port = int(self.env_config.get("port", 11008))
       # print(f"DEBUG: Using port {self.port}")
        
        self.proc = None
        
        if self.env_path is not None and self.env_path != "debug":
            self.env_path = self._set_platform_suffix(self.env_path)

            self.check_platform(self.env_path)  

            # FIX 2: Pass port to Godot via command line
            #print(f"DEBUG: Launching Godot with port {self.port}")
            self._launch_env(self.env_path, self.port, self.show_window == 1, None, self._seed, self.action_repeat, self.speedup)
        else:
            print("No game binary has been provided, please press PLAY in the Godot editor")
        
        self.host_binding = config_kwargs.get("host_binding", False)
        
        # FIX 3: Add debug output for connection
       # print(f"DEBUG: Starting server on port {self.port}...")
       # print("DEBUG PY: (about to listen) port =", self.port)
        self.connection = self._start_server()
       # print("DEBUG: Server started, connection established")
       # print("DEBUG PY: socket bound, waiting for Godot connect...")
        self.num_envs = None
        
        # FIX 4: Add debug output for handshake
        #print("DEBUG: Performing handshake...")
        self._handshake()
        #print("DEBUG: Handshake complete")
        
        self.action_spaces = []
        self.observation_spaces = []     
        
        # FIX 5: Add debug output before sending config
        #print("DEBUG: Sending simulation config to Godot...")
        #print(f"DEBUG: env_config keys: {list(self.env_config.keys())}")
       # print(f"DEBUG: agents_config keys: {list(self.agents_config.keys())}")
        self.send_sim_config(self.env_config, self.agents_config)
        #print("DEBUG: Config sent, waiting for environment info...")
        
        self.action_spaces = []
        self.observation_spaces = []

        env_info = self._get_env_info()
        #print(f"DEBUG: Received env_info: {env_info.keys()}")
        
        self.observation_labels = env_info["observation_labels"]
        # --- expose a flat label list for agent_0 so wrappers can print index->name ---
        first_key = str(101)  # agent_0 in your Godot labeling scheme
        self.observation_labels_flat = list(self.observation_labels[first_key])
        # ===== ONE-TIME OBSERVATION LABEL DUMP (DEBUG) =====
        if not hasattr(self, "_printed_observation_labels"):
            print("\n=== B-ACE Observation Labels (raw, from Godot) ===")
            for agent_id, labels in self.observation_labels.items():
                print(f"\nAgent key: {agent_id}")
                for i, label in enumerate(labels):
                    print(f"{i:>2}: {label}")
            print("=== End Observation Labels ===\n")
            self._printed_observation_labels = True
        # ==================================================        self.action_spaces = env_info["action_spaces"]
                
        self.tuple_action_spaces = [
            spaces.Tuple([v for _, v in action_space.items()]) for action_space in self.action_spaces
        ]
        # Single agent action space processor using the action space(s) of the first agent
        self.action_space_processor = ActionSpaceProcessor(self.tuple_action_spaces[0], convert_action_space)
                
        # For multi-policy envs: The name of each agent's policy set in the env itself (any training_mode
        # AIController instance is treated as an agent)
        self.agent_policy_names

        atexit.register(self._close)
                                
        #Initialization for PettingZoo Paralell
        self.agents = [f'agent_{i}' for i in range(self._num_agents)]  # Initialize agents
        self.possible_agents = self.agents[:]
        
        # Now we can initialize _prev_missiles_fired since possible_agents exists
        self._prev_missiles_fired = {a: 0 for a in self.possible_agents}
        
                        
        self.obs_map= {agent : {label: index for index, label in enumerate(self.observation_labels[str(101 + i)])}  for i,agent in enumerate(self.possible_agents)}               


    def get_hvaa_indices(self, agent_name: str):

        # Use the existing label list under key '101'
        labels = None
        if hasattr(self, "observation_labels"):
            labels = self.observation_labels.get("101", None)

        if labels is None:
            print("[get_hvaa_indices] WARNING: no labels for '101'; returning None")
            return None

        # Build label->index mapping
        label_to_idx = {name: idx for idx, name in enumerate(labels)}

        # Map the four HVAA-related names
        required = ["hvaa_dist", "hvaa_alt_diff", "hvaa_angle_off", "hvaa_detected"]
        missing = [r for r in required if r not in label_to_idx]
        if missing:
            print("[get_hvaa_indices] WARNING: missing HVAA labels:", missing)
            return None

        return {
            "hvaa_dist": label_to_idx["hvaa_dist"],
            "hvaa_alt_diff": label_to_idx["hvaa_alt_diff"],
            "hvaa_angle_off": label_to_idx["hvaa_angle_off"],
            "hvaa_detected": label_to_idx["hvaa_detected"],
        }


        self.agent_idx = [ {agent : i} for i, agent in enumerate(self.possible_agents)]                 
        
        self.observation_space = self.observation_spaces[0]["obs"]
        self.action_space = self.action_spaces[0]["input"]
        
        self._cumulative_rewards = {agent : 0  for agent in self.possible_agents}                
        self.rewards =  {agent : 0  for agent in self.possible_agents}
        self.terminations =  {agent : False  for agent in self.possible_agents}
        self.truncations =  {agent : False  for agent in self.possible_agents} 
        self.observations =  {agent : []  for agent in self.possible_agents}  
        self.info =  {agent : []  for agent in self.possible_agents}  
        self.infos =  {agent : []  for agent in self.possible_agents}
        
        #print("DEBUG: Wrapper initialization complete!")

        self._ep_step = 0
        self._printed_ep_end = False
                            
    def send_sim_config(self, _env_config, _agents_config):
        message = {"type": "config"}        
        message["agents_config"] = _agents_config
        message["env_config"] = _env_config
        
        # FIX 6: Add debug output
        #print(f"DEBUG: Sending config message: type={message['type']}")
        self._send_as_json(message)
        
        # FIX 7: Wait for acknowledgment
        #print("DEBUG: Waiting for config acknowledgment...")
        response = self._get_dict_json_message()
        #print(f"DEBUG: Received response: {response}")
        
        if response.get("type") != "ack":
            print(f"WARNING: Expected 'ack' response, got '{response.get('type')}'")
            
    
    def reset(self, seed=0, options=None):
        #print("DEBUG: Calling reset...")
        obs_dict, _info_from_base = super().reset()
        #print("DEBUG: Reset complete, processing observations...")

        # obs_dict right now looks like:
        # { "agent_0": [ ... ], "agent_1": [ ... ], ... }

        self.observations = {}
        self.info = {}

        for agent_name in self.possible_agents:
            if agent_name not in obs_dict:
                print(f"WARNING: {agent_name} not in obs_dict from Godot; filling zeros")
                self.observations[agent_name] = {
                    "obs": [],
                    "mask": [True for _ in range(4)]
                }
                self.info[agent_name] = {}
                continue

            raw_obs = obs_dict[agent_name]

            # Wrap it like a policy would expect
            self.observations[agent_name] = {
                "obs": raw_obs,
                "mask": [True for _ in range(4)]
            }

            self.info[agent_name] = {}

        # ParallelEnv.reset() should return (observations, infos)
        self._ep_step = 0
        self._printed_ep_end = False

        return self.observations, self.info
    

    
    
    def _observation_space(self, agent):        
        return self.observation_spaces[agent]
    
    def action_space(self, agent = None):        
        return self.action_space_processor.action_space
    
    def seed(self, _seed):
        self.seed = _seed
    
    def step(self, actions, order_ij=True):
        if self.action_type == "Low_Level_Continuous":
            # Build list: per-agent -> [ per-head payloads ]
            # Single head "input" => wrap the 4-vector in a list
            godot_actions = []
            for agent_name in self.possible_agents:
                a = np.asarray(actions[agent_name], dtype=np.float32)  # shape (4,)
                godot_actions.append([a])  # <-- NOTE the extra [ ... ] for the head

        elif self.action_type == "Low_Level_Discrete":
            ordered = []
            for agent_name in self.possible_agents:
                ordered.append(self.decode_action(actions[agent_name]))
            # For discrete if the action space has a single head, you should ALSO wrap once:
            godot_actions = [[np.asarray(ordered[i], dtype=np.int32)]
                            for i in range(len(ordered))]
        else:
            print("GodotPZWrapper::Error:: Unknown Actions Type -> ", self.action_type)

        obs, reward, dones, truncs, info = super().step(godot_actions, order_ij=order_ij)

        # === DEBUG: RAW FROM GODOT Termination Tracking===
        #if any(dones.values()) or any(truncs.values()):
        #    print(f"[GODOT-RAW] step={self._ep_step} dones={dones} truncs={truncs}")
        #for a in self.possible_agents:
        #    ai = info.get(a, {}) if isinstance(info, dict) else {}
        #    if ai.get("episode_over") or ai.get("termination_reason"):
        #        print(f"[GODOT-INFO] {a}: episode_over={ai.get('episode_over')}, reason={ai.get('termination_reason')}")
        # === END DEBUG ===
        
        # DEBUG ==================================
       # Count agent-steps (ParallelEnv step)
        self._ep_step += 1

        # CRITICAL FIX: Convert info to proper dict format BEFORE using it
        if isinstance(info, list):
            info_dict = {}
            for i, agent_name in enumerate(self.agents):
                if i < len(info) and isinstance(info[i], dict):
                    info_dict[agent_name] = info[i]
                else:
                    info_dict[agent_name] = {}
            info = info_dict
        elif not isinstance(info, dict):
            info = {agent_name: {} for agent_name in self.agents}

        ''' Slightly wrong output during evaluations. FIX 15JAN 1600 ==================================
        # --- OPTION A termination gating: define episode_over ALWAYS ---
        episode_over = any(
            bool(info.get(a, {}).get("episode_over", False))
            for a in self.possible_agents
        )

        # Check if any agent is done/truncated (diagnostic only)
        _any_done   = any(bool(dones.get(a, False))  for a in self.possible_agents)
        _any_trunc  = any(bool(truncs.get(a, False)) for a in self.possible_agents)

        # This is now the real episode end condition
        _episode_end = episode_over

        # Apply gating to dones/truncs
        if not episode_over:
            for a in self.possible_agents:
                dones[a] = False
                truncs[a] = False
        else:
            for a in self.possible_agents:
                dones[a] = True
                truncs[a] = False
        '''
        # --- TERMINATION LOGIC: Use episode_over OR original dones ---
        episode_over_from_info = any(
            bool(info.get(a, {}).get("episode_over", False))
            for a in self.possible_agents
        )

        # Check if any agent is done/truncated from Godot's original signal
        # Keep these variable names for the diagnostic print statement
        _any_done = any(bool(dones.get(a, False)) for a in self.possible_agents)
        _any_trunc = any(bool(truncs.get(a, False)) for a in self.possible_agents)

        # Episode ends if EITHER condition is true
        _episode_end = episode_over_from_info or _any_done or _any_trunc

        # Set final termination flags based on combined condition
        for a in self.possible_agents:
            dones[a] = _episode_end
            truncs[a] = False  # Use dones for termination, not truncation


        # --- Derive a per-step 'missile_fired' flag for each agent ---
        for agent_name in self.possible_agents:
            ai = info.get(agent_name, {})

            # TODO: replace 'missiles_fired_total' with the actual key
            # you get from Godot info (e.g. 'blue_missiles_fired', 'weapons_expended', etc.)
            cur_total = ai.get("missiles_fired_total", 0)

            prev_total = self._prev_missiles_fired.get(agent_name, 0)
            fired_this_step = cur_total > prev_total

            ai["missile_fired"] = bool(fired_this_step)
            info[agent_name] = ai

            self._prev_missiles_fired[agent_name] = cur_total
        # --- end missile_fired derivation ---


        # On first episode end, print a compact reason line
        if _episode_end and not self._printed_ep_end:
            self._printed_ep_end = True
            #print(f"[TERM-DEBUG] At EP-END: self.terminations will be set from dones={dones}")
            # Try to pull a sim clock if Godot passes one
            sim_time = None
            for a in self.possible_agents:
                if a in info and isinstance(info[a], dict):
                    sim_time = info[a].get("sim_time_sec") or info[a].get("sim_time") or sim_time

            # Collect per-agent reasons if present
            reasons = []
            for a in self.possible_agents:
                ai = info.get(a, {})
                r = ai.get("termination_reason") or ai.get("reason") or ai.get("end_reason")
                if r:
                    reasons.append(f"{a}:{r}")
            reason_str = "; ".join(reasons) if reasons else "unknown"

            print(f"[EP-END] steps={self._ep_step} any_done={_any_done} any_trunc={_any_trunc} "
                f"sim_time={sim_time} reasons=[{reason_str}]")
    
        # Process observations
        self.observations = {agent_name : {"obs": _obs["obs"], "mask": [True for _ in range(4)]} for agent_name, _obs in obs.items()}
        
        # === DEBUG: What does dones look like right before aggregation? === termination signal
        #if _episode_end:
        #    print(f"[DEBUG-DONES] agents={self.possible_agents}, dones={dones}")
        # === END DEBUG ===

        # Aggregate termination/truncation flags
        self.terminations = _episode_end
        self.truncations = False        
        self.rewards = 0.0
        
        for i, agent in enumerate(self.possible_agents):
            self.rewards += reward[agent]
        
        # Causing termination signal loss 
        #for i, agent in enumerate(self.possible_agents):
        #    self.terminations = self.terminations or dones[agent]
        #    self.truncations = self.truncations or truncs[agent]           
        #    self.rewards += reward[agent]
        
        # === SIMPLER DEBUG: Print whenever _episode_end is True === termination signal
        if _episode_end:
            print(f"[TERM-CHECK] _episode_end={_episode_end} → self.terminations={self.terminations}, self.truncations={self.truncations}")
        # === END DEBUG ===
        
       
        self.info = info
        return self.observations, self.rewards, self.terminations, self.truncations, self.info


    def _process_obs(self, response_obs):
        return response_obs
    
    def last(self, env=None):
        
        """Return the last observations, rewards, and done status."""                        
        return (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.info,
        )
                
    def decode_action(self, encoded_action):
        # Decode back to the original action tuple        
        turn_input = encoded_action % 5
        level_input = (encoded_action // 5.0) % 5
        fire_input = (encoded_action // 25.0) % 2
        return np.array([fire_input, level_input, turn_input])