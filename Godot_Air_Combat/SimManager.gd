extends Node3D

var fighterObj   = preload("res://components/Fighter.tscn")
const SConv      = preload("res://assets/Sim_assets.gd").SConv
const SimGroups  = preload("res://assets/Sim_assets.gd").SimGroups

@onready var mainView = get_tree().root.get_node("B_ACE")
@onready var mainCanvas = mainView.get_node("CanvasLayer")
var tree = null

#const RewardsControl = preload("res://Sim_assets.gd").RewardsControl

var finalState = null 
var allFinalStates = []
var runs_count = 0

var id

var envConfig
var agentsConfig
var rewardsConfig
var simGroups

var agents = []
var enemies = []
var fighters = []
var teams_agents =[[],[]]
var hvaa_assets = []

var team_dl_tracks = [{},{}] #True of False to share the track id by Data Link per team_id
var last_team_dl_tracks = [{},{}] #True of False to share the track id by Data Link per team_id

var agents_alive_control
var enemies_alive_control

var n_action_steps = 0
var action_repeat
var max_cycles
var phy_fps
var physics_updates = 0
var elapsed_time = 0.0
var min_blue_red_separation = INF
var first_shot_team = -1  # -1 = none, 0 = blue, 1 = red
var _prev_blue_killed: int = 0
var _prev_red_killed: int = 0
var _blue_kills_this_step: int = 0
var _red_kills_this_step: int = 0
var _last_info_step: int = -1
var hvaa_spawn_positions: Dictionary = {}  # {hvaa_id: spawn_pos}
var hvaa_target_distance: float = 90.0 * SConv.NM2GDM 


var stop_mission = 1

var stop_simulation = false
var ready_to_reset = true
var initialized = false


func _update_step_event_counters():
	# Compute "kills_this_step" deltas once per sim step.
	# _get_info_from_agents() is called for each agent; without this guard, we'd
	# double-count if we updated prev counters inside the per-agent loop.
	if _last_info_step == n_action_steps:
		return

	var blue_total := 0
	var red_total := 0
	if finalState != null and typeof(finalState) == TYPE_ARRAY and finalState.size() >= 2:
		blue_total = int(finalState[0].get("killed", 0))
		red_total = int(finalState[1].get("killed", 0))

	_blue_kills_this_step = maxi(0, blue_total - _prev_blue_killed)
	_red_kills_this_step  = maxi(0, red_total  - _prev_red_killed)

	_prev_blue_killed = blue_total
	_prev_red_killed  = red_total
	_last_info_step = n_action_steps

func initialize(_id, _tree, _envConfig: Dictionary, _agentsConfig: Dictionary):
	# --- Teardown any previous run (idempotent initialize) ---
	# Stop processing while we rebuild
	set_process_mode_recursively(self, false)

	# --- Apply the new configs first ---
	id = _id
	tree = _tree
	envConfig = _envConfig.duplicate(true)
	agentsConfig = _agentsConfig.duplicate(true)
	simGroups = SimGroups.new(id)  # <-- MOVE THIS UP HERE

	# Clean up missiles first (so they don't tick while we free fighters)
	_teardown_world()

	# Reset all runtime containers/controls
	agents.clear()
	enemies.clear()
	fighters.clear()
	hvaa_assets.clear()
	teams_agents = [[], []]
	team_dl_tracks = [{}, {}]
	last_team_dl_tracks = [{}, {}]
	agents_alive_control = 0
	enemies_alive_control = 0
	stop_simulation = false
	ready_to_reset = true
	initialized = false

	action_repeat = int(envConfig.get("action_repeat", 20))
	max_cycles   = int(envConfig.get("max_cycles", 36000))
	stop_mission = int(envConfig.get("stop_mission", 1))
	phy_fps      = int(envConfig.get("phy_fps", 20))

	# --- Rebuild world & agents with the new configs ---
	_set_agents(tree)
	# OPTIONAL: spawn HVAA/asset planes as Blue “agents”, so Reds track them as enemies
	if agentsConfig.has("hvaa_agents"):
		var n_hvaa := int(agentsConfig["hvaa_agents"].get("num_agents", 0))
		for i in range(n_hvaa):
			var newAsset = fighterObj.instantiate()
			add_child(newAsset)

			newAsset.manager = self
			newAsset.get_node("RenderModel").set_scale(Vector3(4.0, 4.0, 4.0))
			newAsset.phy_fps       = int(envConfig["phy_fps"])
			newAsset.action_repeat = int(envConfig["action_repeat"])
			newAsset.action_type   = envConfig["action_type"]
			newAsset.max_cycles    = max_cycles
			newAsset.add_to_group(simGroups.FIGHTER)
			newAsset.simGroups     = simGroups

			var cfg: Dictionary = agentsConfig["hvaa_agents"].duplicate(true)
			newAsset.team_id = 0  # BLUE side (so REDs will attack it)
			
			hvaa_assets.append(newAsset)

			var num_group: int = _tree.get_nodes_in_group(simGroups.BLUE).size()
			var offset_x: float = (num_group / 2.0) if (num_group % 2 == 0) else (-(num_group - 1) / 2.0 - 1.0)

			newAsset.add_to_group(simGroups.AGENT)
			newAsset.add_to_group(simGroups.BLUE)
			newAsset.id = 100 + len(agents)
			newAsset.set_meta("id", newAsset.id)
			newAsset.team_color = "BLUE"
			newAsset.team_color_group = simGroups.BLUE
			newAsset.max_trail_points = envConfig["max_trail_size"]

		# Force HVAA behavior via config flags (Fighter.gd will read these)
			cfg["is_hvaa"] = true
			cfg["missiles"] = 0
			cfg["const_hdg_enable"] = cfg.get("const_hdg_enable", true)

			cfg["offset_pos"] = Vector3(offset_x * 6, 0.0, 0.0)
			newAsset.update_init_config(cfg, envConfig["RewardsConfig"])
			newAsset.is_hvaa = true
			newAsset.reset()
			newAsset.set_heuristic("AP")
			newAsset._heuristic = "AP"

			fighters.append(newAsset)
			teams_agents[newAsset.team_id].append(newAsset)
			

	
	# For RL agents we want "model" control, not AP/autopilot.
	for a in agents:
		if a.get("is_hvaa") and a.is_hvaa:
				a.set_heuristic("AP")
		else:
			a.set_heuristic("model")
# For enemies you can leave them baseline/AI or also set "model" if you want self-play.
	#for e in enemies:
		#e.set_heuristic("AP") # <- optional, they can still use autopilot if you want them stable

	_reset_simulation()

	initialized = true
	ready_to_reset = false
	set_process_mode_recursively(self, true)
	

func _any_hvaa_alive() -> bool:
	# HVAA is alive if it exists, is activated, and is NOT killed.
	for a in hvaa_assets:
		if not is_instance_valid(a):
			continue

		# Fighter.gd has: var activated = true, var killed = false
		if a.activated and (not a.killed):
			return true

	return false

func set_process_mode_recursively(node, _process_mode):
	node.set_process(_process_mode)
	node.set_physics_process(_process_mode)
	for child in node.get_children():
		set_process_mode_recursively(child, _process_mode)

func apply_blue_spawn(blue_spawn_cfg: Dictionary) -> void:
	#print("[SPAWN] enter apply_blue_spawn, cfg=", blue_spawn_cfg)
	if typeof(blue_spawn_cfg) != TYPE_DICTIONARY:
		return

	# Read config
	var mode := String(blue_spawn_cfg.get("mode", "fixed"))
	var center: Dictionary = blue_spawn_cfg.get("center", {"x":0.0,"z":0.0})
	var cx := float(center.get("x", 0.0))
	var cz := float(center.get("z", 0.0))
	var inner_r := float(blue_spawn_cfg.get("inner_radius", 0.0))
	var outer_r := float(blue_spawn_cfg.get("outer_radius", 0.0))
	var rect: Dictionary = blue_spawn_cfg.get("rect", {"xmin":-10.0,"xmax":10.0,"zmin":-10.0,"zmax":10.0})
	var alt_block: Dictionary = blue_spawn_cfg.get("altitude", {"value":25000.0})
	var y_alt_ft := float(alt_block.get("value", 25000.0))  # feet
	var min_sep := float(blue_spawn_cfg.get("min_sep", 0.0))
	var heading_range: Array = blue_spawn_cfg.get("heading_deg_range", [0.0, 360.0])
	var speed_range: Array   = blue_spawn_cfg.get("speed_range_mps", [220.0, 260.0])

	# RNG
	var rng := RandomNumberGenerator.new()
	rng.randomize()

	# Keep fighters separated (sampling space is in NM/ft)
	var placed_nmft: Array = []

	for a in agents:
		if a.team_id != 0:
			continue  # blue only

		# sample position in NM (x,z) and ft (y)
		var pos_nm := Vector3(cx, y_alt_ft, cz)  # (x_nm, y_ft, z_nm)
		var tries := 0
		while true:
			tries += 1
			if mode == "random_disk":
				var t := rng.randf()
				var r_nm := sqrt((outer_r*outer_r - inner_r*inner_r) * t + inner_r*inner_r)
				var th := rng.randf_range(0.0, TAU)
				pos_nm.x = cx + r_nm * cos(th)
				pos_nm.z = cz + r_nm * sin(th)
			elif mode == "random_rect":
				pos_nm.x = rng.randf_range(float(rect.get("xmin",-10.0)), float(rect.get("xmax",10.0)))
				pos_nm.z = rng.randf_range(float(rect.get("zmin",-10.0)), float(rect.get("zmax",10.0)))
			# else "fixed": leave pos_nm as-is

			var ok := true
			if min_sep > 0.0:
				for q in placed_nmft:
					if Vector3(pos_nm.x, 0.0, pos_nm.z).distance_to(Vector3(q.x, 0.0, q.z)) < min_sep:
						ok = false
						break
			if ok or tries > 100:
				break

		placed_nmft.append(pos_nm)

		# sample heading/speed
		var hdg_deg := rng.randf_range(float(heading_range[0]), float(heading_range[1]))
		var _speed := rng.randf_range(float(speed_range[0]), float(speed_range[1]))  # prefix '_' if unused

		#print("[SPAWN] BLUE pick pos(NM/ft)=", pos_nm, " hdg=", hdg_deg)

		# Convert NM/ft to engine units for Fighter.init_position (Vector3 required)
		#var pos_u := Vector3(
		#	pos_nm.x * SConv.NM2GDM,  # NM → Godot meters
		#	pos_nm.y * SConv.FT2GDM,  # feet → Godot meters
		#	pos_nm.z * SConv.NM2GDM
		#)

		# Build config the Fighter expects (Vector3s, not dictionaries)
		# pos_nm is (x_nm, y_ft, z_nm) sampled above
		# IMPORTANT: Fighter.update_init_config expects init_position as a DICTIONARY in NM/ft
		var bcfg: Dictionary = agentsConfig["blue_agents"].duplicate(true)

		bcfg["init_position"] = {       # <-- Dictionary, same pattern as your JSON
			"x": pos_nm.x,              # NM
			"y": pos_nm.y,              # ft
			"z": pos_nm.z               # NM
		}
		bcfg["init_hdg"] = float(hdg_deg)

		# Ensure offset_pos is a Vector3 (your _set_agents does this before the first call)
		if bcfg.has("offset_pos") and typeof(bcfg["offset_pos"]) != TYPE_VECTOR3:
			var off: Dictionary = bcfg.get("offset_pos", {"x":0.0,"y":0.0,"z":0.0})
			bcfg["offset_pos"] = Vector3(
				float(off.get("x", 0.0)),
				float(off.get("y", 0.0)),
				float(off.get("z", 0.0))
			)

		# Apply and reset so it takes effect immediately this episode and persists for the next reset
		a.update_init_config(bcfg, envConfig["RewardsConfig"])


		#print("[SPAWN] BLUE after apply: pos=", a.global_transform.origin)
		
func apply_hvaa_spawn(hvaa_spawn_cfg: Dictionary) -> void:
	if typeof(hvaa_spawn_cfg) != TYPE_DICTIONARY:
		return
	if hvaa_assets.is_empty():
		return

	# Read config (mirrors B_ACE_sync.gd logic)
	var mode := String(hvaa_spawn_cfg.get("mode", "random_rect"))

	var center: Dictionary = hvaa_spawn_cfg.get("center", {"x": 0.0, "z": 0.0})
	var cx := float(center.get("x", 0.0))
	var cz := float(center.get("z", 0.0))

	var inner_r := float(hvaa_spawn_cfg.get("inner_radius", 0.0))
	var outer_r := float(hvaa_spawn_cfg.get("outer_radius", 0.0))

	var rect: Dictionary = hvaa_spawn_cfg.get("rect", {
		"xmin": -10.0, "xmax": 10.0,
		"zmin": 30.0,  "zmax": 90.0
		})

	var alt_block: Dictionary = hvaa_spawn_cfg.get("altitude", {"value": 25000.0})
	var y_alt_ft := float(alt_block.get("value", 25000.0))  # ft

	var heading_range: Array = hvaa_spawn_cfg.get("heading_deg_range", [0.0, 360.0])
	#var speed_range: Array   = hvaa_spawn_cfg.get("speed_range_mps", [260.0, 290.0])

	# RNG similar to apply_blue_spawn
	var rng := RandomNumberGenerator.new()
	rng.randomize()

	# For each HVAA asset, sample a new position + heading
	for a in hvaa_assets:
		if not is_instance_valid(a):
			continue

		var pos_nm := Vector3()
		pos_nm.y = y_alt_ft

		if mode == "random_rect":
			pos_nm.x = rng.randf_range(float(rect.get("xmin", -10.0)), float(rect.get("xmax", 10.0)))
			pos_nm.z = rng.randf_range(float(rect.get("zmin", 30.0)),  float(rect.get("zmax", 90.0)))
		elif mode == "random_disk":
			var t := rng.randf()
			var r_nm := sqrt((outer_r * outer_r - inner_r * inner_r) * t + inner_r * inner_r)
			var th := rng.randf_range(0.0, TAU)
			pos_nm.x = cx + r_nm * cos(th)
			pos_nm.z = cz + r_nm * sin(th)
		else:
			# fallback: keep current init_position
			continue

		var hdg_deg := rng.randf_range(float(heading_range[0]), float(heading_range[1]))
		# (we ignore speed here; Fighter.gd / const_speed from agentsConfig will handle it)

		# Build HVAA config, mirroring initialize()
		var cfg: Dictionary = agentsConfig["hvaa_agents"].duplicate(true)
		cfg["is_hvaa"] = true
		cfg["missiles"] = 0
		cfg["const_hdg_enable"] = cfg.get("const_hdg_enable", true)

		cfg["init_position"] = {
			"x": pos_nm.x,
			"y": pos_nm.y,
			"z": pos_nm.z
			}
		cfg["init_hdg"] = hdg_deg
		cfg["const_hdg_deg"] = hdg_deg

		# Optional: keep whatever offset_pos logic you want; here we zero it:
		cfg["offset_pos"] = Vector3.ZERO
		
		a.update_init_config(cfg, envConfig["RewardsConfig"])
		#a.reset()
		

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _physics_process(delta):
					
	var donesAgents  =_check_all_done_agents()
	var donesEnemies  = _check_all_done_enemies()		
	
	for blue_agent in agents:
		for red_enemy in enemies:
			if blue_agent.activated and red_enemy.activated:
				var sep = blue_agent.global_position.distance_to(red_enemy.global_position)
				if sep < min_blue_red_separation:
					min_blue_red_separation = sep
	
	if donesAgents and donesEnemies:												
		ready_to_reset = true		
		set_process_mode_recursively(self, false)
									
	last_team_dl_tracks = team_dl_tracks #Use last list pointer
	team_dl_tracks = [{},{}] #reset list before agents do the updates
			
	# Increment the physics update count
	physics_updates += 1    
	elapsed_time += delta	
				
	
	if agents_alive_control == 0 and not _any_hvaa_alive() and not stop_simulation:
		stop_simulation = true
		finalState[0]["end_cond"] = "All_Blue_Destroyed"
		finalState[1]["end_cond"] = "All_Blue_Destroyed"
		for agent in agents:
			agent.done = true
			agent.ownRewards.add_final_episode_reward("Team_Killed", 1.0, agent.missiles)
		for enemy in enemies:
			enemy.done = true
			
	if not stop_simulation:
		for hvaa in hvaa_assets:
			if is_instance_valid(hvaa) and hvaa.activated:
				var spawn_pos = hvaa_spawn_positions.get(hvaa.get_instance_id(), hvaa.global_transform.origin)
				var dist_traveled = hvaa.global_transform.origin.distance_to(spawn_pos)
				if dist_traveled >= hvaa_target_distance:
					stop_simulation = true
					finalState[0]["end_cond"] = "HVAA_Mission_Success"
					finalState[1]["end_cond"] = "HVAA_Mission_Success"
					for agent in agents:
						agent.done = true
						agent.ownRewards.add_final_episode_reward("HVAA_Mission_Success", 1.0, agent.missiles)
						print("[TERM] HVAA_Mission_Success awarding unit terminal reward")
					for enemy in enemies:
						enemy.done = true
					break
			
	
	#if agents_alive_control == 0 and not stop_simulation :		
		
		#var missiles = tree.get_nodes_in_group(simGroups.MISSILE)
		#if len(missiles) == 0:
			#stop_simulation = true
			#finalState[0]["end_cond"] = "Blue_Killed"
			#finalState[1]["end_cond"] = "Blue_Killed"
			#for agent in agents:			
				#agent.ownRewards.add_final_episode_reward("Team_Killed", (max_cycles - n_action_steps) / action_repeat, agent.missiles)				
			
			#for enemy in enemies:
				#enemy.done = true
	
	#if enemies_alive_control == 0 and not stop_simulation:		
		
		#var missiles = tree.get_nodes_in_group(simGroups.MISSILE)		
		#if len(missiles) == 0:
			#stop_simulation = true
			#inalState[0]["end_cond"] = "Red_Killed"
			#finalState[1]["end_cond"] = "Red_Killed"
			#for agent in agents:
				#agent.done = true
				#agent.ownRewards.add_final_episode_reward("Enemies_Killed", (max_cycles - n_action_steps) / action_repeat, agent.missiles)					
			
				
	if n_action_steps % action_repeat != 0:
		n_action_steps += 1
		return
			
	#Reach This part only every ActionRepeat Steps
	n_action_steps += 1
		
	if n_action_steps >= max_cycles:		
		
		for enemy in enemies:
			enemy.done = true 
			enemy.ownRewards.add_final_episode_reward("Max_Cycles", (max_cycles - n_action_steps) / action_repeat, enemy.missiles)
					
		for agent in agents:
			agent.done = true 						
			agent.ownRewards.add_final_episode_reward("Max_Cycles", (max_cycles - n_action_steps) / action_repeat, agent.missiles)
					
		finalState[0]["end_cond"] = "Max_Cycles"
		finalState[1]["end_cond"] = "Max_Cycles"
		
		stop_simulation = true
		
		
	#PROCCESS Global Rewards
	#Enmies Rewards are actually penaulties due to the proximity to the 
	#Enemies targets and also finish the episode in case the target is achieved		
	var enemy_on_target = false
	#var enemy_goal_reward  = 0.0
	
	#Calculate Penaulties for enemy distance to target	
	var tactics = []
	for enemy in enemies:		
		if enemy.activated and not enemy.get_done():			
			tactics.append(str(enemy.id) + ": " + enemy.tatic_status )
			#if enemy.HPT != null:
			#	tactics[-1] += " " + str(enemy.HPT.is_alive) + "/" + str(enemy.HPT.offensive_factor)
			#enemy_goal_reward += -1.0 / enemy.dist2go
			#if enemy.dist2go < 5.0 and stop_mission == 1 and enemy.mission == "striker": #500 meters
				#enemy_on_target = true
				#finalState[1]['mission'] += 1 
				#finalState[0]["end_cond"] = "Red_Mission"
				#finalState[1]["end_cond"] = "Red_Mission"
	
	mainCanvas.update_tactics(tactics)

	if enemy_on_target:
		for agent in agents:
			agent.done = true
			agent.ownRewards.add_final_episode_reward("Enemy_Achieved_Target", 1.0, agent.missiles)
		for enemy in enemies:
			enemy.done = true
			
	#Calculate Penaulties for own distance to defense target	
	var own_goal_reward = 0.0
	for agent in agents:
		if agent.activated and not agent.get_done():										
			#own_goal_reward += agent.dist2go / 185200
			own_goal_reward += 1.0 + (-0.99)/(1 + exp(-0.02 * (agent.dist2go - 370.4)))
				
			#Add the calculated rews
			#agent.ownRewards.add_mission_rew(enemy_goal_reward)
			agent.ownRewards.add_mission_rew(own_goal_reward)
	
func _set_agents(_tree):	
				
	#Scale Vectors only for Visualization	
	const visual_scaleVector = Vector3(4.0,  4.0,  4.0)
	
	var listComponents = []	
	for i in range(int(agentsConfig["blue_agents"]["num_agents"])):
		listComponents.append("Allied_Agent")
	
	for i in range(int(agentsConfig["red_agents"]["num_agents"])):
		listComponents.append("Enemy_Agent")
		
	for comp in listComponents:
		
		var newFigther = null				
		
		newFigther = fighterObj.instantiate()		
		add_child(newFigther)
		
		newFigther.manager = self
		newFigther.get_node("RenderModel").set_scale(visual_scaleVector)
		
		newFigther.phy_fps 		 = int(envConfig["phy_fps"])
		newFigther.action_repeat = int(envConfig["action_repeat"])
		newFigther.action_type 	 = envConfig["action_type"]
		
		newFigther.max_cycles = max_cycles
		
		newFigther.add_to_group(simGroups.FIGHTER)
		newFigther.simGroups = simGroups				
						
		if comp == "Allied_Agent":
			
			var blue_config = agentsConfig["blue_agents"].duplicate(true)
			newFigther.team_id 	= 0			
			agents.append(newFigther)			
			
			var offset_x = 0
			var num_group = _tree.get_nodes_in_group(simGroups.BLUE).size()
			if num_group % 2 == 0:
				offset_x = num_group / 2
			else:
				offset_x = -(num_group -1) / 2 - 1
			
			newFigther.add_to_group(simGroups.AGENT)							
			newFigther.add_to_group(simGroups.BLUE)
			newFigther.id = 100 +  len(agents)
			newFigther.set_meta('id', 100 +  len(agents))
			
			newFigther.team_color = "BLUE"
			newFigther.team_color_group = simGroups.BLUE
			
			newFigther.max_trail_points = envConfig["max_trail_size"] 
									
			blue_config["offset_pos"] = Vector3(offset_x * 6, 0.0, 0.0)			
			newFigther.update_init_config(blue_config, envConfig["RewardsConfig"])			
			newFigther.reset()
			newFigther.behavior = "external"
			newFigther._heuristic = "model"
											
		else:
			var red_config = agentsConfig["red_agents"].duplicate(true)
			newFigther.team_id = 1
			enemies.append(newFigther)

									
			var num_group = _tree.get_nodes_in_group(simGroups.RED).size()
			var offset_x = 0
			if num_group % 2 == 0:
				offset_x = num_group / 2
			else:
				offset_x = -(num_group -1) / 2 - 1 
						
			newFigther.add_to_group(simGroups.ENEMY)							
			newFigther.add_to_group(simGroups.RED)
			newFigther.id = 200 +  len(enemies)
			newFigther.set_meta('id', 200 +  len(enemies))
			newFigther.team_color = "RED"
			newFigther.team_color_group = simGroups.RED
			
			newFigther.max_trail_points = envConfig["max_trail_size"] 
			
			red_config["offset_pos"] = Vector3(offset_x * 6, 0.0, 0.0)			
			newFigther.update_init_config(red_config, envConfig["RewardsConfig"])			
			newFigther.reset()				
																										
		fighters.append(newFigther)
		teams_agents[newFigther.team_id].append(newFigther)
			
	for fighter in fighters:
		fighter.update_scene(tree)
	var i = 0
	for agent in agents:
		agent.agent_name = "agent_" + str(i)
		i = i + 1
		

func _reset_simulation():
	_reset_components()	
	if envConfig.has("BlueSpawn"):
		apply_blue_spawn(envConfig["BlueSpawn"])
	if envConfig.has("HVAASpawn"):
		apply_hvaa_spawn(envConfig["HVAASpawn"])
	_reset_all_uavs()
	
		# Update tracks with correct positions
	for fighter in fighters:
		fighter.update_scene(tree)
	
	# Reset data link tracking
	for team_id in range(2):
		for agent in teams_agents[team_id]:
			for track in agent.radar_track_list:
				team_dl_tracks[team_id][track.id] = false
				track.is_alive = false
	
	min_blue_red_separation = INF
	first_shot_team = -1 
	
	physics_updates = 0
	elapsed_time = 0.0

	agents_alive_control = len(agents)
	enemies_alive_control = len(enemies)
							
	physics_updates = 0
	elapsed_time = 0.0
	
	agents_alive_control = len(agents)
	enemies_alive_control = len(enemies)
					
	if n_action_steps > 3:
		runs_count = runs_count + 1 	
		
		if finalState != null:
			var lastResults = finalState.duplicate(true)						
			var enemy_agent = teams_agents[1][0]			
			
			lastResults.append({"run_number": runs_count, 
								"steps"      : n_action_steps,								
								"enemy_behavior": enemy_agent.behavior,
								"enemy_params" : [enemy_agent.dShot, enemy_agent.lCrank, enemy_agent.lBreak],
								}	
							)
			allFinalStates.append(lastResults)
	
	finalState = []
	
	for i in range(2):
		finalState.append( {	
				"killed" 	: 0,
				"mission"	: 0,
				"reward"  	: 0.0,
				"missile"	: 0,				
				"end_cond"  : null
				})
		# Reset non-terminal event tracking for the new episode
	_prev_blue_killed = 0
	_prev_red_killed = 0
	_blue_kills_this_step = 0
	_red_kills_this_step = 0
	_last_info_step = -1
	
	stop_simulation = false
	n_action_steps = 0
	ready_to_reset = false
	
	set_process_mode_recursively(self, true)	
	if envConfig.has("BlueSpawn"):
		if "apply_blue_spawn" in self:
			#print("[SPAWN] SimManager applying BlueSpawn…")
			apply_blue_spawn(envConfig["BlueSpawn"])
			# Show what actually stuck:
			#for a in agents:
				#if a.team_id == 0:
					#print("[SPAWN] BLUE after apply: pos=", a.global_transform.origin)
		#else:
			#print("[SPAWN] apply_blue_spawn() missing on SimManager")
	hvaa_spawn_positions.clear()
	for hvaa in hvaa_assets:
		if is_instance_valid(hvaa):
			hvaa_spawn_positions[hvaa.get_instance_id()] = hvaa.global_transform.origin
	

func _teardown_world():
	# Free all missiles
	if tree != null and simGroups != null:
		var missiles = tree.get_nodes_in_group(simGroups.MISSILE)
		for missile in missiles:
			if is_instance_valid(missile):
				missile.queue_free()

	# Free all fighters/nodes we spawned previously
	for f in fighters:
		if is_instance_valid(f):
			f.queue_free()


func _reset_components():	
	# Episode cleanup — also guard for first run
	if tree == null or simGroups == null:
		return
	var missiles = tree.get_nodes_in_group(simGroups.MISSILE)
	for missile in missiles:
		if is_instance_valid(missile):
			missile.queue_free()
	
			
func _reset_all_uavs():
	
	if initialized:
		for uav in fighters:
			uav.needs_reset = true
			uav.reactivate()
			uav.reset() 
	
func _get_obs_from_agents():
	
	var obs = []
	for agent in agents:
		obs.append(agent.get_obs())
				
	return obs
	
func _get_reward_from_agents():
	var rewards = [] 
	for agent in agents:
		rewards.append(agent.get_reward())		
		finalState[agent.team_id]["reward"] += rewards[-1]		
	return rewards    
	
func _get_done_from_agents():
	var dones = []
	for agent in agents:
		dones.append(agent.get_done())		
	return dones

func _get_info_from_agents():

	var info_array = []
	_update_step_event_counters()
	for agent in agents:
		var agent_info = {}
		
		agent_info["episode_over"] = stop_simulation or ready_to_reset
		agent_info["episode_steps"] = n_action_steps
		agent_info["sim_time_sec"] = elapsed_time
		if hvaa_assets.size() > 0:
			var hvaa_alive = _any_hvaa_alive()
			agent_info["hvaa_alive"] = hvaa_alive
			agent_info["hvaa_destroyed"] = not hvaa_alive
		else:
			agent_info["hvaa_alive"] = true
			agent_info["hvaa_destroyed"] = false
		# Only populate info if episode is done
		
		# --- Non-terminal combat notifications (always present) ---
		agent_info["blue_killed_total"] = _prev_blue_killed
		agent_info["red_killed_total"]  = _prev_red_killed
		agent_info["blue_kills_this_step"] = _blue_kills_this_step
		agent_info["red_kills_this_step"]  = _red_kills_this_step
		agent_info["blue_killed_event"] = (_blue_kills_this_step > 0)
		agent_info["red_killed_event"]  = (_red_kills_this_step > 0)

		# --- HVAA status every step (always present) ---
		if hvaa_assets.size() > 0:
			var hvaa_alive = _any_hvaa_alive()
			agent_info["hvaa_alive"] = hvaa_alive
			agent_info["hvaa_destroyed"] = not hvaa_alive
		else:
			agent_info["hvaa_alive"] = true
			agent_info["hvaa_destroyed"] = false
			
		# ---------------------------------------------------------
		# Combat + geometry metrics (emit EVERY STEP, not just done)
		# ---------------------------------------------------------
		if finalState != null and typeof(finalState) == TYPE_ARRAY and finalState.size() >= 2:
			var blue_killed := int(finalState[0].get("killed", 0))
			var red_killed  := int(finalState[1].get("killed", 0))
			var blue_missiles := int(finalState[0].get("missile", 0))
			var red_missiles  := int(finalState[1].get("missile", 0))

			agent_info["blue_killed"] = blue_killed
			agent_info["red_killed"] = red_killed
			agent_info["blue_missiles_fired"] = blue_missiles
			agent_info["red_missiles_fired"] = red_missiles

			# Pk definitions (same as you had, but now always present)
			agent_info["blue_pk"] = (float(red_killed) / float(blue_missiles)) if blue_missiles > 0 else 0.0
			agent_info["red_pk"]  = (float(blue_killed) / float(red_missiles)) if red_missiles > 0 else 0.0
		else:
			# Safe defaults
			agent_info["blue_killed"] = 0
			agent_info["red_killed"] = 0
			agent_info["blue_missiles_fired"] = 0
			agent_info["red_missiles_fired"] = 0
			agent_info["blue_pk"] = 0.0
			agent_info["red_pk"] = 0.0

		# Min separation is tracked across the episode; emit every step
		if min_blue_red_separation < INF:
			agent_info["min_separation_nm"] = float(min_blue_red_separation) / float(SConv.NM2GDM)
		else:
			agent_info["min_separation_nm"] = -1.0

		# First-shot flags (derived from first_shot_team) — emit every step
		agent_info["blue_shot_first"] = (first_shot_team == 0)
		agent_info["red_shot_first"]  = (first_shot_team == 1)
		agent_info["no_shots_fired"]  = (first_shot_team == -1)
		
		if agent.get_done():
			# DEBUG once per done agent
			''' 
			print("[DONE-DEBUG] agent=", agent.name, " team_id=", agent.team_id,
				" finalState_type=", typeof(finalState),
				" finalState=", finalState)
				
			if finalState != null and typeof(finalState) == TYPE_ARRAY:
				print("[DONE-DEBUG] finalState.size()=", finalState.size())
				if agent.team_id >= 0 and agent.team_id < finalState.size():
					print("[DONE-DEBUG] finalState[team]=", finalState[agent.team_id])
					print("[DONE-DEBUG] end_cond=", finalState[agent.team_id].get("end_cond", "<missing>"))
					'''
			# Team status
			agent_info["all_red_destroyed"] = (enemies_alive_control == 0)
			agent_info["all_blue_destroyed"] = (agents_alive_control == 0)
			
			# HVAA status
			if hvaa_assets.size() > 0:
				agent_info["hvaa_destroyed"] = not _any_hvaa_alive()
			else:
				agent_info["hvaa_destroyed"] = false
			
			# Termination reason
			if finalState != null and finalState.size() > agent.team_id:
				var end_cond = finalState[agent.team_id].get("end_cond")
				agent_info["termination_reason"] = end_cond if end_cond != null else "unknown"
			else:
				agent_info["termination_reason"] = "unknown"
			
			# Simulation time
			agent_info["sim_time_sec"] = elapsed_time
			agent_info["episode_steps"] = n_action_steps
			
			# Combat metrics
			if finalState != null and finalState.size() > 0:
				var blue_killed = finalState[0].get("killed", 0)
				var red_killed = finalState[1].get("killed", 0)
				var blue_missiles = finalState[0].get("missile", 0)
				var red_missiles = finalState[1].get("missile", 0)
				
				agent_info["blue_killed"] = blue_killed
				agent_info["red_killed"] = red_killed
				agent_info["blue_missiles_fired"] = blue_missiles
				agent_info["red_missiles_fired"] = red_missiles
				
				# Missile efficiency (Pk - probability of kill)
				agent_info["blue_pk"] = (float(red_killed) / float(blue_missiles)) if blue_missiles > 0 else 0.0
				agent_info["red_pk"] = (float(blue_killed) / float(red_missiles)) if red_missiles > 0 else 0.0
			
			# Mission outcome categorization
			var red_dead = agent_info.get("all_red_destroyed", false)
			var blue_dead = agent_info.get("all_blue_destroyed", false)
			var hvaa_dead = agent_info.get("hvaa_destroyed", false)
			
			if red_dead and not hvaa_dead:
				agent_info["mission_outcome"] = "complete_victory"
				agent_info["mission_success"] = true
			elif red_dead and hvaa_dead:
				agent_info["mission_outcome"] = "pyrrhic_victory"
				agent_info["mission_success"] = false
			elif hvaa_dead:
				agent_info["mission_outcome"] = "mission_failure"
				agent_info["mission_success"] = false
			else:
				agent_info["mission_outcome"] = "incomplete"
				agent_info["mission_success"] = false
			
			# Closest approach distance
			if min_blue_red_separation < INF:
				agent_info["min_separation_nm"] = min_blue_red_separation / SConv.NM2GDM
			else:
				agent_info["min_separation_nm"] = -1.0  # No valid measurement
			
			# First shot advantage
			agent_info["blue_shot_first"] = (first_shot_team == 0)
			agent_info["red_shot_first"] = (first_shot_team == 1)
			agent_info["no_shots_fired"] = (first_shot_team == -1)
		
		info_array.append(agent_info)
	
	return info_array

func _get_done_from_enemies():
	var dones = []
	for enemy in enemies:
		dones.append(enemy.get_done())		
	return dones

func _check_all_done_agents():	
	for agent in agents:
		if not agent.get_done():
			return false					
	return true

func _check_all_done_enemies():	
	for enemy in enemies:
		if not enemy.get_done():
			return false					
	return true

func _set_agent_actions(actions):
	for i in range(len(actions)):
		#env.debug_text.add_text("\nAction:" + str(actions[i])) 
		#print(i, actions[i])
		agents[i].set_action(actions[i])
	
func _set_heuristic(heuristic):
	for agent in agents:
		agent.set_heuristic(heuristic)

func _collect_results():	
	return finalState
	
func _collect_last_results():	
	var finalResults = allFinalStates.duplicate(true)
	#allFinalStates = []
	return finalResults
	

func inform_state(team_id, condition, src: Object = null):
	if condition == "killed":
		var is_hvaa := false
		if src != null and is_instance_valid(src):
			is_hvaa = bool(src.get("is_hvaa")) if "is_hvaa" in src else false

		if team_id == 0:
			if not is_hvaa:
				agents_alive_control -= 1
				finalState[team_id]["killed"] += 1   # <-- ONLY count BLUE AGENT deaths
			else:
				# Optional: track HVAA separately if you want
				# finalState[team_id]["hvaa_killed"] = int(finalState[team_id].get("hvaa_killed", 0)) + 1
				pass
		else:
			enemies_alive_control -= 1
			finalState[team_id]["killed"] += 1       # <-- RED deaths still counted

	elif condition == "missile":
		if first_shot_team == -1:
			first_shot_team = team_id
		finalState[team_id]["missile"] += 1

''' BEFORE FIX TO TRACK BLUE FIGHTERS/HVAA Separately
func inform_state(team_id, condition, src: Object = null):
	# src (optional) can be the Fighter node calling this

	if condition == "killed":
		# If the killed unit is HVAA, do NOT count it as a "blue agent"
		var is_hvaa := false
		if src != null and is_instance_valid(src):
			# Fighter.gd sets is_hvaa = true for HVAA assets in your initialize()
			is_hvaa = bool(src.get("is_hvaa")) if "is_hvaa" in src else false

		if team_id == 0:
			if not is_hvaa:
				agents_alive_control -= 1
			# else: HVAA death tracked by _any_hvaa_alive() and hvaa_assets
		else:
			enemies_alive_control -= 1

		finalState[team_id]["killed"] += 1

	elif condition == "missile":
		if first_shot_team == -1:
			first_shot_team = team_id
		finalState[team_id]["missile"] += 1
'''
