#SHOULD BE OLD CODE
extends Node
# --fixed-fps 2000 --disable-render-loop

const MAJOR_VERSION := "0"
const MINOR_VERSION := "1" 
const DEFAULT_PORT := "11008"

@onready var mainView = get_tree().root.get_node("B_ACE")
@onready var mainViewPort = get_tree().root.get_node("B_ACE/Simulations")
@onready var mainCanvas = get_tree().root.get_node("B_ACE/CanvasLayer")
@onready var phy_show = mainView.get_node("CanvasLayer/Control/PHY_Show")
@onready var steps_show = mainView.get_node("CanvasLayer/Control/Steps_Show")

const SimManager = preload("res://SimManager.tscn")
const SConv = preload("res://assets/Sim_assets.gd").SConv

var envConfig

var simConfig
#Default line params GodotRL Config
var action_repeat
var renderize 
var sim_speed = 1.0
var phy_fps
var speed_up 
var seed

# Variables for real-time visual render timer
var visual_fps = 60 
var render_interval_msec: float = 0.0
var last_render_time_msec: int = 0
var accumulated_sim_time: float = 0.0

#Aditional Env Config
var parallel_envs 

#Experiment Mode 
var experiment_mode
var experiment_runs_per_case
var experiment_current_run
var experiment_results
var experiment_in_progress = false

var simulation_list 

var n_action_steps : int = 0
var render_count = 0
var last_check = 0.0

var rng = RandomNumberGenerator.new()

var stream : StreamPeerTCP = null
var connected = false
var message_center
var should_connect = true

var need_to_send_obs = false
var args = null
@onready var start_time = Time.get_ticks_msec()
var initialized = false
var just_reset = false
var stop_simulation = false

# Called when the node enters the scene tree for the first time.
func _ready():
		
	await get_tree().root.ready
	get_tree().set_pause(true) 
	_initialize()
	
	await get_tree().create_timer(1.0).timeout
	get_tree().set_pause(false) 
	
	render_count = 0
	last_check = Time.get_ticks_msec()

	
func _handshake():
	#print("DEBUG: Performing handshake...")
	
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "handshake")
	var major_version = json_dict["major_version"]
	var minor_version = json_dict["minor_version"]
	if major_version != MAJOR_VERSION:
		print("WARNING: major version mismatch ", major_version, " ", MAJOR_VERSION)  
	if minor_version != MINOR_VERSION:
		print("WARNING: minor version mismatch ", minor_version, " ", MINOR_VERSION)
	
	#print("DEBUG: Handshake complete")

func _get_exact(num_bytes: int) -> PackedByteArray:
	var out := PackedByteArray()
	while out.size() < num_bytes:
		stream.poll()
		if stream.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			print("ERROR: stream disconnected while reading")
			return PackedByteArray()  # return empty -> caller will treat as error

		var to_read := num_bytes - out.size()
		var result = stream.get_data(to_read)
		var err = result[0]
		var chunk: PackedByteArray = result[1]

		if err != OK:
			OS.delay_usec(10)
			continue  # try again

		if chunk.size() == 0:
			OS.delay_usec(10)
			continue  # wait for more bytes

		out.append_array(chunk)

	return out


func _get_dict_json_message() -> Dictionary:
	# 1. read 4-byte little-endian length
	var header_bytes := _get_exact(4)
	if header_bytes.size() < 4:
		print("ERROR: Failed to read length header from Python")
		return {}

	var msg_len := (
		header_bytes[0]
		| (header_bytes[1] << 8)
		| (header_bytes[2] << 16)
		| (header_bytes[3] << 24)
	)

	# 2. read body of that length
	var body_bytes := _get_exact(msg_len)
	if body_bytes.size() < msg_len:
		print("ERROR: Failed to read full body from Python")
		return {}

	var message_str := body_bytes.get_string_from_utf8()

	var parsed = JSON.parse_string(message_str)
	if typeof(parsed) != TYPE_DICTIONARY:
		print("ERROR: expected dict JSON, got: ", message_str)
		return {}

	return parsed

func _load_json_file_dict(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("CONFIG: could not open " + path)
		return {}
	var txt := f.get_as_text()
	f.close()
	var json := JSON.new()
	var ok := json.parse(txt)
	if ok != OK:
		push_error("CONFIG: JSON parse error in " + path + " -> " + json.get_error_message())
		return {}
	var data := json.get_data()
	if typeof(data) == TYPE_DICTIONARY:
		return data
	return {}

var episode_idx: int = 0

func _randomize_blue_spawns_for_all_sims():
	# Call this right after each sim._reset_simulation()
	episode_idx += 1
	var bs: Dictionary = envConfig.get("BlueSpawn", {})
	if bs.is_empty():
		return
	for sim in simulation_list:
		_randomize_blue_spawns(sim, bs)

func _randomize_blue_spawns(sim: Node, blue_spawn_cfg):
	# Collect blue fighters robustly
	var blue := _get_blue_fighters(sim)
	if blue.is_empty():
		return

	# Option A: If you want to honor AgentsConfig.blue_agents.rnd_offset_range:
	# _apply_simple_offsets_from_agents_config(sim, blue); return

	# Option B: Use EnvConfig.BlueSpawn (richer shapes/spacing/heading/speed)
	if typeof(blue_spawn_cfg) != TYPE_DICTIONARY:
		return

	var mode := String(blue_spawn_cfg.get("mode", "fixed"))
	var center: Dictionary = blue_spawn_cfg.get("center", {"x":0.0,"z":0.0})
	var cx := float(center.get("x", 0.0))
	var cz := float(center.get("z", 0.0))
	var inner_r := float(blue_spawn_cfg.get("inner_radius", 0.0))
	var outer_r := float(blue_spawn_cfg.get("outer_radius", 0.0))
	var rect: Dictionary = blue_spawn_cfg.get("rect", {"xmin":-10.0,"xmax":10.0,"zmin":-10.0,"zmax":10.0})
	var alt_block: Dictionary = blue_spawn_cfg.get("altitude", {"value":25000.0})
	var y_alt := float(alt_block.get("value", 25000.0))
	var min_sep := float(blue_spawn_cfg.get("min_sep", 0.0))
	var heading_range: Array = blue_spawn_cfg.get("heading_deg_range", [0.0, 360.0])
	var speed_range: Array   = blue_spawn_cfg.get("speed_range_mps", [220.0, 260.0])
	var placed: Array[Vector3] = []

	# seed for reproducibility (base seed + episode)
	var base_seed := int(envConfig.get("seed", 1))
	rng.seed = int(base_seed + episode_idx)

	for f in blue:
		var pos: Vector3 = f.global_transform.origin
		var tries := 0
		var max_tries := 200

		while true:
			tries += 1
			if mode == "random_disk":
				pos = _sample_point_in_annulus(cx, cz, inner_r, outer_r, y_alt)
			elif mode == "random_rect":
				pos = _sample_point_in_rect(rect, y_alt)
			else:
				break  # fixed -> keep whatever _reset_simulation gave us

			if min_sep <= 0.0 or _is_far_from_all(pos, placed, min_sep):
				break
			if tries > max_tries:
				break

		placed.append(pos)

		# apply position OLD - wrong conversion NM/ft to meters
		#var xf: Transform3D = f.global_transform
		#xf.origin = pos
		#f.global_transform = xf
		
		# NEW (RIGHT: convert NM/ft -> Godot meters before applying)
		var pos_gdm := Vector3(
			pos.x * SConv.NM2GDM,   # NM -> m (scaled)
			pos.y * SConv.FT2GDM,   # ft -> m (scaled)
			pos.z * SConv.NM2GDM
			)

		var xf: Transform3D = f.global_transform
		xf.origin = pos_gdm
		f.global_transform = xf

# (optional) sanity print once
		#print("[SPAWN] (NM,ft,NM)=", pos, " -> (m)=", pos_gdm)

		# heading & speed
		var hdg_deg := rng.randf_range(float(heading_range[0]), float(heading_range[1]))
		var speed := rng.randf_range(float(speed_range[0]), float(speed_range[1]))

		# Convert heading to velocity on x–z plane; adjust if your forward axis differs
		var hdg_rad := deg_to_rad(hdg_deg)
		var vx := speed * sin(hdg_rad)
		var vz := -speed * cos(hdg_rad)
		var vel := Vector3(vx, 0.0, vz)

		# Prefer aircraft APIs if available
		if "set_heading_deg" in f:
			f.set_heading_deg(hdg_deg)
		if "set_speed_mps" in f:
			f.set_speed_mps(speed)
		elif "linear_velocity" in f:
			f.linear_velocity = vel

		_face_velocity(f, vel)

func _randomize_hvaa_spawns_for_all_sims():
	var hs: Dictionary = envConfig.get("HVAASpawn", {})
	print("[HVAA DEBUG] _randomize_hvaa_spawns_for_all_sims called. HVAASpawn:", hs)
	if hs.is_empty():
		print("[HVAA DEBUG] HVAASpawn is empty, skipping HVAA randomization.")
		return
	for sim in simulation_list:
		print("[HVAA DEBUG] randomizing HVAA for sim: ", sim.name)
		_randomize_hvaa_spawns(sim, hs)


func _randomize_hvaa_spawns(sim: Node, hvaa_spawn_cfg: Dictionary) -> void:
	var hvaa_arr := _get_hvaa_aircraft(sim)
	print("[HVAA DEBUG] _randomize_hvaa_spawns: found ", hvaa_arr.size(), " HVAA(s) ", sim.name)
	if hvaa_arr.is_empty():
		return

	# Shape and altitude
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
	var y_alt := float(alt_block.get("value", 25000.0))

	var heading_range: Array = hvaa_spawn_cfg.get("heading_deg_range", [0.0, 360.0])
	var speed_range: Array   = hvaa_spawn_cfg.get("speed_range_mps", [250.0, 290.0])

	# Different seed offset from blue so they don't correlate
	var base_seed := int(envConfig.get("seed", 1))
	rng.seed = int(base_seed + 100000 + episode_idx)
	
	print("[HVAA DEBUG] Using HVAASpawn cfg mode:", mode, " rect:", rect, 
		" alt_ft:", y_alt, " heading_range:", heading_range, " speed_range:", speed_range)
	
	for f in hvaa_arr:
		var pos: Vector3

		if mode == "random_rect":
			pos = _sample_point_in_rect(rect, y_alt)
		elif mode == "random_disk":
			pos = _sample_point_in_annulus(cx, cz, inner_r, outer_r, y_alt)
		else:
			# fall back to whatever init_position gave us
			continue

		# Convert NM/ft to Godot meters (same as blue)
		var pos_gdm := Vector3(
			pos.x * SConv.NM2GDM,
			pos.y * SConv.FT2GDM,
			pos.z * SConv.NM2GDM
		)

		var xf: Transform3D = f.global_transform
		xf.origin = pos_gdm
		f.global_transform = xf

		# Heading + speed
		var hdg_deg := rng.randf_range(float(heading_range[0]), float(heading_range[1]))
		var speed := rng.randf_range(float(speed_range[0]), float(speed_range[1]))
		var hdg_rad := deg_to_rad(hdg_deg)

		var vx := speed * sin(hdg_rad)
		var vz := -speed * cos(hdg_rad)
		var vel := Vector3(vx, 0.0, vz)

		if "set_heading_deg" in f:
			f.set_heading_deg(hdg_deg)
		if "set_speed_mps" in f:
			f.set_speed_mps(speed)
		elif "linear_velocity" in f:
			f.linear_velocity = vel
		
		print("[HVAA DEBUG] New HVAA pos (NM,ft,NM):", pos, 
			" -> GDM:", pos_gdm, 
			" hdg_deg:", hdg_deg, " speed:", speed)
		
		_face_velocity(f, vel)

func _get_blue_fighters(sim: Node) -> Array:
	# Try several common ways B-ACE/Godot airframes tag "blue"
	var arr: Array = []
	if "fighters" in sim:
		for n in sim.fighters:
			if n == null: continue
			if "team" in n and int(n.team) == 0:   # team 0 -> blue (common)
				arr.append(n)
			elif "side" in n and str(n.side).to_lower() == "blue":
				arr.append(n)
			elif n.is_in_group("blue") or n.is_in_group("blue_fighters"):
				arr.append(n)
	# If your sim exposes sim.blue_fighters, use that instead:
	# if "blue_fighters" in sim: return sim.blue_fighters
	return arr

func _get_hvaa_aircraft(sim: Node) -> Array:
	var arr: Array = []
	if "fighters" in sim:
		for n in sim.fighters:
			if n == null:
				continue
			
			# Declare with simple, known types first
			var has_flag := false
			var flag_val := false

			# Check for is_hvaa in metadata or as a property
			if n.has_meta("is_hvaa"):
				has_flag = true
				flag_val = n.get_meta("is_hvaa")
			elif "is_hvaa" in n:
				has_flag = true
				flag_val = n.is_hvaa

			print("[HVAA DEBUG] Fighter:", n.name,
				" has is_hvaa property:", has_flag,
				" value:", (flag_val if has_flag else "N/A"))

			if flag_val:
				arr.append(n)

	print("[HVAA DEBUG] _get_hvaa_aircraft found ", arr.size(), " HVAA(s)")
	return arr


func _sample_point_in_annulus(cx: float, cz: float, r_in: float, r_out: float, y_alt: float) -> Vector3:
	var t := rng.randf()
	var r := sqrt((r_out*r_out - r_in*r_in) * t + r_in*r_in)
	var theta := rng.randf_range(0.0, TAU)
	var x := cx + r * cos(theta)
	var z := cz + r * sin(theta)
	return Vector3(x, y_alt, z)

func _sample_point_in_rect(rect: Dictionary, y_alt: float) -> Vector3:
	var x := rng.randf_range(float(rect.get("xmin", -10.0)), float(rect.get("xmax", 10.0)))
	var z := rng.randf_range(float(rect.get("zmin", -10.0)), float(rect.get("zmax", 10.0)))
	return Vector3(x, y_alt, z)

func _is_far_from_all(p: Vector3, pts: Array, min_sep: float) -> bool:
	#for q in pts:
		#if p.distance_to(q) < min_sep:
			#return false
	#return true
	# p and pts are in NM/ft; compare horizontal distance in NM only
	for q in pts:
		var qv: Vector3 = q as Vector3
		var dx: float = p.x - q.x   # NM
		var dz: float = p.z - q.z   # NM
		if sqrt(dx*dx + dz*dz) < min_sep:
			return false
	return true


func _face_velocity(node: Node3D, v: Vector3) -> void:
	if v.length() < 1e-3:
		return
	var forward := v.normalized()
	var up := Vector3.UP
	var right := up.cross(forward).normalized()
	var real_up := forward.cross(right).normalized()
	var basis := Basis(right, real_up, -forward)
	node.global_transform = Transform3D(basis, node.global_transform.origin)

# (Optional) Simple random offsets around AgentsConfig.blue_agents.init_position
func _apply_simple_offsets_from_agents_config(sim: Node, blue: Array):
	if not (simConfig.has("AgentsConfig") and simConfig["AgentsConfig"].has("blue_agents")):
		return
	var b: Dictionary = simConfig["AgentsConfig"]["blue_agents"]
	var base: Dictionary = b.get("init_position", {"x": 0.0, "y": 25000.0, "z": 0.0})
	var off: Dictionary = b.get("rnd_offset_range", {"x": 0.0, "y": 0.0, "z": 0.0})
	for f in blue:
		var px := float(base.get("x",0.0)) + rng.randf_range(-float(off.get("x",0.0)), float(off.get("x",0.0)))
		var py := float(base.get("y",0.0)) + rng.randf_range(-float(off.get("y",0.0)), float(off.get("y",0.0)))
		var pz := float(base.get("z",0.0)) + rng.randf_range(-float(off.get("z",0.0)), float(off.get("z",0.0)))
		var xf: Transform3D = f.global_transform
		xf.origin = Vector3(px, py, pz)
		f.global_transform = xf

func _send_dict_as_json_message(dict: Dictionary) -> void:
	var json_str = JSON.stringify(dict)
	var bytes = json_str.to_utf8_buffer()
	var length = bytes.size()

	# Build [4-byte little-endian length][json bytes]
	var header = PackedByteArray()
	header.resize(4)
	header[0] = length & 0xFF
	header[1] = (length >> 8) & 0xFF
	header[2] = (length >> 16) & 0xFF
	header[3] = (length >> 24) & 0xFF

	stream.put_data(header)
	stream.put_data(bytes)
	
func _send_env_info() -> void:
	print("DEBUG: Sending env_info to Python...")
	# Only send env info if we actually have a sim running
	if simulation_list == null:
		print("DEBUG: simulation_list is null")
		return
	if simulation_list.size() == 0:
		print("DEBUG: simulation_list is empty")
		return
	if simulation_list[0] == null:
		print("DEBUG: simulation_list[0] is null")
		return
	if simulation_list[0].agents == null:
		print("DEBUG: agents is null")
		return
	if simulation_list[0].agents.size() == 0:
		print("DEBUG: agents is empty")
		return

	# Build observation label map
	var observation_labels: Dictionary = {}
	for agent in simulation_list[0].agents:
		var obs_with_labels: Dictionary = agent.get_obs(true)
		# use string key because Python expects string keys
		observation_labels[str(agent.id)] = obs_with_labels["labels"]

	var first_agent = simulation_list[0].agents[0]

	var message: Dictionary = {
		"type": "env_info",
		"observation_space": first_agent.get_obs_space(),
		"observation_labels": observation_labels,
		"action_space": first_agent.get_action_space(),
		"n_agents": simulation_list[0].agents.size() }

	_send_dict_as_json_message(message)
	print("DEBUG: env_info sent successfully")


func connect_to_server() -> bool:
	#print("DEBUG: Trying to connect to Python server...")
	stream = StreamPeerTCP.new()

	var ip = "127.0.0.1"
	var port = _get_port()
	#print("DEBUG: Connecting to ", ip, ":", port)

	var err = stream.connect_to_host(ip, port)
	if err != OK and err != ERR_ALREADY_IN_USE:
		print("ERROR: connect_to_host returned ", err)
		return false

	# Wait up to ~2 seconds for connection to finish
	var elapsed := 0.0
	while elapsed < 2.0:
		stream.poll()
		var status = stream.get_status()
		if status == StreamPeerTCP.STATUS_CONNECTED:
			#print("DEBUG: Connected! stream status = ", status)
			return true
		OS.delay_msec(10)
		elapsed += 0.01

	var final_status = stream.get_status()
	print("DEBUG: Failed to connect after wait. Final status = ", final_status)
	return false

func _get_args():
	
	var arguments = {}
	for argument in OS.get_cmdline_args():
		if argument.find("=") > -1:
			var key_value = argument.split("=")
			arguments[key_value[0].lstrip("--")] = key_value[1]
		else:
			# Options without an argument will be present in the dictionary,
			# with the value set to an empty string.
			arguments[argument.lstrip("--")] = ""

	return arguments   

func _set_view_features():	
	mainCanvas.btn_speed_up.set_text(str(speed_up) + "X")	
		
func _get_port():    
	return args.get("port", DEFAULT_PORT).to_int()
		
func _create_simulations(_agents_config_dict, _experiment_cases=null):
	print("DEBUG _create_simulations: parallel_envs = ", parallel_envs)
	print("DEBUG _create_simulations: _agents_config_dict = ", _agents_config_dict)
	
	for container in mainViewPort.get_children():
		container.queue_free()	
	simulation_list = []
	
	var main_view_size = mainViewPort.get_rect().size
	var cols = min(parallel_envs, 5)
	var rows = ceil(float(parallel_envs) / cols)
	var sim_width = main_view_size.x / cols
	var sim_height = main_view_size.y / rows
	
	# Define a small border size (in pixels)
	var border_size = 1.0
	
	# Calculate the normalized border size as a fraction of the viewport size
	var border_x = border_size / main_view_size.x
	var border_y = border_size / main_view_size.y
	
	var x = 0
	var y = 0
	
	for i in range(parallel_envs):
				
		var case_config = _agents_config_dict.duplicate()
		
		if _experiment_cases != null:
			assert(len(_experiment_cases) == parallel_envs)
			case_config = _agents_config_dict.duplicate()
			update_dict(case_config, _experiment_cases[i]["AgentsConfig"])			
			
						
		var viewport_container = SimManager.instantiate()
		var viewport = viewport_container.get_node("SubViewport")
		var new_simulation = viewport.get_node("SimManager")		
		# Set the size of the viewport
		viewport.size = Vector2(sim_width, sim_height)		
		
		# Enable "Own World" on the viewport
		viewport.world_3d = World3D.new()
		
		# Tell the viewport to only render when we manually request it.
		viewport.render_target_update_mode = SubViewport.UPDATE_ONCE
		viewport.render_target_clear_mode = SubViewport.CLEAR_MODE_ALWAYS
		
		
		# Set the position of the ViewportContainer with a border
		viewport_container.anchor_left = float(x) / cols + (border_x * x)
		viewport_container.anchor_right = viewport_container.anchor_left + 1.0 / cols - 2 * border_x
		viewport_container.anchor_top = float(y) / rows + (border_y * y)
		viewport_container.anchor_bottom = viewport_container.anchor_top + 1.0 / rows - 2 * border_y

		mainViewPort.add_child(viewport_container)
		
		var _tree = get_tree()
		new_simulation.tree = _tree	
		
		new_simulation.initialize(i, _tree, envConfig, case_config)
		viewport.uavs = new_simulation.fighters

		simulation_list.append(new_simulation)

		x += 1
		if x >= cols:
			x = 0
			y += 1
	
	#print("DEBUG: Created ", simulation_list.size(), " simulations")

func disconnect_from_server():
	stream.disconnect_from_host()


func _initialize():
	
	args = _get_args()
	#print("DEBUG _initialize(): args = ", args)
	
	# CRITICAL CHANGE: Initialize with MINIMAL defaults
	# Don't load baked JSON - Python will send everything we need
	simConfig = {
		"EnvConfig": {
			"phy_fps": 20,
			"speed_up": 50,
			"renderize": 1,
			"action_repeat": 20,
			"parallel_envs": 1,
			"seed": 1,
			"action_type": "Low_Level_Continuous",
			"max_cycles": 36000,
			"experiment_mode": 0,
			"stop_mission": 1,
			"max_trail_size": 180,
			"RewardsConfig": {}
		},
		"AgentsConfig": {
			"blue_agents": {
				"num_agents": 2
			},
			"red_agents": {
				"num_agents": 2
			}
		}
	}
	
	print("DEBUG: Using minimal defaults, waiting for Python config...")

	# 2. Apply cmdline args (for standalone mode or overrides)
	update_dict(simConfig["EnvConfig"], args)

	# 4. Cache envConfig reference
	envConfig = simConfig["EnvConfig"]
	print("DEBUG envConfig (before Python) = ", envConfig)
	
	# 5. Seed RNG
	var seed_value := int(envConfig.get("seed", 1))
	rng.seed = seed_value
	rng.randomize()

	# 6. Pull physics/render params from minimal defaults
	phy_fps        = int(envConfig.get("phy_fps", 20))
	speed_up       = int(envConfig.get("speed_up", 50))
	renderize      = int(envConfig.get("renderize", 1))
	action_repeat  = int(envConfig.get("action_repeat", 20))
	parallel_envs  = int(envConfig.get("parallel_envs", 1))
	print("DEBUG parallel_envs = ", parallel_envs)
	experiment_mode= int(envConfig.get("experiment_mode", 0))

	# 7. Apply simulation timing
	Engine.physics_ticks_per_second = speed_up * phy_fps
	Engine.time_scale = speed_up * 1.0
	Engine.max_fps = 0
	RenderingServer.render_loop_enabled = false

	# 8. Try to connect to Python
	connected = connect_to_server()

	if connected:
		# PYTHON-CONTROLLED MODE
		#print("DEBUG: Connected to Python, waiting for configuration...")
		_handshake()
		_wait_for_configuration()  # This will receive FULL config from Python
		
		# Send env_info AFTER sims are created (only for non-experiment mode)
		if not experiment_mode:
			_send_env_info()
	else:
		# STANDALONE / EDITOR MODE - Try to load from baked file
		print("DEBUG: Not connected to Python, trying to load baked config...")
		var baked_config = load_json_file("res://assets/Default_Sim_Config.json")
		if typeof(baked_config) == TYPE_DICTIONARY and baked_config.size() > 0:
			#print("DEBUG: Loaded baked config successfully")
			if baked_config.has("EnvConfig"):
				update_dict(simConfig["EnvConfig"], baked_config["EnvConfig"])
			if baked_config.has("AgentsConfig"):
				update_dict(simConfig["AgentsConfig"], baked_config["AgentsConfig"])
		else:
			print("DEBUG: No baked config found, using minimal defaults")
		
		_create_simulations(simConfig["AgentsConfig"])
		initialized = true
		for sim in simulation_list:
			sim._reset_simulation()

	# 9. Visuals timing setup
	render_interval_msec = 1000.0 / visual_fps
	last_render_time_msec = Time.get_ticks_msec()

	_set_view_features()

func _check_all_sims_done() -> bool:
	if simulation_list == null:
		return false
	if simulation_list.size() == 0:
		return false

	for sim in simulation_list:
		if sim == null:
			return false
		if not sim.ready_to_reset:
			return false
	return true


func _physics_process(delta): 
		
	accumulated_sim_time += delta 
	
	var current_time_msec = Time.get_ticks_msec()
	if current_time_msec - last_render_time_msec >= render_interval_msec:
		# It's time to render a new frame.
		last_render_time_msec = current_time_msec

		if renderize == 1:
		# Request updates from all sub-viewports
			for container in mainViewPort.get_children():
				var viewport = container.get_node("SubViewport")
				viewport.render_target_update_mode = SubViewport.UPDATE_ONCE	            
				# Force the engine to draw this single, complete frame
				RenderingServer.force_draw()  
		
	if Time.get_ticks_msec() - last_check >= 100:
	
	# 1. Get the real time that passed during this interval (in seconds)
		var real_time_passed_sec = (Time.get_ticks_msec() - last_check) / 1000.0
		var practical_speed_up = 0.0
		if real_time_passed_sec > 0:
			practical_speed_up = accumulated_sim_time / real_time_passed_sec
		# --- 2. Dynamic Visual FPS Adjustment Logic ---
		if speed_up >= 1: # Only run logic when sped up
			# Get the performance ratio (e.g., 0.9 means we're running at 90% of target speed)
			var performance_ratio = practical_speed_up / speed_up 
			
			if performance_ratio < 0.80:
				# PERFORMANCE IS POOR: Reduce visual FPS to free up resources.
				visual_fps -= 2.0 # Decrease by 2 FPS
			else:
				# PERFORMANCE IS GOOD: Gradually increase visual FPS back to the target.
				visual_fps += 1.0 # Increase by 1 FPS

			visual_fps = clamp(visual_fps, 5, 60)
			render_interval_msec = 1000.0 / visual_fps
			
			phy_show.text = "%.0f" % practical_speed_up
			accumulated_sim_time = 0.0
			last_check = Time.get_ticks_msec()
				
			
	if n_action_steps % action_repeat != 0 and not stop_simulation:
		n_action_steps += 1						
		return
	
	steps_show.text = str(n_action_steps/action_repeat )
	#Reach This part only every ActionRepeat Steps
	n_action_steps += 1			
	
	if connected:		
		#RL Mode
		if not experiment_mode:
			get_tree().set_pause(true) 
			
			if just_reset:		
							
				just_reset = false
				var obs = _get_obs_from_simulations()				
				
				var obs_dict = {}
				var i = 0
				for agent in simulation_list[0].agents:
					obs_dict[agent.agent_name] = obs[i]
					i = i + 1					
			
				var reply = {
					"type": "reset",
					"obs": obs_dict
				}
				_send_dict_as_json_message(reply)
				# this should go straight to getting the action and setting it checked the agent, no need to perform one phyics tick
				get_tree().set_pause(false) 
				return
			
			if need_to_send_obs:
				need_to_send_obs = false
				var reward = _get_reward_from_simulations()
				var done = _get_dones_from_simulations_agents()
				var info = _get_info_from_simulations() 
				var obs = _get_obs_from_simulations()
				
				var obs_dict  	= {}
				var done_dict 	= {}
				var reward_dict = {}
				var info_dict = {}
				
				var i = 0
				for agent in simulation_list[0].agents:
					obs_dict[agent.agent_name] = obs[i]
					done_dict[agent.agent_name] = done[i]
					reward_dict[agent.agent_name] = reward[i]
					info_dict[agent.agent_name] = info[i] 
					i = i + 1

									
				var reply = {
					"type": "step",
					"obs": obs_dict,
					"reward": reward_dict,
					"done": done_dict,
					"info": info_dict 
				}
				_send_dict_as_json_message(reply)
								
			var handled = handle_message()
		
		#Experiment MODE 
		elif experiment_in_progress:			
						
			_get_reward_from_simulations()
									
			if _check_all_sims_done():
				experiment_current_run += 1
				
				var result = _collect_experiment_result(experiment_current_run)
				
				experiment_results.append(result)

				if experiment_current_run >= experiment_runs_per_case:
					var reply = {
						"type": "experiment_results",
						"results": experiment_results
					}
					_send_dict_as_json_message(reply)
					
					experiment_in_progress = false
													
					_wait_for_configuration()
				else:
					var reply = {
						"type": "experiment_step",
						"run_finished": str(experiment_current_run) 
					}
					_send_dict_as_json_message(reply)
					
					_reset_all_sims()
														
	#Not Connected
	else:					
		_check_all_sims_done()
		_get_reward_from_simulations()
		var obs = _get_obs_from_simulations()
			
		_reset_agents_if_done()	
		
func handle_message() -> bool:
	# Read one JSON message from Python
	var message: Dictionary = _get_dict_json_message()
	var mtype: String = str(message.get("type", ""))

	#if mtype != "action":
	#	print("DEBUG handle_message: received type = ", mtype)

	# ---------- 1. LIVE CONFIG UPDATE ----------
	if mtype == "config":
		#print("DEBUG: Processing config message from Python...")

		# env overrides
		var env_config_msg: Dictionary = message.get("env_config", {})
		print("DEBUG: env_config from Python = ", env_config_msg)
		update_dict(envConfig, env_config_msg)
		update_dict(simConfig["EnvConfig"], env_config_msg)

		phy_fps         = int(envConfig.get("phy_fps", phy_fps))
		speed_up        = int(envConfig.get("speed_up", speed_up))
		renderize       = int(envConfig.get("renderize", renderize))
		action_repeat   = int(envConfig.get("action_repeat", action_repeat))
		parallel_envs   = int(envConfig.get("parallel_envs", parallel_envs))
		experiment_mode = int(envConfig.get("experiment_mode", experiment_mode))

		Engine.physics_ticks_per_second = speed_up * phy_fps
		Engine.time_scale = speed_up * 1.0

		# agent overrides
		var incoming_agents: Dictionary = message.get("agents_config", {})
		print("DEBUG: agents_config from Python = ", incoming_agents)
		var merged_agents: Dictionary = (simConfig["AgentsConfig"] as Dictionary).duplicate(true)
		update_dict(merged_agents, incoming_agents)
		simConfig["AgentsConfig"] = merged_agents

		# experiment or normal sim setup
		if experiment_mode == 1 and message.has("experiment_config"):
			_run_experiment(merged_agents, message["experiment_config"])
		else:
			#print("DEBUG: Creating simulations...")
			_create_simulations(merged_agents)
			initialized = true
			#print("DEBUG: Resetting simulations...")
			for sim in simulation_list:
				sim._reset_simulation()

		#print("DEBUG: Sending ACK to Python...")
		_send_dict_as_json_message({"type": "ack", "event": "config_ok"})
		get_tree().set_pause(false)
		#print("DEBUG: Configuration complete, unpaused")

		return true

	# ---------- 2. CLOSE ----------
	if mtype == "close":
		print("received close message, closing game")
		for sim in simulation_list:
			sim.queue_free()
		simulation_list.clear()
		get_tree().quit()
		get_tree().set_pause(false)
		return true

	# ---------- 3. RESET EPISODE ----------
	if mtype == "reset":
		if simulation_list.size() == 1:
			var results_single = simulation_list[0]._collect_results()
			mainCanvas.update_results(results_single)
			mainCanvas.update_scores(results_single[1]["killed"], results_single[0]["killed"])
			simulation_list[0]._reset_simulation()
		else:
			for idx in range(simulation_list.size()):
				simulation_list[idx]._reset_simulation()
		
		_randomize_blue_spawns_for_all_sims()
		#for agent in simulation_list[0].agents:
		#	if agent.team_id == 0:
		#		print("[POST-RESET] BLUE agent at ", agent.global_transform.origin)
		just_reset = true
		n_action_steps = 0
		get_tree().set_pause(false)
		return true

	# ---------- 4. CALL (debug/introspection) ----------
	if mtype == "call":
		var method := message.get("method", "")
		var results = ""
		if method == "last":
			results = simulation_list[0]._collect_last_results()
		else:
			results = simulation_list[0]._collect_last_results()

		var reply = {
			"type": "call",
			"returns": results
		}
		_send_dict_as_json_message(reply)
		get_tree().set_pause(false)
		return true
	
	# ---------- 5. ACTION STEP FROM PYTHON ----------
	if mtype == "action":
		var actions = message["action"]
		if simulation_list.size() == 1:
			simulation_list[0]._set_agent_actions(actions)
		else:
			for idx in range(simulation_list.size()):
				simulation_list[idx]._set_agent_actions(actions[idx])

		need_to_send_obs = true
		get_tree().set_pause(false)
		return true

	# ---------- 6. EXPERIMENT CONFIG (per-step tweak) ----------
	if mtype == "experiment_config":
		var exp_cfg := message.get("config", {})
		for sim in simulation_list:
			sim.handle_experiment_config(exp_cfg)
		get_tree().set_pause(false)
		return true

	print("handle_message(): message not handled: ", mtype)
	return false
	
func _run_experiment(agents_config_msg, experiment_config):
	
	print("Running experiment with configuration:", experiment_config)
	
	experiment_runs_per_case = experiment_config.get("runs_per_case", 10)
	envConfig["parallel_envs"] = len(experiment_config['cases'])
	parallel_envs 	= envConfig["parallel_envs"]
	
	
	# Create simulations based on the experiment configuration
	_create_simulations(agents_config_msg, experiment_config.get("cases", null))
	
	initialized = true
	experiment_in_progress = true
	experiment_results = []
	experiment_current_run = 0

	_reset_all_sims()
	get_tree().set_pause(false)  
	
func _collect_experiment_result(run_num):
	var _results = []
	for simulation in simulation_list: 
		
		var final_results = simulation._collect_results()
		
		var sim = simulation
		final_results[0]["beh"] = [sim.fighters[0].dShot, sim.fighters[0].lCrank, sim.fighters[0].lBreak]
		final_results[1]["beh"] = [sim.fighters[1].dShot, sim.fighters[1].lCrank, sim.fighters[1].lBreak]
		mainCanvas.update_results(final_results)		
		mainCanvas.update_scores(final_results[1]['killed'], final_results[0]['killed'])
		
		var sim_result = {
			"env_id" : simulation.id,
			"run_num": run_num,
			"final_results" : final_results		
		}
							
		_results.append(sim_result)
		
	return _results
		

func are_all_true(array):
	
	for value in array:
		if not value:
			return false
	return true

func update_dict(org_dict: Dictionary, new_config: Dictionary):
	for key in new_config:
		var new_val = new_config[key]
		if typeof(new_val) == TYPE_DICTIONARY:
			if org_dict.has(key) and typeof(org_dict[key]) == TYPE_DICTIONARY:
				update_nested_dict(org_dict[key], new_val)
			else:
				# if it doesn't exist yet, just copy the whole dictionary in
				org_dict[key] = new_val.duplicate(true)
		else:
			# allow brand new scalar keys too
			org_dict[key] = new_val

func update_nested_dict(existing_dict: Dictionary, new_dict: Dictionary):
	for key in new_dict:
		var new_val = new_dict[key]
		if typeof(new_val) == TYPE_DICTIONARY:
			if existing_dict.has(key) and typeof(existing_dict[key]) == TYPE_DICTIONARY:
				update_nested_dict(existing_dict[key], new_val)
			else:
				existing_dict[key] = new_val.duplicate(true)
		else:
			existing_dict[key] = new_val
				
func load_json_file(file_path: String):
	var file = FileAccess.open(file_path, FileAccess.READ)
	if file:
		var json_string = file.get_as_text()
		file.close()

		var json = JSON.new()
		var parse_result = json.parse(json_string)

		if parse_result == OK:
			return json.get_data()
		else:
			print("JSON Parse Error: ", json.get_error_message())
			return null
	else:
		print("File not found: ", file_path)
		return null

func _reset_agents_if_done() -> Array:
	var finalStatus: Array = []

	if simulation_list == null or simulation_list.size() == 0:
		return finalStatus

	var index := 0
	for sim in simulation_list:
		if sim == null:
			finalStatus.append(null)
			continue

		if sim.ready_to_reset:
			n_action_steps = 0

			var results = sim._collect_results()
			finalStatus.append(results)

			mainCanvas.update_results(results)
			mainCanvas.update_scores(results[1]["killed"], results[0]["killed"])

			sim._reset_simulation()
		else:
			finalStatus.append(null)

		index += 1

	return finalStatus
					
func _input(event):	
	
	if Input.is_action_just_pressed("r_key"):
		just_reset = true
		n_action_steps = 0			
		_reset_all_sims()
	
func clamp_array(arr : Array, min:float, max:float):
	var output : Array = []
	for a in arr:
		output.append(clamp(a, min, max))
	return output	

func _reset_all_sims():	
	for sim in simulation_list:		
		sim._reset_simulation()

	_randomize_blue_spawns_for_all_sims()
	_randomize_hvaa_spawns_for_all_sims()

	# Force all fighters to recompute their radar tracks
	for sim in simulation_list:
		for fighter in sim.fighters:
			if fighter.activated:
				fighter.process_tracks()
	
		
func _reset_all_sim(sim_index):	
	simulation_list[sim_index]._reset_simulation()

	
		
func _get_obs_from_simulations():
	if simulation_list == null or simulation_list.size() == 0:
		return []

	if simulation_list.size() == 1:
		return simulation_list[0]._get_obs_from_agents()

	var envs_obs: Array = []
	for sim in simulation_list:
		if sim != null:
			envs_obs.append(sim._get_obs_from_agents())
		else:
			envs_obs.append([])
	return envs_obs


func _get_reward_from_simulations():
	if simulation_list == null or simulation_list.size() == 0:
		return []

	if simulation_list.size() == 1:
		return simulation_list[0]._get_reward_from_agents()

	var envs_rews: Array = []
	for sim in simulation_list:
		if sim != null:
			envs_rews.append(sim._get_reward_from_agents())
		else:
			envs_rews.append([])
	return envs_rews
	
func _get_dones_from_simulations_agents():
	if simulation_list == null or simulation_list.size() == 0:
		return []

	if simulation_list.size() == 1:
		return simulation_list[0]._get_done_from_agents()

	var agents_dones: Array = []
	for sim in simulation_list:
		if sim != null:
			agents_dones.append(sim._get_done_from_agents())
		else:
			agents_dones.append([])
	return agents_dones
	
func _get_dones_from_simulations_enemies():
	if simulation_list == null or simulation_list.size() == 0:
		return []

	if simulation_list.size() == 1:
		return simulation_list[0]._get_done_from_enemies()

	var enemies_dones: Array = []
	for sim in simulation_list:
		if sim != null:
			enemies_dones.append(sim._get_done_from_enemies())
		else:
			enemies_dones.append([])
	return enemies_dones

func _get_info_from_simulations():
	"""Get info from all simulations (mirrors _get_reward_from_simulations)"""
	if simulation_list == null or simulation_list.size() == 0:
		return []

	if simulation_list.size() == 1:
		return simulation_list[0]._get_info_from_agents()

	var envs_info: Array = []
	for sim in simulation_list:
		if sim != null:
			envs_info.append(sim._get_info_from_agents())
		else:
			envs_info.append([])
	return envs_info

func _wait_for_configuration():
	print("DEBUG: Waiting for config message from Python...")
	var config_message: Dictionary = _get_dict_json_message()
	var mtype: String = str(config_message.get("type", ""))

	print("DEBUG: Received message type: ", mtype)

	if mtype == "close":
		get_tree().quit()
		get_tree().set_pause(false)
		return true

	if mtype != "config":
		print("ERROR: Wrong message received, expected 'config' or 'close', got ", mtype)
		get_tree().quit()
		get_tree().set_pause(false)
		return true

	print("DEBUG: Processing config from Python...")

	# Apply env config
	var env_config_msg: Dictionary = config_message.get("env_config", {})
	#print("DEBUG: env_config = ", env_config_msg)
	update_dict(envConfig, env_config_msg)
	update_dict(simConfig["EnvConfig"], env_config_msg)

	phy_fps         = int(envConfig.get("phy_fps", phy_fps))
	speed_up        = int(envConfig.get("speed_up", speed_up))
	renderize       = int(envConfig.get("renderize", renderize))
	action_repeat   = int(envConfig.get("action_repeat", action_repeat))
	parallel_envs   = int(envConfig.get("parallel_envs", parallel_envs))
	experiment_mode = int(envConfig.get("experiment_mode", experiment_mode))

	#print("DEBUG: Applied settings - speed_up:", speed_up, " parallel_envs:", parallel_envs)

	Engine.physics_ticks_per_second = speed_up * phy_fps
	Engine.time_scale = speed_up * 1.0

	# Merge agent config
	var agents_config_msg: Dictionary = config_message.get("agents_config", {})
	#print("DEBUG: agents_config = ", agents_config_msg)
	var merged_agents: Dictionary = (simConfig["AgentsConfig"] as Dictionary).duplicate(true)
	update_dict(merged_agents, agents_config_msg)
	simConfig["AgentsConfig"] = merged_agents

	# Create either experiment or normal sims
	if experiment_mode == 1 and config_message.has("experiment_config"):
		_run_experiment(merged_agents, config_message["experiment_config"])
	else:
		_create_simulations(merged_agents)
		initialized = true
		for sim in simulation_list:
			sim._reset_simulation()
		_randomize_blue_spawns_for_all_sims()
		_randomize_hvaa_spawns_for_all_sims()

	# Tell Python we're live
	#print("DEBUG: Sending ACK to Python...")
	_send_dict_as_json_message({"type": "ack", "event": "config_ok"})
	get_tree().set_pause(false)
	#print("DEBUG: Initial configuration complete, unpaused")

	return true
