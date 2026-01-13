#Should be OLD CODE

extends Control

var goals = []
var goalsPending = []
var uavs = []

var cameraGlobal: Camera3D
var cameraUav: Camera3D
var uavCamId = 0

const UAV_CAM_SCALE_VECTOR = Vector3(3.0, 3.0, 3.0)
const GLOBAL_CAM_SCALE_VECTOR = Vector3(5.0, 5.0, 5.0)

var mouse_sens = 0.1
var camera_angle_v = 0
var camera_angle_h = 0

var zoom_level = 100
var move_speed = 5

@onready var canvas = get_node("CanvasLayer")
var fighterObj = preload("res://components/Fighter.tscn")

var rng = RandomNumberGenerator.new()
var numTasksDone = 0

func _ready():	
	
	cameraGlobal = get_node("CameraGlobal")
	cameraGlobal.make_current()
	
		# ADD THESE LINES:
	cameraGlobal.position = Vector3(0, 500, 400)  # Zoom out
	cameraGlobal.fov = 100                         # Wide angle
	cameraGlobal.rotation_degrees.x = -40          # Look down
	
	
	self.position.y = zoom_level
	RenderingServer.render_loop_enabled = true
	
