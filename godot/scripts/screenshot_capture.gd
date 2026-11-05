extends Node
## Development helper: renders the running scene to a PNG and quits.
##
## Arguments are passed after `--` so they do not clash with Godot's own options:
##
##   godot --path godot -- --capture=shot.png --capture-frames=120 \
##         --capture-pos=0,1,-34 --capture-yaw=180
##
## Without `--capture` the node does nothing, so it is harmless in normal play
## sessions.

@export var default_wait_frames := 90

var _path := ""
var _wait_frames := -1
var _position := Vector3.INF
var _yaw_degrees := INF
var _pitch_degrees := INF
var _counter := 0
var _finished := false

func _ready() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--capture="):
			_path = argument.trim_prefix("--capture=")
		elif argument.begins_with("--capture-frames="):
			_wait_frames = int(argument.trim_prefix("--capture-frames="))
		elif argument.begins_with("--capture-pos="):
			_position = _parse_vector(argument.trim_prefix("--capture-pos="))
		elif argument.begins_with("--capture-yaw="):
			_yaw_degrees = float(argument.trim_prefix("--capture-yaw="))
		elif argument.begins_with("--capture-pitch="):
			_pitch_degrees = float(argument.trim_prefix("--capture-pitch="))
	if _wait_frames < 0:
		_wait_frames = default_wait_frames
	if _path.is_empty():
		set_process(false)
		return
	call_deferred("_apply_camera_override")

func _apply_camera_override() -> void:
	var player := get_tree().get_first_node_in_group("player") as Node3D
	if player == null:
		return
	if _position != Vector3.INF:
		player.global_position = _position
	if _yaw_degrees != INF:
		player.rotation.y = deg_to_rad(_yaw_degrees)
	if _pitch_degrees != INF:
		var camera := get_tree().get_first_node_in_group("player_camera") as Camera3D
		if camera != null:
			camera.rotation.x = deg_to_rad(_pitch_degrees)

func _process(_delta: float) -> void:
	if _path.is_empty() or _finished:
		return
	_counter += 1
	if _counter < _wait_frames:
		return
	_finished = true
	_capture()

func _capture() -> void:
	await RenderingServer.frame_post_draw
	var image := get_viewport().get_texture().get_image()
	var error := image.save_png(_path)
	if error == OK:
		print("[ScreenshotCapture] Saved %s" % _path)
	else:
		push_error("[ScreenshotCapture] Failed to save %s (error %d)" % [_path, error])
	get_tree().quit()

func _parse_vector(text: String) -> Vector3:
	var parts := text.split(",")
	if parts.size() != 3:
		return Vector3.INF
	return Vector3(float(parts[0]), float(parts[1]), float(parts[2]))
