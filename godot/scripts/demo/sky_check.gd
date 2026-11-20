extends SceneTree
## Item 3 check: render environment.tscn from a recorded trajectory pose in the MAIN
## viewport (no SubViewport) and save one frame, to compare with a demo pane.
##
##   godot --path godot --windowed --script res://scripts/demo/sky_check.gd -- \
##       --out=/tmp/x.png --trajectory=t06_look_around --sample=71 [--yaw-offset=180] [--pitch=35]

var _out := "/tmp/sky_main.png"
var _trajectory := "t06_look_around"
var _sample := 71
var _pitch := 0.0
var _yaw_offset := 0.0

func _initialize() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			_out = a.trim_prefix("--out=")
		elif a.begins_with("--trajectory="):
			_trajectory = a.trim_prefix("--trajectory=")
		elif a.begins_with("--sample="):
			_sample = int(a.trim_prefix("--sample="))
		elif a.begins_with("--pitch="):
			_pitch = float(a.trim_prefix("--pitch="))
		elif a.begins_with("--yaw-offset="):
			_yaw_offset = float(a.trim_prefix("--yaw-offset="))
	_run.call_deferred()

func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()

func _run() -> void:
	root.content_scale_mode = Window.CONTENT_SCALE_MODE_DISABLED
	var world := Node3D.new()
	root.add_child(world)
	world.add_child((load("res://scenes/environment.tscn") as PackedScene).instantiate())

	var path := _repo_root().path_join("dataset/streaming/runs/phase5_engine/scores/%s.json" % _trajectory)
	var scores: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	var cam := Camera3D.new()
	cam.current = true
	cam.fov = 75.0
	if scores is Dictionary:
		var p: Dictionary = scores["pose"][str(_sample)]
		var pos: Array = p["position"]
		var q: Array = p["quaternion"]
		var basis := Basis(Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
		basis = basis * Basis.from_euler(Vector3(deg_to_rad(_pitch), 0.0, 0.0))
		basis = Basis(Vector3.UP, deg_to_rad(_yaw_offset)) * basis
		cam.transform = Transform3D(basis, Vector3(float(pos[0]), float(pos[1]), float(pos[2])))
	world.add_child(cam)

	for i in 45:
		await process_frame
	var image := root.get_texture().get_image()
	if image != null:
		image.save_png(_out)
		print("[skycheck] saved %s %dx%d" % [_out, image.get_width(), image.get_height()])
		var w := image.get_width()
		var h := image.get_height()
		var total := 0.0
		var n := 0
		for y in range(h / 6, h / 2, 8):
			for x in range(w / 5, w * 4 / 5, 8):
				var c := image.get_pixel(x, y)
				total += (c.r + c.g + c.b) / 3.0
				n += 1
		print("[skycheck] upper mean=%.4f" % (total / maxf(1.0, float(n))))
	quit(0)
