extends SceneTree
## Phase 5 / Step 1 - runtime texture-swap spike.
##
## Confirms that a runtime albedo swap works and is visible, and measures the cost.
## A probe quad is placed directly in front of the camera so the comparison is not
## confounded by scene occlusion/framing:
##   00 - the imported 2048 texture (VRAM-compressed, res://)
##   01 - the 256 variant      (Image.load_from_file, uncompressed)
##   02 - the original 2048, loaded uncompressed (the "full" texture)
##
## It also demonstrates the per-object material duplication that the streamer needs
## (the shared .tres material is aliased across every object, so it must be
## duplicated before its albedo can be changed).
##
## Run:
##   godot --path godot --resolution 1280x720 --script res://tests/swap_spike.gd -- \
##       --out=dataset/streaming/runs/step1

const STREAM_IDS := [
	"ambientcg/Asphalt021_Color", "ambientcg/Bricks066_Color", "ambientcg/Concrete012_Color",
	"ambientcg/Grass004_Color", "ambientcg/Ground037_Color", "ambientcg/Metal063_Color",
	"ambientcg/Road001_Color",
]

class LoadTask:
	extends RefCounted
	var path := ""
	var image: Image = null
	var load_ms := 0.0
	var mipmap_ms := 0.0
	var error := -1
	func run() -> void:
		var t0 := Time.get_ticks_usec()
		var img: Image = Image.load_from_file(path)
		load_ms = float(Time.get_ticks_usec() - t0) / 1000.0
		if img == null or img.is_empty():
			error = FAILED
			return
		error = OK
		var t1 := Time.get_ticks_usec()
		img.generate_mipmaps()
		mipmap_ms = float(Time.get_ticks_usec() - t1) / 1000.0
		image = img

var _out := ""
var _shots := ""
var _report: Dictionary = {"step": "swap_spike", "measures": {}, "checks": {}}
var _watchdog := 0

func _initialize() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			_out = a.trim_prefix("--out=")
	_run.call_deferred()

func _process(_delta: float) -> bool:
	_watchdog += 1
	if _watchdog > 3000:
		push_error("[swap_spike] watchdog timeout")
		quit(3)
		return true
	return false

func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()

func _variant_path(albedo_res: String, res: int) -> String:
	if not albedo_res.begins_with("res://"):
		return ""
	var rel := albedo_res.trim_prefix("res://")
	var dir := rel.get_base_dir()
	var stem := rel.get_file().get_basename()
	var pack := dir.get_file()
	return _repo_root().path_join("dataset/textures/variants").path_join(pack).path_join(
		"%s_%d.jpg" % [stem, res])

func _pick_target(env: Node) -> Dictionary:
	var best: Dictionary = {}
	for entry in env.get("tracked_objects"):
		var ids: PackedStringArray = entry.get("textures", PackedStringArray())
		var hit := false
		for sid in STREAM_IDS:
			if ids.has(sid):
				hit = true
				break
		if not hit:
			continue
		if String(entry.get("category", "")) == "shw":
			return entry
		if best.is_empty():
			best = entry
	return best

func _shot(name: String) -> void:
	await process_frame
	await RenderingServer.frame_post_draw
	var img := root.get_texture().get_image()
	img.save_png(_shots.path_join(name + ".png"))
	print("[swap_spike] shot ", name)

func _run() -> void:
	if _out.is_empty():
		_out = "dataset/streaming/runs/step1"
	if not _out.begins_with("/"):
		_out = _repo_root().path_join(_out)
	DirAccess.make_dir_recursive_absolute(_out)
	_shots = _out.path_join("shots")
	DirAccess.make_dir_recursive_absolute(_shots)

	var packed := load("res://scenes/main.tscn") as PackedScene
	var main: Node = packed.instantiate()
	root.add_child(main)
	var recorder: Node = main.find_child("TelemetryRecorder", true, false)
	if recorder != null:
		recorder.set("auto_start", false)
		recorder.set("enabled", false)
		recorder.set("record_jsonl", false)
		recorder.set("record_csv", false)
	for i in 6:
		await process_frame

	var env: Node = get_first_node_in_group("environment")
	var player: Node3D = get_first_node_in_group("player") as Node3D
	var camera: Camera3D = get_first_node_in_group("player_camera") as Camera3D
	var target := _pick_target(env)
	if target.is_empty() or player == null or camera == null:
		push_error("[swap_spike] no target object or player")
		_report["error"] = "no_target"
		_write_report()
		quit(1)
		return

	var node: Node3D = target["node"]
	var mi: MeshInstance3D = node.find_children("*", "MeshInstance3D", true, false)[0]
	var original := mi.material_override as StandardMaterial3D
	var original_texture := original.albedo_texture
	var albedo_res := original_texture.resource_path
	_report["object_id"] = target.get("object_id", "")
	_report["category"] = target.get("category", "")
	_report["texture_ids"] = Array(target.get("textures", []))
	_report["albedo_resource_path"] = albedo_res

	# --- demonstrate per-object material duplication (the streamer's real need) ---
	var dup_t0 := Time.get_ticks_usec()
	var duplicated := original.duplicate() as StandardMaterial3D
	mi.material_override = duplicated
	_report["measures"]["duplicate_material_ms"] = float(Time.get_ticks_usec() - dup_t0) / 1000.0
	_report["checks"]["duplication_isolation"] = (
		duplicated != original
		and duplicated.albedo_texture == original.albedo_texture
		and original.albedo_texture.resource_path != "")

	# --- park the player and put a probe quad right in front of the camera --------
	player.global_position = Vector3(0.0, 1.0, -28.0)
	player.rotation.y = 0.0
	player.velocity = Vector3.ZERO
	player.set_physics_process(false)
	for i in 2:
		await process_frame

	var forward := -camera.global_transform.basis.z
	var probe := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = Vector3(2.6, 2.6, 0.12)
	probe.mesh = box
	var probe_mat := StandardMaterial3D.new()
	probe_mat.albedo_texture = original_texture
	probe_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	probe.material_override = probe_mat
	main.add_child(probe)
	probe.global_position = camera.global_position + forward * 4.0
	probe.look_at(camera.global_position, Vector3.UP)
	for i in 3:
		await process_frame
	await _shot("00_imported_2048")
	_report["checks"]["imported_texture_width"] = original_texture.get_width()

	# --- swap to the 256 variant --------------------------------------------------
	var path256 := _variant_path(albedo_res, 256)
	var path1024 := _variant_path(albedo_res, 1024)
	_report["path_256"] = path256
	_report["path_1024"] = path1024
	if path256.is_empty() or not FileAccess.file_exists(path256):
		_report["error"] = "absolute_path_failed"
		_write_report()
		print("[swap_spike] KILL: variant path not found: ", path256)
		quit(2)
		return
	_report["checks"]["absolute_path_found"] = true

	var t0 := Time.get_ticks_usec()
	var img256: Image = Image.load_from_file(path256)
	_report["measures"]["load256_ms"] = float(Time.get_ticks_usec() - t0) / 1000.0
	if img256 == null or img256.is_empty():
		_report["error"] = "absolute_path_failed"
		_write_report()
		print("[swap_spike] KILL: Image.load_from_file failed: ", path256)
		quit(2)
		return
	var t1 := Time.get_ticks_usec()
	img256.generate_mipmaps()
	_report["measures"]["mipmap256_ms"] = float(Time.get_ticks_usec() - t1) / 1000.0
	var t2 := Time.get_ticks_usec()
	var tex256 := ImageTexture.create_from_image(img256)
	_report["measures"]["create256_ms"] = float(Time.get_ticks_usec() - t2) / 1000.0
	var t3 := Time.get_ticks_usec()
	probe_mat.albedo_texture = tex256
	_report["measures"]["assign256_ms"] = float(Time.get_ticks_usec() - t3) / 1000.0
	_report["bytes_256"] = _bytes(img256)
	_report["checks"]["swapped_texture_width_256"] = probe_mat.albedo_texture.get_width()
	for i in 3:
		await process_frame
	await _shot("01_variant_256")

	# --- thread-load the 1024 variant (largest generated variant) ------------------
	var task1024 := LoadTask.new()
	task1024.path = path1024
	var id1024 := WorkerThreadPool.add_task(Callable(task1024, "run"))
	WorkerThreadPool.wait_for_task_completion(id1024)
	_report["measures"]["thread_load1024_ms"] = task1024.load_ms
	_report["measures"]["thread_mipmap1024_ms"] = task1024.mipmap_ms
	if task1024.error == OK:
		_report["bytes_1024"] = _bytes(task1024.image)

	# --- thread-load the FULL texture (original 2048, uncompressed) ----------------
	var path_full := ProjectSettings.globalize_path(albedo_res)
	_report["path_full"] = path_full
	var task := LoadTask.new()
	task.path = path_full
	var wall0 := Time.get_ticks_usec()
	var id_full := WorkerThreadPool.add_task(Callable(task, "run"))
	WorkerThreadPool.wait_for_task_completion(id_full)
	_report["measures"]["thread_wall_wait_ms"] = float(Time.get_ticks_usec() - wall0) / 1000.0
	_report["measures"]["thread_load_full_ms"] = task.load_ms
	_report["measures"]["thread_mipmap_full_ms"] = task.mipmap_ms
	if task.error != OK:
		_report["error"] = "full_load_failed"
		_write_report()
		quit(2)
		return
	var hitch0 := Time.get_ticks_usec()
	var tex_full := ImageTexture.create_from_image(task.image)
	probe_mat.albedo_texture = tex_full
	_report["measures"]["main_thread_hitch_ms"] = float(Time.get_ticks_usec() - hitch0) / 1000.0
	_report["bytes_full"] = _bytes(task.image)
	_report["checks"]["swapped_texture_width_full"] = probe_mat.albedo_texture.get_width()
	for i in 3:
		await process_frame
	await _shot("02_full_2048_loaded")

	_report["ok"] = true
	_write_report()
	for i in 3:
		await process_frame
	quit(0)

func _bytes(img: Image) -> int:
	return int(img.get_width()) * int(img.get_height()) * 4 * 4 / 3

func _write_report() -> void:
	var f := FileAccess.open(_out.path_join("swap_spike.json"), FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(_report, "  "))
	print("[swap_spike] report -> ", _out.path_join("swap_spike.json"))
	print(JSON.stringify(_report))
