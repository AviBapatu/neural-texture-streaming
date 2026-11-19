class_name StreamingController
extends Node
## Phase 5 / Step 5 - live in-engine streaming demo.
##
## Each 10 Hz sample it (1) computes live visibility for the streamable objects with
## VisibilityService, (2) turns a policy into per-object priorities, (3) advances the
## streamer (TextureStreamer.StreamState - the same code the parity test validates),
## and (4) mirrors residency onto real materials: a 256-variant placeholder, and the
## full 2048 texture loaded on a worker thread.
##
## Keys:
##   1 reactive_visible  2 heuristic_occ  3 mlp_b (scored replay)  4 mlp_b live
##   (4 is unavailable - see docs/phase-5-results.md §6)  5 no streaming
##   0 apply the preset cell (budget 100%, 200 MB/s, latency 1.0 s)
##   F1 toggle the residency tint
##   [ / ]  latency -/+ 0.1 s          - / =  budget -/+ 50%
##   P      print a metrics snapshot
##
## The 53-object scope is stated everywhere: it streams the objects that use one of
## the 7 ambientCG identities that have generated variants.

signal sample_published(stats)

const VARIANT_IDS := [
	"ambientcg/Asphalt021_Color", "ambientcg/Bricks066_Color", "ambientcg/Concrete012_Color",
	"ambientcg/Grass004_Color", "ambientcg/Ground037_Color", "ambientcg/Metal063_Color",
	"ambientcg/Road001_Color",
]
const POLICY_NAMES := ["reactive_visible", "heuristic_occ", "mlp_b", "no_streaming"]
const CANDIDATE_DISTANCE := 25.0
const TINT_PLACEHOLDER := Color(1.0, 1.0, 1.0)
const TINT_LOADING := Color(1.0, 0.92, 0.4)
const TINT_RESIDENT := Color(1.0, 1.0, 1.0)
const TINT_WASTED := Color(1.0, 0.4, 0.4)

class LoadTask:
	extends RefCounted
	var path := ""
	var image: Image = null
	var ms := 0.0
	func run() -> void:
		var t0 := Time.get_ticks_usec()
		var img: Image = Image.load_from_file(path)
		ms = float(Time.get_ticks_usec() - t0) / 1000.0
		if img != null and not img.is_empty():
			img.generate_mipmaps()
			image = img

var camera: Camera3D
var tracked: Array = []
var runner: Node
var score_rows: Dictionary = {}          # "sample" -> [[oid, cov, frustum, occ, hraw, mlp05, mlp10]]

var policy := "reactive_visible"
var budget_bytes := 134217728.0
var base_peak_bytes := 134217728.0
var latency_seconds := 0.5
var bandwidth_bytes_per_s := 200000000.0
var dt := 0.1
var apply_textures := true

var streamable: Array = []               # entries: {oid, node, mi, mat, texture_id, source_path, full_bytes, centre, bounds}
var by_id: Dictionary = {}
var sizes: Dictionary = {}
var centres: Dictionary = {}

var tint_enabled := true
var samples := 0
var _visibility := VisibilityService.new()
var _state := TextureStreamer.StreamState.new()
var _transport: TransportModel
var _sample_index := 0
var _tick := 0
var _viewport := Vector2(1280, 720)
var _last_load_ms := 0.0
var _last_hitch_ms := 0.0
var _pending: Array = []                 # oids queued for a real texture load
var _tasks: Dictionary = {}              # oid -> LoadTask
var last_stats: Dictionary = {}
var _hud: Label


func configure(options: Dictionary) -> void:
	apply_textures = bool(options.get("apply_textures", true))
	if options.has("bandwidth_mb_s"):
		bandwidth_bytes_per_s = float(options["bandwidth_mb_s"]) * 1e6
	if options.has("latency_seconds"):
		latency_seconds = float(options["latency_seconds"])
	if options.has("budget_fraction"):
		budget_bytes = maxf(base_peak_bytes * float(options["budget_fraction"]), 1.0)
	_transport = TransportModel.new(bandwidth_bytes_per_s, latency_seconds, dt)


func _ready() -> void:
	if apply_textures:
		_build_hud()


## Called by the driver once the environment, camera and runner are wired.
func prepare() -> void:
	if camera == null:
		camera = get_tree().get_first_node_in_group("player_camera") as Camera3D
	_visibility.occlusion_enabled = true
	_visibility.occlusion_max_rays = 4
	_visibility.max_observation_distance = 60.0
	_visibility.include_legacy_fields = false
	if camera != null:
		_visibility.camera = camera
		var player := get_tree().get_first_node_in_group("player")
		if player is CollisionObject3D:
			_visibility.excluded_rids.append((player as CollisionObject3D).get_rid())
	if _transport == null:
		_transport = TransportModel.new(bandwidth_bytes_per_s, latency_seconds, dt)
	_collect_streamable()
	if apply_textures:
		_apply_placeholders()


func _collect_streamable() -> void:
	var by_oid := {}
	for entry in tracked:
		by_oid[entry.get("object_id", "")] = entry
	var index_json: Variant = _read_json_path("dataset/streaming/runs/phase5_engine/manifest.json")
	if index_json is Dictionary:
		for o in index_json.get("objects", []):
			var oid := String(o["object_id"])
			var entry: Dictionary = by_oid.get(oid, {})
			var node: Node3D = entry.get("node")
			if node == null or not is_instance_valid(node):
				continue
			var instances := node.find_children("*", "MeshInstance3D", true, false)
			if instances.is_empty():
				continue
			var mi: MeshInstance3D = instances[0]
			var item := {
				"oid": oid, "node": node, "mi": mi,
				"texture_id": String(o["texture_id"]),
				"source_path": String(o["source_path"]),
				"full_bytes": float(o["full_bytes"]),
				"available_resolutions": o["available_resolutions"],
				"centre": o["centre"], "bounds": entry.get("bounds", AABB()),
			}
			streamable.append(item)
			by_id[oid] = item
			sizes[oid] = float(o["full_bytes"])
			centres[oid] = o["centre"]
		print("[streaming] %d streamable objects (%d tracked)" % [streamable.size(), tracked.size()])


## Duplicate each material (the shared .tres would alias every object) and start on
## the free 256 placeholder.
func _apply_placeholders() -> void:
	for item in streamable:
		var mi: MeshInstance3D = item["mi"]
		var duplicate := (mi.material_override as StandardMaterial3D)
		if duplicate != null:
			duplicate = duplicate.duplicate() as StandardMaterial3D
			mi.material_override = duplicate
		var tex := _load_variant(String(item["source_path"]), 256)
		if duplicate != null and tex != null:
			duplicate.albedo_texture = tex
			duplicate.albedo_color = TINT_PLACEHOLDER


func _load_variant(source_path: String, res: int) -> Texture2D:
	var path := _variant_path(source_path, res)
	if path.is_empty() or not FileAccess.file_exists(path):
		return null
	var img: Image = Image.load_from_file(path)
	if img == null or img.is_empty():
		return null
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


func _variant_path(source_path: String, res: int) -> String:
	# source_path is the object index's texture source: either "res://..." or the
	# repository-relative "godot/assets/..." form.
	var rel := source_path.trim_prefix("res://")
	if rel.begins_with("godot/"):
		rel = rel.trim_prefix("godot/")
	if rel.is_empty() or not rel.begins_with("assets/"):
		return ""
	var dir := rel.get_base_dir()
	var stem := rel.get_file().get_basename()
	return _repo_root().path_join("dataset/textures/variants").path_join(dir.get_file()).path_join(
		"%s_%d.jpg" % [stem, res])


## Absolute filesystem path for a res:// or repository-relative asset path.
func _resolve_asset(source_path: String) -> String:
	if source_path.begins_with("res://"):
		return ProjectSettings.globalize_path(source_path)
	if source_path.begins_with("/"):
		return source_path
	return _repo_root().path_join(source_path)


func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()


func _read_json_path(relative: String) -> Variant:
	var path := relative if relative.begins_with("/") else _repo_root().path_join(relative)
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		return null
	return JSON.parse_string(text)


func set_policy(name: String) -> void:
	policy = name
	print("[streaming] policy = ", policy, " (residency state kept; metrics continue)")


func set_budget_fraction(fraction: float) -> void:
	budget_bytes = maxf(base_peak_bytes * fraction, 1.0)
	print("[streaming] budget = %.0f MB" % (budget_bytes / 1e6))


func set_latency(value: float) -> void:
	latency_seconds = clampf(value, 0.0, 2.0)
	if _transport != null:
		_transport.latency_seconds = latency_seconds
	print("[streaming] latency = %.1f s" % latency_seconds)


func _physics_process(_delta: float) -> void:
	if camera == null or runner == null:
		return
	if not runner.call("is_running"):
		return
	# Only sample once the runner is actually RUN (state 2), i.e. after the settle
	# phase - the recorder's session also begins at RUN, so sample 0 is 6 ticks after
	# RUN starts and scored_replay lookups line up.
	if int(runner.get("state")) != 2:
		return
	_tick += 1
	if _tick % int(round(dt / (1.0 / 60.0))) != 0:
		return
	_step_sample()


func _step_sample() -> void:
	_viewport = get_viewport().get_visible_rect().size
	var measured: Dictionary = _visibility.observe(tracked, _viewport)
	var observations: Array = measured["observations"]

	var row_oid := PackedStringArray()
	var row_cov := PackedFloat64Array()
	var row_vis := PackedByteArray()
	var row_frustum := PackedByteArray()
	var row_occ := PackedInt32Array()

	for obs in observations:
		var oid := String(obs.get("object_id", ""))
		if not by_id.has(oid):
			continue
		var item: Dictionary = by_id[oid]
		var bounds: AABB = item["bounds"]
		var distance := _point_to_aabb_distance(camera.global_position,
			VisibilityService.transform_aabb((item["node"] as Node3D).global_transform, bounds))
		if distance > CANDIDATE_DISTANCE:
			continue
		var frustum := 1 if bool(obs.get("frustum_visible", false)) else 0
		var occlusion = obs.get("occlusion_visible")
		var occ := -1
		if occlusion == true:
			occ = 1
		elif occlusion == false:
			occ = 0
		row_oid.append(oid)
		row_cov.append(float(obs.get("coverage", 0.0)))
		row_frustum.append(frustum)
		row_occ.append(occ)
		row_vis.append(1 if (frustum == 1 and occ != 0) else 0)

	var priorities := _priorities(row_oid, row_cov, row_frustum, row_occ, row_vis)
	var rows := PackedInt32Array()
	for i in row_oid.size():
		rows.append(i)

	_state.step(_sample_index, true, rows, priorities, sizes, row_vis, row_cov, row_oid,
		_transport, budget_bytes, Callable(self, "_on_resident"), Callable(self, "_on_evicted"))
	samples += 1
	_sample_index += 1

	if apply_textures:
		_process_texture_queue()
		_update_hud()

	last_stats = {
		"sample": _sample_index, "policy": policy, "visible_rows": row_oid.size(),
		"resident": _state.resident.size(),
		"resident_bytes": _state.resident_bytes, "budget_bytes": budget_bytes,
		"deficit": _state.metrics.summarise(budget_bytes)["deficit"],
		"pop_in_events": _state.metrics.pop_in_events,
		"bytes_streamed": _state.metrics.bytes_streamed,
		"wasted_prefetches": _state.metrics.wasted_prefetches,
	}
	sample_published.emit(last_stats)


func _priorities(row_oid: PackedStringArray, row_cov: PackedFloat64Array,
		row_frustum: PackedByteArray, row_occ: PackedInt32Array,
		row_vis: PackedByteArray) -> PackedFloat64Array:
	var priorities := PackedFloat64Array()
	priorities.resize(row_oid.size())
	if policy == "no_streaming":
		return priorities
	var position: Array = [camera.global_position.x, camera.global_position.y, camera.global_position.z]
	var basis := camera.global_transform.basis.get_rotation_quaternion()
	var quat: Array = [basis.x, basis.y, basis.z, basis.w]
	var velocity: Array = [0.0, 0.0, 0.0]
	var player := get_tree().get_first_node_in_group("player")
	if player != null and player.get("velocity") != null:
		var v: Vector3 = player.get("velocity")
		velocity = [v.x, v.y, v.z]
	var half := HeuristicOcc.half_angles(camera.fov, _viewport.x, _viewport.y)
	var matrix := StreamMath.quaternion_to_matrix(quat)
	var scored: Array = score_rows.get(str(_sample_index), [])
	var scored_by_oid := {}
	for r in scored:
		scored_by_oid[String(r[0])] = r
	for i in row_oid.size():
		var oid := row_oid[i]
		var visible_now := row_vis[i] == 1
		var prio := 0.0
		match policy:
			"reactive_visible":
				prio = 2.0 if visible_now else 0.0
			"heuristic_occ":
				if visible_now:
					prio = 2.0
				else:
					var margin := HeuristicOcc.margin_deg(position, matrix, centres.get(oid, [0.0, 0.0, 0.0]),
						velocity, _world_angular_velocity(), half[0], half[1])
					prio = HeuristicOcc.priority(false, margin, row_occ[i])
			"mlp_b":
				if visible_now:
					prio = 2.0
				elif scored_by_oid.has(oid):
					var r: Array = scored_by_oid[oid]
					prio = maxf(float(r[5]), float(r[6]))
				else:
					prio = 0.0
		priorities[i] = prio
	return priorities


var _prev_quat := Quaternion.IDENTITY
var _prev_quat_valid := false

func _world_angular_velocity() -> Array:
	var q := camera.global_transform.basis.get_rotation_quaternion()
	if not _prev_quat_valid:
		_prev_quat = q
		_prev_quat_valid = true
		return [0.0, 0.0, 0.0]
	var angular := StreamMath.angular_velocity_from_quaternions(
		[[_prev_quat.x, _prev_quat.y, _prev_quat.z, _prev_quat.w], [q.x, q.y, q.z, q.w]],
		PackedFloat64Array([0.0, dt]))
	_prev_quat = q
	return angular[1]


func _point_to_aabb_distance(point: Vector3, box: AABB) -> float:
	var low := box.position
	var high := box.position + box.size
	var d := Vector3(
		maxf(maxf(low.x - point.x, 0.0), point.x - high.x),
		maxf(maxf(low.y - point.y, 0.0), point.y - high.y),
		maxf(maxf(low.z - point.z, 0.0), point.z - high.z))
	return d.length()


# --- real texture application -------------------------------------------------

func _on_resident(oid: String) -> void:
	if not apply_textures:
		return
	if not _pending.has(oid):
		_pending.append(oid)
	_set_tint(oid, TINT_LOADING)


func _on_evicted(oid: String) -> void:
	if not apply_textures:
		return
	if _pending.has(oid):
		_pending.erase(oid)
	var item: Dictionary = by_id.get(oid, {})
	if item.is_empty():
		return
	# A wasted prefetch was evicted before it was ever seen; tint it, else restore
	# the placeholder.
	var wasted := not bool(_state.ever_visible.get(oid, false))
	_set_tint(oid, TINT_WASTED if wasted else TINT_PLACEHOLDER)


func _process_texture_queue() -> void:
	# One completion per frame bounds the main-thread upload hitch.
	for oid in _tasks.keys():
		var task: LoadTask = _tasks[oid]
		if task.image == null:
			continue
		var item: Dictionary = by_id.get(oid, {})
		_tasks.erase(oid)
		if item.is_empty():
			continue
		var hitch0 := Time.get_ticks_usec()
		var tex := ImageTexture.create_from_image(task.image)
		var mat: StandardMaterial3D = (item["mi"] as MeshInstance3D).material_override
		if mat != null:
			mat.albedo_texture = tex
			mat.albedo_color = TINT_RESIDENT
		_last_hitch_ms = float(Time.get_ticks_usec() - hitch0) / 1000.0
		_last_load_ms = task.ms
		return
	if _pending.is_empty():
		return
	var oid: String = _pending.pop_front()
	var item: Dictionary = by_id.get(oid, {})
	if item.is_empty():
		return
	var task := LoadTask.new()
	task.path = _resolve_asset(String(item["source_path"]))
	_tasks[oid] = task
	WorkerThreadPool.add_task(Callable(task, "run"))


func _set_tint(oid: String, color: Color) -> void:
	if not tint_enabled:
		return
	var item: Dictionary = by_id.get(oid, {})
	if item.is_empty():
		return
	var mat: StandardMaterial3D = (item["mi"] as MeshInstance3D).material_override
	if mat != null:
		mat.albedo_color = color


# --- HUD + input --------------------------------------------------------------

func _build_hud() -> void:
	var layer := CanvasLayer.new()
	layer.name = "StreamingHUD"
	var label := Label.new()
	label.name = "Status"
	label.position = Vector2(12, 12)
	label.add_theme_color_override("font_color", Color(1, 1, 1))
	label.add_theme_color_override("font_outline_color", Color(0, 0, 0))
	label.add_theme_constant_override("outline_size", 4)
	layer.add_child(label)
	add_child(layer)
	_hud = label

func _update_hud() -> void:
	if _hud == null:
		return
	var summary := _state.metrics.summarise(budget_bytes)
	var lines := [
		"Neural Adaptive Texture Streaming - Phase 5 demo (53 objects)",
		"policy: %s   [1]reactive_visible [2]heuristic_occ [3]mlp_b [5]no_streaming [0]preset" % policy,
		"latency: %.1f s  [ / ]      budget: %.0f/%.0f MB  [- / =]" % [
			latency_seconds, _state.resident_bytes / 1e6, budget_bytes / 1e6],
		"deficit: %.4f   pop-in: %d   wasted: %d   streamed: %.0f MB" % [
			summary["deficit"], _state.metrics.pop_in_events, _state.metrics.wasted_prefetches,
			_state.metrics.bytes_streamed / 1e6],
		"resident: %d  loads: %d   last load %.1f ms  hitch %.1f ms   F1 tint: %s" % [
			_state.resident.size(), _state.metrics.loads_completed, _last_load_ms, _last_hitch_ms,
			"on" if tint_enabled else "off"],
	]
	_hud.text = "\n".join(lines)


func _input(event: InputEvent) -> void:
	if not (event is InputEventKey) or not event.pressed or event.echo:
		return
	match (event as InputEventKey).keycode:
		KEY_1:
			set_policy("reactive_visible")
		KEY_2:
			set_policy("heuristic_occ")
		KEY_3:
			set_policy("mlp_b")
		KEY_4:
			print("[streaming] mlp_b live inference is unavailable (GDScript inference ",
				"0.497 ms/candidate; see docs/phase-5-results.md §6) - using scored_replay")
			set_policy("mlp_b")
		KEY_5:
			set_policy("no_streaming")
		KEY_0:
			set_budget_fraction(1.0)
			set_latency(1.0)
			bandwidth_bytes_per_s = 200e6
			if _transport != null:
				_transport.bandwidth_bytes_per_s = bandwidth_bytes_per_s
			print("[streaming] preset cell: budget 100%, 200 MB/s, latency 1.0 s")
		KEY_F1:
			tint_enabled = not tint_enabled
			print("[streaming] tint ", "on" if tint_enabled else "off")
		KEY_BRACKETLEFT:
			set_latency(latency_seconds - 0.1)
		KEY_BRACKETRIGHT:
			set_latency(latency_seconds + 0.1)
		KEY_MINUS:
			set_budget_fraction((budget_bytes / base_peak_bytes) - 0.5)
		KEY_EQUAL:
			set_budget_fraction((budget_bytes / base_peak_bytes) + 0.5)
		KEY_P:
			print("[streaming] snapshot: ", JSON.stringify(last_stats))


## Final metrics for the trajectory, closing open episodes.
func finalise() -> Dictionary:
	return _state.finish(budget_bytes)


## Reset streaming state (for a back-to-back comparison run).
func reset_state() -> void:
	_state = TextureStreamer.StreamState.new()
	_sample_index = 0
	_tick = 0
	_prev_quat_valid = false
	_pending.clear()
	_tasks.clear()
