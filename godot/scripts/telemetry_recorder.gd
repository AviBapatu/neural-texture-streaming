extends Node
## Records camera telemetry for the neural texture-streaming dataset.
##
## Schema v2 is additive: every Phase 1 field is unchanged, and the new fields are
## only additions (see `docs/telemetry-schema.md`). Two object arrays are produced:
##
##   * `observations` - every object within `max_visible_distance`, each carrying the
##     three independent visibility signals (`frustum_visible`, `approx_visible`,
##     `occlusion_visible`), distance, screen-space coverage and texture identity.
##   * `visible` - the Phase 1 subset (non-zero coverage). Emitted unless
##     `legacy_visible_array` is disabled to keep dataset files lean.
##
## Sampling is either wall-clock (`sample_mode = "realtime"`, Phase 1 behaviour) or
## physics-tick based (`"physics"`, exactly `sample_interval_ticks` per sample). The
## tick mode makes sample counts and `session_time` reproducible, which is what the
## dataset pipeline uses.
##
## This node provides infrastructure only - no prediction or scheduling happens here.

const SCHEMA_VERSION := 2

const GROUP_ENVIRONMENT := "environment"
const GROUP_CAMERA := "player_camera"
const GROUP_PLAYER := "player"

const MODE_REALTIME := "realtime"
const MODE_PHYSICS := "physics"

const CSV_HEADER := "timestamp,session_time,frame,cam_x,cam_y,cam_z,quat_x,quat_y,quat_z,quat_w,euler_x,euler_y,euler_z,vel_x,vel_y,vel_z,dir_x,dir_y,dir_z,object_id,object_name,object_source,distance,coverage,textures"

@export var enabled := true
@export var auto_start := true
@export var samples_per_second := 10.0
## Maximum distance for candidate observations and for the legacy `visible` subset.
@export var max_visible_distance := 75.0
## Leave empty to write into `<project>/../dataset/telemetry`.
@export var output_directory := ""
@export var record_jsonl := true
@export var record_csv := true
@export var print_progress := false
## Emit the Phase 1 `visible` array alongside `observations`.
@export var legacy_visible_array := true

@export_group("Phase 2 sampling")
## "realtime" (wall clock, Phase 1) or "physics" (every `sample_interval_ticks`).
@export var sample_mode := MODE_REALTIME
@export var sample_interval_ticks := 6

@export_group("Phase 2 occlusion")
@export var occlusion_enabled := false
@export var occlusion_max_rays := 4

var samples_written := 0
var observations_written := 0
var occlusion_rays_cast := 0
var occlusion_time_ms := 0.0
var visibility_time_ms := 0.0
var _occlusion_reasons: Dictionary = {}
var _occlusion_values: Dictionary = {}

var _accumulator := 0.0
var _frame := 0
var _physics_tick := 0
var _session_stem := ""
var _session_metadata: Dictionary = {}
var _camera: Camera3D
var _player: Node3D
var _environment: Node3D
var _visibility := VisibilityService.new()
var _status_label: Label
var _jsonl: FileAccess
var _csv: FileAccess
var _start_ticks := 0.0
var _previous_quaternion := Quaternion.IDENTITY
var _previous_quaternion_valid := false
var _sample_delta := 0.0

func _ready() -> void:
	_start_ticks = Time.get_ticks_msec() / 1000.0
	_visibility.max_observation_distance = max_visible_distance
	_visibility.occlusion_enabled = occlusion_enabled
	_visibility.occlusion_max_rays = occlusion_max_rays
	call_deferred("_resolve_nodes")

func _resolve_nodes() -> void:
	_camera = get_tree().get_first_node_in_group(GROUP_CAMERA) as Camera3D
	_player = get_tree().get_first_node_in_group(GROUP_PLAYER) as Node3D
	_environment = get_tree().get_first_node_in_group(GROUP_ENVIRONMENT) as Node3D
	if _camera == null or _environment == null:
		push_error("[TelemetryRecorder] Camera or environment not found; recording disabled.")
		enabled = false
		return
	_visibility.camera = _camera
	if _player is CollisionObject3D:
		_visibility.excluded_rids.append((_player as CollisionObject3D).get_rid())
	var hud := get_parent().get_node_or_null("HUD/Status")
	if hud is Label:
		_status_label = hud
	if auto_start:
		begin_session(default_session_stem(), {"auto_started": true})

func default_session_stem() -> String:
	return "session_" + Time.get_datetime_string_from_system(true).replace(":", "-")

# --- session control ---------------------------------------------------------

## Opens a new recording. `stem` is the file name without extension.
func begin_session(stem: String, metadata: Dictionary = {}) -> void:
	close_session()
	var directory := _output_directory()
	var error := DirAccess.make_dir_recursive_absolute(directory)
	if error != OK and error != ERR_ALREADY_EXISTS:
		push_error("[TelemetryRecorder] Could not create %s (error %d)." % [directory, error])
		enabled = false
		return
	_session_stem = stem
	_session_metadata = metadata.duplicate(true)
	samples_written = 0
	observations_written = 0
	occlusion_rays_cast = 0
	occlusion_time_ms = 0.0
	visibility_time_ms = 0.0
	_occlusion_reasons.clear()
	_occlusion_values.clear()
	_physics_tick = 0
	_frame = 0
	_accumulator = 0.0
	_previous_quaternion_valid = false
	if record_jsonl:
		_jsonl = FileAccess.open(directory.path_join(stem + ".jsonl"), FileAccess.WRITE)
	if record_csv:
		_csv = FileAccess.open(directory.path_join(stem + ".csv"), FileAccess.WRITE)
		if _csv != null:
			_csv.store_line(CSV_HEADER)
	print("[TelemetryRecorder] recording '%s' into %s" % [stem, directory])

## Closes the current recording and writes its `<stem>.meta.json` sidecar.
func close_session() -> void:
	if _jsonl != null:
		_jsonl.close()
		_jsonl = null
	if _csv != null:
		_csv.close()
		_csv = null
	if not _session_stem.is_empty():
		_write_meta()
		_session_stem = ""

func _output_directory() -> String:
	if not output_directory.is_empty():
		return output_directory
	return ProjectSettings.globalize_path("res://").path_join("../dataset/telemetry").simplify_path()

func _write_meta() -> void:
	var directory := _output_directory()
	var meta := _session_metadata.duplicate(true)
	meta["schema_version"] = SCHEMA_VERSION
	meta["session_stem"] = _session_stem
	meta["samples"] = samples_written
	meta["observations"] = observations_written
	meta["sample_mode"] = sample_mode
	meta["sample_interval_ticks"] = sample_interval_ticks
	meta["physics_ticks_per_second"] = Engine.physics_ticks_per_second
	meta["samples_per_second_effective"] = _effective_sample_rate()
	meta["max_visible_distance"] = max_visible_distance
	meta["legacy_visible_array"] = legacy_visible_array
	meta["environment_seed"] = _environment.get("generation_seed") if _environment != null else null
	meta["occlusion"] = {
		"enabled": occlusion_enabled,
		"max_rays": occlusion_max_rays,
		"min_coverage": _visibility.occlusion_min_coverage,
		"rays_cast": occlusion_rays_cast,
		"occlusion_time_ms": snappedf(occlusion_time_ms, 0.001),
		"visibility_time_ms": snappedf(visibility_time_ms, 0.001),
		"mean_occlusion_ms_per_sample": snappedf(_mean(occlusion_time_ms), 0.0001),
		"mean_visibility_ms_per_sample": snappedf(_mean(visibility_time_ms), 0.0001),
		"occlusion_share_percent": snappedf(_share_percent(), 0.01),
		"reasons": _occlusion_reasons.duplicate(),
		"values": _occlusion_values.duplicate(),
	}
	var path := directory.path_join(_session_stem + ".meta.json")
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("[TelemetryRecorder] Could not write %s" % path)
		return
	file.store_string(JSON.stringify(meta, "  ") + "\n")
	file.close()

func _mean(total: float) -> float:
	return total / float(samples_written) if samples_written > 0 else 0.0

func _share_percent() -> float:
	return (occlusion_time_ms / visibility_time_ms) * 100.0 if visibility_time_ms > 0.0 else 0.0

func _effective_sample_rate() -> float:
	if sample_mode == MODE_PHYSICS:
		return float(Engine.physics_ticks_per_second) / float(maxi(1, sample_interval_ticks))
	return samples_per_second

# --- sampling ----------------------------------------------------------------

func _process(delta: float) -> void:
	_frame += 1
	if sample_mode != MODE_REALTIME or not enabled or _camera == null:
		return
	_accumulator += delta
	var interval := 1.0 / maxf(1.0, samples_per_second)
	if _accumulator < interval:
		return
	_sample_delta = _accumulator
	_accumulator = 0.0
	_write_sample()

func _physics_process(delta: float) -> void:
	_physics_tick += 1
	if sample_mode != MODE_PHYSICS or not enabled or _camera == null:
		return
	if _physics_tick % maxi(1, sample_interval_ticks) != 0:
		return
	_sample_delta = float(sample_interval_ticks) / float(Engine.physics_ticks_per_second)
	_write_sample()

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("toggle_telemetry"):
		enabled = not enabled
		print("[TelemetryRecorder] Recording %s." % ("enabled" if enabled else "paused"))

## Enables or disables recording (used by tests and the HUD).
func set_recording(value: bool) -> void:
	enabled = value

func is_recording() -> bool:
	return enabled

func get_status_text() -> String:
	return "%s | %d samples | %s" % [
		"REC" if enabled else "PAUSED",
		samples_written,
		_session_stem,
	]

func _write_sample() -> void:
	var sample := _build_sample()
	if record_jsonl and _jsonl != null:
		_jsonl.store_line(JSON.stringify(sample))
	if record_csv and _csv != null:
		for entry in sample.get("visible", []):
			_csv.store_line(_csv_row(sample, entry))
	samples_written += 1
	observations_written += sample["observations"].size()
	if _status_label != null and samples_written % 5 == 0:
		_status_label.text = "Telemetry: " + get_status_text()
	if print_progress and samples_written % 50 == 0:
		print("[TelemetryRecorder] %d samples, %d observations."
			% [samples_written, sample["observations"].size()])

func _build_sample() -> Dictionary:
	var camera_transform := _camera.global_transform
	var camera_position := camera_transform.origin
	var quaternion := camera_transform.basis.get_rotation_quaternion()
	var euler := camera_transform.basis.get_euler()
	var view_direction := -camera_transform.basis.z
	var velocity := Vector3.ZERO
	if _player != null:
		velocity = _player.get("velocity")
	var angular_velocity := _angular_velocity(quaternion)
	var viewport_size := get_viewport().get_visible_rect().size

	_visibility.max_observation_distance = max_visible_distance
	_visibility.occlusion_enabled = occlusion_enabled
	_visibility.occlusion_max_rays = occlusion_max_rays
	_visibility.include_legacy_fields = legacy_visible_array
	var objects: Array = _environment.get("tracked_objects")
	var measured: Dictionary = _visibility.observe(objects, viewport_size)
	visibility_time_ms += measured["time_ms"]
	occlusion_rays_cast += measured["rays_cast"]
	occlusion_time_ms += measured["occlusion_time_ms"]
	for reason in measured["occlusion_reasons"]:
		_occlusion_reasons[reason] = _occlusion_reasons.get(reason, 0) + measured["occlusion_reasons"][reason]
	for value in measured["occlusion_counts"]:
		_occlusion_values[value] = _occlusion_values.get(value, 0) + measured["occlusion_counts"][value]

	var environment_seed: int = 0
	if _environment != null and _environment.get("generation_seed") != null:
		environment_seed = _environment.get("generation_seed")

	var sample := {
		"schema_version": SCHEMA_VERSION,
		"timestamp": snappedf(Time.get_unix_time_from_system(), 0.001),
		"session_time": snappedf(_session_time(), 0.001),
		"frame": _frame_index(),
		"physics_tick": _physics_tick,
		"environment_seed": environment_seed,
		"cam_position": _vector(camera_position, 0.001),
		"cam_quaternion": [snappedf(quaternion.x, 0.0001), snappedf(quaternion.y, 0.0001),
			snappedf(quaternion.z, 0.0001), snappedf(quaternion.w, 0.0001)],
		"cam_euler": _vector(euler, 0.0001),
		"cam_velocity": _vector(velocity, 0.001),
		"cam_angular_velocity": _vector(angular_velocity, 0.0001),
		"cam_direction": _vector(view_direction, 0.0001),
		"cam_fov": snappedf(_camera.fov, 0.01),
		"viewport": [int(viewport_size.x), int(viewport_size.y)],
		"observation_count": measured["observations"].size(),
		"visible_count": measured["visible"].size(),
		"culled_count": measured["culled_count"],
		"occlusion_enabled": occlusion_enabled,
		"observations": measured["observations"],
	}
	if legacy_visible_array:
		sample["visible"] = _legacy_visible(measured["visible"])
	return sample

func _session_time() -> float:
	if sample_mode == MODE_PHYSICS:
		return float(_physics_tick) / float(Engine.physics_ticks_per_second)
	return Time.get_ticks_msec() / 1000.0 - _start_ticks

## In physics mode the record is indexed by tick rather than by render frame, so the
## value does not depend on how the engine schedules `_process` against physics.
func _frame_index() -> int:
	return _physics_tick if sample_mode == MODE_PHYSICS else _frame

## Angular velocity as an axis-angle finite difference of consecutive camera
## orientations, in radians per second. Zero for the first sample.
func _angular_velocity(quaternion: Quaternion) -> Vector3:
	if not _previous_quaternion_valid or _sample_delta <= 0.0:
		_previous_quaternion = quaternion
		_previous_quaternion_valid = true
		return Vector3.ZERO
	var delta := _previous_quaternion.inverse() * quaternion
	_previous_quaternion = quaternion
	var angle := delta.get_angle()
	if is_zero_approx(angle):
		return Vector3.ZERO
	var axis := delta.get_axis()
	if not axis.is_finite():
		return Vector3.ZERO
	return axis * (angle / _sample_delta)

func _legacy_visible(observations: Array) -> Array:
	var result: Array = []
	for observation in observations:
		result.append({
			"id": observation["legacy_id"],
			"name": observation["name"],
			"source": observation["source"],
			"distance": observation["distance"],
			"coverage": observation["coverage"],
			"textures": Array(observation["texture_ids"]),
		})
	return result

func _vector(value: Vector3, precision: float) -> Array:
	return [snappedf(value.x, precision), snappedf(value.y, precision), snappedf(value.z, precision)]

func _csv_row(sample: Dictionary, entry: Dictionary) -> String:
	var position: Array = sample["cam_position"]
	var quaternion: Array = sample["cam_quaternion"]
	var euler: Array = sample["cam_euler"]
	var velocity: Array = sample["cam_velocity"]
	var direction: Array = sample["cam_direction"]
	return ",".join([
		"%.3f" % sample["timestamp"],
		"%.3f" % sample["session_time"],
		"%d" % sample["frame"],
		"%.3f" % position[0], "%.3f" % position[1], "%.3f" % position[2],
		"%.4f" % quaternion[0], "%.4f" % quaternion[1], "%.4f" % quaternion[2], "%.4f" % quaternion[3],
		"%.4f" % euler[0], "%.4f" % euler[1], "%.4f" % euler[2],
		"%.3f" % velocity[0], "%.3f" % velocity[1], "%.3f" % velocity[2],
		"%.4f" % direction[0], "%.4f" % direction[1], "%.4f" % direction[2],
		"%d" % entry["id"],
		entry["name"],
		entry["source"],
		"%.3f" % entry["distance"],
		"%.6f" % entry["coverage"],
		"|".join(entry["textures"]),
	])

func _exit_tree() -> void:
	close_session()
