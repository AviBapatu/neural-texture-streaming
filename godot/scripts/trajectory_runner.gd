extends Node
class_name TrajectoryRunner
## Replays a generated camera trajectory without touching the player controller.
##
## The runner only uses the controller's public API and the real InputMap actions:
## it sets `walk_speed`, writes `rotation.y` / the camera pitch, and presses
## `move_forward` / `move_back` / `sprint`. The player therefore moves through
## exactly the same code path a human would drive it through.
##
## It runs with a lower physics process priority than the player so the heading is
## applied before the controller integrates movement for that tick.

const STATE_IDLE := 0
const STATE_SETTLE := 1
const STATE_RUN := 2
const STATE_DONE := 3

const MOVEMENT_ACTIONS := ["move_forward", "move_back", "move_left", "move_right", "sprint"]

## Physics ticks to let gravity settle the player after teleporting to the start.
@export var settle_ticks := 30

var recorder: Node
var player: CharacterBody3D
var camera: Camera3D

var definition: Dictionary = {}
var state := STATE_IDLE
var elapsed := 0.0

var _settle_count := 0
var _queue: Array = []
var _queue_index := -1
var results: Array = []

func _ready() -> void:
	process_physics_priority = -100
	_resolve()

func _resolve() -> void:
	if player == null:
		player = get_tree().get_first_node_in_group("player") as CharacterBody3D
	if camera == null:
		camera = get_tree().get_first_node_in_group("player_camera") as Camera3D

## Queues a list of trajectory definitions and starts the first one.
##
## The runner advances the queue itself, inside `_physics_process`, so the gap
## between one trajectory ending and the next beginning is always exactly one
## physics tick. Driving the sequence from outside (polling a frame callback) would
## make that gap depend on process/physics scheduling and break reproducibility.
func queue(definitions: Array) -> void:
	_queue = definitions
	_queue_index = -1
	state = STATE_IDLE
	_advance_queue()

## Loads a single trajectory definition and begins the settle phase.
func start(new_definition: Dictionary) -> void:
	_resolve()
	definition = new_definition
	state = STATE_SETTLE
	elapsed = 0.0
	_settle_count = 0
	_release_movement()
	_teleport()

func _advance_queue() -> void:
	if _queue_index + 1 >= _queue.size():
		state = STATE_DONE
		return
	_queue_index += 1
	start(_queue[_queue_index])

func is_running() -> bool:
	return state == STATE_SETTLE or state == STATE_RUN

func is_finished() -> bool:
	return state == STATE_DONE and _queue_index >= _queue.size() - 1

func queued_count() -> int:
	return _queue.size()

func completed_count() -> int:
	return clampi(_queue_index + 1, 0, _queue.size())

func trajectory_id() -> String:
	return String(definition.get("trajectory_id", ""))

func _teleport() -> void:
	if player == null:
		return
	var position: Array = definition.get("start_position", [0.0, 0.2, -34.0])
	player.global_position = Vector3(position[0], position[1], position[2])
	player.rotation.y = deg_to_rad(float(definition.get("start_yaw_degrees", 0.0)))
	player.velocity = Vector3.ZERO
	if camera != null:
		camera.rotation.x = deg_to_rad(float(definition.get("start_pitch_degrees", 0.0)))

func _physics_process(delta: float) -> void:
	match state:
		STATE_SETTLE:
			_settle_count += 1
			if _settle_count >= settle_ticks:
				_begin()
		STATE_RUN:
			elapsed += delta
			if elapsed >= float(definition.get("duration_seconds", 0.0)):
				_finish()
			else:
				_apply(elapsed, delta)
		STATE_DONE:
			_advance_queue()
		_:
			pass

func _begin() -> void:
	state = STATE_RUN
	elapsed = 0.0
	if recorder == null:
		return
	recorder.call("begin_session", trajectory_id(), {
		"trajectory_id": trajectory_id(),
		"trajectory_kind": definition.get("kind", ""),
		"trajectory_hash": StableId.hash8(JSON.stringify(definition, "", true)),
		"environment_seed": definition.get("environment_seed", 0),
		"duration_seconds": definition.get("duration_seconds", 0.0),
		"segment_count": definition.get("segments", []).size(),
		"settle_ticks": settle_ticks,
		"definition": definition,
	})

func _finish() -> void:
	_release_movement()
	if player != null:
		player.walk_speed = 6.0
	state = STATE_DONE
	if recorder != null:
		results.append({
			"trajectory_id": trajectory_id(),
			"samples": recorder.get("samples_written"),
			"observations": recorder.get("observations_written"),
		})
		recorder.call("close_session")

## Per-trajectory sample/observation counts, in execution order.
func take_results() -> Array:
	var collected := results
	results = []
	return collected

func _apply(time: float, delta: float) -> void:
	if player == null:
		return
	var segment := _segment_at(time)
	if segment.is_empty():
		return
	var should_move := _apply_heading(segment, time, delta)
	_apply_movement(segment, should_move)

func _segment_at(time: float) -> Dictionary:
	var segments: Array = definition.get("segments", [])
	if segments.is_empty():
		return {}
	for segment in segments:
		if time < float(segment.get("end", 0.0)):
			return segment
	return segments[segments.size() - 1]

func _apply_heading(segment: Dictionary, time: float, delta: float) -> bool:
	if String(segment.get("mode", "control")) == "waypoint":
		return _apply_waypoint_heading(segment, delta)
	var start := float(segment.get("start", 0.0))
	var end := float(segment.get("end", 0.0))
	var local := clampf((time - start) / maxf(0.001, end - start), 0.0, 1.0)
	player.rotation.y = deg_to_rad(lerpf(
		float(segment.get("yaw_from", 0.0)), float(segment.get("yaw_to", 0.0)), local))
	if camera != null:
		camera.rotation.x = deg_to_rad(lerpf(
			float(segment.get("pitch_from", 0.0)), float(segment.get("pitch_to", 0.0)), local))
	return String(segment.get("action", "none")) != "none"

func _apply_waypoint_heading(segment: Dictionary, delta: float) -> bool:
	var target_array: Array = segment.get("target", [0.0, 0.0, 0.0])
	var target := Vector3(target_array[0], target_array[1], target_array[2])
	var offset := target - player.global_position
	offset.y = 0.0
	if camera != null:
		camera.rotation.x = deg_to_rad(float(segment.get("pitch", 0.0)))
	if offset.length() <= float(segment.get("arrive_radius", 1.5)):
		return false
	var desired := atan2(-offset.x, -offset.z)
	var max_step := deg_to_rad(float(segment.get("turn_rate_degrees_per_second", 180.0))) * delta
	player.rotation.y = _approach_angle(player.rotation.y, desired, max_step)
	return true

func _approach_angle(from: float, to: float, max_step: float) -> float:
	return from + clampf(wrapf(to - from, -PI, PI), -max_step, max_step)

func _apply_movement(segment: Dictionary, should_move: bool) -> void:
	_release_movement()
	if player == null:
		return
	player.walk_speed = float(segment.get("speed", 6.0))
	if not should_move:
		return
	var action := String(segment.get("action", "move_forward"))
	if action != "none" and InputMap.has_action(action):
		Input.action_press(action)
	if bool(segment.get("sprint", false)) and InputMap.has_action("sprint"):
		Input.action_press("sprint")

func _release_movement() -> void:
	for action in MOVEMENT_ACTIONS:
		if InputMap.has_action(action):
			Input.action_release(action)
