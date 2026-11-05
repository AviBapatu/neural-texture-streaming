extends SceneTree
## Scripted first-person controller test: gravity/ground collision, WASD movement,
## camera reset and collision against a building.
##
## Run with:
##   godot --headless --path godot --script res://tests/movement_test.gd
##
## The test drives the real controller through the real InputMap actions declared
## in project.godot, so it also verifies those bindings. Exits non-zero on failure.

const SETTLE_FRAMES := 40
const MOVE_FRAMES := 90
const COLLIDE_FRAMES := 120
const MIN_TRAVEL := 2.0

var _failures: Array[String] = []
var _stage := 0
var _frames := 0

var _player: CharacterBody3D
var _main: Node
var _start := Vector3.ZERO
var _settled := Vector3.ZERO
var _moved := Vector3.ZERO
var _target_center := Vector3.ZERO
var _target_half := Vector3.ZERO
var _collision_start := Vector3.ZERO

func _initialize() -> void:
	print("=== first-person controller test ===\n")

func _process(_delta: float) -> bool:
	_frames += 1
	match _stage:
		0:
			return _stage_boot()
		1:
			return _stage_settle()
		2:
			return _stage_move()
		3:
			return _stage_reset()
		4:
			return _stage_collision_move()
		5:
			return _stage_report()
	return true

func _fail(message: String) -> void:
	_failures.append(message)
	print("  FAIL: %s" % message)

func _ok(message: String) -> void:
	print("  ok  : %s" % message)

func _advance(next_stage: int) -> bool:
	_stage = next_stage
	_frames = 0
	return false

func _stage_boot() -> bool:
	var packed := load("res://scenes/main.tscn") as PackedScene
	if packed == null:
		_fail("main.tscn did not load")
		_stage = 5
		return false
	_main = packed.instantiate()
	root.add_child(_main)
	_player = get_first_node_in_group("player") as CharacterBody3D
	if _player == null:
		_fail("no node in group 'player'")
		_stage = 5
		return false
	print("[1] controller attached: %s at %s" % [_player.get_class(), _player.global_position])
	_start = _player.global_position
	return _advance(1)

func _stage_settle() -> bool:
	if _frames < SETTLE_FRAMES:
		return false
	print("[2] gravity and ground collision")
	_settled = _player.global_position
	if absf(_settled.y) > 0.25:
		_fail("player did not come to rest on the ground (y=%.3f)" % _settled.y)
	else:
		_ok("player rests on the ground plane (y=%.3f)" % _settled.y)
	Input.action_press("move_forward")
	return _advance(2)

func _stage_move() -> bool:
	if _frames < MOVE_FRAMES:
		return false
	Input.action_release("move_forward")
	_moved = _player.global_position
	var travel := _moved.z - _settled.z
	print("[3] WASD movement via the 'move_forward' action")
	if travel < MIN_TRAVEL:
		_fail("forward movement too small: %.3f m" % travel)
	else:
		_ok("moved %.2f m along the view direction (%s -> %s)"
			% [travel, _settled.snapped(Vector3(0.01, 0.01, 0.01)), _moved.snapped(Vector3(0.01, 0.01, 0.01))])
	var event := InputEventAction.new()
	event.action = "reset_camera"
	event.pressed = true
	Input.parse_input_event(event)
	return _advance(3)

func _stage_reset() -> bool:
	if _frames < 10:
		return false
	var after := _player.global_position
	print("[4] camera reset")
	if absf(after.x - _start.x) > 0.01 or absf(after.z - _start.z) > 0.01:
		_fail("reset did not restore the start position (%s vs %s)" % [after, _start])
	elif absf(after.y - _start.y) > 0.5:
		_fail("reset restored an unexpected height (y=%.3f, expected ~%.3f)" % [after.y, _start.y])
	else:
		_ok("reset restored the spawn position %s" % after.snapped(Vector3(0.01, 0.01, 0.01)))

	var environment := get_first_node_in_group("environment")
	if environment == null:
		_fail("no node in group 'environment'")
		return _advance(5)
	var objects: Array = environment.get("tracked_objects")
	for entry in objects:
		var source: String = entry["source"]
		if not source.begins_with("building-"):
			continue
		var node: Node3D = entry["node"]
		var bounds: AABB = entry["bounds"]
		var world := _transform_aabb(node.global_transform, bounds)
		if world.size.x > 3.0 and world.size.z > 3.0:
			_target_center = world.get_center()
			_target_half = world.size * 0.5
			break
	if _target_half == Vector3.ZERO:
		_fail("no building with a usable footprint was found")
		return _advance(5)
	_player.global_position = Vector3(_target_center.x, 0.5, _target_center.z + _target_half.z + 4.0)
	_player.rotation.y = 0.0
	_player.velocity = Vector3.ZERO
	_collision_start = _player.global_position
	print("[5] collision against %s (centre %s, half size %s)"
		% ["building", _target_center.snapped(Vector3(0.1, 0.1, 0.1)), _target_half.snapped(Vector3(0.1, 0.1, 0.1))])
	Input.action_press("move_forward")
	return _advance(4)

func _stage_collision_move() -> bool:
	if _frames < COLLIDE_FRAMES:
		return false
	Input.action_release("move_forward")
	var final := _player.global_position
	var front_face := _target_center.z + _target_half.z
	var travelled := _collision_start.z - final.z
	if travelled < 1.0:
		_fail("player did not advance towards the building (%.3f m)" % travelled)
	else:
		_ok("advanced %.2f m towards the building" % travelled)
	if final.z < front_face:
		_fail("player penetrated the building (z=%.3f < front face %.3f)" % [final.z, front_face])
	else:
		_ok("stopped outside the building (z=%.3f, front face %.3f)" % [final.z, front_face])
	return _advance(5)

func _stage_report() -> bool:
	print("\n=== result: %s ===" % ("FAILED" if _failures.size() > 0 else "OK"))
	for failure in _failures:
		print("  FAIL: %s" % failure)
	quit(1 if _failures.size() > 0 else 0)
	return true

func _transform_aabb(xform: Transform3D, aabb: AABB) -> AABB:
	var minimum := xform * aabb.position
	var maximum := minimum
	for i in 8:
		var point := xform * aabb.get_endpoint(i)
		minimum = minimum.min(point)
		maximum = maximum.max(point)
	return AABB(minimum, maximum - minimum)
