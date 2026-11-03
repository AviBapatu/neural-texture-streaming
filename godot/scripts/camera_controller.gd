extends CharacterBody3D
## First-person camera controller used to walk through the generated city block.
##
## Movement uses the InputMap actions declared in `project.godot`, so the bindings
## stay editable from the editor. Pressing the `reset_camera` action restores the
## transform captured on startup. The node also publishes its velocity and view
## direction for the telemetry recorder in `telemetry_recorder.gd`.

@export_group("Movement")
@export var walk_speed := 6.0
@export var sprint_multiplier := 2.0
@export var ground_acceleration := 12.0
@export var air_acceleration := 3.0
@export var jump_velocity := 5.5
@export var gravity := 22.0

@export_group("Camera")
@export var mouse_sensitivity := 0.0022
@export_range(0.0, 89.0) var max_pitch_degrees := 88.0

@onready var camera: Camera3D = $Camera3D

var _pitch := 0.0
var _start_transform := Transform3D()
var _input_enabled := true

func _ready() -> void:
	_start_transform = global_transform
	add_to_group("player")
	camera.add_to_group("player_camera")
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			var motion := event as InputEventMouseMotion
			rotate_y(-motion.relative.x * mouse_sensitivity)
			_pitch = clampf(
				_pitch - motion.relative.y * mouse_sensitivity,
				-deg_to_rad(max_pitch_degrees),
				deg_to_rad(max_pitch_degrees)
			)
			camera.rotation.x = _pitch
		return

	if event is InputEventMouseButton and (event as InputEventMouseButton).pressed:
		if Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
		return

	if event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif event.is_action_pressed("reset_camera"):
		reset_to_start()

func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity.y -= gravity * delta

	var input := Vector2.ZERO
	if _input_enabled:
		input = Input.get_vector("move_left", "move_right", "move_forward", "move_back")

	var direction := transform.basis * Vector3(input.x, 0.0, input.y)
	if direction.length_squared() > 0.0:
		direction = direction.normalized()

	var speed := walk_speed
	if Input.is_action_pressed("sprint"):
		speed *= sprint_multiplier

	var target := direction * speed
	var acceleration := ground_acceleration if is_on_floor() else air_acceleration
	var horizontal := Vector3(velocity.x, 0.0, velocity.z)
	horizontal = horizontal.lerp(target, clampf(acceleration * delta, 0.0, 1.0))
	velocity.x = horizontal.x
	velocity.z = horizontal.z

	if _input_enabled and Input.is_action_just_pressed("jump") and is_on_floor():
		velocity.y = jump_velocity

	move_and_slide()

## Restores the player to the transform it had when the scene was loaded.
func reset_to_start() -> void:
	global_transform = _start_transform
	_pitch = 0.0
	camera.rotation.x = 0.0
	velocity = Vector3.ZERO

## Forward view vector of the camera, used as the telemetry "camera direction".
func get_view_direction() -> Vector3:
	return -camera.global_transform.basis.z

func set_input_enabled(value: bool) -> void:
	_input_enabled = value
