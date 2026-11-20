extends SceneTree
## Item 5 probe: is `Parameter "scenario" is null.` caused by creating a
## SubViewport(own_world_3d) + WorldEnvironment and freeing it in the same frame?
##
##   godot --headless --path godot --script res://scripts/demo/env_noise_probe.gd

func _initialize() -> void:
	_run.call_deferred()

func _run() -> void:
	print("[probe] case A: SubViewport(own_world_3d) + WorldEnvironment, world assigned up front")
	var a := await _make(true, true)
	print("[probe] case A done")

	print("[probe] case B: SubViewport(own_world_3d) + WorldEnvironment, world assigned after tree")
	var b := await _make(true, false)
	print("[probe] case B done")

	print("[probe] case C: bare SubViewport(own_world_3d)")
	var c := await _make(true, false, false)
	print("[probe] case C done")

	print("[probe] case D: plain SubViewport, no own world")
	var d := await _make(false, false, false)
	print("[probe] case D done")

	quit(0)

func _make(own_world: bool, world_first: bool, with_env := true) -> bool:
	var sv := SubViewport.new()
	sv.own_world_3d = own_world
	root.add_child(sv)
	if own_world and world_first and sv.world_3d == null:
		sv.world_3d = World3D.new()
	if own_world and not world_first and sv.world_3d == null:
		sv.world_3d = World3D.new()
	if with_env:
		var we := WorldEnvironment.new()
		we.environment = Environment.new()
		sv.add_child(we)
	await process_frame
	await process_frame
	sv.queue_free()
	await process_frame
	return true
