extends SceneTree
## Writes `dataset/manifests/object_index.json` for the seeded environment.
##
## Run with:
##   godot --headless --path godot --script res://tools/export_object_index.gd
##
## The index maps every generated object to its persistent id, category, transform,
## texture identities and collision flag. It is the join table the labelling
## pipeline uses, and its `generation_hash` identifies the exact environment layout.

const RELATIVE_OUTPUT := "../dataset/manifests/object_index.json"

var _environment: Node
var _stage := 0

func _initialize() -> void:
	print("=== export object index ===")
	var packed := load("res://scenes/environment.tscn") as PackedScene
	if packed == null:
		push_error("environment.tscn did not load")
		quit(1)
		return
	_environment = packed.instantiate()
	root.add_child(_environment)

func _process(_delta: float) -> bool:
	if _stage == 0:
		_stage = 1
		return false
	if _environment == null:
		quit(1)
		return true
	var index: Dictionary = _environment.build_object_index()
	var path := ProjectSettings.globalize_path("res://").path_join(RELATIVE_OUTPUT).simplify_path()
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("could not write %s" % path)
		quit(1)
		return true
	file.store_string(JSON.stringify(index, "  ") + "\n")
	file.close()
	print("  path            : %s" % path)
	print("  environment seed: %d" % index["environment_seed"])
	print("  objects         : %d" % index["object_count"])
	print("  id collisions   : %d" % index["id_collisions"])
	print("  generation hash : %s" % index["generation_hash"])
	print("  texture ids (%d):" % index["texture_ids_used"].size())
	for texture_id in index["texture_ids_used"]:
		print("      %s" % texture_id)
	quit(0)
	return true
