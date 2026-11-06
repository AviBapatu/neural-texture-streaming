extends SceneTree
## Headless verification of the generated environment.
##
## Run with:
##   godot --headless --path godot --script res://tests/environment_report.gd
##
## It loads every material and model, instantiates `environment.tscn`, then prints
## a top-down ASCII map of the layout together with per-category object counts.
## Exits with a non-zero status if any assertion fails.

const MAP_HALF_EXTENT := 50
const MAP_STEP := 2

const EnvironmentManagerScript := preload("res://scripts/environment_manager.gd")

var _failures: Array[String] = []
var _environment: Node
var _stage := 0

func _initialize() -> void:
	print("=== Neural Predictive Texture Streaming - environment report ===\n")
	_check_materials()
	_check_models()

func _process(_delta: float) -> bool:
	match _stage:
		0:
			var packed := load("res://scenes/environment.tscn") as PackedScene
			if packed == null:
				_fail("environment.tscn did not load")
			else:
				_environment = packed.instantiate()
				root.add_child(_environment)
			_stage = 1
			return false
		1:
			_check_environment()
			print("\n=== result: %s ===" % ("FAILED" if _failures.size() > 0 else "OK"))
			for failure in _failures:
				print("  FAIL: %s" % failure)
			quit(1 if _failures.size() > 0 else 0)
			return true
	return true

func _fail(message: String) -> void:
	_failures.append(message)

func _check_materials() -> void:
	print("[materials]")
	for key in EnvironmentManagerScript.MATERIAL_PATHS:
		var path: String = EnvironmentManagerScript.MATERIAL_PATHS[key]
		var material := load(path) as StandardMaterial3D
		if material == null:
			_fail("material '%s' did not load (%s)" % [key, path])
			continue
		var has_albedo := material.albedo_texture != null
		var has_normal := material.normal_texture != null
		print("  %-9s albedo=%s normal=%s roughness=%s" % [
			key, has_albedo, has_normal, material.roughness_texture != null])
		if not has_albedo:
			_fail("material '%s' has no albedo texture" % key)
		if not has_normal:
			_fail("material '%s' has no normal texture" % key)

func _check_models() -> void:
	print("\n[models]")
	var directories := [
		EnvironmentManagerScript.DIR_SUBURBAN,
		EnvironmentManagerScript.DIR_MODULAR,
		EnvironmentManagerScript.DIR_ROADS,
		EnvironmentManagerScript.DIR_VEGETATION,
	]
	var total := 0
	for directory in directories:
		var files := DirAccess.get_files_at(directory)
		var count := 0
		var failed := 0
		for file_name in files:
			if not file_name.ends_with(".glb"):
				continue
			count += 1
			if load(directory + file_name) as PackedScene == null:
				failed += 1
				_fail("model did not load: %s%s" % [directory, file_name])
		total += count
		print("  %-46s %3d models, %d failed" % [directory.trim_prefix("res://assets/models/"), count, failed])
	print("  total: %d models" % total)
	if total == 0:
		_fail("no models found")

func _check_environment() -> void:
	print("\n[environment]")
	if _environment == null:
		return
	var objects: Array = _environment.get("tracked_objects")
	if objects == null or objects.is_empty():
		_fail("environment generated no tracked objects")
		return

	var counts := {}
	for entry in objects:
		var category := _category(entry["source"])
		counts[category] = counts.get(category, 0) + 1

	print("  tracked objects: %d" % objects.size())
	for category in counts:
		print("    %-12s %4d" % [category, counts[category]])

	for required in ["road", "building", "wall", "prop", "material_showcase"]:
		if counts.get(required, 0) <= 0:
			_fail("environment contains no objects of category '%s'" % required)

	print("\n[top-down map] each cell = %d units: '.' road  '#' building  'w' wall  'G' showcase  '*' prop  '+' plaza" % MAP_STEP)
	_print_map(objects)

func _category(source: String) -> String:
	if source.begins_with("road-"):
		return "road"
	if source.begins_with("building-"):
		return "building"
	if source.begins_with("wall_"):
		return "wall"
	if source.begins_with("gallery_"):
		return "material_showcase"
	if source.begins_with("plate_"):
		return "plate"
	return "prop"

func _symbol(category: String) -> String:
	match category:
		"road": return "."
		"building": return "#"
		"wall": return "w"
		"material_showcase": return "G"
		"plate": return "+"
		_: return "*"

func _print_map(objects: Array) -> void:
	var cells := (MAP_HALF_EXTENT * 2) / MAP_STEP + 1
	var rows: Array = []
	for row in cells:
		var line: Array = []
		for column in cells:
			line.append(" ")
		rows.append(line)
	for entry in objects:
		var node: Node3D = entry["node"]
		var position := node.position
		var column := int(round((position.x + MAP_HALF_EXTENT) / float(MAP_STEP)))
		var row := int(round((position.z + MAP_HALF_EXTENT) / float(MAP_STEP)))
		if row < 0 or row >= cells or column < 0 or column >= cells:
			continue
		rows[row][column] = _symbol(_category(entry["source"]))
	for row in cells:
		print("  " + "".join(rows[row]))
