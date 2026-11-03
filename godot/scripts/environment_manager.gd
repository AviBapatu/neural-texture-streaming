extends Node3D
class_name EnvironmentManager
## Procedurally assembles the compact urban block used by the texture-streaming
## experiments.
##
## The layout is fully deterministic (driven by [member generation_seed]) so the
## environment - and therefore any dataset recorded from it - is reproducible
## across runs and machines. Every spawned object is registered in
## [member tracked_objects] so `telemetry_recorder.gd` can observe it.
##
## Kenney models are authored at ~1 unit per tile, so they are scaled by
## [constant CITY_SCALE]. Quaternius vegetation is already metric and is therefore
## spawned with an inverse scale.

signal environment_ready

const CITY_SCALE := 4.0
const TILE := 4.0
const GRID_LINES := [-20.0, 0.0, 20.0]
const EXTENT := 40.0
const ROAD_CLEARANCE := 3.4
const BUILD_SPACING := 7.5
const MAX_OBJECTS := 600
const SPAWN_POINT := Vector3(0.0, 0.2, -34.0)

const DIR_SUBURBAN := "res://assets/models/kenney_city_suburban/"
const DIR_MODULAR := "res://assets/models/kenney_modular_buildings/"
const DIR_ROADS := "res://assets/models/kenney_city_roads/"
const DIR_VEGETATION := "res://assets/models/vegetation/"

const MATERIAL_PATHS := {
	"grass": "res://assets/materials/mat_ground_grass.tres",
	"asphalt": "res://assets/materials/mat_asphalt.tres",
	"road": "res://assets/materials/mat_road.tres",
	"concrete": "res://assets/materials/mat_concrete.tres",
	"brick": "res://assets/materials/mat_brick.tres",
	"metal": "res://assets/materials/mat_metal.tres",
	"dirt": "res://assets/materials/mat_ground_dirt.tres",
}

const MATERIAL_TEXTURE_IDS := {
	"grass": ["ambientcg/Grass004_Color"],
	"asphalt": ["ambientcg/Asphalt021_Color"],
	"road": ["ambientcg/Road001_Color"],
	"concrete": ["ambientcg/Concrete012_Color"],
	"brick": ["ambientcg/Bricks066_Color"],
	"metal": ["ambientcg/Metal063_Color"],
	"dirt": ["ambientcg/Ground037_Color"],
}

const TEX_SUBURBAN := ["kenney/city_suburban_colormap"]
const TEX_MODULAR := ["kenney/modular_buildings_colormap"]
const TEX_ROADS := ["kenney/city_roads_colormap"]
const TEX_VEGETATION := ["quaternius/stylized_nature_atlas"]

@export var generation_seed := 20260501
@export_range(0.0, 1.0) var building_density := 0.8

var tracked_objects: Array[Dictionary] = []

var _rng := RandomNumberGenerator.new()
var _next_id := 1
var _materials: Dictionary = {}
var _models_root: Node3D
var _blocks: Array[Dictionary] = []
## Guards against two objects hashing to the same content-derived id.
var _used_object_ids: Dictionary = {}
var id_collisions := 0

func _ready() -> void:
	add_to_group("environment")
	_rng.seed = generation_seed
	_load_materials()
	_models_root = Node3D.new()
	_models_root.name = "GeneratedModels"
	add_child(_models_root)
	_build_block_layout()
	_build_roads()
	_build_ground_details()
	_build_buildings()
	_build_walls()
	_build_props()
	_build_texture_gallery()
	print("[EnvironmentManager] %d tracked objects generated (seed=%d)."
		% [tracked_objects.size(), generation_seed])
	environment_ready.emit()

## Transform the player should start (and reset to) at.
func get_spawn_position() -> Vector3:
	return SPAWN_POINT

func get_generation_seed() -> int:
	return generation_seed

# --- asset helpers -----------------------------------------------------------

func _load_materials() -> void:
	for key in MATERIAL_PATHS:
		var material := load(MATERIAL_PATHS[key])
		if material == null:
			push_error("[EnvironmentManager] Missing material resource: %s" % MATERIAL_PATHS[key])
		else:
			_materials[key] = material

func _list_models(directory: String, prefix := "") -> Array:
	var result: Array = []
	for file_name in DirAccess.get_files_at(directory):
		if file_name.ends_with(".glb") and (prefix.is_empty() or file_name.begins_with(prefix)):
			result.append(directory + file_name)
	result.sort()
	return result

func _pick(options: Array) -> String:
	if options.is_empty():
		return ""
	return options[_rng.randi_range(0, options.size() - 1)]

# --- layout helpers ----------------------------------------------------------

func _axis_ranges() -> Array:
	var lines: Array = GRID_LINES.duplicate()
	lines.sort()
	var edges: Array = [-EXTENT]
	edges.append_array(lines)
	edges.append(EXTENT)
	var ranges: Array = []
	for i in range(edges.size() - 1):
		var start: float = edges[i] + ROAD_CLEARANCE
		var end: float = edges[i + 1] - ROAD_CLEARANCE
		if end - start > 3.0:
			ranges.append(Vector2(start, end))
	return ranges

func _build_block_layout() -> void:
	_blocks.clear()
	var gallery_index := -1
	var best_distance := INF
	for x_range in _axis_ranges():
		for z_range in _axis_ranges():
			var center := Vector3((x_range.x + x_range.y) * 0.5, 0.0, (z_range.x + z_range.y) * 0.5)
			var roll := _rng.randf()
			var kind := "built"
			if roll < 0.18:
				kind = "plaza"
			elif roll < 0.34:
				kind = "park"
			_blocks.append({"x": x_range, "z": z_range, "kind": kind, "center": center})
	# The block closest to the spawn becomes the texture gallery.
	for i in _blocks.size():
		var distance: float = _blocks[i]["center"].distance_to(SPAWN_POINT)
		if distance < best_distance:
			best_distance = distance
			gallery_index = i
	if gallery_index >= 0:
		_blocks[gallery_index]["kind"] = "gallery"

# --- construction ------------------------------------------------------------

func _build_roads() -> void:
	var straight := DIR_ROADS + "road-straight.glb"
	var crossroad := DIR_ROADS + "road-crossroad.glb"
	var steps := int(EXTENT / TILE)
	for line in GRID_LINES:
		for i in range(-steps, steps + 1):
			var z := float(i) * TILE
			var at_crossing: bool = GRID_LINES.has(z)
			_spawn_model(crossroad if at_crossing else straight,
				Vector3(line, 0.0, z), 0.0, TEX_ROADS, "none")
	for line in GRID_LINES:
		for i in range(-steps, steps + 1):
			var x := float(i) * TILE
			if GRID_LINES.has(x):
				continue
			_spawn_model(straight, Vector3(x, 0.0, line), 90.0, TEX_ROADS, "none")

func _build_ground_details() -> void:
	var plaza_materials := ["asphalt", "road", "concrete", "dirt"]
	var plaza_index := 0
	for block in _blocks:
		var kind: String = block["kind"]
		if kind != "plaza" and kind != "gallery":
			continue
		var size_x: float = block["x"].y - block["x"].x
		var size_z: float = block["z"].y - block["z"].x
		var material_key: String = "concrete"
		if kind == "plaza":
			material_key = plaza_materials[plaza_index % plaza_materials.size()]
			plaza_index += 1
		_spawn_plate(block["center"], size_x, size_z, material_key)

func _build_buildings() -> void:
	var suburban := _list_models(DIR_SUBURBAN, "building-type-")
	var towers := _list_models(DIR_MODULAR, "building-sample-tower-")
	var houses := _list_models(DIR_MODULAR, "building-sample-house-")
	for block in _blocks:
		var kind: String = block["kind"]
		if kind == "park":
			continue
		var density := building_density
		if kind == "plaza":
			density = 0.35
		if kind == "gallery":
			density = 0.0
		var x_range: Vector2 = block["x"]
		var z_range: Vector2 = block["z"]
		var x := x_range.x + BUILD_SPACING * 0.5
		while x <= x_range.y:
			var z := z_range.x + BUILD_SPACING * 0.5
			while z <= z_range.y:
				_place_building_cell(x, z, suburban, towers, houses, density)
				z += BUILD_SPACING
			x += BUILD_SPACING

func _place_building_cell(x: float, z: float, suburban: Array, towers: Array, houses: Array, density: float) -> void:
	if _rng.randf() > density:
		return
	var yaw: float = [0.0, 90.0, 180.0, 270.0][_rng.randi_range(0, 3)]
	var roll := _rng.randf()
	var model_path := ""
	var textures := TEX_SUBURBAN
	if roll < 0.18 and not towers.is_empty():
		model_path = _pick(towers)
		textures = TEX_MODULAR
	elif roll < 0.34 and not houses.is_empty():
		model_path = _pick(houses)
		textures = TEX_MODULAR
	elif not suburban.is_empty():
		model_path = _pick(suburban)
	_spawn_model(model_path, Vector3(x, 0.0, z), yaw, textures, "auto")

func _build_walls() -> void:
	var half := EXTENT + 1.5
	var step := 6.0
	var gap := TILE * 0.7
	var axes := [
		{"axis": "z", "signs": [1.0, -1.0]},
		{"axis": "x", "signs": [1.0, -1.0]},
	]
	for entry in axes:
		var is_z: bool = entry["axis"] == "z"
		for sign in entry["signs"]:
			var fixed: float = sign * half
			var cursor := -half
			while cursor < half:
				var next: float = minf(cursor + step, half)
				var middle := (cursor + next) * 0.5
				var blocked := false
				for line in GRID_LINES:
					if absf(middle - line) < gap + step * 0.5:
						blocked = true
						break
				if not blocked:
					if is_z:
						_spawn_wall(Vector3(cursor, 0.0, fixed), Vector3(next, 0.0, fixed), "concrete", 2.6, 0.5)
					else:
						_spawn_wall(Vector3(fixed, 0.0, cursor), Vector3(fixed, 0.0, next), "concrete", 2.6, 0.5)
				cursor = next
	# Interior yard walls: a low brick segment on the road-facing edge of each block.
	for block in _blocks:
		if block["kind"] != "built":
			continue
		var x_edge: float = block["x"].x - 0.8
		var z_center: float = (block["z"].x + block["z"].y) * 0.5
		_spawn_wall(Vector3(x_edge, 0.0, z_center - 3.0), Vector3(x_edge, 0.0, z_center + 3.0), "brick", 1.6, 0.35)

func _build_props() -> void:
	var lights := [DIR_ROADS + "light-square.glb", DIR_ROADS + "light-curved.glb"]
	var steps := int(EXTENT / TILE)
	for i in GRID_LINES.size():
		var line: float = GRID_LINES[i]
		var side: float = 3.0 if i % 2 == 0 else -3.0
		var k := -steps
		while k <= steps:
			if absf(float(k) * TILE) < EXTENT - 4.0:
				_spawn_model(_pick(lights), Vector3(line + side, 0.0, float(k) * TILE), 0.0, TEX_ROADS, "trunk")
				_spawn_model(_pick(lights), Vector3(float(k) * TILE, 0.0, line - side), 90.0, TEX_ROADS, "trunk")
			k += 5
	# Vegetation in the park blocks and along the city edge.
	for block in _blocks:
		if block["kind"] != "park":
			continue
		var x_range: Vector2 = block["x"]
		var z_range: Vector2 = block["z"]
		for i in 6:
			var pos := Vector3(
				_rng.randf_range(x_range.x, x_range.y),
				0.0,
				_rng.randf_range(z_range.x, z_range.y)
			)
			_spawn_vegetation(pos)
	# A few planters and barriers for street clutter.
	for i in 18:
		var line: float = GRID_LINES[_rng.randi_range(0, GRID_LINES.size() - 1)]
		var pos := Vector3(line + _rng.randf_range(-3.0, 3.0), 0.0, _rng.randf_range(-EXTENT + 4.0, EXTENT - 4.0))
		_spawn_model(DIR_SUBURBAN + "planter.glb", pos, _rng.randf_range(0.0, 360.0), TEX_SUBURBAN, "none")
	for i in 6:
		var pos := Vector3(_rng.randf_range(-EXTENT, EXTENT), 0.0, -EXTENT + 2.0)
		_spawn_model(DIR_ROADS + "construction-cone.glb", pos, 0.0, TEX_ROADS, "trunk")

func _spawn_vegetation(position: Vector3) -> void:
	var options := [
		DIR_VEGETATION + "Tree.glb",
		DIR_VEGETATION + "Pine.glb",
		DIR_VEGETATION + "Bush.glb",
		DIR_VEGETATION + "Bush with Flowers.glb",
		DIR_VEGETATION + "Fern.glb",
		DIR_VEGETATION + "Grass.glb",
		DIR_VEGETATION + "Tall Grass.glb",
		DIR_VEGETATION + "Rock Medium.glb",
		DIR_VEGETATION + "Pebble Round.glb",
	]
	var collide := "trunk"
	var model_path := _pick(options)
	if model_path.contains("Grass") or model_path.contains("Pebble") or model_path.contains("Fern") or model_path.contains("Bush"):
		collide = "none"
	_spawn_model(model_path, position, _rng.randf_range(0.0, 360.0), TEX_VEGETATION, collide, 1.0 / CITY_SCALE)

func _build_texture_gallery() -> void:
	var block := _find_block("gallery")
	if block.is_empty():
		return
	var keys := ["grass", "asphalt", "road", "concrete", "brick", "metal", "dirt"]
	var center: Vector3 = block["center"]
	var per_row := 4
	var spacing := 3.2
	var size := 2.4
	var rows := int(ceil(keys.size() / float(per_row)))
	for i in keys.size():
		var column := i % per_row
		var row := i / per_row
		var items_in_row := mini(per_row, keys.size() - row * per_row)
		var row_width := float(items_in_row - 1) * spacing
		var x := center.x - row_width * 0.5 + float(column) * spacing
		var z := center.z - float(rows - 1) * spacing * 0.5 + float(row) * spacing
		_spawn_material_cube(Vector3(x, size * 0.5, z), keys[i], size)

# --- primitive spawners ------------------------------------------------------

func _spawn_model(model_path: String, position: Vector3, yaw_degrees: float, fallback_textures: Array, collide_mode := "auto", extra_scale := 1.0) -> Node3D:
	if model_path.is_empty() or tracked_objects.size() >= MAX_OBJECTS:
		return null
	var packed := load(model_path) as PackedScene
	if packed == null:
		push_warning("[EnvironmentManager] Could not load model: %s" % model_path)
		return null
	var wrapper := Node3D.new()
	wrapper.name = "obj_%04d" % _next_id
	wrapper.position = position
	wrapper.rotation.y = deg_to_rad(yaw_degrees)
	_models_root.add_child(wrapper)
	var instance := packed.instantiate() as Node3D
	if instance == null:
		wrapper.queue_free()
		return null
	wrapper.add_child(instance)
	var object_scale := CITY_SCALE * extra_scale
	instance.scale = Vector3.ONE * object_scale
	var bounds := _measure_bounds(wrapper, instance)
	var collidable := _add_collision(wrapper, bounds, collide_mode)
	var textures := _collect_texture_ids(wrapper)
	if textures.is_empty():
		textures = PackedStringArray(fallback_textures)
	var source := model_path.get_file()
	_track(wrapper, source, model_path, StableId.category_for_source(source),
		Vector3(0.0, yaw_degrees, 0.0), Vector3.ONE * object_scale, textures, bounds, collidable)
	return wrapper

func _spawn_plate(center: Vector3, size_x: float, size_z: float, material_key: String) -> void:
	if tracked_objects.size() >= MAX_OBJECTS:
		return
	var thickness := 0.12
	var mesh := BoxMesh.new()
	mesh.size = Vector3(size_x, thickness, size_z)
	var mesh_instance := MeshInstance3D.new()
	mesh_instance.mesh = mesh
	mesh_instance.material_override = _materials.get(material_key)
	var wrapper := Node3D.new()
	wrapper.name = "plate_%04d" % _next_id
	wrapper.position = Vector3(center.x, 0.02 - thickness * 0.5, center.z)
	wrapper.add_child(mesh_instance)
	_models_root.add_child(wrapper)
	var bounds := AABB(Vector3(-size_x * 0.5, -thickness * 0.5, -size_z * 0.5), Vector3(size_x, thickness, size_z))
	var textures := _collect_texture_ids(wrapper)
	if textures.is_empty():
		textures = PackedStringArray(MATERIAL_TEXTURE_IDS.get(material_key, []))
	_track(wrapper, "plate_%s" % material_key, "plate_%s" % material_key, StableId.CATEGORY_PLAZA,
		Vector3.ZERO, Vector3.ONE, textures, bounds, false)

func _spawn_wall(from: Vector3, to: Vector3, material_key: String, height: float, thickness: float) -> void:
	if tracked_objects.size() >= MAX_OBJECTS:
		return
	var length := from.distance_to(to)
	if length < 1.0:
		return
	var size := Vector3(thickness, height, length)
	var mesh := BoxMesh.new()
	mesh.size = size
	var mesh_instance := MeshInstance3D.new()
	mesh_instance.mesh = mesh
	mesh_instance.material_override = _materials.get(material_key)
	var wrapper := Node3D.new()
	wrapper.name = "wall_%04d" % _next_id
	var yaw := atan2(to.x - from.x, to.z - from.z)
	wrapper.position = (from + to) * 0.5 + Vector3(0.0, height * 0.5, 0.0)
	wrapper.rotation.y = yaw
	wrapper.add_child(mesh_instance)
	var body := StaticBody3D.new()
	var shape_node := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = size
	shape_node.shape = box
	body.add_child(shape_node)
	wrapper.add_child(body)
	_models_root.add_child(wrapper)
	var textures := _collect_texture_ids(wrapper)
	if textures.is_empty():
		textures = PackedStringArray(MATERIAL_TEXTURE_IDS.get(material_key, []))
	_track(wrapper, "wall_%s" % material_key, "wall_%s" % material_key, StableId.CATEGORY_WALL,
		Vector3(0.0, rad_to_deg(yaw), 0.0), Vector3.ONE, textures, AABB(-size * 0.5, size), true)

func _spawn_material_cube(position: Vector3, material_key: String, size: float) -> void:
	if tracked_objects.size() >= MAX_OBJECTS:
		return
	var dimensions := Vector3(size, size, size)
	var mesh := BoxMesh.new()
	mesh.size = dimensions
	var mesh_instance := MeshInstance3D.new()
	mesh_instance.mesh = mesh
	mesh_instance.material_override = _materials.get(material_key)
	var wrapper := Node3D.new()
	wrapper.name = "gallery_%s" % material_key
	wrapper.position = position
	wrapper.add_child(mesh_instance)
	var body := StaticBody3D.new()
	var shape_node := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = dimensions
	shape_node.shape = box
	body.add_child(shape_node)
	wrapper.add_child(body)
	_models_root.add_child(wrapper)
	var textures := _collect_texture_ids(wrapper)
	if textures.is_empty():
		textures = PackedStringArray(MATERIAL_TEXTURE_IDS.get(material_key, []))
	_track(wrapper, "gallery_%s" % material_key, "gallery_%s" % material_key, StableId.CATEGORY_SHOWCASE,
		Vector3.ZERO, Vector3.ONE, textures, AABB(-dimensions * 0.5, dimensions), true)

func _track(wrapper: Node3D, source: String, model_identity: String, category: String,
		rotation_degrees: Vector3, scale: Vector3, textures: PackedStringArray,
		bounds: AABB, collidable: bool) -> void:
	var transform := wrapper.global_transform
	var object_id := _unique_object_id(StableId.object_id(
		generation_seed, category, model_identity, transform.origin, rotation_degrees, scale))
	if textures.is_empty():
		textures = PackedStringArray(["untextured/%s" % source])
	tracked_objects.append({
		"id": _next_id,
		"object_id": object_id,
		"category": category,
		"name": wrapper.name,
		"source": source,
		"model": model_identity,
		"rotation_degrees": rotation_degrees,
		"scale": scale,
		"transform": transform,
		"textures": textures,
		"collidable": collidable,
		"node": wrapper,
		"bounds": bounds,
	})
	_next_id += 1

## Disambiguates the (extremely unlikely) case of two objects hashing to the same id.
func _unique_object_id(candidate: String) -> String:
	if not _used_object_ids.has(candidate):
		_used_object_ids[candidate] = 1
		return candidate
	var occurrence: int = _used_object_ids[candidate] + 1
	_used_object_ids[candidate] = occurrence
	id_collisions += 1
	return "%s_%d" % [candidate, occurrence]

# --- texture identity --------------------------------------------------------

## Collects the texture identities of the materials an object actually renders with.
func _collect_texture_ids(subtree: Node) -> PackedStringArray:
	var ids := PackedStringArray()
	for child in subtree.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := child as MeshInstance3D
		if mesh_instance == null:
			continue
		_append_material_textures(mesh_instance.material_override, ids)
		if mesh_instance.mesh != null:
			for surface in mesh_instance.mesh.get_surface_count():
				_append_material_textures(mesh_instance.get_active_material(surface), ids)
	return ids

func _append_material_textures(material: Material, ids: PackedStringArray) -> void:
	if material == null:
		return
	var added := false
	var base := material as BaseMaterial3D
	if base != null and base.albedo_texture != null:
		var texture_id := StableId.texture_id_for_path(base.albedo_texture.resource_path)
		if not texture_id.is_empty():
			if not ids.has(texture_id):
				ids.append(texture_id)
			added = true
	if not added and not material.resource_path.is_empty():
		var fallback := "material/%s" % material.resource_path.get_file().get_basename()
		if not ids.has(fallback):
			ids.append(fallback)

# --- object index ------------------------------------------------------------

## Serialisable description of every generated object, written to
## `dataset/manifests/object_index.json` by `tools/export_object_index.gd`.
func build_object_index() -> Dictionary:
	var objects: Array = []
	var ids := PackedStringArray()
	var texture_ids: Array = []
	for entry in tracked_objects:
		var transform: Transform3D = entry["transform"]
		var world_bounds := _transform_aabb(transform, entry["bounds"])
		var textures: PackedStringArray = entry["textures"]
		for texture_id in textures:
			if not texture_ids.has(texture_id):
				texture_ids.append(texture_id)
		ids.append(entry["object_id"])
		objects.append({
			"object_id": entry["object_id"],
			"legacy_id": entry["id"],
			"category": entry["category"],
			"source": entry["source"],
			"model": entry["model"],
			"node_name": entry["name"],
			"position": _vector_to_array(transform.origin),
			"rotation_degrees": _vector_to_array(entry["rotation_degrees"]),
			"scale": _vector_to_array(entry["scale"]),
			"world_bounds": {
				"position": _vector_to_array(world_bounds.position),
				"size": _vector_to_array(world_bounds.size),
			},
			"collidable": entry["collidable"],
			"texture_ids": Array(textures),
		})
	texture_ids.sort()
	return {
		"schema_version": 1,
		"environment_seed": generation_seed,
		"building_density": building_density,
		"object_count": objects.size(),
		"generation_hash": StableId.hash8("|".join(ids)),
		"id_collisions": id_collisions,
		"texture_identity_method": "derived_from_material_albedo_texture_paths",
		"texture_ids_used": texture_ids,
		"objects": objects,
	}

func _vector_to_array(value: Vector3) -> Array:
	return [
		snappedf(value.x, 0.0001),
		snappedf(value.y, 0.0001),
		snappedf(value.z, 0.0001),
	]

func _find_block(kind: String) -> Dictionary:
	for block in _blocks:
		if block["kind"] == kind:
			return block
	return {}

# --- geometry helpers --------------------------------------------------------

func _measure_bounds(root: Node3D, subtree: Node3D) -> AABB:
	var to_local := root.global_transform.affine_inverse()
	var result := AABB()
	var first := true
	for child in subtree.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := child as MeshInstance3D
		if mesh_instance == null or mesh_instance.mesh == null:
			continue
		var local := _transform_aabb(to_local * mesh_instance.global_transform, mesh_instance.get_aabb())
		if first:
			result = local
			first = false
		else:
			result = result.merge(local)
	if first:
		result = AABB(Vector3(-0.5, 0.0, -0.5), Vector3.ONE)
	return result

func _transform_aabb(xform: Transform3D, aabb: AABB) -> AABB:
	var minimum := xform * aabb.position
	var maximum := minimum
	for i in 8:
		var point := xform * aabb.get_endpoint(i)
		minimum = minimum.min(point)
		maximum = maximum.max(point)
	return AABB(minimum, maximum - minimum)

func _add_collision(wrapper: Node3D, bounds: AABB, mode: String) -> bool:
	var should_collide := false
	match mode:
		"none":
			should_collide = false
		"box", "trunk":
			should_collide = true
		_:
			should_collide = bounds.size.y >= 0.6 and maxf(bounds.size.x, bounds.size.z) >= 0.25
	if not should_collide:
		return false
	var body := StaticBody3D.new()
	var shape_node := CollisionShape3D.new()
	if mode == "trunk":
		var height := maxf(0.5, bounds.size.y)
		var cylinder := CylinderShape3D.new()
		cylinder.height = height
		cylinder.radius = clampf(maxf(bounds.size.x, bounds.size.z) * 0.12, 0.12, 0.6)
		shape_node.shape = cylinder
		shape_node.position = Vector3(bounds.get_center().x, bounds.position.y + height * 0.5, bounds.get_center().z)
	else:
		var box := BoxShape3D.new()
		box.size = bounds.size
		shape_node.shape = box
		shape_node.position = bounds.get_center()
	body.add_child(shape_node)
	wrapper.add_child(body)
	return true
