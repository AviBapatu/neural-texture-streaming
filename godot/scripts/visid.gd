class_name StableId
## Deterministic identifier helpers for the dataset pipeline.
##
## Object ids are derived from the *content* of an object - its category, its source
## model and its quantised transform - combined with the environment seed. They are
## therefore:
##
##   * identical across independent runs of the same seed,
##   * unique per object,
##   * independent of the order in which telemetry observations are recorded,
##   * different between seeds, so datasets generated from different environments can
##     never collide.
##
## They never depend on runtime instance ids or random numbers.
##
## Texture identities are derived from the albedo texture resource paths of the
## materials an object actually uses, so they stay correct when the asset set changes
## and do not need to be maintained by hand.

const CATEGORY_ROAD := "rod"
const CATEGORY_BUILDING := "bld"
const CATEGORY_WALL := "wal"
const CATEGORY_PROP := "prp"
const CATEGORY_SHOWCASE := "shw"
const CATEGORY_PLAZA := "plz"

const TRANSLATION_PRECISION := 1000.0
const ROTATION_PRECISION := 100.0
const SCALE_PRECISION := 1000.0

const HEX_DIGITS := "0123456789abcdef"

## 8 hex characters (32 bits) of Godot's deterministic `String.hash()`.
static func hash8(text: String) -> String:
	var value := text.hash() & 0xFFFFFFFF
	var result := ""
	for i in 8:
		result = HEX_DIGITS[value & 0xF] + result
		value = value >> 4
	return result

## The exact string an object id is derived from. Kept separate so tests and the
## object index can reproduce and audit it.
static func canonical_object_key(
		seed: int, category: String, model_identity: String,
		position: Vector3, rotation_degrees: Vector3, scale: Vector3) -> String:
	return "%d|%s|%s|%s|%s|%s" % [
		seed,
		category,
		model_identity,
		_quantise(position, TRANSLATION_PRECISION),
		_quantise(rotation_degrees, ROTATION_PRECISION),
		_quantise(scale, SCALE_PRECISION),
	]

static func object_id(
		seed: int, category: String, model_identity: String,
		position: Vector3, rotation_degrees: Vector3, scale: Vector3) -> String:
	var key := canonical_object_key(seed, category, model_identity, position, rotation_degrees, scale)
	return "s%d_%s_%s" % [seed, category, hash8(key)]

static func _quantise(value: Vector3, precision: float) -> String:
	return "%d,%d,%d" % [
		roundi(value.x * precision),
		roundi(value.y * precision),
		roundi(value.z * precision),
	]

## Maps a generated object's legacy `source` string onto a category prefix.
static func category_for_source(source: String) -> String:
	if source.begins_with("road-"):
		return CATEGORY_ROAD
	if source.begins_with("building-"):
		return CATEGORY_BUILDING
	if source.begins_with("wall_"):
		return CATEGORY_WALL
	if source.begins_with("gallery_"):
		return CATEGORY_SHOWCASE
	if source.begins_with("plate_"):
		return CATEGORY_PLAZA
	return CATEGORY_PROP

## Maps an albedo texture resource path onto a stable texture identifier.
##
##   res://assets/textures/ground/Bricks066/Bricks066_2K-JPG_Color.jpg
##       -> ambientcg/Bricks066_Color
##   res://assets/models/kenney_city_suburban/Textures/colormap.png
##       -> kenney/city_suburban_colormap
##   res://assets/models/vegetation/Tree_Bark_NormalTree.png
##       -> quaternius/Tree_Bark_NormalTree
##
## Unrecognised paths are returned unchanged so nothing is silently dropped.
static func texture_id_for_path(resource_path: String) -> String:
	if resource_path.is_empty():
		return ""
	if resource_path.begins_with("res://assets/textures/ground/"):
		var relative := resource_path.trim_prefix("res://assets/textures/ground/")
		var parts := relative.split("/", false)
		if parts.is_empty():
			return resource_path
		var file_stem := parts[parts.size() - 1].get_basename()
		var tokens := file_stem.split("_", false)
		if tokens.size() >= 3:
			return "ambientcg/%s_%s" % [tokens[0], tokens[tokens.size() - 1]]
		return "ambientcg/%s" % file_stem
	if resource_path.begins_with("res://assets/models/") and resource_path.contains("/Textures/"):
		var relative := resource_path.trim_prefix("res://assets/models/")
		var parts := relative.split("/", false)
		if parts.is_empty():
			return resource_path
		var pack := parts[0].trim_prefix("kenney_")
		return "kenney/%s_%s" % [pack, resource_path.get_file().get_basename()]
	if resource_path.begins_with("res://assets/models/vegetation/"):
		return "quaternius/%s" % resource_path.get_file().get_basename()
	return resource_path
