class_name VisibilityService
extends RefCounted
## Computes the three independent visibility signals used by the dataset.
##
##   frustum_visible  - the object's world AABB projects into the viewport
##                      (i.e. it has non-zero screen-space coverage)
##   approx_visible   - the Phase 1 signal: frustum_visible AND within
##                      `max_observation_distance`. Still ignores occlusion, so it
##                      can report objects that are hidden behind buildings.
##   occlusion_visible - raycast result against real physics geometry. `true`,
##                      `false`, or `null` when the test could not run (see
##                      `occlusion_reason`). `null` is never "invisible".
##
## The signals are deliberately kept separate and the occlusion test is optional
## and measured, so the cheaper signals can be used for normal gameplay.

const RESOLUTION_LADDER := [64, 128, 256, 512, 1024, 2048]

const REASON_TESTED := "tested"
const REASON_DISABLED := "disabled"
const REASON_LOW_COVERAGE := "below_coverage_threshold"
const REASON_NO_SPACE := "no_space_state"
const REASON_BEHIND_NEAR := "behind_near_plane"
const REASON_TOO_CLOSE := "too_close"

## Ray budget including the centre ray. A single centre ray is not used for large
## objects: the centre plus the front-facing face centres are sampled instead.
@export var occlusion_max_rays := 4
@export var occlusion_min_coverage := 0.0005
@export var occlusion_enabled := false
## Emit `name` and `source` on every observation. Only the legacy `visible` array
## needs them, so the dataset pipeline turns this off to keep the files lean.
@export var include_legacy_fields := true

var max_observation_distance := 60.0
var camera: Camera3D
var excluded_rids: Array[RID] = []

var rays_cast := 0
var last_time_ms := 0.0
var occlusion_time_ms := 0.0

## Collider RIDs per object, resolved once per object list instead of once per
## sample (walking every subtree every sample dominated the occlusion cost).
var _body_rids: Dictionary = {}
var _body_rids_source: Array = []

## Fraction of the viewport covered by `world_bounds`, clipped to the viewport.
##
## Projects the eight AABB corners and returns the area of their 2D bounding
## rectangle. Corners behind the camera are skipped, so objects straddling the near
## plane are underestimated rather than rejected. This is an approximation: it is a
## bounding rectangle, not the rasterised area of the mesh.
static func screen_coverage(camera: Camera3D, world_bounds: AABB, viewport_size: Vector2) -> float:
	if camera == null or viewport_size.x <= 0.0 or viewport_size.y <= 0.0:
		return 0.0
	var minimum := Vector2(INF, INF)
	var maximum := Vector2(-INF, -INF)
	var behind_count := 0
	for i in 8:
		var corner := world_bounds.get_endpoint(i)
		if camera.is_position_behind(corner):
			behind_count += 1
			continue
		var projected := camera.unproject_position(corner)
		minimum = minimum.min(projected)
		maximum = maximum.max(projected)
	if behind_count == 8:
		return 0.0
	var rect := Rect2(minimum, maximum - minimum)
	var clipped := rect.intersection(Rect2(Vector2.ZERO, viewport_size))
	if clipped.size.x <= 0.0 or clipped.size.y <= 0.0:
		return 0.0
	return (clipped.size.x * clipped.size.y) / (viewport_size.x * viewport_size.y)

## Estimates the texture resolution an object needs so that roughly one texel maps
## to one screen pixel.
##
##   required_px = sqrt(coverage * viewport_w * viewport_h)
##   required_resolution = smallest ladder step >= required_px
##
## This is an ESTIMATE (see docs/future-demand-labels.md). It assumes the coverage
## rectangle approximates the visible surface, a fronto-parallel surface, and a
## texture that maps once across the object's on-screen extent. Returns 0 when the
## object is not on screen.
static func required_resolution(coverage: float, viewport_size: Vector2) -> int:
	if coverage <= 0.0 or viewport_size.x <= 0.0 or viewport_size.y <= 0.0:
		return 0
	var required_px := sqrt(coverage * viewport_size.x * viewport_size.y)
	for step in RESOLUTION_LADDER:
		if float(step) >= required_px:
			return step
	return RESOLUTION_LADDER[RESOLUTION_LADDER.size() - 1]

## Sample points for the occlusion test: the AABB centre plus the centres of the
## front-facing faces, capped at `max_points`.
static func occlusion_sample_points(world_bounds: AABB, camera_position: Vector3, max_points: int) -> Array:
	var centre := world_bounds.get_center()
	var points: Array = [centre]
	if max_points <= 1:
		return points
	var half := world_bounds.size * 0.5
	var faces := [
		{"normal": Vector3(-1.0, 0.0, 0.0), "point": centre + Vector3(-half.x, 0.0, 0.0)},
		{"normal": Vector3(1.0, 0.0, 0.0), "point": centre + Vector3(half.x, 0.0, 0.0)},
		{"normal": Vector3(0.0, -1.0, 0.0), "point": centre + Vector3(0.0, -half.y, 0.0)},
		{"normal": Vector3(0.0, 1.0, 0.0), "point": centre + Vector3(0.0, half.y, 0.0)},
		{"normal": Vector3(0.0, 0.0, -1.0), "point": centre + Vector3(0.0, 0.0, -half.z)},
		{"normal": Vector3(0.0, 0.0, 1.0), "point": centre + Vector3(0.0, 0.0, half.z)},
	]
	for face in faces:
		if points.size() >= max_points:
			break
		var to_camera: Vector3 = camera_position - face["point"]
		if to_camera.length_squared() <= 0.0001:
			continue
		if to_camera.normalized().dot(face["normal"]) <= 0.05:
			continue
		points.append(face["point"])
	return points

## Observes every object within `max_observation_distance`.
##
## Returns `{observations, visible, culled_count, rays_cast, time_ms}` where
## `visible` is the Phase 1 subset (non-zero coverage) and `observations` is the
## full candidate set with explicit flags on every entry.
func observe(objects: Array, viewport_size: Vector2) -> Dictionary:
	var started := Time.get_ticks_usec()
	rays_cast = 0
	occlusion_time_ms = 0.0
	var observations: Array = []
	var visible: Array = []
	var culled := 0
	var reason_counts: Dictionary = {}
	var occlusion_counts := {"true": 0, "false": 0, "null": 0}
	if camera == null:
		last_time_ms = 0.0
		return {"observations": observations, "visible": visible, "culled_count": 0,
			"rays_cast": 0, "time_ms": 0.0, "occlusion_time_ms": 0.0,
			"occlusion_reasons": reason_counts, "occlusion_counts": occlusion_counts}
	var camera_position := camera.global_position
	var space_state := _space_state()
	_ensure_body_rids(objects)

	for entry in objects:
		var node: Node3D = entry["node"]
		if not is_instance_valid(node):
			continue
		var world_bounds := transform_aabb(node.global_transform, entry["bounds"])
		var distance := camera_position.distance_to(world_bounds.get_center())
		if distance > max_observation_distance:
			culled += 1
			continue
		var coverage := screen_coverage(camera, world_bounds, viewport_size)
		var frustum_visible := coverage > 0.0
		var observation := {
			"object_id": entry.get("object_id", ""),
			"legacy_id": entry["id"],
			"category": entry.get("category", ""),
			"distance": snappedf(distance, 0.001),
			"coverage": snappedf(coverage, 0.000001),
			"frustum_visible": frustum_visible,
			"approx_visible": frustum_visible,
			"occlusion_visible": null,
			"occlusion_reason": REASON_DISABLED,
			"occlusion_rays": [0, 0],
			"texture_ids": Array(entry["textures"]),
			"required_resolution": 0,
		}
		if include_legacy_fields:
			observation["name"] = entry.get("name", "")
			observation["source"] = entry.get("source", "")
		if frustum_visible:
			observation["required_resolution"] = required_resolution(coverage, viewport_size)
			var occlusion := _measure_occlusion(space_state, entry, node, world_bounds, camera_position, distance, coverage)
			observation["occlusion_visible"] = occlusion["visible"]
			observation["occlusion_reason"] = occlusion["reason"]
			observation["occlusion_rays"] = [occlusion["tested"], occlusion["open"]]
			reason_counts[occlusion["reason"]] = reason_counts.get(occlusion["reason"], 0) + 1
			if occlusion["visible"] == null:
				occlusion_counts["null"] += 1
			elif occlusion["visible"]:
				occlusion_counts["true"] += 1
			else:
				occlusion_counts["false"] += 1
			visible.append(observation)
		observations.append(observation)

	last_time_ms = (Time.get_ticks_usec() - started) / 1000.0
	return {
		"observations": observations,
		"visible": visible,
		"culled_count": culled,
		"rays_cast": rays_cast,
		"time_ms": snappedf(last_time_ms, 0.001),
		"occlusion_time_ms": snappedf(occlusion_time_ms, 0.001),
		"occlusion_reasons": reason_counts,
		"occlusion_counts": occlusion_counts,
	}

func _space_state() -> PhysicsDirectSpaceState3D:
	if camera == null or camera.get_world_3d() == null:
		return null
	return camera.get_world_3d().direct_space_state

func _measure_occlusion(space_state: PhysicsDirectSpaceState3D, entry: Dictionary, node: Node3D,
		world_bounds: AABB, camera_position: Vector3, distance: float, coverage: float) -> Dictionary:
	if not occlusion_enabled:
		return {"visible": null, "reason": REASON_DISABLED, "tested": 0, "open": 0}
	if space_state == null:
		return {"visible": null, "reason": REASON_NO_SPACE, "tested": 0, "open": 0}
	if coverage < occlusion_min_coverage:
		return {"visible": null, "reason": REASON_LOW_COVERAGE, "tested": 0, "open": 0}
	if distance < 0.5:
		return {"visible": null, "reason": REASON_TOO_CLOSE, "tested": 0, "open": 0}

	var started := Time.get_ticks_usec()
	var exclude := excluded_rids.duplicate()
	var body_rid: RID = _body_rids.get(entry.get("object_id", ""), RID())
	if body_rid.is_valid():
		exclude.append(body_rid)

	var points := occlusion_sample_points(world_bounds, camera_position, maxi(1, occlusion_max_rays))
	var tested := 0
	var open := 0
	for point in points:
		var offset: Vector3 = point - camera_position
		var length := offset.length()
		if length < 0.05:
			continue
		if camera.is_position_behind(point):
			continue
		tested += 1
		var query := PhysicsRayQueryParameters3D.create(
			camera_position, camera_position + offset / length * (length - 0.05))
		query.exclude = exclude
		query.collide_with_areas = false
		query.collide_with_bodies = true
		rays_cast += 1
		if space_state.intersect_ray(query).is_empty():
			open += 1

	occlusion_time_ms += (Time.get_ticks_usec() - started) / 1000.0
	if tested == 0:
		return {"visible": null, "reason": REASON_BEHIND_NEAR, "tested": 0, "open": 0}
	return {"visible": open > 0, "reason": REASON_TESTED, "tested": tested, "open": open}

func _ensure_body_rids(objects: Array) -> void:
	if is_same(_body_rids_source, objects) and _body_rids.size() == objects.size():
		return
	_body_rids_source = objects
	_body_rids.clear()
	for entry in objects:
		var node: Node3D = entry["node"]
		if is_instance_valid(node):
			_body_rids[entry.get("object_id", "")] = _body_rid(node)

func _body_rid(node: Node3D) -> RID:
	for child in node.find_children("*", "CollisionObject3D", true, false):
		var collision_object := child as CollisionObject3D
		if collision_object != null:
			return collision_object.get_rid()
	return RID()

static func transform_aabb(xform: Transform3D, aabb: AABB) -> AABB:
	var minimum := xform * aabb.position
	var maximum := minimum
	for i in 8:
		var point := xform * aabb.get_endpoint(i)
		minimum = minimum.min(point)
		maximum = maximum.max(point)
	return AABB(minimum, maximum - minimum)
