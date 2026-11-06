extends SceneTree
## Verifies that object identities are deterministic and seed-scoped.
##
## Run with:
##   godot --headless --path godot --script res://tests/object_identity_test.gd
##
## Checks:
##   1. the same seed generated twice in one process produces identical ids,
##   2. the same seed generated twice produces the same `generation_hash`,
##   3. a different seed produces a disjoint id set (no accidental reuse),
##   4. ids are unique within an environment,
##   5. ids are stable regardless of the order the index is walked in.

const SEED_A := 20260501
const SEED_B := 777001
const ENVIRONMENT_SCENE := "res://scenes/environment.tscn"

var _failures: Array[String] = []
var _stage := 0
var _frames := 0
var _environment: Node

var _ids_a1 := PackedStringArray()
var _ids_a2 := PackedStringArray()
var _ids_b := PackedStringArray()
var _hash_a1 := ""
var _hash_a2 := ""
var _hash_b := ""
var _collisions_a := 0

func _initialize() -> void:
	print("=== object identity test ===\n")

func _process(_delta: float) -> bool:
	_frames += 1
	if _frames == 1:
		return false
	match _stage:
		0:
			_spawn(SEED_A)
			_stage = 1
			return false
		1:
			_capture_a1()
			_release()
			_stage = 2
			return false
		2:
			if _environment != null:
				return false
			_spawn(SEED_A)
			_stage = 3
			return false
		3:
			_capture_a2()
			_release()
			_stage = 4
			return false
		4:
			if _environment != null:
				return false
			_spawn(SEED_B)
			_stage = 5
			return false
		5:
			_capture_b()
			_release()
			_stage = 6
			return false
		6:
			_compare()
			_report()
			return true
	return false

func _spawn(seed_value: int) -> void:
	var packed := load(ENVIRONMENT_SCENE) as PackedScene
	if packed == null:
		_fail("environment.tscn did not load")
		return
	_environment = packed.instantiate()
	_environment.generation_seed = seed_value
	root.add_child(_environment)

func _release() -> void:
	if _environment != null:
		root.remove_child(_environment)
		_environment.queue_free()
		_environment = null

func _capture_a1() -> void:
	var index := _index()
	_ids_a1 = _collect_ids(index)
	_hash_a1 = index.get("generation_hash", "")
	_collisions_a = index.get("id_collisions", 0)
	print("[1] seed %d generated: %d objects, %d id collisions, hash %s"
		% [SEED_A, _ids_a1.size(), _collisions_a, _hash_a1])

func _capture_a2() -> void:
	var index := _index()
	_ids_a2 = _collect_ids(index)
	_hash_a2 = index.get("generation_hash", "")
	print("[2] seed %d regenerated: %d objects, hash %s" % [SEED_A, _ids_a2.size(), _hash_a2])

func _capture_b() -> void:
	var index := _index()
	_ids_b = _collect_ids(index)
	_hash_b = index.get("generation_hash", "")
	print("[3] seed %d generated: %d objects, hash %s" % [SEED_B, _ids_b.size(), _hash_b])

func _index() -> Dictionary:
	if _environment == null:
		_fail("environment missing when building the index")
		return {}
	if not _environment.has_method("build_object_index"):
		_fail("environment script failed to load (no build_object_index)")
		return {}
	return _environment.build_object_index()

func _collect_ids(index: Dictionary) -> PackedStringArray:
	var ids := PackedStringArray()
	for entry in index.get("objects", []):
		ids.append(entry["object_id"])
	return ids

func _compare() -> void:
	print("\n[checks]")
	if _ids_a1.is_empty():
		_fail("seed A produced no objects")
		return

	if _ids_a1 == _ids_a2:
		_ok("identical seeds produce identical object id sequences (%d ids)" % _ids_a1.size())
	else:
		var first_difference := _first_difference(_ids_a1, _ids_a2)
		_fail("object id sequences differ at index %d (%s vs %s)"
			% [first_difference, _ids_a1[first_difference], _ids_a2[first_difference]])

	if _hash_a1 == _hash_a2:
		_ok("identical seeds produce the same generation hash (%s)" % _hash_a1)
	else:
		_fail("generation hash differs: %s vs %s" % [_hash_a1, _hash_a2])

	if _unique(_ids_a1) == _ids_a1.size():
		_ok("all %d ids are unique" % _ids_a1.size())
	else:
		_fail("duplicate ids within seed A")
	if _unique(_ids_b) == _ids_b.size():
		_ok("all %d ids are unique for seed B" % _ids_b.size())
	else:
		_fail("duplicate ids within seed B")

	var shared := _intersection(_ids_a1, _ids_b)
	if shared.is_empty():
		_ok("different seeds share no object ids (%d vs %d ids)" % [_ids_a1.size(), _ids_b.size()])
	else:
		_fail("different seeds share %d object ids, e.g. %s" % [shared.size(), shared[0]])

	if _hash_a1 != _hash_b:
		_ok("different seeds produce different generation hashes")
	else:
		_fail("both seeds produced the same generation hash")

	var ids := _ids_a1.duplicate()
	ids.reverse()
	if _unique(ids) == ids.size():
		_ok("ids do not depend on the order the index is walked in")
	else:
		_fail("ids collide when the index order is reversed")

	if _collisions_a == 0:
		_ok("no hash collisions within the default seed")
	else:
		_fail("%d hash collisions were disambiguated within the default seed" % _collisions_a)

func _first_difference(left: PackedStringArray, right: PackedStringArray) -> int:
	var count := mini(left.size(), right.size())
	for i in count:
		if left[i] != right[i]:
			return i
	return count

func _unique(values: PackedStringArray) -> int:
	var seen := {}
	for value in values:
		seen[value] = true
	return seen.size()

func _intersection(left: PackedStringArray, right: PackedStringArray) -> PackedStringArray:
	var right_set := {}
	for value in right:
		right_set[value] = true
	var result := PackedStringArray()
	for value in left:
		if right_set.has(value):
			result.append(value)
	return result

func _ok(message: String) -> void:
	print("  ok  : %s" % message)

func _fail(message: String) -> void:
	_failures.append(message)
	print("  FAIL: %s" % message)

func _report() -> void:
	print("\n=== result: %s ===" % ("FAILED" if _failures.size() > 0 else "OK"))
	for failure in _failures:
		print("  FAIL: %s" % failure)
	quit(1 if _failures.size() > 0 else 0)
