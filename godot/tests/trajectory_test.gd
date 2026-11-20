extends SceneTree
## Verifies that replaying a trajectory reproduces the same state stream.
##
## Run with:
##   godot --headless --fixed-fps 60 --path godot --script res://tests/trajectory_test.gd
##
## Every trajectory in the quick profile is replayed twice and compared sample by
## sample, after removing the wall-clock `timestamp`.
##
## Discrete fields - sample count, `session_time`, `frame`, `physics_tick`, camera
## position/orientation/velocity, and every object id, category, visibility flag,
## texture id and observation count - must match exactly.
##
## `coverage` and `distance` are allowed a small tolerance. Replaying inside one
## process means the second replay starts from the physics state left by the first,
## so contact/depenetration history can perturb the camera transform by ~1e-9. That
## shows up as a one-unit difference in the sixth decimal of `coverage`, and nothing
## else. Two replays in *separate* processes are byte-identical, which is what the
## dataset pipeline relies on (asserted by `tests/run_tests.sh`).

const QUICK_DIR := "../dataset/trajectories/quick"
const RUN_DIRECTORIES := ["user://determinism_a", "user://determinism_b"]
const COVERAGE_TOLERANCE := 0.000005
const DISTANCE_TOLERANCE := 0.005

const EXACT_OBSERVATION_KEYS := [
	"legacy_id", "category", "frustum_visible", "approx_visible",
	"occlusion_visible", "occlusion_reason", "occlusion_rays", "texture_ids",
]
const NUMERIC_OBSERVATION_KEYS := {"coverage": COVERAGE_TOLERANCE, "distance": DISTANCE_TOLERANCE}
const RESOLUTION_LADDER := [64, 128, 256, 512, 1024, 2048]

var _failures: Array[String] = []
var _main: Node
var _recorder: Node
var _runner: TrajectoryRunner
var _definitions: Array = []
var _stage := 0
var _frames := 0
var _run_index := 0

func _initialize() -> void:
	print("=== trajectory determinism test ===\n")
	_load_definitions()
	var packed := load("res://scenes/main.tscn") as PackedScene
	if packed == null:
		push_error("main.tscn did not load")
		quit(1)
		return
	_main = packed.instantiate()
	_recorder = _main.get_node_or_null("TelemetryRecorder")
	if _recorder == null:
		push_error("TelemetryRecorder missing")
		quit(1)
		return
	_recorder.set("auto_start", false)
	_recorder.set("sample_mode", "physics")
	_recorder.set("sample_interval_ticks", 6)
	_recorder.set("record_csv", false)
	_recorder.set("legacy_visible_array", false)
	_recorder.set("occlusion_enabled", false)
	root.add_child(_main)
	_runner = TrajectoryRunner.new()
	_runner.name = "TrajectoryRunner"
	_runner.recorder = _recorder
	_main.add_child(_runner)

func _process(_delta: float) -> bool:
	_frames += 1
	if _frames < 10:
		return false
	match _stage:
		0:
			if _definitions.is_empty():
				_fail("no trajectory definitions loaded from %s" % QUICK_DIR)
				_report()
				return true
			print("replaying %d trajectories into %d directories\n"
				% [_definitions.size(), RUN_DIRECTORIES.size()])
			_next_directory()
			_stage = 1
			return false
		1:
			if not _runner.is_finished():
				return false
			_run_index += 1
			if _run_index >= RUN_DIRECTORIES.size():
				_compare()
				_report()
				return true
			_next_directory()
			return false
	return false

func _next_directory() -> void:
	_recorder.set("output_directory", ProjectSettings.globalize_path(RUN_DIRECTORIES[_run_index]))
	_runner.queue(_definitions)

func _load_definitions() -> void:
	var directory := ProjectSettings.globalize_path("res://").path_join(QUICK_DIR).simplify_path()
	var index_file := FileAccess.open(directory.path_join("index.json"), FileAccess.READ)
	if index_file == null:
		return
	var index: Dictionary = JSON.parse_string(index_file.get_as_text())
	index_file.close()
	for entry in index.get("trajectories", []):
		var path := directory.path_join(String(entry["file"]).get_file())
		var file := FileAccess.open(path, FileAccess.READ)
		if file == null:
			_fail("could not read %s" % path)
			continue
		var definition = JSON.parse_string(file.get_as_text())
		file.close()
		if typeof(definition) == TYPE_DICTIONARY:
			_definitions.append(definition)

func _read_samples(directory: String, identifier: String) -> Array:
	var result: Array = []
	var path := ProjectSettings.globalize_path(directory).path_join(identifier + ".jsonl")
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return result
	while not file.eof_reached():
		var line := file.get_line().strip_edges()
		if line.is_empty():
			continue
		var sample = JSON.parse_string(line)
		if typeof(sample) != TYPE_DICTIONARY:
			continue
		sample.erase("timestamp")
		result.append(sample)
	file.close()
	return result

func _compare() -> void:
	for definition in _definitions:
		var identifier: String = definition["trajectory_id"]
		var first := _read_samples(RUN_DIRECTORIES[0], identifier)
		var second := _read_samples(RUN_DIRECTORIES[1], identifier)
		if first.is_empty():
			_fail("%s: first replay produced no samples" % identifier)
			continue
		if first.size() != second.size():
			_fail("%s: sample count differs (%d vs %d)" % [identifier, first.size(), second.size()])
			continue

		var hard := 0
		var tolerance_only := 0
		var detail := ""
		var max_coverage_delta := 0.0
		var observations := 0
		var visible := 0
		for i in first.size():
			var verdict := _compare_sample(first[i], second[i])
			observations += first[i].get("observations", []).size()
			visible += first[i].get("visible_count", 0)
			if not verdict["hard"].is_empty():
				hard += 1
				if detail.is_empty():
					detail = "\n        row %d: %s" % [i, verdict["hard"]]
			elif not verdict["numeric"].is_empty():
				tolerance_only += 1
				max_coverage_delta = maxf(max_coverage_delta, verdict["max_delta"])

		if hard > 0:
			_fail("%s: %d of %d samples differ in discrete fields%s"
				% [identifier, hard, first.size(), detail])
		elif tolerance_only > 0:
			_ok("%s: %d samples match exactly; %d differ only within numeric tolerance (max |dcoverage| = %.7f)"
				% [identifier, first.size(), tolerance_only, max_coverage_delta])
		else:
			_ok("%s: %d samples identical across both replays" % [identifier, first.size()])

		print("      %s: %d observations, %d frustum-visible" % [identifier, observations, visible])
		if observations == 0:
			_fail("%s: the replay recorded no observations" % identifier)
		if visible == 0:
			_fail("%s: the replay never saw a frustum-visible object" % identifier)

func _compare_sample(left: Dictionary, right: Dictionary) -> Dictionary:
	var hard: Array = []
	var numeric: Array = []
	var max_delta := 0.0
	for key in left:
		if key == "observations":
			continue
		if not right.has(key):
			hard.append("missing key %s" % key)
		elif left[key] != right[key]:
			hard.append("%s: %s vs %s" % [key, left[key], right[key]])
	var left_observations: Array = left.get("observations", [])
	var right_observations: Array = right.get("observations", [])
	if left_observations.size() != right_observations.size():
		hard.append("observation count %d vs %d" % [left_observations.size(), right_observations.size()])
		return {"hard": hard, "numeric": numeric, "max_delta": max_delta}
	for i in left_observations.size():
		var a: Dictionary = left_observations[i]
		var b: Dictionary = right_observations[i]
		if a.get("object_id") != b.get("object_id"):
			hard.append("obs[%d] id %s vs %s" % [i, a.get("object_id"), b.get("object_id")])
			continue
		for key in EXACT_OBSERVATION_KEYS:
			if a.get(key) != b.get(key):
				hard.append("obs[%d].%s: %s vs %s" % [i, key, a.get(key), b.get(key)])
		for key in NUMERIC_OBSERVATION_KEYS:
			var delta: float = absf(float(a.get(key, 0.0)) - float(b.get(key, 0.0)))
			if delta > NUMERIC_OBSERVATION_KEYS[key]:
				hard.append("obs[%d].%s delta %.9f exceeds tolerance" % [i, key, delta])
			elif key == "coverage" and delta > 0.0:
				numeric.append(i)
				max_delta = maxf(max_delta, delta)
		if not _resolution_close(a.get("required_resolution", 0), b.get("required_resolution", 0)):
			hard.append("obs[%d].required_resolution: %s vs %s"
				% [i, a.get("required_resolution"), b.get("required_resolution")])
	return {"hard": hard, "numeric": numeric, "max_delta": max_delta}

## Allows a required resolution to differ by at most one ladder step, which can
## happen when the underlying coverage sits exactly on a ladder boundary.
func _resolution_close(left: int, right: int) -> bool:
	if left == right:
		return true
	var left_index := RESOLUTION_LADDER.find(left)
	var right_index := RESOLUTION_LADDER.find(right)
	if left_index < 0 or right_index < 0:
		return false
	return absi(left_index - right_index) <= 1

func _cleanup() -> void:
	for directory in RUN_DIRECTORIES:
		var path := ProjectSettings.globalize_path(directory)
		for definition in _definitions:
			var identifier: String = definition["trajectory_id"]
			for suffix in [".jsonl", ".meta.json"]:
				DirAccess.remove_absolute(path.path_join(identifier + suffix))

func _ok(message: String) -> void:
	print("  ok  : %s" % message)

func _fail(message: String) -> void:
	_failures.append(message)
	print("  FAIL: %s" % message)

func _report() -> void:
	_cleanup()
	print("\n=== result: %s ===" % ("FAILED" if _failures.size() > 0 else "OK"))
	for failure in _failures:
		print("  FAIL: %s" % failure)
	quit(1 if _failures.size() > 0 else 0)
