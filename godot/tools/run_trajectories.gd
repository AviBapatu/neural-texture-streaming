extends SceneTree
## Runs a set of generated trajectories and records raw telemetry for each.
##
##   godot --headless --fixed-fps 60 --path godot --script res://tools/run_trajectories.gd -- \
##       --trajectories=dataset/trajectories/full/index.json \
##       --output=dataset/raw/run_full \
##       --occlusion=1 --sample-interval=6 --max-distance=60
##
## Arguments (all optional except --trajectories and --output):
##   --trajectories=PATH   trajectory index.json written by generate_trajectories.py
##   --output=DIR          where the raw JSONL + meta pairs are written
##   --occlusion=0|1       enable the occlusion raycast signal (default 0)
##   --legacy-visible=0|1  also emit the Phase 1 `visible` array (default 0)
##   --sample-interval=N   physics ticks between samples (default 6 -> 10 Hz)
##   --max-distance=M      candidate observation distance (default 60)
##   --occlusion-rays=N    ray budget per object (default 4)
##   --limit=N             only run the first N trajectories
##   --warmup-ticks=N      ticks to let the environment settle before the first run
##
## One process runs every trajectory so the environment is generated once, which
## makes the run cheaper and guarantees every trajectory sees exactly the same world.
## Run under `--fixed-fps` so the simulation advances faster than real time.

const MAIN_SCENE := "res://scenes/main.tscn"

var _main: Node
var _recorder: Node
var _runner: TrajectoryRunner
var _definitions: Array = []
var _started_at := 0
var _warmup := 10
var _stage := 0
var _stage_frames := 0
var _run_id := ""

func _initialize() -> void:
	var options := _parse_arguments(OS.get_cmdline_user_args())
	if options.is_empty():
		quit(1)
		return
	_run_id = options["run_id"]
	_load_definitions(options["trajectories"], int(options["limit"]))
	_warmup = int(options["warmup_ticks"])

	var packed := load(MAIN_SCENE) as PackedScene
	if packed == null:
		push_error("main.tscn did not load")
		quit(1)
		return
	_main = packed.instantiate()
	_recorder = _main.get_node_or_null("TelemetryRecorder")
	if _recorder == null:
		push_error("TelemetryRecorder missing from main.tscn")
		quit(1)
		return
	# Configure before entering the tree so _resolve_nodes does not auto-start.
	_recorder.set("auto_start", false)
	_recorder.set("sample_mode", "physics")
	_recorder.set("sample_interval_ticks", int(options["sample_interval"]))
	_recorder.set("max_visible_distance", float(options["max_distance"]))
	_recorder.set("occlusion_enabled", options["occlusion"])
	_recorder.set("occlusion_max_rays", int(options["occlusion_rays"]))
	_recorder.set("legacy_visible_array", options["legacy_visible"])
	_recorder.set("record_csv", false)
	_recorder.set("record_jsonl", true)
	_recorder.set("print_progress", false)
	_recorder.set("output_directory", options["output"])
	root.add_child(_main)

	_runner = TrajectoryRunner.new()
	_runner.name = "TrajectoryRunner"
	_runner.recorder = _recorder
	_main.add_child(_runner)

	print("=== run trajectories ===")
	print("  run id        : %s" % _run_id)
	print("  trajectories  : %d" % _definitions.size())
	print("  output        : %s" % options["output"])
	print("  occlusion     : %s (%d rays max)" % [options["occlusion"], int(options["occlusion_rays"])])
	print("  sample every  : %d physics ticks" % int(options["sample_interval"]))
	print("  max distance  : %s" % options["max_distance"])

func _process(_delta: float) -> bool:
	_stage_frames += 1
	match _stage:
		0:
			if _stage_frames < _warmup:
				return false
			if _definitions.is_empty():
				push_error("no trajectories to run")
				quit(1)
				return true
			_started_at = Time.get_ticks_msec()
			_runner.queue(_definitions)
			_stage = 1
			return false
		1:
			if not _runner.is_finished():
				return false
			_finish()
			return true
	return false

func _finish() -> void:
	var elapsed_ms := Time.get_ticks_msec() - _started_at
	var results: Array = _runner.take_results()
	var total_samples := 0
	var total_observations := 0
	for result in results:
		total_samples += result["samples"]
		total_observations += result["observations"]
	var summary := {
		"schema_version": 1,
		"run_id": _run_id,
		"trajectory_count": results.size(),
		"total_samples": total_samples,
		"total_observations": total_observations,
		"wall_clock_ms": elapsed_ms,
		"trajectories": results,
	}
	var directory: String = _recorder.get("output_directory")
	DirAccess.make_dir_recursive_absolute(directory)
	var path := directory.path_join("run_summary.json")
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file != null:
		file.store_string(JSON.stringify(summary, "  ") + "\n")
		file.close()
	print("\n  wrote %s" % path)
	print("  %d samples, %d observations across %d trajectories in %.1fs"
		% [total_samples, total_observations, results.size(), elapsed_ms / 1000.0])
	print("\n=== done ===")

func _load_definitions(index_path: String, limit: int) -> void:
	var resolved := _resolve_path(index_path)
	var file := FileAccess.open(resolved, FileAccess.READ)
	if file == null:
		push_error("trajectory index not found: %s" % resolved)
		quit(1)
		return
	var index: Dictionary = JSON.parse_string(file.get_as_text())
	file.close()
	var base := resolved.get_base_dir()
	for entry in index.get("trajectories", []):
		if limit > 0 and _definitions.size() >= limit:
			break
		var path := base.path_join(String(entry["file"]).get_file())
		var definition_file := FileAccess.open(path, FileAccess.READ)
		if definition_file == null:
			push_error("could not read trajectory %s" % path)
			continue
		var definition = JSON.parse_string(definition_file.get_as_text())
		definition_file.close()
		if typeof(definition) == TYPE_DICTIONARY:
			_definitions.append(definition)
		else:
			push_error("trajectory %s is not a JSON object" % path)

## Resolves a project-relative path (as documented in the README) against the
## repository root rather than the Godot project directory.
func _resolve_path(path: String) -> String:
	if path.begins_with("res://") or path.begins_with("/"):
		return path
	return ProjectSettings.globalize_path("res://").path_join("..").simplify_path().path_join(path)

func _parse_arguments(arguments: PackedStringArray) -> Dictionary:
	var options := {
		"trajectories": "",
		"output": "",
		"run_id": "run",
		"occlusion": false,
		"legacy_visible": false,
		"sample_interval": 6,
		"max_distance": 60.0,
		"occlusion_rays": 4,
		"limit": 0,
		"warmup_ticks": 10,
	}
	for argument in arguments:
		if argument.begins_with("--trajectories="):
			options["trajectories"] = argument.trim_prefix("--trajectories=")
		elif argument.begins_with("--output="):
			options["output"] = argument.trim_prefix("--output=")
		elif argument.begins_with("--run-id="):
			options["run_id"] = argument.trim_prefix("--run-id=")
		elif argument.begins_with("--occlusion="):
			options["occlusion"] = argument.trim_prefix("--occlusion=") == "1"
		elif argument.begins_with("--legacy-visible="):
			options["legacy_visible"] = argument.trim_prefix("--legacy-visible=") == "1"
		elif argument.begins_with("--sample-interval="):
			options["sample_interval"] = int(argument.trim_prefix("--sample-interval="))
		elif argument.begins_with("--max-distance="):
			options["max_distance"] = float(argument.trim_prefix("--max-distance="))
		elif argument.begins_with("--occlusion-rays="):
			options["occlusion_rays"] = int(argument.trim_prefix("--occlusion-rays="))
		elif argument.begins_with("--limit="):
			options["limit"] = int(argument.trim_prefix("--limit="))
		elif argument.begins_with("--warmup-ticks="):
			options["warmup_ticks"] = int(argument.trim_prefix("--warmup-ticks="))
	if String(options["trajectories"]).is_empty():
		push_error("--trajectories=PATH is required")
		return {}
	if String(options["output"]).is_empty():
		push_error("--output=DIR is required")
		return {}
	options["output"] = _resolve_path(String(options["output"]))
	return options
