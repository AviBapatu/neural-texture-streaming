extends SceneTree
## Phase 5 / Steps 4-5 - live in-engine predictive texture streaming.
##
## Replays scripted trajectories with the streaming controller active. Three modes:
##
##   --mode=demo        interactive: run trajectories, switch policy with 1-4, F1 tint
##   --mode=capture     run one trajectory per policy, screenshotting at a fixed cadence
##                      (same camera each run, so the captures are frame-comparable)
##   --mode=comparison  back-to-back replay of one trajectory under two policies,
##                      writing comparison.json with each policy's metrics
##
## Run (demo):
##   godot --path godot --fixed-fps 60 --resolution 1280x720 \
##       --script res://tools/run_streaming.gd -- --mode=demo \
##       --trajectory=t04_sprint_corner --policy=heuristic_occ \
##       --out=dataset/streaming/runs/demo
##
## Scope note: the engine streams the 53 objects that use one of the 7 ambientCG
## identities with generated variants. Every number is scoped to those 53.

const MAIN_SCENE := "res://scenes/main.tscn"
const ENGINE_DIR := "dataset/streaming/runs/phase5_engine"

var _main: Node
var _runner: TrajectoryRunner
var _controller: StreamingController
var _env: Node
var _camera: Camera3D

var _mode := "demo"
var _out := ""
var _trajectory := "t04_sprint_corner"
var _policies: Array = ["heuristic_occ"]
var _budget_fraction := 1.0
var _latency := 0.5
var _bandwidth_mb_s := 200.0
var _capture_every := 10
var _warmup := 20
var _duration := 0.0

var _stage := 0
var _stage_frames := 0
var _plan_index := 0
var _results: Array = []
var _capture_count := 0
var _definitions: Array = []
var _score_rows: Dictionary = {}
var _peak := 1.0


func _initialize() -> void:
	if not _parse_args():
		quit(1)
		return
	_run.call_deferred()


func _run() -> void:
	var packed := load(MAIN_SCENE) as PackedScene
	if packed == null:
		push_error("[streaming] main.tscn did not load")
		quit(1)
		return
	_main = packed.instantiate()
	var recorder: Node = _main.get_node_or_null("TelemetryRecorder")
	if recorder != null:
		recorder.set("auto_start", false)
		recorder.set("enabled", false)
		recorder.set("record_jsonl", false)
		recorder.set("record_csv", false)
	root.add_child(_main)

	_runner = TrajectoryRunner.new()
	_runner.name = "TrajectoryRunner"
	_runner.recorder = null
	_main.add_child(_runner)

	_controller = StreamingController.new()
	_controller.name = "StreamingController"
	_main.add_child(_controller)

	_load_definitions()


func _process(_delta: float) -> bool:
	_stage_frames += 1
	match _stage:
		0:
			if _stage_frames < _warmup:
				return false
			_resolve()
			_controller.configure({
				"apply_textures": _mode == "demo" or _mode == "capture",
				"bandwidth_mb_s": _bandwidth_mb_s,
				"latency_seconds": _latency,
				"budget_fraction": _budget_fraction,
			})
			_controller.base_peak_bytes = _peak
			_controller.budget_bytes = maxf(_peak * _budget_fraction, 1.0)
			_begin_plan_entry()
			_stage = 1
			return false
		1:
			_capture_if_due()
			if _runner.is_finished():
				_results.append(_record_entry())
				_plan_index += 1
				if _plan_index >= _policies.size():
					_finish()
					return true
				_begin_plan_entry()
			return false
	return false


func _resolve() -> void:
	_env = get_first_node_in_group("environment")
	_camera = get_first_node_in_group("player_camera") as Camera3D
	_controller.camera = _camera
	_controller.tracked = _env.get("tracked_objects")
	_controller.runner = _runner
	var manifest: Variant = _read_json(ENGINE_DIR.path_join("manifest.json"))
	if manifest is Dictionary:
		var peaks: Dictionary = manifest.get("peak_working_set_bytes", {})
		if peaks.has(_trajectory):
			_peak = float(peaks[_trajectory])
	_controller.base_peak_bytes = _peak
	_controller.budget_bytes = maxf(_peak * _budget_fraction, 1.0)
	_controller.prepare()
	var scores: Variant = _read_json(ENGINE_DIR.path_join("scores").path_join(_trajectory + ".json"))
	if scores is Dictionary:
		_score_rows = scores.get("rows", {})
	_controller.score_rows = _score_rows
	print("[streaming] mode=%s trajectory=%s peak_ws=%.1f MB budget=%.1f MB latency=%.1fs bw=%sMB/s" % [
		_mode, _trajectory, _peak / 1e6, _controller.budget_bytes / 1e6, _latency, _bandwidth_mb_s])


func _begin_plan_entry() -> void:
	var policy: String = _policies[_plan_index]
	_controller.set_policy(policy)
	_controller.reset_state()
	_controller.score_rows = _score_rows
	_capture_count = 0
	_runner.queue([_definitions[0]])
	print("[streaming] run %d/%d policy=%s" % [_plan_index + 1, _policies.size(), policy])


func _record_entry() -> Dictionary:
	var metrics: Dictionary = _controller.finalise()
	metrics["policy"] = _policies[_plan_index]
	metrics["trajectory"] = _trajectory
	metrics["budget_bytes"] = _controller.budget_bytes
	metrics["latency_seconds"] = _latency
	metrics["bandwidth_mb_s"] = _bandwidth_mb_s
	print("[streaming] %s deficit=%.4f bytes=%.1fMB wasted=%d loads=%d popin=%d" % [
		metrics["policy"], metrics["deficit"], metrics["bytes_streamed"] / 1e6,
		metrics["wasted_prefetches"], metrics["loads_completed"], metrics["pop_in_events"]])
	return metrics


## Capture mode: shoot at a fixed cadence while the runner is active.
func _capture_if_due() -> void:
	if _mode != "capture":
		return
	if _stage != 1 or not _runner.is_running():
		return
	var shots := _out.path_join("shots_%s" % _policies[_plan_index])
	DirAccess.make_dir_recursive_absolute(shots)
	var sample: int = int(_controller.get("samples"))
	if sample > _capture_count * _capture_every and sample > 0:
		_capture_count += 1
		var img := root.get_texture().get_image()
		img.save_png(shots.path_join("sample_%04d.png" % sample))


func _finish() -> void:
	DirAccess.make_dir_recursive_absolute(_out)
	var summary := {
		"mode": _mode, "trajectory": _trajectory, "scope": "53 streamable objects",
		"budget_fraction": _budget_fraction, "latency_seconds": _latency,
		"bandwidth_mb_s": _bandwidth_mb_s, "peak_working_set_bytes": _peak,
		"runs": _results,
	}
	var name := "comparison.json" if _mode == "comparison" else "streaming_%s.json" % _mode
	var f := FileAccess.open(_out.path_join(name), FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(summary, "  "))
	print("[streaming] wrote ", _out.path_join(name))
	print(JSON.stringify(summary))


func _load_definitions() -> void:
	var index_path := "dataset/trajectories/full/index.json"
	var index: Variant = _read_json(index_path)
	if not (index is Dictionary):
		return
	var base := _repo_root().path_join("dataset/trajectories/full")
	for entry in index.get("trajectories", []):
		if String(entry["trajectory_id"]) != _trajectory:
			continue
		var definition: Variant = _read_json(base.path_join(String(entry["file"]).get_file()))
		if definition is Dictionary:
			_definitions.append(definition)
	if _definitions.is_empty():
		push_error("[streaming] trajectory not found: " + _trajectory)


func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()


func _read_json(relative: String) -> Variant:
	var path := relative if relative.begins_with("/") else _repo_root().path_join(relative)
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		return null
	return JSON.parse_string(text)


func _parse_args() -> bool:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--mode="):
			_mode = a.trim_prefix("--mode=")
		elif a.begins_with("--out="):
			_out = a.trim_prefix("--out=")
		elif a.begins_with("--trajectory="):
			_trajectory = a.trim_prefix("--trajectory=")
		elif a.begins_with("--policy="):
			_policies = [a.trim_prefix("--policy=")]
		elif a.begins_with("--policies="):
			_policies = a.trim_prefix("--policies=").split(",")
		elif a.begins_with("--budget-fraction="):
			_budget_fraction = float(a.trim_prefix("--budget-fraction="))
		elif a.begins_with("--latency="):
			_latency = float(a.trim_prefix("--latency="))
		elif a.begins_with("--bandwidth-mb-s="):
			_bandwidth_mb_s = float(a.trim_prefix("--bandwidth-mb-s="))
		elif a.begins_with("--capture-every="):
			_capture_every = int(a.trim_prefix("--capture-every="))
		elif a.begins_with("--warmup="):
			_warmup = int(a.trim_prefix("--warmup="))
	if _out.is_empty():
		_out = "dataset/streaming/runs/%s" % _mode
	if not _out.begins_with("/"):
		_out = _repo_root().path_join(_out)
	if _mode == "comparison" and _policies.size() < 2:
		_policies = ["reactive_visible", "mlp_b"]
	return true
