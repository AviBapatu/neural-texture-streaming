extends SceneTree
## Measures the cost of the occlusion raycast pass.
##
## Run with:
##   godot --headless --path godot --script res://tools/benchmark_visibility.gd
##
## Walks the same camera path twice - once with occlusion disabled and once with it
## enabled - and reports the visibility cost per sample for both, so the overhead is
## a measured number rather than a guess. Writes benchmarks/occlusion_overhead.json.

const ENVIRONMENT_SCENE := "res://scenes/main.tscn"
const WARMUP_FRAMES := 40
const MEASURE_FRAMES := 300
const SAMPLE_INTERVAL_TICKS := 6
const OUTPUT_RELATIVE := "../benchmarks/occlusion_overhead.json"
const TELEMETRY_RELATIVE := "../benchmarks/telemetry"

var _stage := 0
var _frames := 0
var _stage_frames := 0
var _main: Node
var _player: CharacterBody3D
var _recorder: Node
var _off := {}
var _on := {}

func _initialize() -> void:
	print("=== occlusion overhead benchmark ===\n")
	var packed := load(ENVIRONMENT_SCENE) as PackedScene
	if packed == null:
		push_error("main.tscn did not load")
		quit(1)
		return
	_main = packed.instantiate()
	# Configure the recorder before it enters the tree, so its deferred _resolve_nodes
	# does not open an auto-started session.
	_recorder = _main.get_node_or_null("TelemetryRecorder")
	if _recorder != null:
		_recorder.set("auto_start", false)
		_recorder.set("record_jsonl", false)
		_recorder.set("record_csv", false)
		_recorder.set("sample_mode", "physics")
		_recorder.set("sample_interval_ticks", SAMPLE_INTERVAL_TICKS)
		_recorder.set("occlusion_enabled", false)
		_recorder.set("output_directory", _path(TELEMETRY_RELATIVE))
	root.add_child(_main)

func _process(_delta: float) -> bool:
	_frames += 1
	_stage_frames += 1
	if _frames < 3:
		return false
	match _stage:
		0:
			_player = _main.get_node_or_null("Player") as CharacterBody3D
			if _player == null or _recorder == null:
				push_error("Player or TelemetryRecorder missing")
				quit(1)
				return true
			_start_run("benchmark_occlusion_off")
			_enter(1)
			return false
		1:
			if _stage_frames < WARMUP_FRAMES + MEASURE_FRAMES:
				return false
			_off = _capture("occlusion disabled")
			_input_off()
			_player.reset_to_start()
			_recorder.set("occlusion_enabled", true)
			_start_run("benchmark_occlusion_on")
			_enter(2)
			return false
		2:
			if _stage_frames < WARMUP_FRAMES + MEASURE_FRAMES:
				return false
			_on = _capture("occlusion enabled")
			_recorder.call("close_session")
			_report()
			return true
	return false

func _enter(next_stage: int) -> void:
	_stage = next_stage
	_stage_frames = 0

func _start_run(stem: String) -> void:
	Input.action_press("move_forward")
	_recorder.call("begin_session", stem, {
		"purpose": "occlusion overhead benchmark",
		"warmup_frames": WARMUP_FRAMES,
		"measure_frames": MEASURE_FRAMES,
	})

func _input_off() -> void:
	Input.action_release("move_forward")
	Input.action_release("sprint")

func _capture(label: String) -> Dictionary:
	_input_off()
	var samples: int = _recorder.get("samples_written")
	var visibility_ms: float = _recorder.get("visibility_time_ms")
	var occlusion_ms: float = _recorder.get("occlusion_time_ms")
	var rays: int = _recorder.get("occlusion_rays_cast")
	var result := {
		"label": label,
		"occlusion_enabled": _recorder.get("occlusion_enabled"),
		"samples": samples,
		"rays_cast": rays,
		"total_visibility_ms": snappedf(visibility_ms, 0.001),
		"mean_visibility_ms": snappedf(visibility_ms / maxf(1.0, samples), 0.0001),
		"total_occlusion_ms": snappedf(occlusion_ms, 0.001),
		"mean_occlusion_ms": snappedf(occlusion_ms / maxf(1.0, samples), 0.0001),
		"mean_rays_per_sample": snappedf(float(rays) / maxf(1.0, samples), 0.01),
	}
	print("[%s] samples=%d rays=%d mean visibility=%.4f ms occlusion=%.4f ms"
		% [label, samples, rays, result["mean_visibility_ms"], result["mean_occlusion_ms"]])
	return result

func _report() -> void:
	var delta_ms: float = _on["mean_visibility_ms"] - _off["mean_visibility_ms"]
	var overhead_percent := 0.0
	if _off["mean_visibility_ms"] > 0.0:
		overhead_percent = delta_ms / _off["mean_visibility_ms"] * 100.0
	var report := {
		"schema_version": 1,
		"sample_interval_ticks": SAMPLE_INTERVAL_TICKS,
		"physics_ticks_per_second": Engine.physics_ticks_per_second,
		"warmup_frames": WARMUP_FRAMES,
		"measure_frames": MEASURE_FRAMES,
		"occlusion_off": _off,
		"occlusion_on": _on,
		"mean_overhead_ms_per_sample": snappedf(delta_ms, 0.0001),
		"overhead_percent_of_visibility": snappedf(overhead_percent, 0.1),
		"sample_budget_ms": snappedf(1000.0 * float(SAMPLE_INTERVAL_TICKS) / float(Engine.physics_ticks_per_second), 0.001),
	}
	var path := _path(OUTPUT_RELATIVE)
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file != null:
		file.store_string(JSON.stringify(report, "  ") + "\n")
		file.close()
	print("\n  sample budget           : %.2f ms (every %d physics ticks)"
		% [report["sample_budget_ms"], SAMPLE_INTERVAL_TICKS])
	print("  visibility without rays : %.4f ms/sample" % _off["mean_visibility_ms"])
	print("  visibility with rays    : %.4f ms/sample" % _on["mean_visibility_ms"])
	print("  occlusion added         : %.4f ms/sample (%.1f%%), %.1f rays/sample"
		% [delta_ms, overhead_percent, _on["mean_rays_per_sample"]])
	print("  report                  : %s" % path)
	print("\n=== done ===")

func _path(relative: String) -> String:
	return ProjectSettings.globalize_path("res://").path_join(relative).simplify_path()
