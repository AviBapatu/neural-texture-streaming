extends SceneTree
## Phase 5 demo - pick the demo segment.
##
## Scans the four held-out test trajectories for the window where reactive_visible and
## mlp_b tau=0.5 differ most *on screen*. Per 10 Hz sample the metric is
##     (#objects visible but not resident) * (their total viewport coverage)
## i.e. "how much low-resolution placeholder is on screen, times how many objects".
## A window is scored by sum(metric_reactive - metric_mlp); the biggest single-sample
## reactive metric is the headline pop-in event to replay in slow motion.
##
##   godot --headless --path godot --script res://scripts/demo/scan_segments.gd -- \
##       --out=dataset/streaming/runs/demo

const ENGINE_DIR := "dataset/streaming/runs/phase5_engine"
const TRAJECTORIES := ["t01_walk_straight", "t03_walk_corner", "t04_sprint_corner", "t06_look_around"]
const WINDOW_SAMPLES := 60   # 6 s at 10 Hz
const WARMUP_SAMPLES := 20   # skip the cold-start transient (nothing resident yet)

var _out := ""

func _initialize() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			_out = a.trim_prefix("--out=")
	_run.call_deferred()

func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()

func _read_json(relative: String) -> Variant:
	var path := relative if relative.begins_with("/") else _repo_root().path_join(relative)
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		return null
	return JSON.parse_string(text)

func _run() -> void:
	if _out.is_empty():
		_out = "dataset/streaming/runs/demo"
	if not _out.begins_with("/"):
		_out = _repo_root().path_join(_out)
	DirAccess.make_dir_recursive_absolute(_out)

	var manifest: Variant = _read_json(ENGINE_DIR.path_join("manifest.json"))
	if not (manifest is Dictionary):
		quit(1)
		return
	var sizes := {}
	for o in manifest["objects"]:
		sizes[String(o["object_id"])] = float(o["full_bytes"])
	var peaks: Dictionary = manifest["peak_working_set_bytes"]

	var candidates: Array = []
	var events: Array = []
	for traj in TRAJECTORIES:
		var scores: Variant = _read_json(ENGINE_DIR.path_join("scores").path_join(traj + ".json"))
		if not (scores is Dictionary):
			push_warning("[scan] missing %s" % traj)
			continue
		var samples: Array = scores["samples"]
		var rows: Dictionary = scores["rows"]
		var pose: Dictionary = scores["pose"]
		var snap: Array = scores.get("snap_samples", [])
		var peak := float(peaks.get(traj, 1.0))
		var reactive := _profile(samples, rows, pose, snap, sizes, peak, "reactive_visible")
		var mlp := _profile(samples, rows, pose, snap, sizes, peak, "mlp_b_tau0.5")
		var n: int = samples.size()
		var metric_r := PackedFloat64Array()
		var metric_m := PackedFloat64Array()
		for i in n:
			metric_r.append(reactive["count"][i] * reactive["coverage"][i])
			metric_m.append(mlp["count"][i] * mlp["coverage"][i])
		var best_event := mini(WARMUP_SAMPLES, maxi(0, n - 1))
		for i in range(best_event, n):
			if metric_r[i] > metric_r[best_event]:
				best_event = i
		events.append({
			"trajectory": traj, "event_sample": samples[best_event],
			"event_seconds": samples[best_event] * 0.1,
			"reactive_metric": metric_r[best_event], "mlp_metric": metric_m[best_event],
			"reactive_count": reactive["count"][best_event], "mlp_count": mlp["count"][best_event],
		})
		var peak_count_traj := 0
		for i in n:
			peak_count_traj = maxi(peak_count_traj, reactive["count"][i])
		print("[scan] %s: %d samples, reactive pop-in count max=%d" % [traj, n, peak_count_traj])
		for start in range(WARMUP_SAMPLES, n - WINDOW_SAMPLES):
			var score := 0.0
			var peak_count := 0
			var peak_cov := 0.0
			for i in range(start, start + WINDOW_SAMPLES):
				score += metric_r[i] - metric_m[i]
				peak_count = maxi(peak_count, reactive["count"][i])
				peak_cov = maxf(peak_cov, reactive["coverage"][i])
			candidates.append({
				"trajectory": traj, "start_sample": samples[start],
				"end_sample": samples[start + WINDOW_SAMPLES - 1],
				"start_seconds": samples[start] * 0.1,
				"end_seconds": samples[start + WINDOW_SAMPLES - 1] * 0.1,
				"score": score, "peak_lowres_count": peak_count, "peak_popin_coverage": peak_cov,
			})
	candidates.sort_custom(func(a, b): return a["score"] > b["score"])
	events.sort_custom(func(a, b): return a["reactive_metric"] > b["reactive_metric"])

	var best: Dictionary = {} if candidates.is_empty() else candidates[0]
	var best_event: Dictionary = {} if events.is_empty() else events[0]
	print("[scan] top 3 windows (score = sum count*coverage, reactive - mlp):")
	for c in candidates.slice(0, 3):
		print("[scan]   %s  %.1f-%.1fs  score=%.4f  peak_lowres=%d  peak_cov=%.3f" % [
			c["trajectory"], c["start_seconds"], c["end_seconds"], c["score"],
			c["peak_lowres_count"], c["peak_popin_coverage"]])
	print("[scan] biggest pop-in events:")
	for e in events:
		print("[scan]   %s  s=%d (%.1fs)  reactive metric=%.4f (%d objs)  mlp=%.4f (%d objs)" % [
			e["trajectory"], e["event_sample"], e["event_seconds"], e["reactive_metric"],
			e["reactive_count"], e["mlp_metric"], e["mlp_count"]])

	# The event that belongs to the winning trajectory is the one to stage.
	var stage_event: Dictionary = best_event
	for e in events:
		if e["trajectory"] == best.get("trajectory", ""):
			stage_event = e
			break
	var report := {
		"window_samples": WINDOW_SAMPLES, "candidates": candidates.slice(0, 3),
		"events": events, "best": best, "stage_event": stage_event,
		"replay_start_sample": maxi(0, int(stage_event.get("event_sample", 0)) - 15),
		"replay_end_sample": int(stage_event.get("event_sample", 0)) + 15,
	}
	var f := FileAccess.open(_out.path_join("demo_segment.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(report, "  "))
	print("[scan] chosen: %s  replay samples %d-%d  (%.1f-%.1fs), event at %.1fs" % [
		best.get("trajectory", "?"), report["replay_start_sample"], report["replay_end_sample"],
		int(report["replay_start_sample"]) * 0.1, int(report["replay_end_sample"]) * 0.1,
		stage_event.get("event_seconds", 0.0)])
	quit(0)

## Per-sample pop-in count and coverage for one policy.
func _profile(samples: Array, rows: Dictionary, pose: Dictionary, snap: Array, sizes: Dictionary,
		peak: float, policy: String) -> Dictionary:
	var ds := DemoStream.new()
	ds.setup("scan", samples, rows, pose, snap, sizes, peak, 200.0e6, 1.0, policy)
	var count := PackedInt32Array()
	var coverage := PackedFloat64Array()
	var visible := PackedInt32Array()
	for i in samples.size():
		ds.advance(i)
		var entries := ds.visible_popin()
		var cov := 0.0
		for entry in entries:
			cov += float(entry["coverage"])
		count.append(entries.size())
		coverage.append(cov)
		visible.append(ds.visible_count())
	return {"count": count, "coverage": coverage, "visible": visible}
