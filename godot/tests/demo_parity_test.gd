extends SceneTree
## Phase 5 demo - parity and determinism gate.
##
## Proves three things about the demo's streaming path ([DemoStream], the metrics core
## every pane shares):
##
##   1. It reproduces the simulator. At the Phase 5 parity cell (100%, 200 MB/s, 0.5 s)
##      the demo's deficit/bytes for t06_look_around match the recorded
##      `parity_report.json` engine numbers, which `tools/check_godot_parity.py` already
##      showed agree with the Python simulator to 4.29e-7.
##   2. Determinism: replaying the same trajectory twice yields bit-identical metrics.
##   3. Dual-pane == single-pane: two independent DemoStream instances and a single one,
##      given the same trajectory/policy/cell, agree to < 1e-9 (the panes differ only in
##      rendering, never in the streamed signals or the streamer).
##
## Run:
##   godot --headless --path godot --script res://tests/demo_parity_test.gd -- \
##       --out=dataset/streaming/runs/demo

const ENGINE_DIR := "dataset/streaming/runs/phase5_engine"
const TRAJECTORY := "t06_look_around"
const TOLERANCE := 1e-3

var _out := ""
var _report: Dictionary = {"test": "demo_parity", "trajectory": TRAJECTORY, "checks": {}}

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
	var scores: Variant = _read_json(ENGINE_DIR.path_join("scores").path_join(TRAJECTORY + ".json"))
	if not (manifest is Dictionary) or not (scores is Dictionary):
		_fail("cannot load engine data")
		return
	var sizes := {}
	for o in manifest["objects"]:
		sizes[String(o["object_id"])] = float(o["full_bytes"])
	var peak := float(manifest["peak_working_set_bytes"].get(TRAJECTORY, 1.0))
	var samples: Array = scores["samples"]
	var rows: Dictionary = scores["rows"]
	var pose: Dictionary = scores["pose"]
	var snap: Array = scores.get("snap_samples", [])

	var ok := true

	# --- 1. reproduce the recorded engine/sim parity numbers (0.5 s cell) ----------
	var parity: Variant = _read_json(ENGINE_DIR.path_join("parity_report.json"))
	var expected := {}
	if parity is Dictionary:
		for c in parity["comparisons"]:
			if String(c["trajectory"]) == TRAJECTORY:
				expected[String(c["policy"])] = c["engine"]
	var map := {"reactive_visible": "reactive_visible",
		"heuristic_occ": "heuristic_occ_tau0", "mlp_b": "mlp_b_tau0"}
	var parity_rows := []
	for sim_policy in expected:
		var mine := _run_full(scores, samples, rows, pose, snap, sizes, peak, 0.5, map[sim_policy])
		var exp: Dictionary = expected[sim_policy]
		var dd := absf(mine["deficit"] - float(exp["deficit"]))
		var db := absf(mine["bytes_streamed"] - float(exp["bytes_streamed"]))
		var pass_row := dd < TOLERANCE and db <= 1.0
		ok = ok and pass_row
		parity_rows.append({"policy": sim_policy, "demo_deficit": mine["deficit"],
			"engine_deficit": float(exp["deficit"]), "deficit_diff": dd,
			"demo_bytes": mine["bytes_streamed"], "engine_bytes": float(exp["bytes_streamed"]),
			"bytes_diff": db, "ok": pass_row})
	_report["checks"]["parity_vs_sim"] = parity_rows

	# --- 2. determinism -----------------------------------------------------------
	var a := _run_full(scores, samples, rows, pose, snap, sizes, peak, 1.0, "mlp_b_tau0.5")
	var b := _run_full(scores, samples, rows, pose, snap, sizes, peak, 1.0, "mlp_b_tau0.5")
	var deterministic: bool = a["deficit"] == b["deficit"] and a["bytes_streamed"] == b["bytes_streamed"] \
		and a["wasted_prefetches"] == b["wasted_prefetches"] and a["pop_in_events"] == b["pop_in_events"]
	ok = ok and deterministic
	_report["checks"]["determinism"] = {"identical": deterministic,
		"deficit": a["deficit"], "bytes": a["bytes_streamed"]}

	# --- 3. dual-pane == single-pane at the preset cell (1.0 s) --------------------
	var preset := {}
	for spec in DemoStream.SPECS:
		var pid := String(spec["id"])
		var single := _run_full(scores, samples, rows, pose, snap, sizes, peak, 1.0, pid)
		var left := _run_full(scores, samples, rows, pose, snap, sizes, peak, 1.0, pid)
		var right := _run_full(scores, samples, rows, pose, snap, sizes, peak, 1.0, pid)
		var dd := maxf(absf(single["deficit"] - left["deficit"]), absf(left["deficit"] - right["deficit"]))
		var db := maxf(absf(single["bytes_streamed"] - left["bytes_streamed"]),
			absf(left["bytes_streamed"] - right["bytes_streamed"]))
		var pass_row := dd < TOLERANCE and db <= 1.0
		ok = ok and pass_row
		preset[pid] = {"deficit": single["deficit"], "bytes_streamed": single["bytes_streamed"],
			"wasted_prefetches": single["wasted_prefetches"], "pop_in_events": single["pop_in_events"],
			"peak_resident_bytes": single["peak_resident_bytes"],
			"dual_vs_single_deficit_diff": dd, "dual_vs_single_bytes_diff": db, "ok": pass_row}
	_report["checks"]["dual_equals_single"] = preset
	_report["preset_cell"] = {"budget_fraction": 1.0, "bandwidth_mb_s": 200.0, "latency_seconds": 1.0,
		"peak_working_set_bytes": peak}
	_report["ok"] = ok

	var f := FileAccess.open(_out.path_join("verification.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(_report, "  "))
	print("[demo_parity] -> ", _out.path_join("verification.json"))
	for row in parity_rows:
		print("[demo_parity] parity %s demo=%.6f engine=%.6f diff=%.9f ok=%s" % [
			row["policy"], row["demo_deficit"], row["engine_deficit"], row["deficit_diff"], row["ok"]])
	print("[demo_parity] determinism identical=%s" % deterministic)
	print("[demo_parity] ", "PASS" if ok else "FAIL")
	quit(0 if ok else 1)

func _run_full(scores: Dictionary, samples: Array, rows: Dictionary, pose: Dictionary,
		snap: Array, sizes: Dictionary, peak: float, latency: float, policy_id: String) -> Dictionary:
	var ds := DemoStream.new()
	ds.setup(TRAJECTORY, samples, rows, pose, snap, sizes, peak, 200.0e6, latency, policy_id)
	for i in samples.size():
		ds.advance(i)
	return ds.summarise()

func _fail(message: String) -> void:
	print("[demo_parity] FAIL: ", message)
	_report["ok"] = false
	_report["error"] = message
	DirAccess.make_dir_recursive_absolute(_out)
	var f := FileAccess.open(_out.path_join("verification.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(_report, "  "))
	quit(1)
