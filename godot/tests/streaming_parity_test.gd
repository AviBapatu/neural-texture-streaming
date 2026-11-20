extends SceneTree
## Phase 5 / Step 4 - engine-side parity run.
##
## Replays the exported per-(trajectory, sample, object) signals through the GDScript
## streaming core (TextureStreamer) for reactive_visible, heuristic_occ and mlp_b
## (scored_replay), and writes the engine's metrics for comparison against
## sim/simulate.py run on the same 53-object scope.
##
## Priorities:
##   reactive_visible - 2.0 when ground-truth visible, else 0 (VisibilityService logic)
##   heuristic_occ    - the ported frozen heuristic (HeuristicOcc), tau = 0
##   mlp_b            - scored_replay: the precomputed leave-trajectory-out MLP scores
##
## Also checks the heuristic port against the exported raw score for every row.
##
## Run:
##   godot --headless --path godot --script res://tests/streaming_parity_test.gd -- \
##       --out=dataset/streaming/runs/phase5_engine

var _base := ""
var _engine_dir := ""
var _watchdog := 0

func _initialize() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			_base = a.trim_prefix("--out=")
	_run.call_deferred()

func _process(_delta: float) -> bool:
	_watchdog += 1
	if _watchdog > 20000:
		push_error("[parity] watchdog timeout")
		quit(3)
		return true
	return false

func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()

func _read_json(path: String) -> Variant:
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		push_error("[parity] cannot read " + path)
		return null
	return JSON.parse_string(text)

func _run() -> void:
	if _base.is_empty():
		_base = "dataset/streaming/runs/phase5_engine"
	if not _base.begins_with("/"):
		_base = _repo_root().path_join(_base)
	_engine_dir = _base.path_join("engine")
	DirAccess.make_dir_recursive_absolute(_engine_dir)

	var cells: Dictionary = _read_json(_base.path_join("cells.json"))
	var manifest: Dictionary = _read_json(_base.path_join("manifest.json"))
	if cells == null or manifest == null:
		quit(2)
		return

	var sizes := {}
	var centres := {}
	for o in manifest["objects"]:
		sizes[o["object_id"]] = float(o["full_bytes"])
		centres[o["object_id"]] = o["centre"]

	var policies: Array = cells["policies"]
	var streamer := TextureStreamer.new()
	var all_metrics := {}
	var port_max := 0.0
	var port_sum := 0.0
	var port_count := 0

	for traj in manifest["trajectories"]:
		var data: Dictionary = _read_json(_base.path_join("scores").path_join(traj + ".json"))
		if data == null:
			continue
		var pose: Dictionary = data["pose"]
		var rows_by_sample_raw: Dictionary = data["rows"]

		var samples := PackedInt32Array()
		for s in data["samples"]:
			samples.append(int(s))

		var row_oid := PackedStringArray()
		var row_cov := PackedFloat64Array()
		var row_vis := PackedByteArray()
		var row_frustum := PackedByteArray()
		var row_occ := PackedInt32Array()
		var row_hraw := PackedFloat64Array()
		var row_mlp05 := PackedFloat64Array()
		var row_mlp10 := PackedFloat64Array()
		var rows_by_sample := {}
		for s in samples:
			var list: Array = rows_by_sample_raw.get(str(s), [])
			var idx := PackedInt32Array()
			for r in list:
				idx.append(row_oid.size())
				var oid := String(r[0])
				var frustum := int(r[2])
				var occ := int(r[3])
				row_oid.append(oid)
				row_cov.append(float(r[1]))
				row_frustum.append(frustum)
				row_occ.append(occ)
				row_vis.append(1 if (frustum == 1 and occ != 0) else 0)
				row_hraw.append(float(r[4]))
				row_mlp05.append(float(r[5]))
				row_mlp10.append(float(r[6]))
			rows_by_sample[s] = idx

		var excluded := {}
		var snap_window := int(cells["snap_window_samples"])
		for s in data["snap_samples"]:
			for offset in range(-snap_window, snap_window + 1):
				excluded[int(s) + offset] = true

		# --- priorities per policy ----------------------------------------------
		var budget := float(cells["budget_bytes"].get(traj, 1.0))
		var bw := float(cells["bandwidth_bytes_per_s"])
		var lat := float(cells["latency_seconds"])
		var dt := float(manifest["sample_interval_seconds"])
		all_metrics[traj] = {}
		for policy in policies:
			var priorities := PackedFloat64Array()
			priorities.resize(row_oid.size())
			for s in samples:
				var p: Dictionary = pose.get(str(s), {})
				var half := [0.0, 0.0]
				if not p.is_empty():
					half = HeuristicOcc.half_angles(float(p["fov"]),
						float(p["viewport"][0]), float(p["viewport"][1]))
				for r in rows_by_sample[s]:
					var visible_now := row_vis[r] == 1
					var prio := 0.0
					if policy == "reactive_visible":
						prio = 2.0 if visible_now else 0.0
					elif policy == "heuristic_occ":
						if visible_now:
							prio = 2.0
						else:
							var oid := row_oid[r]
							var centre: Array = centres.get(oid, [0.0, 0.0, 0.0])
							var position: Array = p["position"]
							var velocity: Array = p["velocity"]
							var angular: Array = p["angular_velocity"]
							var matrix := StreamMath.quaternion_to_matrix(p["quaternion"])
							var margin := HeuristicOcc.margin_deg(position, matrix, centre,
								velocity, angular, half[0], half[1])
							prio = HeuristicOcc.priority(false, margin, row_occ[r])
							# port accuracy
							var mine := HeuristicOcc.raw_score(margin, row_occ[r])
							var diff := absf(mine - row_hraw[r])
							port_max = maxf(port_max, diff)
							port_sum += diff
							port_count += 1
					elif policy == "mlp_b":
						prio = 2.0 if visible_now else maxf(row_mlp05[r], row_mlp10[r])
					priorities[r] = prio
			var metrics := streamer.simulate(samples, rows_by_sample, excluded, priorities,
				sizes, row_vis, row_cov, row_oid, bw, lat, budget, dt)
			all_metrics[traj][policy] = metrics

	var out := {
		"scope": "53 streamable objects (7 ambientCG identities)",
		"budget_fraction": cells["budget_fraction"],
		"latency_seconds": cells["latency_seconds"],
		"bandwidth_mb_s": cells["bandwidth_mb_s"],
		"policies": policies,
		"metrics": all_metrics,
		"heuristic_port": {
			"rows_checked": port_count,
			"max_abs_diff_deg": port_max,
			"mean_abs_diff_deg": (port_sum / float(port_count)) if port_count > 0 else 0.0,
		},
	}
	var f := FileAccess.open(_engine_dir.path_join("engine_metrics.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(out, "  "))
	print("[parity] engine metrics -> ", _engine_dir.path_join("engine_metrics.json"))
	print("[parity] heuristic port: rows=", port_count, " max_abs_diff_deg=", port_max,
		" mean_abs_diff_deg=", (port_sum / float(port_count)) if port_count > 0 else 0.0)
	quit(0)
