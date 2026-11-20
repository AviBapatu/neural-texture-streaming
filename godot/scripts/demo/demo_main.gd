extends Control
## Phase 5 presenter demo driver.
##
## Two panes, each its own World3D + environment + streamer, replaying one held-out
## trajectory at the same tick. Launch through `./run_demo.sh`.

const ENGINE_DIR := "dataset/streaming/runs/phase5_engine"
const SPLIT_MANIFEST := "dataset/manifests/experimental_split_v1.json"
const DT := 0.1

const PRESETS := [
	{"left": "reactive_visible", "right": "mlp_b_tau0",
		"caption": "Preset 1 - reactive_visible vs mlp_b tau=0: prefetching cuts the mean deficit by 0.0469 at this cell (Phase 5 report, section 5), but at tau=0 it pays 5.80x the bytes (Phase 5sel summary)."},
	{"left": "heuristic_occ_tau0.5", "right": "mlp_b_tau0.5",
		"caption": "Preset 2 - heuristic_occ tau=0.5 vs mlp_b tau=0.5: the threshold cuts the 5.80x overhead to 1.20x and 1.60x (Phase 5sel summary) while the deficit barely moves - prediction degrades gracefully."},
	{"left": "heuristic_occ_tau0", "right": "mlp_b_tau0",
		"caption": "Preset 3 - heuristic_occ vs mlp_b at tau=0: the tuned heuristic and the MLP land within 0.0024 deficit and 0.03x bytes of each other (Phase 5sel summary) - the ranking is at parity."},
]

const POLICY_CYCLE := ["reactive_visible", "heuristic_occ_tau0", "heuristic_occ_tau0.5",
	"mlp_b_tau0", "mlp_b_tau0.5", "no_streaming"]

var _mode := "dual"
var _trajectory := "t06_look_around"
var _policy_left := "reactive_visible"
var _policy_right := "mlp_b_tau0.5"
var _budget_fraction := 1.0
var _latency := 1.0
var _bandwidth_mb_s := 200.0
var _peak := 1.0
var _quality_low := true
var _pane_scale := 1
var _fullscreen := true
var _slowmo := true
var _no_vsync := false

var _start_sample := 56
var _end_sample := 87
var _start_overridden := false
var _end_overridden := false
var _replay_lowres := 3
var _replay_mlp := 0

var _panes: Array[DemoPane] = []
var _root_box: VBoxContainer
var _pane_box: GridContainer
var _gpu_label: Label
var _footer: Label
var _caption: Label
var _summary: PanelContainer
var _summary_grid: GridContainer
var _help: PanelContainer
var _help_visible := false

var _samples: Array = []
var _rows_by_sample: Dictionary = {}
var _pose: Dictionary = {}
var _snap: Array = []
var _sizes: Dictionary = {}
var _manifest_objects: Array = []

var _elapsed := 0.0
var _next_sample := 60
var _paused := false
var _playing := false
var _worlds_ready := false

var _tour := false
var _tour_index := 0
var _tour_timer := 0.0
const TOUR_SECONDS := 30.0

var _exit_after_frames := 0
var _frame_count := 0
var _bench_start_usec := 0
var _record_dir := ""
var _record_every := 1
var _record_count := 0

var _shot_dir := ""
var _shot_list: Array = []
var _shot_index := 0
var _shot_sample := 71
var _shot_wait := 0
var _shot_stage := 0

func _ready() -> void:
	_parse_args()
	var win := get_window()
	# No letterbox: lay the UI out at the real window size, and follow resizes.
	win.content_scale_mode = Window.CONTENT_SCALE_MODE_DISABLED
	if _fullscreen:
		win.mode = Window.MODE_FULLSCREEN
	if _no_vsync:
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	win.size_changed.connect(_on_window_resized)
	if not _load_data():
		push_error("[demo] failed to load demo data")
		return
	_verify_held_out()
	_build_ui()
	_build_panes()

func _on_window_resized() -> void:
	_apply_layout()

## Grid columns follow the window aspect: side by side when landscape, stacked when tall.
func _apply_layout() -> void:
	if _pane_box == null or _panes.is_empty():
		return
	var size := get_viewport_rect().size
	var portrait := size.y > size.x
	_pane_box.columns = 1 if (portrait and _panes.size() > 1) else _panes.size()
	print("[demo] layout: viewport=%s portrait=%s columns=%d" % [
		str(size), str(portrait), _pane_box.columns])
	var ui_scale := clampf(size.y / 900.0, 0.75, 1.8)
	for pane in _panes:
		pane.set_ui_scale(ui_scale)
		pane.sync_viewport_size()

func _load_data() -> bool:
	var manifest: Variant = _read_json(ENGINE_DIR.path_join("manifest.json"))
	var scores: Variant = _read_json(ENGINE_DIR.path_join("scores").path_join(_trajectory + ".json"))
	if not (manifest is Dictionary) or not (scores is Dictionary):
		return false
	_manifest_objects = manifest.get("objects", [])
	_samples = scores.get("samples", [])
	_rows_by_sample = scores.get("rows", {})
	_pose = scores.get("pose", {})
	_snap = scores.get("snap_samples", [])
	for o in _manifest_objects:
		_sizes[String(o["object_id"])] = float(o["full_bytes"])
	var peaks: Dictionary = manifest.get("peak_working_set_bytes", {})
	_peak = float(peaks.get(_trajectory, 134217728.0))
	_apply_segment()
	if _end_sample <= 0 or _end_sample > _samples.size():
		_end_sample = mini(_samples.size(), _start_sample + 90)
	return true

## Pull the scan's chosen replay window and caption numbers, if it has been run.
func _apply_segment() -> void:
	var seg: Variant = _read_json("dataset/streaming/runs/demo/demo_segment.json")
	if not (seg is Dictionary):
		return
	var event: Dictionary = (seg as Dictionary).get("stage_event", {})
	if String(event.get("trajectory", "")) != _trajectory:
		return
	_replay_lowres = int(event.get("reactive_count", 0))
	_replay_mlp = int(event.get("mlp_count", 0))
	if not _start_overridden:
		_start_sample = int(seg.get("replay_start_sample", _start_sample))
	if not _end_overridden:
		_end_sample = int(seg.get("replay_end_sample", _end_sample)) + 1

func _verify_held_out() -> void:
	var split: Variant = _read_json(SPLIT_MANIFEST)
	if not (split is Dictionary):
		print("[demo] VERIFY: split manifest missing")
		return
	var train: Array = []
	for entry in split.get("splits", {}).get("train", []):
		train.append(String(entry["trajectory_id"]))
	var held := not train.has(_trajectory)
	print("[demo] VERIFY: %s %s the MLP training set (7 train trajectories)" % [
		_trajectory, "is NOT in" if held else "IS IN (LEAK!)"])

func _parse_args() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--panes="):
			_mode = "single" if a.trim_prefix("--panes=") == "1" else "dual"
		elif a.begins_with("--trajectory="):
			_trajectory = a.trim_prefix("--trajectory=")
		elif a.begins_with("--policy-left="):
			_policy_left = a.trim_prefix("--policy-left=")
		elif a.begins_with("--policy-right="):
			_policy_right = a.trim_prefix("--policy-right=")
		elif a.begins_with("--budget="):
			_budget_fraction = float(a.trim_prefix("--budget="))
		elif a.begins_with("--latency="):
			_latency = float(a.trim_prefix("--latency="))
		elif a.begins_with("--bandwidth="):
			_bandwidth_mb_s = float(a.trim_prefix("--bandwidth="))
		elif a.begins_with("--start-sample="):
			_start_sample = int(a.trim_prefix("--start-sample="))
			_start_overridden = true
		elif a.begins_with("--end-sample="):
			_end_sample = int(a.trim_prefix("--end-sample="))
			_end_overridden = true
		elif a.begins_with("--quality="):
			_quality_low = a.trim_prefix("--quality=") != "high"
		elif a.begins_with("--pane-scale="):
			_pane_scale = maxi(1, int(a.trim_prefix("--pane-scale=")))
		elif a.begins_with("--exit-after-frames="):
			_exit_after_frames = int(a.trim_prefix("--exit-after-frames="))
		elif a.begins_with("--record="):
			_record_dir = a.trim_prefix("--record=")
		elif a.begins_with("--record-every="):
			_record_every = maxi(1, int(a.trim_prefix("--record-every=")))
		elif a == "--tour":
			_tour = true
		elif a == "--fullscreen":
			_fullscreen = true
		elif a == "--windowed":
			_fullscreen = false
		elif a == "--no-slowmo":
			_slowmo = false
		elif a == "--no-vsync":
			_no_vsync = true
		elif a.begins_with("--shot-dir="):
			_shot_dir = a.trim_prefix("--shot-dir=")
		elif a.begins_with("--shot-sample="):
			_shot_sample = int(a.trim_prefix("--shot-sample="))

# --- UI ---------------------------------------------------------------------

func _build_ui() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_root_box = VBoxContainer.new()
	_root_box.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_root_box.add_theme_constant_override("separation", 0)
	_root_box.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_root_box)

	# Top bar
	var top := PanelContainer.new()
	top.custom_minimum_size = Vector2(0, 30)
	top.add_theme_stylebox_override("panel", _bar_style())
	_root_box.add_child(top)
	var top_box := HBoxContainer.new()
	top.add_child(top_box)
	var title := Label.new()
	title.text = "  Neural Predictive Texture Streaming - Phase 5"
	title.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_style_label(title, 15, Color(1, 1, 1))
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top_box.add_child(title)
	_gpu_label = Label.new()
	_gpu_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_style_label(_gpu_label, 12, Color(0.7, 0.85, 1.0))
	top_box.add_child(_gpu_label)

	# Panes
	_pane_box = GridContainer.new()
	_pane_box.columns = 2
	_pane_box.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_pane_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_pane_box.add_theme_constant_override("separation", 2)
	_pane_box.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_root_box.add_child(_pane_box)

	# Footer
	var foot := PanelContainer.new()
	foot.custom_minimum_size = Vector2(0, 26)
	foot.add_theme_stylebox_override("panel", _bar_style())
	_root_box.add_child(foot)
	_footer = Label.new()
	_footer.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_footer.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_style_label(_footer, 13, Color(0.9, 0.95, 1.0))
	foot.add_child(_footer)

	var overlay := CanvasLayer.new()
	add_child(overlay)

	_caption = Label.new()
	_caption.set_anchors_and_offsets_preset(Control.PRESET_CENTER_BOTTOM)
	_caption.offset_left = -640
	_caption.offset_right = 640
	_caption.offset_top = -150
	_caption.offset_bottom = -100
	_caption.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_caption.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_style_label(_caption, 15, Color(1, 0.95, 0.6))
	_caption.visible = false
	overlay.add_child(_caption)

	_build_summary(overlay)
	_build_help(overlay)

func _bar_style() -> StyleBoxFlat:
	var s := StyleBoxFlat.new()
	s.bg_color = Color(0.04, 0.06, 0.09, 0.9)
	s.border_color = Color(0.3, 0.5, 0.8, 0.5)
	s.border_width_bottom = 1
	s.border_width_top = 1
	s.set_content_margin_all(4)
	return s

func _build_summary(overlay: CanvasLayer) -> void:
	_summary = PanelContainer.new()
	_summary.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	_summary.offset_left = -430
	_summary.offset_right = 430
	_summary.offset_top = -190
	_summary.offset_bottom = 190
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.03, 0.05, 0.08, 0.95)
	style.border_color = Color(0.4, 0.7, 1.0)
	style.set_border_width_all(2)
	style.set_corner_radius_all(8)
	style.set_content_margin_all(16)
	_summary.add_theme_stylebox_override("panel", style)
	_summary.visible = false
	overlay.add_child(_summary)
	_summary_grid = GridContainer.new()
	_summary_grid.columns = 3
	_summary_grid.add_theme_constant_override("h_separation", 28)
	_summary_grid.add_theme_constant_override("v_separation", 6)
	_summary.add_child(_summary_grid)

func _build_help(overlay: CanvasLayer) -> void:
	_help = PanelContainer.new()
	_help.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	_help.offset_left = -360
	_help.offset_right = 360
	_help.offset_top = -210
	_help.offset_bottom = 210
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.02, 0.04, 0.07, 0.96)
	style.border_color = Color(0.5, 0.75, 1.0)
	style.set_border_width_all(2)
	style.set_corner_radius_all(10)
	style.set_content_margin_all(20)
	_help.add_theme_stylebox_override("panel", style)
	_help.visible = false
	overlay.add_child(_help)
	var label := Label.new()
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_style_label(label, 14, Color(0.92, 0.96, 1.0))
	label.text = "\n".join([
		"KEYS  (H to hide)", "",
		"SPACE   pause / resume the replay",
		"R       restart the replay",
		"H       toggle this help",
		"F1      tint overlay (low-res / loading / resident / wasted)",
		"X       swap-outline flash on/off",
		"[ ]     latency -/+ 0.1 s",
		"-  =    budget -/+ 50%",
		"T       summary table (both policies)",
		"1/2/3   presets 1, 2, 3",
		"A       auto-tour (~90 s, captions)",
		"Q / W   cycle LEFT pane policy",
		"O / P   cycle RIGHT pane policy",
		"Esc     quit",
		"", "Trajectory: %s (held out)  preset cell 100%% / 200 MB/s / 1.0 s" % _trajectory,
	])
	_help.add_child(label)

func _build_panes() -> void:
	var count := 1 if _mode == "single" else 2
	var pane_px := Vector2(800.0 if count > 1 else 1600.0, 840.0)
	var args := {
		"trajectory": _trajectory, "samples": _samples, "rows": _rows_by_sample,
		"pose": _pose, "snap": _snap, "sizes": _sizes,
		"budget": _peak * _budget_fraction, "bandwidth": _bandwidth_mb_s * 1e6,
		"latency": _latency,
	}
	for i in count:
		var pane := DemoPane.new()
		pane.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		pane.size_flags_vertical = Control.SIZE_EXPAND_FILL
		pane.custom_minimum_size = Vector2(1, 1)
		_pane_box.add_child(pane)
		var pane_args := args.duplicate()
		pane_args["policy"] = _policy_left if i == 0 else _policy_right
		pane.setup(pane_px, "L" if i == 0 else "R", _manifest_objects, pane_args, _pane_scale, _quality_low)
		_panes.append(pane)

func _on_worlds_ready() -> void:
	_worlds_ready = true
	_apply_layout()
	var size := get_viewport_rect().size
	print("[demo] window=%dx%d fullscreen=%s" % [int(size.x), int(size.y), str(_fullscreen)])
	_gpu_label.text = "%s  ·  %s  " % [
		RenderingServer.get_video_adapter_name(), RenderingServer.get_video_adapter_vendor()]
	print("[demo] GPU: %s (%s)" % [
		RenderingServer.get_video_adapter_name(), RenderingServer.get_video_adapter_vendor()])
	print("[demo] mode=%s trajectory=%s (held out) cell=%.0f%% %.0fMB/s %.1fs segment=%.1f-%.1fs quality=%s" % [
		_mode, _trajectory, _budget_fraction * 100.0, _bandwidth_mb_s, _latency,
		_start_sample * DT, _end_sample * DT, "low" if _quality_low else "high"])
	_restart()
	_caption.text = "%s  %.1f-%.1f s @ 0.25x   ·   reactive: %d low-res visible; mlp_b tau0.5: %d" % [
		_trajectory, _start_sample * DT, (_end_sample - 1) * DT, _replay_lowres, _replay_mlp]
	_caption.visible = true
	if _tour:
		_tour_index = 0
		_tour_timer = 0.0
		_apply_preset(0)
		_caption.visible = true
	if not _record_dir.is_empty():
		DirAccess.make_dir_recursive_absolute(_record_dir)
	if not _shot_dir.is_empty():
		_begin_shots()

func _apply_preset(index: int) -> void:
	if index < 0:
		return
	var preset: Dictionary = PRESETS[index % PRESETS.size()]
	_policy_left = String(preset["left"])
	_policy_right = String(preset["right"])
	_set_pane_policy(0, _policy_left)
	_set_pane_policy(1, _policy_right)
	_caption.text = String(preset["caption"])
	if _tour:
		_caption.visible = true
	print("[demo] preset %d: left=%s right=%s" % [index + 1, _policy_left, _policy_right])

func _set_pane_policy(index: int, policy: String) -> void:
	if index < _panes.size():
		_panes[index].set_policy(policy)

# --- playback ---------------------------------------------------------------

func _restart() -> void:
	_seek_to(_start_sample)
	_paused = false
	_playing = true

## Reset every pane and replay from sample 0 up to `sample`, so the streamer's
## residency/warm-up matches a continuous run of the whole trajectory (starting mid-way
## would show spurious pop-in), then park the camera there.
func _seek_to(sample: int) -> void:
	for pane in _panes:
		pane.reset_stream()
	for i in range(0, sample):
		for pane in _panes:
			pane.advance(i)
	_elapsed = float(sample) * DT
	_next_sample = sample
	for pane in _panes:
		pane.set_time(_elapsed)

func _advance_to(target: int) -> void:
	while _next_sample <= target and _next_sample < _end_sample:
		for pane in _panes:
			pane.advance(_next_sample)
		_next_sample += 1
	if _next_sample >= _end_sample:
		if not _shot_dir.is_empty():
			_playing = false
		else:
			_restart()

func _process(delta: float) -> void:
	if not _worlds_ready:
		if _panes.size() > 0 and _panes[0].world_ready:
			_on_worlds_ready()
		_frame_count += 1
		return
	if not _paused and _playing:
		_elapsed += delta * (0.25 if _slowmo else 1.0)
		_advance_to(int(floor(_elapsed / DT)))
	for pane in _panes:
		pane.set_time(_elapsed)
	if _tour:
		_tick_tour(delta)
	_update_footer()
	if not _capture_step():
		_frame_count += 1
		if not _record_dir.is_empty() and _frame_count % _record_every == 0:
			var image := get_viewport().get_texture().get_image()
			if image != null:
				image.save_png(_record_dir.path_join("frame_%05d.png" % _record_count))
				_record_count += 1
		if _frame_count == 30:
			_bench_start_usec = Time.get_ticks_usec()
		if _exit_after_frames > 0 and _frame_count >= _exit_after_frames:
			_finish_report()
			get_tree().quit()

func _tick_tour(delta: float) -> void:
	_tour_timer += delta
	if _tour_timer >= TOUR_SECONDS:
		_tour_timer = 0.0
		_tour_index = (_tour_index + 1) % PRESETS.size()
		_apply_preset(_tour_index)

func _update_footer() -> void:
	_footer.text = "%.0f FPS  ·  %s  ·  %s  ·  %.1f-%.1f s%s  ·  %s" % [
		Engine.get_frames_per_second(), _mode, _trajectory,
		_start_sample * DT, _end_sample * DT, "  (0.25x)" if _slowmo else "",
		"PAUSED" if _paused else "playing"]

func _finish_report() -> void:
	var fps := Engine.get_frames_per_second()
	if _bench_start_usec > 0 and _frame_count > 30:
		var seconds := float(Time.get_ticks_usec() - _bench_start_usec) / 1e6
		if seconds > 0.0:
			fps = float(_frame_count - 30) / seconds
	var frame_ms := (1000.0 / fps) if fps > 0.0 else 0.0
	print("[demo] FINAL  gpu=%s  average_fps=%.1f  frame_ms=%.2f  frames=%d  quality=%s" % [
		RenderingServer.get_video_adapter_name(), fps, frame_ms, _frame_count,
		"low" if _quality_low else "high"])
	for pane in _panes:
		var m := pane.stream.summarise()
		print("[demo]   %s %s deficit=%.6f bytes=%.3f MB wasted=%d loads=%d popin=%d resident=%.1f/%.1f MB" % [
			pane.side, pane.stream.policy_id, m["deficit"], m["bytes_streamed"] / 1e6,
			m["wasted_prefetches"], m["loads_completed"], m["pop_in_events"],
			m["peak_resident_bytes"] / 1e6, m["budget_bytes"] / 1e6])

func panes() -> Array:
	return _panes

# --- capture ----------------------------------------------------------------

func _begin_shots() -> void:
	DirAccess.make_dir_recursive_absolute(_shot_dir)
	_shot_list = [0, 1, 2]
	_shot_index = 0
	_shot_stage = 0
	_shot_wait = 0
	_paused = false
	_playing = false

## Returns true while a capture sequence is in progress (playback is frozen).
func _capture_step() -> bool:
	if _shot_dir.is_empty():
		return false
	if _shot_stage == 0:
		_apply_preset(_shot_list[_shot_index])
		_seek_to(_shot_sample)
		_paused = true
		_playing = false
		_shot_stage = 1
		_shot_wait = 0
		return true
	if _shot_stage == 1:
		_shot_wait += 1
		# Let the per-frame texture queue drain so resident geometry shows full detail.
		var pending := false
		for p in _panes:
			pending = pending or p.has_pending_textures()
		if _shot_wait >= 25 and not pending:
			var img := get_viewport().get_texture().get_image()
			if img != null:
				img.save_png(_shot_dir.path_join("preset_%d.png" % (_shot_list[_shot_index] + 1)))
			print("[demo] shot preset %d -> %s" % [
				_shot_list[_shot_index] + 1,
				_shot_dir.path_join("preset_%d.png" % (_shot_list[_shot_index] + 1))])
			_shot_index += 1
			if _shot_index >= _shot_list.size():
				print("[demo] capture done")
				get_tree().quit()
				return true
			_shot_stage = 0
		return true
	return true

# --- summary ----------------------------------------------------------------

func _toggle_summary() -> void:
	_summary.visible = not _summary.visible
	if _summary.visible:
		_refresh_summary()

func _refresh_summary() -> void:
	for child in _summary_grid.get_children():
		child.queue_free()
	var left := _panes[0].stream.summarise()
	var right := (_panes[1].stream.summarise() if _panes.size() > 1 else left)
	_add_cell("METRIC", true)
	_add_cell("LEFT: " + _panes[0].stream.spec(_policy_left)["label"], true)
	_add_cell("RIGHT: " + (_panes[1].stream.spec(_policy_right)["label"] if _panes.size() > 1 else "-"), true)
	_add_row("deficit", "%.4f" % left["deficit"], "%.4f" % right["deficit"])
	_add_row("pop-in events", "%d" % left["pop_in_events"], "%d" % right["pop_in_events"])
	_add_row("MB streamed", "%.1f" % (left["bytes_streamed"] / 1e6), "%.1f" % (right["bytes_streamed"] / 1e6))
	_add_row("x reactive bytes", "%.2f" % _bytes_ratio(left["bytes_streamed"]), "%.2f" % _bytes_ratio(right["bytes_streamed"]))
	_add_row("wasted loads", "%d" % left["wasted_prefetches"], "%d" % right["wasted_prefetches"])
	_add_row("resident MB", "%.0f / %.0f" % [left["peak_resident_bytes"] / 1e6, left["budget_bytes"] / 1e6],
		"%.0f / %.0f" % [right["peak_resident_bytes"] / 1e6, right["budget_bytes"] / 1e6])
	_add_row("video samples", "%d / %d" % [min(_next_sample, _end_sample), _end_sample],
		"%d / %d" % [min(_next_sample, _end_sample), _end_sample])

func _bytes_ratio(bytes: float) -> float:
	if _reactive_bytes < 0.0:
		var ref := DemoStream.new()
		ref.setup(_trajectory, _samples, _rows_by_sample, _pose, _snap, _sizes,
			_peak * _budget_fraction, _bandwidth_mb_s * 1e6, _latency, "reactive_visible")
		for i in _samples.size():
			ref.advance(i)
		_reactive_bytes = ref.bytes_streamed()
	if _reactive_bytes <= 0.0:
		return 0.0
	return bytes / _reactive_bytes

var _reactive_bytes := -1.0

func _add_cell(text: String, header := false) -> void:
	var label := Label.new()
	label.text = text
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_style_label(label, 14 if header else 13,
		Color(1, 1, 1) if header else Color(0.85, 0.9, 1.0))
	_summary_grid.add_child(label)

func _add_row(name: String, a: String, b: String) -> void:
	_add_cell(name)
	_add_cell(a)
	_add_cell(b)

func _style_label(label: Label, size: int, color: Color) -> void:
	label.add_theme_font_size_override("font_size", size)
	label.add_theme_color_override("font_color", color)
	label.add_theme_color_override("font_outline_color", Color(0, 0, 0))
	label.add_theme_constant_override("outline_size", 5)
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE

# --- input ------------------------------------------------------------------

func _cycle(pane_index: int, step: int) -> void:
	if pane_index >= _panes.size():
		return
	var current := _panes[pane_index].stream.policy_id
	var idx := POLICY_CYCLE.find(current)
	if idx < 0:
		idx = 0
	idx = (idx + step + POLICY_CYCLE.size()) % POLICY_CYCLE.size()
	var policy: String = POLICY_CYCLE[idx]
	if pane_index == 0:
		_policy_left = policy
	else:
		_policy_right = policy
	_panes[pane_index].set_policy(policy)
	_reactive_bytes = -1.0

func _input(event: InputEvent) -> void:
	if not (event is InputEventKey) or not event.pressed or event.echo:
		return
	match (event as InputEventKey).keycode:
		KEY_SPACE:
			_paused = not _paused
		KEY_R:
			_restart()
		KEY_H:
			_help_visible = not _help_visible
			_help.visible = _help_visible
		KEY_F1:
			for pane in _panes:
				pane.tint_enabled = not pane.tint_enabled
				pane._apply_tints()
		KEY_X:
			for pane in _panes:
				pane.outline_enabled = not pane.outline_enabled
		KEY_S:
			_slowmo = not _slowmo
		KEY_BRACKETLEFT:
			_set_latency(_latency - 0.1)
		KEY_BRACKETRIGHT:
			_set_latency(_latency + 0.1)
		KEY_MINUS:
			_set_budget(_budget_fraction - 0.5)
		KEY_EQUAL:
			_set_budget(_budget_fraction + 0.5)
		KEY_T:
			_toggle_summary()
		KEY_1:
			_tour = false
			_caption.visible = false
			_apply_preset(0)
		KEY_2:
			_tour = false
			_caption.visible = false
			_apply_preset(1)
		KEY_3:
			_tour = false
			_caption.visible = false
			_apply_preset(2)
		KEY_A:
			_tour = not _tour
			_tour_index = 0
			_tour_timer = 0.0
			_caption.visible = _tour
			if _tour:
				_apply_preset(0)
		KEY_Q:
			_cycle(0, -1)
		KEY_W:
			_cycle(0, 1)
		KEY_O:
			_cycle(1, -1)
		KEY_P:
			_cycle(1, 1)
		KEY_ESCAPE:
			get_tree().quit()

func _set_latency(value: float) -> void:
	_latency = clampf(value, 0.0, 2.0)
	for pane in _panes:
		if pane.stream != null:
			pane.stream.set_transport(_bandwidth_mb_s * 1e6, _latency)

func _set_budget(fraction: float) -> void:
	_budget_fraction = maxf(fraction, 0.5)
	for pane in _panes:
		if pane.stream != null:
			pane.stream.set_budget(_budget_fraction, _peak)
	_reactive_bytes = -1.0

# --- helpers ----------------------------------------------------------------

func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()

func _read_json(relative: String) -> Variant:
	var path := relative if relative.begins_with("/") else _repo_root().path_join(relative)
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		return null
	return JSON.parse_string(text)
