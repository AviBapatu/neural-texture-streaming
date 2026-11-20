extends SceneTree
## Phase 5 demo - 256-vs-full visibility capture.
##
## Freezes the demo at a preset sample, forces every object visible at that sample to
## the 256 variant, captures, then forces the same objects to the full (original 2048)
## texture, captures, and compares mean luminance-gradient sharpness. Writes both PNGs
## and a report so the "is the swap noticeable" claim is reproducible.
##
## Run:
##   godot --path godot --resolution 1280x720 \
##       --script res://tests/demo_capture.gd -- --out=dataset/streaming/runs/demo \
##       --sample=78 --panes=2

const SAMPLE_DEFAULT := 78

var _out := ""
var _sample := SAMPLE_DEFAULT
var _panes := 2

func _initialize() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--out="):
			_out = a.trim_prefix("--out=")
		elif a.begins_with("--sample="):
			_sample = int(a.trim_prefix("--sample="))
		elif a.begins_with("--panes="):
			_panes = int(a.trim_prefix("--panes="))
	_run.call_deferred()

func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()

func _run() -> void:
	if _out.is_empty():
		_out = "dataset/streaming/runs/demo"
	if not _out.begins_with("/"):
		_out = _repo_root().path_join(_out)
	DirAccess.make_dir_recursive_absolute(_out)

	var demo: Control = (load("res://scenes/demo_streaming.tscn") as PackedScene).instantiate()
	root.add_child(demo)
	DisplayServer.window_set_size(Vector2i(1280, 720))
	root.size = Vector2i(1280, 720)
	# Wait for the panes, environment and materials to be built.
	for i in 240:
		await process_frame
		if demo.has_method("panes") and demo.panes().size() > 0:
			break
	for i in 20:
		await process_frame
	print("[demo_capture] root size=", root.size, " demo size=", demo.size)

	demo.force_sample(_sample)
	for i in 8:
		await process_frame
	var visible: Array = demo.visible_oids_at(_sample)
	var panes: Array = demo.panes()

	var report := {
		"test": "demo_capture", "trajectory": "t06_look_around", "sample": _sample,
		"panes": panes.size(), "visible_objects": visible.size(),
		"visible_ids": visible, "shots": {},
	}

	var shot_256 := _out.path_join("preset_256.png")
	var applied_256 := 0
	for pane in panes:
		applied_256 += int(pane.force_resolution(256, visible))
	await _snap(shot_256)
	var img256 := root.get_texture().get_image()
	var s256 := _sharpness(img256)

	var shot_full := _out.path_join("preset_full.png")
	var applied_full := 0
	for pane in panes:
		applied_full += int(pane.force_resolution(0, visible))
	await _snap(shot_full)
	var imgfull := root.get_texture().get_image()
	var sfull := _sharpness(imgfull)

	report["applied_256"] = applied_256
	report["applied_full"] = applied_full

	report["shots"]["256"] = {"path": shot_256, "sharpness": s256}
	report["shots"]["full_2048"] = {"path": shot_full, "sharpness": sfull}
	report["ratio_full_over_256"] = (sfull / s256) if s256 > 0.0 else 0.0
	report["mean_abs_pixel_diff"] = _mean_abs_diff(img256, imgfull)
	report["noticeable"] = sfull > s256 * 1.05

	var f := FileAccess.open(_out.path_join("capture_report.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(report, "  "))
	print("[demo_capture] 256 sharpness=%.4f  full sharpness=%.4f  ratio=%.3f  mean_abs_pixel_diff=%.4f  noticeable=%s" % [
		s256, sfull, report["ratio_full_over_256"], report["mean_abs_pixel_diff"], report["noticeable"]])
	print("[demo_capture] -> ", _out.path_join("capture_report.json"))
	quit(0)

## Mean per-channel absolute pixel difference between two frames (0..1 scale).
func _mean_abs_diff(a: Image, b: Image) -> float:
	if a == null or b == null:
		return 0.0
	var w := mini(a.get_width(), b.get_width())
	var h := mini(a.get_height(), b.get_height())
	var total := 0.0
	var samples := 0
	var step := 2
	var y := 0
	while y < h:
		var x := 0
		while x < w:
			var ca := a.get_pixel(x, y)
			var cb := b.get_pixel(x, y)
			total += absf(ca.r - cb.r) + absf(ca.g - cb.g) + absf(ca.b - cb.b)
			samples += 1
			x += step
		y += step
	return total / float(maxi(1, samples))

func _snap(path: String) -> void:
	for i in 4:
		await process_frame
	await RenderingServer.frame_post_draw
	var image := root.get_texture().get_image()
	print("[demo_capture] _snap ", path, " image=", image.get_width(), "x", image.get_height())
	image.save_png(path)

## Mean luminance gradient magnitude (a standard sharpness proxy). Higher = sharper.
func _sharpness(image: Image) -> float:
	if image == null:
		return 0.0
	var w := image.get_width()
	var h := image.get_height()
	if w < 2 or h < 2:
		return 0.0
	var total := 0.0
	var samples := 0
	var step := 2
	var y := 0
	while y < h - step:
		var x := 0
		while x < w - step:
			var c := image.get_pixel(x, y)
			var cx := image.get_pixel(x + step, y)
			var cy := image.get_pixel(x, y + step)
			var l := 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
			var lx := 0.299 * cx.r + 0.587 * cx.g + 0.114 * cx.b
			var ly := 0.299 * cy.r + 0.587 * cy.g + 0.114 * cy.b
			total += absf(lx - l) + absf(ly - l)
			samples += 1
			x += step
		y += step
	return (total / float(maxi(1, samples))) * 100.0
