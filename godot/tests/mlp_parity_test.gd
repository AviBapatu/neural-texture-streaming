extends SceneTree
## Phase 5 / Step 7 gate - the GDScript MLP vs training/models.py forward_numpy.
##
## Reads the exported model and probe (tools/export_model_for_godot.py), computes the
## forward pass in GDScript, and checks it against the reference logits in the probe.
## Exits non-zero if the max absolute difference exceeds the probe tolerance.

const MODEL := "res://assets/models/visibility_mlp_b_seed1.json"
const PROBE := "res://assets/models/visibility_mlp_b_seed1_probe.json"

var _watchdog := 0

func _initialize() -> void:
	_run.call_deferred()

func _process(_delta: float) -> bool:
	_watchdog += 1
	if _watchdog > 600:
		push_error("[mlp_parity] watchdog timeout")
		quit(3)
		return true
	return false

func _run() -> void:
	var model := VisibilityMlp.load_from(MODEL)
	if not model.loaded:
		quit(2)
		return
	var probe: Variant = JSON.parse_string(FileAccess.get_file_as_string(PROBE))
	if not (probe is Dictionary):
		push_error("[mlp_parity] cannot read probe")
		quit(2)
		return
	var tolerance := float(probe.get("tolerance", 1e-4))
	var features: Array = probe["features"]
	var expected: Array = probe["expected_logits"]
	var max_diff := 0.0
	for r in features.size():
		var logits := model.forward(PackedFloat64Array(features[r]))
		for j in logits.size():
			max_diff = maxf(max_diff, absf(logits[j] - float(expected[r][j])))
	print("[mlp_parity] inputs=%d outputs=%d hidden=%s activation=%s" % [
		model.feature_names.size(), model.output_names.size(), str(model.weights.size()), model.activation])
	print("[mlp_parity] feature[0]=%s  feature[65]=%s  outputs=%s" % [
		model.feature_names[0], model.feature_names[65], ", ".join(model.output_names)])
	print("[mlp_parity] rows=%d  max_abs_diff=%.8f  tolerance=%.8f  %s" % [
		features.size(), max_diff, tolerance, "PASS" if max_diff <= tolerance else "FAIL"])
	quit(0 if max_diff <= tolerance else 1)
