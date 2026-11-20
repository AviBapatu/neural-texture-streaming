extends SceneTree
const MODEL := "res://assets/models/visibility_mlp_b_seed1.json"
var _w := 0
func _initialize() -> void: _run.call_deferred()
func _process(_d: float) -> bool:
	_w += 1
	if _w > 600: quit(3); return true
	return false
func _run() -> void:
	var model := VisibilityMlp.load_from(MODEL)
	var n := 300
	var xs: Array = []
	var rng := RandomNumberGenerator.new(); rng.seed = 7
	for i in n:
		var v := PackedFloat64Array(); v.resize(66)
		for j in 66: v[j] = rng.randfn(0.0, 1.0)
		xs.append(v)
	# warm up
	for i in 5:
		model.forward(xs[i])
	var t0 := Time.get_ticks_usec()
	for r in 3:
		for v in xs: model.forward(v)
	var ms := float(Time.get_ticks_usec() - t0) / 1000.0 / 3.0
	print("[mlp_bench] %d candidates: %.3f ms/batch (%.4f ms/candidate)" % [n, ms, ms / float(n)])
	quit(0)
