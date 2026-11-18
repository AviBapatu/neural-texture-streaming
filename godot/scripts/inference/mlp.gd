class_name VisibilityMlp
extends RefCounted
## Godot inference for the Phase 3b visibility MLP (variant b, seed 1).
##
## Plain dense stack: normalise, then Linear -> ReLU -> Linear -> ReLU -> Linear.
## The feature order and output names come from the exported JSON - nothing about
## the model is hard-coded, so the file is the single source of truth.
##
## Feature order and output order are read from `feature_names` / `output_names`;
## the 66th feature is `extra_0`, the frozen frustum-tuned heuristic score.

var feature_names := PackedStringArray()
var output_names := PackedStringArray()
var activation := "relu"
var mean := PackedFloat64Array()
var std := PackedFloat64Array()
var weights: Array = []     # Array[PackedFloat64Array], row-major (out x in)
var biases: Array = []      # Array[PackedFloat64Array]
var loaded := false


static func load_from(path_relative: String) -> VisibilityMlp:
	var model := VisibilityMlp.new()
	var text := FileAccess.get_file_as_string(path_relative)
	if text.is_empty():
		push_error("[mlp] cannot read " + path_relative)
		return model
	var parsed: Variant = JSON.parse_string(text)
	if not (parsed is Dictionary):
		push_error("[mlp] invalid model JSON")
		return model
	var data: Dictionary = parsed
	model.activation = String(data.get("activation", "relu"))
	model.mean = PackedFloat64Array(data["normaliser"]["mean"])
	model.std = PackedFloat64Array(data["normaliser"]["std"])
	for n in data["feature_names"]:
		model.feature_names.append(String(n))
	for n in data["output_names"]:
		model.output_names.append(String(n))
	for layer in data["layers"]:
		var rows: Array = []
		for row in layer["weight"]:
			rows.append(PackedFloat64Array(row))
		model.weights.append(rows)
		model.biases.append(PackedFloat64Array(layer["bias"]))
	model.loaded = true
	return model


## Raw logits for one feature vector (in feature_names order).
func forward(features: PackedFloat64Array) -> PackedFloat64Array:
	var a := PackedFloat64Array()
	a.resize(features.size())
	for i in features.size():
		a[i] = (features[i] - mean[i]) / std[i]
	for layer_index in weights.size():
		var rows: Array = weights[layer_index]
		var bias: PackedFloat64Array = biases[layer_index]
		var out := PackedFloat64Array()
		out.resize(rows.size())
		for i in rows.size():
			var row: PackedFloat64Array = rows[i]
			var sum := float(bias[i])
			for j in row.size():
				sum += row[j] * a[j]
			out[i] = sum
		if layer_index < weights.size() - 1:
			if activation == "relu":
				for i in out.size():
					out[i] = maxf(out[i], 0.0)
			else:
				for i in out.size():
					out[i] = tanh(out[i])
		a = out
	return a


## Sigmoid probability for each output.
func probabilities(features: PackedFloat64Array) -> PackedFloat64Array:
	var logits := forward(features)
	var result := PackedFloat64Array()
	result.resize(logits.size())
	for i in logits.size():
		result[i] = 1.0 / (1.0 + exp(-logits[i]))
	return result
