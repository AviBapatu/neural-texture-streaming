# exported_model/

Reserved for the exported neural network that the Godot runtime will consume.

The training phase (see `../training/`) produces the weights; they land here in a form Godot
can load, for example ONNX loaded through an `ONNXRuntime`-style binding or a TorchScript/ONNX
model converted to a plain weight format that a small GDScript or GDExtension runner can read.

Nothing here yet - no model has been trained or exported.
