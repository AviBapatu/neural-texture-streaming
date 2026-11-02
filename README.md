# Neural Predictive Texture Streaming

Semester project: **Neural Predictive Texture Streaming for Real-Time 3D Environments.**

The goal is a system that predicts which textures a player is about to look at, based on
camera movement and scene content, so the right texture resolution can be prefetched before
it is needed - reducing texture pop-in under a constrained memory budget.

## Layout

```
neural-texture-streaming/
├── godot/         Godot 4.7 project (the environment recorded for the dataset)
├── assets_src/    untouched upstream asset repository (provenance)
├── dataset/       telemetry, labels, splits, manifests and reports
├── tools/         Python utilities and pipeline scripts
├── training/      model training and evaluation code
├── sim/           offline streaming simulator
├── benchmarks/    occlusion overhead measurement
└── docs/          documentation
```

## Requirements

- Godot **4.7.2** stable (GDScript only, no C#)
- Python 3 with **Pillow** and **NumPy**

## Launch

```bash
godot --path godot        # play the environment
godot --path godot -e     # open the editor
```
