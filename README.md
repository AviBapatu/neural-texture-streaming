# Neural Predictive Texture Streaming

Semester project: **Neural Predictive Texture Streaming for Real-Time 3D Environments.**

The system predicts which textures a player is about to look at - from camera motion and
scene content - so a lower-resolution placeholder can be replaced by a full-texture request
before the object becomes visible, reducing texture pop-in under a constrained memory
budget. The work spans the whole stack: a playable Godot environment, a validated telemetry
and labelling pipeline, non-neural baselines, a small visibility MLP, a trace-driven
streaming simulator, and an in-engine streaming runtime with a live side-by-side demo.

## Headline result

Predictive prefetching helps within a **bounded latency band**, not universally. In a
268-object simulator, occlusion culling captures **88.9%** of the reactive-to-oracle gap in
the memory-tight regime; once latency becomes the binding constraint a thresholded
predictor (`mlp_b`, tau = 0.5) reaches a deficit of **0.1361 at 1.6x** the bytes of reactive
culling and helps in **12 of 12** prediction-relevant cells. The learned model matches a
strong geometric heuristic at full recall and degrades more gracefully once selection is
required. The full argument, numbers and caveats are in
[docs/final-report.md](docs/final-report.md).

## Layout

```
neural-texture-streaming/
├── godot/                  Godot 4.7 project (launch this)
│   ├── project.godot
│   ├── scenes/             main.tscn, environment.tscn, player.tscn, demo_streaming.tscn
│   ├── scripts/            camera_controller.gd, environment_manager.gd, visid.gd,
│   │                       visibility_service.gd, telemetry_recorder.gd,
│   │                       trajectory_runner.gd, screenshot_capture.gd
│   │   ├── inference/      GDScript MLP + heuristic ports, streaming math
│   │   ├── streaming/      texture_streamer.gd, transport_model.gd, streaming_controller.gd
│   │   └── demo/           dual-pane demo driver
│   ├── tools/              export_object_index.gd, run_trajectories.gd,
│   │                       run_streaming.gd, benchmark_visibility.gd
│   ├── assets/             models/, textures/, materials/, hdri/
│   └── tests/              headless tests + run_tests.sh
├── tools/                  Python utilities and pipeline scripts
├── training/               feature pipeline, baselines, visibility MLP, evaluation
├── sim/                    offline streaming simulator and sweeps
├── benchmarks/             occlusion overhead measurement
├── dataset/                manifests/, trajectories/ (raw and processed runs are generated)
└── docs/                   documentation + validation screenshots
```

Generated run outputs (`dataset/raw`, `dataset/processed`, `dataset/telemetry`,
`dataset/streaming`, `dataset/textures`, `dataset/reports`) and the Godot import cache
(`godot/.godot`) are not tracked; they are reproduced by the pipeline below.

## Requirements

- Godot **4.7.2** stable (GDScript only, no C#)
- Python 3 with **NumPy** and (for the texture tooling) **Pillow**

## Launch

```bash
godot --path godot                      # play the environment
godot --path godot -e                   # open the editor
```

| Input | Action |
|---|---|
| `W` `A` `S` `D` | Move |
| Mouse | Look around |
| `Shift` | Sprint |
| `Space` | Jump |
| `R` | Reset camera to the starting position |
| `F1` | Start / pause telemetry recording |
| `Esc` | Release the mouse cursor (click to re-capture) |

## Generate the dataset

```bash
tools/run_dataset_pipeline.sh --profile full --run-id run_full_p25
```

Runs all eight stages: import, object index, texture index, trajectories, splits, raw
capture, labelling and validation. `--run-id` keeps earlier runs intact. See
[docs/dataset-generation.md](docs/dataset-generation.md).

## Train and evaluate

```bash
python3 training/run_baselines.py     # non-neural baselines and evaluation harness
python3 training/train.py             # visibility MLP
python3 sim/run_sweep.py --config sim/config_4bplus.yaml   # streaming simulator sweep
```

## Validate

```bash
godot/tests/run_tests.sh                                     # headless Phase 1 + Phase 2 checks
python3 tools/validate_dataset.py --run run_full_p25         # dual-target label set
```

See [docs/validation.md](docs/validation.md) for what each check covers.

## Live demo

```bash
./run_demo.sh
```

Opens the dual-pane streaming demo (held-out trajectory `t06_look_around`, preset cell
budget 100% / 200 MB/s / 1.0 s). See [docs/final-report.md](docs/final-report.md) for the
captured results and the demo keys.

## Documentation

| Document | Contents |
|---|---|
| [docs/environment.md](docs/environment.md) | the generated city and how to change it |
| [docs/telemetry.md](docs/telemetry.md) | Phase 1 telemetry fields |
| [docs/telemetry-schema.md](docs/telemetry-schema.md) | full schema v2, visibility signals, occlusion cost |
| [docs/future-demand-labels.md](docs/future-demand-labels.md) | the prediction targets, resolution estimate, eligibility masks |
| [docs/dataset-generation.md](docs/dataset-generation.md) | pipeline stages, identity scheme, splits, knobs |
| [docs/texture-variants.md](docs/texture-variants.md) | downscaling tool and manifest |
| [docs/validation.md](docs/validation.md) | every executed check |
| [docs/experimental-split-review.md](docs/experimental-split-review.md) | split v1 design and rationale |
| [docs/known-issues.md](docs/known-issues.md) | open defects with reproduction |
| [docs/limitations.md](docs/limitations.md) | known gaps and scope boundaries |
| [docs/final-report.md](docs/final-report.md) | the full project report and results |
| [docs/asset-sources.md](docs/asset-sources.md) | asset provenance and licensing |

## Asset provenance

All assets come from the [FPS Buildings Environment Kit](https://github.com/petroulacl/fps-buildings-env-kit)
(commit `1dd8e932402deb82422920df2e7fd079391e55d3`), which collects CC0 / public-domain work
from Kenney, Quaternius, Poly Haven and ambientCG.
