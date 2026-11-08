# Dataset generation

How the dataset is produced, how to reproduce it, and how the pieces fit together.

## Reproduce a complete run

```bash
# from the repository root
python3 -m pip install --user Pillow          # one-off, needed by the texture tooling

tools/run_dataset_pipeline.sh --profile full  # ~2-4 minutes, ~233 MB of output
```

That single command runs all eight stages. To reproduce the exact dataset reported in
[phase-2-report.md](phase-2-report.md), use no extra options:

```bash
tools/run_dataset_pipeline.sh --profile full --occlusion 1
```

Set `GODOT=/path/to/Godot_v4.7.2-stable_linux.x86_64` if `godot` is not on `PATH`.

## The eight stages

| # | Stage | Command | Output |
|---|---|---|---|
| 1 | import assets | `godot --headless --path godot --import` | `godot/.godot/imported/` |
| 2 | export object index | `godot ... --script res://tools/export_object_index.gd` | `dataset/manifests/object_index.json` |
| 3 | build texture index | `python3 tools/build_texture_index.py` | `dataset/manifests/texture_index.json` |
| 4 | generate trajectories | `python3 tools/generate_trajectories.py --profile full` | `dataset/trajectories/full/` |
| 5 | assign splits | `python3 tools/split_dataset.py --profile full` | `dataset/manifests/splits.json` |
| 6 | record raw telemetry | `godot ... --script res://tools/run_trajectories.gd` | `dataset/raw/<run>/` |
| 7 | label future demand | `python3 tools/label_future_demand.py --run <run>` | `dataset/processed/<run>/` |
| 8 | validate | `python3 tools/validate_dataset.py --run <run>` | `dataset/reports/<run>-validation.md` |

Each stage is independently runnable, so a failure can be isolated.

## Output layout

```
dataset/
├── raw/<run>/                          original telemetry, nothing derived
│   ├── <trajectory_id>.jsonl           one record per sample
│   ├── <trajectory_id>.meta.json       trajectory metadata, seeds, occlusion cost
│   └── run_summary.json                run totals
├── processed/<run>/                    labelled, model-ready
│   ├── <trajectory_id>.samples.jsonl   camera state, one row per sample
│   └── <trajectory_id>.labels.jsonl    object state + future labels
├── trajectories/<profile>/             generated trajectory definitions + index.json
├── manifests/
│   ├── object_index.json               every object: id, category, transform, textures
│   ├── texture_index.json              every texture id: source path, resolutions, variants
│   ├── splits.json                     train / validation / test assignment
│   └── dataset_manifest.json           what the run was built with
└── reports/<run>-validation.md         human-readable validation report
```

**Raw and processed are never mixed.** `dataset/raw/` holds only what the engine recorded;
`dataset/processed/` holds only derived rows. The Phase 1 interactive recordings in
`dataset/telemetry/` are a separate, legacy location used when you simply play the scene.

## Stable object identities

Object ids look like `s20260501_bld_3f2a1b7c`:

```
object_id = "s<seed>_<category>_<hash8>"

hash8 = first 8 hex digits of Godot's String.hash() over

        "<seed>|<category>|<model>|<x,y,z quantised>|<rotation quantised>|<scale quantised>"
```

where `model` is the `.glb` resource path for imported models, or a generated name such as
`wall_brick` for primitives.

Properties this gives:

| Property | Why it holds |
|---|---|
| Same seed ⇒ same ids | generation is deterministic and the hash is over rounded values |
| Unique per object | collisions are detected during generation and disambiguated with a `_2` suffix; the default seed produces none |
| Independent of observation order | the id is a hash of what the object *is*, not of when it was seen |
| Different seeds never collide | the seed is both hashed and prefixed |
| Consistent type/transform/texture | those values are the hash input |

Ids are **not** runtime instance ids and are never drawn from a random number generator.

`dataset/manifests/object_index.json` records, per object: `object_id`, the Phase 1 `legacy_id`,
`category`, `source`, `model`, node name, position, rotation, scale, world AABB, collision
flag, `texture_ids`, and the resolved texture detail (source path, original resolution,
available resolutions, whether variants exist). The file also carries a `generation_hash` over
all ids in generation order, which changes if the environment layout changes.

## Camera trajectories

`tools/generate_trajectories.py` writes declarative JSON. Two segment modes:

- **`control`** - open loop: an input action, a speed, and linear yaw/pitch ramps
- **`waypoint`** - closed loop: walk to a target point, turning towards it at a capped rate

Targets for the waypoint segments are derived from `object_index.json`, so "approach a
building" walks at a real building and stops just outside its bounding box, and "walk into a
plaza" arrives at a spot that is provably clear of every building.

The 14 trajectories in the `full` profile cover all ten behaviours from the specification,
with four additional speed/orientation variants:

| Id | Behaviour | Length |
|---|---|---|
| `t01_walk_straight` | straight walking | 18 s |
| `t02_sprint_straight` | straight sprinting | 16 s |
| `t03_walk_corner` | walking around a corner | 22 s |
| `t04_sprint_corner` | sprinting around a corner | 20 s |
| `t05_turn_left_right` | turning left and right while moving | 24 s |
| `t06_look_around` | looking around while stationary | 20 s |
| `t07_stop_start` | stopping and restarting | 22 s |
| `t08_forward_reverse` | moving forward and reversing | 20 s |
| `t09_approach_building` | approaching a building | 18 s |
| `t10_street_then_plaza` | narrow street then open plaza | 28 s |
| `t11_slow_walk_down` | slow walk, camera angled down (speed variant) | 18 s |
| `t12_sprint_weave` | sprint with continuous turning (orientation variant) | 20 s |
| `t13_approach_prop` | approaching a street prop (target variant) | 16 s |
| `t14_plaza_look` | panning around an open plaza (orientation variant) | 22 s |

`--profile quick` takes `t01`, `t06` and `t09` scaled to 40% duration (22.4 s total) for fast
test runs.

## Execution model

`godot/scripts/trajectory_runner.gd` replays a trajectory **without touching the player
controller**. It only uses the controller's public surface:

- sets the exported `walk_speed`
- writes `rotation.y` and the camera pitch
- presses `move_forward` / `move_back` / `sprint` through the real InputMap

So the player moves through exactly the code path a human would drive it through. The runner
has a lower physics process priority than the player, so the heading is applied before the
controller integrates movement for that tick.

The runner owns the trajectory queue rather than being driven from outside. That matters: if
the harness polled a frame callback to start the next trajectory, the gap between trajectories
would depend on how the engine schedules `_process` against physics and the replay would no
longer be reproducible.

Run under `--fixed-fps 60` so the simulation advances faster than real time: the reported run
covers 284 seconds of camera movement in about 8 seconds of wall clock.

## Splits

`tools/split_dataset.py` assigns whole trajectories, deterministically:

```
8 train / 3 validation / 3 test    (for 14 trajectories; 60/20/20 with a minimum of one each)
```

The assignment is asserted to be **exhaustive** (every trajectory in exactly one split) and
**disjoint**. Every processed row carries its trajectory's `split`, and the validator re-checks
the disjointness at row level, so a mislabelled row would be caught. Test trajectories are
never used for training or model selection.

Splitting at trajectory level is what prevents leakage: neighbouring frames of one camera path
are highly correlated, so a frame-level split would put near-duplicates on both sides.

## Tuning knobs

| Variable | Default | Effect |
|---|---|---|
| `MAX_DISTANCE` | 40 | radius, in metres, of the raw observation capture |
| `CANDIDATE_DISTANCE` | 25 | radius for labelling candidates |
| `SAMPLE_INTERVAL` | 6 | physics ticks per sample (6 = 10 Hz) |
| `HORIZONS` | `0.5,1.0` | prediction horizons in seconds |
| `OCCLUSION` | `1` | enable the occlusion raycast signal |

`MAX_DISTANCE` should exceed `CANDIDATE_DISTANCE` by at least `max_speed × max_horizon`
(12 m at sprint speed over a 1 s horizon); the validator warns if it does not.

## Validating a run

```bash
python3 tools/validate_dataset.py --run run_full
```

Writes `dataset/reports/run_full-validation.md` plus a JSON twin. See
[validation.md](validation.md).

## Determinism

Two runs of the same profile on the same Godot build and platform produce **byte-identical**
raw telemetry, except for the wall-clock `timestamp` field. This is asserted by the test
suite. Note that under `--fixed-fps` several samples can share the same millisecond of wall
clock; `physics_tick` and `session_time` are the authoritative keys and are always unique.

Cross-platform float reproducibility is *not* guaranteed - generate the dataset on one machine.
