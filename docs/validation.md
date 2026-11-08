# Validation

Everything below was executed on the development machine (Linux, NVIDIA GeForce MX550,
Vulkan 1.4 Forward+). Nothing in this file is claimed without having been run.

## Running the checks

```bash
# everything: Phase 1 environment checks plus Phase 2 dataset checks
godot/tests/run_tests.sh

# individual pieces
godot --headless --path godot --import
godot --headless --path godot --script res://tests/environment_report.gd
godot --headless --path godot --script res://tests/movement_test.gd
godot --headless --path godot --script res://tests/object_identity_test.gd
godot --headless --fixed-fps 60 --path godot --script res://tests/trajectory_test.gd
godot --headless --path godot --quit-after 360
godot --headless --path godot --script res://tools/benchmark_visibility.gd
python3 tools/validate_telemetry.py dataset/telemetry
python3 tools/validate_texture_variants.py
python3 tools/validate_dataset.py --run run_full        # Phase 2 label set
python3 tools/validate_dataset.py --run run_full_p25    # Phase 2.5 dual-target label set

# full dataset generation, then visual check (the latter needs a graphical session)
tools/run_dataset_pipeline.sh --profile full
tools/capture_screenshots.sh
```

Set `GODOT=/path/to/Godot_v4.7.2-stable_linux.x86_64` if the binary is not on `PATH`.

## Result

`godot/tests/run_tests.sh` exits `0` with **ALL CHECKS PASSED**:

| # | Check | Result |
|---|---|---|
| 1 | Godot version | `4.7.2.stable.official.ed1daf0bf` |
| 2 | Asset import | no script errors |
| 3 | Materials | all 7 PBR materials load with albedo + normal + roughness textures |
| 4 | Models | 234 `.glb` files load, 0 failures |
| 5 | Environment | 268 tracked objects generated (seed 20260501) |
| 6 | Controller | ground collision, WASD movement, camera reset, building collision |
| 7 | Headless run | main scene runs 360 frames with no script errors |
| 8 | Telemetry | Phase 1 session validates |
| 9 | Texture variants | 99 variants verified against the manifest |
| 10 | Object identity | same seed ⇒ identical ids and hash; different seed ⇒ disjoint ids |
| 11 | Object index across processes | identical, hash `f3b18df8` |
| 12 | Dataset pipeline (quick) | `PIPELINE OK`, `run_quick: PASSED` |
| 13 | Trajectory replay determinism | discrete fields identical; `coverage` within 1e-6 |
| 14 | Raw capture across processes | 3/3 trajectories byte-identical |
| 15 | Occlusion overhead benchmark | 0.92-1.35 ms/sample added |

(The suite has 12 numbered steps; steps 2-6 above correspond to step 2 of the script.)

### Controller test detail

The controller is driven through the real `InputMap` actions declared in `project.godot`:

```
[2] gravity and ground collision   ok : player rests on the ground plane (y=0.000)
[3] WASD movement                  ok : moved 3.40 m along the view direction
[4] camera reset                   ok : reset restored the spawn position (0.0, 0.0, -34.0)
[5] collision against a building   ok : advanced 3.60 m, stopped outside the building
                                        (z=-30.259, front face -30.659)
```

### Determinism

Two different guarantees, both asserted:

| Property | Method | Result |
|---|---|---|
| Same seed ⇒ same object ids | `object_identity_test.gd`, three generations in one process | identical id sequence, same `generation_hash` |
| Different seed ⇒ no shared ids | same test, seed `777001` | 0 shared ids of 268/262 |
| Same trajectory replayed in one process | `trajectory_test.gd` | every discrete field identical; worst `coverage` delta 1e-6 |
| Same trajectory replayed in two processes | `run_tests.sh` step 11 | **byte-identical** (wall-clock timestamps excluded) |

Replaying inside a shared process starts from the physics state left by the previous
trajectory, so contact and depenetration history can perturb the camera transform by ~1e-9.
That surfaces as a one-unit difference in the sixth decimal of `coverage` and nothing else.
Fresh processes are exact, which is what dataset generation relies on.

Under `--fixed-fps` several samples can share a wall-clock millisecond, so `physics_tick` and
`session_time` are the authoritative keys. The validator reports the shared milliseconds as
warnings, not errors.

### Validation is not vacuous

Deliberately corrupting four processed rows - flipping a `visible_in` label, setting an
off-ladder `required_resolution`, setting a negative `first_visible_time`, and flipping
`resolution_insufficient` - produced **7 specific errors** covering all four, and the report
returned to `PASSED` after restoring the file.

### Phase 2.5 label checks

`tools/validate_dataset.py` is schema-aware: it detects whether a label file carries the Phase 2
label set or the Phase 2.5 dual-target set and applies the matching rules. This is what lets the
preserved `run_full` keep validating after the new targets were introduced.

| Run | Detected label set | Result |
|---|---|---|
| `run_full` | phase-2 (single target) | PASSED, 0 errors |
| `run_full_p25` | phase-2.5 (dual targets) | PASSED, 0 errors |
| `run_quick` | phase-2 (single target) | PASSED, 0 errors |

Checks specific to the Phase 2.5 label set, all passing on `run_full_p25`:

| Check | Result |
|---|---|
| occlusion-aware target is tri-state, unknown not treated as invisible | 75,685 true / 295,865 false / 5,388 unknown |
| frustum and occlusion targets are recorded separately | 53,062 rows frustum-yes but definitively occluded |
| resolution eligibility mask matches the texture index | 30,469 eligible / 158,000 ineligible |
| future labels recomputed independently | 1,174 labels cross-checked |

Per-row assertions:

- `frustum_visible_in_<h>s` agrees with the legacy `visible_in_<h>s` on every row.
- `occlusion_visible_in_<h>s` is `true` / `false` / `null` and nothing else.
- `null` never appears without `occlusion_unknown_in_<h>s` being true.
- `null` never appears for an object that was never in the frustum - leaving the frustum is
  sufficient to know an object is not visible, so that case must be `false`.
- `true` always carries a first-visibility time inside the horizon and a non-zero required
  resolution; `false` never carries a first-visibility time.
- `occlusion_visible_in_<h>s == true` implies `frustum_visible_in_<h>s == true`.
- `resolution_eligible` matches the texture index exactly; eligible rows carry exactly the
  texture index's `resolution_variants` and `variant_texture_ids`; ineligible rows carry none.
- Both targets are recomputed from the raw stream by a naive, independent linear scan and
  compared field by field.

### Preserved splits

The pipeline compares a new run's trajectory assignment against a reference run and **fails** if
they differ. It skips the comparison when the two runs use different trajectory profiles, since
those are expected to differ.

```
checking the assignment against run_full
split assignment identical to run_full: True          # run_full_p25, same profile
different profiles (full vs quick) - nothing to compare   # run_quick
```

### Graphical launch

The project was launched through the graphical Godot application (not headless) and rendered
correctly with `Vulkan 1.4.354 - Forward+ - NVIDIA GeForce MX550`. The environment, models,
materials, HDRI sky, shadows and the HUD all render. Screenshots captured this way are in
`docs/screenshots/`:

| File | View |
|---|---|
| `01_spawn.png` | the player spawn on the southern avenue |
| `02_street_center.png` | looking north up the main avenue |
| `03_crossroads.png` | a road intersection with crosswalks and street furniture |
| `05_gallery.png` | the seven-cube material showcase seen from the road |
| `06_skyline.png` | a side street with buildings at several distances |
| `07_aerial.png` | elevated view over the block grid |
| `08_gallery_close.png` | elevated view of the material showcase and its plaza |

### Telemetry sample

```
samples        : 24
observations   : 4560
coverage range : 0.000004 .. 0.228439
distance range : 2.537 .. 74.193
texture ids    : 9 distinct
CSV rows       : 4560          # matches the JSONL observation count
```

## Re-importing after asset changes

```bash
godot --headless --path godot --import
```

Godot regenerates `.godot/imported/` on the first run. The class cache (used for `class_name`
lookups) is refreshed by the same command.
