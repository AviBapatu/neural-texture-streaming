# Known issues

Open defects and gaps found so far, with reproduction. Nothing here is fixed in Phase
3a.1 - fixes belong to the phase that owns the affected code.

---

## 1. `cam_angular_velocity` uses the wrong arc (quaternion double cover)

**Status: NOT FIXED.** Worked around in the feature pipeline only.
**Owner when fixed:** Phase 4 (Godot / recorder).
**Severity:** high for any feature that reads the field directly; low in the current
pipeline, which recomputes it.

### What is wrong

`godot/scripts/telemetry_recorder.gd:325-338`:

```gdscript
func _angular_velocity(quaternion: Quaternion) -> Vector3:
	if not _previous_quaternion_valid or _sample_delta <= 0.0:
		_previous_quaternion = quaternion
		_previous_quaternion_valid = true
		return Vector3.ZERO
	var delta := _previous_quaternion.inverse() * quaternion
	_previous_quaternion = quaternion
	var angle := delta.get_angle()
	if is_zero_approx(angle):
		return Vector3.ZERO
	var axis := delta.get_axis()
	if not axis.is_finite():
		return Vector3.ZERO
	return axis * (angle / _sample_delta)
```

`q` and `-q` are the same rotation. When `previous.inverse() * current` lands on the `-q`
representation, `get_angle()` returns `2*pi - theta` and `get_axis()` returns the negated
axis. The code never checks the scalar part, so it reports a near-full-turn rotation in
the wrong direction instead of the small rotation that actually happened.

### Reproduction

Every affected sample has the signature `recorded_angle + true_angle = 360 deg`:

| trajectory | sample index | recorded (deg/s) | correct (deg/s) | recorded vector (rad/s) |
|---|---|---|---|---|
| t03_walk_corner | 106 | 3563.9 | 35.92 | `[0, -62.2017, 0]` |
| t05_turn_left_right | 110 | 3570.0 | 30.05 | `[0, -62.3085, 0]` |
| t05_turn_left_right | 140 | 3585.3 | 15.04 | `[0, +62.5758, 0]` |
| t06_look_around | **33** | **3582.5** | **17.97** | `[0, -62.5272, 0]` |
| t06_look_around | 55 | 3546.1 | 54.05 | `[0, +61.8910, 0]` |
| t14_plaza_look | 200 | 3570.0 | 30.05 | `[0, -62.3085, 0]` |

Worked example, `t06_look_around` sample 33: the camera is panning steadily at 18 deg/s
(samples 31, 32 and 34 are all 18.0 deg/s). Sample 33 records 3582.5 deg/s, i.e. 358.25
deg in one 0.1 s step. `358.25 + 1.79 = 360.04`, so the true rotation is 1.79 deg in the
opposite direction. `-62.5272 rad/s * 0.1 s = 6.2527 rad = 358.2 deg`.

**Extent: 6 of 2,826 sample pairs (0.21%)**, in 4 of 14 trajectories.

### Impact

- **`cam_angular_velocity` itself** is wrong at those samples.
- **Downstream features explode.** Before correction, `cam_ang_accel_deg_s2` for the
  affected rows spikes to **36,005 deg/s²**; its 99th percentile across the dataset was
  36,001 and became 180 after correction. Any normalisation fit over those rows is
  dominated by the spike.
- **`cam_yaw_rate_deg_s`, `cam_pitch_rate_deg_s`, `cam_roll_rate_deg_s` and
  `cam_ang_speed_deg_s`** inherit it, since they project the same vector.
- **Row counts:** 6 sample pairs, each with up to ~110 candidate rows, so roughly 450 of
  188,469 label rows (0.24%).

### How it is worked around today

`training/features.py::angular_velocity_from_quaternions` recomputes the angular velocity
from the raw orientations, negating the quaternion difference when its scalar part is
negative:

```python
negate = delta[:, 3] < 0.0
delta[negate] *= -1.0
```

This is exact for any rotation under 180 degrees per sample, which 10 Hz sampling can be
taken to guarantee for this camera. It returns a flag array marking the samples where it
disagreed with the recorded value, so the extent of the defect stays measurable. Unit
tests cover both the correction and the flagged-sample count.

### Proposed fix

In `_angular_velocity`, choose the shorter arc before decomposing:

```gdscript
var delta := _previous_quaternion.inverse() * quaternion
if delta.w < 0.0:
	delta = Quaternion(-delta.x, -delta.y, -delta.z, -delta.w)
```

That is the whole fix. It cannot be applied in Phase 3a.1 because `godot/` is out of scope,
and applying it would change recorded telemetry and therefore invalidate the existing
datasets, which must not be regenerated.

### Verification once fixed

Re-run the reproduction above: all six rows must report a value close to the corrected
column, and the `flagged` count in
`training/tests/test_features.py::test_recorded_angular_velocity_defect_is_bounded` must
fall to zero.

---

## 2. `control` segments snap the camera at `waypoint` boundaries

**Status: NOT FIXED.** Documented, and the affected rows are excluded from headline
metrics; the affected trajectories are held out of both evaluation sets in split v1.
**Owner when fixed:** Phase 4 (trajectory generator and/or runner).
**Severity:** high for evaluation; the affected trajectories' visibility labels are
artefacts.

### What is wrong

`godot/scripts/trajectory_runner.gd:176-177` assigns the heading **absolutely**:

```gdscript
player.rotation.y = deg_to_rad(lerpf(
    float(segment.get("yaw_from", 0.0)), float(segment.get("yaw_to", 0.0)), local))
```

There is no continuity constraint against the heading left behind by the preceding
segment. A `waypoint` segment (`trajectory_runner.gd:183-195`) leaves the heading wherever
pursuit steering converged:

```gdscript
var desired := atan2(-offset.x, -offset.z)
player.rotation.y = _approach_angle(player.rotation.y, desired, max_step)
```

When a `waypoint` segment is followed by a `control` segment whose `yaw_from` differs from
the converged heading, the camera turns instantly.

### Reproduction

All three snaps are at a `waypoint` -> `control` boundary with `yaw_from = 0.0`:

| trajectory | sample | time | heading before | heading after | change | definition |
|---|---|---|---|---|---|---|
| t09_approach_building | 140 | 14.1 s | -180.00 deg | 0.00 deg | 180.00 deg | `generate_trajectories.py:301` |
| t13_approach_prop | 120 | 12.1 s | -90.00 deg | 0.00 deg | 90.00 deg | `generate_trajectories.py:352` |
| t14_plaza_look | 100 | 10.1 s | 81.87 deg | 1.67 deg | 80.20 deg | `generate_trajectories.py:363` |

The camera position does not change at any of the three. `t14` then ramps correctly from
the new heading (`yaw_to = 120` over 6 s); only the starting point is discontinuous.

`t09` and `t13` declare `yaw_from == yaw_to == 0.0`, a no-op ramp, which is consistent with
the field having been written as if it meant "do not rotate" rather than "face world
yaw 0".

### Why the authoring-intent reading is plausible

Every place `control` segments were chained so that one's `yaw_to` equals the next's
`yaw_from`, there is no snap: `t12` chains 150->210->150->210->180; `t14` chains its
segment-1 `yaw_to=120` into segment-2 `yaw_from=120`; `t11`'s single `control` segment
declares `yaw_from=90`, matching its `start_yaw_degrees=90`. The runner is faithful; the
segments say something that was not intended.

### Impact

- 3 samples in 3 trajectories.
- **Every newly-visible event in `t09_approach_building` (175) and `t13_approach_prop` (65)
  is within +/-10 samples of a snap.** Their visibility labels describe a camera flip, not
  a camera moving. `t14_plaza_look` has 66 of 292 (23%).
- This is what made the Phase 3a headline (primary-subset PR-AUC 0.189) an artefact: 65 of
  its 100 test positives came from `t13`. Removing `t13` gives 0.904.

### Proposed fixes, in order of preference

1. **Make the generator author the intended behaviour.** Derive each `control` segment's
   `yaw_from` from the heading the preceding `waypoint` segment will have converged to, and
   assert continuity across every segment boundary when generating the definitions. Add a
   generator-side check that fails if any boundary exceeds the turn-rate limit.
2. **Add a "hold current heading" mode** to `control` segments, so the common intent
   ("stop moving and keep looking where you are") has a way to be expressed, and treat a
   missing `yaw_from` as that rather than as 0.
3. **Rate-limit in the runner**: clamp any per-tick yaw change to the segment's
   `turn_rate_degrees_per_second`. This would hide the symptom rather than fix the
   semantics, and would silently change existing definitions, so it is the least
   preferred.

### Interim mitigation (what Phase 3a.1 does)

- The three trajectories are placed in **training** in split v1, so neither evaluation set
  is scored on artefacts.
- Snap-affected rows are **excluded from headline metrics** and reported separately.
- The flags are available per `(trajectory, sample)` in
  `training/experiments/phase3a1/eval_flags.csv`.
- **Recommended for Phase 3b:** exclude snap-affected rows from training too, so the model
  is not taught that a 180 deg rotation in 0.1 s is a normal camera event.

---

## 3. `trajectory_kind` in the processed samples is the id suffix, not the declared kind

**Status: NOT FIXED.** Worked around by reading the trajectory index instead.
**Owner when fixed:** Phase 4 (labeller / dataset generation).
**Severity:** low for modelling; medium for any analysis that groups by movement type.

`tools/label_future_demand.py` sets the `trajectory_kind` column of each processed sample
from `trajectory_id.split("_", 1)[-1]` rather than from the trajectory definition's
declared `kind`. Three trajectories disagree with the trajectory index:

| trajectory | declared kind | recorded `trajectory_kind` |
|---|---|---|
| `t11_slow_walk_down` | `walk_straight` | `slow_walk_down` |
| `t12_sprint_weave` | `sprint_corner` | `sprint_weave` |
| `t14_plaza_look` | `look_around` | `plaza_look` |

The other eleven happen to agree because their id suffix matches their kind.

**Impact:** any grouping by `trajectory_kind` from the processed samples misclassifies
three of fourteen trajectories - and does so inconsistently, since the suffix is a
human-readable label rather than a category. The audit and split v1 use
`dataset/trajectories/full/index.json` as authoritative
(`training/audit.py::declared_kinds`).

**Fix:** populate the column from the trajectory definition, or drop it and join to the
index. Test `test_recorded_trajectory_kind_column_is_not_the_declared_kind` pins the
current mismatch so the workaround is removed deliberately rather than by accident.

---

## 4. `required_resolution` is an estimate, and the resolution target is ambiguous before visibility

**Status: accepted as designed; a recommendation is recorded.**
**Owner:** Phase 3b.

`required_resolution` is derived from the projected AABB rectangle, not from ground truth
(see `docs/future-demand-labels.md`). Two consequences that Phase 3a.1 measured:

- The coverage-threshold baseline is near-perfect on already-visible rows (macro-F1 0.994,
  under-prediction 1.2%) and **exactly useless on not-yet-visible rows**: `coverage` is
  identically 0.0 outside the frustum, so it predicts resolution 0 for all 482 (0.5 s) and
  960 (1.0 s) primary rows, and under-predicts 100% of them.
- The future requirement differs from what current coverage implies on **10.9%** of eligible
  rows at 0.5 s and **14.6%** at 1.0 s. All of those differences are in the region a
  predictor would have to cover.

Only 7 of 17 texture identities have generated variants, so the target is actionable for a
minority of objects - 30,469 of 188,469 rows, and only 482 of them in the primary subset.

---

## 5. Trains/serve parity - features that must be recomputed identically in GDScript

**Status: open by construction.** Phase 3a.1 computes features in Python for training.
Phase 4 will need the same values at runtime in GDScript. Any difference between the two
implementations is a silent accuracy loss, and it will not show up in offline metrics.

The rule for Phase 4: **every feature below must be implemented in GDScript and verified
against the Python values on recorded telemetry, before anything is deployed.** The
verification is cheap - the Python pipeline already writes its feature matrix, so a
GDScript exporter can be diffed against it row by row on a recorded trajectory.

| Feature group | Features | Why parity is hard |
|---|---|---|
| Camera-frame transform | `obj_cam_x/y/z`, `obj_cam_radius` | quaternion -> basis -> world-to-camera convention; Godot's `Basis` ordering must match `features.quaternion_to_matrix` exactly |
| Angle extraction | `azimuth_deg`, `elevation_deg` | `atan2` argument order and the sign of the forward axis |
| Bearing rate | `bearing_rate_deg_s`, `bearing_rate_valid` | **must wrap the azimuth delta into [-180, 180]**; an unwrapped difference jumps by 360 at the wrap point |
| Closing speed | `closing_speed_m_s`, `closing_speed_valid` | needs the previous AABB distance, so the previous camera pose must be retained |
| Angular velocity | `cam_yaw_rate_deg_s`, `cam_pitch_rate_deg_s`, `cam_roll_rate_deg_s`, `cam_ang_speed_deg_s` | **must use the shortest-arc correction** (issue 1); must project onto camera-local axes, not world axes |
| Linear acceleration | `cam_lin_accel_mag`, `cam_accel_cam_x/y/z`, `cam_lin_accel_valid` | backward difference of `cam_velocity`, with the interval taken from `session_time` rather than an assumed 0.1 s |
| Angular acceleration | `cam_ang_accel_deg_s2`, `cam_ang_accel_valid` | backward difference of the **corrected** angular velocity; inherits issue 1 if not |
| Pose history | `hist_dpos_1..4`, `hist_dyaw_1..4`, `hist_dpitch_1..4`, `hist_speed_1..4`, `hist_valid_1..4` | a rolling 4-slot buffer that must be **reset at session start**, and must never carry across a trajectory or session boundary; the validity flags must be exported, not inferred |
| Screen state | `coverage`, `log_coverage`, `frustum_visible`, `occlusion_true/false/unknown` | `log_coverage` is `log1p(coverage * 1e6)`; the occlusion tri-state must keep "unknown" distinct from "false" |
| Resolution state | `required_resolution`, `log_required_resolution` | `log2` of the estimate, 0 when the object is not visible; the underlying map is already GDScript and is the canonical source |
| Static object facts | `current_resolution`, `n_available_resolutions`, `max_available_resolution`, `has_variants`, `resolution_ratio` | derived from the texture manifest; must match `build_texture_index.py` |
| Category one-hots | `cat_bld/plz/prp/rod/shw/wal` | fixed alphabetical order, shared constant |
| Normalisation | all columns | mean/std must be loaded from the JSON written by `fit_normaliser` and applied in the same order; a mismatched column order is silent and total |

**Not required at runtime, and deliberately so:** the evaluation sidecar columns
(`snap_affected`, `movement_type`) and every target field. They are analysis-only and the
allow-list rejects them.

**Recommended Phase 4 gate:** an offline parity test that loads one recorded trajectory,
computes every feature in GDScript, and asserts equality with the Python matrix within a
documented tolerance. Without it, train/serve skew is unmeasurable.
