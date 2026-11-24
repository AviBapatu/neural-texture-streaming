# Known limitations

Everything here is a deliberate scope boundary or a known gap, not a hidden defect. Items that
later phases resolved are marked as such so the history is clear.

## Scope

- **No neural network.** Phases 1, 2 and 2.5 deliver the environment, the texture pipeline, the
  telemetry infrastructure, the trajectories and the labels. There is no model, no prediction
  and no prefetch scheduler.
- **No runtime texture streaming.** Godot's own texture handling is untouched; the generated
  variants live in `dataset/` and are not loaded by the running project. Wiring a memory budget
  and mip switching into the renderer is the next phase.
- **`training/` and `exported_model/` are empty.** Model training happens on Kaggle; nothing has
  been trained or exported. `benchmarks/` holds the occlusion overhead measurement.

## Labels (Phase 2.5)

- **The test split is only three trajectories.** The trajectory-level split is 8 train /
  3 validation / 3 test. The three test trajectories are `t11_slow_walk_down`,
  `t13_approach_prop` and `t03_walk_corner` - so the test split contains **no sprinting and no
  pure-rotation trajectory**, even though rotation and approach are what generate most
  new-visibility events. Three evaluation trajectories is a thin sample: one trajectory's
  behaviour can dominate the score. Treat any test-set metric as indicative rather than as a
  stable estimate. More trajectories should be added only if evaluation shows insufficient
  coverage.
- **Only 7 of the 17 texture identities are eligible for resolution prediction.** Generated
  variants exist only for the ambientCG PBR albedos, so 83.8% of label rows
  (`resolution_eligible: false`) cannot contribute to a resolution loss. Visibility prediction
  is unrestricted. Generating variants for the Kenney and Quaternius textures is future work
  and was explicitly out of scope for Phase 2.5.
- **The occlusion-aware target carries an unknown bucket.** 5,388 rows (2.9%) are `null`
  because the occlusion state was never determined. They are unknown, not negative. Training on
  them requires an explicit choice - drop, reweight or mask - and the dataset does not make
  that choice. `occlusion_unknown_in_<h>s` is provided so the choice is available.
- **The occlusion target is strictly harder than the frustum target.** `True` on the occlusion
  target implies `True` on the frustum target, so the positive class is smaller. In
  `run_full_p25` it is 75,685 rows versus 134,135 across both horizons.
- **Counts in the reports are dataset diagnostics, not evidence of prediction.** "2,221 objects
  newly visible at 0.5 s" describes the scene and the trajectory set. It says nothing about
  whether a model can predict it.
- **Two target families plus compatibility columns make the labels large.** 
  `dataset/processed/run_full_p25` is 259 MB, of which 72 MB is the Phase 2
  `visible_in_*` / `occlusion_visible_*` compatibility block. `--drop-legacy-labels` removes it.
- **`visibility_in_*` (Phase 2) is byte-identical to `frustum_visible_in_*`.** They are the same
  quantity under two names; keeping both is a compatibility cost, not extra information.

## Visibility

- **Visibility is still an approximation.** It is explicitly *three* approximations rather than
  one, but none of them is the renderer's own answer.
  - `frustum_visible` uses the projected AABB, not the mesh, so it overestimates sparse shapes.
  - `approx_visible` is Phase 1's method and ignores occlusion entirely.
  - `occlusion_visible` casts at most `occlusion_max_rays` rays at a box approximation of the
    object. Aggregation is "any sample point reaches", so a partly-occluded object reads as
    visible.
- ~~Visibility can count objects behind buildings~~ - **addressed**: the occlusion signal is
  separate and measured to disagree with the frustum signal on about two thirds of tested
  observations, and the label set now exposes both as distinct targets. It remains an
  approximation, and it is off by default.
- **Occlusion only sees collidable geometry.** Roads, plaza slabs, grass and planters have no
  colliders, so they cannot occlude and are not meaningfully self-occluded. Buildings and
  walls - the real occluders - are boxes and are covered.
- **Objects outside the observation radius are invisible to the labeller.** The raw capture
  records to 40 m and labelling candidates to 25 m, with a documented 12 m margin for the 1 s
  horizon at sprint speed. Raising `HORIZONS` or camera speed without raising `MAX_DISTANCE`
  would silently truncate some future visibility.

## Telemetry

- **Sampling is tick-based in the dataset pipeline (10 Hz)** and wall-clock based when you
  simply play the scene. Under `--fixed-fps` several samples can share a wall-clock
  millisecond, so `timestamp` is not a unique key - `physics_tick` and `session_time` are.
- **No GPU, memory or texture-residency counters.** There is nothing yet to correlate a memory
  budget against; only camera motion and object observations are recorded.
- ~~Object identifiers are per-run and not persisted~~ - **addressed**: ids are content-derived
  and seed-scoped, and every one is written to `dataset/manifests/object_index.json`.
- **Object ids are stable, but the object-to-texture mapping is not versioned beyond
  `object_index.json`.** Regenerating the environment with a different seed produces a
  different index, and old telemetry would no longer join to it. `generation_hash` exists to
  detect that.

## Labels (resolution)

- **`required_resolution` is an estimate, not ground truth.** It is derived from the projected
  AABB rectangle and assumes one texel per screen pixel on a fronto-parallel surface. The
  scene has no real mip ground truth to measure against.
- **Labels are calculated from a 10 Hz sample stream.** A visibility event that begins and ends
  between two samples is missed. The 0.5 s horizon has five future samples, so this is unlikely
  but not impossible.
- **Straight-line trajectories produce almost no new-visibility events.** `t01_walk_straight`
  legitimately yields zero "becomes visible" labels; new visibility comes from rotation and
  approach. This is a property of the scene, and it means the positive class is dominated by
  the rotating and approaching trajectories.

## Environment

- **No LODs and no occlusion culling.** Every object is full detail at all distances. Fine for
  268 objects, but it is not a realistic streaming workload yet.
- **The player can walk off the edge of the ground plane** (200x200 units; the city is 80x80
  inside it) if they leave through a perimeter road gap.
- **One directional light only**, with no reflection probes, so the fully metallic showcase
  cube relies on the sky for its reflections.
- **Grass uses a triplanar-projected photo texture** rather than a terrain shader; close up it
  reads as a tiled texture rather than geometry.
- **No audio, no menus, no save/load.** The only UI is the control hint, crosshair and
  telemetry status label.

## Tooling and reproducibility

- `tools/generate_texture_variants.py`, `tools/build_texture_index.py` and
  `tools/validate_dataset.py` require **Pillow**.
- **Cross-machine reproducibility is not guaranteed.** Two runs on the same Godot build and
  platform are byte-identical; floating point across builds or architectures is not guaranteed
  to be, so generate a dataset on one machine.
- **Replaying a trajectory inside one process is not bit-exact.** The second replay starts from
  the physics state left by the first, which perturbs the camera transform by ~1e-9; `coverage`
  can move by one unit in the sixth decimal. Separate processes are exact, which is what the
  pipeline uses.
- **`dataset/manifests/splits.json` and `dataset_manifest.json` describe the most recent run.**
  Stable per-run copies are kept in `dataset/manifests/runs/<run-id>/`, and the validator
  prefers those, so an earlier dataset keeps validating after a later run with another profile.
- **Every dataset run duplicates its raw telemetry.** `run_full` and `run_full_p25` each carry
  their own 100 MB copy of bit-identical raw data. There is no hard-link or shared-raw mode.
- **The full pipeline takes roughly 2-4 minutes** and produces ~359 MB for `run_full_p25`.
- `tools/capture_screenshots.sh` needs a graphical session and a GPU; it will not work over
  plain SSH without a display.

## Assets

- The kit ships FBX/OBJ copies of the same models; only the GLB versions were imported.
- Vehicle and pedestrian models are **not** in the source kit, so the streets are static.
- All assets are CC0; see [asset-sources.md](asset-sources.md) for the full record.
