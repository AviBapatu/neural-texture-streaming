# Telemetry schema (v2)

`godot/scripts/telemetry_recorder.gd` writes one JSONL record per sample. Schema v2 is
**additive**: every Phase 1 field still exists with the same name and meaning, and the
25-column CSV is byte-compatible with Phase 1. `schema_version` is present in every dataset
file.

## Where the data lands

| Location | Written by | Purpose |
|---|---|---|
| `dataset/telemetry/` | playing the game (`main.tscn`), or `--legacy-visible=1` runs | Phase 1 behaviour, unchanged. Includes the `visible` array and CSV. |
| `dataset/raw/<run>/<trajectory_id>.jsonl` | `godot/tools/run_trajectories.gd` | pipeline raw capture |
| `dataset/raw/<run>/<trajectory_id>.meta.json` | ditto | per-trajectory metadata and cost measurements |
| `dataset/raw/<run>/run_summary.json` | ditto | run-level totals |

## Sampling modes

`sample_mode` is either:

- `realtime` - Phase 1 behaviour, wall-clock based at `samples_per_second` (10 Hz)
- `physics` - one sample every `sample_interval_ticks` physics ticks (6 ticks = 10 Hz at 60 ticks/s)

The dataset pipeline uses `physics`. In that mode `session_time == physics_tick / 60` and
`frame` is the physics tick, so the record no longer depends on how the engine schedules
`_process` against physics. This is what makes two runs byte-identical.

## Per-sample fields

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | `2` |
| `timestamp` | float | Unix wall clock. The only field that legitimately differs between replays. |
| `session_time` | float | seconds since the session began |
| `frame` | int | process frame (realtime) or physics tick (physics) |
| `physics_tick` | int | physics ticks since the session began |
| `environment_seed` | int | the seed of the environment that produced this sample |
| `cam_position` | `[x,y,z]` | camera world position |
| `cam_quaternion` | `[x,y,z,w]` | camera orientation |
| `cam_euler` | `[x,y,z]` | camera orientation in radians (Phase 1) |
| `cam_velocity` | `[x,y,z]` | player velocity, units/s |
| `cam_angular_velocity` | `[x,y,z]` | axis-angle finite difference of consecutive orientations, rad/s |
| `cam_direction` | `[x,y,z]` | unit forward vector |
| `cam_fov` | float | vertical field of view in degrees |
| `viewport` | `[w,h]` | viewport size in pixels |
| `observation_count` | int | length of `observations` |
| `visible_count` | int | how many of those are `frustum_visible` |
| `culled_count` | int | objects beyond the observation distance |
| `occlusion_enabled` | bool | whether the raycast pass ran for this recording |
| `observations` | array | every object within `max_visible_distance`, see below |
| `visible` | array | Phase 1 array, only when `legacy_visible_array` is on |

## Per-observation fields

`observations` contains **every object within `max_visible_distance`**, not just the ones on
screen. An object that is outside the frustum is still listed, with `frustum_visible: false`,
rather than being silently dropped. `culled_count` accounts for the rest.

| Field | Type | Notes |
|---|---|---|
| `object_id` | string | stable, seed-scoped id (see [dataset-generation.md](dataset-generation.md)) |
| `legacy_id` | int | the Phase 1 per-run ordinal |
| `category` | string | `rod` `bld` `wal` `prp` `shw` `plz` |
| `distance` | float | camera to AABB **centre**, metres |
| `coverage` | float | fraction of the viewport covered, `[0,1]` |
| `frustum_visible` | bool | the AABB projects into the viewport (`coverage > 0`) |
| `approx_visible` | bool | the Phase 1 signal: frustum-visible and within distance |
| `occlusion_visible` | `true`/`false`/`null` | raycast result; `null` = not determined |
| `occlusion_reason` | string | why the occlusion value is what it is |
| `occlusion_rays` | `[tested, open]` | rays cast and rays that reached the object |
| `texture_ids` | array | the object's texture identities |
| `required_resolution` | int | estimated resolution needed at this coverage, `0` when off screen |
| `name` / `source` | string | only emitted when `legacy_visible_array` is on |

`occlusion_reason` is one of:

| Value | Meaning |
|---|---|
| `tested` | rays were cast; `occlusion_visible` is `true` or `false` |
| `disabled` | the pass is off for this recording; `occlusion_visible` is `null` |
| `below_coverage_threshold` | the object is too small on screen to be worth a ray; `null` |
| `no_space_state` | no physics space was available; `null` |
| `behind_near_plane` | every sample point was behind the near plane; `null` |
| `too_close` | the object is closer than 0.5 m; `null` |

**`null` is never "invisible".** It means the question was not answered. A consumer that wants
a definitive answer must check `occlusion_reason`, or treat `null` as "unknown" and fall back
to `approx_visible`.

## The three visibility signals are kept separate

| Signal | Answers | Cost | Known weakness |
|---|---|---|---|
| `frustum_visible` | is it in the view frustum? | cheap | ignores everything between camera and object |
| `approx_visible` | Phase 1's method | cheap | reports objects hidden behind buildings |
| `occlusion_visible` | do rays reach it? | ~0.97 ms/sample with 4 rays | only sees collidable geometry |

Measured on the full city: with occlusion enabled, **1497 of 2289 tested observations were
actually occluded** - about 65% of what the frustum method calls "visible" is behind a
building. That is the whole reason the third signal exists.

## Cost

Per sample over ~240 observations, measured by `godot/tools/benchmark_visibility.gd`:

| | mean visibility cost |
|---|---|
| occlusion disabled | 1.53 ms |
| occlusion enabled | 2.56 ms |
| **added by occlusion** | **~0.97 ms** (311 rays/sample) |

Sampling runs every 6 physics ticks = a 100 ms budget, so the occlusion pass costs about
**1% of the budget**. It is an exported property (`occlusion_enabled`, `occlusion_max_rays`)
and defaults to off, so normal gameplay never pays for it.

## Validating

```bash
python3 tools/validate_telemetry.py dataset/telemetry      # Phase 1 files
python3 tools/validate_dataset.py --run <run-id>           # full dataset schema and labels
```
