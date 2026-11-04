# Camera telemetry (Phase 1)

> **This page documents the Phase 1 `dataset/telemetry/` recordings.** For the full schema the
> dataset pipeline writes - including the three visibility signals, occlusion, per-frame
> fields and the processing-tick sampling mode - see
> [telemetry-schema.md](telemetry-schema.md). The fields below are unchanged in schema v2; they
> are simply no longer the whole story.

`godot/scripts/telemetry_recorder.gd` records what the camera could see while the environment
is played. It is deliberately only infrastructure - no prediction, prefetching or scheduling
happens here.

## Output

Two files are written per session into `dataset/telemetry/`:

- `session_<timestamp>.jsonl` - one JSON object per sample, with a nested `visible` array
- `session_<timestamp>.csv` - one row per *(sample, visible object)* pair, for spreadsheet and
  dataframe workflows

The directory is `<project root>/dataset/telemetry`, derived from the Godot project directory
rather than hard-coded. Override it with the `output_directory` export if needed.

## Sampling

Samples are taken at `samples_per_second` (default **10 Hz**) of wall-clock time, so the number
of samples depends on how long the session ran, not on the frame rate. Recording starts
automatically and can be paused/resumed at runtime with `F1`. Objects further away than
`max_visible_distance` (default 75 units) are ignored.

## JSONL schema

```json
{
  "timestamp": 1793909153.056,
  "session_time": 1.6,
  "frame": 96,
  "cam_position": [0.0, 0.0, -34.0],
  "cam_quaternion": [0.0, 1.0, 0.0, 0.0],
  "cam_euler": [0.0, 3.1416, 0.0],
  "cam_velocity": [0.0, 0.0, 3.3],
  "cam_direction": [0.0, 0.0, 1.0],
  "visible": [
    {
      "id": 24,
      "name": "obj_0024",
      "source": "road-straight.glb",
      "distance": 2.537,
      "coverage": 0.004777,
      "textures": ["kenney/city_roads_colormap"]
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `timestamp` | Unix wall-clock time of the sample |
| `session_time` | seconds since the recorder started |
| `frame` | engine frame counter |
| `cam_position` | camera world position (x, y, z) |
| `cam_quaternion` | camera orientation as a quaternion (x, y, z, w) |
| `cam_euler` | camera orientation as Euler angles in radians |
| `cam_velocity` | player velocity (x, y, z), units/second |
| `cam_direction` | unit forward vector of the camera |
| `visible[]` | objects deemed visible from this camera pose |
| `visible[].id` | stable per-run object identifier |
| `visible[].name` | node name in the generated scene |
| `visible[].source` | originating model file (or `wall_*` / `gallery_*` / `plate_*`) |
| `visible[].distance` | metres from the camera to the object AABB centre |
| `visible[].coverage` | approximate fraction of the viewport covered |
| `visible[].textures` | identifiers of the textures that object resolves to |

`visible` is sorted by ascending distance.

Texture identifiers are **derived from the albedo textures the objects actually use**, not from
a hand-written list (Phase 2 fixed a Phase 1 inaccuracy here). 17 are in use across the scene:
7 `ambientcg/*_Color` PBR albedos, 3 `kenney/*_colormap` atlases and 7 `quaternius/*` model
textures. `dataset/manifests/texture_index.json` is the authoritative list, with each id's
source path, original resolution and available variants.

## CSV columns

```
timestamp, session_time, frame,
cam_x, cam_y, cam_z, quat_x, quat_y, quat_z, quat_w, euler_x, euler_y, euler_z,
vel_x, vel_y, vel_z, dir_x, dir_y, dir_z,
object_id, object_name, object_source, distance, coverage, textures
```

`textures` is a `|`-separated list.

## How visibility is approximated

There is **no occlusion test**: an object on the far side of a building still counts as
visible if it is close enough and projects into the viewport. For each object the recorder
projects the eight corners of its world-space AABB with `Camera3D.unproject_position`, takes
the bounding rectangle of the projected points, clips it to the viewport, and reports the
ratio of areas. Corners behind the camera are skipped, so objects straddling the near plane
are underestimated rather than rejected.

This is a cheap, deterministic approximation that is good enough to drive a prefetching
dataset, but it is not the renderer's own visibility information. Replacing it with
`VisibleOnScreenNotifier3D` or a GPU readback is a natural next step.

## Validating a session

```bash
python3 tools/validate_telemetry.py dataset/telemetry
```

The validator checks the schema, that all values are finite, that coverage stays in `[0, 1]`,
that distances are non-negative, that no record has an empty texture list, and that the CSV
row count matches the number of JSONL observations.
