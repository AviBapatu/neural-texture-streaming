# Future texture-demand labels

The point of the dataset is to answer, for every moment and every object near the camera:
*will this object need a high-resolution texture soon, and how high?* This document defines
exactly how those labels are computed and what they are worth.

## Two prediction targets

Since Phase 2.5 the dataset carries **two distinct future-prediction targets**, so it can be
determined whether predicting frustum entry is sufficient or whether occlusion-aware
prediction actually improves streaming decisions.

| Target | Question | Type | Fields |
|---|---|---|---|
| **Frustum demand** | Will the object's bounding box enter the camera frustum? | boolean | `frustum_visible_in_<h>s` |
| **Occlusion-aware visible demand** | Will the object *actually* become visible, after accounting for occlusion? | **tri-state** | `occlusion_visible_in_<h>s` |

They are separate columns. They are not interchangeable, and the difference is large: in
`run_full_p25`, **53,062 rows** (across both horizons) enter the frustum but are definitively
occluded, and **5,388 rows** are genuinely unanswerable.

### The occlusion target is tri-state, not boolean

An object's occlusion state is not always known - the raycast pass may have been disabled, the
object may have been too small on screen to be worth a ray, or it may have been behind the near
plane. The target therefore has three values:

| Value | Meaning |
|---|---|
| `True` | at least one observation in the window was frustum-visible **and** the raycast confirmed it was not blocked |
| `False` | the object was never frustum-visible in the window, or only at moments the raycast confirmed were blocked |
| `null` | it was frustum-visible, but every such moment has an unknown occlusion state |

**`null` is never "not visible".** `occlusion_unknown_in_<h>s` (a boolean) accompanies the
target so a consumer can filter or reweight those rows rather than silently training on them.

Leaving the frustum is *not* a "maybe". Being outside the frustum is sufficient to know an
object is not visible, so an empty window is `False` and never `null`; `null` is reserved for
"frustum-visible, occlusion undetermined". The validator asserts both directions of this rule.

## Horizons

Labels are produced for two prediction horizons by default, **0.5 s and 1.0 s**, configurable
with `--horizons 0.5,1.0,2.0`. The horizon appears in the field name, with the decimal point
replaced by an underscore:

| Horizon | Suffix |
|---|---|
| 0.5 s | `_0_5s` |
| 1.0 s | `_1_0s` |

So a 2.0 s horizon would produce `frustum_visible_in_2_0s`, and so on.

## The window

For a sample at time `t`, the label window is **`[t, t + horizon]`** - inclusive of the current
moment. `frustum_visible_in_<h>s` therefore answers *"is this object in the frustum at any
point from now until t+h?"*.

The case that actually drives prefetching is an object that is **not on screen now but will
be**:

```
frustum_visible_in_<h>s == true  AND  frustum_visible == false
```

and for those rows *every* label value is derived from observations strictly after `t`, so
nothing leaks backwards. The test suite asserts that such rows always have
`first_visible_time > 0`, and the validator recomputes labels from the raw stream with an
independent implementation and compares.

## Fields

For each candidate object and each horizon:

### Frustum target

| Field | Meaning |
|---|---|
| `frustum_visible_in_<h>s` | in the frustum at some point in `[t, t+h]` |
| `frustum_first_visible_time_<h>s` | seconds from `t` until first in the frustum (`0.0` if already, `null` if never) |
| `frustum_max_coverage_<h>s` | largest screen-space coverage in the window (`0.0` if never) |
| `frustum_required_resolution_<h>s` | estimated resolution needed at first frustum visibility (`null` if never) |

### Occlusion-aware target

| Field | Meaning |
|---|---|
| `occlusion_visible_in_<h>s` | **tri-state**: `true` / `false` / `null` (unknown) |
| `occlusion_first_visible_time_<h>s` | seconds from `t` until first *confirmed* visibility (`null` unless `true`) |
| `occlusion_max_coverage_<h>s` | largest coverage among confirmed-visible observations |
| `occlusion_required_resolution_<h>s` | estimated resolution needed at first confirmed visibility |
| `occlusion_unknown_in_<h>s` | whether the window contained any unknown occlusion state |

### Compatibility labels (Phase 2, unchanged)

`visible_in_<h>s`, `first_visible_time_<h>s`, `max_coverage_<h>s`,
`required_resolution_<h>s`, `required_resolution_max_<h>s`, `occlusion_visible_<h>s`,
`resolution_insufficient_<h>s`.

`visible_in_<h>s` is **byte-identical to `frustum_visible_in_<h>s`** - it is the same quantity
under the Phase 2 name, kept so that anything built on the Phase 2 dataset keeps working. The
validator asserts the two agree on every row.

Note the deliberate naming distinction: `occlusion_visible_<h>s` is the Phase 2 *reading* at
first visibility (it can be `null`), whereas `occlusion_visible_in_<h>s` is the Phase 2.5
*target*. They are different fields with different meanings.

These columns cost about **72 MB (27% of the label bytes)** in `run_full_p25`. Pass
`--drop-legacy-labels` to omit them if nothing downstream needs the Phase 2 names.

### Per-row object state

`frustum_visible`, `approx_visible`, `occlusion_visible`, `coverage`, `required_resolution`,
`distance`, `category`, `current_resolution`, `resolution_eligible`.

## Resolution eligibility masks

Visibility prediction is **not** restricted: both targets are computed for every candidate
object, using all **17 texture identities**.

Resolution prediction is restricted, because only textures that actually have generated
variants can be scored against a resolution target:

| Texture family | Original | Variants | Eligible for resolution loss |
|---|---|---|---|
| `ambientcg/*` (walls, plaza slabs, showcase cubes) | 2048 | 256, 512, 1024, 2048 | **yes** (7 identities) |
| `kenney/*_colormap` (roads, buildings, props) | 512 | none | no (3 identities) |
| `quaternius/*` (vegetation) | 1024 | none | no (7 identities) |

Two explicit mask fields are written per row:

| Field | Meaning |
|---|---|
| `resolution_eligible` | `true` only if at least one of the object's texture identities has generated variants |
| `resolution_variants` | the resolutions a streaming system could actually choose (present only when eligible) |
| `variant_texture_ids` | which of the object's texture identities are variant-capable (present only when eligible) |

In `run_full_p25`, **30,469 of 188,469 rows (16.2%)** are resolution-eligible. Excluding the
other 158,000 rows from a resolution loss is a filter, not a data loss: their visibility
targets remain fully usable.

## Candidates

A row is emitted for every object whose axis-aligned bounding box is within
`--candidate-distance` metres of the camera (default **25 m**). Distance is measured from the
camera to the *box*, not to its centre, so a large building counts as close when you are
standing against it.

### Why 25 m, and why the raw capture uses 40 m

The raw capture records observations out to `--max-distance` (default **40 m**). A candidate's
look-ahead must stay inside the radius the raw capture actually recorded, otherwise its future
visibility would be invisible to the labeller.

The worst case is the camera moving directly away at maximum sprint speed:

```
candidate radius + max speed x max horizon  =  25 m + 12 m/s x 1.0 s  =  37 m  <  40 m
```

so a 40 m observation radius covers a 25 m candidate radius with margin. The validator checks
this relationship and warns if the margin drops below 12 m.

## How required resolution is estimated

**This is an estimate, not ground truth.** The scene has no real texture-mip ground truth to
measure against, so the value is derived from screen-space coverage:

```
covered_px  = coverage * viewport_width * viewport_height     # px^2 of viewport covered
required_px = sqrt(covered_px)                                # equivalent edge length in px
required_resolution = smallest step on the ladder >= required_px
```

The ladder is `64, 128, 256, 512, 1024, 2048`, so the result is always a power of two a texture
system could actually pick.

The reasoning is texel density: to render at roughly one texel per screen pixel, a surface
whose visible extent covers `N` pixels needs a texture about `N` texels across. `required_px`
is that `N`, derived from the covered area rather than from the mesh.

The assumptions are:

1. **Coverage is the projected AABB rectangle**, clipped to the viewport. It is a bounding
   rectangle, not the rasterised area, so it **overestimates** sparse or non-box meshes and
   therefore tends to overestimate the required resolution for them.
2. **One texel per pixel** is assumed, with a fronto-parallel surface. Neither viewing-angle
   stretch nor anisotropic filtering is modelled.
3. **The texture is assumed to map once across the object's on-screen extent.** For the
   Kenney and Quaternius atlases, which are shared across many objects, this is a heuristic
   rather than a fact.
4. `coverage == 0` yields `required_resolution == 0`, meaning "not on screen".

## Reading the labels

The processed output is split across two files per trajectory, joined on
`(trajectory_id, sample_index)`:

- `<id>.samples.jsonl` - one row per sample: camera pose, velocity, fov, viewport, seed
- `<id>.labels.jsonl` - one row per `(sample_index, object_id)`: object state + labels

```python
import pandas as pd

samples = pd.read_json("dataset/processed/run_full_p25/t01_walk_straight.samples.jsonl", lines=True)
labels = pd.read_json("dataset/processed/run_full_p25/t01_walk_straight.labels.jsonl", lines=True)
data = labels.merge(samples, on=["trajectory_id", "sample_index"], suffixes=("", "_sample"))

# frustum target: near, not in the frustum now, in the frustum within 1 s
frustum_new = data[~data.frustum_visible & data.frustum_visible_in_1_0s]

# occlusion target: only rows where the answer is actually known
known = data[data.occlusion_visible_in_1_0s.notna()]
occlusion_new = known[~known.frustum_visible & (known.occlusion_visible_in_1_0s == True)]

# resolution loss only where the mask allows it
scored = data[data.resolution_eligible]
```

Keeping the camera state out of the label rows removes ~250 duplicated bytes from every one of
~188k rows.

## What the labels are not

- **Not proof that neural prediction works.** Counts like "2,221 objects newly visible at
  0.5 s" are dataset diagnostics that describe the scene and the trajectory set. They say
  nothing about whether a model can predict them.
- **Not occlusion ground truth.** The occlusion signal is a small number of rays against box
  approximations, not the renderer's own answer. Roughly two thirds of frustum-visible
  observations are occluded, which is why the two targets differ - but the occlusion target
  itself carries a `null` bucket.
- **Not a demand signal for a specific mip chain.** They are resolution estimates from screen
  coverage.
- **Not memory-budget aware.** Making a prefetch decision under a budget is the next phase's
  job.
