# Environment

The playable environment is a compact city block generated at runtime by
`godot/scripts/environment_manager.gd`. Generation is **deterministic**: it is driven by the
`generation_seed` export on the manager node (default `20260501`), so every run produces the
same layout and any dataset recorded from it is reproducible.

## Layout

| Constant | Value | Meaning |
|---|---|---|
| `TILE` | 4.0 | world size of one Kenney road tile |
| `GRID_LINES` | `[-20, 0, 20]` | road centre lines on both axes |
| `EXTENT` | 40.0 | half-size of the city |
| `CITY_SCALE` | 4.0 | scale applied to the unit-sized Kenney models |
| `ROAD_CLEARANCE` | 3.4 | gap kept between a road edge and the nearest building |
| `BUILD_SPACING` | 7.5 | grid pitch used to fill a block |

Three road lines on each axis produce a **4x4 grid of 16 blocks** inside an 80x80 unit area.
Every block is assigned a kind:

- `built` - filled with buildings on a jittered grid
- `plaza` - a large PBR-textured slab (asphalt / road / concrete / ground)
- `park` - vegetation
- `gallery` - reserved for the seven-cube material showcase; this is the block closest to the
  player spawn

Roads are laid tile-by-tile with `road-crossroad` at intersections and `road-straight`
elsewhere. A walled perimeter (with gaps where the roads exit) surrounds the whole city, and
each `built` block gets a low brick yard wall facing the street.

## Object inventory

The default seed produces **268 tracked objects**:

| Category | Count |
|---|---|
| road tiles | 117 |
| buildings | 44 |
| walls | 42 |
| props (street lights, planters, trees, cones) | 54 |
| material showcase cubes | 7 |
| plaza slabs | 4 |

`godot/tests/environment_report.gd` prints this breakdown plus a top-down ASCII map of the
layout.

## Materials

Models from the Kenney kits use their own embedded `colormap` texture. On top of that, seven
`StandardMaterial3D` resources in `godot/assets/materials/` wrap the ambientCG PBR sets
(grass, asphalt, road, concrete, brick, metal, ground). They use **triplanar projection** with
a scale of 1 tile per ~4 world units, so the same material can be shared by the ground plane,
walls and cubes of different sizes without stretching.

The sun is a single shadow-casting `DirectionalLight3D`; the sky is a `PanoramaSkyMaterial`
using `urban_street_01_2k.hdr`, which also drives ambient light. The environment uses ACES
tonemapping, SSAO and a mild glow.

## Collision

| Object | Collider |
|---|---|
| Ground | one large box, top surface at `y = 0` |
| Buildings | box fitted to the model AABB |
| Walls, showcase cubes | box |
| Trees, street lights | cylinder around the trunk |
| Road tiles, planters, grass/pebbles | none (visual only) |

Road tiles are intentionally non-colliding: they are only 8 cm thick and would otherwise be
unwalkable steps. The player walks on the ground plane at `y = 0`, which the road tiles sit
just above.

## Regenerating with a different layout

Set `generation_seed` or `building_density` on the `Environment` node in
`scenes/environment.tscn` (or from code) and the whole city changes. Block kinds are drawn
from the same seeded generator, so the gallery block moves with the seed.
