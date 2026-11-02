# Asset sources and licensing

## Repository

| | |
|---|---|
| Name | FPS Buildings Environment Kit |
| URL | https://github.com/petroulacl/fps-buildings-env-kit |
| Commit | `1dd8e932402deb82422920df2e7fd079391e55d3` (2026-07-03) |
| Clone | `assets_src/fps-buildings-env-kit/` (full repository, 621 MB, 1997 files) |
| Licence | **CC0 1.0 Public Domain Dedication** for every bundled pack |

The repository README and the `License.txt` file inside each pack both state CC0:
*"You can use this content for personal, educational, and commercial purposes"*, with
attribution optional. The following `License.txt` files were read during setup:

- `buildings/kenney-city-kit-suburban/License.txt`
- `buildings/kenney-modular-buildings/License.txt`
- `props/kenney-city-kit-roads/License.txt`

The full clone is kept untouched in `assets_src/` as the provenance record; nothing in it is
modified. Only copies are imported into `godot/assets/`.

## Packs used in the project

| Pack | Upstream | Licence | Copied into |
|---|---|---|---|
| Kenney City Kit Suburban (2.0) | kenney.nl | CC0 | `godot/assets/models/kenney_city_suburban/` |
| Kenney Modular Buildings (2.1) | kenney.nl | CC0 | `godot/assets/models/kenney_modular_buildings/` |
| Kenney City Kit Roads (2.0) | kenney.nl | CC0 | `godot/assets/models/kenney_city_roads/` |
| Quaternius Stylized Nature MegaKit | quaternius.com | CC0 | `godot/assets/models/vegetation/` (14 models) |
| Poly Haven HDRIs | polyhaven.com | CC0 | `godot/assets/hdri/` (4 HDRIs, 2K) |
| ambientCG PBR textures | ambientcg.com | CC0 | `godot/assets/textures/ground/` (7 packs) |

Formats imported: `.glb` for all models (Godot imports glTF natively) and `.hdr` for the
skyboxes. The FBX/OBJ copies that ship alongside the GLB files were left in `assets_src/` and
not imported.

### ambientCG packs

`Asphalt021`, `Bricks066`, `Concrete012`, `Grass004`, `Ground037`, `Metal063`, `Road001` -
2048x2048 JPG maps (Color, NormalGL, Roughness, plus AmbientOcclusion / Metalness /
Displacement where the pack provides them). The DirectX-convention normal maps were skipped
(Godot uses the OpenGL convention).

### Poly Haven HDRIs

`urban_street_01_2k.hdr`, `urban_alley_01_2k.hdr`, `kloppenheim_02_puresky_2k.hdr`,
`kloppenheim_06_puresky_2k.hdr`. `urban_street_01_2k.hdr` is the one wired up as the sky in
`scenes/environment.tscn`; the others are available to swap in.

## Downloaded but not used

Downloaded as part of the full repository clone, but not imported into the Godot project:

- `military/kenney-desert-shooter-pack/` - 2D sprites and sound effects, not relevant to a 3D
  texture-streaming prototype.
- `buildings/kenney-tiny-town/` - 2D tilesets.
- `environment/vegetation/polypizza-stylized-nature-gltf/` - the remaining Quaternius models
  beyond the 14 that are used.

They remain available in `assets_src/` if a later phase wants them.
