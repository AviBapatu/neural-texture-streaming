# Texture variants

`tools/generate_texture_variants.py` derives lower-resolution versions of the source textures
so the future streaming experiments have real multi-resolution data to switch between.

## Usage

```bash
python3 -m pip install --user Pillow          # one-off dependency

python3 tools/generate_texture_variants.py    # defaults, run from the repository root
python3 tools/generate_texture_variants.py --dry-run     # report only
python3 tools/generate_texture_variants.py --output godot/assets/textures/variants
python3 tools/generate_texture_variants.py --resolutions 1024,512,256 --quality 92
```

| Option | Default | Meaning |
|---|---|---|
| `--source` | `godot/assets/textures/ground` | directory scanned for originals |
| `--output` | `dataset/textures/variants` | where variants are written |
| `--manifest` | `dataset/textures/manifest.json` | JSON manifest path |
| `--resolutions` | `2048,1024,512,256` | target resolutions (longest edge) |
| `--quality` | `95` | JPEG/WEBP quality |
| `--overwrite` | off | regenerate variants that already exist |
| `--dry-run` | off | report what would be generated, write nothing |

## Behaviour

- Originals are **only read**, never modified or replaced.
- Originals stay in place; variants go to a completely separate tree that mirrors the source
  folder structure, with the resolution appended to the file stem:
  `Bricks066/Bricks066_2K-JPG_Color.jpg` -> `Bricks066/Bricks066_2K-JPG_Color_1024.jpg`.
- A variant is only produced when the original actually permits it - a target equal to or
  larger than the original's longest edge is refused and recorded under `skipped`. Because the
  ambientCG maps are 2048x2048, the `2048` target is skipped for every one of them and the
  `1024`, `512` and `256` variants are generated.
- Aspect ratio is always preserved. Target resolutions are interpreted as the length of the
  *longest* edge.
- Downsampling uses an exact box filter (`PIL.Image.reduce`) when the source divides by an
  integer factor, and Lanczos (`PIL.Image.Resampling.LANCZOS`) otherwise. Both are high
  quality; the box filter is the standard choice for building mip chains.
- Output format follows the input format; JPEG is written at 4:4:4 chroma (no subsampling)
  with the requested quality so the normal maps degrade as little as possible.

## Current output

33 originals -> **99 variants** (26.3 MB), 0 errors.

```
dataset/textures/variants/<Pack>/<name>_1024.jpg
dataset/textures/variants/<Pack>/<name>_512.jpg
dataset/textures/variants/<Pack>/<name>_256.jpg
```

## Manifest

`dataset/textures/manifest.json`:

```json
{
  "generated_at": "2026-11-05T20:12:56+00:00",
  "source_root": "godot/assets/textures/ground",
  "output_root": "dataset/textures/variants",
  "target_resolutions": [2048, 1024, 512, 256],
  "resolution_semantics": "target is the longest edge of the variant; aspect ratio is preserved",
  "downsampling": { "integer_factors": "PIL Image.reduce (exact box filter)",
                    "other_factors": "PIL Image.resize with LANCZOS" },
  "jpeg_quality": 95,
  "summary": { "originals": 33, "variants": 99, "errors": 0 },
  "textures": [
    {
      "source": "godot/assets/textures/ground/Bricks066/Bricks066_2K-JPG_Color.jpg",
      "relative_path": "Bricks066/Bricks066_2K-JPG_Color.jpg",
      "width": 2048, "height": 2048, "mode": "RGB", "format": "JPEG", "bytes": 3401234,
      "variants": [
        { "resolution": 1024, "width": 1024, "height": 1024,
          "method": "box-reduce x2", "path": "dataset/textures/variants/..._1024.jpg", "bytes": 812345 },
        { "resolution": 512, "...": "..." },
        { "resolution": 256, "...": "..." }
      ],
      "skipped": [
        { "resolution": 2048, "reason": "original longest edge is 2048px - no downscale possible" }
      ]
    }
  ]
}
```

The manifest is the authoritative mapping from each original texture to its variants, so a
later phase can build the texture streamer without re-deriving anything.

## Validating the output

```bash
python3 tools/validate_texture_variants.py
```

Checks that every listed variant exists, matches its recorded dimensions, is strictly smaller
than its original and keeps the original's aspect ratio, and that every original is still
present and unmodified in size.
