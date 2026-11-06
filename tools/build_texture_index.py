#!/usr/bin/env python3
"""Build the texture index and enrich the object index with texture details.

The object index produced by Godot records *which* texture identities each object
uses. This step resolves those identities to real files on disk, joins them with the
variant manifest produced in Phase 1, and writes:

    dataset/manifests/texture_index.json

It then enriches `dataset/manifests/object_index.json` in place so every object
carries the available resolutions of its textures (a union, plus the per-texture
detail), which is what the labelling pipeline needs to decide whether a required
resolution is actually obtainable.

Usage:
    python3 tools/build_texture_index.py
    python3 tools/build_texture_index.py --check          # verify only, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("Pillow is required: python3 -m pip install --user Pillow", file=sys.stderr)
    raise SystemExit(2)

DEFAULT_OBJECT_INDEX = "dataset/manifests/object_index.json"
DEFAULT_VARIANT_MANIFEST = "dataset/textures/manifest.json"
DEFAULT_OUTPUT = "dataset/manifests/texture_index.json"

GROUND_ROOT = "godot/assets/textures/ground"
MODEL_ROOT = "godot/assets/models"
VEGETATION_ROOT = "godot/assets/models/vegetation"


def resolve_source_path(texture_id: str, project_root: Path) -> Path:
    """Maps a stable texture identity back onto the file it came from."""
    parts = texture_id.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"unrecognised texture id: {texture_id}")
    family, name = parts

    if family == "ambientcg":
        # ambientcg/<Pack>_<Map> -> ground/<Pack>/<Pack>_2K-JPG_<Map>.jpg
        pack, _, map_name = name.partition("_")
        candidate = project_root / GROUND_ROOT / pack / f"{pack}_2K-JPG_{map_name}.jpg"
        return _require(candidate, texture_id)

    if family == "kenney":
        # kenney/<pack>_colormap -> models/kenney_<pack>/Textures/colormap.png
        pack = name.removesuffix("_colormap")
        candidate = project_root / MODEL_ROOT / f"kenney_{pack}" / "Textures" / "colormap.png"
        return _require(candidate, texture_id)

    if family == "quaternius":
        candidate = project_root / VEGETATION_ROOT / f"{name}.png"
        return _require(candidate, texture_id)

    raise ValueError(f"unrecognised texture family '{family}' in {texture_id}")


def _require(path: Path, texture_id: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{texture_id}: expected texture at {path}")
    return path


def load_variant_manifest(path: Path) -> dict[str, dict]:
    """Maps original source path -> {resolution: variant path}."""
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_source: dict[str, dict] = {}
    for entry in manifest.get("textures", []):
        variants = {variant["resolution"]: variant["path"] for variant in entry.get("variants", [])}
        by_source[entry["source"]] = variants
    return by_source


def classify(texture_id: str) -> str:
    if texture_id.startswith("ambientcg/"):
        return "pbr_albedo"
    if texture_id.startswith("kenney/"):
        return "atlas"
    if texture_id.startswith("quaternius/"):
        return "model_texture"
    return "unknown"


def image_size(path: Path) -> list[int]:
    with Image.open(path) as image:
        return [image.width, image.height]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--object-index", default=DEFAULT_OBJECT_INDEX)
    parser.add_argument("--variant-manifest", default=DEFAULT_VARIANT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="verify only, write nothing")
    arguments = parser.parse_args(argv)

    project_root = Path.cwd()
    object_index_path = Path(arguments.object_index)
    if not object_index_path.is_file():
        print(f"object index not found: {object_index_path}", file=sys.stderr)
        print("run: godot --headless --path godot --script res://tools/export_object_index.gd", file=sys.stderr)
        return 2

    object_index = json.loads(object_index_path.read_text(encoding="utf-8"))
    variants_by_source = load_variant_manifest(Path(arguments.variant_manifest))

    texture_ids: list[str] = sorted(set(object_index.get("texture_ids_used", [])))
    if not texture_ids:
        # Fall back to whatever the objects reference.
        texture_ids = sorted({t for o in object_index.get("objects", []) for t in o.get("texture_ids", [])})

    problems: list[str] = []
    textures: dict[str, dict] = {}
    for texture_id in texture_ids:
        try:
            resolved = resolve_source_path(texture_id, project_root)
        except (ValueError, FileNotFoundError) as error:
            problems.append(str(error))
            continue
        # Keys in the variant manifest are project-relative, so store ours the same way.
        try:
            source = resolved.relative_to(project_root)
        except ValueError:
            source = resolved
        source_key = str(source).replace("\\", "/")
        variants = variants_by_source.get(source_key, {})
        original = image_size(resolved)
        available = sorted({max(original), *variants.keys()})
        textures[texture_id] = {
            "texture_id": texture_id,
            "kind": classify(texture_id),
            "source_path": source_key,
            "original_resolution": original,
            "available_resolutions": available,
            "variant_paths": {str(resolution): path for resolution, path in sorted(variants.items())},
            "has_variants": bool(variants),
            "max_resolution": max(available),
        }
        print(f"  {texture_id:<44} {str(original):<12} variants={len(variants)} -> {available}")

    if problems:
        print(f"\n{len(problems)} texture id(s) could not be resolved:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    with_variants = sum(1 for t in textures.values() if t["has_variants"])
    print(f"\n{len(textures)} texture ids resolved, {with_variants} with generated variants")
    print(f"{len(textures) - with_variants} only exist at their original resolution")

    if arguments.check:
        print("check mode: nothing written")
        return 0

    index = {
        "schema_version": 1,
        "environment_seed": object_index.get("environment_seed"),
        "source_variant_manifest": str(arguments.variant_manifest).replace("\\", "/"),
        "texture_count": len(textures),
        "textures_with_variants": with_variants,
        "note": "available_resolutions is the union of the original resolution and the generated variants",
        "textures": textures,
    }
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")

    enrich_object_index(object_index_path, object_index, textures)
    return 0


def enrich_object_index(path: Path, object_index: dict, textures: dict[str, dict]) -> None:
    """Adds resolved texture detail to every object in the object index."""
    for entry in object_index.get("objects", []):
        details = []
        union: set[int] = set()
        for texture_id in entry.get("texture_ids", []):
            texture = textures.get(texture_id)
            if texture is None:
                continue
            details.append({
                "texture_id": texture_id,
                "source_path": texture["source_path"],
                "original_resolution": texture["original_resolution"],
                "available_resolutions": texture["available_resolutions"],
                "has_variants": texture["has_variants"],
            })
            union.update(texture["available_resolutions"])
        entry["textures"] = details
        entry["available_resolutions"] = sorted(union)
        entry["has_variants"] = any(detail["has_variants"] for detail in details)
    object_index["texture_index"] = "dataset/manifests/texture_index.json"
    path.write_text(json.dumps(object_index, indent=2) + "\n", encoding="utf-8")
    print(f"enriched {path} with per-object texture detail")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
