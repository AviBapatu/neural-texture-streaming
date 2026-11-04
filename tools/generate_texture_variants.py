#!/usr/bin/env python3
"""Generate lower-resolution variants of the source textures.

Originals are only ever *read*: every variant is written to a separate output
tree and described in a JSON manifest that maps each original texture to the
resolution variants that could be derived from it.

A variant is only produced when the original actually permits it, i.e. when the
target resolution is strictly smaller than the longest edge of the original.
Target resolutions are interpreted as the length of the *longest* edge, so the
aspect ratio of the source is always preserved.

Downsampling uses an exact box filter (``Image.reduce``) whenever the source can
be divided by an integer factor, and a Lanczos filter otherwise. The originals
are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("Pillow is required: python3 -m pip install --user Pillow", file=sys.stderr)
    raise SystemExit(2)

DEFAULT_RESOLUTIONS = (2048, 1024, 512, 256)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
FORMAT_BY_SUFFIX = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
}


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create lower-resolution variants of texture maps for the "
                    "neural texture-streaming experiments.",
    )
    parser.add_argument("--source", default="godot/assets/textures/ground",
                        help="directory containing the original textures")
    parser.add_argument("--output", default="dataset/textures/variants",
                        help="directory the variants are written to")
    parser.add_argument("--manifest", default="dataset/textures/manifest.json",
                        help="where the JSON manifest is written")
    parser.add_argument("--resolutions", default=",".join(str(r) for r in DEFAULT_RESOLUTIONS),
                        help="comma separated target resolutions (longest edge)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG/WEBP quality for the generated variants")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate variants that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be generated without writing files")
    return parser.parse_args(argv)


def parse_resolutions(text: str) -> list[int]:
    resolutions: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError(f"resolution must be positive, got {value}")
        resolutions.append(value)
    if not resolutions:
        raise ValueError("at least one resolution is required")
    return sorted(set(resolutions), reverse=True)


def find_source_images(source_root: Path) -> list[Path]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_root}")
    images: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return images


def plan_variants(width: int, height: int, resolutions: list[int]) -> tuple[list[dict], list[dict]]:
    longest = max(width, height)
    plans: list[dict] = []
    skipped: list[dict] = []
    for target in resolutions:
        if target >= longest:
            skipped.append({
                "resolution": target,
                "reason": f"original longest edge is {longest}px - no downscale possible",
            })
            continue
        if longest % target == 0:
            factor = longest // target
            new_width = width // factor
            new_height = height // factor
            method = f"box-reduce x{factor}"
        else:
            scale = target / longest
            factor = None
            new_width = max(1, round(width * scale))
            new_height = max(1, round(height * scale))
            method = "lanczos"
        plans.append({
            "resolution": target,
            "width": new_width,
            "height": new_height,
            "factor": factor,
            "method": method,
        })
    return plans, skipped


def downsample(image: "Image.Image", plan: dict) -> "Image.Image":
    factor = plan["factor"]
    if factor is not None and factor > 1:
        return image.reduce(factor)
    return image.resize((plan["width"], plan["height"]), Image.Resampling.LANCZOS)


def save_variant(image: "Image.Image", destination: Path, source_suffix: str,
                 quality: int, icc_profile: bytes | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_format = FORMAT_BY_SUFFIX.get(source_suffix, "PNG")
    save_options: dict = {}
    if icc_profile:
        save_options["icc_profile"] = icc_profile
    if image_format == "JPEG":
        save_options.update(quality=quality, subsampling=0, optimize=True)
    elif image_format == "WEBP":
        save_options.update(quality=quality, method=6)
    elif image_format == "PNG":
        save_options.update(optimize=True)
    image.save(destination, format=image_format, **save_options)


def process_texture(image_path: Path, source_root: Path, output_root: Path,
                    resolutions: list[int], quality: int, overwrite: bool,
                    dry_run: bool) -> dict:
    relative = image_path.relative_to(source_root)
    entry: dict = {
        "source": str(image_path).replace("\\", "/"),
        "relative_path": str(relative).replace("\\", "/"),
        "variants": [],
        "skipped": [],
    }
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            entry.update({
                "width": width,
                "height": height,
                "mode": image.mode,
                "format": image.format,
                "bytes": image_path.stat().st_size,
            })
            icc_profile = image.info.get("icc_profile")
            plans, skipped = plan_variants(width, height, resolutions)
            entry["skipped"] = skipped
            for plan in plans:
                variant_relative = relative.with_name(
                    f"{relative.stem}_{plan['resolution']}{relative.suffix.lower()}"
                )
                destination = output_root / variant_relative
                record = {
                    "resolution": plan["resolution"],
                    "width": plan["width"],
                    "height": plan["height"],
                    "method": plan["method"],
                    "path": str(destination).replace("\\", "/"),
                    "bytes": destination.stat().st_size if destination.is_file() else 0,
                }
                if not dry_run:
                    if destination.is_file() and not overwrite:
                        record["reused"] = True
                    else:
                        resized = downsample(image, plan)
                        save_variant(resized, destination, relative.suffix.lower(), quality, icc_profile)
                        record["bytes"] = destination.stat().st_size
                entry["variants"].append(record)
    except Exception as error:  # noqa: BLE001 - report and continue with the next texture
        entry["error"] = str(error)
    return entry


def main(argv: list[str]) -> int:
    arguments = parse_arguments(argv)
    resolutions = parse_resolutions(arguments.resolutions)
    source_root = Path(arguments.source)
    output_root = Path(arguments.output)
    manifest_path = Path(arguments.manifest)

    images = find_source_images(source_root)
    if not images:
        print(f"No images found under {source_root}", file=sys.stderr)
        return 1

    print(f"Source      : {source_root} ({len(images)} images)")
    print(f"Output      : {output_root}")
    print(f"Manifest    : {manifest_path}")
    print(f"Resolutions : {', '.join(str(r) for r in resolutions)}")
    if arguments.dry_run:
        print("Mode        : dry run (nothing is written)")

    textures = []
    variant_count = 0
    for index, image_path in enumerate(images, start=1):
        entry = process_texture(
            image_path, source_root, output_root, resolutions,
            arguments.quality, arguments.overwrite, arguments.dry_run,
        )
        variant_count += len(entry["variants"])
        textures.append(entry)
        status = "error" if "error" in entry else f"{len(entry['variants'])} variants"
        print(f"  [{index:>3}/{len(images)}] {entry['relative_path']} -> {status}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(source_root).replace("\\", "/"),
        "output_root": str(output_root).replace("\\", "/"),
        "target_resolutions": resolutions,
        "resolution_semantics": "target is the longest edge of the variant; aspect ratio is preserved",
        "downsampling": {
            "integer_factors": "PIL Image.reduce (exact box filter)",
            "other_factors": "PIL Image.resize with LANCZOS",
        },
        "jpeg_quality": arguments.quality,
        "dry_run": arguments.dry_run,
        "summary": {
            "originals": len(textures),
            "variants": variant_count,
            "errors": sum(1 for t in textures if "error" in t),
        },
        "textures": textures,
    }

    if not arguments.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote manifest: {manifest_path}")

    print(f"Originals: {manifest['summary']['originals']}, "
          f"variants: {variant_count}, errors: {manifest['summary']['errors']}")
    return 0 if manifest["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
