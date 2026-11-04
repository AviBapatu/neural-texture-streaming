#!/usr/bin/env python3
"""Validate the texture-variant manifest against the files on disk.

Verifies that every variant listed in `dataset/textures/manifest.json` exists,
has the recorded dimensions, preserves the aspect ratio of its original, is
strictly smaller than the original, and that no original was overwritten.

Usage:
    python3 tools/validate_texture_variants.py
    python3 tools/validate_texture_variants.py --manifest dataset/textures/manifest.json
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="dataset/textures/manifest.json")
    parser.add_argument("--min-variants", type=int, default=1)
    arguments = parser.parse_args(argv)

    manifest_path = Path(arguments.manifest)
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    variants_checked = 0
    total_variant_bytes = 0

    for texture in manifest.get("textures", []):
        source = Path(texture["source"])
        relative = texture["relative_path"]
        if "error" in texture:
            problems.append(f"{relative}: manifest records an error: {texture['error']}")
            continue
        if not source.is_file():
            problems.append(f"{relative}: original missing at {source}")
        else:
            with Image.open(source) as image:
                if image.size != (texture["width"], texture["height"]):
                    problems.append(
                        f"{relative}: original is {image.size}, manifest says "
                        f"({texture['width']}, {texture['height']})"
                    )
        variants = texture.get("variants", [])
        if len(variants) < arguments.min_variants:
            problems.append(f"{relative}: only {len(variants)} variants")
        for variant in variants:
            variants_checked += 1
            path = Path(variant["path"])
            if not path.is_file():
                problems.append(f"{relative}: variant file missing at {path}")
                continue
            total_variant_bytes += path.stat().st_size
            with Image.open(path) as image:
                if image.size != (variant["width"], variant["height"]):
                    problems.append(
                        f"{path.name}: on disk {image.size}, manifest "
                        f"({variant['width']}, {variant['height']})"
                    )
            if variant["width"] >= texture["width"] or variant["height"] >= texture["height"]:
                problems.append(f"{path.name}: variant is not smaller than the original")
            original_ratio = texture["width"] / texture["height"]
            variant_ratio = variant["width"] / variant["height"]
            if abs(original_ratio - variant_ratio) > 1e-6:
                problems.append(
                    f"{path.name}: aspect ratio changed "
                    f"({original_ratio:.4f} -> {variant_ratio:.4f})"
                )

    summary = manifest.get("summary", {})
    print(f"manifest     : {manifest_path}")
    print(f"generated at : {manifest.get('generated_at')}")
    print(f"originals    : {summary.get('originals')} listed, "
          f"{len(manifest.get('textures', []))} processed")
    print(f"variants     : {variants_checked} verified "
          f"({total_variant_bytes / 1e6:.1f} MB)")
    print(f"resolutions  : {manifest.get('target_resolutions')}")

    if problems:
        print(f"\nINVALID - {len(problems)} problem(s):")
        for problem in problems[:40]:
            print(f"  - {problem}")
        return 1
    print("\nVALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
