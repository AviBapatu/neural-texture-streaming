#!/usr/bin/env bash
#
# Render the validation screenshots into docs/screenshots/.
#
#   tools/capture_screenshots.sh
#
# Uses the --capture* arguments understood by godot/scripts/screenshot_capture.gd.
# Requires a graphical session (DISPLAY / WAYLAND_DISPLAY). Override the Godot
# binary with GODOT=/path/to/godot.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GODOT_PROJECT="$PROJECT_ROOT/godot"
OUTPUT_DIR="$PROJECT_ROOT/docs/screenshots"
cd "$PROJECT_ROOT" || exit 1

GODOT_BIN="${GODOT:-}"
if [[ -z "$GODOT_BIN" ]]; then
	if command -v godot >/dev/null 2>&1; then
		GODOT_BIN="$(command -v godot)"
	else
		GODOT_BIN="$HOME/.local/bin/godot"
	fi
fi
if [[ ! -x "$GODOT_BIN" ]]; then
	printf 'Godot binary not found at %s. Set GODOT=/path/to/godot.\n' "$GODOT_BIN" >&2
	exit 127
fi

mkdir -p "$OUTPUT_DIR"
status=0

capture() {
	local name="$1" position="$2" yaw="$3" pitch="${4:-0}" frames="${5:-140}"
	printf 'capturing %-24s pos=%-14s yaw=%-5s pitch=%s\n' "$name" "$position" "$yaw" "$pitch"
	timeout 180 "$GODOT_BIN" --path "$GODOT_PROJECT" --resolution 1280x720 -- \
		"--capture=$OUTPUT_DIR/$name" \
		"--capture-frames=$frames" \
		"--capture-pos=$position" \
		"--capture-yaw=$yaw" \
		"--capture-pitch=$pitch" >/dev/null 2>&1 || status=1
}

capture 01_spawn.png          "0,0.3,-34"    180
capture 02_street_center.png  "0,0.3,-10"    180
capture 03_crossroads.png     "0,0.3,4"      135
capture 05_gallery.png        "-20,0.3,-30"  270
capture 06_skyline.png        "24,0.3,-10"   135
capture 07_aerial.png         "0,55,45"      180 -35 25
capture 08_gallery_close.png  "-19,7,-37"    226 -22 25

printf '\nscreenshots written to %s\n' "$OUTPUT_DIR"
exit $status
