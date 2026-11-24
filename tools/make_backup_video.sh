#!/usr/bin/env bash
#
# Phase 5 demo - backup video of the auto-tour.
#
# Records the presenter demo running its A auto-tour to PNG frames and encodes them to
# an MP4 with ffmpeg, so there is a recorded fallback if the live demo fails.
#
#   tools/make_backup_video.sh [output.mp4]
#
# The demo runs with --fixed-fps so the recording is deterministic (the tour advances a
# fixed 1/fps per frame regardless of how slowly PNG capture runs), and the encoded
# video length matches the tour length exactly.
#
# Environment:
#   DEMO_GODOT      path to the Godot 4.7 binary
#   BACKUP_SECONDS  tour seconds to record (default 120)
#   BACKUP_FPS      capture/encode fps (default 24)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GODOT_PROJECT="$HERE/godot"
OUT="${1:-$HERE/dataset/streaming/runs/demo/backup_tour.mp4}"
SECONDS_TO_RECORD="${BACKUP_SECONDS:-120}"
FPS="${BACKUP_FPS:-24}"
FRAMES="$(mktemp -d "${TMPDIR:-/tmp}/nts_demo_frames.XXXXXX")"

resolve_godot() {
	if [[ -n "${DEMO_GODOT:-}" ]]; then printf '%s' "$DEMO_GODOT"; return; fi
	if [[ -n "${GODOT:-}" ]]; then printf '%s' "$GODOT"; return; fi
	local name
	for name in godot godot4 godot4.7 Godot; do
		if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return; fi
	done
	local candidate
	for candidate in \
		"$HOME/.local/bin/godot" \
		"$HOME/.local/opt/godot/"Godot_v*-stable_linux.x86_64; do
		if [[ -x "$candidate" ]]; then printf '%s' "$candidate"; return; fi
	done
	printf ''
}

GODOT_BIN="$(resolve_godot)"
if [[ -z "$GODOT_BIN" ]]; then
	printf 'Godot 4.7 was not found. Set DEMO_GODOT=/path/to/godot and re-run.\n' >&2
	exit 127
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
	printf 'ffmpeg was not found on PATH.\n' >&2
	exit 127
fi

# Encoders to try, best first. Capability is probed by actually encoding, because e.g.
# h264_nvenc can be listed but fail to open (no NVENC access under the NVK driver).
# The list is captured up front because `grep -q` on a live ffmpeg pipe trips `pipefail`.
ENCODER_LIST="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
ENCODER_CANDIDATES="libx264 h264_nvenc libopenh264 h264_vaapi mpeg4"

TOTAL_FRAMES=$((SECONDS_TO_RECORD * FPS))
printf 'Recording %ds of auto-tour at %sfps (%d frames) -> %s\n' \
	"$SECONDS_TO_RECORD" "$FPS" "$TOTAL_FRAMES" "$OUT"

mkdir -p "$(dirname "$OUT")"
cd "$HERE" || exit 1

"$GODOT_BIN" --path "$GODOT_PROJECT" --fullscreen --fixed-fps "$FPS" \
	res://scenes/demo_streaming.tscn -- \
	--fullscreen --panes=2 --tour --quality=low --record="$FRAMES" --record-every=1 \
	--exit-after-frames="$TOTAL_FRAMES" || {
		printf 'Godot recording failed.\n' >&2
		rm -rf "$FRAMES"
		exit 1
	}

COUNT="$(find "$FRAMES" -name 'frame_*.png' | wc -l)"
if [[ "$COUNT" -lt 2 ]]; then
	printf 'No frames were captured.\n' >&2
	rm -rf "$FRAMES"
	exit 1
fi

ENCODE_OK=0
for E in $ENCODER_CANDIDATES; do
	grep -qE "[[:space:]]${E}([[:space:]]|$)" <<< "$ENCODER_LIST" || continue
	printf 'trying encoder: %s ... ' "$E"
	if [[ "$E" == "mpeg4" ]]; then
		ffmpeg -y -loglevel error -framerate "$FPS" -i "$FRAMES/frame_%05d.png" \
			-c:v mpeg4 -qscale:v 3 -pix_fmt yuv420p "$OUT" >/dev/null 2>&1
	else
		ffmpeg -y -loglevel error -framerate "$FPS" -i "$FRAMES/frame_%05d.png" \
			-c:v "$E" -crf 23 -pix_fmt yuv420p "$OUT" >/dev/null 2>&1
	fi
	if [[ -s "$OUT" ]]; then
		ENCODER="$E"
		ENCODE_OK=1
		printf 'ok\n'
		break
	fi
	rm -f "$OUT"
	printf 'failed\n'
done
if [[ "$ENCODE_OK" != "1" ]]; then
	printf 'ffmpeg encoding failed with every candidate encoder.\n' >&2
	rm -rf "$FRAMES"
	exit 1
fi

rm -rf "$FRAMES"
printf 'Wrote %s (%s frames, %.1fs)\n' "$OUT" "$COUNT" "$(echo "$COUNT $FPS" | awk '{print $1/$2}')"
