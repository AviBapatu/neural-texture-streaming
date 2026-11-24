#!/usr/bin/env bash
#
# Phase 5 presenter demo - one command, no arguments needed.
#
#   ./run_demo.sh
#
# Default: borderless fullscreen dual-pane, t06_look_around (a held-out test trajectory
# the MLP did not train on), preset cell (budget 100%, 200 MB/s, 1.0 s latency), replaying
# the 5.6-8.7 s segment at 0.25x - 1.5 s either side of the largest pop-in event, where
# reactive_visible has 3 low-resolution objects on screen and mlp_b tau0.5 has 0.
# On a portrait window the two panes stack vertically instead of side by side.
#
# Env overrides: DEMO_PANES=1 (single-pane fallback), DEMO_TRAJECTORY=, DEMO_GODOT=,
# DEMO_QUALITY=high (effects on), DEMO_PANE_SCALE=2 (half-res panes), DEMO_NO_OFFLOAD=1,
# DEMO_WINDOWED=1 (windowed 1600x900 instead of fullscreen).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GODOT_PROJECT="$HERE/godot"

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

PANES="${DEMO_PANES:-2}"
TRAJECTORY="${DEMO_TRAJECTORY:-t06_look_around}"
QUALITY="${DEMO_QUALITY:-low}"
PANE_SCALE="${DEMO_PANE_SCALE:-1}"

# Discrete-GPU offload. On this machine Godot already selects the NVIDIA MX550, but the
# PRIME variables are set so an integrated-GPU default is overridden too.
if [[ "${DEMO_NO_OFFLOAD:-0}" != "1" ]]; then
	export __NV_PRIME_RENDER_OFFLOAD=1
	export __GLX_VENDOR_LIBRARY_NAME=nvidia
	export __VK_LAYER_NV_optimus=NVIDIA_only
	export DRI_PRIME=1
fi

printf '\n\033[1mPhase 5 presenter demo\033[0m\n'
if [[ "${DEMO_WINDOWED:-0}" == "1" ]]; then
	printf '  window      : windowed 1600x900 (panes: %s, pane scale: 1/%s)\n' "$PANES" "$PANE_SCALE"
else
	printf '  window      : borderless fullscreen (panes: %s, pane scale: 1/%s)\n' "$PANES" "$PANE_SCALE"
fi
printf '  preset cell : budget 100%%, 200 MB/s, latency 1.0 s\n'
printf '  trajectory  : %s  (held out - the MLP did not train on it)\n\n' "$TRAJECTORY"
printf '  \033[1mKEYS\033[0m\n'
printf '    SPACE ......... pause / resume the replay\n'
printf '    R ............. restart the replay\n'
printf '    S ............. slow motion 0.25x (default on for the 3 s pop-in window)\n'
printf '    H ............. toggle the key-help overlay\n'
printf '    F1 ............ tint overlay (low-res red / loading yellow / resident green / wasted purple)\n'
printf '    X ............. toggle the swap-outline flash\n'
printf '    [  ] .......... latency -/+ 0.1 s\n'
printf '    -  = .......... budget -/+ 50%%\n'
printf '    T ............. summary table (both policies)\n'
printf '    1 / 2 / 3 ..... presets: reactive vs mlp tau0 | heuristic tau0.5 vs mlp tau0.5 | heuristic vs mlp tau0\n'
printf '    A ............. auto-tour (~90 s, captions)\n'
printf '    Q / W ......... cycle LEFT pane policy\n'
printf '    O / P ......... cycle RIGHT pane policy\n'
printf '    Esc ........... quit\n\n'

MODE_ARGS=(--fullscreen)
if [[ "${DEMO_WINDOWED:-0}" == "1" ]]; then MODE_ARGS=(--windowed); fi

ARGS=(--path "$GODOT_PROJECT" --resolution 1600x900)
ARGS+=(res://scenes/demo_streaming.tscn --)
ARGS+=("${MODE_ARGS[@]}" --panes="$PANES" --trajectory="$TRAJECTORY" --quality="$QUALITY" --pane-scale="$PANE_SCALE")

cd "$HERE" || exit 1
exec "$GODOT_BIN" "${ARGS[@]}"
