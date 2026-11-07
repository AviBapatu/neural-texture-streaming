#!/usr/bin/env bash
#
# End-to-end dataset generation.
#
#   tools/run_dataset_pipeline.sh --profile full
#   tools/run_dataset_pipeline.sh --profile quick --run-id run_quick
#
# Steps, in order:
#   1. import Godot assets
#   2. export the object index            -> dataset/manifests/object_index.json
#   3. build the texture index            -> dataset/manifests/texture_index.json
#   4. generate camera trajectories       -> dataset/trajectories/<profile>/
#   5. assign train/validation/test       -> dataset/manifests/splits.json
#   6. record raw telemetry               -> dataset/raw/<run-id>/
#   7. compute future-demand labels       -> dataset/processed/<run-id>/
#   8. validate and report                -> dataset/reports/<run-id>-validation.md
#
# Environment variables:
#   GODOT              path to the Godot 4.7 binary
#   SAMPLE_INTERVAL    physics ticks between samples (default 6 -> 10 Hz)
#   OCCLUSION          1 to enable the occlusion raycasts (default 1)
#   MAX_DISTANCE       candidate observation distance in metres (default 60)
#   CANDIDATE_DISTANCE labelling candidate distance in metres (default 35)
#   HORIZONS           prediction horizons in seconds (default 0.5,1.0)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GODOT_PROJECT="$PROJECT_ROOT/godot"
cd "$PROJECT_ROOT" || exit 1

PROFILE="full"
RUN_ID=""
OCCLUSION="${OCCLUSION:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-6}"
MAX_DISTANCE="${MAX_DISTANCE:-40}"
CANDIDATE_DISTANCE="${CANDIDATE_DISTANCE:-25}"
HORIZONS="${HORIZONS:-0.5,1.0}"
# Existing run whose trajectory-level split assignment must be preserved exactly.
REFERENCE_RUN="${REFERENCE_RUN:-run_full}"
SKIP_IMPORT=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--profile) PROFILE="$2"; shift 2 ;;
		--run-id) RUN_ID="$2"; shift 2 ;;
		--occlusion) OCCLUSION="$2"; shift 2 ;;
		--skip-import) SKIP_IMPORT=1; shift ;;
		-h|--help) sed -n '2,25p' "$0"; exit 0 ;;
		*) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
	esac
done
[[ -z "$RUN_ID" ]] && RUN_ID="run_${PROFILE}"

resolve_godot() {
	if [[ -n "${GODOT:-}" ]]; then printf '%s' "$GODOT"; return; fi
	local name
	for name in godot godot4 godot4.7 Godot; do
		if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return; fi
	done
	local candidate
	for candidate in "$HOME/.local/bin/godot" \
		"$HOME/.local/opt/godot/"Godot_v*-stable_linux.x86_64; do
		if [[ -x "$candidate" ]]; then printf '%s' "$candidate"; return; fi
	done
	printf ''
}

GODOT_BIN="$(resolve_godot)"
status=0
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
fail() { printf 'FAILED: %s\n' "$1"; status=1; }

if [[ -z "$GODOT_BIN" ]]; then
	printf 'Godot 4.7 was not found. Set GODOT=/path/to/godot.\n' >&2
	exit 127
fi

printf 'run id     : %s\n' "$RUN_ID"
printf 'profile    : %s\n' "$PROFILE"
printf 'godot      : %s\n' "$GODOT_BIN"
printf 'occlusion  : %s\n' "$OCCLUSION"

if [[ $SKIP_IMPORT -eq 0 ]]; then
	step "1/8 import assets"
	"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --import >/tmp/nts_pipeline_import.log 2>&1 \
		|| fail "asset import"
	grep -qE "SCRIPT ERROR|Parse Error" /tmp/nts_pipeline_import.log \
		&& { grep -E "SCRIPT ERROR|Parse Error" /tmp/nts_pipeline_import.log; fail "script errors"; } \
		|| printf 'no script errors\n'
fi

step "2/8 export object index"
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --script res://tools/export_object_index.gd \
	2>&1 | sed 's/^/  /' || fail "object index export"

step "3/8 build texture index"
python3 tools/build_texture_index.py 2>&1 | tail -4 | sed 's/^/  /' || fail "texture index"

step "4/8 generate trajectories ($PROFILE)"
python3 tools/generate_trajectories.py --profile "$PROFILE" 2>&1 | tail -3 | sed 's/^/  /' \
	|| fail "trajectory generation"

step "5/8 assign train/validation/test splits"
python3 tools/split_dataset.py --profile "$PROFILE" --run-id "$RUN_ID" 2>&1 | head -5 | sed 's/^/  /' \
	|| fail "split"

if [[ "$RUN_ID" != "$REFERENCE_RUN" && -f "dataset/manifests/runs/$REFERENCE_RUN/splits.json" ]]; then
	printf '\n  checking the assignment against %s\n' "$REFERENCE_RUN"
	python3 -c "
import json, sys
first = json.load(open('dataset/manifests/runs/$REFERENCE_RUN/splits.json'))
second = json.load(open('dataset/manifests/runs/$RUN_ID/splits.json'))
strip = lambda m: {k: [e['trajectory_id'] for e in v] for k, v in m['splits'].items()}

if first['profile'] != second['profile']:
    # A run over a different trajectory profile is expected to have a different
    # assignment; there is nothing to preserve between them.
    print('  different profiles (%s vs %s) - nothing to compare'
          % (first['profile'], second['profile']))
    sys.exit(0)

same = strip(first) == strip(second)
print('  split assignment identical to $REFERENCE_RUN:', same)
if not same:
    print('    reference:', strip(first))
    print('    new      :', strip(second))
sys.exit(0 if same else 1)
" || fail "trajectory split assignment changed for the same profile"
fi

step "6/8 record raw telemetry"
rm -rf "dataset/raw/$RUN_ID"
"$GODOT_BIN" --headless --fixed-fps 60 --path "$GODOT_PROJECT" \
	--script res://tools/run_trajectories.gd -- \
	"--trajectories=dataset/trajectories/$PROFILE/index.json" \
	"--output=dataset/raw/$RUN_ID" \
	"--run-id=$RUN_ID" \
	"--occlusion=$OCCLUSION" \
	"--sample-interval=$SAMPLE_INTERVAL" \
	"--max-distance=$MAX_DISTANCE" 2>&1 | tail -5 | sed 's/^/  /' || fail "trajectory recording"

step "7/8 compute future-demand labels"
rm -rf "dataset/processed/$RUN_ID"
python3 tools/label_future_demand.py --run "$RUN_ID" \
	--horizons "$HORIZONS" --candidate-distance "$CANDIDATE_DISTANCE" 2>&1 | tail -6 | sed 's/^/  /' \
	|| fail "labelling"

step "8/8 validate dataset"
python3 tools/validate_dataset.py --run "$RUN_ID" --horizons "$HORIZONS" \
	--candidate-distance "$CANDIDATE_DISTANCE" 2>&1 | tail -6 | sed 's/^/  /' \
	|| fail "validation"

printf '\n\033[1m=== %s ===\033[0m\n' "$([[ $status -eq 0 ]] && echo 'PIPELINE OK' || echo 'PIPELINE FAILED')"
"$GODOT_BIN" --version >/dev/null 2>&1
printf 'raw       : dataset/raw/%s\n' "$RUN_ID"
printf 'processed : dataset/processed/%s\n' "$RUN_ID"
printf 'report    : dataset/reports/%s-validation.md\n' "$RUN_ID"
du -sh "dataset/raw/$RUN_ID" "dataset/processed/$RUN_ID" 2>/dev/null
exit $status
