#!/usr/bin/env bash
#
# End-to-end validation of the environment and the dataset pipeline.
#
#   godot/tests/run_tests.sh
#
# Environment variables:
#   GODOT   path to the Godot 4.7 binary (defaults to `godot` on PATH, then to a
#           couple of common install locations)
#   FRAMES  frames for the Phase 1 headless run (default 360)
#   PROFILE dataset profile used for the Phase 2 pipeline check (default quick)
#
# Exits non-zero if any step fails. Phase 1 steps are unchanged.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GODOT_PROJECT="$PROJECT_ROOT/godot"
cd "$PROJECT_ROOT" || exit 1

resolve_godot() {
	if [[ -n "${GODOT:-}" ]]; then
		printf '%s' "$GODOT"
		return
	fi
	local name
	for name in godot godot4 godot4.7 Godot; do
		if command -v "$name" >/dev/null 2>&1; then
			command -v "$name"
			return
		fi
	done
	local candidate
	for candidate in \
		"$HOME/.local/bin/godot" \
		"$HOME/.local/opt/godot/"Godot_v*-stable_linux.x86_64; do
		if [[ -x "$candidate" ]]; then
			printf '%s' "$candidate"
			return
		fi
	done
	printf ''
}

GODOT_BIN="$(resolve_godot)"
FRAMES="${FRAMES:-360}"
PROFILE="${PROFILE:-quick}"
status=0
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
fail() { printf 'FAILED: %s\n' "$1"; status=1; }

step "Godot binary"
if [[ -z "$GODOT_BIN" ]]; then
	printf 'Godot 4.7 was not found. Set GODOT=/path/to/godot and re-run.\n' >&2
	exit 127
fi
printf 'using: %s\n' "$GODOT_BIN"
"$GODOT_BIN" --version || fail "godot --version"

# ---------------------------------------------------------------- Phase 1 ----

step "1/12 import assets"
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --import >/tmp/nts_import.log 2>&1 || fail "asset import"
if grep -qE "SCRIPT ERROR|Parse Error" /tmp/nts_import.log; then
	grep -E "SCRIPT ERROR|Parse Error" /tmp/nts_import.log
	fail "script errors during import"
else
	printf 'no script errors during import\n'
fi

step "2/12 environment report (materials, models, layout)"
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --script res://tests/environment_report.gd \
	|| fail "environment report"

step "3/12 first-person controller (movement, reset, collision)"
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --script res://tests/movement_test.gd \
	|| fail "controller test"

step "4/12 run main scene headless ($FRAMES frames)"
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --quit-after "$FRAMES" 2>&1 \
	| tee /tmp/nts_run.log | grep -E "EnvironmentManager|TelemetryRecorder|ERROR" || true
if grep -qE "SCRIPT ERROR|Parse Error" /tmp/nts_run.log; then
	fail "script errors while running the main scene"
fi

step "5/12 validate Phase 1 telemetry"
python3 tools/validate_telemetry.py dataset/telemetry || fail "telemetry validation"

step "6/12 validate texture variants"
python3 tools/validate_texture_variants.py || fail "texture variant validation"

# ---------------------------------------------------------------- Phase 2 ----

step "7/12 object identity determinism"
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --script res://tests/object_identity_test.gd \
	|| fail "object identity test"

step "8/12 object index reproducible across processes"
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --script res://tools/export_object_index.gd \
	>/dev/null 2>&1 || fail "object index export (first)"
cp dataset/manifests/object_index.json /tmp/nts_index_a.json
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" --script res://tools/export_object_index.gd \
	>/dev/null 2>&1 || fail "object index export (second)"
python3 -c "
import json, sys
first = json.load(open('/tmp/nts_index_a.json'))
second = json.load(open('dataset/manifests/object_index.json'))
ids_a = [o['object_id'] for o in first['objects']]
ids_b = [o['object_id'] for o in second['objects']]
same = ids_a == ids_b and first['generation_hash'] == second['generation_hash']
print('  %d ids, generation hash %s, identical across processes: %s'
      % (len(ids_b), second['generation_hash'], same))
sys.exit(0 if same else 1)
" || fail "object index is not reproducible across processes"

step "9/12 dataset pipeline ($PROFILE profile) incl. validation"
GODOT="$GODOT_BIN" tools/run_dataset_pipeline.sh --profile "$PROFILE" --skip-import | tail -12 \
	|| fail "dataset pipeline"

step "10/12 trajectory replay determinism"
"$GODOT_BIN" --headless --fixed-fps 60 --path "$GODOT_PROJECT" \
	--script res://tests/trajectory_test.gd || fail "trajectory determinism test"

step "11/12 raw capture byte-identical across processes"
DETERMINISM_DIR="dataset/raw/_determinism_check"
rm -rf "$DETERMINISM_DIR"
"$GODOT_BIN" --headless --fixed-fps 60 --path "$GODOT_PROJECT" \
	--script res://tools/run_trajectories.gd -- \
	"--trajectories=dataset/trajectories/$PROFILE/index.json" \
	"--output=$DETERMINISM_DIR" --occlusion=1 --max-distance=40 \
	--run-id=_determinism_check >/dev/null 2>&1 || fail "determinism re-run"
python3 -c "
import glob, json, os, sys
ok, checked = True, 0
for path in sorted(glob.glob('dataset/raw/run_${PROFILE}/*.jsonl')):
    name = os.path.basename(path)
    other = os.path.join('$DETERMINISM_DIR', name)
    if not os.path.isfile(other):
        print('  missing', other); ok = False; continue
    first = [json.loads(l) for l in open(path)]
    second = [json.loads(l) for l in open(other)]
    for row in first: row.pop('timestamp')
    for row in second: row.pop('timestamp')
    checked += 1
    if first != second:
        print('  differs:', name); ok = False
print('  %d trajectories byte-identical (wall-clock timestamps excluded): %s' % (checked, ok))
sys.exit(0 if ok and checked else 1)
" || fail "raw capture is not reproducible across processes"
rm -rf "$DETERMINISM_DIR"

step "12/12 occlusion overhead benchmark"
"$GODOT_BIN" --headless --path "$GODOT_PROJECT" \
	--script res://tools/benchmark_visibility.gd || fail "occlusion benchmark"

# ---------------------------------------------------------------- Phase 5 ----

P5_DIR="dataset/streaming/runs/phase5_engine"
P5_GODOT="--headless --path $GODOT_PROJECT"

step "Phase 5 / 13: engine streaming parity vs the simulator (53-object scope)"
python3 tools/export_scores_for_godot.py --out "$P5_DIR" >/dev/null 2>&1 \
	|| fail "phase 5 export"
python3 sim/run_sweep.py --config sim/config_5engine.yaml >/dev/null 2>&1 \
	|| fail "phase 5 simulator run"
"$GODOT_BIN" $P5_GODOT --script res://tests/streaming_parity_test.gd -- "--out=$P5_DIR" \
	>/tmp/nts_p5_parity.log 2>&1 || { tail -5 /tmp/nts_p5_parity.log; fail "phase 5 engine parity run"; }
grep -E "heuristic port" /tmp/nts_p5_parity.log || true
python3 tools/check_godot_parity.py \
	--engine "$P5_DIR/engine/engine_metrics.json" \
	--sim "$P5_DIR/sim_results/results.csv" | tail -9 || fail "phase 5 parity within tolerance"

step "Phase 5 / 14: streaming determinism (engine metrics reproducible)"
cp "$P5_DIR/engine/engine_metrics.json" /tmp/nts_p5_metrics_a.json
"$GODOT_BIN" $P5_GODOT --script res://tests/streaming_parity_test.gd -- "--out=$P5_DIR" \
	>/dev/null 2>&1 || fail "phase 5 determinism re-run"
python3 -c "
import json, sys
a = json.load(open('/tmp/nts_p5_metrics_a.json'))
b = json.load(open('$P5_DIR/engine/engine_metrics.json'))
print('  identical:', a == b)
sys.exit(0 if a == b else 1)
" || fail "engine metrics are not reproducible across runs"

step "Phase 5 / 15: MLP port vs training/models.py forward_numpy"
python3 tools/export_model_for_godot.py >/dev/null 2>&1 || fail "phase 5 model export"
"$GODOT_BIN" $P5_GODOT --script res://tests/mlp_parity_test.gd | grep -E "mlp_parity" \
	|| fail "MLP port does not match the reference forward pass"

step "Phase 5 / 16: presenter demo parity + determinism"
"$GODOT_BIN" $P5_GODOT --script res://tests/demo_parity_test.gd -- \
	"--out=dataset/streaming/runs/demo" >/tmp/nts_p5_demo.log 2>&1 \
	|| { tail -20 /tmp/nts_p5_demo.log; fail "presenter demo parity run"; }
grep -E "\[demo_parity\]" /tmp/nts_p5_demo.log || true
grep -q "\[demo_parity\] PASS" /tmp/nts_p5_demo.log \
	|| fail "presenter demo parity/determinism did not pass"

printf '\n\033[1m=== %s ===\033[0m\n' "$([[ $status -eq 0 ]] && echo 'ALL CHECKS PASSED' || echo 'CHECKS FAILED')"
exit $status
