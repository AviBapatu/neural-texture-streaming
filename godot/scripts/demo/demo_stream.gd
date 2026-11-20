class_name DemoStream
extends RefCounted
## Per-pane streaming metrics core for the Phase 5 presenter demo.
##
## This is the render-free half of a demo pane: it replays one trajectory's recorded
## per-sample signals through [TextureStreamer.StreamState] under a selectable policy
## and exposes the live metrics. The visible half ([DemoPane]) shares the same
## [member state] instance so what the HUD reports is exactly what drives the swaps.
##
## Visibility is the recorded per-(trajectory, sample, object) signal set from
## `dataset/streaming/runs/phase5_engine/scores/<trajectory>.json` - the same signals
## the Phase 5 parity test replays, which is why the demo reproduces the simulator to
## 4.3e-7 rather than to raycast noise. The streamer, transport, residency, eviction and
## metrics are the live engine code.

const SAMPLE_INTERVAL := 0.1
const SNAP_WINDOW := 10

## The six selectable policies, in cycle order. `neural` marks the precomputed-MLP
## policies that the HUD labels "NEURAL (precomputed scores)".
const SPECS := [
	{"id": "reactive_visible", "base": "reactive_visible", "tau": 0.0,
		"label": "reactive_visible", "neural": false},
	{"id": "heuristic_occ_tau0", "base": "heuristic_occ", "tau": 0.0,
		"label": "heuristic_occ tau=0", "neural": false},
	{"id": "heuristic_occ_tau0.5", "base": "heuristic_occ", "tau": 0.5,
		"label": "heuristic_occ tau=0.5", "neural": false},
	{"id": "mlp_b_tau0", "base": "mlp_b", "tau": 0.0,
		"label": "mlp_b tau=0", "neural": true},
	{"id": "mlp_b_tau0.5", "base": "mlp_b", "tau": 0.5,
		"label": "mlp_b tau=0.5", "neural": true},
	{"id": "no_streaming", "base": "no_streaming", "tau": 0.0,
		"label": "no_streaming", "neural": false},
]

static func spec(policy_id: String) -> Dictionary:
	for s in SPECS:
		if String(s["id"]) == policy_id:
			return s
	return SPECS[0]

static func policy_ids() -> PackedStringArray:
	var out := PackedStringArray()
	for s in SPECS:
		out.append(String(s["id"]))
	return out

var policy_id := "reactive_visible"
var policy_base := "reactive_visible"
var tau := 0.0
var neural := false
var trajectory := ""

var samples: Array = []
var rows_by_sample: Dictionary = {}
var pose: Dictionary = {}
var excluded: Dictionary = {}
var sizes: Dictionary = {}
var budget_bytes := 1.0

var state: TextureStreamer.StreamState
var transport: TransportModel
var index := -1

func setup(p_trajectory: String, p_samples: Array, p_rows: Dictionary, p_pose: Dictionary,
		p_snap: Array, p_sizes: Dictionary, p_budget: float, p_bandwidth: float,
		p_latency: float, p_policy_id: String) -> void:
	trajectory = p_trajectory
	samples = p_samples
	rows_by_sample = p_rows
	pose = p_pose
	sizes = p_sizes
	budget_bytes = p_budget
	transport = TransportModel.new(p_bandwidth, p_latency, SAMPLE_INTERVAL)
	excluded.clear()
	for s in p_snap:
		for offset in range(-SNAP_WINDOW, SNAP_WINDOW + 1):
			excluded[int(s) + offset] = true
	set_policy(p_policy_id)
	reset()

func set_policy(p_id: String) -> void:
	var s := spec(p_id)
	policy_id = String(s["id"])
	policy_base = String(s["base"])
	tau = float(s["tau"])
	neural = bool(s["neural"])

func set_budget(fraction: float, peak_bytes: float) -> void:
	budget_bytes = maxf(peak_bytes * fraction, 1.0)

func set_transport(bandwidth_bytes_per_s: float, latency_seconds: float) -> void:
	if transport == null:
		transport = TransportModel.new(bandwidth_bytes_per_s, latency_seconds, SAMPLE_INTERVAL)
	else:
		transport.bandwidth_bytes_per_s = bandwidth_bytes_per_s
		transport.latency_seconds = latency_seconds

## All-zero priorities (the no-prefetch reference).
func is_reactive_only() -> bool:
	return policy_base == "reactive_visible" or policy_base == "no_streaming"

func reset() -> void:
	state = TextureStreamer.StreamState.new()
	index = -1

func sample_count() -> int:
	return samples.size()

func current_sample() -> int:
	return index

func pose_at(i: int) -> Dictionary:
	if i < 0 or i >= samples.size():
		return {}
	return pose.get(str(int(samples[i])), {})

## Advance one sample. Returns false when the trajectory is finished.
func advance(i: int, on_load := Callable(), on_evict := Callable()) -> bool:
	if i < 0 or i >= samples.size():
		return false
	index = i
	var sample := int(samples[i])
	var list: Array = rows_by_sample.get(str(sample), [])
	var row_oid := PackedStringArray()
	var row_cov := PackedFloat64Array()
	var row_vis := PackedByteArray()
	var row_occ := PackedInt32Array()
	for r in list:
		row_oid.append(String(r[0]))
		row_cov.append(float(r[1]))
		var occ := int(r[3])
		row_occ.append(occ)
		row_vis.append(1 if (int(r[2]) == 1 and occ != 0) else 0)
	var priorities := priorities_for(list, row_occ, row_vis)
	var rows := PackedInt32Array()
	for j in row_oid.size():
		rows.append(j)
	var scored := not excluded.has(sample)
	state.step(sample, scored, rows, priorities, sizes, row_vis, row_cov, row_oid,
		transport, budget_bytes, on_load, on_evict)
	return true

## Priorities for one sample's rows, matching sim/policies.py exactly.
##
##   visible now            -> 2.0
##   heuristic_occ          -> squash(raw margin score)
##   mlp_b                  -> max(occlusion_0_5, occlusion_1_0)
##   tau > 0                -> 0 unless the score is strictly greater than tau
##   reactive_visible       -> 0 when not visible
func priorities_for(list: Array, row_occ: PackedInt32Array,
		row_vis: PackedByteArray) -> PackedFloat64Array:
	var priorities := PackedFloat64Array()
	priorities.resize(list.size())
	if policy_base == "no_streaming":
		return priorities
	for i in list.size():
		var visible_now := row_vis[i] == 1
		var prio := 0.0
		if visible_now:
			prio = 2.0
		match policy_base:
			"reactive_visible":
				pass
			"heuristic_occ":
				if not visible_now:
					prio = _threshold(HeuristicOcc.squash(float(list[i][4])), tau)
			"mlp_b":
				if not visible_now:
					prio = _threshold(maxf(float(list[i][5]), float(list[i][6])), tau)
			_:
				prio = 0.0
		priorities[i] = prio
	return priorities

static func _threshold(score: float, tau_value: float) -> float:
	if tau_value <= 0.0:
		return score
	return score if score > tau_value else 0.0

func summarise() -> Dictionary:
	return state.metrics.summarise(budget_bytes)

func resident_bytes() -> float:
	return state.resident_bytes

func resident_count() -> int:
	return state.resident.size()

func deficit() -> float:
	var s := state.metrics.summarise(budget_bytes)
	return float(s["deficit"])

func pop_in_events() -> int:
	return state.metrics.pop_in_events

func bytes_streamed() -> float:
	return state.metrics.bytes_streamed

func wasted_prefetches() -> int:
	return state.metrics.wasted_prefetches

## Objects visible now (ground truth) whose full texture is not resident - i.e. on
## screen right now at the low-resolution placeholder. Returns [{oid, coverage}].
func visible_popin() -> Array:
	var out: Array = []
	if index < 0:
		return out
	var list: Array = rows_by_sample.get(str(int(samples[index])), [])
	for r in list:
		if int(r[2]) == 1 and int(r[3]) != 0 and not state.resident.has(String(r[0])):
			out.append({"oid": String(r[0]), "coverage": float(r[1])})
	return out

## Number of ground-truth-visible objects at the current sample.
func visible_count() -> int:
	if index < 0:
		return 0
	var n := 0
	for r in rows_by_sample.get(str(int(samples[index])), []):
		if int(r[2]) == 1 and int(r[3]) != 0:
			n += 1
	return n
