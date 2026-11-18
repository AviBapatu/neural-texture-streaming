class_name TextureStreamer
extends RefCounted
## Faithful GDScript port of sim/simulate.py `simulate()` + `summarise()`.
##
## Two-level residency: a low-resolution placeholder is always resident and free;
## an object becomes "good" only once its full texture is resident. A shared
## per-step bandwidth pool serves requests in priority order, a request becomes
## eligible `latency` after it is first made, and residency is capped at the budget
## with lowest-priority-first eviction.
##
## Ground truth is the project's convention: visible = frustum_visible AND
## occlusion_state != definitively-false, with unknown counting as visible.
##
## `StreamState` is the stateful core (one trajectory); `simulate()` is a loop over
## it so the offline comparison and the live engine share one implementation.


## Stateful streaming core. One instance per trajectory (or per live session).
class StreamState:
	extends RefCounted

	var resident := {}
	var resident_bytes := 0.0
	var loaded := {}
	var requested_at := {}
	var ever_visible := {}
	var episode_open := {}
	var extended := {}
	var metrics := StreamingMetrics.new()

	## Advance one sample. `rows` is the row indices belonging to this sample.
	func step(sample: int, scored: bool, rows: PackedInt32Array, priorities: PackedFloat64Array,
			sizes: Dictionary, ground_truth: PackedByteArray, coverage: PackedFloat64Array,
			object_ids: PackedStringArray, transport: TransportModel, budget_bytes: float,
			on_load := Callable(), on_evict := Callable()) -> void:
		metrics.steps += 1
		if scored:
			metrics.scored_steps += 1

		var object_priority := {}
		for row in rows:
			var oid: String = object_ids[row]
			var value: float = priorities[row]
			if value > float(object_priority.get(oid, 0.0)):
				object_priority[oid] = value

		for oid in object_priority:
			if float(object_priority[oid]) <= 0.0 or resident.has(oid):
				continue
			if not requested_at.has(oid):
				requested_at[oid] = sample

		var available := transport.step_budget()
		var eligible: Array = []
		for oid in requested_at:
			if resident.has(oid):
				continue
			if transport.is_eligible(sample, int(requested_at[oid])):
				eligible.append(oid)
		eligible.sort_custom(_eligible_before.bind(object_priority, requested_at))

		var completing: Array = []
		for oid in eligible:
			if available <= 0.0:
				break
			var need := float(sizes.get(oid, 0.0)) - float(loaded.get(oid, 0.0))
			if need <= 0.0:
				completing.append(oid)
				continue
			var take := minf(need, available)
			loaded[oid] = float(loaded.get(oid, 0.0)) + take
			metrics.bytes_streamed += take
			available -= take
			if float(sizes.get(oid, 0.0)) - float(loaded[oid]) <= 1e-9:
				completing.append(oid)

		for oid in completing:
			var size := float(sizes.get(oid, 0.0))
			if size > budget_bytes:
				loaded.erase(oid)
				requested_at.erase(oid)
				continue
			if resident_bytes + size > budget_bytes:
				var evictable: Array = []
				for o in resident:
					if float(object_priority.get(o, 0.0)) < float(object_priority.get(oid, 0.0)):
						evictable.append(o)
				evictable.sort_custom(_evictable_before.bind(object_priority, requested_at))
				for victim in evictable:
					if resident_bytes + size <= budget_bytes:
						break
					_evict(victim, on_evict)
			if resident_bytes + size <= budget_bytes:
				resident[oid] = size
				resident_bytes += size
				if not ever_visible.has(oid):
					ever_visible[oid] = false
				loaded.erase(oid)
				requested_at.erase(oid)
				metrics.loads_completed += 1
				if on_load.is_valid():
					on_load.call(oid)
			else:
				loaded.erase(oid)
				requested_at.erase(oid)

		metrics.peak_resident_bytes = maxf(metrics.peak_resident_bytes, resident_bytes)

		extended.clear()
		for row in rows:
			if ground_truth[row] == 0:
				continue
			var oid: String = object_ids[row]
			var covered := coverage[row]
			if scored:
				metrics.visible_coverage += covered
			if resident.has(oid):
				ever_visible[oid] = true
				if episode_open.has(oid):
					metrics.pop_in_durations.append(int(episode_open[oid]))
					episode_open.erase(oid)
			else:
				if scored:
					metrics.deficit += covered
				extended[oid] = true
				if not episode_open.has(oid):
					metrics.pop_in_events += 1
					episode_open[oid] = 0
				episode_open[oid] = int(episode_open[oid]) + 1

		var closed: Array = []
		for oid in episode_open:
			if not extended.has(oid):
				closed.append(oid)
		for oid in closed:
			metrics.pop_in_durations.append(int(episode_open[oid]))
			episode_open.erase(oid)

	func _evict(victim: String, on_evict: Callable) -> void:
		resident_bytes -= float(resident.get(victim, 0.0))
		resident.erase(victim)
		if not bool(ever_visible.get(victim, false)):
			metrics.wasted_prefetches += 1
		requested_at.erase(victim)
		if on_evict.is_valid():
			on_evict.call(victim)

	## Sort key for eligible requests: (-priority, requested_at, object_id).
	func _eligible_before(a, b, object_priority: Dictionary, requested_at: Dictionary) -> bool:
		var pa := float(object_priority.get(a, 0.0))
		var pb := float(object_priority.get(b, 0.0))
		if pa != pb:
			return pa > pb
		var ra := int(requested_at.get(a, 0))
		var rb := int(requested_at.get(b, 0))
		if ra != rb:
			return ra < rb
		return String(a) < String(b)

	## Sort key for eviction victims: (priority, requested_at, object_id).
	func _evictable_before(a, b, object_priority: Dictionary, requested_at: Dictionary) -> bool:
		var pa := float(object_priority.get(a, 0.0))
		var pb := float(object_priority.get(b, 0.0))
		if pa != pb:
			return pa < pb
		var ra := int(requested_at.get(a, 0))
		var rb := int(requested_at.get(b, 0))
		if ra != rb:
			return ra < rb
		return String(a) < String(b)

	## Close open episodes and evict everything (end of trajectory).
	func finish(budget_bytes: float) -> Dictionary:
		for oid in episode_open:
			metrics.pop_in_durations.append(int(episode_open[oid]))
		episode_open.clear()
		for oid in resident.keys():
			if not bool(ever_visible.get(oid, false)):
				metrics.wasted_prefetches += 1
		resident.clear()
		return metrics.summarise(budget_bytes)

	func is_resident(oid: String) -> bool:
		return resident.has(oid)


## Replay one trajectory's rows under one policy (offline parity path).
func simulate(samples: PackedInt32Array, rows_by_sample: Dictionary, excluded: Dictionary,
		priorities: PackedFloat64Array, sizes: Dictionary, ground_truth: PackedByteArray,
		coverage: PackedFloat64Array, object_ids: PackedStringArray,
		bandwidth_bytes_per_s: float, latency_seconds: float, budget_bytes: float, dt: float,
		on_load := Callable(), on_evict := Callable()) -> Dictionary:
	var transport := TransportModel.new(bandwidth_bytes_per_s, latency_seconds, dt)
	var state := StreamState.new()
	for sample in samples:
		state.step(sample, not excluded.has(sample),
			rows_by_sample.get(sample, PackedInt32Array()), priorities, sizes,
			ground_truth, coverage, object_ids, transport, budget_bytes, on_load, on_evict)
	return state.finish(budget_bytes)

