class_name StreamingMetrics
extends RefCounted
## Metrics accumulator - an exact mirror of sim/simulate.py's StepMetrics and
## summarise() (sim/simulate.py:164-176, 323-345).

var deficit := 0.0
var visible_coverage := 0.0
var pop_in_events := 0
var pop_in_durations: Array = []
var wasted_prefetches := 0
var bytes_streamed := 0.0
var peak_resident_bytes := 0.0
var steps := 0
var scored_steps := 0
var loads_completed := 0


## Flatten to the metrics dict the simulator's summarise() produces.
## `budget_bytes` should be negative to omit the budget context.
func summarise(budget_bytes := -1.0) -> Dictionary:
	var durations: Array = pop_in_durations
	var mean_duration := 0.0
	if durations.size() > 0:
		var total := 0.0
		for d in durations:
			total += float(d)
		mean_duration = total / float(durations.size())
	var max_duration := 0
	for d in durations:
		max_duration = maxi(max_duration, int(d))
	var result := {
		"deficit": (deficit / visible_coverage) if visible_coverage > 0.0 else 0.0,
		"deficit_bytes_weighted": deficit,
		"visible_coverage": visible_coverage,
		"pop_in_events": pop_in_events,
		"pop_in_mean_duration_samples": mean_duration,
		"pop_in_max_duration_samples": max_duration,
		"wasted_prefetches": wasted_prefetches,
		"loads_completed": loads_completed,
		"bytes_streamed": bytes_streamed,
		"peak_resident_bytes": peak_resident_bytes,
		"steps": steps,
		"scored_steps": scored_steps,
	}
	if budget_bytes >= 0.0:
		result["budget_bytes"] = budget_bytes
		result["peak_over_budget"] = (peak_resident_bytes / budget_bytes) if budget_bytes > 0.0 else INF
	return result
