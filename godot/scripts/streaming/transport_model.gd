class_name TransportModel
extends RefCounted
## Transport model - an exact mirror of the simulator's bandwidth/latency handling
## (sim/simulate.py:239-255).
##
## A request becomes eligible `latency_seconds` after it was first made, and each
## step offers a shared pool of `bandwidth_bytes_per_s * dt` bytes, consumed in
## priority order. There is no real disk timing here on purpose: the point is to
## reproduce the simulator's numbers, not to measure the SSD.

var bandwidth_bytes_per_s := 0.0
var latency_seconds := 0.0
var dt := 0.1


func _init(p_bandwidth_bytes_per_s := 0.0, p_latency_seconds := 0.0, p_dt := 0.1) -> void:
	bandwidth_bytes_per_s = p_bandwidth_bytes_per_s
	latency_seconds = p_latency_seconds
	dt = p_dt


## Bytes available to spend this step.
func step_budget() -> float:
	return bandwidth_bytes_per_s * dt


## True when a request made at `requested_sample` may start streaming at `sample`.
## Mirrors `t - since >= latency - 1e-9` with t = sample * dt.
func is_eligible(sample: int, requested_sample: int) -> bool:
	return float(sample - requested_sample) * dt >= latency_seconds - 1e-9
