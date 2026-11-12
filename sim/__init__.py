"""Phase 4a: an offline, trace-driven texture-streaming simulator.

Replays the recorded telemetry through a simple two-level residency model and
compares streaming policies on visible texture deficit under a memory budget.

Nothing here touches Godot, the datasets or the manifests. It reads the frozen
per-row scores produced by ``training/save_oof_scores.py`` and the labels that
already exist on disk.
"""
