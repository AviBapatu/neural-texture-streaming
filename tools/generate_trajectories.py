#!/usr/bin/env python3
"""Generate the deterministic camera trajectories used to record the dataset.

Trajectories are plain JSON so they can be inspected, diffed and replayed by
`godot/scripts/trajectory_runner.gd`. Every value is derived from fixed constants
and the seeded object index - there is no run-time randomness anywhere.

Two segment modes are supported:

  control   open loop: an input action, a speed, and linear yaw/pitch ramps
  waypoint  closed loop: walk to a target point, turning towards it at a capped rate

Usage:
    python3 tools/generate_trajectories.py --profile full
    python3 tools/generate_trajectories.py --profile quick     # scaled to 40% duration
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROAD_LINES = (-20.0, 0.0, 20.0)
EXTENT = 40.0
SPAWN = (0.0, 0.2, -34.0)
DEFAULT_SEED = 20260501
DEFAULT_OBJECT_INDEX = "dataset/manifests/object_index.json"
OUTPUT_ROOT = "dataset/trajectories"

WALK_SPEED = 6.0
SLOW_SPEED = 3.0
SPRINT_SPEED = 6.0

QUICK_PROFILE = ("t01_walk_straight", "t06_look_around", "t09_approach_building")
QUICK_DURATION_SCALE = 0.4


# --- geometry helpers --------------------------------------------------------

def centre(bounds: dict) -> tuple[float, float, float]:
    position, size = bounds["position"], bounds["size"]
    return (position[0] + size[0] / 2.0, position[1] + size[1] / 2.0, position[2] + size[2] / 2.0)


def support_distance(bounds: dict, direction: tuple[float, float]) -> float:
    """Distance from the AABB centre to its surface along a unit XZ direction."""
    size = bounds["size"]
    return abs(direction[0]) * size[0] / 2.0 + abs(direction[1]) * size[2] / 2.0


def outside_point(bounds: dict, from_point: tuple[float, float, float], margin: float = 2.0) -> list[float]:
    """A point just outside `bounds`, on the side facing `from_point`."""
    cx, _, cz = centre(bounds)
    dx, dz = from_point[0] - cx, from_point[2] - cz
    length = math.hypot(dx, dz)
    if length < 1e-6:
        dx, dz, length = 0.0, 1.0, 1.0
    ux, uz = dx / length, dz / length
    reach = support_distance(bounds, (ux, uz))
    return [round(cx + ux * (reach + margin), 3), SPAWN[1], round(cz + uz * (reach + margin), 3)]


def horizontal_gap(bounds: dict, point: tuple[float, float, float]) -> float:
    cx, _, cz = centre(bounds)
    size = bounds["size"]
    dx = max(0.0, abs(point[0] - cx) - size[0] / 2.0)
    dz = max(0.0, abs(point[2] - cz) - size[2] / 2.0)
    return math.hypot(dx, dz)


def yaw_towards(origin: tuple[float, float, float], target: list[float]) -> float:
    """Player yaw (degrees in [0, 360)) that makes the camera face `target`."""
    return math.degrees(math.atan2(-(target[0] - origin[0]), -(target[2] - origin[2]))) % 360.0


# --- object index queries ----------------------------------------------------

def load_objects(path: Path) -> tuple[list[dict], int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"object index not found: {path}\n"
            "run: godot --headless --path godot --script res://tools/export_object_index.gd"
        )
    index = json.loads(path.read_text(encoding="utf-8"))
    return index["objects"], index["environment_seed"]


def nearest_of_category(objects: list[dict], category: str, point: tuple[float, float, float],
                        minimum_size: float = 0.0) -> dict | None:
    best, best_distance = None, math.inf
    for entry in objects:
        if entry["category"] != category:
            continue
        bounds = entry["world_bounds"]
        if min(bounds["size"][0], bounds["size"][2]) < minimum_size:
            continue
        cx, _, cz = centre(bounds)
        distance = math.hypot(cx - point[0], cz - point[2])
        if distance < best_distance:
            best, best_distance = entry, distance
    return best


def safe_point_on_plate(plate: dict, obstacles: list[dict], clearance: float = 6.0) -> list[float] | None:
    """Finds a spot on a plaza slab that is clear of every building."""
    bounds = plate["world_bounds"]
    cx, _, cz = centre(bounds)
    half_x, half_z = bounds["size"][0] / 2.0 - 2.0, bounds["size"][2] / 2.0 - 2.0
    if half_x <= 0.0 or half_z <= 0.0:
        return None
    buildings = [o for o in obstacles if o["category"] == "bld"]
    for step_x in range(-3, 4):
        for step_z in range(-3, 4):
            candidate = (cx + step_x * half_x / 3.0, SPAWN[1], cz + step_z * half_z / 3.0)
            if all(horizontal_gap(b["world_bounds"], candidate) >= clearance for b in buildings):
                return [round(candidate[0], 3), SPAWN[1], round(candidate[2], 3)]
    return None


def safe_plaza_point(objects: list[dict], near: tuple[float, float, float] = None) -> list[float] | None:
    """A clear spot on a plaza slab, preferring generous clearance then proximity.

    `near` biases the choice towards a plaza the camera can actually walk to in a
    straight line (plaza slabs are open, so a short straight approach is safe).
    """
    plates = sorted((o for o in objects if o["category"] == "plz"), key=lambda o: o["object_id"])
    for clearance in (6.0, 3.0):
        candidates = []
        for plate in plates:
            point = safe_point_on_plate(plate, objects, clearance)
            if point is None:
                continue
            if near is None:
                return point
            candidates.append((math.hypot(point[0] - near[0], point[2] - near[2]), point))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
    return None


def narrow_street_waypoints(objects: list[dict]) -> tuple[list[float], list[float]] | None:
    """Finds a road spot with buildings close on both sides, plus a point beyond it."""
    buildings = [o for o in objects if o["category"] == "bld"]
    for line in (0.0, -20.0, 20.0):
        for z in range(-30, 31, 4):
            here = (line, SPAWN[1], float(z))
            ahead = (line, SPAWN[1], float(z + 14))
            left = sum(1 for b in buildings if horizontal_gap(b["world_bounds"], (-12.0, 0.0, float(z))) < 8.0)
            right = sum(1 for b in buildings if horizontal_gap(b["world_bounds"], (12.0, 0.0, float(z))) < 8.0)
            near = sum(1 for b in buildings if horizontal_gap(b["world_bounds"], here) < 13.0
                       and horizontal_gap(b["world_bounds"], ahead) < 13.0)
            if left and right and near >= 2:
                return ([round(line, 3), SPAWN[1], float(z)], [round(line, 3), SPAWN[1], float(z + 14)])
    return None


# --- segment builders --------------------------------------------------------

def control(start: float, end: float, action: str, speed: float, yaw_from: float, yaw_to: float,
            pitch_from: float = 0.0, pitch_to: float = 0.0, sprint: bool = False) -> dict:
    return {
        "mode": "control", "start": round(start, 3), "end": round(end, 3),
        "action": action, "speed": speed, "sprint": sprint,
        "yaw_from": yaw_from, "yaw_to": yaw_to,
        "pitch_from": pitch_from, "pitch_to": pitch_to,
    }


def waypoint(start: float, end: float, speed: float, target: list[float],
             arrive_radius: float = 1.6, turn_rate: float = 150.0, pitch: float = 0.0,
             sprint: bool = False) -> dict:
    return {
        "mode": "waypoint", "start": round(start, 3), "end": round(end, 3),
        "action": "move_forward", "speed": speed, "sprint": sprint,
        "target": target, "arrive_radius": arrive_radius,
        "turn_rate_degrees_per_second": turn_rate, "pitch": pitch,
    }


def trajectory(identifier: str, kind: str, description: str, seed: int, duration: float,
               segments: list[dict], start: list[float] = None, yaw: float = 180.0,
               pitch: float = 0.0) -> dict:
    return {
        "schema_version": 1,
        "trajectory_id": identifier,
        "kind": kind,
        "description": description,
        "environment_seed": seed,
        "start_position": list(start) if start else list(SPAWN),
        "start_yaw_degrees": yaw,
        "start_pitch_degrees": pitch,
        "duration_seconds": round(duration, 3),
        "segments": segments,
    }


# --- the trajectory set ------------------------------------------------------

def build_trajectories(objects: list[dict], seed: int) -> list[dict]:
    results: list[dict] = []
    south = [0.0, SPAWN[1], -34.0]
    north = [0.0, SPAWN[1], 34.0]

    # 1. straight walking
    results.append(trajectory(
        "t01_walk_straight", "walk_straight",
        "Walk north along the central avenue from the spawn point.",
        seed, 18.0,
        [control(0.0, 18.0, "move_forward", WALK_SPEED, 180.0, 180.0)],
        start=south, yaw=180.0))

    # 2. straight sprinting
    results.append(trajectory(
        "t02_sprint_straight", "sprint_straight",
        "Sprint north along the central avenue using the sprint action.",
        seed, 16.0,
        [control(0.0, 16.0, "move_forward", SPRINT_SPEED, 180.0, 180.0, sprint=True)],
        start=[0.0, SPAWN[1], -38.0], yaw=180.0))

    # 3. walking around a corner
    results.append(trajectory(
        "t03_walk_corner", "walk_corner",
        "Walk north, then turn left at the intersection and continue east.",
        seed, 22.0,
        [control(0.0, 9.0, "move_forward", WALK_SPEED, 180.0, 180.0),
         control(9.0, 11.5, "move_forward", WALK_SPEED, 180.0, 270.0),
         control(11.5, 22.0, "move_forward", WALK_SPEED, 270.0, 270.0)],
        start=[0.0, SPAWN[1], -14.0], yaw=180.0))

    # 4. sprinting around a corner
    results.append(trajectory(
        "t04_sprint_corner", "sprint_corner",
        "Sprint north, turn right at the intersection at speed, continue east.",
        seed, 20.0,
        [control(0.0, 7.0, "move_forward", SPRINT_SPEED, 180.0, 180.0, sprint=True),
         control(7.0, 9.0, "move_forward", SPRINT_SPEED, 180.0, 90.0, sprint=True),
         control(9.0, 20.0, "move_forward", SPRINT_SPEED, 90.0, 90.0, sprint=True)],
        start=[0.0, SPAWN[1], -16.0], yaw=180.0))

    # 5. turning left and right while moving
    results.append(trajectory(
        "t05_turn_left_right", "turn_left_right",
        "Walk forward while sweeping the camera left and right across the street.",
        seed, 24.0,
        [control(0.0, 6.0, "move_forward", WALK_SPEED, 180.0, 90.0),
         control(6.0, 12.0, "move_forward", WALK_SPEED, 90.0, 270.0),
         control(12.0, 18.0, "move_forward", WALK_SPEED, 270.0, 180.0),
         control(18.0, 24.0, "move_forward", WALK_SPEED, 180.0, 180.0)],
        start=south, yaw=180.0))

    # 6. looking around while stationary
    results.append(trajectory(
        "t06_look_around", "look_around",
        "Stand still at the spawn point and pan a full 360 degrees.",
        seed, 20.0,
        [control(0.0, 5.0, "none", 0.0, 180.0, 270.0),
         control(5.0, 10.0, "none", 0.0, 270.0, 0.0),
         control(10.0, 15.0, "none", 0.0, 0.0, 90.0),
         control(15.0, 20.0, "none", 0.0, 90.0, 180.0)],
        start=south, yaw=180.0))

    # 7. stopping and restarting
    results.append(trajectory(
        "t07_stop_start", "stop_start",
        "Walk in bursts: forward, stop, forward, stop, forward.",
        seed, 22.0,
        [control(0.0, 4.0, "move_forward", WALK_SPEED, 180.0, 180.0),
         control(4.0, 7.0, "none", 0.0, 180.0, 180.0),
         control(7.0, 12.0, "move_forward", WALK_SPEED, 180.0, 190.0),
         control(12.0, 15.0, "none", 0.0, 190.0, 190.0),
         control(15.0, 22.0, "move_forward", WALK_SPEED, 190.0, 180.0)],
        start=south, yaw=180.0))

    # 8. moving forward and reversing
    results.append(trajectory(
        "t08_forward_reverse", "forward_reverse",
        "Walk forward, stop, reverse back down the avenue, then forward again.",
        seed, 20.0,
        [control(0.0, 6.0, "move_forward", WALK_SPEED, 180.0, 180.0),
         control(6.0, 7.0, "none", 0.0, 180.0, 180.0),
         control(7.0, 14.0, "move_back", WALK_SPEED, 180.0, 180.0),
         control(14.0, 15.0, "none", 0.0, 180.0, 180.0),
         control(15.0, 20.0, "move_forward", WALK_SPEED, 180.0, 200.0)],
        start=[0.0, SPAWN[1], -30.0], yaw=180.0))

    # 9. approaching a building
    building = nearest_of_category(objects, "bld", (0.0, 0.0, -18.0), minimum_size=5.0)
    if building is not None:
        look_from = (0.0, SPAWN[1], -30.0)
        target = outside_point(building["world_bounds"], look_from, margin=2.5)
        start = [target[0], SPAWN[1], target[2] - 14.0]
        results.append(trajectory(
            "t09_approach_building", "approach_building",
            f"Walk directly at {building['object_id']} and stop just in front of it.",
            seed, 18.0,
            [waypoint(0.0, 14.0, WALK_SPEED, target, arrive_radius=2.0, pitch=2.0),
             control(14.0, 18.0, "none", 0.0, 0.0, 0.0, 2.0, -6.0)],
            start=start, yaw=yaw_towards(tuple(start), target)))

    # 10. narrow street and open plaza
    street = narrow_street_waypoints(objects)
    safe_plaza = safe_plaza_point(objects, near=(0.0, SPAWN[1], -30.0))
    if street and safe_plaza:
        enter, beyond = street
        # Route stays on the road grid until the final turn into the open plaza, so
        # the straight-line waypoint steering never has to path around a building.
        plaza_gate = [0.0, SPAWN[1], -30.0]
        start = [0.0, SPAWN[1], -34.0]
        results.append(trajectory(
            "t10_street_then_plaza", "street_then_plaza",
            "Walk up a narrow street lined with buildings, double back, then turn west "
            "out of the street into an open plaza.",
            seed, 28.0,
            [waypoint(0.0, 10.0, WALK_SPEED, enter, arrive_radius=2.0),
             waypoint(10.0, 16.0, WALK_SPEED, plaza_gate, arrive_radius=2.0),
             waypoint(16.0, 28.0, SLOW_SPEED, safe_plaza, arrive_radius=2.0, turn_rate=120.0)],
            start=start, yaw=yaw_towards(tuple(start), enter)))

    # 11. slow walk looking down at the pavement
    results.append(trajectory(
        "t11_slow_walk_down", "walk_straight",
        "Slow walk with the camera angled down at the road surface.",
        seed, 18.0,
        [control(0.0, 18.0, "move_forward", SLOW_SPEED, 90.0, 90.0, -18.0, -18.0)],
        start=[20.0, SPAWN[1], -34.0], yaw=90.0, pitch=-18.0))

    # 12. sprint with continuous turning
    results.append(trajectory(
        "t12_sprint_weave", "sprint_corner",
        "Sprint along the avenue while weaving the camera from side to side.",
        seed, 20.0,
        [control(0.0, 5.0, "move_forward", SPRINT_SPEED, 150.0, 210.0, sprint=True),
         control(5.0, 10.0, "move_forward", SPRINT_SPEED, 210.0, 150.0, sprint=True),
         control(10.0, 15.0, "move_forward", SPRINT_SPEED, 150.0, 210.0, sprint=True),
         control(15.0, 20.0, "move_forward", SPRINT_SPEED, 210.0, 180.0, sprint=True)],
        start=[0.0, SPAWN[1], -36.0], yaw=150.0))

    # 13. approaching a street prop
    prop = nearest_of_category(objects, "prp", (-20.0, 0.0, -20.0))
    if prop is not None:
        target = outside_point(prop["world_bounds"], (-24.0, SPAWN[1], -20.0), margin=1.6)
        start = [target[0] - 9.0, SPAWN[1], target[2]]
        results.append(trajectory(
            "t13_approach_prop", "approach_prop",
            f"Walk up to {prop['object_id']} and inspect it from close range.",
            seed, 16.0,
            [waypoint(0.0, 12.0, SLOW_SPEED, target, arrive_radius=1.4, pitch=-4.0),
             control(12.0, 16.0, "none", 0.0, 0.0, 0.0, -4.0, 8.0)],
            start=start, yaw=yaw_towards(tuple(start), target)))

    # 14. plaza look-around
    if safe_plaza:
        start = [safe_plaza[0] + 7.0, SPAWN[1], safe_plaza[2] + 1.0]
        results.append(trajectory(
            "t14_plaza_look", "look_around",
            "Walk into an open plaza and pan around it from the middle.",
            seed, 22.0,
            [waypoint(0.0, 10.0, WALK_SPEED, safe_plaza, arrive_radius=2.5),
             control(10.0, 16.0, "none", 0.0, 0.0, 120.0),
             control(16.0, 22.0, "none", 0.0, 120.0, 300.0)],
            start=start, yaw=yaw_towards(tuple(start), safe_plaza)))

    return results


def scale_definition(definition: dict, factor: float) -> dict:
    scaled = json.loads(json.dumps(definition))
    scaled["duration_seconds"] = round(definition["duration_seconds"] * factor, 3)
    scaled["segments"] = [
        {**segment, "start": round(segment["start"] * factor, 3), "end": round(segment["end"] * factor, 3)}
        for segment in definition["segments"]
    ]
    return scaled


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=("full", "quick"), default="full")
    parser.add_argument("--object-index", default=DEFAULT_OBJECT_INDEX)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    arguments = parser.parse_args(argv)

    objects, seed = load_objects(Path(arguments.object_index))
    definitions = build_trajectories(objects, seed)

    if arguments.profile == "quick":
        wanted = set(QUICK_PROFILE)
        definitions = [d for d in definitions if d["trajectory_id"] in wanted]
        definitions = [scale_definition(d, QUICK_DURATION_SCALE) for d in definitions]
        missing = wanted - {d["trajectory_id"] for d in definitions}
        if missing:
            print(f"warning: quick profile could not build {sorted(missing)}", file=sys.stderr)

    output = Path(arguments.output_root) / arguments.profile
    output.mkdir(parents=True, exist_ok=True)

    index_entries = []
    for definition in definitions:
        path = output / f"{definition['trajectory_id']}.json"
        path.write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
        index_entries.append({
            "trajectory_id": definition["trajectory_id"],
            "kind": definition["kind"],
            "description": definition["description"],
            "file": str(path).replace("\\", "/"),
            "duration_seconds": definition["duration_seconds"],
            "segment_count": len(definition["segments"]),
            "modes": sorted({segment["mode"] for segment in definition["segments"]}),
        })
        print(f"  {definition['trajectory_id']:<26} {definition['duration_seconds']:>6.1f}s  "
              f"{len(definition['segments'])} segment(s)  {definition['kind']}")

    index = {
        "schema_version": 1,
        "profile": arguments.profile,
        "environment_seed": seed,
        "duration_scale": QUICK_DURATION_SCALE if arguments.profile == "quick" else 1.0,
        "trajectory_count": len(index_entries),
        "total_duration_seconds": round(sum(e["duration_seconds"] for e in index_entries), 3),
        "trajectories": index_entries,
    }
    index_path = output / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(index_entries)} trajectories ({index['total_duration_seconds']}s) -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
