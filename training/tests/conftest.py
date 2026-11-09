"""Shared fixtures.

Tests run against a three-trajectory slice of `run_full_p25` copied into a temporary
directory. The slice keeps the suite fast while still covering a straight walk, a
rotation-heavy trajectory and a sprint, which is what most of the edge cases need.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "dataset" / "processed" / "run_full_p25"
CONFIG_PATH = ROOT / "training" / "configs" / "features_v1.yaml"

FIXTURE_TRAJECTORIES = ("t01_walk_straight", "t06_look_around", "t12_sprint_weave")


@pytest.fixture(scope="session")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def tiny_config(config) -> dict:
    """The real config with the fold structure resized for three trajectories."""
    import copy
    tiny = copy.deepcopy(config)
    tiny["splits"]["folds"] = 3
    tiny["splits"]["fold_size"] = 1
    tiny["baselines"]["predictive"]["grid"] = {
        "damping": [0.0, 0.5],
        "angular_damping": [0.0, 1.0],
        "integrate_steps": [1, 10],
    }
    tiny["baselines"]["predictive"]["margin_thresholds_deg"] = [-5.0, 0.0]
    tiny["baselines"]["predictive"]["occlusion_variants"] = True
    return tiny


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    """A temporary dataset directory holding the three fixture trajectories."""
    directory = tmp_path_factory.mktemp("dataset_slice")
    for trajectory_id in FIXTURE_TRAJECTORIES:
        for kind in ("labels", "samples"):
            source = DATASET / f"{trajectory_id}.{kind}.jsonl"
            if source.is_file():
                shutil.copy(source, directory / source.name)
    return directory


@pytest.fixture(scope="session")
def split_manifest_path() -> Path:
    return ROOT / "dataset" / "manifests" / "runs" / "run_full_p25" / "splits.json"


@pytest.fixture(scope="session")
def fixture_split(tmp_path_factory, fixture_dir) -> Path:
    """A split manifest covering the fixture trajectories.

    The real manifest puts all three fixture trajectories in `train`, which would
    leave validation and test empty and make the end-to-end tests vacuous. This one
    assigns one trajectory to each partition.
    """
    directory = tmp_path_factory.mktemp("split_manifest")
    path = directory / "splits.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "profile": "fixture",
        "seed": 20261105,
        "rule": "trajectory-level split over the test fixture trajectories",
        "trajectory_count": len(FIXTURE_TRAJECTORIES),
        "counts": {"train": 1, "validation": 1, "test": 1},
        "splits": {
            "train": [{"trajectory_id": FIXTURE_TRAJECTORIES[0]}],
            "validation": [{"trajectory_id": FIXTURE_TRAJECTORIES[1]}],
            "test": [{"trajectory_id": FIXTURE_TRAJECTORIES[2]}],
        },
    }, indent=2) + "\n", encoding="utf-8")
    return path
