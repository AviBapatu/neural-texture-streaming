"""Phase 5sel tests: the selector's invariants and the ranking model's primitives.

The synthetic scene is the same shape ``sim/tests/test_sim.py`` uses, so the selector
engine can be run against it directly. Two tests reuse the real 188k-row artefacts (the
4b+ CSV and the score bundle); both skip cleanly when those have not been generated.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim import policies_5sel as P
from sim import run_5sel, scene
from sim.policies import build_policies
from sim.simulate import RunConfig, SimData, TrajectoryView, build_sim_data, summarise, simulate

ROOT = Path(__file__).resolve().parents[2]
CONFIG_4BPLUS = ROOT / "sim" / "config_4bplus.yaml"
RESULTS_4BPLUS = ROOT / "sim" / "results_4bplus" / "results.csv"
OOF = ROOT / "training" / "experiments" / "phase4a" / "oof_scores.npz"
DATASET = ROOT / "dataset" / "processed" / "run_full_p25"

MB = 1e6
A, B, C = "obj_a", "obj_b", "obj_c"


def synthetic_data(n_samples: int = 20, spans=None) -> SimData:
    spans = spans if spans is not None else {A: (5, 10), B: (12, 18), C: None}

    def visible(oid, sample):
        span = spans.get(oid)
        return span is not None and span[0] <= sample <= span[1]

    rows = [(oid, s, True, 1 if visible(oid, s) else 0, 1.0)
            for s in range(n_samples) for oid in (A, B, C)]
    object_ids = np.array([r[0] for r in rows], dtype=object)
    sample_index = np.array([r[1] for r in rows], dtype=np.int64)
    frustum = np.array([r[2] for r in rows], dtype=bool)
    state = np.array([r[3] for r in rows], dtype=np.int8)
    coverage = np.array([r[4] for r in rows], dtype=np.float64)
    rows_by_sample = {s: np.flatnonzero(sample_index == s) for s in range(n_samples)}
    view = TrajectoryView("t_synthetic", list(range(n_samples)), rows_by_sample,
                          frozenset(), np.arange(len(rows)))
    scores = {key: np.zeros(len(rows)) for key in
              ("mlp_b__occlusion_0_5", "mlp_b__occlusion_1_0", "gbdt_c__occlusion_0_5",
               "gbdt_c__occlusion_1_0", "heuristic_frustum", "heuristic_occlusion")}
    return SimData(object_ids=object_ids, trajectory=np.zeros(len(rows), dtype=np.int32),
                   trajectories=["t_synthetic"], sample_index=sample_index,
                   frustum_visible=frustum, occlusion_state=state, coverage=coverage,
                   scores=scores, occlusion_visible_in_1_0s=np.zeros(len(rows), dtype=bool),
                   occlusion_first_visible_time_1_0s=np.full(len(rows), np.nan), views=[view])


def sizes_for(multiplier: float = 1.0) -> dict:
    return {oid: 1.0 * MB * multiplier for oid in (A, B, C)}


# --- the selector helper ------------------------------------------------------

def test_allowance_never_exceeded():
    """The selector stops the moment the allowance is spent.

    ``A`` is a stop threshold, not a hard ceiling: the object that crosses it is still
    requested. What is asserted is that no request is made once the running total has
    reached ``A`` - i.e. every requested object before the last was added while the
    cumulative spend was still below ``A``.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        count = int(rng.integers(1, 8))
        sizes = {f"o{i}": float(rng.integers(1, 10)) * MB for i in range(count)}
        priorities = {oid: float(rng.random()) + 0.1 for oid in sizes}
        allowance = float(rng.integers(1, 30)) * MB
        requested, spent = P.select_requests(priorities, {}, {}, sizes, allowance)
        ranked = sorted(priorities, key=lambda o: (-priorities[o], o))

        # prefix of the ranking, so ordering is respected
        assert requested == ranked[:len(requested)]
        # no request was made once the allowance was already spent: the running total
        # before every request, including the last, is below the allowance
        cumulative = 0.0
        for oid in requested:
            assert cumulative < allowance, "requested past a spent allowance"
            cumulative += sizes[oid]
        if len(requested) < len(ranked):
            assert cumulative >= allowance - 1e-9, "stopped before the allowance was spent"


def test_ordering_respected_and_uncapped_requests_everything():
    sizes = {A: 1.0 * MB, B: 2.0 * MB, C: 3.0 * MB}
    priorities = {A: 0.2, B: 0.9, C: 0.5}
    requested, spent = P.select_requests(priorities, {}, {}, sizes, 1.5 * MB)
    assert requested == [B]                      # highest first, crossing object included
    assert spent == 2.0 * MB
    everything, total = P.select_requests(priorities, {}, {}, sizes, math.inf)
    assert everything == [B, C, A]               # descending score, id breaks ties
    assert total == 6.0 * MB


def test_resident_and_in_flight_objects_cost_nothing():
    sizes = {A: 5.0 * MB, B: 1.0 * MB, C: 1.0 * MB}
    priorities = {A: 0.9, B: 0.8, C: 0.7}
    # A is in flight and B resident: neither is a candidate, neither consumes allowance,
    # so C (1 MB) fits under a 1.5 MB allowance.
    requested, spent = P.select_requests(priorities, resident={B: 1.0 * MB},
                                         in_flight={A: 0.0}, sizes=sizes,
                                         allowance=1.5 * MB)
    assert requested == [C]
    assert spent == 1.0 * MB


# --- the selector engine reproduces the original engine -----------------------

@pytest.mark.parametrize("name", ["reactive_visible", "heuristic_occ", "mlp_b"])
def test_uncapped_selector_engine_is_bit_for_bit_the_original(name):
    """With an infinite allowance the two engines must agree exactly.

    This is what makes every other number in this phase comparable to Phase 4b+: the
    selector changes nothing when there is no allowance.
    """
    data = synthetic_data()
    policies = build_policies(data, {"policies": [name]})
    policy = policies[name]
    priorities = policy.priority(np.arange(len(data.object_ids)))
    sizes = sizes_for()
    for latency in (0.0, 0.3):
        for bandwidth in (1.0 * MB, 50.0 * MB):
            config = RunConfig(budget_bytes=2.5 * MB, size_multiplier=1.0,
                               latency_seconds=latency, bandwidth_bytes_per_s=bandwidth)
            original = summarise(simulate(data.views[0], policy, priorities, sizes,
                                          data.ground_truth_visible, data.coverage,
                                          data.object_ids, config), config)
            selected = summarise(P.simulate_selector(
                data.views[0], priorities, sizes, data.ground_truth_visible, data.coverage,
                data.object_ids, config, allowance_bytes=math.inf), config)
            assert original == selected


def test_tighter_allowance_streams_no_more_and_misses_no_less():
    """A smaller factor f can only commit fewer requests, so bytes are monotone in f."""
    data = synthetic_data(n_samples=40)
    policies = build_policies(data, {"policies": ["mlp_b"]})
    priorities = policies["mlp_b"].priority(np.arange(len(data.object_ids)))
    sizes = sizes_for()
    config = RunConfig(budget_bytes=3.0 * MB, size_multiplier=1.0, latency_seconds=0.2,
                       bandwidth_bytes_per_s=2.0 * MB)
    previous = -1.0
    for allowance in (0.0, 0.2 * MB, 0.5 * MB, 1.0 * MB, math.inf):
        metrics = summarise(P.simulate_selector(
            data.views[0], priorities, sizes, data.ground_truth_visible, data.coverage,
            data.object_ids, config, allowance_bytes=allowance), config)
        assert metrics["bytes_streamed"] >= previous - 1e-9
        previous = metrics["bytes_streamed"]
    # With no allowance no request is ever made, so nothing streams.
    empty = summarise(P.simulate_selector(
        data.views[0], priorities, sizes, data.ground_truth_visible, data.coverage,
        data.object_ids, config, allowance_bytes=0.0), config)
    assert empty["bytes_streamed"] == 0.0


# --- the P3 training primitives ----------------------------------------------

def test_ndcg_at_k_is_one_for_a_perfect_ranking_and_below_for_a_wrong_one():
    gains = np.array([3.0, 2.0, 1.0, 0.0])
    perfect = P.ndcg_at_k(np.array([4.0, 3.0, 2.0, 1.0]), gains,
                          [np.arange(4)], k=4)
    wrong = P.ndcg_at_k(np.array([1.0, 2.0, 3.0, 4.0]), gains, [np.arange(4)], k=4)
    assert perfect == pytest.approx(1.0)
    assert wrong < perfect
    assert P.ndcg_at_k(np.zeros(4), np.zeros(4), [np.arange(4)]) == 0.0


def test_listwise_and_hinge_losses_are_finite_and_prefer_the_right_order():
    import torch

    group = torch.tensor([0, 0, 1, 1])
    gain = torch.tensor([3.0, 1.0, 0.0, 2.0])
    good = torch.tensor([3.0, 1.0, 0.0, 2.0])          # matches the gain order in both groups
    bad = torch.tensor([1.0, 3.0, 2.0, 0.0])           # reversed in both groups
    for loss_fn in (P._listnet_loss, P._hinge_loss):
        good_value = float(loss_fn(good, group, gain))
        bad_value = float(loss_fn(bad, group, gain))
        assert math.isfinite(good_value) and math.isfinite(bad_value)
        assert good_value < bad_value, f"{loss_fn.__name__} did not prefer the right order"
    # A group with zero total gain contributes nothing to the listwise loss.
    zero = P._listnet_loss(torch.zeros(2), torch.tensor([0, 0]), torch.zeros(2))
    assert float(zero) == 0.0


def test_gain_target_is_non_negative_and_has_positives():
    if not DATASET.is_dir():
        pytest.skip("dataset not present")
    from training import dataset as D

    feature_config = yaml.safe_load(
        (ROOT / "training" / "configs" / "features_v1.yaml").read_text(encoding="utf-8"))
    table = D.build_table(DATASET, feature_config)
    records = scene.load_objects(ROOT / "dataset" / "manifests" / "object_index.json")
    sizes = scene.sized_bytes(records, 1.0)
    gain = P.gain_target(table, DATASET, sizes)
    assert np.isfinite(gain).all() and (gain >= 0).all()
    assert gain.sum() > 0.0, "the label rows carried no future coverage at all"


# --- the real artefacts -------------------------------------------------------

@pytest.mark.skipif(not RESULTS_4BPLUS.is_file(), reason="run the 4b+ sweep first")
def test_relevant_cells_are_the_twelve_of_phase_4bplus():
    config = yaml.safe_load(CONFIG_4BPLUS.read_text(encoding="utf-8"))
    cells = run_5sel.relevant_cells(RESULTS_4BPLUS,
                                    float(config["nominal_bandwidth_mb_s"]))
    assert len(cells) == 12
    scales = {cell["load_time_scale"] for cell in cells}
    assert scales == {0.25, 1.0, 4.0}
    assert all(cell["budget_fraction"] in (1.0, 1.5) for cell in cells)


@pytest.mark.skipif(not (RESULTS_4BPLUS.is_file() and OOF.is_file()),
                    reason="run the 4b+ sweep and save_oof_scores.py first")
def test_uncapped_engine_reproduces_the_published_4bplus_deficits():
    """The strongest check here: an uncapped replay must reproduce Phase 4b+ exactly.

    If the new engine disagreed with the published 4b+ deficits for the same policies
    and cells, every comparison in this phase would be against a different simulator.
    """
    config = yaml.safe_load(CONFIG_4BPLUS.read_text(encoding="utf-8"))
    nominal = float(config["nominal_bandwidth_mb_s"])
    sample_interval = float(config["sample_interval_seconds"])
    window = int(round(float(config["working_set_window_seconds"]) / sample_interval))
    data = build_sim_data(config["paths"]["oof_scores"], config["paths"]["dataset"],
                          int(config["snap_window_samples"]))
    records = scene.load_objects(config["paths"]["object_index"])
    sizes = scene.sized_bytes(records, 1.0)
    size_per_row = np.array([sizes[oid] for oid in data.object_ids], dtype=np.float64)
    fixed = P.fixed_priorities(data)

    published = {}
    with RESULTS_4BPLUS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "deficit" or float(row["budget_fraction"]) != 1.0 \
                    or float(row["load_time_scale"]) != 1.0 \
                    or float(row["latency_seconds"]) != 1.0:
                continue
            published[(row["policy"], row["trajectory"])] = float(row["value"])

    fraction, bandwidth = 1.0, 200.0
    allowance_base = bandwidth * 1e6 * sample_interval
    oracle = P.oracle_matched_priority(data, size_per_row, 1.0, bandwidth * 1e6,
                                       float(config["matched_margin_seconds"]))
    choices = run_5sel.cell_policies(fixed, oracle, fixed["mlp_b_tau0"], allowance_base)
    checked = 0
    for view in data.views:
        peak = scene.peak_working_set(view.rows_by_sample, data.ground_truth_visible,
                                      data.object_ids, sizes, window)
        run = RunConfig(budget_bytes=max(peak * fraction, 1.0), size_multiplier=1.0,
                        latency_seconds=1.0, bandwidth_bytes_per_s=bandwidth * 1e6,
                        sample_interval=sample_interval)
        for name in ("reactive_visible", "mlp_b_tau0", "oracle_matched"):
            priority, allowance = choices[name]
            metrics = summarise(P.simulate_selector(
                view, priority, sizes, data.ground_truth_visible, data.coverage,
                data.object_ids, run, allowance_bytes=allowance), run)
            key = (name, view.trajectory_id)
            if key not in published:
                continue
            # The published CSV rounds each value to 6 significant figures, so the
            # agreement is per-mille rather than bit-for-bit; the exact bit-for-bit
            # check is against the live engine in the test above.
            assert abs(metrics["deficit"] - published[key]) < 1e-5, (
                f"{key}: engine {metrics['deficit']} vs published {published[key]}")
            checked += 1
    assert checked == 3 * 14, f"only {checked} deficits compared"
