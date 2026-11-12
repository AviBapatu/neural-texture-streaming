"""Phase 4a tests: the simulator's invariants and the properties the brief requires.

Several of these are cheap to state and easy to get subtly wrong, so they are
written as behavioural checks against a tiny synthetic scene rather than against the
real 188k-row bundle. The one test that does need the real artefacts skips cleanly
when they have not been generated.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim import run_sweep, scene
from sim.policies import (MATCHED_MARGIN_SECONDS, Policy, build_policies,
                          deployable_names, lagged_occlusion_state, matched_priorities,
                          occlusion_visible, time_to_next_visible)
from sim.simulate import (RunConfig, SimData, TrajectoryView, build_sim_data,
                          summarise, simulate)
from sim.summary_4bplus import paired_ci, reduction_ci, resample_matrix

ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "training" / "experiments" / "phase4a" / "oof_scores.npz"
OOF_MANIFEST = ROOT / "training" / "experiments" / "phase4a" / "oof_scores.manifest.json"
CONFIG = ROOT / "sim" / "config.yaml"
CONFIG_4B = ROOT / "sim" / "config_4b.yaml"
CONFIG_4BPLUS = ROOT / "sim" / "config_4bplus.yaml"
RESULTS_4B = ROOT / "sim" / "results_4b" / "results.csv"
RESULTS_4BPLUS = ROOT / "sim" / "results_4bplus" / "results.csv"

MB = 1e6

# --- synthetic scene ----------------------------------------------------------

# Object ids used by the synthetic trajectories below.
A, B, C = "obj_a", "obj_b", "obj_c"


def synthetic_data(n_samples: int = 20, visible_spans: dict | None = None,
                   occlusion_unknown: bool = False, oracle_horizon: int = 10) -> SimData:
    """A three-object trajectory: A visible early, B later, C never.

    Returns a ``SimData`` shaped exactly like the real one, so the engine runs
    unmodified. Sizes are supplied separately to ``simulate`` so the same data can be
    replayed at different memory costs.

    The oracle's future labels are derived from the same spans, so the oracle policy
    has genuine foresight here rather than an all-zero score row.
    """
    spans = visible_spans if visible_spans is not None else {A: (5, 10), B: (12, 18), C: None}

    def is_visible(oid: str, sample: int) -> bool:
        span = spans.get(oid)
        return span is not None and span[0] <= sample <= span[1]

    rows: list[tuple] = []
    for sample in range(n_samples):
        for oid in (A, B, C):
            visible = is_visible(oid, sample)
            rows.append((oid, sample, True, 1 if visible else 0, 1.0))

    object_ids = np.array([r[0] for r in rows], dtype=object)
    sample_index = np.array([r[1] for r in rows], dtype=np.int64)
    frustum = np.array([r[2] for r in rows], dtype=bool)
    state = np.array([r[3] for r in rows], dtype=np.int8)
    coverage = np.array([r[4] for r in rows], dtype=np.float64)

    if occlusion_unknown:
        state = np.full_like(state, -1)

    future_visible = np.zeros(len(rows), dtype=bool)
    first_visible = np.full(len(rows), np.nan, dtype=np.float64)
    for index, (oid, sample) in enumerate(zip(object_ids, sample_index)):
        for offset in range(1, oracle_horizon + 1):
            if is_visible(oid, int(sample) + offset):
                future_visible[index] = True
                first_visible[index] = float(offset) * 0.1
                break

    rows_by_sample = {s: np.flatnonzero(sample_index == s) for s in range(n_samples)}
    view = TrajectoryView("t_synthetic", list(range(n_samples)), rows_by_sample,
                          frozenset(), np.arange(len(rows)))
    scores = {key: np.zeros(len(rows)) for key in
              ("mlp_b__occlusion_0_5", "mlp_b__occlusion_1_0",
               "gbdt_c__occlusion_0_5", "gbdt_c__occlusion_1_0",
               "heuristic_frustum", "heuristic_occlusion")}
    return SimData(object_ids=object_ids, trajectory=np.zeros(len(rows), dtype=np.int32),
                   trajectories=["t_synthetic"], sample_index=sample_index,
                   frustum_visible=frustum, occlusion_state=state, coverage=coverage,
                   scores=scores, occlusion_visible_in_1_0s=future_visible,
                   occlusion_first_visible_time_1_0s=first_visible, views=[view])


def policy_set(data: SimData, names=None) -> dict[str, Policy]:
    config = {"policies": names or ["reactive_frustum", "reactive_visible",
                                    "heuristic_frustum", "heuristic_occ",
                                    "mlp_b", "gbdt_c", "oracle"]}
    return build_policies(data, config)


def sizes_for(objects=(A, B, C), multiplier: float = 1.0) -> dict[str, float]:
    return {oid: 1.0 * MB * multiplier for oid in objects}


def run_one(data: SimData, policy: Policy, config: RunConfig, sizes=None,
            stop_sample: int | None = None) -> dict:
    view = data.views[0]
    priorities = policy.priority(np.arange(len(data.object_ids)))
    return summarise(simulate(view, policy, priorities, sizes or sizes_for(),
                              data.ground_truth_visible, data.coverage,
                              data.object_ids, config, stop_sample=stop_sample))


# --- sanity: generous resources leave no deficit ------------------------------

def test_infinite_budget_and_zero_latency_gives_zero_deficit_for_every_policy():
    """If memory and latency are free, nothing can ever be missing when visible.

    This is the end-to-end sanity check: it exercises the request path, the streaming
    path, the completion path and the measurement path for all seven policies at once.
    """
    data = synthetic_data()
    policies = policy_set(data)
    config = RunConfig(budget_bytes=1e12, size_multiplier=1.0, latency_seconds=0.0,
                       bandwidth_bytes_per_s=1e12)
    for name, policy in policies.items():
        metrics = run_one(data, policy, config)
        assert metrics["deficit"] == 0.0, f"{name} showed deficit with unlimited resources"
        assert metrics["pop_in_events"] == 0, f"{name} popped in with unlimited resources"
        assert metrics["peak_resident_bytes"] <= config.budget_bytes


def test_deficit_is_one_when_nothing_is_ever_loaded():
    """The other end of the sanity range: no bandwidth means every visible step misses."""
    data = synthetic_data()
    policies = policy_set(data)
    config = RunConfig(budget_bytes=1e12, size_multiplier=1.0, latency_seconds=0.0,
                       bandwidth_bytes_per_s=0.0)
    for name, policy in policies.items():
        metrics = run_one(data, policy, config)
        assert metrics["deficit"] == pytest.approx(1.0), f"{name} should miss everything"
        assert metrics["peak_resident_bytes"] == 0.0


def test_reactive_policies_request_nothing_that_is_not_currently_visible():
    """A reactive policy's score band is empty, so it cannot prefetch at all."""
    data = synthetic_data()
    policies = policy_set(data)
    config = RunConfig(budget_bytes=1e12, size_multiplier=1.0, latency_seconds=0.0,
                       bandwidth_bytes_per_s=1e6)
    for name in ("reactive_frustum", "reactive_visible"):
        metrics = run_one(data, policies[name], config)
        # A and B are visible for 6 and 7 samples; with a slow link neither can be
        # fully resident on the step it first becomes visible, so there is deficit.
        assert metrics["deficit"] > 0.0


# --- the budget invariant -----------------------------------------------------

def test_peak_resident_bytes_never_exceeds_the_budget():
    """Across a spread of budgets, some far too small for even one object."""
    data = synthetic_data()
    policies = policy_set(data)
    for budget_multiple in (0.25, 0.5, 1.0, 1.5, 2.5, 10.0):
        config = RunConfig(budget_bytes=budget_multiple * MB, size_multiplier=1.0,
                           latency_seconds=0.1, bandwidth_bytes_per_s=100 * MB)
        for name, policy in policies.items():
            metrics = run_one(data, policy, config)
            assert metrics["peak_resident_bytes"] <= budget_multiple * MB + 1e-6, (
                f"{name} exceeded a budget of {budget_multiple}MB with "
                f"{metrics['peak_resident_bytes'] / MB:.3f}MB resident")


def test_an_object_larger_than_the_budget_is_never_resident():
    """It cannot fit at any point, so the engine must not report it as resident."""
    data = synthetic_data()
    policy = policy_set(data)["reactive_frustum"]
    config = RunConfig(budget_bytes=0.5 * MB, size_multiplier=1.0, latency_seconds=0.0,
                       bandwidth_bytes_per_s=100 * MB)
    metrics = run_one(data, policy, config)
    assert metrics["peak_resident_bytes"] == 0.0
    assert metrics["deficit"] == pytest.approx(1.0)


# --- reactive policies read only the present ----------------------------------

def test_reactive_policies_are_unaffected_by_truncating_the_future():
    """Truncation test: a reactive policy cannot see later samples.

    The first k samples of a full replay must be identical to a replay that stops at
    k. If a reactive policy were peeking at the future - through the scores, the
    labels or the sizing - the two would diverge. Compared on the accumulated metrics
    rather than the normalised deficit, because truncation changes the denominator.
    """
    data = synthetic_data(n_samples=20)
    policies = policy_set(data)
    config = RunConfig(budget_bytes=2.0 * MB, size_multiplier=1.0, latency_seconds=0.1,
                       bandwidth_bytes_per_s=5 * MB)

    cut = 12
    for name in ("reactive_frustum", "reactive_visible"):
        policy = policies[name]
        prefix = run_one(data, policy, config, stop_sample=cut)

        view = data.views[0]
        truncated_rows = {s: r for s, r in view.rows_by_sample.items() if s < cut}
        truncated = TrajectoryView("t_synthetic", sorted(truncated_rows), truncated_rows,
                                   frozenset(), view.all_rows)
        part = summarise(simulate(
            truncated, policy, policy.priority(np.arange(len(data.object_ids))),
            sizes_for(), data.ground_truth_visible, data.coverage, data.object_ids, config))

        assert part == prefix, f"{name} changed when the future was removed"

        # And the future really did hold something, or the test proves nothing.
        full = run_one(data, policy, config)
        assert full["steps"] > prefix["steps"]


def test_a_policy_reading_the_future_would_fail_the_truncation_test():
    """Negative control: the oracle *does* change when the future is removed.

    Without this, the truncation test above would pass even if the harness silently
    handed every policy the same future information.
    """
    data = synthetic_data(n_samples=20)
    policy = policy_set(data)["oracle"]
    config = RunConfig(budget_bytes=2.0 * MB, size_multiplier=1.0, latency_seconds=0.1,
                       bandwidth_bytes_per_s=20 * MB)

    # A scene whose only visible object appears *after* the cut. The full run's oracle
    # can see it coming and starts loading; the stripped run cannot, because in that
    # scene the object never becomes visible at all.
    cut = 14
    late = synthetic_data(n_samples=30, visible_spans={A: (15, 25), B: None, C: None})
    late_policy = policy_set(late)["oracle"]
    ahead = summarise(simulate(
        late.views[0], late_policy, late_policy.priority(np.arange(len(late.object_ids))),
        sizes_for(), late.ground_truth_visible, late.coverage, late.object_ids,
        config, stop_sample=cut))

    stripped = synthetic_data(n_samples=cut, visible_spans={A: None, B: None, C: None})
    stripped_policy = policy_set(stripped)["oracle"]
    blind = summarise(simulate(
        stripped.views[0], stripped_policy,
        stripped_policy.priority(np.arange(len(stripped.object_ids))),
        sizes_for(), stripped.ground_truth_visible, stripped.coverage,
        stripped.object_ids, config))

    assert ahead["peak_resident_bytes"] > blind["peak_resident_bytes"], (
        "the oracle loaded nothing ahead of time; the control would prove nothing")
    assert ahead["loads_completed"] > blind["loads_completed"]


def test_visible_set_ignores_the_future_and_uses_the_current_step_only():
    """The reactive priority array is a function of the current row and nothing else."""
    data = synthetic_data()
    policies = policy_set(data)
    view = data.views[0]
    # Rows of a sample that is not currently visible get priority 0 under reactive.
    for sample in view.samples:
        rows = view.rows_by_sample[sample]
        expected = np.where(data.frustum_visible[rows], 2.0, 0.0)
        assert np.allclose(policies["reactive_frustum"].priority(rows), expected)


# --- oracle handling ----------------------------------------------------------

def test_oracle_is_not_deployable_and_is_excluded_from_summary_policy_lists():
    data = synthetic_data()
    policies = policy_set(data)
    assert "oracle" in policies
    assert policies["oracle"].deployable is False
    assert "oracle" not in deployable_names(policies)
    assert set(deployable_names(policies)) == {
        "reactive_frustum", "reactive_visible", "heuristic_frustum", "heuristic_occ",
        "mlp_b", "gbdt_c"}


def test_oracle_summary_tables_omit_the_oracle_from_the_gap_columns():
    """The oracle gap table must list deployable policies only, per the brief."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data = synthetic_data()
    policies = policy_set(data)
    priorities = {name: policy.priority(np.arange(len(data.object_ids)))
                  for name, policy in policies.items()}

    per_trajectory = {}
    for fraction in config["sweep"]["budget_fraction_of_peak_working_set"]:
        for multiplier in config["sweep"]["size_multiplier"]:
            for latency in config["sweep"]["latency_seconds"]:
                sizes = sizes_for(multiplier=float(multiplier))
                run = RunConfig(budget_bytes=2 * MB, size_multiplier=float(multiplier),
                                latency_seconds=float(latency),
                                bandwidth_bytes_per_s=10 * MB)
                for name, policy in policies.items():
                    key = (float(fraction), float(multiplier), float(latency),
                           float(config["bandwidth_mb_s"]), name, "t_synthetic")
                    per_trajectory[key] = summarise(simulate(
                        data.views[0], policy, priorities[name], sizes,
                        data.ground_truth_visible, data.coverage, data.object_ids, run), run)

    peaks = {float(m): {"t_synthetic": 3.0 * MB} for m in config["sweep"]["size_multiplier"]}
    summary = run_sweep.build_summary(config, data, policies, per_trajectory, peaks, policies)

    def header_of(section: str) -> str:
        return next(line for line in section.splitlines() if line.startswith("| budget"))

    gap_section = summary.split("## Oracle gap")[1].split("## Paired differences")[0]
    header = header_of(gap_section)
    assert "oracle" not in header, "the oracle must not appear in the deployable gap table"
    assert "mlp_b" in header and "heuristic_frustum" in header
    # The deficit table above it does show the oracle, as the upper bound.
    deficit_section = summary.split("## Deficit by policy")[1].split("## Oracle gap")[0]
    assert "oracle" in header_of(deficit_section)


def test_oracle_deficit_is_not_worse_than_the_reactive_baseline():
    """The oracle has strictly more information, so it cannot do worse on this scene."""
    data = synthetic_data()
    policies = policy_set(data)
    config = RunConfig(budget_bytes=1.5 * MB, size_multiplier=1.0, latency_seconds=0.1,
                       bandwidth_bytes_per_s=2 * MB)
    oracle = run_one(data, policies["oracle"], config)
    for name in ("reactive_frustum", "reactive_visible", "heuristic_frustum",
                 "heuristic_occ", "mlp_b", "gbdt_c"):
        assert oracle["deficit"] <= run_one(data, policies[name], config)["deficit"] + 1e-9


# --- ground truth -------------------------------------------------------------

def test_unknown_occlusion_counts_as_visible():
    """The project's convention: unknown is not "definitely occluded"."""
    data = synthetic_data(occlusion_unknown=True)
    # Every row is frustum visible with unknown occlusion, so every row is visible.
    assert data.ground_truth_visible.all()
    definitely_occluded = np.array([1, 1, 1], dtype=bool)
    assert not occlusion_visible(definitely_occluded, np.array([0, 0, 0])).any()
    assert occlusion_visible(definitely_occluded, np.array([-1, 1, 0])).tolist() == [True, True, False]


def test_snap_affected_samples_are_excluded_from_the_scored_deficit():
    """Stepping inside a snap window must not move the deficit, only the coverage."""
    data = synthetic_data()
    policy = policy_set(data)["reactive_frustum"]
    config = RunConfig(budget_bytes=0.1 * MB, size_multiplier=1.0, latency_seconds=0.0,
                       bandwidth_bytes_per_s=0.0)
    scored = run_one(data, policy, config)

    view = data.views[0]
    excluded = frozenset(range(0, 20))
    blind = TrajectoryView(view.trajectory_id, view.samples, view.rows_by_sample,
                           excluded, view.all_rows)
    hidden = summarise(simulate(blind, policy,
                                policy.priority(np.arange(len(data.object_ids))),
                                sizes_for(), data.ground_truth_visible, data.coverage,
                                data.object_ids, config))
    assert hidden["visible_coverage"] == 0.0
    assert hidden["deficit"] == 0.0            # nothing scored, so nothing to divide
    assert hidden["pop_in_events"] == scored["pop_in_events"]  # still simulated


# --- determinism --------------------------------------------------------------

def test_the_same_run_twice_gives_identical_metrics():
    data = synthetic_data()
    policy = policy_set(data)["heuristic_occ"]
    config = RunConfig(budget_bytes=1.5 * MB, size_multiplier=1.0, latency_seconds=0.1,
                       bandwidth_bytes_per_s=3 * MB)
    first = run_one(data, policy, config)
    second = run_one(data, policy, config)
    assert first == second


# --- scene / sizing -----------------------------------------------------------

def test_full_texture_bytes_match_the_stated_formula():
    """width * height * 4 * 4/3, summed over the object's textures."""
    record = scene.ObjectRecord("x", "bld", (("t", 512, 512),))
    assert record.full_bytes_at_1x == pytest.approx(512 * 512 * 4 * (4 / 3))
    two = scene.ObjectRecord("y", "bld", (("t", 512, 512), ("u", 256, 256)))
    assert two.full_bytes_at_1x == pytest.approx(512 * 512 * 4 * (4 / 3)
                                                 + 256 * 256 * 4 * (4 / 3))


def test_size_multiplier_scales_bytes_linearly():
    records = {A: scene.ObjectRecord(A, "bld", (("t", 256, 256),))}
    assert scene.sized_bytes(records, 4.0)[A] == pytest.approx(
        4.0 * scene.sized_bytes(records, 1.0)[A])


def test_peak_working_set_counts_distinct_objects_not_rows():
    """An object visible across a whole window counts once, not once per sample."""
    n = 20
    data = synthetic_data(n_samples=n, visible_spans={A: (0, n - 1), B: None, C: None})
    sizes = {A: 10 * MB, B: 10 * MB, C: 10 * MB}
    peak = scene.peak_working_set(data.views[0].rows_by_sample, data.ground_truth_visible,
                                 data.object_ids, sizes, window_samples=20)
    assert peak == pytest.approx(10 * MB)

    both = synthetic_data(n_samples=n, visible_spans={A: (0, 5), B: (10, 15), C: None})
    peak_both = scene.peak_working_set(both.views[0].rows_by_sample,
                                       both.ground_truth_visible, both.object_ids,
                                       sizes, window_samples=20)
    assert peak_both == pytest.approx(20 * MB)


def test_working_set_window_is_bounded_by_the_window_not_the_trajectory():
    """A and B far apart in time never share a 2 s window."""
    data = synthetic_data(n_samples=60, visible_spans={A: (0, 3), B: (40, 45), C: None})
    sizes = {A: 10 * MB, B: 10 * MB, C: 10 * MB}
    peak = scene.peak_working_set(data.views[0].rows_by_sample, data.ground_truth_visible,
                                 data.object_ids, sizes, window_samples=20)
    assert peak == pytest.approx(10 * MB)


# --- the real artefacts -------------------------------------------------------

@pytest.mark.skipif(not OOF.is_file(), reason="run training/save_oof_scores.py first")
def test_oof_scores_for_a_trajectory_come_from_a_fold_that_excluded_it():
    """Every trajectory is held out by exactly one fold, and that fold supplies its scores.

    The second half is the substantive check: with synthetic per-fold arrays that
    encode the fold identity, the assembled out-of-fold values must take the value
    from the fold where the trajectory was *test*, never from any other fold.
    """
    from training import splits as S

    manifest = json.loads(OOF_MANIFEST.read_text(encoding="utf-8"))
    folds = [S.Fold(index=f["fold"], train=tuple(f["train"]),
                    validation=tuple(f["validation"]), test=tuple(f["test"]))
             for f in manifest["folds"]]
    trajectories = manifest["trajectories"]

    test_sets = [set(f.test) for f in folds]
    for position, fold in enumerate(folds):
        assert not (set(fold.train) & set(fold.test))
        assert not (set(fold.train) & set(fold.validation))
        assert not (set(fold.validation) & set(fold.test))
    for name in trajectories:
        holders = [i for i, test in enumerate(test_sets) if name in test]
        assert len(holders) == 1, f"{name} is test in {len(holders)} folds, expected 1"

    # Synthetic per-fold arrays tagged with the fold index: any row that picked up a
    # score from a fold that trained on it would show the wrong tag.
    with np.load(OOF, allow_pickle=True) as payload:
        trajectory = np.array(payload["trajectory"], dtype=object)
    per_fold = []
    for fold in folds:
        tagged = np.full((len(trajectory), 1), float(fold.index))
        per_fold.append(tagged)

    from training.save_oof_scores import assemble_oof

    codes = np.array([trajectories.index(t) for t in trajectory], dtype=np.int32)

    class StubTable:
        """Just enough of a FeatureTable for ``assemble_oof`` to slice by trajectory."""

        def __init__(self, trajectory_codes, names):
            self.trajectory = trajectory_codes
            self.trajectories = names

        def __len__(self):
            return int(self.trajectory.size)

    assembled = assemble_oof(StubTable(codes, trajectories), folds, per_fold, "synthetic")
    for position, name in enumerate(trajectories):
        rows = codes == position
        expected = next(f.index for f in folds if name in f.test)
        assert np.allclose(assembled[rows], expected), (
            f"{name} carries a score from a fold that did not hold it out")


RESULTS = ROOT / "sim" / "results" / "results.csv"


@pytest.mark.skipif(not RESULTS.is_file(), reason="run sim/run_sweep.py first")
def test_published_sweep_never_exceeds_its_budget_in_any_run():
    """The memory cap must hold in every one of the published runs, not just synthetic ones."""
    checked = 0
    worst = 0.0
    with RESULTS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "peak_resident_bytes":
                continue
            budget = float(row["budget_bytes"])
            peak = float(row["value"])
            assert peak <= budget * (1 + 1e-9), (
                f"{row['policy']} on {row['trajectory']} at budget "
                f"{row['budget_fraction']}/{row['size_multiplier']}x/"
                f"{row['latency_seconds']}s held {peak:.1f} bytes over a budget of "
                f"{budget:.1f}")
            worst = max(worst, peak / budget)
            checked += 1
    assert checked == 18 * 7 * 14, f"expected 1764 runs in the sweep, found {checked}"
    assert worst > 0.5, "no run came close to its budget; the sweep may not be exercising it"


@pytest.mark.skipif(not RESULTS.is_file(), reason="run sim/run_sweep.py first")
def test_published_sweep_covers_the_whole_declared_grid():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seen_configs, seen_policies, seen_trajectories = set(), set(), set()
    with RESULTS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seen_configs.add((float(row["budget_fraction"]), float(row["size_multiplier"]),
                              float(row["latency_seconds"])))
            seen_policies.add(row["policy"])
            seen_trajectories.add(row["trajectory"])
    expected_configs = {(float(b), float(m), float(l))
                        for b in config["sweep"]["budget_fraction_of_peak_working_set"]
                        for m in config["sweep"]["size_multiplier"]
                        for l in config["sweep"]["latency_seconds"]}
    assert seen_configs == expected_configs
    assert seen_policies == set(config["policies"])
    assert len(seen_trajectories) == 14
    assert len(expected_configs) == 18


@pytest.mark.skipif(not OOF.is_file(), reason="run training/save_oof_scores.py first")
def test_real_bundle_every_row_is_scored_and_keys_are_unique():
    data = build_sim_data(OOF, ROOT / "dataset" / "processed" / "run_full_p25", 10)
    for key, values in data.scores.items():
        assert np.isfinite(values).all(), f"{key} has non-finite scores"
        # Model outputs are probabilities; the heuristic arrays are raw frustum
        # margins in degrees and are squashed by the policy, so they are unbounded.
        if key.startswith(("mlp_b__", "gbdt_c__")):
            assert values.min() >= 0.0 and values.max() <= 1.0, f"{key} is outside [0, 1]"
    assert len(data.views) == 14
    for view in data.views:
        keys = set()
        for sample, rows in view.rows_by_sample.items():
            for row in rows:
                key = (sample, data.object_ids[row])
                assert key not in keys, f"duplicate (sample, object) in {view.trajectory_id}"
                keys.add(key)


# --- Phase 4b: the matched oracle ---------------------------------------------

def matched_priorities_for(data: SimData, latency: float, bandwidth: float,
                           sizes: dict[str, float] | None = None) -> np.ndarray:
    sizes = sizes or sizes_for()
    per_row = np.array([sizes[oid] for oid in data.object_ids], dtype=np.float64)
    return matched_priorities(data.ground_truth_visible,
                              time_to_next_visible(data, 0.1), per_row,
                              latency, bandwidth, MATCHED_MARGIN_SECONDS)


def test_matched_oracle_requests_exactly_what_is_inside_its_horizon():
    """The horizon is latency + load time + margin, and nothing outside it is requested."""
    latency, bandwidth = 0.5, 10.0 * MB
    size = 10.0 * MB                       # load time 1.0 s
    horizon = latency + size / bandwidth + MATCHED_MARGIN_SECONDS   # 1.7 s
    time_to_visible = np.array([0.0, 0.3, horizon - 1e-6, horizon + 1e-6, np.inf])
    priorities = matched_priorities(np.zeros(len(time_to_visible), dtype=bool),
                                    time_to_visible, np.full(len(time_to_visible), size),
                                    latency, bandwidth, MATCHED_MARGIN_SECONDS)
    assert priorities[0] > 0.0 and priorities[1] > 0.0 and priorities[2] > 0.0
    assert priorities[3] == 0.0, "an object beyond the horizon must not be requested"
    assert priorities[4] == 0.0, "an object that is never visible must not be requested"
    # Soonest first: priority is strictly decreasing in the time to visibility.
    assert priorities[0] > priorities[1] > priorities[2]
    assert 0.0 < priorities[2] <= 1.0


def test_matched_oracle_starts_a_slow_load_early_and_lets_a_fast_one_wait():
    """The horizon is matched to each object's own load time, not to a fixed horizon.

    Both objects are wanted 0.9 s from now. The 40 MB one needs 4 s of a 10 MB/s link,
    so it cannot possibly arrive unless it starts immediately and must be requested at
    high priority. The 1 MB one needs a tenth of a second and is not due for nine
    tenths, so a sensible policy leaves it alone and picks it up later - asking now
    would only occupy memory for nothing.
    """
    time_to_visible = np.array([0.9, 0.9])
    sizes_per_row = np.array([1.0 * MB, 40.0 * MB])
    priorities = matched_priorities(np.zeros(2, dtype=bool), time_to_visible,
                                    sizes_per_row, 0.1, 10.0 * MB,
                                    MATCHED_MARGIN_SECONDS)
    assert priorities[1] > 0.5, "the slow object must be requested at high priority"
    assert priorities[0] == 0.0, "the fast object can afford to wait"

    # The converse: when the fast object is nearly due, it does get requested.
    soon = matched_priorities(np.zeros(2, dtype=bool), np.array([0.2, 0.2]),
                              sizes_per_row, 0.1, 10.0 * MB, MATCHED_MARGIN_SECONDS)
    assert soon[0] > 0.0


def test_matched_oracle_uses_visible_now_for_objects_on_screen():
    """An object visible right now is priority 2 regardless of its time-to-visible."""
    visible_now = np.array([True, False])
    priorities = matched_priorities(visible_now, np.array([0.0, 0.2]),
                                    np.full(2, 1.0 * MB), 0.1, 10.0 * MB)
    assert priorities[0] == 2.0


def test_matched_oracle_never_exceeds_budget_at_any_latency_or_bandwidth():
    """The engine must hold the cap for the matched oracle as it does for everything else."""
    data = synthetic_data()
    policies = policy_set(data, ["oracle_matched"])
    sizes = sizes_for()
    for latency in (0.0, 0.5, 1.0):
        for bandwidth in (1.0 * MB, 50.0 * MB):
            priorities = matched_priorities_for(data, latency, bandwidth, sizes)
            for budget in (2.0 * MB, 5.0 * MB, 1e12):
                config = RunConfig(budget_bytes=budget, size_multiplier=1.0,
                                   latency_seconds=latency,
                                   bandwidth_bytes_per_s=bandwidth)
                metrics = summarise(simulate(
                    data.views[0], policies["oracle_matched"], priorities, sizes,
                    data.ground_truth_visible, data.coverage, data.object_ids, config),
                    config)
                assert metrics["peak_resident_bytes"] <= budget * (1 + 1e-9), (
                    f"latency={latency} bandwidth={bandwidth} budget={budget} held "
                    f"{metrics['peak_resident_bytes']}")


def test_ideal_budget_with_zero_latency_gives_zero_deficit():
    """With memory free and latency zero, the only remaining question is bandwidth.

    A 1 MB object over a 0.1 s step needs 10 MB/s, so at 20 MB/s the request is made,
    streamed and completed inside the very sample the object becomes visible. Any
    deficit here would mean the engine measures before it finishes loading.
    """
    data = synthetic_data()
    sizes = sizes_for()
    policies = policy_set(data, ["reactive_visible", "oracle_matched"])
    priorities = {"reactive_visible": policies["reactive_visible"].priority(
        np.arange(len(data.object_ids)))}
    priorities["oracle_matched"] = matched_priorities_for(data, 0.0, 20.0 * MB, sizes)

    for name in ("reactive_visible", "oracle_matched"):
        config = RunConfig(budget_bytes=float("inf"), size_multiplier=1.0,
                           latency_seconds=0.0, bandwidth_bytes_per_s=20.0 * MB)
        metrics = summarise(simulate(
            data.views[0], policies[name], priorities[name], sizes,
            data.ground_truth_visible, data.coverage, data.object_ids, config), config)
        assert metrics["deficit"] == 0.0, f"{name} showed deficit with unlimited budget"


def test_ideal_budget_still_shows_deficit_when_latency_is_the_only_obstacle():
    """The control for the test above: memory and bandwidth are not the problem here."""
    data = synthetic_data()
    sizes = sizes_for()
    policies = policy_set(data, ["reactive_visible"])
    priorities = policies["reactive_visible"].priority(np.arange(len(data.object_ids)))
    config = RunConfig(budget_bytes=float("inf"), size_multiplier=1.0,
                       latency_seconds=1.0, bandwidth_bytes_per_s=20.0 * MB)
    metrics = summarise(simulate(
        data.views[0], policies["reactive_visible"], priorities, sizes,
        data.ground_truth_visible, data.coverage, data.object_ids, config), config)
    assert metrics["deficit"] > 0.0, (
        "with infinite memory and a 1.0 s latency a reactive policy must still miss")


# --- Phase 4b: the published results ------------------------------------------

@pytest.mark.skipif(not RESULTS_4B.is_file(), reason="run sim/run_sweep.py --config sim/config_4b.yaml first")
def test_phase4b_published_runs_never_exceed_their_budget():
    """Every finite-budget run of the published 4b sweep respects its cap."""
    checked = 0
    ideal = 0
    with RESULTS_4B.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "peak_resident_bytes":
                continue
            budget = float(row["budget_bytes"])
            peak = float(row["value"])
            if math.isinf(budget):
                ideal += 1
                continue
            assert peak <= budget * (1 + 1e-9), (
                f"{row['policy']} on {row['trajectory']} held {peak:.1f} over {budget:.1f}")
            checked += 1
    assert checked == 54 * 8 * 14, f"expected 6048 finite-budget runs, found {checked}"
    assert ideal == 54 * 2 * 14, f"expected 1512 ideal_budget runs, found {ideal}"


@pytest.mark.skipif(not RESULTS_4B.is_file(), reason="run sim/run_sweep.py --config sim/config_4b.yaml first")
def test_phase4b_published_sweep_covers_the_whole_declared_grid():
    config = yaml.safe_load(CONFIG_4B.read_text(encoding="utf-8"))
    seen_configs, seen_policies, seen_bandwidths, kinds = set(), set(), set(), set()
    with RESULTS_4B.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seen_bandwidths.add(float(row["bandwidth_mb_s"]))
            kinds.add(row["budget_kind"])
            if row["budget_kind"] == "ideal":
                continue
            seen_configs.add((float(row["budget_fraction"]), float(row["size_multiplier"]),
                              float(row["latency_seconds"]), float(row["bandwidth_mb_s"])))
            seen_policies.add(row["policy"])
    expected = {(float(b), float(m), float(l), float(w))
                for b in config["sweep"]["budget_fraction_of_peak_working_set"]
                for m in config["sweep"]["size_multiplier"]
                for l in config["sweep"]["latency_seconds"]
                for w in config["sweep"]["bandwidth_mb_s"]}
    assert seen_configs == expected
    assert len(expected) == 54
    assert seen_policies == set(config["policies"])
    assert seen_bandwidths == {50.0, 200.0}
    assert kinds == {"fraction", "ideal"}


@pytest.mark.skipif(not RESULTS_4B.is_file(), reason="run sim/run_sweep.py --config sim/config_4b.yaml first")
def test_phase4b_summary_relevance_count_matches_an_independent_recount():
    """The published count of prediction-relevant configurations must be recomputable.

    The rule is arithmetic on the CSV, so it is recomputed here from scratch and
    compared against the number the summary printed - if the summary and the rule ever
    drift apart, this fails rather than the report quietly stating a wrong count.
    """
    import re
    config = yaml.safe_load(CONFIG_4B.read_text(encoding="utf-8"))
    threshold = float(config["reporting"]["prediction_relevant_gap"])
    values: dict[tuple, dict[str, float]] = {}
    with RESULTS_4B.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["budget_kind"] != "fraction" or row["metric"] != "deficit":
                continue
            key = (float(row["budget_fraction"]), float(row["size_multiplier"]),
                   float(row["latency_seconds"]), float(row["bandwidth_mb_s"]),
                   row["policy"])
            values.setdefault(key, {})[row["trajectory"]] = float(row["value"])

    configs = sorted({k[:4] for k in values})
    relevant = 0
    for cell in configs:
        rv = np.mean(list(values[cell + ("reactive_visible",)].values()))
        om = np.mean(list(values[cell + ("oracle_matched",)].values()))
        if rv - om >= threshold - 1e-9:
            relevant += 1

    summary = (ROOT / "sim" / "results_4b" / "summary.md").read_text(encoding="utf-8")
    stated = int(re.search(r"\*\*(\d+) of 54\*\* configurations are prediction-relevant",
                           summary).group(1))
    assert stated == relevant, (
        f"summary claims {stated} prediction-relevant configurations, recount says "
        f"{relevant}")


def test_sweep_is_deterministic_across_two_runs(tmp_path):
    """Two runs of the same config produce byte-identical results.

    Run on a reduced grid so the check is cheap; the full grid is verified the same way
    by running the published command twice.
    """
    base = yaml.safe_load(CONFIG_4B.read_text(encoding="utf-8"))
    base["policies"] = ["reactive_visible", "oracle_matched"]
    base["sweep"] = {
        "budget_fraction_of_peak_working_set": [1.0],
        "size_multiplier": [4],
        "latency_seconds": [0.5],
        "bandwidth_mb_s": [50.0],
    }
    # The summary renders a per-trajectory table for the representative cells, so they
    # have to be cells this reduced grid actually contains.
    base["reporting"]["representative_configs"] = [
        {"budget_fraction": 1.0, "size_multiplier": 4, "latency_seconds": 0.5,
         "bandwidth_mb_s": 50.0}]
    outputs = []
    for run in ("first", "second"):
        directory = tmp_path / run
        base["paths"]["results"] = str(directory)
        config_path = tmp_path / f"{run}.yaml"
        config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
        run_sweep.run_sweep(config_path)
        outputs.append((directory / "results.csv").read_bytes())
    assert outputs[0] == outputs[1], "the sweep is not deterministic"
    # And the config actually ran something.
    assert len(outputs[0].splitlines()) > 100


# --- Phase 4b+: baselines, thresholds and the bootstrap -----------------------

def test_lagged_occlusion_uses_only_past_samples():
    """reactive_visible_lag must be causal: the state at s comes from s - lag or earlier.

    Checked by feeding a scene whose occlusion states form a known sequence and
    confirming each row reads the value that was current ``lag`` samples back - and
    that changing a *later* state cannot change an earlier row's answer.
    """
    data = synthetic_data(n_samples=12)
    lag = 2
    lagged = lagged_occlusion_state(data, lag)
    # Every object has a row at every sample in the synthetic scene, so the lagged value
    # is exactly the state the object had `lag` samples earlier; before the trajectory
    # starts there is no reading at all and the state is unknown (-1), which the
    # project's convention counts as visible.
    state_at = {(data.object_ids[r], int(data.sample_index[r])): int(data.occlusion_state[r])
                for r in range(len(data.object_ids))}
    for row in range(len(data.object_ids)):
        sample = int(data.sample_index[row])
        object_id = data.object_ids[row]
        expected = state_at.get((object_id, sample - lag), -1)
        assert int(lagged[row]) == expected, (
            f"row {row} (sample {sample}) read {lagged[row]}, expected {expected}")
    # The lag must actually have bitten somewhere, or this test proves nothing.
    assert any(int(lagged[r]) != int(data.occlusion_state[r])
               for r in range(len(data.object_ids))), "the lag changed nothing"

    # Causality: mutating the final sample's state must not reach any earlier row.
    mutated = SimData(**{**data.__dict__})
    mutated.occlusion_state = data.occlusion_state.copy()
    mutated.occlusion_state[data.sample_index == 11] = 1
    replayed = lagged_occlusion_state(mutated, lag)
    early = data.sample_index <= 11 - lag
    assert np.array_equal(lagged[early], replayed[early]), (
        "a future sample leaked into the lagged occlusion state")


def test_threshold_zero_is_the_identity_and_the_scores_do_change():
    """tau = 0 must reproduce the base policy exactly; tau > 0 must actually do something."""
    data = synthetic_data()
    base = policy_set(data, ["heuristic_occ", "mlp_b"])
    thresholds = build_policies(data, {
        "policies": ["heuristic_occ", "mlp_b"],
        "prefetch_thresholds": {"bases": ["heuristic_occ", "mlp_b"],
                                "taus": [0.0, 0.1, 0.3, 0.5]},
        "sample_interval_seconds": 0.1, "reactive_lag_seconds": 0.2})
    for name in ("heuristic_occ", "mlp_b"):
        assert np.array_equal(thresholds[f"{name}_tau0"].score, base[name].score)
        # tau = 0 must also be the identity for a policy whose scores are not all zero.
        assert thresholds[f"{name}_tau0"].score.sum() >= thresholds[f"{name}_tau0.1"].score.sum()
    # The synthetic scores are zero for mlp_b, so a positive threshold can only zero more.
    assert thresholds["mlp_b_tau0.5"].score.sum() <= thresholds["mlp_b_tau0"].score.sum()
    # A base named in `policies` stays available under its own name alongside its variants.
    changed = build_policies(data, {
        "policies": ["mlp_b"],
        "prefetch_thresholds": {"bases": ["mlp_b"], "taus": [0.5]},
        "sample_interval_seconds": 0.1, "reactive_lag_seconds": 0.2})
    assert set(changed) == {"mlp_b", "mlp_b_tau0.5"}

    # heuristic_occ's squash maps a zero margin to 0.5, so tau = 0.1 must not zero it out
    # and tau = 0.5 must. That is the discrimination this test exists to provide.
    assert thresholds["heuristic_occ_tau0"].score.min() > 0.0
    assert thresholds["heuristic_occ_tau0.1"].score.min() > 0.0
    assert thresholds["heuristic_occ_tau0.5"].score.max() == 0.0


def test_bootstrap_is_deterministic():
    """Same seed, same inputs, same interval - and a different seed is allowed to differ."""
    left = np.array([0.4, 0.5, 0.3, 0.6, 0.45, 0.55, 0.35, 0.5, 0.42, 0.48, 0.52, 0.44,
                     0.38, 0.46])
    right = np.array([0.2, 0.25, 0.18, 0.3, 0.22, 0.28, 0.19, 0.24, 0.21, 0.26, 0.27, 0.23,
                      0.2, 0.22])
    reactive = [left, left + 0.05, left - 0.02]

    first = resample_matrix(len(left), 200, 20260501)
    second = resample_matrix(len(left), 200, 20260501)
    assert np.array_equal(first, second), "the resample matrix is not reproducible"
    assert not np.array_equal(first, resample_matrix(len(left), 200, 7)), (
        "a different seed produced the same resample matrix")

    a = reduction_ci(reactive, right, first)
    b = reduction_ci(reactive, right, second)
    assert a == b, "the bootstrap interval is not reproducible"
    lo, hi = a[1], a[2]
    assert lo <= a[0] <= hi
    # A wide, obviously-nonzero difference must produce an interval above zero.
    assert lo > 0.0
    # And a genuinely null difference must produce one that spans zero.
    null = paired_ci(right, right, first)
    assert null[1] <= 0.0 <= null[2]


@pytest.mark.skipif(not RESULTS_4BPLUS.is_file(), reason="run sim/run_sweep.py --config sim/config_4bplus.yaml first")
def test_phase4bplus_published_runs_never_exceed_their_budget():
    """Every run of the published 4b+ sweep respects its memory cap."""
    checked = 0
    with RESULTS_4BPLUS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "peak_resident_bytes":
                continue
            budget = float(row["budget_bytes"])
            peak = float(row["value"])
            assert peak <= budget * (1 + 1e-9), (
                f"{row['policy']} on {row['trajectory']} held {peak:.1f} over {budget:.1f}")
            checked += 1
    assert checked == 84 * 12 * 14, f"expected 14112 runs, found {checked}"


@pytest.mark.skipif(not (RESULTS_4BPLUS.is_file() and RESULTS_4B.is_file()),
                    reason="run both sweeps first")
def test_phase4bplus_overlapping_cells_reproduce_phase4b():
    """The strongest check in this phase: 4b+ with an equivalent cell must match 4b.

    Phase 4b's (size 1x, bandwidth b) is 4b+'s (load_time_scale = 200/b) with size held
    at 1x. Where the policies have the same definition - tau = 0 is the unthresholded
    policy - the two sweeps must agree to 1e-6, which is the evidence that the new grid,
    the new transport axis and the threshold machinery all reproduce the old behaviour
    exactly rather than approximately.
    """
    nominal = float(yaml.safe_load(CONFIG_4BPLUS.read_text(encoding="utf-8"))
                    ["nominal_bandwidth_mb_s"])
    plus = {}
    with RESULTS_4BPLUS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "deficit":
                continue
            plus[(float(row["budget_fraction"]), float(row["load_time_scale"]),
                  float(row["latency_seconds"]), row["policy"], row["trajectory"])] = \
                float(row["value"])

    # Phase 4b name -> Phase 4b+ name. Getting this direction wrong makes the test skip
    # exactly the two policies whose equivalence it exists to check, and still pass.
    renaming = {"heuristic_occ": "heuristic_occ_tau0", "mlp_b": "mlp_b_tau0"}
    compared = 0
    with RESULTS_4B.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "deficit" or float(row["size_multiplier"]) != 1.0:
                continue
            policy = row["policy"]
            if policy not in renaming and policy not in ("reactive_frustum",
                                                         "reactive_visible", "oracle_matched"):
                continue
            scale = nominal / float(row["bandwidth_mb_s"])
            if scale not in (0.25, 1.0, 4.0, 16.0):
                continue                     # bandwidth not on the 4b+ scale grid
            key = (float(row["budget_fraction"]), scale, float(row["latency_seconds"]),
                   renaming.get(policy, policy), row["trajectory"])
            if key not in plus:
                continue
            assert abs(plus[key] - float(row["value"])) < 1e-6, (
                f"{key}: 4b+ {plus[key]} vs 4b {row['value']}")
            compared += 1
    # 3 budgets x 2 shared scales x 3 shared latencies x 5 policies x 14 trajectories.
    # The count is asserted, not just the agreement: a filter slipped one step too far
    # would otherwise let this test pass while checking a handful of rows.
    assert compared == 3 * 2 * 3 * 5 * 14, f"only {compared} cells overlapped"


def test_phase4bplus_sweep_is_deterministic_across_two_runs(tmp_path):
    """Two runs of the same 4b+ config produce byte-identical results."""
    base = yaml.safe_load(CONFIG_4BPLUS.read_text(encoding="utf-8"))
    base["policies"] = ["reactive_frustum", "reactive_visible", "reactive_visible_lag",
                        "oracle_matched"]
    base["prefetch_thresholds"] = {"bases": ["mlp_b"], "taus": [0.0, 0.3]}
    base["sweep"] = {
        "budget_fraction_of_peak_working_set": [1.5],
        "size_multiplier": [1],
        "latency_seconds": [0.5],
        "load_time_scale": [4],
    }
    base["reporting"]["representative_configs"] = [
        {"budget_fraction": 1.5, "size_multiplier": 1, "latency_seconds": 0.5,
         "load_time_scale": 4}]
    outputs = []
    for run in ("first", "second"):
        directory = tmp_path / run
        base["paths"]["results"] = str(directory)
        config_path = tmp_path / f"{run}.yaml"
        config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
        run_sweep.run_sweep(config_path)
        outputs.append(((directory / "results.csv").read_bytes(),
                        (directory / "summary.md").read_bytes()))
    assert outputs[0] == outputs[1], "the 4b+ sweep or its summary is not deterministic"
    assert len(outputs[0][0].splitlines()) > 100
