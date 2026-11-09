"""Leakage, boundary, mask, metric and determinism tests.

The central test is :func:`test_features_are_truncation_invariant`: rebuild the
features for a row using a copy of the trajectory with every sample after that row
deleted, and require the result to be identical. Any feature that reads the future
fails it. :func:`test_truncation_test_detects_an_injected_future_feature` is the
negative control - it injects exactly such a feature and asserts the comparison
catches it, so the passing test is known to have teeth.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import dataset as D
from training import evaluate as E
from training import features as F
from training import run_baselines as R
from training import splits as S


# --- the leakage test ---------------------------------------------------------

def _catalogue(config):
    return F.ObjectCatalogue.from_manifest(config["paths"]["object_index"])


def _rows_by_key(rows, meta):
    return {(m["sample_index"], m["object_id"]): row for row, m in zip(rows, meta)}


def test_features_are_truncation_invariant(fixture_dir, config):
    """Features for a row must not change when all later samples are deleted."""
    trajectory_id = "t06_look_around"
    samples = D.load_labels(fixture_dir / f"{trajectory_id}.samples.jsonl")
    labels = D.load_labels(fixture_dir / f"{trajectory_id}.labels.jsonl")
    catalogue = _catalogue(config)
    allow = list(config["features"]["allow"])
    history_k = int(config["history_k"])

    full_rows, full_meta = _build(samples, labels, catalogue, history_k, allow)
    full = _rows_by_key(full_rows, full_meta)

    checked = 0
    # Sample indices spread through the trajectory, always leaving the truncated
    # copy long enough to compute history (index >= history_k).
    for sample_index in (5, 17, 40, 88, 150, len(samples) - 1):
        labels_up_to = [row for row in labels if row["sample_index"] <= sample_index]
        truncated_rows, truncated_meta = _build(
            samples[:sample_index + 1], labels_up_to, catalogue, history_k, allow)
        truncated = _rows_by_key(truncated_rows, truncated_meta)
        for key, row in truncated.items():
            if key[0] != sample_index:
                continue
            reference = full[key]
            for name, value in row.items():
                other = reference[name]
                if isinstance(value, float) and isinstance(other, float):
                    assert abs(value - other) < 1e-12, (
                        f"feature {name} for {key} changed when the future was removed "
                        f"({value} vs {other}) - it reads a later sample")
                else:
                    assert value == other, (
                        f"feature {name} for {key} changed when the future was removed")
            checked += 1
            break
    assert checked == 6, f"expected to check 6 rows, checked {checked}"


def _build(samples, labels, catalogue, history_k, allow):
    return D.build_trajectory_rows(samples, labels, catalogue, history_k, allow)


def _build_with_injected_future(samples, labels, catalogue, history_k, allow, offset=1):
    """A deliberately leaking builder, used only as a negative control.

    Adds a feature that reads the *session time of a later sample*. Truncating the
    trajectory removes that sample, so the injected feature must change. A leak of
    the next sample's session time is certain to be detectable (it moves by 0.1 s),
    which is what a negative control needs - a leaked coverage value could
    coincidentally be equal between two adjacent samples and make the control pass
    for the wrong reason.
    """
    rows, meta = D.build_trajectory_rows(samples, labels, catalogue, history_k, allow)
    for row, entry in zip(rows, meta):
        victim = min(int(entry["sample_index"]) + offset, len(samples) - 1)
        row["leak_future_session_time"] = float(samples[victim]["session_time"])
    return rows, meta


def test_truncation_test_detects_an_injected_future_feature(fixture_dir, config):
    """Negative control: the leakage test must fail on a feature that reads the future."""
    trajectory_id = "t06_look_around"
    samples = D.load_labels(fixture_dir / f"{trajectory_id}.samples.jsonl")
    labels = D.load_labels(fixture_dir / f"{trajectory_id}.labels.jsonl")
    catalogue = _catalogue(config)
    allow = list(config["features"]["allow"]) + ["leak_future_session_time"]
    history_k = int(config["history_k"])

    full_rows, full_meta = _build_with_injected_future(samples, labels, catalogue, history_k, allow)
    full = _rows_by_key(full_rows, full_meta)

    sample_index = 40
    labels_up_to = [row for row in labels if row["sample_index"] <= sample_index]
    truncated_rows, truncated_meta = _build_with_injected_future(
        samples[:sample_index + 1], labels_up_to, catalogue, history_k, allow)
    truncated = _rows_by_key(truncated_rows, truncated_meta)

    differences = 0
    checked = 0
    for key, row in truncated.items():
        if key[0] != sample_index:
            continue
        checked += 1
        if row["leak_future_session_time"] != full[key]["leak_future_session_time"]:
            differences += 1
    assert checked > 0, "no rows were compared"
    assert differences > 0, (
        "the truncation comparison did NOT detect an injected future feature - "
        "the leakage test has no teeth")


def test_truncation_test_detects_leakage_in_the_production_builder(monkeypatch, fixture_dir, config):
    """The strongest form of the negative control.

    The injected feature is added *inside* the real geometry function rather than
    wrapped around the row builder, so this exercises exactly the code path the
    passing truncation test covers. A leak of the next sample's session time is used
    because it changes every sample regardless of whether the camera is moving -
    leaking a position instead would be undetectable on a stationary trajectory and
    the control would pass for the wrong reason.
    """
    trajectory_id = "t06_look_around"
    samples = D.load_labels(fixture_dir / f"{trajectory_id}.samples.jsonl")
    labels = D.load_labels(fixture_dir / f"{trajectory_id}.labels.jsonl")
    catalogue = _catalogue(config)
    allow = list(config["features"]["allow"])
    history_k = int(config["history_k"])

    original = F.geometry_features

    def leaky(catalogue_, sample_features, sample_index, object_ids):
        output = original(catalogue_, sample_features, sample_index, object_ids)
        victim = min(sample_index + 1, sample_features.count - 1)   # reads the FUTURE
        output["leaked_future_time"] = np.full(len(object_ids),
                                               float(sample_features.times[victim]))
        return output

    monkeypatch.setattr(F, "geometry_features", leaky)
    # build_trajectory_rows resolves the module attribute at call time, so the patch
    # takes effect without touching dataset.py.
    assert D.F.geometry_features is leaky

    sample_index = 25
    comparison = allow + ["leaked_future_time"]
    full_rows, full_meta = _build(samples, labels, catalogue, history_k, comparison)
    full = _rows_by_key(full_rows, full_meta)

    labels_up_to = [row for row in labels if row["sample_index"] <= sample_index]
    truncated_rows, truncated_meta = _build(
        samples[:sample_index + 1], labels_up_to, catalogue, history_k, comparison)
    truncated = _rows_by_key(truncated_rows, truncated_meta)

    checked = differences = 0
    for key, row in truncated.items():
        if key[0] != sample_index:
            continue
        checked += 1
        if row["leaked_future_time"] != full[key]["leaked_future_time"]:
            differences += 1
    assert checked > 0
    assert differences == checked, (
        "a feature reading the next sample was not detected by the truncation "
        "comparison - the leakage test has no teeth")


def test_history_never_crosses_a_trajectory_boundary(fixture_dir, config):
    """Every trajectory starts with an invalid history; none is inherited."""
    catalogue = _catalogue(config)
    history_k = int(config["history_k"])
    allow = ["hist_valid_1", "hist_valid_2", "hist_valid_3", "hist_valid_4"]
    for trajectory_id in ("t01_walk_straight", "t06_look_around"):
        samples = D.load_labels(fixture_dir / f"{trajectory_id}.samples.jsonl")
        labels = D.load_labels(fixture_dir / f"{trajectory_id}.labels.jsonl")
        rows, meta = D.build_trajectory_rows(samples, labels, catalogue, history_k, allow)
        for row, entry in zip(rows, meta):
            index = int(entry["sample_index"])
            for k in range(1, history_k + 1):
                expected = 1.0 if index >= k else 0.0
                assert row[f"hist_valid_{k}"] == expected, (
                    f"{trajectory_id} sample {index}: hist_valid_{k} is "
                    f"{row[f'hist_valid_{k}']}, expected {expected}")


def test_no_forbidden_column_reaches_the_feature_matrix(fixture_dir, config):
    """The allow-list is the enforcement; this checks nothing forbidden slipped in."""
    table = D.build_table(fixture_dir, config)
    forbidden_exact = {
        "object_id", "trajectory_id", "sample_index", "timestamp", "session_time",
        "physics_tick", "cam_position", "cam_quaternion", "cam_direction", "cam_euler",
        "position", "world_bounds", "split", "resolution_eligible",
        "resolution_variants", "variant_texture_ids", "coverage_0_5s",
    }
    forbidden_suffixes = ("_in_0_5s", "_in_1_0s", "_0_5s", "_1_0s")

    overlap = forbidden_exact.intersection(table.names)
    assert not overlap, f"forbidden columns in the feature matrix: {sorted(overlap)}"

    suffixed = [name for name in table.names if name.endswith(forbidden_suffixes)]
    assert not suffixed, f"future-derived columns in the feature matrix: {suffixed}"

    # Nothing outside the allow-list may appear either.
    allow = list(config["features"]["allow"])
    assert set(table.names) <= set(allow) | {n for n in table.names if n.startswith("tex_")}


def test_features_do_not_read_the_target_columns(fixture_dir, config):
    """Deleting every label column from the input must not change the features.

    The feature builder is called directly, with the label dictionaries stripped of
    all target fields. A feature that read a target would raise or change; the target
    builder itself is checked separately since it obviously *does* read them.
    """
    trajectory_id = "t06_look_around"
    samples = D.load_labels(fixture_dir / f"{trajectory_id}.samples.jsonl")
    labels = D.load_labels(fixture_dir / f"{trajectory_id}.labels.jsonl")
    catalogue = _catalogue(config)
    allow = list(config["features"]["allow"])
    history_k = int(config["history_k"])

    target_markers = ("_in_0_5s", "_in_1_0s", "first_visible_time", "max_coverage",
                      "required_resolution_", "resolution_insufficient",
                      "occlusion_unknown_in", "occlusion_visible_")
    stripped = [{k: v for k, v in row.items() if not any(m in k for m in target_markers)}
                for row in labels]
    removed = sum(len(row) - len(clean) for row, clean in zip(labels, stripped))
    assert removed > 0, "no target columns were stripped; the test is vacuous"

    full_rows, _ = D.build_trajectory_rows(samples, labels, catalogue, history_k, allow)
    clean_rows, _ = D.build_trajectory_rows(samples, stripped, catalogue, history_k, allow)

    assert len(full_rows) == len(clean_rows)
    for index, (a, b) in enumerate(zip(full_rows, clean_rows)):
        assert a == b, f"row {index} changed when the target columns were removed"


# --- masks --------------------------------------------------------------------

def test_unknown_occlusion_rows_contribute_nothing(fixture_dir, config):
    """Unknown occlusion is excluded from metrics and counted, never negative."""
    table = D.build_table(fixture_dir, config)
    for column in range(len(table.horizons)):
        y_true, valid, _ = E.target_arrays(table, "occlusion", column)
        unknown = table.y_occlusion[:, column] == D.OCCLUSION_UNKNOWN
        assert unknown.sum() > 0, "fixture has no unknown occlusion rows"
        assert not valid[unknown].any()
        assert not y_true[unknown].any()

        score = np.random.default_rng(0).random(len(table))
        predicted = score > 0.5
        masked = E.binary_metrics(y_true[valid], predicted[valid], score[valid])
        # Flipping every unknown row's prediction must not move the masked metrics.
        flipped = predicted.copy()
        flipped[unknown] = ~flipped[unknown]
        again = E.binary_metrics(y_true[valid], flipped[valid], score[valid])
        assert masked == again


def test_resolution_ineligible_rows_contribute_nothing(fixture_dir, config):
    """Only eligible rows carry a resolution target, so only they can be scored."""
    table = D.build_table(fixture_dir, config)
    ineligible = ~table.resolution_eligible
    assert ineligible.sum() > 0, "fixture has no ineligible rows"
    assert np.isnan(table.y_resolution[ineligible]).all()


# --- metrics ------------------------------------------------------------------

def test_binary_metrics_match_a_naive_implementation():
    rng = np.random.default_rng(7)
    y_true = rng.random(2000) < 0.3
    score = rng.random(2000)
    predicted = score > 0.6

    result = E.binary_metrics(y_true, predicted, score)

    tp = sum(1 for t, p in zip(y_true, predicted) if t and p)
    fp = sum(1 for t, p in zip(y_true, predicted) if not t and p)
    fn = sum(1 for t, p in zip(y_true, predicted) if t and not p)
    tn = sum(1 for t, p in zip(y_true, predicted) if not t and not p)
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (tp, fp, tn, fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    assert abs(result["precision"] - precision) < 1e-12
    assert abs(result["recall"] - recall) < 1e-12
    assert abs(result["f1"] - f1) < 1e-12

    assert abs(result["pr_auc"] - _naive_average_precision(y_true, score)) < 1e-9


def _naive_average_precision(y_true, score):
    """Average precision by explicit threshold sweep, for cross-checking."""
    thresholds = sorted(set(score), reverse=True)
    positives = sum(y_true)
    if positives == 0:
        return float("nan")
    previous_recall = 0.0
    area = 0.0
    for threshold in thresholds:
        selected = score >= threshold
        true_positive = sum(1 for t, s in zip(y_true, selected) if t and s)
        if true_positive == 0:
            continue
        precision = true_positive / sum(selected)
        recall = true_positive / positives
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def test_average_precision_is_nan_without_positives():
    assert np.isnan(E.average_precision(np.zeros(10, dtype=bool), np.arange(10, dtype=float)))


def test_recall_at_k_matches_a_naive_implementation(fixture_dir, config):
    table = D.build_table(fixture_dir, config)
    rng = np.random.default_rng(11)
    score = rng.random(len(table))
    y_true = table.y_frustum[:, 0]

    for k in (5, 10, 20):
        assert abs(E.recall_at_k(table, y_true, score, k)
                   - _naive_recall_at_k(table, y_true, score, k)) < 1e-12


def _naive_recall_at_k(table, y_true, score, k):
    groups = {}
    for index in range(len(table)):
        key = (int(table.trajectory[index]), int(table.sample_index[index]))
        groups.setdefault(key, []).append(index)
    values = []
    for key in sorted(groups):
        members = groups[key]
        positives = [i for i in members if y_true[i]]
        if not positives:
            continue
        ranked = sorted(members, key=lambda i: -score[i])[:k]
        values.append(sum(1 for i in ranked if y_true[i]) / len(positives))
    return float(np.mean(values)) if values else float("nan")


def test_resolution_metrics_under_prediction_rate():
    y_true = np.array([256, 256, 512, 1024, 128])
    y_pred = np.array([256, 128, 512, 2048, 128])
    result = E.resolution_metrics(y_true, y_pred)
    # one under (256 vs 128), one over (2048 vs 1024), three exact
    assert result["under_prediction_rate"] == pytest.approx(1 / 5)
    assert result["over_prediction_rate"] == pytest.approx(1 / 5)
    assert result["exact_resolution_rate"] == pytest.approx(3 / 5)
    # macro-F1 must only average classes with support
    assert set(result["classes_with_support"]) == {128, 256, 512, 1024}
    assert result["macro_f1"] > 0.5


# --- splits -------------------------------------------------------------------

def test_folds_never_overlap(fixture_dir, config):
    """No trajectory may appear in more than one partition of any fold."""
    folds = S.build_folds(sorted(["t01", "t02", "t03", "t04", "t05", "t06", "t07", "t08",
                                  "t09", "t10", "t11", "t12", "t13", "t14"]),
                          folds=7, fold_size=2, seed=20261105)
    assert len(folds) == 7
    seen_as_test = []
    for fold in folds:
        S.assert_disjoint(sorted(set(fold.train) | set(fold.validation) | set(fold.test)),
                          {"train": fold.train, "validation": fold.validation, "test": fold.test},
                          context=f"fold {fold.index}")
        seen_as_test.extend(fold.test)
    assert sorted(seen_as_test) == sorted(f"t{i:02d}" for i in range(1, 15)), \
        "every trajectory should be held out as test exactly once"

    with pytest.raises(ValueError):
        S.assert_disjoint(["a", "b"], {"train": ["a"], "test": ["a"]})


def test_real_splits_are_disjoint(split_manifest_path):
    assignment = S.load_split_manifest(split_manifest_path)
    partitions: dict[str, list[str]] = {}
    for trajectory_id, name in assignment.items():
        partitions.setdefault(name, []).append(trajectory_id)
    S.assert_disjoint(sorted(assignment), partitions)
    assert sorted(partitions) == ["test", "train", "validation"]


# --- determinism --------------------------------------------------------------

def test_running_twice_gives_identical_results(tmp_path, fixture_dir, fixture_split, tiny_config):
    """The whole harness is deterministic for a fixed config and dataset."""
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config), encoding="utf-8")

    common = ["--config", str(config_path), "--dataset", str(fixture_dir),
              "--phase2_5-split", str(fixture_split)]
    first = R.run(common + ["--output", str(tmp_path / "a")])
    second = R.run(common + ["--output", str(tmp_path / "b")])

    assert first["tuning"]["chosen"] == second["tuning"]["chosen"]
    assert first["row_count"] == second["row_count"]
    assert first["features"] == second["features"]
    assert np.isfinite(first["tuning"]["chosen_validation_score"]["frustum"]), \
        "selection score is not finite; the fixture split left validation empty"
    assert json.dumps(first["final_test"], sort_keys=True) == \
        json.dumps(second["final_test"], sort_keys=True), "final test results differ between runs"

    written = json.loads((tmp_path / "a" / "results.json").read_text(encoding="utf-8"))
    assert written["tuning"]["chosen"] == first["tuning"]["chosen"]
    assert (tmp_path / "a" / "results.csv").read_text(encoding="utf-8") == \
        (tmp_path / "b" / "results.csv").read_text(encoding="utf-8")
