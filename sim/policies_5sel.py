"""Phase 5sel: a per-step byte allowance (rank-and-cap selector) and a ranking model.

Two things live here, both new in Phase 5sel and both imported by ``sim/run_5sel.py``:

* ``simulate_selector`` - a byte-for-byte copy of ``sim.simulate.simulate`` whose only
  change is that new requests are throttled by a per-step allowance ``A``. With
  ``allowance_bytes = inf`` it reproduces the Phase 4b+ engine exactly, which the tests
  check; that reproduction is what makes every other number in this phase comparable to
  Phase 4b+.

* the P3 ranking model - a dense 1-logit network trained to rank a sample's candidates
  by future-occlusion coverage per byte, with a listwise (ListNet) and a pairwise-hinge
  objective, three seeds, and leave-trajectory-out out-of-fold scoring so no trajectory
  that supplies a simulator score was ever seen in training.

Nothing here writes to any existing file. The protocol is
``docs/phase-5sel-protocol.md``; it was written before this ran.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sim.policies import build_policies, matched_priorities, time_to_next_visible
from sim.simulate import StepMetrics, SimData, build_sim_data, load_labels_by_key

# --- the rank-and-cap selector -------------------------------------------------

def select_requests(object_priority: dict, resident: dict, in_flight: dict,
                    sizes: dict, allowance: float) -> tuple[list[str], float]:
    """The objects that receive a *new* request this step, and the bytes committed.

    ``object_priority`` maps every object present this sample to its priority. A
    candidate is an object with priority > 0 that is neither ``resident`` nor
    ``in_flight`` (already requested and not yet completed or evicted); resident and
    in-flight objects therefore cost nothing and are never re-requested while in flight.

    Candidates are taken in (priority desc, object id asc) order and requested until the
    cumulative full-texture size reaches ``allowance``: the requested set is the shortest
    prefix whose running total reaches ``allowance`` (or every candidate, if their total
    is below it). ``allowance`` is a stop threshold, so the crossing object is included
    and the top-ranked candidate is never skipped - a hard ``sum <= allowance`` ceiling
    would leave any texture larger than ``allowance`` permanently unloadable.
    ``allowance = inf`` requests every candidate; ``allowance <= 0`` requests none.
    """
    ranked = sorted(((oid, float(value)) for oid, value in object_priority.items()
                     if float(value) > 0.0 and oid not in resident and oid not in in_flight),
                    key=lambda item: (-item[1], item[0]))
    requested: list[str] = []
    spent = 0.0
    for oid, _value in ranked:
        if spent >= allowance:
            break
        requested.append(oid)
        spent += float(sizes[oid])
    return requested, spent


def simulate_selector(view, priorities: np.ndarray, sizes: dict[str, float],
                      ground_truth: np.ndarray, coverage: np.ndarray,
                      object_ids: np.ndarray, config,
                      allowance_bytes: float = math.inf,
                      stop_sample: int | None = None,
                      trace: list | None = None) -> StepMetrics:
    """Replay one trajectory, throttling new requests to ``allowance_bytes`` per step.

    Identical to ``sim.simulate.simulate`` apart from the request block. Eviction,
    completion, the residency cap and the deficit measurement are unchanged. When
    ``allowance_bytes`` is infinite every object with priority > 0 is requested, exactly
    as the original engine does, so the two agree bit for bit.

    ``trace``, when given, collects one record per sample describing the ranking and the
    selection, so the tests can assert the allowance and ordering invariants on a real
    replay rather than only on the helper.
    """
    dt = config.sample_interval
    resident: dict[str, float] = {}
    resident_bytes = 0.0
    loaded: dict[str, float] = {}
    requested_at: dict[str, float] = {}
    ever_visible: dict[str, bool] = {}
    episode_open: dict[str, int] = {}
    extended: set[str] = set()

    metrics = StepMetrics()

    def close_episode(object_id: str) -> None:
        if object_id in episode_open:
            metrics.pop_in_durations.append(episode_open.pop(object_id))

    def evict(object_id: str) -> None:
        nonlocal resident_bytes
        resident_bytes -= resident.pop(object_id)
        if not ever_visible.get(object_id, False):
            metrics.wasted_prefetches += 1
        requested_at.pop(object_id, None)

    for sample in view.samples:
        if stop_sample is not None and sample >= stop_sample:
            break
        t = sample * dt
        rows = view.rows_by_sample[sample]
        scored = sample not in view.excluded_samples
        metrics.steps += 1
        metrics.scored_steps += int(scored)

        object_priority: dict[str, float] = {}
        row_of: dict[str, int] = {}
        for row in rows:
            oid = object_ids[row]
            value = float(priorities[row])
            if value > object_priority.get(oid, 0.0):
                object_priority[oid] = value
                row_of[oid] = row

        # --- requests, throttled by the per-step allowance
        requested, spent = select_requests(object_priority, resident, requested_at,
                                           sizes, allowance_bytes)
        if trace is not None:
            ranked = sorted(((oid, value) for oid, value in object_priority.items()
                             if value > 0.0 and oid not in resident
                             and oid not in requested_at),
                            key=lambda item: (-item[1], item[0]))
            trace.append({"sample": sample, "allowance": float(allowance_bytes),
                          "ranked": ranked, "requested": list(requested), "spent": spent})
        for oid in requested:
            requested_at.setdefault(oid, t)

        # --- stream the step's bandwidth, in priority order (unchanged)
        available = config.bandwidth_bytes_per_s * dt
        eligible = [oid for oid, since in requested_at.items()
                    if oid not in resident and t - since >= config.latency_seconds - 1e-9]
        eligible.sort(key=lambda o: (-object_priority.get(o, 0.0), requested_at[o], o))

        completing: list[str] = []
        for oid in eligible:
            if available <= 0.0:
                break
            need = sizes[oid] - loaded.get(oid, 0.0)
            if need <= 0.0:
                completing.append(oid)
                continue
            take = min(need, available)
            loaded[oid] = loaded.get(oid, 0.0) + take
            metrics.bytes_streamed += take
            available -= take
            if sizes[oid] - loaded[oid] <= 1e-9:
                completing.append(oid)

        # --- complete loads, enforcing the budget by eviction (unchanged)
        for oid in completing:
            size = sizes[oid]
            if size > config.budget_bytes:
                loaded.pop(oid, None)
                requested_at.pop(oid, None)
                continue
            if resident_bytes + size > config.budget_bytes:
                evictable = sorted(
                    (o for o in resident if object_priority.get(o, 0.0)
                     < object_priority.get(oid, 0.0)),
                    key=lambda o: (object_priority.get(o, 0.0), requested_at.get(o, 0.0), o))
                for victim in evictable:
                    if resident_bytes + size <= config.budget_bytes:
                        break
                    evict(victim)
            if resident_bytes + size <= config.budget_bytes:
                resident[oid] = size
                resident_bytes += size
                ever_visible.setdefault(oid, False)
                loaded.pop(oid, None)
                requested_at.pop(oid, None)
                metrics.loads_completed += 1
            else:
                loaded.pop(oid, None)
                requested_at.pop(oid, None)

        metrics.peak_resident_bytes = max(metrics.peak_resident_bytes, resident_bytes)

        extended.clear()
        for row in rows:
            if not ground_truth[row]:
                continue
            oid = object_ids[row]
            covered = float(coverage[row])
            if scored:
                metrics.visible_coverage += covered
            if oid in resident:
                ever_visible[oid] = True
                close_episode(oid)
            else:
                if scored:
                    metrics.deficit += covered
                extended.add(oid)
                if oid not in episode_open:
                    metrics.pop_in_events += 1
                    episode_open[oid] = 0
                episode_open[oid] += 1

        for oid in list(episode_open):
            if oid not in extended:
                close_episode(oid)

    for oid in list(episode_open):
        close_episode(oid)
    for oid in list(resident):
        evict(oid)

    return metrics


# --- priority arrays -----------------------------------------------------------

BASE_POLICY_CONFIG = {
    "policies": ["reactive_frustum", "reactive_visible"],
    "prefetch_thresholds": {"bases": ["heuristic_occ", "mlp_b"],
                            "taus": [0.0, 0.1, 0.3, 0.5]},
}


def fixed_priorities(data: SimData) -> dict[str, np.ndarray]:
    """Priority arrays for every configuration-independent policy, by name."""
    policies = build_policies(data, BASE_POLICY_CONFIG)
    rows = np.arange(len(data.object_ids))
    return {name: policy.priority(rows) for name, policy in policies.items()}


def oracle_matched_priority(data: SimData, size_per_row: np.ndarray, latency: float,
                            bandwidth_bytes_per_s: float, margin: float) -> np.ndarray:
    """The Phase 4b+ matched-oracle reference priorities (uncapped, never selected)."""
    return matched_priorities(data.ground_truth_visible,
                              time_to_next_visible(data, 0.1), size_per_row,
                              latency, bandwidth_bytes_per_s, margin)


def sigmoid(scores: np.ndarray) -> np.ndarray:
    """Monotone map to (0, 1): ordering preserved, every candidate stays requestable."""
    return 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float64)))


# --- P3: the ranking model -----------------------------------------------------

GAIN_FIELD = "occlusion_max_coverage_1_0s"
HIDDEN = [128, 64]
DROPOUT = 0.1
SEEDS = (1, 2, 3)
MAX_EPOCHS = 60
GROUPS_PER_BATCH = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MIN_EPOCHS = 3
NDCG_K = 10
HINGE_MARGIN = 1.0
LOSS_NAMES = ("listwise", "hinge")

_GROUP_SCALE = 1_000_000


def gain_target(table, dataset_dir: str | Path, size_bytes: dict[str, float]) -> np.ndarray:
    """Future occlusion-aware coverage at 1.0 s divided by the object's full bytes.

    The label rows carry ``occlusion_max_coverage_1_0s`` on every row and it is never
    null, so no proxy is needed. Coverage per byte is the quantity a byte-overhead
    question wants to rank on.
    """
    lookup = load_labels_by_key(dataset_dir, (GAIN_FIELD,))
    gain = np.zeros(len(table), dtype=np.float64)
    missing = 0
    for index in range(len(table)):
        key = (table.trajectories[table.trajectory[index]],
               int(table.sample_index[index]), table.object_id[index])
        hit = lookup.get(key)
        if hit is None or hit[0] is None:
            missing += 1
            continue
        gain[index] = float(hit[0]) / float(size_bytes[table.object_id[index]])
    if missing:
        raise RuntimeError(f"{missing} rows had no {GAIN_FIELD} label")
    return gain


def group_codes(table) -> np.ndarray:
    """A per-row group id: rows of the same (trajectory, sample) share one."""
    return (table.trajectory.astype(np.int64) * _GROUP_SCALE
            + table.sample_index.astype(np.int64))


def group_lists(codes: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Row-index arrays per group, restricted to ``mask`` (groups with no rows dropped)."""
    mask = np.asarray(mask, dtype=bool)
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return []
    order = selected[np.argsort(codes[selected], kind="stable")]
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    return list(np.split(order, boundaries))


def ndcg_at_k(scores: np.ndarray, gains: np.ndarray,
              groups: list[np.ndarray], k: int = NDCG_K) -> float:
    """Mean NDCG@k over groups whose total gain is positive."""
    values = []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for rows in groups:
        group_gain = gains[rows]
        if group_gain.sum() <= 0.0:
            continue
        order = np.argsort(-scores[rows], kind="stable")[:k]
        ranked = group_gain[order]
        dcg = float(np.sum(ranked * discounts[:ranked.size]))
        ideal = np.sort(group_gain)[::-1][:k]
        idcg = float(np.sum(ideal * discounts[:ideal.size]))
        values.append(dcg / idcg if idcg > 0.0 else 0.0)
    return float(np.mean(values)) if values else 0.0


def _listnet_loss(scores, group, gain):
    """ListNet softmax cross-entropy: target distribution is the normalised gain."""
    import torch
    count = int(group.max().item()) + 1
    device = scores.device
    total = torch.zeros(count, device=device).index_add_(0, group, gain)
    valid = total > 0.0
    share = torch.where(total[group] > 0.0,
                        gain / total[group].clamp_min(1e-30),
                        torch.zeros_like(gain))
    maximum = torch.full((count,), float("-inf"), device=device)
    maximum.index_reduce_(0, group, scores, "amax", include_self=True)
    maximum = torch.where(valid, maximum, torch.zeros_like(maximum))
    shifted = scores - maximum[group]
    exp_sum = torch.zeros(count, device=device).index_add_(0, group, torch.exp(shifted))
    log_sum = maximum + torch.log(exp_sum.clamp_min(1e-30))
    log_prob = scores - log_sum[group]
    contribution = torch.zeros(count, device=device).index_add_(0, group, share * log_prob)
    return -(contribution[valid].sum() / valid.sum().clamp_min(1))


def _hinge_loss(scores, group, gain, margin: float = HINGE_MARGIN):
    """Pairwise hinge over adjacent same-sample candidate pairs with differing gain."""
    import torch
    order = torch.argsort(group, stable=True)
    left, right = order[:-1], order[1:]
    same = group[left] == group[right]
    left, right = left[same], right[same]
    left_gain, right_gain = gain[left], gain[right]
    differs = left_gain != right_gain
    left, right = left[differs], right[differs]
    left_gain, right_gain = left_gain[differs], right_gain[differs]
    if left.numel() == 0:
        return scores.sum() * 0.0
    high = torch.where(left_gain > right_gain, left, right)
    low = torch.where(left_gain > right_gain, right, left)
    return torch.clamp(margin - (scores[high] - scores[low]), min=0.0).mean()


LOSSES = {"listwise": _listnet_loss, "hinge": _hinge_loss}


def train_rank_fold(X: np.ndarray, gain: np.ndarray, train_groups: list[np.ndarray],
                    val_rows: np.ndarray, val_groups_global: list[np.ndarray],
                    config: dict, seed: int, loss_name: str) -> dict:
    """Train one P3 fold with the named loss; early-stop on validation NDCG@10.

    Deterministic given ``seed``: the batch-group order comes from a numpy RNG seeded
    with the same value, and ``torch.manual_seed`` is set immediately before the model
    is built.
    """
    import torch
    from training.models import VisibilityMLP, count_parameters

    settings = config["training"]
    torch.manual_seed(int(seed))
    device = torch.device("cpu")
    model = VisibilityMLP(input_dim=X.shape[1], hidden=HIDDEN, output_dim=1,
                          dropout=DROPOUT, activation="relu").to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    all_features = torch.from_numpy(X)
    gain_tensor = torch.from_numpy(gain.astype(np.float32))

    local_of = np.full(X.shape[0], -1, dtype=np.int64)
    local_of[val_rows] = np.arange(val_rows.size)
    val_groups_local = [local_of[rows] for rows in val_groups_global]
    val_features = torch.from_numpy(X[val_rows])
    val_gain = gain[val_rows]

    rng = np.random.default_rng(int(seed))
    loss_fn = LOSSES[loss_name]
    best_state, best_epoch, best_ndcg, stale = None, -1, -math.inf, 0
    history: list[dict] = []

    def run_batch(batch_groups: list[np.ndarray]) -> float:
        index = np.concatenate(batch_groups)
        group = np.concatenate([np.full(rows.size, position, dtype=np.int64)
                                for position, rows in enumerate(batch_groups)])
        permutation = rng.permutation(index.size)
        index, group = index[permutation], group[permutation]
        feature_index = torch.from_numpy(index)
        group_tensor = torch.from_numpy(group)
        scores = model(all_features[feature_index]).squeeze(-1)
        loss = loss_fn(scores, group_tensor, gain_tensor[feature_index])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return float(loss.item())

    max_epochs = int(settings.get("max_epochs", MAX_EPOCHS))
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(train_groups))
        epoch_loss, batches = 0.0, 0
        pending: list[np.ndarray] = []
        for position in order:
            pending.append(train_groups[position])
            if len(pending) >= GROUPS_PER_BATCH:
                epoch_loss += run_batch(pending)
                batches += 1
                pending = []
        if pending:
            epoch_loss += run_batch(pending)
            batches += 1

        model.eval()
        with torch.no_grad():
            val_scores = model(val_features).squeeze(-1).numpy()
        ndcg = ndcg_at_k(val_scores, val_gain, val_groups_local, NDCG_K)
        history.append({"epoch": epoch, "loss": epoch_loss / max(1, batches),
                        "val_ndcg": ndcg})
        if ndcg > best_ndcg + 1e-9:
            best_ndcg, best_epoch = ndcg, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE and epoch >= MIN_EPOCHS:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        raw = model(all_features).squeeze(-1).numpy()
    return {"raw_scores": raw.astype(np.float64), "model": model,
            "best_epoch": best_epoch, "best_val_ndcg": best_ndcg, "history": history,
            "parameters": count_parameters(model), "loss": loss_name, "seed": int(seed)}


def normalise_fold(train_matrix: np.ndarray, train_rows: np.ndarray,
                   full_matrix: np.ndarray):
    """Per-fold standardisation fitted on the training rows only."""
    subset = train_matrix[train_rows]
    mean = subset.mean(axis=0)
    std = subset.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return ((full_matrix - mean) / std).astype(np.float32), mean, std
