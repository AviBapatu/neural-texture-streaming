# Experimental split v1 - design and rationale

**This rationale was written before any baseline metric was computed on the split.**
The manifest `dataset/manifests/experimental_split_v1.json` was written to disk first:

```
sha256  a3cb2243ec890e70c4f1719a8d69ef78f0b3c8370d37c32174559511fa4873c8
mtime   2026-11-06 12:06:16
```

The results in the last section were produced afterwards and the split was not adjusted
after seeing them. The original Phase 2.5 manifest
(`dataset/manifests/runs/run_full_p25/splits.json`) is untouched - verified by checksum
immediately after writing split v1:

```
$ sha256sum -c before_split.sha256
dataset/manifests/runs/run_full_p25/splits.json: OK
```

## Why a new split at all

The Phase 2.5 split (8 train / 3 validation / 3 test) has two problems that Phase 3a
measured rather than suspected:

1. **The test set is the easy set.** Its three trajectories
   (`t11_slow_walk_down`, `t13_approach_prop`, `t03_walk_corner`) are all walking pace.
   Phase 3a found the heuristic scores PR-AUC 0.590 on average across folds but only
   0.189 on that fixed test set - a threefold gap that is mostly the test set being
   unrepresentative, not the fold estimate being optimistic.
2. **It shares `t13_approach_prop` with the snap contamination discovered in Phase 3a.1.**
   Scoring on a trajectory with a generator defect mixes an artefact into the headline
   number.

## Selection inputs

Only these were used:

| Input | Source |
|---|---|
| movement type | `dataset/trajectories/full/index.json` (declared kind) |
| duration | same |
| snap status | Phase 3a.1 Task A audit |

No baseline metric, label distribution or model score was consulted, because none had
been computed for this split.

## Test set - the four required roles

| Role | Trajectory | Kind | Evidence from the audit |
|---|---|---|---|
| sprinting | `t04_sprint_corner` | sprint_corner | mean speed 18.2 m/s, peak 84.1 m/s |
| pure rotation / look-around | `t06_look_around` | look_around | **stationary** (mean speed 0.0 m/s) with the highest sustained rotation of any trajectory (mean 26.8 deg/s) |
| translation-dominant | `t01_walk_straight` | walk_straight | constant 6 m/s with **exactly zero** rotation for all 180 samples |
| walking | `t03_walk_corner` | walk_corner | 6 m/s with one corner (mean rotation 4.1 deg/s) |

`t06` and `t14` share the `look_around` kind; `t06` was chosen for test because `t14` is
snap-contaminated and `t06` is not, and because `t06` is the purer rotation case
(stationary versus `t14`'s mean speed of 0.2 m/s with a snap).

## Validation set - diversity over regimes

| Trajectory | Kind | Regime |
|---|---|---|
| `t05_turn_left_right` | turn_left_right | rotation while moving (mean rotation 14.9 deg/s, mean speed 4.6 m/s) |
| `t07_stop_start` | stop_start | intermittent translation, almost no rotation (mean rotation 0.9 deg/s) |
| `t08_forward_reverse` | forward_reverse | translation including reverse (mean speed 5.4 m/s) |

Three different motion regimes from three different singleton kinds, none snap-contaminated.

## Training set

`t02`, `t09`, `t10`, `t11`, `t12`, `t13`, `t14`.

It retains every movement type that has more than one member, so no *replicated*
behaviour is missing from training:

| Type | Kept in training because |
|---|---|
| walk_straight | `t11` (test holds out `t01`) |
| sprint_corner | `t12` (test holds out `t04`) |
| look_around | `t14` (test holds out `t06`) |

## The singleton trade-off, stated plainly

**Eight of the eleven movement types are represented by exactly one trajectory:**

`sprint_straight`, `walk_corner`, `turn_left_right`, `stop_start`, `forward_reverse`,
`approach_building`, `street_then_plaza`, `approach_prop`.

Only three types have a second member (`walk_straight`, `sprint_corner`, `look_around`).
The test and validation sets together must hold out **seven** trajectories, and at most
three of those can come from the replicated types. Therefore **at least four singleton
types cannot appear in training**, and this split holds out exactly four:

`walk_corner` (test), `turn_left_right` (validation), `stop_start` (validation),
`forward_reverse` (validation).

**Trade-off.** Holding out a singleton type means the model never sees that behaviour
during training, so any score for it measures *generalisation to unseen behaviour*
rather than interpolation. That is arguably the more useful and more honest number - a
model that only works on behaviours it has memorised is not useful - but it means
`walk_corner` test metrics are not comparable to Phase 2.5's, where `walk_corner` was in
training. Any comparison between split v1 and the Phase 2.5 split must account for this.

The alternative - keeping all singleton types in training - would force the test set to
be drawn only from `walk_straight`, `sprint_corner` and `look_around`, which cannot
supply a walking-and-cornering trajectory and would make validation redundant with test.

## Snap-contaminated trajectories: placement and policy

All three snap-contaminated trajectories (`t09_approach_building`,
`t13_approach_prop`, `t14_plaza_look`) are placed in **training**.

Reason: Phase 3a.1 Task A concludes the three yaw snaps are **generator defects**, not
genuine camera motion - the runner assigns `player.rotation.y` absolutely on a
`control` segment with no continuity constraint against the heading the preceding
`waypoint` segment produced. Placing defective trajectories in test or validation would
score the baselines partly on artefacts.

Because contaminated trajectories can still put contaminated *rows* into training, the
recommended policy - applied here to metrics, and recommended for Phase 3b training -
is:

- **exclude snap-affected rows from headline metrics** and report them separately;
- exclude the same rows from training, so the model is not taught that a 180 deg camera
  rotation in 0.1 s is a normal event.

`snap_affected` is available per `(trajectory, sample)` in
`training/experiments/phase3a1/eval_flags.csv`, with `snap_affected_yaw` and
`snap_affected_position` split out. The split matters: 115 of the 118 flagged samples
are position jumps in the sprint trajectories that are **consistent with the recorded
velocity** (20-120 m/s real motion), not defects. Only the 3 yaw snaps are defects.

## Reporting protocol for Phase 3b

**Primary - leave-trajectory-out grouped cross-validation across all 14 trajectories:**

- 7 folds of 2, deterministic seed, as in Phase 3a.
- Report per-fold, then per-trajectory and per-movement-type tables.
- Report **mean and standard deviation** across folds.
- The fold spread is the headline uncertainty; a single point estimate is not.
- Fold variance conflates movement-type composition with difficulty, because the folds
  are not stratified by kind.

**Secondary - fixed split v1 test set:**

- Scored **once, at the very end**, after every design decision is frozen.
- Four trajectories, 4+3+7 partition, as recorded above.
- Snap-affected rows excluded from the headline and reported separately.
- Because four singleton types are absent from training, these numbers measure
  generalisation to unseen behaviour. State that whenever they are quoted.

**Tuning:** hyper-parameters are selected on the **validation** partition of whichever
protocol is active, never on test. In the primary protocol, a trajectory's rows may not
influence the parameters used to score that trajectory: use out-of-fold parameters per
fold.

**Never:** report a number computed with parameters that saw the test trajectories, or
report a split-v1 number after adjusting split v1.

## Guarantees this split satisfies

| Requirement | Status |
|---|---|
| Complete trajectories only | yes - 14 whole trajectories |
| train / validation / test disjoint | yes - verified programmatically |
| Together cover all 14 | yes - 7 + 3 + 4 = 14 |
| Test includes a sprinting trajectory | yes - `t04_sprint_corner` |
| Test includes a pure rotation / look-around | yes - `t06_look_around` |
| Test includes a translation-dominant trajectory | yes - `t01_walk_straight` |
| Test includes a walking trajectory | yes - `t03_walk_corner` |
| Validation diverse | yes - three regimes |
| Training retains every replicated movement type | yes - `walk_straight`, `sprint_corner`, `look_around` |
| Singleton types held out are declared | yes - four, with the trade-off above |
| Original manifest untouched | yes - checksum verified |

## Results on split v1 (computed after the above was recorded)

See the "Split v1" section of [phase-3a1-report.md](phase-3a1-report.md) and
`training/experiments/phase3a1/split_v1_baselines.json`. The split was not modified
after these numbers were produced.
