class_name HeuristicOcc
extends RefCounted
## Port of the frozen occlusion-tuned heuristic: training/baselines.py `margin_deg`
## (lines 154-190) and `predictive_score` (lines 253-269), with the parameters the
## project froze in training/experiments/phase3a/results.json:
##   damping = 1.0, angular_damping = 0.0, integrate_steps = 2,
##   occlusion_penalty_deg = 30.0, margin threshold = +10 deg.
##
## With damping = 1.0 the camera does not translate (1 - damping = 0), so only the
## rotation term matters; that is kept general here rather than special-cased.

const SIGMA_DEG := 10.0
const THRESHOLD_DEG := 10.0
const OCCLUSION_PENALTY_DEG := 30.0
const HORIZON_SECONDS := 0.5
const INTEGRATE_STEPS := 2
const DAMPING := 1.0
const ANGULAR_DAMPING := 0.0

const OCCLUSION_FALSE := 0


## Max frustum margin over the forward offsets, in degrees. Positive = inside.
static func margin_deg(position: Array, matrix: Array, centre: Array, velocity: Array,
		angular_velocity: Array, half_vertical: float, half_horizontal: float,
		horizon: float = HORIZON_SECONDS, integrate_steps: int = INTEGRATE_STEPS,
		damping: float = DAMPING, angular_damping: float = ANGULAR_DAMPING) -> float:
	var steps := maxi(1, integrate_steps)
	var speed := sqrt(float(angular_velocity[0]) * float(angular_velocity[0])
		+ float(angular_velocity[1]) * float(angular_velocity[1])
		+ float(angular_velocity[2]) * float(angular_velocity[2]))
	var moving := speed > 1e-9
	var safe_speed := speed if moving else 1.0
	var axis := [float(angular_velocity[0]) / safe_speed,
		float(angular_velocity[1]) / safe_speed, float(angular_velocity[2]) / safe_speed]
	var best := -INF
	for k in steps:
		var offset := horizon * float(k + 1) / float(steps)
		var position_at: Array = position
		var matrix_at: Array = matrix
		if offset > 0.0:
			position_at = [
				float(position[0]) + float(velocity[0]) * (1.0 - damping) * offset,
				float(position[1]) + float(velocity[1]) * (1.0 - damping) * offset,
				float(position[2]) + float(velocity[2]) * (1.0 - damping) * offset,
			]
			if moving:
				var angle := speed * (1.0 - angular_damping) * offset
				matrix_at = StreamMath.mat_mul(StreamMath.rotation_matrix(axis, angle), matrix)
			else:
				matrix_at = matrix
		var rel := [float(centre[0]) - float(position_at[0]),
			float(centre[1]) - float(position_at[1]), float(centre[2]) - float(position_at[2])]
		var local := StreamMath.mat_transpose_mul(matrix_at, rel)
		var azimuth := rad_to_deg(atan2(float(local[0]), -float(local[2])))
		var elevation := rad_to_deg(atan2(float(local[1]), sqrt(float(local[0]) * float(local[0]) + float(local[2]) * float(local[2]))))
		var value := minf(half_vertical - absf(elevation), half_horizontal - absf(azimuth))
		best = maxf(best, value)
	return best


## Half-angles from the vertical FOV and the viewport aspect (baselines.py:46-53).
static func half_angles(fov_degrees: float, width: float, height: float) -> Array:
	var vertical := deg_to_rad(fov_degrees)
	var aspect := (width / height) if height else 1.0
	var half_v := fov_degrees * 0.5
	var half_h := rad_to_deg(atan(tan(vertical * 0.5) * aspect))
	return [half_v, half_h]


## Raw occlusion-aware score (margin, minus the penalty when definitively occluded).
static func raw_score(margin: float, occlusion_state: int,
		occlusion_penalty_deg: float = OCCLUSION_PENALTY_DEG) -> float:
	return (margin - occlusion_penalty_deg) if occlusion_state == OCCLUSION_FALSE else margin


## Monotone squash into (0, 1); order-preserving, so ranking is unchanged.
static func squash(margin_deg_value: float, sigma: float = SIGMA_DEG) -> float:
	return 1.0 / (1.0 + exp(-margin_deg_value / sigma))


## The frozen heuristic as a priority: 2.0 when visible now, else the squashed score.
static func priority(visible_now: bool, margin: float, occlusion_state: int) -> float:
	if visible_now:
		return 2.0
	return squash(raw_score(margin, occlusion_state))
