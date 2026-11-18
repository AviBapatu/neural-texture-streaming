class_name StreamMath
extends RefCounted
## Float64 ports of the geometry helpers in training/features.py and
## training/baselines.py. Everything is plain double arithmetic (not Vector3, whose
## components are float32) so the heuristic port can agree with Python closely
## enough that priorities - and therefore eviction order - do not drift.

## Rotation matrix (world-from-local) for a Godot quaternion [x, y, z, w].
## Row-major 3x3, identical to features.quaternion_to_matrix.
static func quaternion_to_matrix(q: Array) -> Array:
	var x := float(q[0])
	var y := float(q[1])
	var z := float(q[2])
	var w := float(q[3])
	return [
		[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
		[2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
		[2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
	]


## Axis-angle rotation matrix, row-major. Mirrors baselines._rotation_matrices.
static func rotation_matrix(axis: Array, angle: float) -> Array:
	var x := float(axis[0])
	var y := float(axis[1])
	var z := float(axis[2])
	var c := cos(angle)
	var s := sin(angle)
	var C := 1.0 - c
	return [
		[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
		[y * x * C + z * s, c + y * y * C, y * z * C - x * s],
		[z * x * C - y * s, z * y * C + x * s, c + z * z * C],
	]


static func mat_mul(a: Array, b: Array) -> Array:
	var out := [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
	for i in 3:
		for j in 3:
			out[i][j] = (float(a[i][0]) * float(b[0][j])
				+ float(a[i][1]) * float(b[1][j])
				+ float(a[i][2]) * float(b[2][j]))
	return out


## local = M^T v  (component i is the dot of M's i-th column with v).
static func mat_transpose_mul(m: Array, v: Array) -> Array:
	return [
		float(m[0][0]) * float(v[0]) + float(m[1][0]) * float(v[1]) + float(m[2][0]) * float(v[2]),
		float(m[0][1]) * float(v[0]) + float(m[1][1]) * float(v[1]) + float(m[2][1]) * float(v[2]),
		float(m[0][2]) * float(v[0]) + float(m[1][2]) * float(v[1]) + float(m[2][2]) * float(v[2]),
	]


## Angular velocity from consecutive orientations, world rad/s, shorter arc.
## Mirrors features.angular_velocity_from_quaternions.
static func angular_velocity_from_quaternions(quaternions: Array, times: PackedFloat64Array) -> Array:
	var count := quaternions.size()
	var angular: Array = []
	for i in count:
		angular.append([0.0, 0.0, 0.0])
	if count < 2:
		return angular
	for i in range(1, count):
		var prev: Array = quaternions[i - 1]
		var cur: Array = quaternions[i]
		var inv := [-float(prev[0]), -float(prev[1]), -float(prev[2]), float(prev[3])]
		var d := _quat_mul(inv, cur)
		var norm := sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
		if norm < 1e-12:
			norm = 1.0
		for k in 4:
			d[k] = float(d[k]) / norm
		if float(d[3]) < 0.0:
			for k in 4:
				d[k] = -float(d[k])
		var dt := float(times[i]) - float(times[i - 1])
		if dt <= 0.0:
			continue
		var angle := 2.0 * acos(clampf(float(d[3]), -1.0, 1.0))
		var sin_half := sqrt(maxf(0.0, 1.0 - float(d[3]) * float(d[3])))
		if sin_half <= 1e-8:
			continue
		var scale := angle / dt / sin_half
		angular[i] = [float(d[0]) * scale, float(d[1]) * scale, float(d[2]) * scale]
	return angular


static func _quat_mul(a: Array, b: Array) -> Array:
	var ax := float(a[0])
	var ay := float(a[1])
	var az := float(a[2])
	var aw := float(a[3])
	var bx := float(b[0])
	var by := float(b[1])
	var bz := float(b[2])
	var bw := float(b[3])
	return [
		aw * bx + ax * bw + ay * bz - az * by,
		aw * by - ax * bz + ay * bw + az * bx,
		aw * bz + ax * by - ay * bx + az * bw,
		aw * bw - ax * bx - ay * by - az * bz,
	]
