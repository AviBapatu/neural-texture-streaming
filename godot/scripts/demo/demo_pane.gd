class_name DemoPane
extends Control
## One demo pane: its own World3D, environment instance, camera, streamer and HUD.
##
## The world content is created one frame after the SubViewport is in the tree, because
## a SubViewport does not expose a usable [member SubViewport.world_3d] during the same
## frame it is added; adding a WorldEnvironment before that leaves the pane unlit (black).
## After the world exists this code binds the environment explicitly and copies the
## main scene's lighting, so a pane renders exactly like the normal single-scene run.

const VARIANT_IDS := [
	"ambientcg/Asphalt021_Color", "ambientcg/Bricks066_Color", "ambientcg/Concrete012_Color",
	"ambientcg/Grass004_Color", "ambientcg/Ground037_Color", "ambientcg/Metal063_Color",
	"ambientcg/Road001_Color",
]
const UNTINTED := Color(1.0, 1.0, 1.0)
const TINT_LOWRES := Color(1.0, 0.30, 0.25)
const TINT_LOADING := Color(1.0, 0.85, 0.25)
const TINT_RESIDENT := Color(0.35, 1.0, 0.40)
const TINT_WASTED := Color(0.72, 0.35, 1.0)
const OUTLINE_SECONDS := 0.45

class LoadTask:
	extends RefCounted
	var path := ""
	var image: Image = null
	var ms := 0.0
	func run() -> void:
		var t0 := Time.get_ticks_usec()
		var img: Image = Image.load_from_file(path)
		ms = float(Time.get_ticks_usec() - t0) / 1000.0
		if img != null and not img.is_empty():
			img.generate_mipmaps()
			image = img

var side := "L"
var stream: DemoStream
var viewport: SubViewport
var camera: Camera3D
var world_env: WorldEnvironment
var sun: DirectionalLight3D

var streamable: Array = []
var by_id: Dictionary = {}

var tint_enabled := true
var outline_enabled := true
var textures_enabled := true
var quality_low := true

var world_ready := false
var lowres_count := 0
var _container: SubViewportContainer
var _hud_panel: PanelContainer
var _hud: Label
var _lowres: Label
var _legend: Label
var _env: Node3D
var _pending: Array = []
var _tasks: Dictionary = {}
var _outlines: Array = []
var _last_visible: Dictionary = {}
var _last_load_ms := 0.0
var _last_hitch_ms := 0.0
var _swap_count := 0
var _time := 0.0

var _init_manifest: Array = []
var _init_args: Dictionary = {}

func setup(pane_size: Vector2, p_side: String, manifest_objects: Array, stream_args: Dictionary,
		p_stretch_shrink: int, p_low_quality: bool) -> void:
	side = p_side
	quality_low = p_low_quality
	_init_manifest = manifest_objects
	_init_args = stream_args
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	clip_contents = true

	_container = SubViewportContainer.new()
	_container.name = "Pane" + side
	_container.stretch = true
	_container.stretch_shrink = maxi(1, p_stretch_shrink)
	_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_container.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(_container)

	viewport = SubViewport.new()
	viewport.own_world_3d = true
	viewport.size = Vector2i(pane_size)
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	viewport.msaa_3d = Viewport.MSAA_DISABLED
	viewport.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	viewport.use_debanding = false
	viewport.audio_listener_enable_3d = false
	_container.add_child(viewport)

	_build_hud()
	_create_world()

## Runs one idle frame after the viewport exists.
## A SubViewport does not create a World3D by itself here, and it has no environment until
## one is assigned, so make the world explicitly and bind the environment to it.
func _create_world() -> void:
	if viewport.world_3d == null:
		viewport.world_3d = World3D.new()

	var packed := load("res://scenes/environment.tscn") as PackedScene
	_env = packed.instantiate() as Node3D
	viewport.add_child(_env)

	# Copy the main scene's lighting into this pane, then apply the demo's quality settings.
	world_env = _env.get_node_or_null("WorldEnvironment") as WorldEnvironment
	if world_env != null:
		world_env.environment = world_env.environment.duplicate()
		viewport.world_3d.environment = world_env.environment
		_configure_environment(world_env.environment)
	sun = _env.get_node_or_null("Sun") as DirectionalLight3D
	if sun != null:
		_configure_light(sun)

	camera = Camera3D.new()
	camera.fov = 75.0
	camera.near = 0.05
	camera.far = 600.0
	camera.keep_aspect = Camera3D.KEEP_HEIGHT
	camera.current = true
	viewport.add_child(camera)

	_collect_streamable(_init_manifest)
	init_stream()
	world_ready = true
	print("[pane %s] world ready: env=%s sun=%s objects=%d shrink=%d lowq=%s bg_mode=%d sky=%s" % [
		side, str(viewport.world_3d.environment != null), str(sun != null),
		streamable.size(), _container.stretch_shrink, str(quality_low),
		(world_env.environment.background_mode if world_env != null else -1),
		str(world_env.environment.sky != null if world_env != null else false)])

func _configure_environment(environment: Environment) -> void:
	if environment == null:
		return
	environment.ssao_enabled = false
	environment.ssil_enabled = false
	environment.ssr_enabled = false
	environment.sdfgi_enabled = false
	environment.glow_enabled = false
	environment.fog_enabled = false
	environment.volumetric_fog_enabled = false
	if quality_low:
		environment.background_energy_multiplier = 1.0
		environment.ambient_light_energy = 0.9

func _configure_light(light: DirectionalLight3D) -> void:
	light.shadow_enabled = not quality_low
	light.directional_shadow_mode = DirectionalLight3D.SHADOW_ORTHOGONAL
	light.directional_shadow_max_distance = 120.0
	light.shadow_bias = 0.1

# --- world content ----------------------------------------------------------

func _collect_streamable(manifest_objects: Array) -> void:
	var by_oid := {}
	for entry in _env.get("tracked_objects"):
		by_oid[entry.get("object_id", "")] = entry
	for o in manifest_objects:
		var oid := String(o["object_id"])
		var entry: Dictionary = by_oid.get(oid, {})
		var node: Node3D = entry.get("node")
		if node == null or not is_instance_valid(node):
			continue
		var instances := node.find_children("*", "MeshInstance3D", true, false)
		if instances.is_empty():
			continue
		var mi: MeshInstance3D = instances[0]
		var original := mi.material_override as StandardMaterial3D
		if original != null:
			original = original.duplicate() as StandardMaterial3D
			mi.material_override = original
		var item := {
			"oid": oid, "node": node, "mi": mi,
			"source_path": String(o["source_path"]),
			"full_bytes": float(o["full_bytes"]),
			"centre": o["centre"],
		}
		streamable.append(item)
		by_id[oid] = item
		if textures_enabled and original != null:
			var tex := _load_variant(String(o["source_path"]), 256)
			if tex != null:
				original.albedo_texture = tex

func init_stream() -> void:
	if stream != null:
		return
	stream = DemoStream.new()
	stream.setup(
		String(_init_args.get("trajectory", "")),
		_init_args.get("samples", []), _init_args.get("rows", {}), _init_args.get("pose", {}),
		_init_args.get("snap", []), _init_args.get("sizes", {}),
		float(_init_args.get("budget", 1.0)), float(_init_args.get("bandwidth", 200.0e6)),
		float(_init_args.get("latency", 1.0)), String(_init_args.get("policy", "reactive_visible")))
	_apply_camera(0)

func set_policy(p_id: String) -> void:
	if stream != null:
		stream.set_policy(p_id)

func reset_stream() -> void:
	if stream == null:
		return
	stream.reset()
	_pending.clear()
	_tasks.clear()
	for item in streamable:
		var mat := (item["mi"] as MeshInstance3D).material_override as StandardMaterial3D
		if mat != null:
			mat.albedo_color = UNTINTED
	_apply_camera(0)

func advance(i: int) -> void:
	if stream == null:
		return
	stream.advance(i, Callable(self, "_on_resident"), Callable(self, "_on_evicted"))
	_last_visible.clear()
	for r in stream.rows_by_sample.get(str(int(stream.samples[i])), []):
		if int(r[2]) == 1 and int(r[3]) != 0:
			_last_visible[String(r[0])] = true
	_apply_tints()

func set_time(t: float) -> void:
	_time = t

## Scale the in-pane HUD with the window (fonts + panel margins).
func set_ui_scale(scale: float) -> void:
	if _hud == null:
		return
	_style(_hud, int(round(14.0 * scale)), Color(1, 1, 1))
	_style(_lowres, int(round(14.0 * scale)), _lowres.get_theme_color("font_color"))
	_style(_legend, int(round(11.0 * scale)), Color(0.75, 0.8, 0.9))
	_hud_panel.position = Vector2(10.0 * scale, 8.0 * scale)
	var style := _hud_panel.get_theme_stylebox("panel") as StyleBoxFlat
	if style != null:
		style.set_content_margin_all(8.0 * scale)

func total_swaps() -> int:
	return _swap_count

func has_pending_textures() -> bool:
	return not _pending.is_empty() or not _tasks.is_empty()

func container_size() -> Vector2:
	return _container.size

## Keep the SubViewport at the container's pixel size (SubViewportContainer only does this
## when it can, and we need it exact so the render fills the pane with no letterbox).
func sync_viewport_size() -> void:
	if _container == null or viewport == null:
		return
	var want := Vector2i((_container.size / float(_container.stretch_shrink)).floor())
	if want.x > 0 and want.y > 0 and viewport.size != want:
		viewport.size = want

func _process(delta: float) -> void:
	if stream == null:
		return
	_apply_camera(_time)
	_process_texture_queue()
	_age_outlines(delta)
	_update_hud()

func _physics_process(_delta: float) -> void:
	_gc_outlines()

# --- camera -----------------------------------------------------------------

func _apply_camera(t: float) -> void:
	if stream == null:
		return
	var n := stream.sample_count()
	if n == 0:
		return
	var f := clampf(t / DemoStream.SAMPLE_INTERVAL, 0.0, float(n - 1))
	var i0 := int(floor(f))
	var i1 := mini(i0 + 1, n - 1)
	var frac := f - float(i0)
	var a := stream.pose_at(i0)
	var b := stream.pose_at(i1)
	if a.is_empty():
		return
	var pa := _vec(a["position"])
	var qa := _quat(a["quaternion"])
	if b.is_empty() or i0 == i1 or frac <= 0.0:
		camera.global_transform = Transform3D(Basis(qa), pa)
		camera.fov = float(a.get("fov", 75.0))
		return
	var pb := _vec(b["position"])
	var qb := _quat(b["quaternion"])
	camera.global_transform = Transform3D(Basis(qa.slerp(qb, frac)), pa.lerp(pb, frac))
	camera.fov = lerpf(float(a.get("fov", 75.0)), float(b.get("fov", 75.0)), frac)

static func _vec(v: Array) -> Vector3:
	return Vector3(float(v[0]), float(v[1]), float(v[2]))

static func _quat(q: Array) -> Quaternion:
	return Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

# --- textures + tint --------------------------------------------------------

func _on_resident(oid: String) -> void:
	if not textures_enabled:
		return
	if not _pending.has(oid):
		_pending.append(oid)

func _on_evicted(oid: String) -> void:
	if not textures_enabled:
		return
	if _pending.has(oid):
		_pending.erase(oid)

## Recolour every streamed object by state. On (default) this is a diagnostic overlay:
## red = visible on the low-resolution placeholder, yellow = loading, green = resident
## full texture, purple = wasted prefetch. Off restores the natural look.
func _apply_tints() -> void:
	for item in streamable:
		var oid: String = item["oid"]
		var mat := (item["mi"] as MeshInstance3D).material_override as StandardMaterial3D
		if mat == null:
			continue
		if not tint_enabled:
			mat.albedo_color = UNTINTED
			continue
		if stream.state.resident.has(oid):
			mat.albedo_color = TINT_RESIDENT
		elif _last_visible.has(oid):
			mat.albedo_color = TINT_LOWRES
		elif stream.state.requested_at.has(oid):
			mat.albedo_color = TINT_LOADING
		elif not bool(stream.state.ever_visible.get(oid, true)):
			mat.albedo_color = TINT_WASTED
		else:
			mat.albedo_color = UNTINTED

func _process_texture_queue() -> void:
	if not textures_enabled:
		return
	for oid in _tasks.keys():
		var task: LoadTask = _tasks[oid]
		if task.image == null:
			continue
		var item: Dictionary = by_id.get(oid, {})
		_tasks.erase(oid)
		if item.is_empty():
			continue
		var hitch0 := Time.get_ticks_usec()
		var tex := ImageTexture.create_from_image(task.image)
		var mat := (item["mi"] as MeshInstance3D).material_override as StandardMaterial3D
		if mat != null:
			mat.albedo_texture = tex
			mat.albedo_color = TINT_RESIDENT if tint_enabled else UNTINTED
		_last_hitch_ms = float(Time.get_ticks_usec() - hitch0) / 1000.0
		_last_load_ms = task.ms
		_swap_count += 1
		if outline_enabled and _last_visible.has(oid):
			_flash_outline(item)
		return
	if _pending.is_empty():
		return
	var oid: String = _pending.pop_front()
	var item: Dictionary = by_id.get(oid, {})
	if item.is_empty():
		return
	var task := LoadTask.new()
	task.path = _resolve_asset(String(item["source_path"]))
	_tasks[oid] = task
	WorkerThreadPool.add_task(Callable(task, "run"))

func _load_full(source_path: String) -> Texture2D:
	var path := _resolve_asset(source_path)
	if not FileAccess.file_exists(path):
		return null
	var img: Image = Image.load_from_file(path)
	if img == null or img.is_empty():
		return null
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)

func force_resolution(res: int, oids: Array) -> int:
	_pending.clear()
	_tasks.clear()
	var applied := 0
	for oid in oids:
		var item: Dictionary = by_id.get(String(oid), {})
		if item.is_empty():
			continue
		var mat := (item["mi"] as MeshInstance3D).material_override as StandardMaterial3D
		if mat == null:
			continue
		var tex: Texture2D = _load_full(String(item["source_path"])) if res <= 0 \
			else _load_variant(String(item["source_path"]), res)
		if tex == null:
			continue
		mat.albedo_texture = tex
		mat.albedo_color = UNTINTED
		applied += 1
	return applied

func _load_variant(source_path: String, res: int) -> Texture2D:
	var path := _variant_path(source_path, res)
	if path.is_empty() or not FileAccess.file_exists(path):
		return null
	var img: Image = Image.load_from_file(path)
	if img == null or img.is_empty():
		return null
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)

func _variant_path(source_path: String, res: int) -> String:
	var rel := source_path.trim_prefix("res://")
	if rel.begins_with("godot/"):
		rel = rel.trim_prefix("godot/")
	if rel.is_empty() or not rel.begins_with("assets/"):
		return ""
	var dir := rel.get_base_dir()
	var stem := rel.get_file().get_basename()
	return _repo_root().path_join("dataset/textures/variants").path_join(dir.get_file()).path_join(
		"%s_%d.jpg" % [stem, res])

func _resolve_asset(source_path: String) -> String:
	if source_path.begins_with("res://"):
		return ProjectSettings.globalize_path(source_path)
	if source_path.begins_with("/"):
		return source_path
	return _repo_root().path_join(source_path)

static func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").simplify_path().path_join("..").simplify_path()

# --- outline flash ----------------------------------------------------------

func _flash_outline(item: Dictionary) -> void:
	var mi: MeshInstance3D = item["mi"]
	if mi.mesh == null:
		return
	_gc_outlines()
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = Color(1.0, 1.0, 1.0)
	mat.cull_mode = BaseMaterial3D.CULL_FRONT
	mat.grow = true
	mat.grow_amount = 0.05
	var outline := MeshInstance3D.new()
	outline.mesh = mi.mesh
	outline.material_override = mat
	outline.name = "SwapOutline"
	mi.add_child(outline)
	_outlines.append({"node": outline, "ttl": OUTLINE_SECONDS})

func _age_outlines(delta: float) -> void:
	for entry in _outlines:
		entry["ttl"] = float(entry["ttl"]) - delta

func _gc_outlines() -> void:
	var kept: Array = []
	for entry in _outlines:
		if float(entry["ttl"]) <= 0.0:
			var node: Node = entry["node"]
			if is_instance_valid(node):
				node.queue_free()
		else:
			kept.append(entry)
	_outlines = kept

# --- HUD --------------------------------------------------------------------

func _build_hud() -> void:
	_hud_panel = PanelContainer.new()
	_hud_panel.name = "HUDPanel" + side
	_hud_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_hud_panel.position = Vector2(10, 8)
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.02, 0.03, 0.05, 0.62)
	style.border_color = Color(0.5, 0.7, 1.0, 0.5)
	style.set_border_width_all(1)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(8)
	_hud_panel.add_theme_stylebox_override("panel", style)
	add_child(_hud_panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 2)
	_hud_panel.add_child(vbox)

	_hud = Label.new()
	_hud.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_style(_hud, 14, Color(1, 1, 1))
	vbox.add_child(_hud)

	_lowres = Label.new()
	_lowres.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_style(_lowres, 14, Color(1, 0.5, 0.5))
	vbox.add_child(_lowres)

	_legend = Label.new()
	_legend.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_style(_legend, 11, Color(0.75, 0.8, 0.9))
	_legend.text = "tint: low-res red · loading yellow · resident green · wasted purple"
	vbox.add_child(_legend)

func _style(label: Label, size: int, color: Color) -> void:
	label.add_theme_font_size_override("font_size", size)
	label.add_theme_color_override("font_color", color)
	label.add_theme_color_override("font_outline_color", Color(0, 0, 0))
	label.add_theme_constant_override("outline_size", 5)

func _update_hud() -> void:
	if _hud == null or stream == null:
		return
	var m := stream.summarise()
	var neural_tag := "  NEURAL (precomputed scores)" if stream.neural else ""
	_hud.text = "\n".join([
		"[%s] %s%s" % [side, stream.spec(stream.policy_id)["label"], neural_tag],
		"deficit %.4f    pop-in %d" % [m["deficit"], stream.pop_in_events()],
		"streamed %.0f MB    resident %.0f/%.0f MB" % [
			stream.bytes_streamed() / 1e6, stream.resident_bytes() / 1e6, stream.budget_bytes / 1e6],
		"wasted %d   swaps %d   load %.1fms" % [
			stream.wasted_prefetches(), _swap_count, _last_load_ms],
	])
	var low := 0
	for oid in _last_visible.keys():
		if not stream.state.resident.has(oid):
			low += 1
	lowres_count = low
	_lowres.text = "LOW-RES VISIBLE: %d objects" % low
	_lowres.add_theme_color_override("font_color",
		Color(1.0, 0.35, 0.3) if low > 0 else Color(0.5, 1.0, 0.6))
