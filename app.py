from __future__ import annotations

from rendercanvas.auto import RenderCanvas, loop
import wgpu
import numpy as np
import PIL.Image as Image
import struct
import time
from pathlib import Path


# ---- ImGui UI (from reference project) ----
try:
    from imgui_bundle import imgui  # type: ignore
    from wgpu.utils.imgui import ImguiRenderer  # type: ignore

    HAS_IMGUI = True
except Exception:
    HAS_IMGUI = False


from primitives import cloth_grid, sphere
from camera import Camera

from scene.cloth import ClothSim
from scene.sphere import SphereMesh
from gpu.resources import load_texture_2d, read_wgsl


WORKGROUP_SIZE = 256

# Put your textures next to cloth_app.py, or change these paths
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SHADERS = ROOT / "shaders"

CLOTH_TEX_PATH = ASSETS / "texel_checker.png"
SPHERE_TEX_PATH = ASSETS / "db.png"  # if missing, we fallback to CLOTH_TEX_PATH


def make_cloth_wire_indices(width: int, height: int) -> np.ndarray:
    """Line indices to visualize the cloth triangle mesh (wireframe).

    For each quad cell we add the 4 edges + the diagonal used by the triangle split.
    Duplicates are fine (simple + fast) and still look good.
    """
    edges: list[int] = []
    w = width
    h = height
    for y in range(h - 1):
        for x in range(w - 1):
            i0 = y * w + x
            i1 = i0 + 1
            i2 = i0 + w
            i3 = i2 + 1
            # quad edges
            edges += [i0, i1, i0, i2, i1, i3, i2, i3]
            # diagonal matching the triangle split (i0,i2,i1) & (i1,i2,i3)
            edges += [i1, i2]
    return np.asarray(edges, dtype=np.uint32)


def make_initial_state(width: int, height: int, spacing: float, y0: float, mass: float = 1.0):
    """Returns pos (vec4) and vel (vec4).

    pos.xyz = position
    pos.w   = inverse mass (0 => pinned)
    """
    n = width * height
    pos = np.zeros((n, 4), dtype=np.float32)
    vel = np.zeros((n, 4), dtype=np.float32)

    invm = 1.0 / mass
    origin = np.array(
        [-0.5 * (width - 1) * spacing, y0, -0.5 * (height - 1) * spacing],
        dtype=np.float32,
    )
    for y in range(height):
        for x in range(width):
            i = y * width + x
            p = origin + np.array([x * spacing, 0.0, y * spacing], dtype=np.float32)
            pos[i, 0:3] = p
            # No pinning: every vertex is free
            pos[i, 3] = invm
    return pos, vel


class App:
    def __init__(self):
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync()

        self.size = (0, 0)
        self.canvas = RenderCanvas(
            size=(900, 650),
            title="Cloth Simulation (WebGPU)",
            update_mode="continuous",
            max_fps=60,
        )
        self.context = self.canvas.get_wgpu_context()

        self.render_format = self.context.get_preferred_format(self.device.adapter)
        self.context.configure(device=self.device, format=self.render_format)

        # ---- UI (ImGui) ----
        # ImguiRenderer handles its own render pass.
        self.imgui_renderer = ImguiRenderer(self.device, self.canvas) if HAS_IMGUI else None
        self.ui_rect = (10.0, 10.0, 360.0, 255.0)  # x, y, w, h (gate camera events)

        # ---- camera ----
        self.camera = Camera(45, 900 / 650, 0.1, 200, 4.0, np.pi / 4, np.pi / 6)
        self.canvas.add_event_handler(
            self.process_event, "pointer_up", "pointer_down", "pointer_move", "wheel"
        )  # type: ignore

        # ---- render bind group layout (uniform + texture + sampler) ----
        bg_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {}},
                {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
            ]
        )
        p_layout = self.device.create_pipeline_layout(bind_group_layouts=[bg_layout])

        # light(vec4) + view(mat4) + proj(mat4) + viewport(vec2) + point_size(f32) + pad = 160 bytes
        self.render_params_buffer = self.device.create_buffer(
            size=160, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        # Texture loader (see gpu/resources.py)

        # Two different textures
        self.tex_cloth = load_texture_2d(self.device, CLOTH_TEX_PATH, CLOTH_TEX_PATH)
        self.tex_sphere = load_texture_2d(self.device, SPHERE_TEX_PATH, CLOTH_TEX_PATH)

        self.sampler = self.device.create_sampler()

        # Two bind groups (same layout), one per object
        self.bind_group_cloth = self.device.create_bind_group(
            layout=bg_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.render_params_buffer,
                        "offset": 0,
                        "size": self.render_params_buffer.size,
                    },
                },
                {"binding": 1, "resource": self.tex_cloth.create_view()},
                {"binding": 2, "resource": self.sampler},
            ],
        )
        self.bind_group_sphere = self.device.create_bind_group(
            layout=bg_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.render_params_buffer,
                        "offset": 0,
                        "size": self.render_params_buffer.size,
                    },
                },
                {"binding": 1, "resource": self.tex_sphere.create_view()},
                {"binding": 2, "resource": self.sampler},
            ],
        )

        # ---- cloth simulation (geometry + buffers + compute pipelines) ----
        self.cloth_w = 80
        self.cloth_h = 60
        self.spacing = 0.05

        compute_code = read_wgsl(SHADERS / "cloth_compute.wgsl")
        self.cloth = ClothSim(
            self.device,
            cloth_w=self.cloth_w,
            cloth_h=self.cloth_h,
            spacing=self.spacing,
            y0=1.2,
            mass=0.25,
            compute_wgsl_code=compute_code,
        )

        self.nverts = self.cloth.nverts
        self.cloth_vertex_stride = self.cloth.vertex_stride

        self.cloth_vertex_buffer = self.cloth.vertex_buffer
        self.cloth_index_buffer = self.cloth.index_buffer
        self.cloth_index_count = self.cloth.index_count

        self.cloth_wire_index_buffer = self.cloth.wire_index_buffer
        self.cloth_wire_index_count = self.cloth.wire_index_count

        # Expose sim buffers/pipelines (used in the draw loop)
        self.pos_a = self.cloth.pos_a
        self.vel_a = self.cloth.vel_a
        self.pos_b = self.cloth.pos_b
        self.vel_b = self.cloth.vel_b
        self.sim_params_buffer = self.cloth.sim_params_buffer
        self.pipe_step = self.cloth.pipe_step
        self.pipe_write_vtx = self.cloth.pipe_write_vtx
        self.compute_bg_ab = self.cloth.compute_bg_ab
        self.compute_bg_ba = self.cloth.compute_bg_ba

# ---- sphere (render mesh + sim params) ----
        self.draw_sphere = True

        # skins
        # sphere: 0 textured, 1 solid
        # cloth : 0 textured, 1 wire mesh
        self.sphere_skin = 0
        self.cloth_skin = 0

        self.sphere_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.sphere_radius = 0.75
        self.sphere_radius_target = float(self.sphere_radius)
        self.sphere_radius_min = 0.25
        self.sphere_radius_max = 1.50
        self.sphere_radius_step = 0.05

        # Fix A: limit how fast radius can change (prevents energy injection)
        self.sphere_radius_max_speed = 0.30  # radius units per second

        # Sphere mesh at radius=1 centered at origin (we update positions on the GPU buffer)
        tmpl_vtx, sph_idx = sphere(48, 24, center=(0.0, 0.0, 0.0), radius=1.0)
        self.sphere_mesh = SphereMesh(self.device, tmpl_vtx, sph_idx)

        self.sphere_vertex_buffer = self.sphere_mesh.vertex_buffer
        self.sphere_index_buffer = self.sphere_mesh.index_buffer
        self.sphere_index_count = self.sphere_mesh.index_count

        # Upload initial sphere positions
        self.sphere_mesh.update(self.sphere_center, self.sphere_radius, force=True)

# ---- compute (cloth simulation) ----
        # The compute pipelines/bind groups are created inside ClothSim (scene/cloth.py).
        # We just reuse self.sim_params_buffer / self.pipe_step / self.pipe_write_vtx / self.compute_bg_ab / self.compute_bg_ba below.

# ---- render pipeline ----
        vertex_buffer_descriptor = {
            "array_stride": self.cloth_vertex_stride,
            "step_mode": wgpu.VertexStepMode.vertex,
            "attributes": [
                {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                {"format": wgpu.VertexFormat.float32x3, "offset": 3 * 4, "shader_location": 1},
                {"format": wgpu.VertexFormat.float32x2, "offset": 6 * 4, "shader_location": 2},
            ],
        }

        with open(SHADERS / "render.wgsl", "r") as file:
            rshader = self.device.create_shader_module(code=file.read())

        # Textured pipeline (shared for both cloth and sphere when skin is textured)
        self.pipeline_textured = self.device.create_render_pipeline(
            layout=p_layout,
            vertex={"module": rshader, "entry_point": "vs_main", "buffers": [vertex_buffer_descriptor]},
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,  # show cloth both sides
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            fragment={
                "module": rshader,
                "entry_point": "fs_main",
                "targets": [{"format": self.render_format}],
            },
        )

        # Solid-color pipeline (sphere skin without texture)
        solid_wgsl = """
struct RenderParams {
    light: vec4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec2<f32>,
    point_size: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: RenderParams;
// Keep bindings compatible with the existing bind group layout:
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var samplr: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip = params.proj * params.view * vec4<f32>(in.position, 1.0);
    out.position = in.position;
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(params.light.xyz - in.position);
    let shading = clamp(dot(light_dir, normalize(in.normal)), 0.15, 1.0);
    let base = vec3<f32>(0.85, 0.85, 0.90);
    return vec4<f32>(base * shading, 1.0);
}
"""
        solid_shader = self.device.create_shader_module(code=solid_wgsl)
        self.pipeline_sphere_solid = self.device.create_render_pipeline(
            layout=p_layout,
            vertex={"module": solid_shader, "entry_point": "vs_main", "buffers": [vertex_buffer_descriptor]},
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.back,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            fragment={
                "module": solid_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.render_format}],
            },
        )

        # Wireframe pipeline for cloth mesh view
        wire_wgsl = """
struct RenderParams {
    light: vec4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec2<f32>,
    point_size: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: RenderParams;
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var samplr: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip = params.proj * params.view * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(v: VertexOutput) -> @location(0) vec4<f32> {
    _ = v.clip; // phony use
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
}
"""
        wire_shader = self.device.create_shader_module(code=wire_wgsl)
        self.pipeline_cloth_wire = self.device.create_render_pipeline(
            layout=p_layout,
            vertex={"module": wire_shader, "entry_point": "vs_main", "buffers": [vertex_buffer_descriptor]},
            primitive={
                "topology": wgpu.PrimitiveTopology.line_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": False,
                "depth_compare": wgpu.CompareFunction.less_equal,
            },
            fragment={
                "module": wire_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.render_format}],
            },
        )

        
        # Points pipeline (debug cloth particles)
        vertex_buffer_descriptor_points = {
            "array_stride": 8 * 4,  # same vertex stride, but only read position
            "step_mode": wgpu.VertexStepMode.instance,
            "attributes": [
                {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
            ],
        }

        self.pipeline_cloth_points = self.device.create_render_pipeline(
            layout=p_layout,
            vertex={"module": rshader, "entry_point": "vs_points", "buffers": [vertex_buffer_descriptor_points]},
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": False,
                "depth_compare": wgpu.CompareFunction.less_equal,
            },
            fragment={
                "module": rshader,
                "entry_point": "fs_points",
                "targets": [{"format": self.render_format}],
            },
        )

# ---- simulation tuning ----
        self.substeps = 16
        self.k_struct = 1200.0
        self.k_shear = 900.0
        self.k_bend = 250.0
        self.c_d = 1.5
        self.vel_damping = 0.995
        self.friction = 0.2
        self.collision_eps = 0.02
        self.gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)

        self.t = 0.0
        self.last = time.perf_counter()

        # ---- reset support ----
        self._reset_requested = False
        self._default_sphere_radius = 0.75
        self._default_sphere_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._default_sphere_skin = 0
        self._default_cloth_skin = 0
        self._default_point_size = 6.0
        self._default_mass = 0.25
        self._default_y0 = 1.2
        self._cloth_vtx0 = self.cloth.vtx0.copy()

        # The draw loop updates the camera uniform every frame (including point_size).
        # Ensure this exists immediately, even before the first reset/UI interaction.
        self.point_size = float(self._default_point_size)

    def reset_all(self):
        # Reset time
        self.t = 0.0
        self.last = time.perf_counter()

        # Reset skins
        self.sphere_skin = int(self._default_sphere_skin)
        self.cloth_skin = int(self._default_cloth_skin)
        self.point_size = float(self._default_point_size)

        # Reset sphere params
        self.sphere_center = self._default_sphere_center.copy()
        self.sphere_radius_target = float(self._default_sphere_radius)
        self.sphere_radius = float(self._default_sphere_radius)

        # Upload sphere mesh immediately
        self.sphere_mesh.update(self.sphere_center, self.sphere_radius, force=True)

# Reset cloth simulation buffers (pos/vel) and visible mesh
        self.cloth.reset(y0=self._default_y0, mass=self._default_mass)

# Reset camera (optional but nice)
        self.camera = Camera(45, 900 / 650, 0.1, 200, 4.0, np.pi / 4, np.pi / 6)
        if self.size != (0, 0):
            self.camera.aspect = self.size[0] / self.size[1]

    def process_event(self, event):
        # Disable camera interaction over the UI area (and when ImGui wants mouse).
        if HAS_IMGUI:
            # Hard UI rectangle gate
            try:
                x = float(event.get("x", -1.0))
                y = float(event.get("y", -1.0))
                ux, uy, uw, uh = self.ui_rect
                if ux <= x <= (ux + uw) and uy <= y <= (uy + uh):
                    return
            except Exception:
                pass

            # ImGui capture flag (if available)
            try:
                io = imgui.get_io()
                if io.want_capture_mouse:
                    return
            except Exception:
                pass

        self.camera.process_event(event)

    def update_gui(self):
        if not HAS_IMGUI:
            return

        imgui.set_next_window_pos((self.ui_rect[0], self.ui_rect[1]), imgui.Cond_.always)
        imgui.set_next_window_size((self.ui_rect[2], self.ui_rect[3]), imgui.Cond_.always)

        imgui.begin("Controls", True)

        imgui.text("Sphere radius")
        changed, val = imgui.slider_float(
            "##SphereRadius",
            float(self.sphere_radius_target),
            float(self.sphere_radius_min),
            float(self.sphere_radius_max),
            "%.3f",
        )
        if changed:
            self.sphere_radius_target = float(val)

        if imgui.button("-"):
            self.sphere_radius_target = max(self.sphere_radius_min, self.sphere_radius_target - self.sphere_radius_step)
        imgui.same_line()
        if imgui.button("+"):
            self.sphere_radius_target = min(self.sphere_radius_max, self.sphere_radius_target + self.sphere_radius_step)

        imgui.text(f"R current: {self.sphere_radius:.3f}")
        imgui.text(f"R target : {self.sphere_radius_target:.3f}")

        imgui.separator()
        imgui.text("Skins")

        changed, skin = imgui.combo("Sphere", int(self.sphere_skin), ["Textured", "Solid"])
        if changed:
            self.sphere_skin = int(skin)

        changed, skin = imgui.combo("Cloth", int(self.cloth_skin), ["Textured", "Mesh", "Points"])
        if changed:
            self.cloth_skin = int(skin)

        imgui.text("Point size")
        changed, val = imgui.slider_float(
            "##PointSize",
            float(self.point_size),
            1.0,
            20.0,
            "%.1f",
        )
        if changed:
            self.point_size = float(val)


        imgui.separator()
        if imgui.button("Reset all"):
            self._reset_requested = True

        imgui.end()

    def loop(self):
        screen_texture: wgpu.GPUTexture = self.context.get_current_texture()  # type: ignore
        size = screen_texture.size
        if size[:2] != self.size:
            self.depth_texture = self.device.create_texture(
                size=size,
                format=wgpu.TextureFormat.depth32float,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self.size = size[:2]
            self.camera.aspect = size[0] / size[1]

        # Apply reset at frame start
        if self._reset_requested:
            self._reset_requested = False
            self.reset_all()

        now = time.perf_counter()
        dt_frame = min(1.0 / 60.0, now - self.last)
        self.last = now
        self.t += dt_frame
        dt = dt_frame / self.substeps

        # ---- sphere radius change: speed-limited (Fix A) ----
        dr = float(self.sphere_radius_target - self.sphere_radius)
        max_dr = float(self.sphere_radius_max_speed) * dt_frame
        if dr > max_dr:
            dr = max_dr
        if dr < -max_dr:
            dr = -max_dr
        self.sphere_radius = float(self.sphere_radius + dr)

        # Update sphere mesh (only uploads if changed)
        if self.draw_sphere:
            self.sphere_mesh.update(self.sphere_center, self.sphere_radius)

# ---- update render uniforms ----
        light_position = np.array([-10, 10, 10, 0], dtype=np.float32)  # vec4 for alignment
        proj_matrix, view_matrix = self.camera.get_matrices()
        render_params_data = (
            light_position.tobytes()
            + view_matrix.T.tobytes()
            + proj_matrix.T.tobytes()
            + np.array([float(self.size[0]), float(self.size[1])], dtype=np.float32).tobytes()
            + np.array([float(self.point_size), 0.0], dtype=np.float32).tobytes()
        )
        self.device.queue.write_buffer(self.render_params_buffer, 0, render_params_data)

        # ---- update sim uniforms (matches cloth_compute.wgsl) ----
        sim_bytes = struct.pack(
            "20f4I",
            # vec4 #0
            dt, self.t, self.spacing, self.c_d,
            # vec4 #1
            self.k_struct, self.k_shear, self.k_bend, self.vel_damping,
            # vec4 #2
            self.friction, self.collision_eps, 0.0, 0.0,
            # gravity vec4
            float(self.gravity[0]), float(self.gravity[1]), float(self.gravity[2]), 0.0,
            # sphere vec4
            float(self.sphere_center[0]), float(self.sphere_center[1]), float(self.sphere_center[2]), float(self.sphere_radius),
            # dims (u32)
            self.cloth_w, self.cloth_h, 0, 0,
        )
        self.device.queue.write_buffer(self.sim_params_buffer, 0, sim_bytes)

        encoder = self.device.create_command_encoder()

        # ---- compute pass (ping-pong) ----
        n_groups = (self.nverts + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        cpass = encoder.begin_compute_pass()

        cpass.set_pipeline(self.pipe_step)
        for _ in range(self.substeps):
            if self.cloth.ping == 0:
                cpass.set_bind_group(0, self.compute_bg_ab)
                cpass.dispatch_workgroups(n_groups, 1, 1)
                self.cloth.ping = 1
            else:
                cpass.set_bind_group(0, self.compute_bg_ba)
                cpass.dispatch_workgroups(n_groups, 1, 1)
                self.cloth.ping = 0

        # Write vertices from the CURRENT state
        cpass.set_pipeline(self.pipe_write_vtx)
        if self.cloth.ping == 0:
            cpass.set_bind_group(0, self.compute_bg_ab)  # pos_in = A
        else:
            cpass.set_bind_group(0, self.compute_bg_ba)  # pos_in = B
        cpass.dispatch_workgroups(n_groups, 1, 1)
        cpass.end()

        # ---- render pass ----
        rpass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": screen_texture.create_view(),
                    "clear_value": (0.9, 0.9, 0.9, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_texture.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )

        # Sphere first
        if self.draw_sphere:
            rpass.set_bind_group(0, self.bind_group_sphere)
            if int(self.sphere_skin) == 0:
                rpass.set_pipeline(self.pipeline_textured)
            else:
                rpass.set_pipeline(self.pipeline_sphere_solid)
            rpass.set_vertex_buffer(0, self.sphere_vertex_buffer)
            rpass.set_index_buffer(self.sphere_index_buffer, wgpu.IndexFormat.uint32)
            rpass.draw_indexed(self.sphere_index_count)

        # Cloth
        rpass.set_bind_group(0, self.bind_group_cloth)
        if int(self.cloth_skin) == 0:
            # Textured mesh
            rpass.set_pipeline(self.pipeline_textured)
            rpass.set_vertex_buffer(0, self.cloth_vertex_buffer)
            rpass.set_index_buffer(self.cloth_index_buffer, wgpu.IndexFormat.uint32)
            rpass.draw_indexed(self.cloth_index_count)
        elif int(self.cloth_skin) == 1:
            # Wireframe (lines)
            rpass.set_pipeline(self.pipeline_cloth_wire)
            rpass.set_vertex_buffer(0, self.cloth_vertex_buffer)
            rpass.set_index_buffer(self.cloth_wire_index_buffer, wgpu.IndexFormat.uint32)
            rpass.draw_indexed(self.cloth_wire_index_count)
        else:
            # Debug: points (one billboard quad per particle)
            rpass.set_pipeline(self.pipeline_cloth_points)
            rpass.set_vertex_buffer(0, self.cloth_vertex_buffer)
            rpass.draw(6, self.cloth_w * self.cloth_h)

        rpass.end()

        self.device.queue.submit([encoder.finish()])

        # Render ImGui UI
        if self.imgui_renderer is not None:
            self.imgui_renderer.render()

    def run(self):
        self.canvas.request_draw(self.loop)
        if self.imgui_renderer is not None:
            self.imgui_renderer.set_gui(self.update_gui)
        loop.run()


if __name__ == "__main__":
    App().run()
