from __future__ import annotations

import numpy as np
import wgpu

from primitives import cloth_grid


WORKGROUP_SIZE = 256


def make_cloth_wire_indices(width: int, height: int) -> np.ndarray:
    """Line indices to visualize the cloth triangle mesh (wireframe)."""
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
            edges += [i0, i1]
            edges += [i1, i3]
            edges += [i3, i2]
            edges += [i2, i0]
            # diagonal used by triangles
            edges += [i1, i2]
    return np.asarray(edges, dtype=np.uint32)


def make_initial_state(width: int, height: int, spacing: float, y0: float, mass: float = 1.0):
    """Returns pos (vec4) and vel (vec4).

    pos.xyz = position
    pos.w   = inverse mass (0 => pinned)
    """
    n = width * height
    # [GRID POINT 1 - CREATION]
    # This is where the giant numpy array containing ALL vertexes are created!
    # n = width * height (e.g. 80 * 60 = 4800 vertexes).
    # This array is then uploaded to the GPU buffers.
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
            pos[i, 3] = invm
    return pos, vel


class ClothSim:
    """Holds cloth buffers + compute pipelines/bindgroups.

    App keeps the high-level parameters (k_struct, gravity, etc.) and calls the pipelines.
    """

    def __init__(
        self,
        device: wgpu.GPUDevice,
        *,
        cloth_w: int,
        cloth_h: int,
        spacing: float,
        y0: float,
        mass: float,
        compute_wgsl_code: str,
    ):
        self.device = device
        self.w = int(cloth_w)
        self.h = int(cloth_h)
        self.spacing = float(spacing)
        self.y0_default = float(y0)
        self.mass_default = float(mass)
        self.nverts = self.w * self.h

        # ---- geometry (render mesh) ----
        vtx0, idx0 = cloth_grid(self.w, self.h, self.spacing, y0=self.y0_default)
        self.vtx0 = vtx0.astype(np.float32, copy=True)
        self.vertex_stride = 8 * 4  # pos3 + nrm3 + uv2

        # [GRID POINT 1 - PROOF]
        # The buffer size is calculated here based on (width * height).
        # Changing self.cloth_w/h in app.py directly changes the memory allocated here.
        self.vertex_buffer = device.create_buffer_with_data(
            data=self.vtx0,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.index_buffer = device.create_buffer_with_data(data=idx0, usage=wgpu.BufferUsage.INDEX)
        self.index_count = int(idx0.size)

        wire_idx = make_cloth_wire_indices(self.w, self.h)
        self.wire_index_buffer = device.create_buffer_with_data(data=wire_idx, usage=wgpu.BufferUsage.INDEX)
        self.wire_index_count = int(wire_idx.size)

        # ---- sim state (ping-pong) ----
        pos0, vel0 = make_initial_state(self.w, self.h, self.spacing, y0=self.y0_default, mass=self.mass_default)
        self.pos_a = device.create_buffer_with_data(
            data=pos0, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self.vel_a = device.create_buffer_with_data(
            data=vel0, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self.pos_b = device.create_buffer(size=self.pos_a.size, usage=wgpu.BufferUsage.STORAGE)
        self.vel_b = device.create_buffer(size=self.vel_a.size, usage=wgpu.BufferUsage.STORAGE)

        self.ping = 0  # 0 => current in A, 1 => current in B

        # ---- sim params uniform ----
        # Matches SimParams in cloth_compute.wgsl: 20 floats + 4 u32 = 96 bytes.
        self.sim_params_buffer = device.create_buffer(
            size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        # ---- compute pipelines + bind groups ----
        compute_bg_layout = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
            ]
        )
        compute_layout = device.create_pipeline_layout(bind_group_layouts=[compute_bg_layout])

        cshader = device.create_shader_module(code=compute_wgsl_code)
        self.pipe_step = device.create_compute_pipeline(layout=compute_layout, compute={"module": cshader, "entry_point": "step"})
        self.pipe_write_vtx = device.create_compute_pipeline(layout=compute_layout, compute={"module": cshader, "entry_point": "write_vertices"})

        def make_compute_bg(pos_in, vel_in, pos_out, vel_out):
            return device.create_bind_group(
                layout=compute_bg_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": self.sim_params_buffer, "offset": 0, "size": self.sim_params_buffer.size},
                    },
                    {"binding": 1, "resource": {"buffer": pos_in, "offset": 0, "size": pos_in.size}},
                    {"binding": 2, "resource": {"buffer": vel_in, "offset": 0, "size": vel_in.size}},
                    {"binding": 3, "resource": {"buffer": pos_out, "offset": 0, "size": pos_out.size}},
                    {"binding": 4, "resource": {"buffer": vel_out, "offset": 0, "size": vel_out.size}},
                    {
                        "binding": 5,
                        "resource": {"buffer": self.vertex_buffer, "offset": 0, "size": self.vertex_buffer.size},
                    },
                ],
            )

        self.compute_bg_ab = make_compute_bg(self.pos_a, self.vel_a, self.pos_b, self.vel_b)
        self.compute_bg_ba = make_compute_bg(self.pos_b, self.vel_b, self.pos_a, self.vel_a)

    def reset(self, *, y0: float | None = None, mass: float | None = None) -> None:
        """Reset pos/vel buffers and the visible vertex buffer."""
        if y0 is None:
            y0 = self.y0_default
        if mass is None:
            mass = self.mass_default

        pos0, vel0 = make_initial_state(self.w, self.h, self.spacing, y0=float(y0), mass=float(mass))
        self.device.queue.write_buffer(self.pos_a, 0, pos0.tobytes())
        self.device.queue.write_buffer(self.vel_a, 0, vel0.tobytes())
        self.device.queue.write_buffer(self.pos_b, 0, pos0.tobytes())
        self.device.queue.write_buffer(self.vel_b, 0, vel0.tobytes())
        self.ping = 0

        # visual reset
        self.device.queue.write_buffer(self.vertex_buffer, 0, self.vtx0.tobytes())
