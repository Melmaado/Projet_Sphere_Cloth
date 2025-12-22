from __future__ import annotations

import numpy as np
import wgpu


class SphereMesh:
    """GPU sphere mesh whose positions can be updated by (center, radius).

    Vertex layout expected: [x,y,z, nx,ny,nz, u,v] float32 (same as primitives.sphere()).
    """

    def __init__(self, device: wgpu.GPUDevice, template_vtx: np.ndarray, indices: np.ndarray):
        self.device = device
        self._template_vtx = np.asarray(template_vtx, dtype=np.float32)
        self._last_center = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._last_radius = float("nan")

        # Buffers
        self.vertex_buffer = device.create_buffer_with_data(
            data=self._template_vtx,  # will be overwritten by update(force=True)
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        self.index_buffer = device.create_buffer_with_data(data=indices, usage=wgpu.BufferUsage.INDEX)
        self.index_count = int(indices.size)

    def update(self, center: np.ndarray, radius: float, *, force: bool = False) -> None:
        c = np.asarray(center, dtype=np.float32).reshape(3)
        r = float(radius)

        if (not force) and np.allclose(c, self._last_center) and abs(r - self._last_radius) <= 1e-6:
            return

        vtx = self._template_vtx.copy()
        vtx[:, 0:3] = c[None, :] + r * self._template_vtx[:, 0:3]
        self.device.queue.write_buffer(self.vertex_buffer, 0, vtx.tobytes())

        self._last_center = c
        self._last_radius = r
