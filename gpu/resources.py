from __future__ import annotations

from pathlib import Path
from typing import Union

import PIL.Image as Image
import wgpu


PathLike = Union[str, Path]


def load_texture_2d(device: wgpu.GPUDevice, path: PathLike, fallback_path: PathLike) -> wgpu.GPUTexture:
    """Load an RGBA texture, fallback if missing."""
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        img = Image.open(fallback_path).convert("RGBA")

    texture_size = (img.size[0], img.size[1], 1)
    tex = device.create_texture(
        size=texture_size,
        format=wgpu.TextureFormat.rgba8unorm_srgb,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        img.tobytes(),
        {"bytes_per_row": img.size[0] * 4, "rows_per_image": img.size[1]},
        texture_size,
    )
    return tex


def read_wgsl(path: PathLike) -> str:
    return Path(path).read_text(encoding="utf-8")
