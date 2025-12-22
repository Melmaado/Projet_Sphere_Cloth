import numpy as np

def cube():
    vertex_data = np.array(
    [
        #   x,    y,    z,   xn,   yn,   zn,    u,    v
        [ 0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0,  0.0],
        [-0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0,  1.0],
        [-0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0,  1.0],
        [ 0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0,  0.0],

        [ 0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0,  1.0],
        [ 0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  1.0,  1.0],
        [ 0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0,  0.0],

        [ 0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0,  0.0],
        [ 0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0,  1.0],
        [-0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0,  1.0],
        [-0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0,  0.0],

        [-0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  0.0,  0.0],
        [-0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  0.0,  1.0],
        [-0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  1.0,  1.0],
        [-0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  1.0,  0.0],

        [ 0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0,  0.0],
        [ 0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0,  1.0],
        [-0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0,  1.0],
        [-0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0,  0.0],

        [ 0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0,  0.0],
        [-0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0,  1.0],
        [-0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0,  1.0],
        [ 0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0,  0.0],
        ],
        dtype=np.float32,
    )

    index_data = np.array(
        [
            0,  1,  2,  0,  2,  3,
            4,  5,  6,  4,  6,  7,
            8,  9, 10,  8, 10, 11,
            12, 13, 14, 12, 14, 15,
            16, 17, 18, 16, 18, 19,
            20, 21, 22, 20, 22, 23,
        ],
        dtype=np.uint32,
    )
    return vertex_data, index_data


def cloth_grid(width: int, height: int, spacing: float = 0.05, y0: float = 1.2,
               inv_mass: float = 1.0):
    """Generate a subdivided plane (cloth) as a triangle mesh.

    Vertex layout: [x,y,z, nx,ny,nz, u,v] (float32)
    Indices: uint32 triangle list.

    Note: the simulation stores inverse-mass in a separate buffer; this function
    only generates a render mesh.
    """
    n = width * height
    vtx = np.zeros((n, 8), dtype=np.float32)

    origin = np.array(
        [-0.5 * (width - 1) * spacing, y0, -0.5 * (height - 1) * spacing],
        dtype=np.float32,
    )

    for y in range(height):
        for x in range(width):
            i = y * width + x
            p = origin + np.array([x * spacing, 0.0, y * spacing], dtype=np.float32)
            u = x / (width - 1)
            v = y / (height - 1)
            vtx[i, 0:3] = p
            vtx[i, 3:6] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            vtx[i, 6:8] = np.array([u, v], dtype=np.float32)

    idx = []
    for y in range(height - 1):
        for x in range(width - 1):
            i0 = y * width + x
            i1 = i0 + 1
            i2 = i0 + width
            i3 = i2 + 1
            # CCW
            idx += [i0, i2, i1, i1, i2, i3]

    return vtx, np.array(idx, dtype=np.uint32)


def sphere(slices: int = 32, stacks: int = 16,
           center=(0.0, 0.0, 0.0), radius: float = 1.0):
    """UV sphere.

    Vertex layout: [x,y,z, nx,ny,nz, u,v] (float32)
    """
    cx, cy, cz = center
    verts = []
    for j in range(stacks + 1):
        v = j / stacks
        phi = np.pi * v  # 0..pi
        sp, cp = np.sin(phi), np.cos(phi)
        for i in range(slices + 1):
            u = i / slices
            theta = 2 * np.pi * u  # 0..2pi
            st, ct = np.sin(theta), np.cos(theta)

            nx = ct * sp
            ny = cp
            nz = st * sp

            x = cx + radius * nx
            y = cy + radius * ny
            z = cz + radius * nz

            verts.append([x, y, z, nx, ny, nz, u, 1.0 - v])

    verts = np.array(verts, dtype=np.float32)

    idx = []
    w = slices + 1
    for j in range(stacks):
        for i in range(slices):
            i0 = j * w + i
            i1 = i0 + 1
            i2 = i0 + w
            i3 = i2 + 1
            idx += [i0, i2, i1, i1, i2, i3]

    return verts, np.array(idx, dtype=np.uint32)