struct SimParams {
    // 0..16
    dt: f32,
    time: f32,
    spacing: f32,
    c_d: f32,

    // 16..32
    k_struct: f32,
    k_shear: f32,
    k_bend: f32,
    vel_damping: f32,

    // 32..48
    friction: f32,
    collision_eps: f32,
    _pad0: vec2<f32>,

    // 48..64
    gravity: vec4<f32>,

    // 64..80
    // [GRID POINT 2] RECEPTION FROM CPU
    // This matches the Python struct.pack order.
    // xyz = center, w = radius
    sphere: vec4<f32>, // xyz=center, w=radius

    // 80..96
    dims: vec2<u32>,
    _pad1: vec2<u32>,
};

@group(0) @binding(0) var<uniform> sim: SimParams;

// Ping-pong buffers to avoid race conditions.
// pos*.w = inverse mass (0 => pinned)
@group(0) @binding(1) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pos_out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel_out: array<vec4<f32>>;

// Interleaved vertex buffer as float array: [px,py,pz, nx,ny,nz, u,v] per vertex
@group(0) @binding(5) var<storage, read_write> vtx: array<f32>;

fn xy_to_idx(x: u32, y: u32) -> u32 {
    return y * sim.dims.x + x;
}

// [GRID POINT 6 - FORMULA]
// This function implements Hooke's Law: F = k * (dist - rest_length) * dir
fn spring_force(pi: vec3<f32>, pj: vec3<f32>, rest: f32, k: f32) -> vec3<f32> {
    let d = pj - pi;
    let dist = length(d);
    if (dist < 1e-6) { return vec3<f32>(0.0); }
    let dir = d / dist;
    let dl = dist - rest;
    // Force on i (towards j if stretched)
    return k * dl * dir;
}

@compute @workgroup_size(256)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = sim.dims.x * sim.dims.y;
    if (i >= n) { return; }

    let p4 = pos_in[i];
    let invm = p4.w;
    if (invm == 0.0) {
        // pinned
        pos_out[i] = p4;
        vel_out[i] = vec4<f32>(0.0);
        return;
    }

    let m = 1.0 / invm;
    let pi = p4.xyz;
    let vi = vel_in[i].xyz;

    let w = sim.dims.x;
    let h = sim.dims.y;
    let x = i % w;
    let y = i / w;

    let s = sim.spacing;
    let sd = s * 1.41421356;
    let s2 = 2.0 * s;

    var R = vec3<f32>(0.0);

    // --- Springs (Hooke's law) ---
    // [GRID POINT 6 & 7] SPRINGS (HOOKE'S LAW)
    // Three types of springs to simulate realistic cloth behavior:

    // structural
    if (x > 0u)       { R += spring_force(pi, pos_in[xy_to_idx(x - 1u, y)].xyz, s,  sim.k_struct); }
    if (x + 1u < w)   { R += spring_force(pi, pos_in[xy_to_idx(x + 1u, y)].xyz, s,  sim.k_struct); }
    if (y > 0u)       { R += spring_force(pi, pos_in[xy_to_idx(x, y - 1u)].xyz, s,  sim.k_struct); }
    if (y + 1u < h)   { R += spring_force(pi, pos_in[xy_to_idx(x, y + 1u)].xyz, s,  sim.k_struct); }

    // shear
    if (x > 0u && y > 0u)             { R += spring_force(pi, pos_in[xy_to_idx(x - 1u, y - 1u)].xyz, sd, sim.k_shear); }
    if (x + 1u < w && y > 0u)         { R += spring_force(pi, pos_in[xy_to_idx(x + 1u, y - 1u)].xyz, sd, sim.k_shear); }
    if (x > 0u && y + 1u < h)         { R += spring_force(pi, pos_in[xy_to_idx(x - 1u, y + 1u)].xyz, sd, sim.k_shear); }
    if (x + 1u < w && y + 1u < h)     { R += spring_force(pi, pos_in[xy_to_idx(x + 1u, y + 1u)].xyz, sd, sim.k_shear); }

    // bend
    if (x > 1u)       { R += spring_force(pi, pos_in[xy_to_idx(x - 2u, y)].xyz, s2, sim.k_bend); }
    if (x + 2u < w)   { R += spring_force(pi, pos_in[xy_to_idx(x + 2u, y)].xyz, s2, sim.k_bend); }
    if (y > 1u)       { R += spring_force(pi, pos_in[xy_to_idx(x, y - 2u)].xyz, s2, sim.k_bend); }
    if (y + 2u < h)   { R += spring_force(pi, pos_in[xy_to_idx(x, y + 2u)].xyz, s2, sim.k_bend); }

    // --- Damping force ---
    R += -sim.c_d * vi;

    // --- Gravity force ---
    // [GRID POINT 4] GRAVITY
    // Apply gravitational force (F = m * g)
    R += m * sim.gravity.xyz;

    // --- Sphere friction (Coulomb-like, based on resultant of other forces) ---
    let c = sim.sphere.xyz;
    let r = sim.sphere.w;
    let d0 = pi - c;
    let dist_contact0 = length(d0);
    if (dist_contact0 <= r + sim.collision_eps) {
        // Avoid `select(a, b, cond)` with `b = d0/dist` because some compilers
        // still evaluate both branches, which can divide by zero when dist ~ 0.
        var nrm = vec3<f32>(0.0, 1.0, 0.0);
        if (dist_contact0 > 1e-6) {
            nrm = d0 / dist_contact0;
        }
        let Ron = dot(R, nrm);
        let Rot = R - Ron * nrm;
        let Rot_len = length(Rot);
        if (Rot_len > 1e-6) {
            let ft_mag = min(Rot_len, sim.friction * abs(Ron));
            R += -ft_mag * (Rot / Rot_len);
        }
    }

    // --- Integrate ---
    let a = R * invm;
    var v_new = vi + sim.dt * a;
    v_new *= sim.vel_damping;
    var x_new = pi + sim.dt * v_new;

    // --- Sphere collision (robust + NO "teleport through") ---
    // IMPORTANT: If a vertex gets pulled *deep* into the sphere in one step,
    // projecting using the direction of p1 (=x_new) can put it on the *far side*
    // of the sphere, which visually looks like it "passed through".
    // We therefore resolve penetration using a normal consistent with the *previous*
    // position p0 (=pi) whenever possible.

    let r_eff = r + sim.collision_eps;
    let p0 = pi;
    var p1 = x_new;

    let to0 = p0 - c;
    let dist0 = length(to0);
    let to1 = p1 - c;
    let dist1 = length(to1);

    var collided = false;
    var nrm = vec3<f32>(0.0, 1.0, 0.0);

    // [GRID POINT 5] CONTINUOUS COLLISION DETECTION (CCD)
    // Check if the segment p0->p1 intersects the sphere to prevent "tunneling"
    // Continuous test for the segment [p0..p1] crossing the sphere.
    // try this FIRST (even if p1 is inside) to avoid far-side projection.
    let dseg = p1 - p0;
    let a_seg = dot(dseg, dseg);
    if (a_seg > 1e-12) {
        let f = p0 - c;
        let b_seg = 2.0 * dot(f, dseg);
        let c_seg = dot(f, f) - r_eff * r_eff;
        let disc = b_seg * b_seg - 4.0 * a_seg * c_seg;
        if (disc >= 0.0) {
            let sdisc = sqrt(disc);
            let t0 = (-b_seg - sdisc) / (2.0 * a_seg);
            let t1 = (-b_seg + sdisc) / (2.0 * a_seg);

            // Choose a valid impact time in [0,1].
            // - If we start outside: take entry t0 if valid, else t1.
            // - If we start inside: take exit t1 if valid, else t0.
            let start_outside = dist0 >= r_eff;
            var t = -1.0;
            if (start_outside) {
                if (t0 >= 0.0 && t0 <= 1.0) { t = t0; }
                else if (t1 >= 0.0 && t1 <= 1.0) { t = t1; }
            } else {
                if (t1 >= 0.0 && t1 <= 1.0) { t = t1; }
                else if (t0 >= 0.0 && t0 <= 1.0) { t = t0; }
            }

            if (t >= 0.0) {
                let hit = p0 + t * dseg;
                let tohit = hit - c;
                let lhit = length(tohit);
                nrm = select(vec3<f32>(0.0, 1.0, 0.0), tohit / lhit, lhit > 1e-6);
                p1 = c + nrm * r_eff;
                collided = true;
            }
        }
    }

    // If still penetrating (common with very stiff springs), push out on the SAME side
    // as the previous position (p0), not the far side.
    if (!collided && dist1 < r_eff) {
        if (dist0 > 1e-6) {
            nrm = to0 / dist0;
        } else if (dist1 > 1e-6) {
            nrm = to1 / dist1;
        } else {
            nrm = vec3<f32>(0.0, 1.0, 0.0);
        }
        p1 = c + nrm * r_eff;
        collided = true;
    }

    x_new = p1;

    if (collided) {
        v_new = (x_new - pi) / sim.dt;
        // Cancel velocity going into the sphere
        let vn = dot(v_new, nrm);
        if (vn < 0.0) {
            v_new = v_new - vn * nrm;
        }
        // [GRID POINT 8] FRICTION
        // Tangential friction reduces velocity along the sphere surface
        // Tangential friction while in contact (0..1)
        let fr = clamp(sim.friction, 0.0, 1.0);
        let vt = v_new - dot(v_new, nrm) * nrm;
        v_new = v_new - fr * vt;
    }

    pos_out[i] = vec4<f32>(x_new, invm);
    vel_out[i] = vec4<f32>(v_new, 0.0);
}

@compute @workgroup_size(256)
fn write_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let w = sim.dims.x;
    let h = sim.dims.y;
    let n = w * h;
    if (i >= n) { return; }

    let x = i % w;
    let y = i / w;

    let p = pos_in[i].xyz;

    // Cheap normal from neighbors
    var nrm = vec3<f32>(0.0, 1.0, 0.0);
    if (x > 0u && x + 1u < w && y > 0u && y + 1u < h) {
        let pL = pos_in[xy_to_idx(x - 1u, y)].xyz;
        let pR = pos_in[xy_to_idx(x + 1u, y)].xyz;
        let pD = pos_in[xy_to_idx(x, y - 1u)].xyz;
        let pU = pos_in[xy_to_idx(x, y + 1u)].xyz;

        var acc = cross(pR - p, pU - p)
                + cross(pU - p, pL - p)
                + cross(pL - p, pD - p)
                + cross(pD - p, pR - p);
        let len = length(acc);
        if (len > 1e-6) { nrm = acc / len; }
    }

    let u = f32(x) / f32(w - 1u);
    let v = f32(y) / f32(h - 1u);

    let base = i * 8u;
    vtx[base + 0u] = p.x;
    vtx[base + 1u] = p.y;
    vtx[base + 2u] = p.z;
    vtx[base + 3u] = nrm.x;
    vtx[base + 4u] = nrm.y;
    vtx[base + 5u] = nrm.z;
    vtx[base + 6u] = u;
    vtx[base + 7u] = v;
}
