struct RenderParams {
    light: vec4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec2<f32>,     // (width, height) in pixels
    point_size: f32,         // point radius in pixels
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
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip = params.proj * params.view * vec4<f32>(in.position, 1.0);
    out.position = in.position;
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);

    // lumière point (params.light.xyz = position de la lumière en world)
    let Lvec = params.light.xyz - in.position;
    let L = normalize(Lvec);

    // Lambert propre = max(dot, 0) + un ambient séparé
    let ndotl = max(dot(N, L), 0.0);

    let ambient = 0.18;
    let shading = ambient + (1.0 - ambient) * ndotl;

    let albedo = textureSample(texture, samplr, in.uv);
    return vec4<f32>(albedo.xyz * shading, 1.0);
}


// --- Debug cloth points (billboard quads) ---
struct PointsInstanceInput {
    @location(0) position: vec3<f32>,
};

struct PointsVertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,   // -1..1 quad coordinates
};

@vertex
fn vs_points(inst: PointsInstanceInput, @builtin(vertex_index) vid: u32) -> PointsVertexOutput {
    // 2 triangles (6 vertices) forming a camera-facing quad around each particle
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    var out: PointsVertexOutput;

    var clip = params.proj * params.view * vec4<f32>(inst.position, 1.0);

    // Convert a pixel radius to NDC, then to clip-space (multiply by w)
    let vp = max(params.viewport, vec2<f32>(1.0, 1.0));
    let offset_ndc = corners[vid] * (params.point_size * 2.0) / vp;
    // WGSL does not allow assigning to swizzles (e.g. clip.xy = ...)
    clip.x = clip.x + offset_ndc.x * clip.w;
    clip.y = clip.y + offset_ndc.y * clip.w;

    out.clip = clip;
    out.local = corners[vid];
    return out;
}

@fragment
fn fs_points(in: PointsVertexOutput) -> @location(0) vec4<f32> {
    // Round point: discard outside unit circle
    if (length(in.local) > 1.0) {
        discard;
    }
    return vec4<f32>(0.08, 0.08, 0.08, 1.0);
}
