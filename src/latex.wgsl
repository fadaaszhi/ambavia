@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var msdf: texture_2d<f32>;
@group(0) @binding(2) var bilinear: sampler;

struct Uniforms {
    resolution: vec2f,
}

struct Vertex {
    @location(0) position: vec2f,
    @location(1) uv: vec2f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

fn flip_y(v: vec2f) -> vec2f {
    return vec2(v.x, -v.y);
}

@vertex
fn vs_latex(v: Vertex) -> VertexOutput {
    let p_clip = vec4(flip_y(2.0 * v.position - uniforms.resolution) / uniforms.resolution, 0.0, 1.0);
    return VertexOutput(p_clip, v.uv);
}

fn median(x: f32, y: f32, z: f32) -> f32 {
    return max(min(x, y), min(max(x, y), z));
}

@fragment
fn fs_latex(in: VertexOutput) -> @location(0) vec4f {
    // https://github.com/Chlumsky/msdfgen
    let unit_range = 4.0 / vec2f(textureDimensions(msdf, 0));
    let screen_tex_size = 1.0 / fwidth(in.uv);
    let screen_px_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);
    let msd = textureSample(msdf, bilinear, in.uv).rgb;

    if all(in.uv == vec2(-1.0)) {
        // Cursor
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    if all(in.uv == vec2(-2.0)) {
        // Selection highlight
        return vec4(0.706, 0.835, 0.996, 1.0);
    }

    if all(in.uv == vec2(-3.0)) {
        // Expression list separator
        return vec4(0.847, 0.847, 0.847, 1.0);
    }

    let sd = median(msd.r, msd.g, msd.b);
    let screen_px_distance = screen_px_range * (sd - 0.5);
    let opacity = saturate(screen_px_distance + 0.5);
    return vec4(0.0, 0.0, 0.0, opacity);
}
