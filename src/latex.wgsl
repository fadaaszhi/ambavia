@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var msdf: texture_2d<f32>;
@group(0) @binding(2) var bilinear: sampler;

struct Uniforms {
    resolution: vec2f,
}

const MSDF_GLYPH = 0u;
const TRANSLUCENT_MSDF_GLYPH = 1u;
const BLACK_BOX = 2u;
const TRANSLUCENT_BLACK_BOX = 3u;
const HIGHLIGHT_BOX = 4u;
const GRAY_BOX = 5u;
const TRANSPARENT_TO_WHITE_GRADIENT = 6u;

struct Vertex {
    @location(0) position: vec2f,
    @location(1) uv: vec2f,
    @location(2) kind: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) @interpolate(flat) kind: u32,
}

fn flip_y(v: vec2f) -> vec2f {
    return vec2(v.x, -v.y);
}

@vertex
fn vs_latex(v: Vertex) -> VertexOutput {
    let p_clip = vec4(flip_y(2.0 * v.position - uniforms.resolution) / uniforms.resolution, 0.0, 1.0);
    return VertexOutput(p_clip, v.uv, v.kind);
}

fn median(x: f32, y: f32, z: f32) -> f32 {
    return max(min(x, y), min(max(x, y), z));
}

@fragment
fn fs_latex(in: VertexOutput) -> @location(0) vec4f {
    // 'fwidth' must only be called from uniform control flow so we do this here
    // before the switch statement
    let fwidth_uv = fwidth(in.uv);

    switch in.kind {
        case BLACK_BOX {
            return vec4(0.0, 0.0, 0.0, 1.0);
        }
        case TRANSLUCENT_BLACK_BOX {
            return vec4(0.0, 0.0, 0.0, 0.2);
        }
        case HIGHLIGHT_BOX {
            return vec4(0.706, 0.835, 0.996, 1.0);
        }
        case GRAY_BOX {
            return vec4(0.847, 0.847, 0.847, 1.0);
        }
        case TRANSPARENT_TO_WHITE_GRADIENT {
            return vec4(1.0, 1.0, 1.0, in.uv.x);
        }
        default {
            // https://github.com/Chlumsky/msdfgen
            let unit_range = 4.0 / vec2f(textureDimensions(msdf, 0));
            let screen_tex_size = 1.0 / fwidth_uv;
            let screen_px_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);
            let msd = textureSampleLevel(msdf, bilinear, in.uv, 0.0).rgb;
            let sd = median(msd.r, msd.g, msd.b);
            let screen_px_distance = screen_px_range * (sd - 0.5);
            var opacity = saturate(screen_px_distance + 0.5);
            if in.kind == TRANSLUCENT_MSDF_GLYPH {
                opacity *= 0.2;
            }
            return vec4(0.0, 0.0, 0.0, opacity);
        }
    }
}
