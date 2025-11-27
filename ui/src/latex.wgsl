@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var msdf: texture_2d<f32>;
@group(0) @binding(2) var bilinear: sampler;

struct Uniforms {
    resolution: vec2f,
    scale_factor: f32,
}

const MSDF_GLYPH = 0u;
const TRANSLUCENT_MSDF_GLYPH = 1u;
const BLACK_BOX = 2u;
const TRANSLUCENT_BLACK_BOX = 3u;
const HIGHLIGHT_BOX = 4u;
const GRAY_BOX = 5u;
const TRANSPARENT_TO_WHITE_GRADIENT = 6u;
const OUTPUT_VALUE_BOX = 7u;
const SLIDER_BAR = 8u;
const SLIDER_POINT_OUTER = 9u;
const SLIDER_POINT_INNER = 10u;
const PLACEHOLDER_MSDF_GLYPH = 11u;
const PLACEHOLDER_BLACK_BOX = 12u;
const DOMAIN_BOUND_UNFOCUSSED = 13u;
const DOMAIN_BOUND_FOCUSSED = 14u;
const DOMAIN_BOUND_ERROR = 15u;

const PLACEHOLDER_OPACITY = 0.47;

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

// https://www.shadertoy.com/view/4llXD7
fn sd_rounded_box(p: vec2f, b: vec2f, r: vec4f) -> f32 {
    var r1 = select(r.zw, r.xy, p.x > 0.0);
    r1.x  = select(r1.y, r1.x, p.y > 0.0);
    let q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r.x;
}

@fragment
fn fs_latex(in: VertexOutput) -> @location(0) vec4f {
    // Derivatives must only be called from uniform control flow so we do this here
    // before the switch statement
    let fwidth_uv = fwidth(in.uv);
    let size = 1.0 / vec2(dpdx(in.uv.x), dpdy(in.uv.y));

    switch in.kind {
        case BLACK_BOX {
            return vec4(0.0, 0.0, 0.0, 1.0);
        }
        case TRANSLUCENT_BLACK_BOX {
            return vec4(0.0, 0.0, 0.0, 0.2);
        }
        case PLACEHOLDER_BLACK_BOX {
            return vec4(0.0, 0.0, 0.0, PLACEHOLDER_OPACITY);
        }
        case DOMAIN_BOUND_UNFOCUSSED {
            return vec4(0.8, 0.8, 0.8, 1.0);
        }
        case DOMAIN_BOUND_FOCUSSED {
            return vec4(0.18, 0.45, 0.86, 1.0);
        }
        case DOMAIN_BOUND_ERROR {
            return vec4(0.882, 0.345, 0.333, 1.0);
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
        case OUTPUT_VALUE_BOX {
            const RADIUS = 4.0;
            const STROKE_COLOR = vec3(0.84);
            const FILL_COLOR = vec3(0.96);
            const STROKE_WIDTH = 1.0;

            let radius = RADIUS * uniforms.scale_factor;
            let stroke_width = max(round(STROKE_WIDTH * uniforms.scale_factor), 1.0);

            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(radius));
            let color = mix(STROKE_COLOR, FILL_COLOR, saturate(0.5 - (sd + stroke_width)));
            return vec4(color, saturate(0.5 - sd));
        }
        case SLIDER_BAR, SLIDER_POINT_OUTER, SLIDER_POINT_INNER {
            var color: vec4f;
            if in.kind == SLIDER_BAR {
                color = vec4(0.898, 0.898, 0.898, 1.0);
            } else {
                color = vec4(0.184, 0.447, 0.863, select(1.0, 0.35, in.kind == SLIDER_POINT_OUTER));
            }

            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(size.y / 2.0));
            return color * vec4(1.0, 1.0, 1.0, saturate(0.5 - sd));
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
            } else if in.kind == PLACEHOLDER_MSDF_GLYPH {
                opacity *= PLACEHOLDER_OPACITY;
            }
            return vec4(0.0, 0.0, 0.0, opacity);
        }
    }
}
