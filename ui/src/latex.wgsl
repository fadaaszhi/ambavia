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
const SLIDER_STEP_TICK = 9u;
const SLIDER_ZERO_TICK = 10u;
const SLIDER_POINT_OUTER = 11u;
const SLIDER_POINT_INNER = 12u;
const PLACEHOLDER_MSDF_GLYPH = 13u;
const PLACEHOLDER_BLACK_BOX = 14u;
const DOMAIN_BOUND_UNFOCUSSED = 15u;
const DOMAIN_BOUND_FOCUSSED = 16u;
const DOMAIN_BOUND_ERROR = 17u;
const GRAYED_MSDF_GLYPH = 18u;
const GRAYED_BLACK_BOX = 19u;

const PLACEHOLDER_OPACITY = 0.47;
const GRAYED_OPACITY = 0.6;

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

// Calculate the Jacobian matrix for bilinear texture sampling
fn jacobian(texture: texture_2d<f32>, uv: vec2f) -> mat2x3f {
    let dimensions = vec2i(textureDimensions(texture, 0));
    let p = uv * vec2f(dimensions);
    let q = floor(p - 0.5);
    let w = p - q - 0.5;
    let r = vec2i(q);
    let a = clamp(r, vec2(0), dimensions - 1);
    let b = clamp(r + 1, vec2(0), dimensions - 1);
    let f00 = textureLoad(texture, vec2(a.x, a.y), 0).rgb;
    let f10 = textureLoad(texture, vec2(b.x, a.y), 0).rgb;
    let f01 = textureLoad(texture, vec2(a.x, b.y), 0).rgb;
    let f11 = textureLoad(texture, vec2(b.x, b.y), 0).rgb;
    let dfdu = mix(f10 - f00, f11 - f01, w.y) * f32(dimensions.x);
    let dfdv = mix(f01 - f00, f11 - f10, w.x) * f32(dimensions.y);
    return mat2x3(dfdu, dfdv);
}

fn sqr(x: vec3f) -> vec3f {
    return x * x;
}

@diagnostic(off, derivative_uniformity)
@fragment
fn fs_latex(in: VertexOutput) -> @location(0) vec4f {
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
        case GRAYED_BLACK_BOX {
            return vec4(0.0, 0.0, 0.0, GRAYED_OPACITY);
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
        case SLIDER_BAR, SLIDER_STEP_TICK, SLIDER_ZERO_TICK, SLIDER_POINT_OUTER, SLIDER_POINT_INNER {
            var color: vec4f;
            switch in.kind {
                case SLIDER_BAR         { color = vec4(0.898, 0.898, 0.898, 1.000); }
                case SLIDER_STEP_TICK   { color = vec4(1.000, 1.000, 1.000, 1.000); }
                case SLIDER_ZERO_TICK   { color = vec4(0.000, 0.000, 0.000, 0.353); }
                case SLIDER_POINT_OUTER { color = vec4(0.184, 0.447, 0.863, 0.350); }
                case SLIDER_POINT_INNER { color = vec4(0.184, 0.447, 0.863, 1.000); }
                default {}
            }
            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(size.y / 2.0));
            return color * vec4(1.0, 1.0, 1.0, saturate(0.5 - sd));
        }
        default {
            // Based off the example snippet from https://github.com/Chlumsky/msdfgen
            // but adjusted to handle non-uniform scaling
            let px_range = 4.0; // set during MSDF atlas creation
            let unit_range = px_range / vec2f(textureDimensions(msdf, 0));
            let msd = textureSampleLevel(msdf, bilinear, in.uv, 0.0).rgb;
            let dmsduv = jacobian(msdf, in.uv);
            let duvdx = dpdx(in.uv);
            let duvdy = dpdy(in.uv);
            let screen_px_range = max(vec3(1.0), select(
                sqrt((sqr(dmsduv[0] * unit_range.x) + sqr(dmsduv[1] * unit_range.y)) /
                     (sqr(dmsduv * duvdx) + sqr(dmsduv * duvdy))),
                vec3(sqrt(2.0) / length(vec4(duvdx, duvdy))),
                (dmsduv[0] == vec3(0.0)) & (dmsduv[1] == vec3(0.0))
            ));
            var msd_screen = screen_px_range * (msd - 0.5);
            let screen_px_distance = median(msd_screen.r, msd_screen.g, msd_screen.b);
            var opacity = saturate(screen_px_distance + 0.5);
            if in.kind == TRANSLUCENT_MSDF_GLYPH {
                opacity *= 0.2;
            } else if in.kind == PLACEHOLDER_MSDF_GLYPH {
                opacity *= PLACEHOLDER_OPACITY;
            } else if in.kind == GRAYED_MSDF_GLYPH {
                opacity *= GRAYED_OPACITY;
            }
            return vec4(0.0, 0.0, 0.0, opacity);
        }
    }
}
