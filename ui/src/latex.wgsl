@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var msdf: texture_2d<f32>;
@group(0) @binding(2) var bilinear: sampler;

struct Uniforms {
    resolution: vec2f,
    scale_factor: f32,
}

// QuadKind
const Rectangle = 0u;
const Pill = 1u;
const MsdfGlyph = 2u;
const AlphaGradientU = 3u;
const AlphaGradientV2 = 4u;
const OutputValueBox = 5u;
const SliderPausedButton = 6u;
const SliderPlayingButton = 7u;

struct Vertex {
    @location(0) position: vec2f,
    @location(1) color: vec4f,
    @location(2) kind: u32,
    @location(3) uv: vec2f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) @interpolate(flat) kind: u32,
    @location(2) uv: vec2f,
}

fn flip_y(v: vec2f) -> vec2f {
    return vec2(v.x, -v.y);
}

@vertex
fn vs_latex(v: Vertex) -> VertexOutput {
    let p_clip = vec4(flip_y(2.0 * v.position - uniforms.resolution) / uniforms.resolution, 0.0, 1.0);
    return VertexOutput(p_clip,  v.color, v.kind, v.uv);
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
        case Rectangle, default {
            return in.color;
        }
        case Pill {
            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(size.y / 2.0));
            return in.color * vec4(1.0, 1.0, 1.0, saturate(0.5 - sd));
        }
        case MsdfGlyph {
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
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case AlphaGradientU {
            return in.color * vec4(1.0, 1.0, 1.0, in.uv.x);
        }
        case AlphaGradientV2 {
            return in.color * vec4(1.0, 1.0, 1.0, in.uv.y * in.uv.y);
        }
        case OutputValueBox {
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
        case SliderPausedButton, SliderPlayingButton {
            let p = in.uv * 2.0 - 1.0;
            var sd = abs(length(p) - 0.935) - 0.065;

            if in.kind == SliderPausedButton {
                sd = min(sd, max(0.5 * p.x + sqrt(0.75) * abs(p.y), -p.x) - 0.21);
            } else {
                sd = min(sd, max(abs(abs(p.x) - 0.22) - 0.14, abs(p.y) - 0.35));
            }

            sd *= size.x / 2.0;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
    }
}
