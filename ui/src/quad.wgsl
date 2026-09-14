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
const AlphaGradientU2 = 4u;
const AlphaGradientV2 = 5u;
const OutputValueBox = 6u;
const SliderPausedButton = 7u;
const SliderPlayingButton = 8u;
const GraphButtonShadow = 9u;
const GraphButton = 10u;
const GraphButtonUpper = 11u;
const GraphButtonLower = 12u;
const HomeIcon = 13u;
const LoopForwardReverseIcon = 14u;
const LoopForwardIcon = 15u;
const PlayOnceIcon = 16u;
const PlayIndefinitelyIcon = 17u;
const PopupShadow = 18u;
const PopupBackground = 19u;
const PopupArrow = 20u;
const PopupRadioLeft = 21u;
const PopupRadioMiddle = 22u;
const PopupRadioRight = 23u;
const PopupRadioSelectedLeft = 24u;
const PopupRadioSelectedMiddle = 25u;
const PopupRadioSelectedRight = 26u;
const PopupButton = 27u;
const IncreaseSliderSpeedIcon = 28u;


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
fn vs_quad(v: Vertex) -> VertexOutput {
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
    let q = abs(p) - b + r1.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r1.x;
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

fn hypot(x: f32, y: f32) -> f32 {
    return length(vec2(x, y));
}

fn modf32(x: f32, y: f32) -> f32 {
    return x - floor(x / y) * y;
}

@diagnostic(off, derivative_uniformity)
fn rounded_box_shadow(uv: vec2f, color: vec4f, shadow_radius_: f32, box_radius_: f32) -> vec4f {
    let size = 1.0 / abs(vec2(dpdx(uv.x), dpdy(uv.y)));
    let shadow_radius = shadow_radius_ * uniforms.scale_factor;
    let box_radius = box_radius_ * uniforms.scale_factor;
    let sd = sd_rounded_box(size * (uv - 0.5), size / 2.0 - 2.0 * shadow_radius, vec4(box_radius));
    let shadow = saturate(1.0 - sd / (2.0 * shadow_radius));
    return color * vec4(1.0, 1.0, 1.0, smoothstep(0.0, 1.0, shadow));
}

fn stroked(sd: f32, fill_color: vec4f, stroke_width_: f32, stroke_color: vec4f) -> vec4f {
    let stroke_width = max(round(stroke_width_ * uniforms.scale_factor), 1.0);
    let color = mix(stroke_color, fill_color, saturate(0.5 - (sd + stroke_width)));
    let opacity = saturate(0.5 - sd);
    return color * vec4(1.0, 1.0, 1.0, opacity);
}

fn stroked2(sd: f32, fill_color: vec4f, stroke_width: f32, stroke_brightness: f32) -> vec4f {
    return stroked(sd, fill_color, stroke_width, fill_color * vec4(vec3(stroke_brightness), 1.0));
}

@diagnostic(off, derivative_uniformity)
@fragment
fn fs_quad(in: VertexOutput) -> @location(0) vec4f {
    let size = 1.0 / abs(vec2(dpdx(in.uv.x), dpdy(in.uv.y)));

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
        case AlphaGradientU2 {
            return in.color * vec4(1.0, 1.0, 1.0, in.uv.x * in.uv.x);
        }
        case AlphaGradientV2 {
            return in.color * vec4(1.0, 1.0, 1.0, in.uv.y * in.uv.y);
        }
        case OutputValueBox {
            const RADIUS = 4.0;
            let radius = RADIUS * uniforms.scale_factor;
            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(radius));
            return stroked2(sd, in.color, 1.0, 0.873);
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
        case GraphButtonShadow {
            return rounded_box_shadow(in.uv, in.color, 5.0, 5.0);
        }
        case GraphButton, GraphButtonUpper, GraphButtonLower {
            const RADIUS = 5.0;

            let radius = RADIUS * uniforms.scale_factor;

            var roundness: vec4f;
            switch in.kind {
                case GraphButton, default {
                    roundness = vec4(radius);
                }
                case GraphButtonUpper {
                    roundness = vec4(0.0, radius, 0.0, radius);
                }
                case GraphButtonLower {
                    roundness = vec4(radius, 0.0, radius, 0.0);
                }
            }

            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, roundness);
            return stroked2(sd, in.color, 1.0, 0.9);
        }
        case HomeIcon {
            let p = in.uv * 2.0 - 1.0;
            let y = p.y;
            let x = abs(p.x);

            var sd = min(min(
                max(abs(x - y - 0.86) - 0.13, x + y - 1.0) / sqrt(2.0),
                max(max((x - y) / sqrt(2.0) - 0.4, x - 0.7), min(y - 0.4, 0.16 - x))),
                max(abs(p.x - 0.55) - 0.15, 0.86 + y - p.x)
            );

            sd *= size.x / 2.0;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case LoopForwardReverseIcon, LoopForwardIcon, PlayOnceIcon, PlayIndefinitelyIcon {
            let p = (in.uv * 2.0 - 1.0) * vec2(1.0, -0.912);
            var x = p.x;
            var y = p.y;

            switch in.kind {
                case LoopForwardReverseIcon {
                    y -= 0.0694;
                    if y < 0.07 {
                        x = -x;
                        y = -y;
                    }
                }
                case LoopForwardIcon {
                    y += select(-0.0694, 0.8566, y < 0.07);
                }
                default {
                    y += 0.275;
                }
            }

            var sd = min(
                max(0.77 * abs(y - 0.463) + 0.65 * x - 0.65, 0.554 - x),
                max(max(
                    select(abs(y - 0.294) - 0.294, hypot(x + 0.65, y - 0.238) - 0.35, x < -0.65),
                    select(0.338 - y, 0.48 - hypot(x + 0.56, y + 0.143), x < -0.56)),
                    x - 0.77
                )
            );
            x = p.x;
            y = p.y;

            if in.kind == PlayOnceIcon {
                sd = min(sd, max(max(
                    hypot(x + 0.478, y + 0.495) - 0.42,
                    min(0.06 - abs(x + 0.463), 0.29 - abs(y + 0.49))),
                    min(0.085 - abs(y - x - 0.238), 0.15 - abs(y + x + 0.873)) / sqrt(2.0)
                ));
            }
            
            if in.kind == PlayIndefinitelyIcon && abs(x + 0.003) < 0.557 {
                sd = hypot(0.154875 - abs(modf32(x + 0.56, 0.30975) - 0.154875), y - 0.188) - 0.125;
            }

            sd *= size.x / 2.0;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case PopupShadow {
            return rounded_box_shadow(in.uv, in.color, 10.0, 6.0);
        }
        case PopupBackground {
            const RADIUS = 6.0;
            const STROKE_BRIGHTNESS = 0.733;
            const STROKE_WIDTH = 1.0;

            let radius = RADIUS * uniforms.scale_factor;
            let stroke_width = max(round(STROKE_WIDTH * uniforms.scale_factor), 1.0);
            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(radius));
            let stroke = mix(STROKE_BRIGHTNESS, 1.0, saturate(0.5 - (sd + stroke_width)));
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(vec3(stroke), opacity);
        }
        case PopupArrow {
            const STROKE_BRIGHTNESS = 0.733;
            const STROKE_WIDTH = 1.0;

            let stroke_width = max(round(STROKE_WIDTH * uniforms.scale_factor), 1.0);
            let p = size * vec2(in.uv.x, abs(in.uv.y - 0.5));
            let sd = dot(p, normalize(vec2(-size.y / 2.0, size.x - stroke_width)));
            let stroke = mix(STROKE_BRIGHTNESS, 1.0, saturate(0.5 - (sd + stroke_width)));
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(vec3(stroke), opacity);
        }
        case PopupRadioLeft, PopupRadioMiddle, PopupRadioRight,
             PopupRadioSelectedLeft, PopupRadioSelectedMiddle, PopupRadioSelectedRight {
            const RADIUS = 3.0;
            const STROKE_BRIGHTNESS = 0.7;
            const STROKE_WIDTH = 1.0;

            let radius = RADIUS * uniforms.scale_factor;
            let stroke_width = max(round(STROKE_WIDTH * uniforms.scale_factor), 1.0);

            var roundness: vec4f;
            switch in.kind {
                case PopupRadioMiddle, PopupRadioSelectedMiddle, default {
                    roundness = vec4(0.0);
                }
                case PopupRadioLeft, PopupRadioSelectedLeft {
                    roundness = vec4(0.0, 0.0, radius, radius);
                }
                case PopupRadioRight, PopupRadioSelectedRight {
                    roundness = vec4(radius, radius, 0.0, 0.0);
                }
            }

            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, roundness);

            if in.kind == PopupRadioMiddle || in.kind == PopupRadioLeft || in.kind == PopupRadioRight {
                return stroked2(sd, in.color, 1.0, 0.7);
            } else {
                return stroked(sd, in.color * vec4(vec3(1.0), 0.25), 1.0, in.color);
            }
        }
        case IncreaseSliderSpeedIcon {
            let p = (in.uv * 2.0 - 1.0) * vec2(0.914, 1.0);
            let y = abs(p.y);
            let x = vec2(p.x, p.x - 0.76);

            let sd2 = max(0.75 * y - 0.66 * x - 1.217, abs(0.66 * y + 0.75 * x + 0.01) - 0.12);
            var sd = min(sd2.x, sd2.y);

            sd *= size.y / 2.0;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case PopupButton {
            const RADIUS = 4.0;
            let radius = RADIUS * uniforms.scale_factor;
            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(radius));
            return stroked2(sd, in.color, 1.0, 0.8);
        }
    }
}
