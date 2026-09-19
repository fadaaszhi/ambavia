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
const ExpressionHiddenIcon = 29u;
const ExpressionShownIcon = 30u;
const SineSolidIcon = 31u;
const SineDashedIcon = 32u;
const SineDottedIcon = 33u;
const SineFilledIcon = 34u;
const PolygonSolidIcon = 35u;
const PolygonDashedIcon = 36u;
const PolygonDottedIcon = 37u;
const PolygonFilledIcon = 38u;
const GutterPointPointIcon = 39u;
const GutterPointOpenIcon = 40u;
const GutterPointCrossIcon = 41u;
const GutterPointSquareIcon = 42u;
const GutterPointPlusIcon = 43u;
const GutterPointTriangleIcon = 44u;
const GutterPointDiamondIcon = 45u;
const GutterPointStarIcon = 46u;
const PointsIcon = 47u;
const LinesIcon = 48u;
const InequalityDashedIcon = 49u;
const InequalityFilledIcon = 50u;
const PopupToggleShadow = 51u;
const OpacityIcon = 52u;
const ThicknessIcon = 53u;
const LineStyleSolidIcon = 54u;
const LineStyleDashedIcon = 55u;
const LineStyleDottedIcon = 56u;
const PointStylePointIcon = 57u;
const PointStyleOpenIcon = 58u;
const PointStyleCrossIcon = 59u;
const PointStyleSquareIcon = 60u;
const PointStylePlusIcon = 61u;
const PointStyleTriangleIcon = 62u;
const PointStyleDiamondIcon = 63u;
const PointStyleStarIcon = 64u;
const PopupRadioTopLeft = 65u;
const PopupRadioTopRight = 66u;
const PopupRadioBottomLeft = 67u;
const PopupRadioBottomRight = 68u;
const PopupRadioSelectedTopLeft = 69u;
const PopupRadioSelectedTopRight = 70u;
const PopupRadioSelectedBottomLeft = 71u;
const PopupRadioSelectedBottomRight = 72u;
const GutterDragXIcon = 73u;
const GutterDragYIcon = 74u;
const GutterDragXYIcon = 75u;
const PopupDragXIcon = 76u;
const PopupDragYIcon = 77u;
const PopupDragXYIcon = 78u;
const PopupColorSwatch = 79u;
const PopupColorSwatchHighlight = 80u;
const TickIcon = 81u;

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

const PI = 3.1415927;

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

// https://youtu.be/62-pRVZuS5c
fn sd_box(p: vec2f, b: vec2f) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

// https://www.shadertoy.com/view/XsXSz4
fn sd_triangle(p: vec2f, p0: vec2f, p1: vec2f, p2: vec2f) -> f32 {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;
    let v0 = p - p0;
    let v1 = p - p1;
    let v2 = p - p2;
    let pq0 = v0 - e0 * saturate(dot(v0, e0) / dot(e0, e0));
    let pq1 = v1 - e1 * saturate(dot(v1, e1) / dot(e1, e1));
    let pq2 = v2 - e2 * saturate(dot(v2, e2) / dot(e2, e2));
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d = min(min(
        vec2(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
        vec2(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
        vec2(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)),
    );
    return -sqrt(d.x) * sign(d.y);
}

// https://www.shadertoy.com/view/3tdSDj
fn sd_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let ap = p - a;
    let ab = b - a;
    return distance(ap, ab * saturate(dot(ap, ab) / dot(ab, ab)));
}

// https://www.desmos.com/calculator/x7vbtu1o40
fn sd_segment_dashed(p: vec2f, a: vec2f, b: vec2f, n: f32, k: f32) -> f32 {
    let ap = p - a;
    let ab = b - a;
    var t = saturate(dot(ap, ab) / dot(ab, ab));
    let i = round(n * t);
    t = clamp(t, (i - k / 2.0) / n, (i + k / 2.0) / n);
    return distance(ap, ab * t);
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

// Returns the x-coordinate of the closest point on y=amplitude*sin(frequency*x) to p
fn sin_closest_x(p: vec2f, amplitude: f32, frequency: f32, n_iterations: i32) -> f32 {
    let f = amplitude * frequency;
    let q = p / amplitude;
    var x = round(q.x * (f / PI)) * (PI / f);
    // Newton's method
    for (var i = 0; i < n_iterations; i++) {
        let c = cos(f * x);
        let s = sin(f * x);
        let b = q.y - s;
        x += (b * c * f + q.x - x) / (f * f * (b * s + c * c) + 1.0);
    }
    return amplitude * x;
}

// Mixes straight-alpha colors
fn mix_straight(x: vec4f, y: vec4f, t: f32) -> vec4f {
    let z = mix(vec4(x.rgb * x.a, x.a), vec4(y.rgb * y.a, y.a), t);
    return select(vec4(0.0), vec4(z.rgb / z.a, z.a), z.a > 0.0);
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
fn apply_shadow_and_circle_mask(sd: f32, uv: vec2f, color: vec4f) -> vec4f {
    let size = 1.0 / abs(vec2(dpdx(uv.x), dpdy(uv.y)));
    let opacity = saturate(0.5 - sd);
    let shadow_radius = 5.0 * uniforms.scale_factor;
    let shadow = saturate(1.0 - sd / shadow_radius);
    let shadow_color = vec4(vec3(0.0), shadow * shadow * 0.1);
    var result = mix_straight(shadow_color, color, opacity);
    result.a *= saturate(0.5 - length((uv - 0.5) * size) + min(size.x, size.y) / 2.0);
    return result;
}

// p in [-1,1]^2
fn sd_point(p: vec2f, kind: u32) -> f32 {
    switch kind {
        case GutterPointPointIcon, default {
            return length(p) - 0.57;
        }
        case PointStylePointIcon {
            return length(p) - 0.776;
        }
        case GutterPointOpenIcon, PointStyleOpenIcon {
            return abs(length(p) - 0.612) - 0.164;
        }
        case GutterPointCrossIcon, PointStyleCrossIcon {
            let q = abs(p);
            return hypot(max(q.x + q.y - 1.21, 0.0), abs(q.x - q.y)) - 0.238;
        }
        case GutterPointSquareIcon, PointStyleSquareIcon, GutterPointDiamondIcon, PointStyleDiamondIcon {
            let is_diamond = kind == GutterPointDiamondIcon || kind == PointStyleDiamondIcon;
            var q = select(p, (p + vec2(p.y, -p.x)) / sqrt(2.0), is_diamond);
            return sd_box(q, vec2(0.673));
        }
        case GutterPointPlusIcon, PointStylePlusIcon {
            let q = abs(p);
            return hypot(max(max(q.x, q.y) - 0.641, 0.0), min(q.x, q.y)) - 0.179;
        }
        case GutterPointTriangleIcon, PointStyleTriangleIcon {
            // https://www.shadertoy.com/view/Xl2yDW
            let k = sqrt(3.0);
            let r = k / 2.0;
            var q = p;
            q.y = -q.y;
            if kind == PointStyleTriangleIcon {
                q.y += 0.13;
            }
            q.x = abs(q.x);
            q -= vec2(0.5, 0.5 * k) * max(q.x + k * q.y, 0.0);
            q -= vec2(clamp(q.x, -r, r), -r / k);
            return length(q) * sign(-q.y);
        }
        case GutterPointStarIcon, PointStyleStarIcon {
            // https://www.shadertoy.com/view/3tSGDy
            let an = PI / 5.0;
            let en = PI * 0.3;
            let racs = vec2(cos(an), sin(an));
            let ecs = vec2(cos(en), sin(en));
            var q = p;
            let bn = modf32(atan2(q.x, -q.y), 2.0 * an) - an;
            q = length(q) * vec2(cos(bn), abs(sin(bn)));
            q -= racs;
            q += ecs * clamp(-dot(q, ecs), 0.0, racs.y / ecs.y);
            return length(q) * sign(q.x);
        }
    }
}

// p in [-0.5,0.5]^2
fn sd_draggable(p: vec2f, kind: u32) -> f32 {
    let a = abs(p);
    let b = a.yx;
    let sd = min(max(b - 0.04, a - 0.23), max((a + b - 0.33) / sqrt(2.0), 0.188 - a));
    
    switch kind {
        case GutterDragXIcon, PopupDragXIcon {
            return sd.x;
        }
        case GutterDragYIcon, PopupDragYIcon {
            return sd.y;
        }
        case GutterDragXYIcon, PopupDragXYIcon, default {
            return min(sd.x, sd.y);
        }
    }
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
            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(min(size.x, size.y) / 2.0));
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
        // TODO add extra data field to vertex to avoid combinatorial explosion
        case PopupRadioLeft, PopupRadioMiddle, PopupRadioRight,
             PopupRadioSelectedLeft, PopupRadioSelectedMiddle, PopupRadioSelectedRight,
             PopupRadioTopLeft, PopupRadioTopRight, PopupRadioBottomLeft, PopupRadioBottomRight,
             PopupRadioSelectedTopLeft, PopupRadioSelectedTopRight,
             PopupRadioSelectedBottomLeft, PopupRadioSelectedBottomRight {
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
                case PopupRadioTopLeft, PopupRadioSelectedTopLeft {
                    roundness = vec4(0.0, 0.0, 0.0, radius);
                }
                case PopupRadioTopRight, PopupRadioSelectedTopRight {
                    roundness = vec4(0.0, radius, 0.0, 0.0);
                }
                case PopupRadioBottomLeft, PopupRadioSelectedBottomLeft {
                    roundness = vec4(0.0, 0.0, radius, 0.0);
                }
                case PopupRadioBottomRight, PopupRadioSelectedBottomRight {
                    roundness = vec4(radius, 0.0, 0.0, 0.0);
                }
            }

            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, roundness);

            if in.kind == PopupRadioMiddle || in.kind == PopupRadioLeft || in.kind == PopupRadioRight ||
               in.kind == PopupRadioTopLeft || in.kind == PopupRadioTopRight || in.kind == PopupRadioBottomLeft ||
               in.kind == PopupRadioBottomRight {
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
        case ExpressionHiddenIcon {
            const THICKNESS = 5.0;
            let thickness = THICKNESS * uniforms.scale_factor;
            let radius = min(size.x, size.y) / 2.0;
            let p = (in.uv - 0.5) * size;
            let r = length(p);
            let sd = abs(r - radius + thickness / 2.0) - thickness / 2.0;
            let t = saturate((r - radius) / thickness + 1.0);
            let opacity = saturate(0.5 - sd) * mix(0.55, 1.0, t);
            return in.color * vec4(vec3(1.0), opacity);
        }
        case ExpressionShownIcon {
            const DARKENING_WIDTH = 4.0;
            let darkening_width = DARKENING_WIDTH * uniforms.scale_factor;
            let radius = min(size.x, size.y) / 2.0;
            let p = (in.uv - 0.5) * size;
            let r = length(p);
            let sd = r - radius;
            let t = saturate((r - radius) / darkening_width + 1.0);
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(vec3(mix(1.0, 0.9, t * t)), opacity);
        }
        case SineSolidIcon, SineDashedIcon, SineDottedIcon, SineFilledIcon {
            var p = in.uv - 0.5;
            let amplitude = 0.265;
            let frequency = 8.26;
            let radius = 0.075;
            var sd: f32;

            if in.kind == SineDottedIcon { 
                let x1 = 0.0465;
                let x2 = PI / (2.0 * frequency);
                let a = vec2(x1, amplitude * sin(frequency * x1));
                let b = vec2(x2, amplitude * sin(frequency * x2));
                p.x = 2.0 * clamp(p.x, -b.x, b.x) - p.x;
                p *= select(-1.0, 1.0, dot(a, p) > 0.0);
                sd = distance(p, select(a, b, dot(a - b, p) < dot((a + b) / 2.0, a - b)));
            } else {
                let n_iterations = select(2, 4, in.kind == SineFilledIcon);
                var x = sin_closest_x(p, amplitude, frequency, n_iterations);

                if in.kind == SineDashedIcon {
                    let a = floor(x * (frequency / PI)) * (PI / frequency);
                    x = clamp(x, 0.05 + a, PI / frequency - 0.05 + a);
                }

                sd = distance(p, vec2(x, amplitude * sin(frequency * x)));
            }

            if in.kind != SineFilledIcon {
                sd -= radius;
            } else if p.y > amplitude * sin(frequency * p.x) {
                sd = -sd;
            }

            sd *= size.x;
            return apply_shadow_and_circle_mask(sd, in.uv, in.color);
        }
        case PolygonSolidIcon, PolygonDashedIcon, PolygonDottedIcon, PolygonFilledIcon {
            var p = in.uv - 0.5;
            let p0 = vec2(-0.256, -0.243);
            let p1 = vec2(-0.1195, 0.2375);
            let p2 = vec2(0.259, 0.051);
            var sd: f32;

            if in.kind == PolygonFilledIcon {
                sd = sd_triangle(p, p0, p1, p2);
            } else {
                var k = 1.0;
                var n = vec3(1.0);
                if in.kind == PolygonDashedIcon {
                    k = 0.44;
                    n = vec3(3.0, 2.0, 3.0);
                } else if in.kind == PolygonDottedIcon {
                    k = 0.0;
                    n = vec3(4.0, 4.0, 5.0);
                }
                sd = min(min(
                    sd_segment_dashed(p, p0, p1, n.x, k),
                    sd_segment_dashed(p, p1, p2, n.y, k)),
                    sd_segment_dashed(p, p2, p0, n.z, k),
                ) - 0.0375;
            }

            sd *= size.x;
            return apply_shadow_and_circle_mask(sd, in.uv, in.color);
        }
        case GutterPointPointIcon, GutterPointOpenIcon, GutterPointCrossIcon, GutterPointSquareIcon,
             GutterPointPlusIcon, GutterPointTriangleIcon, GutterPointDiamondIcon, GutterPointStarIcon {
            let p = (in.uv - 0.5) / 0.223;
            var sd = sd_point(p, in.kind);
            sd *= size.x * 0.223;
            return apply_shadow_and_circle_mask(sd, in.uv, in.color);
        }
        case PointsIcon, LinesIcon {
            let p = in.uv - 0.5;
            let p0 = vec2(-0.2216, 0.23);
            let p1 = vec2(-0.155, -0.124);
            let p2 = vec2(0.09, 0.0915);
            let p3 = vec2(0.221, -0.2326);
            var sd: f32;

            if in.kind == PointsIcon {
                sd = min(min(min(
                    distance(p, p0),
                    distance(p, p1)),
                    distance(p, p2)),
                    distance(p, p3),
                ) - 0.086;
            } else {
                sd = min(min(
                    sd_segment(p, p0, p1),
                    sd_segment(p, p1, p2)),
                    sd_segment(p, p2, p3),
                ) - 0.0363;
            }

            sd *= size.x;
            return apply_shadow_and_circle_mask(sd, in.uv, in.color);
        }
        case InequalityDashedIcon {
            let p = in.uv - 0.5;
            let q = vec2(modf32((p.x - p.y) / sqrt(2.0), 0.385) - 0.1925, p.x + p.y);
            var sd = sd_box(q, vec2(0.108, 0.075));
            sd *= size.x;
            return apply_shadow_and_circle_mask(sd, in.uv, in.color);
        }
        case InequalityFilledIcon {
            var sd = 1.0 - in.uv.x - in.uv.y;
            sd *= size.x;
            return apply_shadow_and_circle_mask(sd, in.uv, in.color);
        }
        case PopupToggleShadow {
            let size = 1.0 / abs(vec2(dpdx(in.uv.x), dpdy(in.uv.y)));
            let shadow_radius = 2.0 * uniforms.scale_factor;
            let s = size - 4.0 * shadow_radius;
            let sd = sd_rounded_box(size * (in.uv - 0.5), s / 2.0, vec4(min(s.y, s.x) / 2.0));
            let shadow = saturate(1.0 - sd / (2.0 * shadow_radius));
            return in.color * vec4(1.0, 1.0, 1.0, smoothstep(0.0, 1.0, shadow));
        }
        case OpacityIcon {
            let p = in.uv - 0.5;
            let a = sd_rounded_box(p - 0.11, vec2(0.357), vec4(0.123));
            let b = sd_rounded_box(p + 0.11, vec2(0.357), vec4(0.123));
            var sd = min(min(abs(a), abs(b)), max(a, b)) - 0.033;
            sd *= size.x;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case ThicknessIcon {
            let y = in.uv.y;
            let opacity = f32(
                (0.13 < y && y < 0.21) ||
                (0.37 < y && y < 0.5) ||
                (0.65 < y && y < 0.86)
            );
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case LineStyleSolidIcon, LineStyleDashedIcon, LineStyleDottedIcon {
            let p = in.uv - 0.5;
            let r = 0.0737;
            let a = vec2(-0.5 + r, 0.5 - r);
            let b = vec2(0.5 - r, -0.5 + r);
            let ap = p - a;
            let ab = b - a;
            var t = dot(ap, ab) / dot(ab, ab);
            switch in.kind {
                case LineStyleSolidIcon, default {
                    t = saturate(t);
                }
                case LineStyleDashedIcon {
                    let b = 0.212;
                    if t < (2.0 - b) / 6.0 {
                        t = clamp(t, 0.0, (1.0 - 2.0 * b) / 3.0);
                    } else if t < (4.0 + b) / 6.0 {
                        t = clamp(t, (1.0 + b) / 3.0, (2.0 - b) / 3.0);
                    } else {
                        t = clamp(t, (2.0 + 2.0 * b) / 3.0, 1.0);
                    }
                }
                case LineStyleDottedIcon {
                    t = round(t * 5.0) / 5.0;
                }
            }
            var sd = distance(ap, ab * t) - r;
            sd *= size.x;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case PointStylePointIcon, PointStyleOpenIcon, PointStyleCrossIcon, PointStyleSquareIcon,
             PointStylePlusIcon, PointStyleTriangleIcon, PointStyleDiamondIcon, PointStyleStarIcon {
            let p = in.uv * 2.0 - 1.0;
            var sd = sd_point(p, in.kind);
            sd *= size.x / 2.0;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case GutterDragXIcon, GutterDragYIcon, GutterDragXYIcon {
            let p = in.uv - 0.5;
            var sd = sd_draggable(p, in.kind);
            sd *= size.x;
            return apply_shadow_and_circle_mask(sd, in.uv, in.color);
        }
        case PopupDragXIcon, PopupDragYIcon, PopupDragXYIcon {
            let p = in.uv - 0.5;
            var sd = sd_draggable(p, in.kind);
            sd *= size.x;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case PopupColorSwatch {
            const RADIUS = 3.0;
            let radius = RADIUS * uniforms.scale_factor;
            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(radius));
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case PopupColorSwatchHighlight {
            const RADIUS = 5.0;
            let radius = RADIUS * uniforms.scale_factor;
            let sd = sd_rounded_box(size * (in.uv - 0.5), size / 2.0, vec4(radius));
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
        case TickIcon {
            let p = (in.uv - 0.5) * vec2(1.0, 0.75);
            var sd = max(
                abs(p.x - 0.105) - p.y - 0.624,
                abs(p.y + abs(p.x + 0.105) - 0.234) - 0.137
            ) / sqrt(2.0);
            sd *= size.x;
            let opacity = saturate(0.5 - sd);
            return in.color * vec4(1.0, 1.0, 1.0, opacity);
        }
    }
}
