use std::collections::HashMap;

use bytemuck::{Zeroable, offset_of};
use derive_more::{Add, From, Into, Sub};
use glam::{DVec2, U16Vec2, Vec2, dvec2, u16vec2, uvec2, vec2};
use typed_index_collections::{TiVec, ti_vec};
use winit::{
    event::{ElementState, MouseButton},
    window::CursorIcon,
};

use crate::{
    graph::{
        Geometry,
        GeometryKind::{Line, Plot, Point},
    },
    math_field::{Cursor, Interactiveness, MathField, Message, UserSelection},
    ui::{Bounds, Context, Event, QuadKind, Response},
    utility::{mix, unmix},
};
use eval::{
    compiler::compile_assignments,
    vm::{self, Vm},
};
use parse::{
    analyze_expression_list::{ExpressionResult, PlotKind, analyze_expression_list},
    ast,
    ast_parser::{parse_standalone_expression, parse_statement},
    latex_parser::parse_latex,
    latex_tree::{self, Bracket},
    name_resolver::{Domain, ExpressionIndex, ExpressionListEntry},
    type_checker::Type,
};

#[derive(Default)]
struct Output {
    ui: OutputUi,
    data: OutputData,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DomainFocusState {
    None,
    Hovered,
    Focussed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DomainState {
    state: DomainFocusState,
    error: bool,
}

#[derive(Default)]
enum OutputUi {
    #[default]
    None,
    Slider {
        value: f64,
        min: f64,
        max: f64,
        dragging: Option<f64>,
        hovered: bool,
    },
    Field(MathField),
    Domain {
        name: String,
        name_field: MathField,
        min_state: DomainState,
        max_state: DomainState,
    },
}

impl OutputUi {
    fn field_from_latex(latex: &[latex_tree::Node]) -> OutputUi {
        let mut field = MathField::from(latex);
        field.interactiveness = Interactiveness::Select;
        field.scale = 18.0;
        field.left_padding = 0.22;
        field.right_padding = 0.4;
        field.bottom_padding = 0.19;
        field.top_padding = 0.25;
        OutputUi::Field(field)
    }

    fn set_domain(&mut self, name: &str) {
        if let OutputUi::Domain { name: existing, .. } = self
            && existing == name
        {
            return;
        }

        use latex_tree::Node;

        let mut latex = vec![];
        latex.push(Node::CtrlSeq("le"));
        let mut parts = name.split("_");
        let a = parts.next().unwrap();
        if a.chars().count() > 1 {
            latex.push(Node::CtrlSeq(a));
        } else {
            latex.push(Node::Char(a.chars().next().unwrap()));
        }
        if let Some(b) = parts.next() {
            let b = b.strip_prefix("{").unwrap().strip_suffix("}").unwrap();
            latex.push(Node::SubSup {
                sub: Some(b.chars().map(Node::Char).collect()),
                sup: None,
            });
        }
        latex.push(Node::CtrlSeq("le"));

        let mut field = MathField::from(&latex);
        field.interactiveness = Interactiveness::None;
        field.scale = 18.0;
        *self = OutputUi::Domain {
            name: name.into(),
            name_field: field,
            min_state: DomainState {
                state: DomainFocusState::None,
                error: false,
            },
            max_state: DomainState {
                state: DomainFocusState::None,
                error: false,
            },
        }
    }
}

#[derive(Debug, Default)]
enum OutputData {
    #[default]
    None,
    Error(String),
    DraggablePoint(Geometry),
    Geometry(Vec<Geometry>),
}

impl Output {
    const NONE: Output = Output {
        ui: OutputUi::None,
        data: OutputData::None,
    };

    fn new_error(error: String) -> Output {
        Output {
            ui: OutputUi::None,
            data: OutputData::Error(error),
        }
    }

    fn set_slider(&mut self, new_value: f64, new_min: f64, new_max: f64) {
        self.data = OutputData::None;
        if let OutputUi::Slider {
            value, min, max, ..
        } = self.ui
            && value == new_value
            && min == new_min
            && max == new_max
        {
            return;
        }
        self.ui = OutputUi::Slider {
            value: new_value,
            min: new_min,
            max: new_max,
            dragging: None,
            hovered: false,
        };
    }

    const SLIDER_BAR_RADIUS: f64 = 3.0;
    const SLIDER_POINT_RADIUS: f64 = 11.0;
    const DOMAIN_VALUE_MIN_WIDTH: f64 = 35.0;
    const DOMAIN_VALUE_MAX_WIDTH: f64 = 70.0;

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        padding: f64,
        top_left: DVec2,
        width: f64,
        parametric_domain: &mut Domain<(MathField, Result<ast::Expression, String>)>,
    ) -> (Response, Option<f64>, Option<Message>, Bounds) {
        match &mut self.ui {
            OutputUi::None => (Response::default(), None, None, Bounds::default()),
            OutputUi::Slider {
                value,
                min,
                max,
                dragging,
                hovered,
            } => {
                let mut response = Response::default();
                let point_radius = ctx.round_nonzero(Self::SLIDER_POINT_RADIUS);
                let left = top_left.x + padding;
                let right = top_left.x + width - padding;
                let mut point = dvec2(
                    mix(left, right, unmix(*value, *min, *max).clamp(0.0, 1.0)),
                    top_left.y + point_radius,
                );
                let point_bounds = Bounds {
                    pos: point - point_radius,
                    size: DVec2::splat(2.0 * point_radius),
                };
                let new_hovered = point_bounds.contains(ctx.cursor);
                let mut new_value = None;

                match event {
                    Event::CursorMoved { .. } if dragging.is_some() => {
                        let offset = dragging.unwrap();
                        point.x = (ctx.cursor.x + offset).clamp(left, right);
                        *value = mix(*min, *max, unmix(point.x, left, right));
                        new_value = Some(*value);
                        response.consume_event();
                        response.request_redraw();
                    }
                    Event::MouseInput(ElementState::Pressed, MouseButton::Left) if *hovered => {
                        *dragging = Some(point.x - ctx.cursor.x);
                        response.consume_event();
                    }
                    Event::MouseInput(ElementState::Released, MouseButton::Left)
                        if dragging.is_some() =>
                    {
                        *dragging = None;
                        response.consume_event();
                    }
                    _ => {}
                }

                let new_hovered = new_hovered || dragging.is_some();

                if *hovered != new_hovered {
                    *hovered = new_hovered;
                    response.request_redraw();
                }

                #[cfg(not(windows))]
                let (grab, grabbing) = (CursorIcon::Grab, CursorIcon::Grabbing);

                // https://github.com/rust-windowing/winit/issues/1043
                #[cfg(windows)]
                let (grab, grabbing) = (CursorIcon::EwResize, CursorIcon::EwResize);

                if dragging.is_some() {
                    response.cursor_icon = grabbing;
                } else if *hovered {
                    response.cursor_icon = grab;
                }

                let bounds = Bounds {
                    pos: top_left,
                    size: dvec2(width, point_radius * 2.0),
                };

                (response, new_value, None, bounds)
            }
            OutputUi::Field(field) => {
                let size = field.expression_size().map(|s| ctx.ceil(s));
                let right = top_left.x + width - 0.5 * padding;
                let left = (right - size.x).max(top_left.x + padding);
                let bounds = Bounds {
                    pos: dvec2(left, top_left.y),
                    size: dvec2(right - left, size.y),
                };
                let (mut response, _) = field.update(ctx, event, bounds);

                if let Some(UserSelection { anchor, focus }) = field.get_selection() {
                    let mut clamp = |mut cursor: Cursor| {
                        let index = cursor
                            .path
                            .first_mut()
                            .map_or(&mut cursor.index, |(index, _)| index);
                        if *index == 0 {
                            *index = 1;
                            response.request_redraw();
                        }
                        cursor
                    };
                    field.set_selection((clamp(anchor.clone()), clamp(focus.clone())));
                }

                (response, None, None, bounds)
            }
            OutputUi::Domain {
                name: _,
                name_field,
                min_state,
                max_state,
            } => {
                let mut size1 = parametric_domain
                    .min
                    .0
                    .expression_size()
                    .map(|s| ctx.ceil(s));
                let size2 = name_field.expression_size().map(|s| ctx.ceil(s));
                let mut size3 = parametric_domain
                    .max
                    .0
                    .expression_size()
                    .map(|s| ctx.ceil(s));

                size1.x = size1
                    .x
                    .clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);
                size3.x = size3
                    .x
                    .clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);

                let height = size1.y.max(size2.y).max(size3.y);

                let left1 = top_left.x + padding;
                let left2 = left1 + size1.x;
                let left3 = left2 + size2.x;

                let bounds1 = Bounds {
                    pos: dvec2(left1, top_left.y + (height - size1.y) / 2.0),
                    size: size1,
                };
                let bounds2 = Bounds {
                    pos: dvec2(left2, top_left.y + (height - size2.y) / 2.0),
                    size: size2,
                };
                let bounds3 = Bounds {
                    pos: dvec2(left3, top_left.y + (height - size3.y) / 2.0),
                    size: size3,
                };

                let (mut response1, mut message1) =
                    parametric_domain.min.0.update(ctx, event, bounds1);
                let (response2, _) = name_field.update(ctx, event, bounds2);
                let (mut response3, mut message3) =
                    parametric_domain.max.0.update(ctx, event, bounds3);

                match message1 {
                    Some(Message::ContentsChanged) => {
                        let mut latex = parametric_domain.min.0.to_latex();
                        if latex.is_empty() {
                            latex = parametric_domain.min.0.get_placeholder();
                        }
                        parametric_domain.min.1 = parse_standalone_expression(&latex)
                    }
                    Some(Message::Left | Message::Remove) => message1 = None,
                    Some(Message::Right) => {
                        message1 = None;
                        parametric_domain.min.0.unfocus();
                        parametric_domain.max.0.select_all();
                    }
                    Some(Message::Up | Message::Down | Message::Add) => {
                        parametric_domain.min.0.unfocus()
                    }
                    None => {}
                }

                match message3 {
                    Some(Message::ContentsChanged) => {
                        let mut latex = parametric_domain.max.0.to_latex();
                        if latex.is_empty() {
                            latex = parametric_domain.max.0.get_placeholder();
                        }
                        parametric_domain.max.1 = parse_standalone_expression(&latex)
                    }
                    Some(Message::Left) => {
                        message3 = None;
                        parametric_domain.max.0.unfocus();
                        parametric_domain.min.0.select_all();
                    }
                    Some(Message::Right | Message::Remove) => message3 = None,
                    Some(Message::Up | Message::Down | Message::Add) => {
                        parametric_domain.max.0.unfocus()
                    }
                    None => {}
                }

                let f = |f: &MathField, b: Bounds, s: &mut DomainFocusState, r: &mut Response| {
                    let new = if f.has_focus() {
                        DomainFocusState::Focussed
                    } else if b.contains(ctx.cursor) {
                        DomainFocusState::Hovered
                    } else {
                        DomainFocusState::None
                    };
                    if new != *s {
                        *s = new;
                        r.request_redraw();
                    }
                };
                f(
                    &parametric_domain.min.0,
                    bounds1,
                    &mut min_state.state,
                    &mut response1,
                );
                f(
                    &parametric_domain.max.0,
                    bounds3,
                    &mut max_state.state,
                    &mut response3,
                );

                let response = response1.or(response2).or(response3);

                let bounds = bounds1.union(bounds2).union(bounds3);

                (response, None, message1.or(message3), bounds)
            }
        }
    }

    fn render(
        &mut self,
        ctx: &Context,
        padding: f64,
        top_left: DVec2,
        width: f64,
        parametric_domain: &mut Domain<(MathField, Result<ast::Expression, String>)>,
        draw_quad: &mut impl FnMut(DVec2, DVec2, QuadKind),
    ) -> f64 {
        match &mut self.ui {
            OutputUi::None => 0.0,
            OutputUi::Slider {
                value,
                min,
                max,
                hovered,
                ..
            } => {
                let point_radius = ctx.round_nonzero(Self::SLIDER_POINT_RADIUS);
                let bar_radius = ctx.round_nonzero(Self::SLIDER_BAR_RADIUS);
                let left = top_left.x + padding;
                let right = top_left.x + width - padding;
                let point = dvec2(
                    mix(left, right, unmix(*value, *min, *max).clamp(0.0, 1.0)),
                    top_left.y + point_radius,
                );
                let bounds = Bounds {
                    pos: top_left,
                    size: dvec2(width, point_radius * 2.0),
                };

                draw_quad(
                    ctx.scale_factor * (dvec2(left, point.y) - bar_radius),
                    ctx.scale_factor * (dvec2(right, point.y) + bar_radius),
                    QuadKind::SliderBar,
                );
                draw_quad(
                    ctx.scale_factor * (point - point_radius),
                    ctx.scale_factor * (point + point_radius),
                    QuadKind::SliderPointOuter,
                );
                let inner_radius = if *hovered { point_radius } else { bar_radius };
                draw_quad(
                    ctx.scale_factor * (point - inner_radius),
                    ctx.scale_factor * (point + inner_radius),
                    QuadKind::SliderPointInner,
                );

                bounds.size.y
            }
            OutputUi::Field(field) => {
                let size = field.expression_size().map(|s| ctx.ceil(s));
                let right = top_left.x + width - 0.5 * padding;
                let left = (right - size.x).max(top_left.x + padding);
                let bounds = Bounds {
                    pos: dvec2(left, top_left.y),
                    size: dvec2(right - left, size.y),
                };
                draw_quad(
                    ctx.scale_factor * bounds.pos,
                    ctx.scale_factor * (bounds.pos + bounds.size),
                    QuadKind::OutputValueBox,
                );
                field.render(ctx, bounds, draw_quad);
                bounds.size.y
            }
            OutputUi::Domain {
                name: _,
                name_field,
                min_state,
                max_state,
            } => {
                let mut size1 = parametric_domain
                    .min
                    .0
                    .expression_size()
                    .map(|s| ctx.ceil(s));
                let size2 = name_field.expression_size().map(|s| ctx.ceil(s));
                let mut size3 = parametric_domain
                    .max
                    .0
                    .expression_size()
                    .map(|s| ctx.ceil(s));

                size1.x = size1
                    .x
                    .clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);
                size3.x = size3
                    .x
                    .clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);

                let height = size1.y.max(size2.y).max(size3.y);

                let left1 = top_left.x + padding;
                let left2 = left1 + size1.x;
                let left3 = left2 + size2.x;

                let bounds1 = Bounds {
                    pos: dvec2(left1, top_left.y + (height - size1.y) / 2.0),
                    size: size1,
                };
                let bounds2 = Bounds {
                    pos: dvec2(left2, top_left.y + (height - size2.y) / 2.0),
                    size: size2,
                };
                let bounds3 = Bounds {
                    pos: dvec2(left3, top_left.y + (height - size3.y) / 2.0),
                    size: size3,
                };

                let mut f = |b: Bounds, s: DomainState| {
                    draw_quad(
                        ctx.scale_factor * (b.pos + dvec2(0.0, b.size.y - 1.0)),
                        ctx.scale_factor
                            * (b.pos
                                + b.size
                                + dvec2(
                                    0.0,
                                    if s.state == DomainFocusState::None && !s.error {
                                        0.0
                                    } else {
                                        1.0
                                    },
                                )),
                        match s.state {
                            _ if s.error => QuadKind::DomainBoundError,
                            DomainFocusState::None | DomainFocusState::Hovered => {
                                QuadKind::DomainBoundUnfocussed
                            }
                            DomainFocusState::Focussed => QuadKind::DomainBoundFocussed,
                        },
                    )
                };

                f(bounds1, *min_state);
                f(bounds3, *max_state);

                parametric_domain.min.0.render(ctx, bounds1, draw_quad);
                name_field.render(ctx, bounds2, draw_quad);
                parametric_domain.max.0.render(ctx, bounds3, draw_quad);

                let bounds = bounds1.union(bounds2).union(bounds3);
                bounds.size.y
            }
        }
    }
}

struct Expression {
    field: MathField,
    parametric_domain: Domain<(MathField, Result<parse::ast::Expression, String>)>,
    ast: Option<Result<parse::ast::Statement, String>>,
    output: Output,
}

impl Default for Expression {
    fn default() -> Self {
        use latex_tree::Node;
        let f = |c| {
            let mut f = MathField::default();
            f.set_placeholder(&[Node::Char(c)]);
            f.scale = 15.7;
            f.bottom_padding = 0.25;
            f.top_padding = 0.25;
            f
        };
        Self {
            field: Default::default(),
            parametric_domain: Domain {
                min: (f('0'), Ok(ast::Expression::Number(0.0))),
                max: (f('1'), Ok(ast::Expression::Number(1.0))),
            },
            ast: None,
            output: Default::default(),
        }
    }
}

impl From<&[latex_tree::Node<'_>]> for Expression {
    fn from(latex: &[latex_tree::Node]) -> Self {
        let mut e = Expression::default();
        e.set_latex(latex);
        e
    }
}

impl Expression {
    const PADDING: f64 = 16.0;

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        top_left: DVec2,
        width: f64,
    ) -> (Response, Option<Message>, f64) {
        let mut response = Response::default();
        let mut message = None;
        let mut height = 0.0;

        let padding = ctx.round(Self::PADDING);
        height += padding;
        let field_bounds = Bounds {
            pos: top_left + dvec2(padding, height),
            size: dvec2(
                width - padding * 1.5,
                ctx.ceil(self.field.expression_size().y),
            ),
        };
        height += field_bounds.size.y;
        height += 0.5 * padding;
        let (output_response, new_value, output_message, output_bounds) = self.output.update(
            ctx,
            event,
            padding,
            top_left + dvec2(0.0, height),
            width,
            &mut self.parametric_domain,
        );

        // Update new value from slider
        if let Some(value) = new_value {
            use latex_tree::Node::Char as C;
            let name = self
                .field
                .to_latex()
                .iter()
                .take_while(|n| n != &&C('='))
                .cloned()
                .collect::<Vec<_>>();
            let mut latex = name;
            latex.push(C('='));
            latex.extend(value.to_string().chars().map(C));
            self.set_latex(&latex);
            message = Some(Message::ContentsChanged);
        }

        response = response.or(output_response);
        height += output_bounds.size.y;
        height += 0.5 * padding;

        let bounds = Bounds {
            pos: top_left,
            size: dvec2(width, height),
        };

        let mut r = Response::default();
        if bounds.contains(ctx.cursor)
            && !field_bounds.contains(ctx.cursor)
            && !output_bounds.contains(ctx.cursor)
        {
            r.cursor_icon = CursorIcon::Pointer;
            if event == &Event::MouseInput(ElementState::Pressed, MouseButton::Left) {
                r.consume_event();
                if !self.field.has_focus() {
                    self.field.focus();
                    r.request_redraw();
                }
            }
        }
        response = response.or(r.or_else(|| {
            let (r, m) = self.field.update(ctx, event, field_bounds);
            if m == Some(Message::ContentsChanged) {
                self.parse_ast();
            }
            if message.is_none() {
                message = m;
            }
            r
        }));

        // Maybe the parametric domain got changed
        if let Some(m) = output_message {
            message = match m {
                Message::ContentsChanged | Message::Down | Message::Add => Some(m),
                Message::Left | Message::Right | Message::Remove => unreachable!(),
                Message::Up => {
                    self.focus();
                    None
                }
            };
        }

        (response, message, height)
    }

    fn set_latex(&mut self, latex: &[latex_tree::Node]) {
        self.field = MathField::from(latex);
        self.parse_ast();
    }

    fn parse_ast(&mut self) {
        let latex = self.field.to_latex();
        self.ast = latex
            .iter()
            .any(|n| n != &latex_tree::Node::Char(' '))
            .then(|| parse_statement(&latex));
    }

    fn focus(&mut self) {
        self.field.focus();
    }

    fn unfocus(&mut self) {
        self.field.unfocus();
    }

    fn has_focus(&self) -> bool {
        self.field.has_focus()
    }

    fn render(
        &mut self,
        ctx: &Context,
        top_left: DVec2,
        width: f64,
        draw_quad: &mut impl FnMut(DVec2, DVec2, QuadKind),
    ) -> f64 {
        let mut height = 0.0;

        let padding = ctx.round(Self::PADDING);
        height += padding;
        let field_bounds = Bounds {
            pos: top_left + dvec2(padding, height),
            size: dvec2(
                width - padding * 1.5,
                ctx.ceil(self.field.expression_size().y),
            ),
        };
        height += field_bounds.size.y;
        height += 0.5 * padding;
        height += self.output.render(
            ctx,
            padding,
            top_left + dvec2(0.0, height),
            width,
            &mut self.parametric_domain,
            draw_quad,
        );
        height += 0.5 * padding;

        self.field.render(ctx, field_bounds, draw_quad);

        height
    }
}

#[derive(Debug, Clone, Copy, From, Into, Add, Sub, PartialEq)]
pub struct ExpressionId(usize);

pub struct ExpressionList {
    expressions: TiVec<ExpressionId, Expression>,
    expressions_changed: bool,
    scroll: f64,
    height: f64,
    expression_bottoms: TiVec<ExpressionId, f64>,
    vm_vars: vm::Vars,

    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniforms_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Uniforms {
    resolution: Vec2,
    scale_factor: f32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Vertex {
    position: Vec2,
    uv: U16Vec2,
    kind: u32,
}

fn create_index_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("latex_index_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDEX,
        mapped_at_creation: false,
    })
}

fn create_vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("latex_vertex_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

impl ExpressionList {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("latex.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("latex_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("latex"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("latex_pipeline_layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_latex"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as _,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: offset_of!(Vertex::zeroed(), Vertex, position) as _,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm16x2,
                            offset: offset_of!(Vertex::zeroed(), Vertex, uv) as _,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: offset_of!(Vertex::zeroed(), Vertex, kind) as _,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_latex"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let index_buffer = create_index_buffer(device, 256);
        let vertex_buffer = create_vertex_buffer(device, 256);

        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms_buffer"),
            size: size_of::<Uniforms>().next_multiple_of(16) as _,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let font_image = image::load_from_memory(include_bytes!("KaTeX.png")).unwrap();
        let font_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("font_texture"),
            size: wgpu::Extent3d {
                width: font_image.width(),
                height: font_image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &font_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &font_image.to_rgba8(),
            wgpu::TexelCopyBufferLayout {
                bytes_per_row: Some(4 * font_image.width()),
                ..Default::default()
            },
            font_texture.size(),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("latex_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        uniforms_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &font_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&device.create_sampler(
                        &wgpu::SamplerDescriptor {
                            label: Some("bilinear"),
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            ..Default::default()
                        },
                    )),
                },
            ],
        });

        let expressions = [
            // "n=10",
            // "p_{00}=\\left(-1.6,-5.35\\right)",
            // "p_{10}=\\left(6.05,0.15\\right)",
            // "p_{01}=\\left(-5.5,2.73\\right)",
            // "p_{11}=\\left(2.3,4.62\\right)",
            // "",
            // "q_{1}=p_{10}-p_{00}",
            // "q_{2}=p_{11}-p_{01}",
            // "q_{3}=p_{01}-p_{00}",
            // "q_{4}=p_{11}-p_{10}",
            // "W\\left(x,y\\right)=x.xy.y-x.yy.x",
            // "u=0.52",
            // "v=0.7",
            // "\\alpha_{1}=\\frac{W\\left(p_{11}-p_{00},q_{2}\\right)}{W\\left(uq_{1}+q_{4},q_{2}\\right)}q_{1}",
            // "\\alpha_{2}=\\frac{W\\left(p_{10}-p_{01},q_{1}\\right)}{W\\left(uq_{2}-q_{4},q_{1}\\right)}q_{2}",
            // "\\alpha=\\alpha_{2}-\\alpha_{1}",
            // "p=p_{00}+u\\alpha_{1}+v\\frac{W\\left(\\alpha_{1},q_{3}\\right)}{W\\left(\\alpha_{2}-v\\alpha,q_{3}\\right)}\\left(q_{3}+u\\alpha\\right)",
            // "p\\operatorname{for}u=\\frac{\\left[0...n\\right]}{n},v=\\frac{\\left[0...n\\right]}{n}",
        ];
        Self {
            expressions: expressions
                .iter()
                .chain(Some(&""))
                .map(|s| Expression::from(parse_latex(s).unwrap().as_slice()))
                .collect(),
            expressions_changed: true,
            scroll: 0.0,
            height: 0.0,
            expression_bottoms: ti_vec![],
            vm_vars: Default::default(),

            pipeline,
            vertex_buffer,
            index_buffer,
            uniforms_buffer,
            bind_group,
        }
    }

    pub fn point_dragged(&mut self, i: ExpressionId, p: DVec2) {
        use parse::latex_tree::Node::{self, Char as C};
        let name = self.expressions[i]
            .field
            .to_latex()
            .iter()
            .take_while(|n| n != &&C('='))
            .cloned()
            .collect::<Vec<_>>();
        let mut latex = name;
        latex.push(C('='));
        let mut inner = vec![];
        inner.extend(p.x.to_string().chars().map(C));
        inner.push(C(','));
        inner.extend(p.y.to_string().chars().map(C));
        latex.push(Node::DelimitedGroup {
            left: Bracket::Paren,
            right: Bracket::Paren,
            inner,
        });
        self.expressions[i].set_latex(&latex);
        self.expressions_changed = true;
    }

    // Positive `delta` moves the expressions down
    fn scroll(&mut self, delta: f64) {
        const SCROLL_EXTRA: f64 = 50.0;
        self.scroll = (self.scroll - delta)
            .min(SCROLL_EXTRA + self.expression_bottoms.last().unwrap_or(&0.0) - self.height)
            .max(0.0);
    }

    fn scroll_into_view(&mut self, i: ExpressionId) {
        const SCROLL_PADDING: f64 = 25.0;
        let bottom = self.expression_bottoms[i];
        let top = if i.0 == 0 {
            0.0
        } else {
            self.expression_bottoms[i - 1.into()]
        };
        self.scroll((self.height - SCROLL_PADDING - (bottom - self.scroll)).min(0.0));
        self.scroll((SCROLL_PADDING - (top - self.scroll)).max(0.0));
    }

    const SEPARATOR_WIDTH: f64 = 1.0;

    pub fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        bounds: Bounds,
    ) -> (Response, Option<(Vec<Geometry>, vm::Vars)>) {
        self.height = bounds.size.y;
        let mut response = Response::default();
        let mut redraw_geometry = false;

        match event {
            Event::MouseWheel(delta)
                if bounds.contains(ctx.cursor) && delta.abs().y >= delta.x.abs() =>
            {
                self.scroll(delta.y);
                response.consume_event();
                response.request_redraw();
            }
            _ => {
                let mut next_y = bounds.pos.y - self.scroll;
                let mut message = None;
                let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
                let expression_width = bounds.size.x - separator_width;
                let mut original_focus = None;
                self.expression_bottoms.clear();

                for (i, expression) in self.expressions.iter_mut_enumerated() {
                    if expression.has_focus() {
                        original_focus = Some(i);
                    }

                    let (r, m, height) = expression.update(
                        ctx,
                        event,
                        dvec2(bounds.pos.x, next_y),
                        expression_width,
                    );
                    next_y += height;
                    response = response.or(r);
                    message = message.or(m.map(|m| (i, m)));
                    next_y += separator_width;
                    self.expression_bottoms
                        .push(next_y - (bounds.pos.y - self.scroll));
                }

                if let Some((i, m)) = message {
                    match m {
                        Message::ContentsChanged => {
                            self.expressions_changed = true;
                            self.scroll_into_view(i);
                        }
                        Message::Left | Message::Right => {}
                        Message::Up => {
                            if i.0 > 0 {
                                self.expressions[i].unfocus();
                                self.expressions[i - 1.into()].focus();
                                response.request_redraw();
                            }
                        }
                        Message::Down => {
                            if i.0 == self.expressions.len() - 1 {
                                self.expressions.push(Default::default());
                            }
                            self.expressions[i].unfocus();
                            self.expressions[i + 1.into()].focus();
                            response.request_redraw();
                        }
                        Message::Add => {
                            self.expressions_changed = true;
                            self.expressions.insert(i + 1.into(), Default::default());
                            self.expressions[i].unfocus();
                            self.expressions[i + 1.into()].focus();
                            response.request_redraw();
                        }
                        Message::Remove => {
                            self.expressions.remove(i);
                            self.expressions_changed = true;
                            if self.expressions.is_empty() {
                                self.expressions.push(Default::default());
                            }
                            self.expressions[ExpressionId(i.0.saturating_sub(1))].focus();
                            response.request_redraw();
                        }
                    }
                }

                if self.expressions.last().unwrap().has_focus() {
                    self.expressions.push(Default::default());
                    response.request_redraw();
                }

                let new_focus = self
                    .expressions
                    .iter_enumerated()
                    .find_map(|(i, e)| e.has_focus().then_some(i));
                redraw_geometry |= self.expressions_changed || original_focus != new_focus;

                if let Some(i) = new_focus
                    && original_focus != new_focus
                {
                    self.scroll_into_view(i);
                }

                if self.expressions_changed {
                    use latex_tree::Node::{self, Char as C};
                    let number = |nodes: &mut Vec<Node>, mut x: f64| {
                        if x.is_nan() {
                            nodes.push(Node::Frac {
                                num: vec![C('0')],
                                den: vec![C('0')],
                            });
                            return;
                        }
                        if x.is_sign_negative() {
                            nodes.push(C('-'));
                            x = -x;
                        }
                        if x.is_infinite() {
                            nodes.push(Node::CtrlSeq("infty"));
                            return;
                        }
                        let mut buffer = ryu::Buffer::new();
                        let mut s = buffer.format_finite(x).split('e');
                        let m = s.next().unwrap();
                        nodes.extend(m.strip_suffix(".0").unwrap_or(m).chars().map(C));
                        if let Some(e) = s.next() {
                            nodes.extend([
                                Node::CtrlSeq("times"),
                                C('1'),
                                C('0'),
                                Node::SubSup {
                                    sub: None,
                                    sup: Some(e.chars().map(C).collect()),
                                },
                            ]);
                        }
                    };
                    let colors = [
                        [0.780, 0.267, 0.251, 1.0],
                        [0.176, 0.439, 0.702, 1.0],
                        [0.204, 0.522, 0.263, 1.0],
                        [0.376, 0.259, 0.651, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ];
                    let point = |nodes: &mut Vec<Node>, x: f64, y: f64| {
                        let mut inner = vec![];
                        number(&mut inner, x);
                        inner.push(C(','));
                        number(&mut inner, y);
                        nodes.push(Node::DelimitedGroup {
                            left: Bracket::Paren,
                            right: Bracket::Paren,
                            inner,
                        });
                    };

                    let mut ei_to_oi: TiVec<ExpressionIndex, ExpressionId> = ti_vec![];
                    let mut list: TiVec<ExpressionIndex, _> = ti_vec![];

                    for (i, e) in self.expressions.iter_mut_enumerated() {
                        let ast = match &e.ast {
                            Some(Ok(ast)) => ast,
                            Some(Err(err)) => {
                                e.output = Output::new_error(format!("parse error: {err}"));
                                continue;
                            }
                            None => {
                                e.output = Output::NONE;
                                continue;
                            }
                        };
                        fn get_number(e: &parse::ast::Expression) -> Option<f64> {
                            match e {
                                parse::ast::Expression::Number(x) => Some(*x),
                                parse::ast::Expression::Op {
                                    operation: parse::op::OpName::Neg,
                                    args: arguments,
                                } => Some(-get_number(
                                    arguments.first().expect("neg should have one argument"),
                                )?),
                                _ => None,
                            }
                        }
                        if let parse::ast::Statement::Assignment { value, .. } = ast {
                            if let Some(value) = get_number(value) {
                                e.output.set_slider(value, -10.0, 10.0);
                            } else if let parse::ast::Expression::Op {
                                operation: parse::op::OpName::Point,
                                args: arguments,
                            } = value
                                && let Some(x) = get_number(&arguments[0])
                                && let Some(y) = get_number(&arguments[1])
                            {
                                let mut latex = vec![C('=')];
                                point(&mut latex, x, y);
                                e.output = Output {
                                    ui: OutputUi::None,
                                    data: OutputData::DraggablePoint(Geometry {
                                        width: 8.0,
                                        color: colors[i.0 % colors.len()],
                                        kind: Point {
                                            p: dvec2(x, y),
                                            draggable: Some(i),
                                        },
                                    }),
                                };
                            } else {
                                e.output.data = OutputData::None;
                            }
                        } else {
                            e.output.data = OutputData::None;
                        }
                        list.push(ExpressionListEntry {
                            expression: ast,
                            parametric_domain: Domain {
                                min: match &e.parametric_domain.min.1 {
                                    Ok(ast) => ast,
                                    Err(_) => &ast::Expression::Number(0.0),
                                },
                                max: match &e.parametric_domain.max.1 {
                                    Ok(ast) => ast,
                                    Err(_) => &ast::Expression::Number(1.0),
                                },
                            },
                        });
                        ei_to_oi.push(i);
                    }

                    let analysis = analyze_expression_list(&list, false);

                    let mut function_id_map = HashMap::new();
                    let (program, mut functions, var_indices) = compile_assignments(
                        analysis.constants.iter().map(|&i| &analysis.assignments[i]),
                        analysis.results.iter_enumerated().filter_map(|(i, r)| {
                            let ExpressionResult::Plot {
                                parameters,
                                assignments,
                                ..
                            } = r
                            else {
                                return None;
                            };
                            function_id_map.insert(i, function_id_map.len());
                            Some((
                                parameters.iter().cloned(),
                                assignments.iter().map(|&i| &analysis.assignments[i]),
                            ))
                        }),
                    );
                    let mut functions = function_id_map
                        .into_iter()
                        .map(|(id, i)| (id, std::mem::take(&mut functions[i])))
                        .collect::<HashMap<_, _>>();
                    let mut vm = Vm::new(&program, Default::default());
                    vm.run(false);

                    'results_loop: for (ei, r) in analysis.results.into_iter_enumerated() {
                        let i = ei_to_oi[ei];
                        let expression = &mut self.expressions[i];
                        let output = &mut expression.output;

                        if let OutputData::Error(_) = output.data {
                            continue 'results_loop;
                        }

                        match r {
                            ExpressionResult::None => *output = Output::NONE,
                            ExpressionResult::Err(e) => {
                                *output = Output::new_error(format!("analysis error: {e}"))
                            }
                            ExpressionResult::Value(id, ty)
                            | ExpressionResult::Plot { value: id, ty, .. } => {
                                let mut nodes = vec![C('=')];

                                let color = colors[i.0 % colors.len()];
                                let mut geometry = vec![];
                                let mut draw_point = |x: f64, y: f64| {
                                    geometry.push(Geometry {
                                        width: 8.0,
                                        color,
                                        kind: Point {
                                            p: dvec2(x, y),
                                            draggable: None,
                                        },
                                    });
                                };
                                let list_limit = 10;

                                if let ExpressionResult::Plot {
                                    ref kind,
                                    value,
                                    ref parameters,
                                    ..
                                } = r
                                {
                                    let kind = match kind {
                                        PlotKind::Normal => PlotKind::Normal,
                                        PlotKind::Inverse => PlotKind::Inverse,
                                        PlotKind::Parametric(d) => {
                                            output
                                                .ui
                                                .set_domain(&analysis.freevars[&parameters[0]]);

                                            let min = match &expression.parametric_domain.min.1 {
                                                Ok(_) => match &d.min {
                                                    Ok(id) => match vm.vars[var_indices[id]]
                                                        .clone()
                                                        .number()
                                                    {
                                                        x if x.is_finite() => Ok(x),
                                                        _ => Err(
                                                            "value error: domain bound should be finite".into(),
                                                        ),
                                                    },
                                                    Err(e) => Err(format!("analysis error: {e}")),
                                                },
                                                Err(e) => Err(format!("parse error: {e}")),
                                            };
                                            let max = match &expression.parametric_domain.max.1 {
                                                Ok(_) => match &d.max {
                                                    Ok(id) => match vm.vars[var_indices[id]]
                                                        .clone()
                                                        .number()
                                                    {
                                                        x if x.is_finite() => Ok(x),
                                                        _ => Err(
                                                            "value error: domain bound should be finite".into(),
                                                        ),
                                                    },
                                                    Err(e) => Err(format!("analysis error: {e}")),
                                                },
                                                Err(e) => Err(format!("parse error: {e}")),
                                            };
                                            let OutputUi::Domain {
                                                min_state,
                                                max_state,
                                                ..
                                            } = &mut output.ui
                                            else {
                                                unreachable!()
                                            };
                                            min_state.error = min.is_err();
                                            max_state.error = max.is_err();

                                            let (min, max) = match (min, max) {
                                                (Ok(min), Ok(max)) => (min, max),
                                                (min, max) => {
                                                    output.data = OutputData::Error(
                                                        [("min", min), ("max", max)]
                                                            .into_iter()
                                                            .filter_map(|(n, m)| {
                                                                m.err().map(|e| {
                                                                    format!("(parametric {n}) {e}")
                                                                })
                                                            })
                                                            .collect::<Vec<_>>()
                                                            .join("\n"),
                                                    );
                                                    continue 'results_loop;
                                                }
                                            };

                                            if min > max {
                                                output.data = OutputData::Error("invalid domain limits: min should be less than max".into());
                                                min_state.error = true;
                                                max_state.error = true;
                                                continue 'results_loop;
                                            }

                                            PlotKind::Parametric(Domain { min, max })
                                        }
                                        PlotKind::Implicit => PlotKind::Implicit,
                                    };
                                    output.data = OutputData::Geometry(vec![Geometry {
                                        width: 2.5,
                                        color,
                                        kind: Plot {
                                            kind,
                                            inputs: parameters
                                                .iter()
                                                .map(|p| var_indices[p])
                                                .collect(),
                                            output: var_indices[&value],
                                            instructions: functions.remove(&ei).unwrap(),
                                        },
                                    }]);
                                }

                                if match r {
                                    ExpressionResult::Plot { parameters, .. } => {
                                        parameters.is_empty()
                                    }
                                    _ => true,
                                } {
                                    let v = var_indices[&id];
                                    match ty {
                                        Type::Number => {
                                            number(&mut nodes, vm.vars[v].clone().number())
                                        }
                                        Type::NumberList => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, x) in a.borrow().as_slice().iter().enumerate() {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    number(&mut inner, *x);
                                                } else {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                    break;
                                                }
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Point => {
                                            let x = vm.vars[v].clone().number();
                                            let y = vm.vars[v + 1.into()].clone().number();
                                            draw_point(x, y);
                                            point(&mut nodes, x, y);
                                        }
                                        Type::PointList => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, p) in a.borrow().chunks(2).enumerate() {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    point(&mut inner, p[0], p[1]);
                                                } else if i == list_limit {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                }
                                                draw_point(p[0], p[1]);
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Polygon => {
                                            let a = vm.vars[v].clone().list();
                                            let a = a.borrow();
                                            geometry.push(Geometry {
                                                width: 2.5,
                                                color,
                                                kind: Line(
                                                    a.chunks(2)
                                                        .chain(a.chunks(2).take(if a.len() > 2 {
                                                            1
                                                        } else {
                                                            0
                                                        }))
                                                        .map(|p| dvec2(p[0], p[1]))
                                                        .collect(),
                                                ),
                                            });
                                        }
                                        Type::PolygonList => {
                                            let a = vm.vars[v].clone().polygon_list();
                                            geometry.extend(a.borrow().iter().map(|a| {
                                                let a = a.borrow();
                                                Geometry {
                                                    width: 2.5,
                                                    color,
                                                    kind: Line(
                                                        a.chunks(2)
                                                            .chain(a.chunks(2).take(
                                                                if a.len() > 2 { 1 } else { 0 },
                                                            ))
                                                            .map(|p| dvec2(p[0], p[1]))
                                                            .collect(),
                                                    ),
                                                }
                                            }));
                                        }
                                        Type::Bool | Type::BoolList => unreachable!(),
                                        Type::EmptyList => nodes.push(Node::DelimitedGroup {
                                            left: Bracket::Square,
                                            right: Bracket::Square,
                                            inner: vec![],
                                        }),
                                    }

                                    if ty.as_single() == Type::Polygon {
                                        output.ui = OutputUi::None;
                                    } else if let OutputUi::None = output.ui
                                        && !matches!(output.data, OutputData::DraggablePoint(_))
                                    {
                                        output.ui = OutputUi::field_from_latex(&nodes);
                                    }
                                    if let OutputData::None = output.data {
                                        output.data = OutputData::Geometry(geometry);
                                    }
                                }
                            }
                        }
                    }

                    self.vm_vars = vm.vars;

                    let mut has_error = false;
                    for (i, e) in self.expressions.iter().enumerate() {
                        if let OutputData::Error(e) = &e.output.data {
                            println!("expression {} {e}", i + 1);
                            has_error = true;
                        }
                    }
                    if has_error {
                        println!();
                    }
                }
            }
        }

        let mut geometry = None;

        if redraw_geometry {
            let mut regular_geometry = vec![];
            let mut draggable_points = vec![];
            let mut focussed_geometry = vec![];

            for e in &self.expressions {
                match &e.output.data {
                    OutputData::DraggablePoint(p) => {
                        let mut p = p.clone();
                        if e.has_focus() {
                            p.width *= 1.15;
                            draggable_points.push(p);
                        } else {
                            draggable_points.push(p);
                        }
                    }
                    OutputData::Geometry(geometry) => {
                        if e.has_focus() {
                            for mut g in geometry.iter().cloned() {
                                g.width *= match g.kind {
                                    Line(_) | Plot { .. } => 1.4,
                                    Point { .. } => 1.2,
                                };
                                focussed_geometry.push(g);
                            }
                        } else {
                            regular_geometry.extend_from_slice(geometry);
                        }
                    }
                    _ => {}
                }
            }

            regular_geometry.append(&mut draggable_points);
            regular_geometry.append(&mut focussed_geometry);

            geometry = Some((regular_geometry, self.vm_vars.clone()));
        }

        self.expressions_changed = false;

        if response.requested_redraw {
            // If something wanted a redraw then some heights probably got
            // altered so it would be good to reclamp the scroll
            self.scroll(0.0);
        }

        (response, geometry)
    }

    pub fn render(
        &mut self,
        ctx: &Context,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        config: &wgpu::SurfaceConfiguration,
        encoder: &mut wgpu::CommandEncoder,
        bounds: Bounds,
    ) {
        let mut indices = vec![];
        let mut vertices = vec![];
        let draw_quad = &mut |p0: DVec2, p1: DVec2, kind: QuadKind| {
            let p0 = p0.as_vec2();
            let p1 = p1.as_vec2();
            let (uv0, uv1) = match kind {
                QuadKind::MsdfGlyph(uv0, uv1)
                | QuadKind::TranslucentMsdfGlyph(uv0, uv1)
                | QuadKind::PlaceholderMsdfGlyph(uv0, uv1) => (uv0, uv1),
                _ => (DVec2::splat(0.0), DVec2::splat(1.0)),
            };
            let kind = kind.index();
            let uv0 = uv0
                .map(|x| (x.clamp(0.0, 1.0) * 65535.0).round())
                .as_u16vec2();
            let uv1 = uv1
                .map(|x| (x.clamp(0.0, 1.0) * 65535.0).round())
                .as_u16vec2();

            indices.push(vertices.len() as u32);
            indices.push(vertices.len() as u32 + 1);
            indices.push(vertices.len() as u32 + 2);
            indices.push(vertices.len() as u32 + 3);
            indices.push(0xffffffff);

            vertices.push(Vertex {
                position: p0,
                uv: uv0,
                kind,
            });
            vertices.push(Vertex {
                position: vec2(p1.x, p0.y),
                uv: u16vec2(uv1.x, uv0.y),
                kind,
            });
            vertices.push(Vertex {
                position: vec2(p0.x, p1.y),
                uv: u16vec2(uv0.x, uv1.y),
                kind,
            });
            vertices.push(Vertex {
                position: p1,
                uv: uv1,
                kind,
            });
        };
        let mut next_y = bounds.pos.y - self.scroll;
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
        let expression_width = bounds.size.x - separator_width;

        for expression in &mut self.expressions {
            let height = expression.render(
                ctx,
                dvec2(bounds.pos.x, next_y),
                expression_width,
                draw_quad,
            );
            next_y += height;
            let p0 = dvec2(bounds.pos.x, next_y);
            let p1 = p0 + dvec2(bounds.size.x, separator_width);
            draw_quad(
                ctx.scale_factor * p0,
                ctx.scale_factor * p1,
                QuadKind::GrayBox,
            );
            next_y += separator_width;
        }

        {
            let p0 = dvec2(bounds.right() - separator_width, bounds.top());
            let p1 = dvec2(bounds.right(), bounds.bottom());
            draw_quad(
                ctx.scale_factor * p0,
                ctx.scale_factor * p1,
                QuadKind::GrayBox,
            );
        }

        let indices_size = size_of_val(&indices[..]) as u64;
        if indices_size > self.index_buffer.size() {
            self.index_buffer = create_index_buffer(device, indices_size);
        }

        let vertices_size = size_of_val(&vertices[..]) as u64;
        if vertices_size > self.vertex_buffer.size() {
            self.vertex_buffer = create_vertex_buffer(device, vertices_size);
        }

        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                resolution: uvec2(config.width, config.height).as_vec2(),
                scale_factor: ctx.scale_factor as f32,
            }]),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("latex"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        ctx.set_scissor_rect(&mut pass, bounds);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw_indexed(0..indices.len() as _, 0, 0..1);
    }
}
