use ambavia::{
    ast_parser::parse_expression_list_entry,
    compiler::compile_assignments,
    latex_tree,
    name_resolver::{ExpressionIndex, resolve_names},
    type_checker::{Type, type_check},
    vm::Vm,
};
use bytemuck::{Zeroable, offset_of};
use derive_more::{From, Into};
use glam::{DVec2, U16Vec2, Vec2, dvec2, u16vec2, uvec2, vec2};
use typed_index_collections::{TiVec, ti_vec};
use winit::{
    event::{ElementState, MouseButton},
    window::CursorIcon,
};

use crate::{
    math_field::{Cursor, Interactiveness, MathField, Message, UserSelection},
    ui::{Bounds, Context, Event, QuadKind, Response},
};

#[derive(Default)]
struct Expression {
    field: MathField,
    ast: Option<Result<ambavia::ast::ExpressionListEntry, String>>,
    output: Option<Result<MathField, String>>,
}

impl Expression {
    const PADDING: f64 = 16.0;
    const OUTPUT_LEFT_PADDING: f64 = 4.0;
    const OUTPUT_RIGHT_PADDING: f64 = 8.0;
    const OUTPUT_TOP_PADDING: f64 = 4.5;
    const OUTPUT_BOTTOM_PADDING: f64 = 3.5;

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
        let output_bounds = if let Some(Ok(output)) = &self.output {
            height += 0.5 * padding;
            let size = output.expression_size().map(|s| ctx.ceil(s));
            let right = top_left.x + width - 0.5 * padding;
            let left = (right - size.x).max(top_left.x + padding);
            let bounds = Bounds {
                pos: dvec2(left, top_left.y + height),
                size: dvec2(right - left, size.y),
            };
            height += bounds.size.y;
            height += 0.5 * padding;
            bounds
        } else {
            height += padding;
            Bounds::default()
        };
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
            message = m;
            r
        }));

        if let Some(Ok(output)) = &mut self.output {
            let mut r = output.update(ctx, event, output_bounds).0;

            if let Some(UserSelection { anchor, focus }) = output.get_selection() {
                let mut clamp = |mut cursor: Cursor| {
                    let index = cursor
                        .path
                        .first_mut()
                        .map_or(&mut cursor.index, |(index, _)| index);
                    if *index == 0 {
                        *index = 1;
                        r.requested_redraw = true;
                    }
                    cursor
                };
                output.set_selection((clamp(anchor.clone()), clamp(focus.clone())));
            }

            response = response.or(r);
        }

        (response, message, height)
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

        let output_bounds = if let Some(Ok(output)) = &self.output {
            height += 0.5 * padding;
            let size = output.expression_size().map(|s| ctx.ceil(s));
            let right = top_left.x + width - 0.5 * padding;
            let left = (right - size.x).max(top_left.x + padding);
            let bounds = Bounds {
                pos: dvec2(left, top_left.y + height),
                size: dvec2(right - left, size.y),
            };
            height += bounds.size.y;
            height += 0.5 * padding;
            bounds
        } else {
            height += padding;
            Bounds::default()
        };

        self.field.render(ctx, field_bounds, draw_quad);

        if let Some(Ok(output)) = &mut self.output {
            draw_quad(
                ctx.scale_factor * output_bounds.pos,
                ctx.scale_factor * (output_bounds.pos + output_bounds.size),
                QuadKind::RoundedBox,
            );
            output.render(ctx, output_bounds, draw_quad);
        }

        height
    }
}

pub struct ExpressionList {
    expressions: Vec<Expression>,

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

        Self {
            expressions: vec![Default::default()],
            pipeline,
            vertex_buffer,
            index_buffer,
            uniforms_buffer,
            bind_group,
        }
    }

    const SEPARATOR_WIDTH: f64 = 1.0;

    pub fn update(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> Response {
        let mut response = Response::default();
        let mut next_y = bounds.pos.y;
        let mut message = None;
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
        let expression_width = bounds.size.x - separator_width;

        for (i, expression) in self.expressions.iter_mut().enumerate() {
            let (r, m, height) =
                expression.update(ctx, event, dvec2(bounds.pos.x, next_y), expression_width);
            next_y += height;
            response = response.or(r);
            message = message.or(m.map(|m| (i, m)));
            next_y += separator_width;
        }

        let mut needs_reevaluation = false;

        if let Some((i, m)) = message {
            match m {
                Message::ContentsChanged => {
                    let e = &mut self.expressions[i];
                    let latex = e.field.to_latex();
                    e.ast = latex
                        .iter()
                        .any(|n| n != &latex_tree::Node::Char(' '))
                        .then(|| parse_expression_list_entry(&latex));
                    needs_reevaluation = true;
                }
                Message::Up => {
                    if i > 0 {
                        self.expressions[i].unfocus();
                        self.expressions[i - 1].focus();
                        response.request_redraw();
                    }
                }
                Message::Down => {
                    if i == self.expressions.len() - 1 {
                        self.expressions.push(Default::default());
                    }
                    self.expressions[i].unfocus();
                    self.expressions[i + 1].focus();
                    response.request_redraw();
                }
                Message::Add => {
                    self.expressions.insert(i + 1, Default::default());
                    self.expressions[i].unfocus();
                    self.expressions[i + 1].focus();
                    response.request_redraw();
                }
                Message::Remove => {
                    self.expressions.remove(i);
                    needs_reevaluation = true;
                    if self.expressions.is_empty() {
                        self.expressions.push(Default::default());
                    }
                    self.expressions[i.saturating_sub(1)].focus();
                    response.request_redraw();
                }
            }
        }

        if self.expressions.last().unwrap().has_focus() {
            self.expressions.push(Default::default());
            response.request_redraw();
        }

        if needs_reevaluation {
            #[derive(From, Into, Clone, Copy)]
            struct OutputIndex(usize);

            let mut ei_to_oi: TiVec<ExpressionIndex, OutputIndex> = ti_vec![];
            let mut list: TiVec<ExpressionIndex, _> = ti_vec![];

            for (i, e) in self.expressions.iter_mut().enumerate() {
                let i = OutputIndex(i);
                e.output = None;
                list.push(match &e.ast {
                    Some(Ok(ast)) => ast,
                    Some(Err(err)) => {
                        e.output = Some(Err(format!("parse error: {err}")));
                        continue;
                    }
                    None => continue,
                });
                ei_to_oi.push(i);
            }

            let (assignments, ei_to_nr) = resolve_names(list.as_slice());
            let (assignments, nr_to_tc) = type_check(&assignments);
            let (program, vars) = compile_assignments(&assignments);
            let mut vm = Vm::with_program(program);
            vm.run(false);

            for (ei, nr) in ei_to_nr.into_iter_enumerated() {
                let i: OutputIndex = ei_to_oi[ei];
                let output = &mut self.expressions[usize::from(i)].output;
                if output.is_none() {
                    *output = match nr {
                        Some(Ok(nr)) => match nr_to_tc[nr].clone() {
                            Ok(tc) => {
                                use latex_tree::Node::{self, Char as C};
                                let ty = assignments[tc].value.ty;
                                let v = vars[tc];
                                let mut nodes = vec![C('=')];

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
                                let point = |nodes: &mut Vec<Node>, x: f64, y: f64| {
                                    let mut point = vec![C('(')];
                                    number(&mut point, x);
                                    point.push(C(','));
                                    number(&mut point, y);
                                    point.push(C(')'));
                                    nodes.push(Node::DelimitedGroup(point));
                                };

                                match ty {
                                    Type::Number => number(&mut nodes, vm.vars[v].clone().number()),
                                    Type::NumberList => {
                                        let a = vm.vars[v].clone().list();
                                        let mut list = vec![C('[')];
                                        for (i, x) in a.borrow().as_slice().iter().enumerate() {
                                            if i > 0 {
                                                list.push(C(','));
                                            }
                                            number(&mut list, *x);
                                        }
                                        list.push(C(']'));
                                        nodes.push(Node::DelimitedGroup(list));
                                    }
                                    Type::Point => point(
                                        &mut nodes,
                                        vm.vars[v].clone().number(),
                                        vm.vars[v + 1].clone().number(),
                                    ),
                                    Type::PointList => {
                                        let a = vm.vars[v].clone().list();
                                        let mut list = vec![C('[')];
                                        for (i, p) in a.borrow().chunks(2).enumerate() {
                                            if i > 0 {
                                                list.push(C(','));
                                            }
                                            point(&mut list, p[0], p[1]);
                                        }
                                        list.push(C(']'));
                                        nodes.push(Node::DelimitedGroup(list));
                                    }
                                    Type::Bool | Type::BoolList => unreachable!(),
                                    Type::EmptyList => {
                                        nodes.push(Node::DelimitedGroup(vec![C('['), C(']')]))
                                    }
                                }

                                let mut field = MathField::from(&nodes);
                                field.interactiveness = Interactiveness::Select;
                                field.scale = 18.0;
                                field.left_padding =
                                    ctx.round(Expression::OUTPUT_LEFT_PADDING) / field.scale;
                                field.right_padding =
                                    ctx.round(Expression::OUTPUT_RIGHT_PADDING) / field.scale;
                                field.bottom_padding =
                                    ctx.round(Expression::OUTPUT_BOTTOM_PADDING) / field.scale;
                                field.top_padding =
                                    ctx.round(Expression::OUTPUT_TOP_PADDING) / field.scale;
                                Some(Ok(field))
                            }
                            Err(e) => Some(Err(format!("type error: {e}"))),
                        },
                        Some(Err(e)) => Some(Err(format!("name error: {e}"))),
                        None => None,
                    }
                }
            }

            let mut has_error = false;
            for (i, e) in self.expressions.iter().enumerate() {
                if let Some(Err(e)) = &e.output {
                    println!("expression {} {e}", i + 1);
                    has_error = true;
                }
            }
            if has_error {
                println!();
            }
        }

        response
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
                QuadKind::MsdfGlyph(uv0, uv1) | QuadKind::TranslucentMsdfGlyph(uv0, uv1) => {
                    (uv0, uv1)
                }
                QuadKind::BlackBox
                | QuadKind::TranslucentBlackBox
                | QuadKind::HighlightBox
                | QuadKind::GrayBox
                | QuadKind::TransparentToWhiteGradient
                | QuadKind::RoundedBox => (DVec2::splat(0.0), DVec2::splat(1.0)),
            };
            let kind = kind.index();
            let uv0 = uv0
                .map(|x| (x.clamp(0.0, 1.0) * 65535.0).round())
                .as_u16vec2();
            let uv1 = uv1
                .map(|x| (x.clamp(0.0, 1.0) * 65535.0).round())
                .as_u16vec2();

            indices.push(vertices.len() as u32 + 0);
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
        let mut next_y = bounds.pos.y;
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
