use ambavia::latex_parser::parse_latex;
use bytemuck::{Zeroable, offset_of};
use glam::{DVec2, U16Vec2, Vec2, dvec2, u16vec2, uvec2, vec2};

use crate::{
    math_field::{MathField, Message},
    ui::{Bounds, Context, Event, QuadKind, Response},
};

pub struct ExpressionList {
    expressions: Vec<MathField>,

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
                    visibility: wgpu::ShaderStages::VERTEX,
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

        let s = r"f\left(x,y,\beta_{1}\right)=\sum_{n=\prod_{n=4}^{5}}^{10}\prod_{n=\frac{\frac{3}{4}}{5}}^{\frac{23ojlkjaf}{dlfj}}ljk";
        // let s = r"f\left(x,y,\beta_{1}\right)=\sum_{n=\frac{\frac{1}{1}}{\frac{1}{1}}}^{10}abc";
        Self {
            expressions: vec![
                MathField::from(&parse_latex(s).unwrap()),
                Default::default(),
            ],
            pipeline,
            vertex_buffer,
            index_buffer,
            uniforms_buffer,
            bind_group,
        }
    }

    const SEPARATOR_WIDTH: f64 = 1.0;
    const PADDING: f64 = 16.0;

    pub fn update(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> Response {
        let mut response = Response::default();
        let mut y_offset = 0.0;
        let mut message = None;
        let padding = ctx.round(Self::PADDING);
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);

        for (i, expression) in self.expressions.iter_mut().enumerate() {
            let y_size = ctx.ceil(expression.expression_size().y);
            let (r, m) = expression.update(
                ctx,
                event,
                Bounds {
                    pos: bounds.pos + dvec2(0.0, y_offset) + padding,
                    size: dvec2(bounds.size.x - 1.5 * padding - separator_width, y_size),
                },
            );

            if let Some(m) = m {
                message = Some((i, m));
            }

            response = response.or(r);
            y_offset += y_size + 2.0 * padding + separator_width;
        }

        if let Some((i, m)) = message {
            match m {
                Message::Up => {
                    if i > 0 {
                        response = response
                            .or(self.expressions[i].unfocus())
                            .or(self.expressions[i - 1].focus());
                        response.request_redraw();
                    }
                }
                Message::Down => {
                    if i == self.expressions.len() - 1 {
                        self.expressions.push(MathField::default());
                    }

                    response = response
                        .or(self.expressions[i].unfocus())
                        .or(self.expressions[i + 1].focus());
                    response.request_redraw();
                }
                Message::Add => {
                    self.expressions.insert(i + 1, MathField::default());
                    response = response
                        .or(self.expressions[i].unfocus())
                        .or(self.expressions[i + 1].focus());
                    response.request_redraw();
                }
                Message::Remove => {
                    response = response.or(self.expressions[i].unfocus());
                    self.expressions.remove(i);

                    if self.expressions.is_empty() {
                        self.expressions.push(MathField::default());
                    }

                    response = response.or(self.expressions[i.saturating_sub(1)].focus());
                    response.request_redraw();
                }
            }
        }

        if self.expressions.last().unwrap().has_focus() {
            self.expressions.push(MathField::default());
            response.request_redraw();
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
                | QuadKind::TransparentToWhiteGradient => (DVec2::splat(0.0), DVec2::splat(1.0)),
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
        let mut y_offset = 0.0;
        let padding = ctx.round(Self::PADDING);
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);

        for expression in &mut self.expressions {
            let y_size = ctx.ceil(expression.expression_size().y);
            expression.render(
                ctx,
                Bounds {
                    pos: bounds.pos + dvec2(0.0, y_offset) + padding,
                    size: dvec2(bounds.size.x - 1.5 * padding - separator_width, y_size),
                },
                draw_quad,
            );
            y_offset += y_size + 2.0 * padding;

            let p0 = bounds.pos + dvec2(0.0, y_offset);
            let p1 = dvec2(bounds.right(), bounds.top() + y_offset + separator_width);
            draw_quad(
                ctx.scale_factor * p0,
                ctx.scale_factor * p1,
                QuadKind::GrayBox,
            );

            y_offset += separator_width;
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
