use bytemuck::offset_of;
use glam::{DVec2, DVec4, Vec2, dvec2, uvec2, vec2};

use crate::{
    AppGraphics,
    ui::{Color, Context},
    utility::mix,
};

#[derive(Debug, Clone, Copy)]
pub enum QuadKind {
    Rectangle,
    Pill,
    MsdfGlyph,
    AlphaGradientU,
    AlphaGradientU2,
    AlphaGradientV2,
    OutputValueBox,
    SliderPausedButton,
    SliderPlayingButton,
}

pub struct Quad {
    pub kind: QuadKind,
    pub p0: DVec2,
    pub p1: DVec2,
    pub uv0: DVec2,
    pub uv1: DVec2,
    pub color: DVec4,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            kind: QuadKind::Rectangle,
            p0: DVec2::ZERO,
            p1: DVec2::ZERO,
            uv0: DVec2::ZERO,
            uv1: DVec2::ONE,
            color: DVec4::ZERO,
        }
    }
}

impl Quad {
    pub fn rectangle(p0: impl Into<DVec2>, p1: impl Into<DVec2>, color: impl Color) -> Quad {
        Quad {
            kind: QuadKind::Rectangle,
            p0: p0.into(),
            p1: p1.into(),
            color: color.to_rgbaf64(),
            ..Default::default()
        }
    }

    pub fn pill(p0: impl Into<DVec2>, p1: impl Into<DVec2>, color: impl Color) -> Quad {
        Quad {
            kind: QuadKind::Pill,
            p0: p0.into(),
            p1: p1.into(),
            color: color.to_rgbaf64(),
            ..Default::default()
        }
    }

    pub fn pixel_snap(self, ctx: &Context) -> Quad {
        let f = |a, b| if a < b { ctx.floor(a) } else { ctx.ceil(a) };
        let p0 = dvec2(f(self.p0.x, self.p1.x), f(self.p0.y, self.p1.y));
        let p1 = dvec2(f(self.p1.x, self.p0.x), f(self.p1.y, self.p0.y));
        Quad {
            p0,
            p1,
            uv0: mix(self.uv0, self.uv1, (p0 - self.p0) / (self.p1 - self.p0)),
            uv1: mix(self.uv0, self.uv1, (p1 - self.p0) / (self.p1 - self.p0)),
            ..self
        }
    }

    pub fn into_triangles(self, ctx: &Context, vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>) {
        let kind = self.kind as u32;
        let p0 = (ctx.scale_factor * self.p0).as_vec2();
        let p1 = (ctx.scale_factor * self.p1).as_vec2();
        let to_unorm = |x: f64, s: f64| (x.clamp(0.0, 1.0) * s).round();
        let uv0 = self.uv0.as_vec2();
        let uv1 = self.uv1.as_vec2();
        let color = self.color.to_array().map(|x| to_unorm(x, 255.0) as u8);

        indices.push(vertices.len() as u32);
        indices.push(vertices.len() as u32 + 1);
        indices.push(vertices.len() as u32 + 2);
        indices.push(vertices.len() as u32 + 3);
        indices.push(0xffffffff);

        vertices.push(Vertex {
            position: p0,
            color,
            kind,
            uv: uv0,
        });
        vertices.push(Vertex {
            position: vec2(p1.x, p0.y),
            color,
            kind,
            uv: vec2(uv1.x, uv0.y),
        });
        vertices.push(Vertex {
            position: vec2(p0.x, p1.y),
            color,
            kind,
            uv: vec2(uv0.x, uv1.y),
        });
        vertices.push(Vertex {
            position: p1,
            color,
            kind,
            uv: uv1,
        });
    }
}

pub struct QuadRenderer {
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

#[derive(Default, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec2,
    pub color: [u8; 4],
    pub kind: u32,
    pub uv: Vec2,
}

fn create_index_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quad_index_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDEX,
        mapped_at_creation: false,
    })
}

fn create_vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quad_vertex_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

impl QuadRenderer {
    pub fn new(
        AppGraphics {
            device,
            queue,
            config,
            ..
        }: &AppGraphics,
    ) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("quad.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quad_bind_group_layout"),
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
            label: Some("quad"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("quad_pipeline_layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_quad"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as _,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: offset_of!(Vertex, position) as _,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm8x4,
                            offset: offset_of!(Vertex, color) as _,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: offset_of!(Vertex, kind) as _,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: offset_of!(Vertex, uv) as _,
                            shader_location: 3,
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
                entry_point: Some("fs_quad"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            cache: None,
            multiview_mask: None,
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
            label: Some("quad_bind_group"),
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
            pipeline,
            vertex_buffer,
            index_buffer,
            uniforms_buffer,
            bind_group,
        }
    }

    pub fn render(
        &mut self,
        ctx: &Context,
        AppGraphics {
            device,
            queue,
            config,
            ..
        }: &AppGraphics,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        vertices: &[Vertex],
        indices: &[u32],
    ) {
        let indices_size = size_of_val(indices) as u64;
        if indices_size > self.index_buffer.size() {
            self.index_buffer = create_index_buffer(device, indices_size);
        }

        let vertices_size = size_of_val(vertices) as u64;
        if vertices_size > self.vertex_buffer.size() {
            self.vertex_buffer = create_vertex_buffer(device, vertices_size);
        }

        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                resolution: uvec2(config.width, config.height).as_vec2(),
                scale_factor: ctx.scale_factor as f32,
            }]),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("quad"),
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
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw_indexed(0..indices.len() as _, 0, 0..1);
    }
}
