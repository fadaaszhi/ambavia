use glam::{DVec2, Vec2, dvec2, uvec2};
use winit::{
    event::{ElementState, MouseButton},
    window::CursorIcon,
};

use crate::{
    Bounds, Context, Event, Response,
    expression_list::ExpressionId,
    utility::{flip_y, snap},
};

struct Viewport {
    center: DVec2,
    width: f64,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            center: DVec2::ZERO,
            width: 20.0,
        }
    }
}

#[derive(Clone)]
pub struct Point {
    pub p: DVec2,
    pub width: f32,
    pub color: [f32; 4],
    pub draggable: Option<ExpressionId>,
}

pub struct GraphPaper {
    viewport: Viewport,
    dragging: Option<Option<ExpressionId>>,
    hovered_point: Option<ExpressionId>,
    points: Vec<Point>,

    graph_texture: wgpu::Texture,
    depth_texture: wgpu::Texture,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniforms_buffer: wgpu::Buffer,
    shapes_capacity: usize,
    shapes_buffer: wgpu::Buffer,
    vertices_capacity: usize,
    vertices_buffer: wgpu::Buffer,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Uniforms {
    resolution: Vec2,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Shape {
    color: [f32; 4],
    width: f32,
    kind: u32,
    padding: [u32; 2],
}

impl Shape {
    const LINE: u32 = 0;
    const POINT: u32 = 1;

    fn line(color: [f32; 4], width: f32) -> Self {
        Self {
            color,
            width,
            kind: Shape::LINE,
            padding: [0; 2],
        }
    }

    fn point(color: [f32; 4], width: f32) -> Self {
        Self {
            color,
            width,
            kind: Shape::POINT,
            padding: [0; 2],
        }
    }
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Vertex {
    position: Vec2,
    shape: u32,
    padding: [u32; 1],
}

impl Vertex {
    const BREAK: Self = Self {
        position: Vec2::ZERO,
        shape: !0,
        padding: [0; 1],
    };

    fn new(position: impl Into<Vec2>, shape: u32) -> Self {
        Self {
            position: position.into(),
            shape,
            padding: [0; 1],
        }
    }
}

const MSAA_SAMPLE_COUNT: u32 = 4;

fn create_graph_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("graph_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("graph_depth_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn shapes_buffer_with_capacity(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shapes_buffer"),
        size: (size_of::<Shape>() * capacity) as _,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn vertices_buffer_with_capacity(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertices_buffer"),
        size: (size_of::<Vertex>() * capacity) as _,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms_buffer: &wgpu::Buffer,
    shapes_buffer: &wgpu::Buffer,
    vertices_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("graph_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(uniforms_buffer.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(shapes_buffer.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(vertices_buffer.as_entire_buffer_binding()),
            },
        ],
    })
}

fn draggable_point_width(width: f32) -> f32 {
    32f32.clamp(width, 2.0 * width) + width
}

impl GraphPaper {
    fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shapes: &[Shape],
        vertices: &[Vertex],
    ) {
        let mut new_buffers = false;
        let grow = |x, y: usize| y.max(x);

        if shapes.len() > self.shapes_capacity {
            new_buffers = true;
            self.shapes_capacity = grow(self.shapes_capacity, shapes.len());
            self.shapes_buffer = shapes_buffer_with_capacity(device, self.shapes_capacity);
        }

        if vertices.len() > self.vertices_capacity {
            new_buffers = true;
            self.vertices_capacity = grow(self.vertices_capacity, vertices.len());
            self.vertices_buffer = vertices_buffer_with_capacity(device, self.vertices_capacity);
        }

        if new_buffers {
            self.bind_group = create_bind_group(
                device,
                &self.layout,
                &self.uniforms_buffer,
                &self.shapes_buffer,
                &self.vertices_buffer,
            )
        }

        queue.write_buffer(&self.shapes_buffer, 0, bytemuck::cast_slice(shapes));
        queue.write_buffer(&self.vertices_buffer, 0, bytemuck::cast_slice(vertices));
    }

    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> GraphPaper {
        let module = device.create_shader_module(wgpu::include_wgsl!("graph.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph_bind_group_layout"),
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let graph_texture =
            create_graph_texture(device, config.width, config.height, config.format);
        let depth_texture = create_depth_texture(device, config.width, config.height);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("graph"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("graph_pipeline_layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_graph"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_texture.format(),
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_graph"),
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
        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms_buffer"),
            size: size_of::<Uniforms>().next_multiple_of(16) as _,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let shapes_capacity = 1;
        let shapes_buffer = shapes_buffer_with_capacity(device, shapes_capacity);
        let vertices_capacity = 1;
        let vertices_buffer = vertices_buffer_with_capacity(device, vertices_capacity);
        let bind_group = create_bind_group(
            device,
            &layout,
            &uniforms_buffer,
            &shapes_buffer,
            &vertices_buffer,
        );
        GraphPaper {
            viewport: Default::default(),
            dragging: None,
            hovered_point: None,
            points: vec![],

            graph_texture,
            depth_texture,
            pipeline,
            layout,
            bind_group,
            uniforms_buffer,
            shapes_capacity,
            shapes_buffer,
            vertices_capacity,
            vertices_buffer,
        }
    }

    pub fn set_geometry(&mut self, points: Vec<Point>) {
        self.points = points;
    }

    pub fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        bounds: Bounds,
    ) -> (Response, Option<(ExpressionId, DVec2)>) {
        let mut response = Response::default();

        let to_vp = |vp: &Viewport, p: DVec2| {
            flip_y(p - bounds.pos - 0.5 * bounds.size) / bounds.size.x * vp.width + vp.center
        };
        let from_vp = |vp: &Viewport, p: DVec2| {
            flip_y(p - vp.center) / vp.width * bounds.size.x + bounds.pos + 0.5 * bounds.size
        };
        let zoom = |vp: &mut Viewport, amount: f64| {
            let origin = from_vp(vp, DVec2::ZERO);
            let p = if amount > 1.0 && (ctx.cursor - origin).abs().max_element() < 25.0 {
                origin
            } else {
                ctx.cursor
            };
            let p_vp = to_vp(vp, p);
            vp.width /= amount;
            vp.center += p_vp - to_vp(vp, p);
        };
        let mut dragged_point = None;
        let new_hovered_point = self.dragging.unwrap_or_else(|| {
            for p in self.points.iter().rev() {
                if let Some(i) = p.draggable
                    && from_vp(&self.viewport, p.p).distance(ctx.cursor)
                        < draggable_point_width(p.width) as f64 / 2.0
                {
                    return Some(i);
                }
            }
            None
        });
        if new_hovered_point != self.hovered_point {
            self.hovered_point = new_hovered_point;
            response.request_redraw();
        }

        match event {
            Event::MouseInput(ElementState::Pressed, MouseButton::Left)
                if bounds.contains(ctx.cursor) =>
            {
                self.dragging = Some(self.hovered_point.or(None));
                response.consume_event();
            }
            Event::MouseInput(ElementState::Released, MouseButton::Left)
                if self.dragging.is_some() =>
            {
                self.dragging = None;
                response.consume_event();
            }
            Event::CursorMoved { previous_cursor } => {
                if let Some(point) = self.dragging {
                    let diff =
                        to_vp(&self.viewport, ctx.cursor) - to_vp(&self.viewport, *previous_cursor);

                    if let Some(i) = point {
                        if let Some(p) = self.points.iter().find(|p| p.draggable == Some(i)) {
                            dragged_point = Some((i, p.p + diff));
                        } else {
                            self.dragging = None;
                        }
                    } else {
                        self.viewport.center -= diff;
                    }
                    response.request_redraw();
                    response.consume_event();
                }
            }
            Event::MouseWheel(delta) if bounds.contains(ctx.cursor) => {
                zoom(&mut self.viewport, (delta.y * 0.0015).exp2());
                response.request_redraw();
                response.consume_event();
            }
            Event::PinchGesture(delta) if bounds.contains(ctx.cursor) => {
                zoom(&mut self.viewport, delta.exp());
                response.request_redraw();
                response.consume_event();
            }
            _ => {}
        }

        if self.hovered_point.is_some() {
            response.cursor_icon = CursorIcon::AllScroll;
        }

        (response, dragged_point)
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
        if bounds.is_empty() {
            return;
        }

        let (shapes, vertices) = self.generate_geometry(ctx, bounds);

        if vertices.is_empty() {
            return;
        }

        if self.graph_texture.width() != config.width
            || self.graph_texture.height() != config.height
        {
            self.graph_texture =
                create_graph_texture(device, config.width, config.height, config.format);
        }

        if self.depth_texture.width() != config.width
            || self.depth_texture.height() != config.height
        {
            self.depth_texture = create_depth_texture(device, config.width, config.height);
        }

        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                resolution: uvec2(config.width, config.height).as_vec2(),
            }]),
        );
        self.write(device, queue, &shapes, &vertices);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("graph_paper"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.graph_texture.create_view(&Default::default()),
                resolve_target: Some(view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.create_view(&Default::default()),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        ctx.set_scissor_rect(&mut pass, bounds);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        pass.draw(
            0..if shapes[vertices.last().unwrap().shape as usize].kind == Shape::LINE {
                vertices.len() as u32 - 1
            } else {
                vertices.len() as u32
            } * 6,
            0..1,
        );
    }

    fn generate_geometry(&self, ctx: &Context, bounds: Bounds) -> (Vec<Shape>, Vec<Vertex>) {
        let mut shapes = vec![];
        let mut vertices = vec![];
        let vp = &self.viewport;
        let vp_size = dvec2(vp.width, vp.width * bounds.size.y / bounds.size.x);
        let physical = ctx.to_physical(bounds);

        let s = vp.width / bounds.size.x * 80.0;
        let (mut major, mut minor) = (f64::INFINITY, 0.0);
        for (a, b) in [(1.0, 5.0), (2.0, 4.0), (5.0, 5.0)] {
            let c = a * 10f64.powf((s / a).log10().ceil());
            if c < major {
                major = c;
                minor = c / b;
            }
        }

        let mut draw_grid = |step: f64, color: [f32; 4], width: u32| {
            let shape = shapes.len() as u32;
            shapes.push(Shape::line(color, width as f32));
            let s = DVec2::splat(step);
            let a = (0.5 * vp_size / step).ceil();
            let n = 2 * a.as_uvec2() + 2;
            let b = flip_y(s / vp_size * physical.size);
            let c = (0.5 - flip_y(vp.center.rem_euclid(s) + a * s) / vp_size) * physical.size
                + physical.pos;

            for i in 0..n.x {
                let x = snap(i as f64 * b.x + c.x, width) as f32;
                vertices.push(Vertex::BREAK);
                vertices.push(Vertex::new((x, physical.top() as f32), shape));
                vertices.push(Vertex::new((x, physical.bottom() as f32), shape));
            }

            for i in 0..n.y {
                let y = snap(i as f64 * b.y + c.y, width) as f32;
                vertices.push(Vertex::BREAK);
                vertices.push(Vertex::new((physical.left() as f32, y), shape));
                vertices.push(Vertex::new((physical.right() as f32, y), shape));
            }
        };

        draw_grid(
            minor,
            [0.88, 0.88, 0.88, 1.0],
            ctx.round_nonzero_as_physical(1.0),
        );
        draw_grid(
            major,
            [0.6, 0.6, 0.6, 1.0],
            ctx.round_nonzero_as_physical(1.0),
        );

        let to_physical = |p: DVec2| {
            flip_y(p - vp.center) / vp.width * physical.size.x + 0.5 * physical.size + physical.pos
        };

        let w = ctx.round_nonzero_as_physical(1.5);
        let origin = to_physical(DVec2::ZERO).map(|x| snap(x, w)).as_vec2();
        let shape = shapes.len() as u32;
        shapes.push(Shape::line([0.098, 0.098, 0.098, 1.0], w as f32));
        vertices.push(Vertex::new((origin.x, physical.top() as f32), shape));
        vertices.push(Vertex::new((origin.x, physical.bottom() as f32), shape));
        vertices.push(Vertex::BREAK);
        vertices.push(Vertex::new((physical.left() as f32, origin.y), shape));
        vertices.push(Vertex::new((physical.right() as f32, origin.y), shape));

        for Point {
            p,
            width,
            color,
            draggable,
        } in &self.points
        {
            let p = to_physical(*p).as_vec2();
            let mut width = *width;

            if draggable.is_some() {
                let shape = shapes.len() as u32;
                let mut color = *color;
                color[3] *= 0.35;
                let draggable_width = draggable_point_width(width);
                shapes.push(Shape::point(
                    color,
                    ctx.scale_factor as f32 * draggable_width,
                ));
                vertices.push(Vertex::new(p, shape));

                if self.hovered_point == *draggable {
                    width = draggable_width;
                }
            }

            let shape = shapes.len() as u32;
            shapes.push(Shape::point(*color, ctx.scale_factor as f32 * width));
            vertices.push(Vertex::new(p, shape));
        }

        (shapes, vertices)
    }
}
