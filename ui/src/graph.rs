mod sample_explicit;
mod sample_implicit;
mod tile_fill;

use std::{
    collections::HashMap,
    iter::zip,
    time::{Duration, Instant},
};

use bytemuck::Zeroable;
use eval::vm::{self, Instruction, VarIndex, Vm};
use glam::{DVec2, Vec2, dvec2, uvec2};
use parse::analyze_expression_list::PlotKind;
use winit::{
    event::{ElementState, MouseButton},
    window::CursorIcon,
};

use crate::{
    Bounds, Context, Event, Response,
    expression_list::ExpressionId,
    graph::{
        sample_explicit::sample_explicit,
        sample_implicit::sample_implicit,
        tile_fill::{Segment, TILE_SIZE, Tile},
    },
    ui::CursorMode,
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

#[derive(Debug, Clone)]
pub enum GeometryKind {
    Line(Vec<DVec2>),
    Point {
        p: DVec2,
        draggable: Option<ExpressionId>,
    },
    Fill(Vec<DVec2>),
    Plot {
        kind: PlotKind<f64>,
        inputs: Vec<VarIndex>,
        output: VarIndex,
        instructions: Vec<Instruction>,
    },
}

#[derive(Debug, Clone)]
pub struct Geometry {
    pub width: f32,
    pub color: [f32; 4],
    pub kind: GeometryKind,
}

pub struct GraphPaper {
    viewport: Viewport,
    dragging: Option<Option<ExpressionId>>,
    hovered_point: Option<ExpressionId>,
    geometry: Vec<Geometry>,
    vm_vars: vm::Vars,

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
    segments_capacity: usize,
    segments_buffer: wgpu::Buffer,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Uniforms {
    resolution: Vec2,
    tile_size: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Shape {
    color: [f32; 4],
    width: f32,
    kind: u32,
    tile: Tile,
    padding: [u32; 3],
}

impl Shape {
    const LINE: u32 = 0;
    const POINT: u32 = 1;
    const RECTANGLE: u32 = 2;
    const TILE: u32 = 3;

    fn line(color: [f32; 4], width: f32) -> Self {
        Self {
            color,
            width,
            kind: Shape::LINE,
            ..Shape::zeroed()
        }
    }

    fn point(color: [f32; 4], width: f32) -> Self {
        Self {
            color,
            width,
            kind: Shape::POINT,
            ..Shape::zeroed()
        }
    }

    fn rectangle(color: [f32; 4], width: f32) -> Self {
        Self {
            color,
            width,
            kind: Shape::RECTANGLE,
            ..Shape::zeroed()
        }
    }

    fn tile(color: [f32; 4], tile: Tile) -> Self {
        Self {
            color,
            kind: Shape::TILE,
            tile,
            ..Shape::zeroed()
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

// Must be 1 or 4
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
        format,
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

fn buffer_with_capacity<T>(device: &wgpu::Device, label: &str, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (size_of::<T>() * capacity) as _,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn shapes_buffer_with_capacity(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    buffer_with_capacity::<Shape>(device, "shapes", capacity)
}

fn vertices_buffer_with_capacity(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    buffer_with_capacity::<Vertex>(device, "vertices", capacity)
}

fn segments_buffer_with_capacity(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    buffer_with_capacity::<Segment>(device, "segments", capacity)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms_buffer: &wgpu::Buffer,
    shapes_buffer: &wgpu::Buffer,
    vertices_buffer: &wgpu::Buffer,
    segments_buffer: &wgpu::Buffer,
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(segments_buffer.as_entire_buffer_binding()),
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
        segments: &[Segment],
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

        if segments.len() > self.segments_capacity {
            new_buffers = true;
            self.segments_capacity = grow(self.segments_capacity, segments.len());
            self.segments_buffer = segments_buffer_with_capacity(device, self.segments_capacity);
        }

        if new_buffers {
            self.bind_group = create_bind_group(
                device,
                &self.layout,
                &self.uniforms_buffer,
                &self.shapes_buffer,
                &self.vertices_buffer,
                &self.segments_buffer,
            )
        }

        queue.write_buffer(&self.shapes_buffer, 0, bytemuck::cast_slice(shapes));
        queue.write_buffer(&self.vertices_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.segments_buffer, 0, bytemuck::cast_slice(segments));
    }

    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> GraphPaper {
        let module = device.create_shader_module(wgpu::include_wgsl!("graph.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph_bind_group_layout"),
            entries: &[
                // uniforms
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
                // shapes
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
                // vertices
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
                // segments
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
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
            multiview_mask: None,
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
        let segments_capacity = 1;
        let segments_buffer = segments_buffer_with_capacity(device, segments_capacity);
        let bind_group = create_bind_group(
            device,
            &layout,
            &uniforms_buffer,
            &shapes_buffer,
            &vertices_buffer,
            &segments_buffer,
        );
        GraphPaper {
            viewport: Default::default(),
            dragging: None,
            hovered_point: None,
            geometry: vec![],
            vm_vars: Default::default(),

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
            segments_capacity,
            segments_buffer,
        }
    }

    pub fn set_geometry(&mut self, geometry: Vec<Geometry>, vm_vars: vm::Vars) {
        self.geometry = geometry;
        self.vm_vars = vm_vars;
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
            for g in self.geometry.iter().rev() {
                if let GeometryKind::Point {
                    p,
                    draggable: Some(i),
                    ..
                } = g.kind
                    && from_vp(&self.viewport, p).distance(ctx.cursor)
                        < draggable_point_width(g.width) as f64 / 2.0
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
                        if let Some(p) = self.geometry.iter().find_map(|g| {
                            if let GeometryKind::Point { p, draggable } = g.kind
                                && draggable == Some(i)
                            {
                                Some(p)
                            } else {
                                None
                            }
                        }) {
                            dragged_point = Some((i, p + diff));
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
            response.cursor_mode = CursorMode::Icon(CursorIcon::AllScroll);
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

        let (shapes, vertices, segments) = self.generate_geometry(ctx, bounds);

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
                tile_size: TILE_SIZE,
            }]),
        );
        self.write(device, queue, &shapes, &vertices, &segments);
        let graph_texture_view = self.graph_texture.create_view(&Default::default());

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("graph_paper"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: if MSAA_SAMPLE_COUNT > 1 {
                    &graph_texture_view
                } else {
                    view
                },
                depth_slice: None,
                resolve_target: if MSAA_SAMPLE_COUNT > 1 {
                    Some(view)
                } else {
                    None
                },
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
        pass.draw(0..vertices.len() as u32 * 6, 0..1);
    }

    fn generate_geometry(
        &mut self,
        ctx: &Context,
        bounds: Bounds,
    ) -> (Vec<Shape>, Vec<Vertex>, Vec<Segment>) {
        let mut shapes = vec![];
        let mut vertices = vec![];
        let mut segments = vec![];
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

        for Geometry { width, color, kind } in &self.geometry {
            match kind {
                GeometryKind::Plot {
                    kind,
                    inputs,
                    output,
                    instructions,
                } => {
                    let shape = shapes.len() as u32;
                    shapes.push(Shape::line(*color, ctx.scale_factor as f32 * width));

                    let mut vm = Vm::new(instructions, std::mem::take(&mut self.vm_vars));
                    let pixels_per_math = vp.width / physical.size.x;

                    let buffer = 0.5 * ctx.scale_factor * *width as f64 * pixels_per_math;
                    let vp_min = vp.center - vp_size * 0.5 - buffer;
                    let vp_max = vp.center + vp_size * 0.5 + buffer;
                    let tolerance = 1.0 * pixels_per_math;

                    const TRACK_STATS: bool = false;
                    const CACHE_IMPLICIT_EVALUATIONS: bool = true;

                    let mut f_eval_count = 0;
                    let mut cache_hits = 0;
                    let mut f_elapsed = Duration::ZERO;
                    let mut run =
                        |vm: &mut Vm, input_indices: &[VarIndex], input_values: &[f64]| {
                            if TRACK_STATS {
                                f_eval_count += 1;
                            }
                            for (index, value) in zip(input_indices, input_values) {
                                // Sometimes the input is optimized out of the program (e.g., f(x)=2)
                                // so we need to check if it actually exists first
                                if let Some(input) = vm.vars.get_mut(*index) {
                                    *input = vm::Value::Number(*value);
                                }
                            }
                            if TRACK_STATS {
                                let start = Instant::now();
                                vm.run(false);
                                f_elapsed += start.elapsed();
                            } else {
                                vm.run(false);
                            }
                        };

                    let start = Instant::now();
                    let points = if *kind == PlotKind::Implicit {
                        let mut cache = HashMap::new();
                        sample_implicit(
                            |p| {
                                if CACHE_IMPLICIT_EVALUATIONS {
                                    let key = [p.x.to_bits(), p.y.to_bits()];
                                    if let Some(f) = cache.get(&key) {
                                        if TRACK_STATS {
                                            cache_hits += 1;
                                        }
                                        *f
                                    } else {
                                        run(&mut vm, inputs, &[p.x, p.y]);
                                        let f = vm.vars[*output].clone().number();
                                        cache.insert(key, f);
                                        f
                                    }
                                } else {
                                    run(&mut vm, inputs, &[p.x, p.y]);
                                    vm.vars[*output].clone().number()
                                }
                            },
                            vp_min,
                            vp_max,
                        )
                    } else {
                        let n_uniform_samples = match kind {
                            // Desmos seems to do 4 per physical pixel
                            PlotKind::Normal => (physical.size.x * 4.0) as usize,
                            PlotKind::Inverse => (physical.size.y * 4.0) as usize,
                            // Desmos seems to do 2000
                            PlotKind::Parametric(_) => 2000,
                            PlotKind::Implicit => unreachable!(),
                        };

                        match kind {
                            PlotKind::Normal => {
                                let f = |x: f64| {
                                    run(&mut vm, inputs, &[x]);
                                    let y = vm.vars[*output].clone().number();
                                    dvec2(x, y)
                                };
                                sample_explicit(
                                    f,
                                    vp_min.x,
                                    vp_max.x,
                                    vp_min,
                                    vp_max,
                                    tolerance,
                                    n_uniform_samples,
                                )
                            }
                            PlotKind::Inverse => {
                                let f = |y: f64| {
                                    run(&mut vm, inputs, &[y]);
                                    let x = vm.vars[*output].clone().number();
                                    dvec2(x, y)
                                };
                                sample_explicit(
                                    f,
                                    vp_min.y,
                                    vp_max.y,
                                    vp_min,
                                    vp_max,
                                    tolerance,
                                    n_uniform_samples,
                                )
                            }
                            PlotKind::Parametric(t) => {
                                let f = |t: f64| {
                                    run(&mut vm, inputs, &[t]);
                                    let x = vm.vars[*output].clone().number();
                                    let y = vm.vars[*output + 1.into()].clone().number();
                                    dvec2(x, y)
                                };
                                sample_explicit(
                                    f,
                                    t.min,
                                    t.max,
                                    vp_min,
                                    vp_max,
                                    tolerance,
                                    n_uniform_samples,
                                )
                            }
                            PlotKind::Implicit => unreachable!(),
                        }
                    };
                    let elapsed = start.elapsed();

                    if TRACK_STATS {
                        println!();
                        println!("points.len() = {}", points.len());
                        println!("cache hits   = {}", cache_hits);
                        println!("f eval count = {}", f_eval_count);
                        println!("f eval time  = {:?}", f_elapsed);
                        println!("total time   = {:?}", elapsed);
                    }

                    for p in points {
                        let p = to_physical(p).as_vec2();
                        vertices.push(Vertex::new(p, shape));
                    }

                    self.vm_vars = vm.vars;
                }
                GeometryKind::Line(points) => {
                    let shape = shapes.len() as u32;
                    shapes.push(Shape::line(*color, ctx.scale_factor as f32 * width));
                    for p in points {
                        let p = to_physical(*p).as_vec2();
                        vertices.push(Vertex::new(p, shape));
                    }
                }
                GeometryKind::Point { p, draggable } => {
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
                GeometryKind::Fill(points) => {
                    tile_fill::tile_fill(
                        physical,
                        &points.iter().cloned().map(to_physical).collect::<Vec<_>>(),
                        &mut segments,
                        |position, item| {
                            let shape = shapes.len() as u32;
                            shapes.push(match item {
                                tile_fill::Item::Rectangle { width } => {
                                    Shape::rectangle(*color, width)
                                }
                                tile_fill::Item::Tile(tile) => Shape::tile(*color, tile),
                            });
                            vertices.push(Vertex::new(position, shape));
                        },
                    );
                }
            }
        }

        // The vertex shader will check an extra vertex when drawing lines, so
        // we push this to avoid an out-of-bounds access in the shader
        vertices.push(Vertex::BREAK);

        (shapes, vertices, segments)
    }
}
