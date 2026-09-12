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
use glam::{DVec2, DVec4, Vec2, dvec2, uvec2};
use parse::analyze_expression_list::PlotKind;
use winit::{
    event::{ElementState, MouseButton},
    window::CursorIcon,
};

use crate::{
    AppGraphics, Bounds, Context, Event, Response,
    expression_list::ExpressionId,
    graph::{
        sample_explicit::sample_explicit,
        sample_implicit::sample_implicit,
        tile_fill::{Segment, TILE_SIZE, Tile},
    },
    quad_renderer::{Quad, QuadKind},
    ui::{AnimatedValue, Color, CursorMode},
    utility::{ClampToBounds, IfFiniteElse, flip_y, mix, set, snap},
};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Viewport {
    center: DVec2,
    width: f64,
    height: Option<f64>,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            center: DVec2::ZERO,
            width: 20.0,
            height: None,
        }
    }
}

impl Viewport {
    fn size(&self, bounds: Bounds) -> DVec2 {
        let h = self.width * (bounds.size.y / bounds.size.x).if_finite_else(1.0);
        dvec2(self.width, self.height.unwrap_or(h))
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

#[derive(Clone, Copy, PartialEq)]
enum Axis {
    X,
    Y,
}

impl Axis {
    fn get(self, p: DVec2) -> f64 {
        match self {
            Axis::X => p.x,
            Axis::Y => p.y,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum DragTarget {
    Point(ExpressionId),
    Axis {
        axis: Axis,
        inital_viewport: Viewport,
        initial_cursor: DVec2,
    },
    GraphPaper,
}

impl DragTarget {
    fn axis(&self) -> Option<Axis> {
        if let DragTarget::Axis { axis, .. } = self {
            Some(*axis)
        } else {
            None
        }
    }
}

pub struct GraphPaper {
    graph_buttons: GraphButtons,
    viewport: Viewport,
    dragging: Option<DragTarget>,
    hovered: DragTarget,
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
        let position = position.into();

        if position.is_finite() {
            Self {
                position,
                shape,
                padding: [0; 1],
            }
        } else {
            Self::BREAK
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

    pub fn new(AppGraphics { device, config, .. }: &AppGraphics) -> GraphPaper {
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
            graph_buttons: GraphButtons::new(),
            viewport: Default::default(),
            dragging: None,
            hovered: DragTarget::GraphPaper,
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
        let (mut response, buttons_hovered) =
            self.graph_buttons
                .update(ctx, event, bounds, &mut self.viewport);

        if response.consumed_event {
            return (response, None);
        }

        let to_vp = |vp: &Viewport, p: DVec2| {
            flip_y(p - bounds.pos - 0.5 * bounds.size) / bounds.size * vp.size(bounds) + vp.center
        };
        let from_vp = |vp: &Viewport, p: DVec2| {
            flip_y(p - vp.center) / vp.size(bounds) * bounds.size + bounds.pos + 0.5 * bounds.size
        };
        let zoom = |vp: &mut Viewport, amount: f64, about: DVec2, axis: Option<Axis>| {
            let origin = from_vp(vp, DVec2::ZERO);
            let p = if amount > 1.0 && (about - origin).abs().max_element() < 25.0 {
                origin
            } else {
                about
            };
            let p_vp = to_vp(vp, p);

            if let Some(axis) = axis {
                let mut size = vp.size(bounds);
                match axis {
                    Axis::X => size.x /= amount,
                    Axis::Y => size.y /= amount,
                }
                vp.width = size.x;
                vp.height = Some(size.y);
            } else {
                vp.width /= amount;
                if let Some(height) = &mut vp.height {
                    *height /= amount;
                }
            }

            vp.center += p_vp - to_vp(vp, p);
        };

        let mut dragged_point = None;
        let origin_clamped = from_vp(&self.viewport, DVec2::ZERO).clampb(bounds);
        let new_hovered = self.dragging.unwrap_or_else(|| {
            if buttons_hovered || ctx.left_mouse_button_already_pressed {
                return DragTarget::GraphPaper;
            }

            if ctx.modifiers.shift_key() && bounds.contains(ctx.cursor) {
                let offset = (origin_clamped - ctx.cursor).abs();
                let axis_resize_radius = 40.0;
                if offset.x < axis_resize_radius
                    && (self.hovered.axis() == Some(Axis::Y)
                        || offset.x < offset.y && self.hovered.axis() != Some(Axis::X))
                {
                    return DragTarget::Axis {
                        axis: Axis::Y,
                        inital_viewport: self.viewport,
                        initial_cursor: ctx.cursor,
                    };
                }
                if offset.y < axis_resize_radius {
                    return DragTarget::Axis {
                        axis: Axis::X,
                        inital_viewport: self.viewport,
                        initial_cursor: ctx.cursor,
                    };
                }
            }

            for g in self.geometry.iter().rev() {
                if let GeometryKind::Point {
                    p,
                    draggable: Some(i),
                    ..
                } = g.kind
                    && from_vp(&self.viewport, p).distance(ctx.cursor)
                        < draggable_point_width(g.width) as f64 / 2.0
                {
                    return DragTarget::Point(i);
                }
            }

            DragTarget::GraphPaper
        });
        if set(&mut self.hovered, new_hovered) {
            response.request_redraw();
        }

        match event {
            Event::MouseInput(ElementState::Pressed, MouseButton::Left)
                if bounds.contains(ctx.cursor) =>
            {
                self.dragging = Some(self.hovered);
                response.consume_event();
            }
            Event::MouseInput(ElementState::Released, MouseButton::Left)
                if self.dragging.is_some() =>
            {
                self.dragging = None;
                // TODO check why consuming release
                response.consume_event();
            }
            Event::CursorMoved { previous_cursor } => {
                if let Some(target) = &mut self.dragging {
                    let diff =
                        to_vp(&self.viewport, ctx.cursor) - to_vp(&self.viewport, *previous_cursor);

                    match target {
                        DragTarget::Point(i) => {
                            if let Some(p) = self.geometry.iter().find_map(|g| {
                                if let GeometryKind::Point { p, draggable } = g.kind
                                    && draggable == Some(*i)
                                {
                                    Some(p)
                                } else {
                                    None
                                }
                            }) {
                                dragged_point = Some((*i, p + diff));
                            } else {
                                self.dragging = None;
                            }
                        }
                        DragTarget::Axis {
                            axis,
                            inital_viewport,
                            initial_cursor,
                        } => {
                            let min = 5.0;
                            let origin = axis.get(origin_clamped);
                            let initial = axis.get(*initial_cursor) - origin;
                            let current = axis.get(ctx.cursor) - origin;
                            if initial.abs() < min {
                                // cursor is too close to origin. let the direction the user drags
                                // in determine what ends up happening
                                *initial_cursor = ctx.cursor.clampb(bounds);
                            } else {
                                let current = if initial > 0.0 {
                                    current.max(min)
                                } else {
                                    current.min(-min)
                                };
                                self.viewport = *inital_viewport;
                                zoom(
                                    &mut self.viewport,
                                    current / initial,
                                    origin_clamped,
                                    Some(*axis),
                                )
                            };
                        }
                        DragTarget::GraphPaper => self.viewport.center -= diff,
                    }

                    response.request_redraw();
                    response.consume_event();
                }
            }
            Event::MouseWheel(delta) if bounds.contains(ctx.cursor) => {
                zoom(
                    &mut self.viewport,
                    (delta.y * 0.0015).exp2(),
                    ctx.cursor,
                    self.hovered.axis(),
                );
                response.request_redraw();
                response.consume_event();
            }
            Event::PinchGesture(delta) if bounds.contains(ctx.cursor) => {
                zoom(
                    &mut self.viewport,
                    delta.exp(),
                    ctx.cursor,
                    self.hovered.axis(),
                );
                response.request_redraw();
                response.consume_event();
            }
            _ => {}
        }

        if !buttons_hovered {
            response.cursor_mode = match self.hovered {
                DragTarget::Point(_) => CursorMode::Icon(CursorIcon::AllScroll),
                DragTarget::Axis { axis: Axis::X, .. } => CursorMode::Icon(CursorIcon::EwResize),
                DragTarget::Axis { axis: Axis::Y, .. } => CursorMode::Icon(CursorIcon::NsResize),
                DragTarget::GraphPaper => CursorMode::NoPreference,
            };
        }

        (response, dragged_point)
    }

    pub fn render(
        &mut self,
        ctx: &Context,
        AppGraphics {
            device,
            config,
            queue,
            ..
        }: &AppGraphics,
        view: &wgpu::TextureView,
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
        let vp_size = vp.size(bounds);
        let physical = ctx.to_physical(bounds);

        let s = vp_size / bounds.size * 80.0;
        let (mut major, mut minor) = (DVec2::INFINITY, DVec2::ZERO);
        for (a, b) in [(1.0, 5.0), (2.0, 4.0), (5.0, 5.0)] {
            let c = a * (s / a).map(f64::log10).ceil().map(|x| 10f64.powf(x));
            if c.x < major.x {
                major.x = c.x;
                minor.x = c.x / b;
            }
            if c.y < major.y {
                major.y = c.y;
                minor.y = c.y / b;
            }
        }

        let mut draw_grid = |step: DVec2, color: [f32; 4], width: u32| {
            let shape = shapes.len() as u32;
            shapes.push(Shape::line(color, width as f32));
            let a = (0.5 * vp_size / step).ceil();
            let n = 2 * a.as_uvec2() + 2;
            let b = flip_y(step / vp_size * physical.size);
            let c = (0.5 - flip_y(vp.center.rem_euclid(step) + a * step) / vp_size) * physical.size
                + physical.pos;

            for i in 0..n.x {
                let x = i as f64 * b.x + c.x;
                if x <= physical.left() || physical.right() <= x {
                    // The default home viewport with sidebar shadow looks weird
                    // if you draw the edge lines so skip them
                    continue;
                }
                let x = snap(x, width) as f32;
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
            flip_y(p - vp.center) / vp_size * physical.size + 0.5 * physical.size + physical.pos
        };

        let w = ctx.round_nonzero_as_physical(1.5);
        let origin = to_physical(DVec2::ZERO).map(|x| snap(x, w)).as_vec2();
        let axis_color = |axis| {
            if self.hovered.axis() == Some(axis) {
                [0.4, 0.7, 0.9, 1.0]
            } else {
                [0.098, 0.098, 0.098, 1.0]
            }
        };
        let mut shape_y = shapes.len() as u32;
        shapes.push(Shape::line(axis_color(Axis::Y), w as f32));
        let mut shape_x = shapes.len() as u32;
        shapes.push(Shape::line(axis_color(Axis::X), w as f32));
        if self.hovered.axis() == Some(Axis::Y) {
            // make it render on top
            shapes.swap(shape_x as usize, shape_y as usize);
            (shape_x, shape_y) = (shape_y, shape_x);
        }
        vertices.push(Vertex::new((origin.x, physical.top() as f32), shape_y));
        vertices.push(Vertex::new((origin.x, physical.bottom() as f32), shape_y));
        vertices.push(Vertex::new((physical.left() as f32, origin.y), shape_x));
        vertices.push(Vertex::new((physical.right() as f32, origin.y), shape_x));

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

                    let mut vm = Vm::new(instructions, std::mem::take(&mut self.vm_vars), []);
                    let pixels_per_math = vp_size / physical.size;

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

                    if let Some(id) = *draggable {
                        let shape = shapes.len() as u32;
                        let mut color = *color;
                        color[3] *= 0.35;
                        let draggable_width = draggable_point_width(width);
                        shapes.push(Shape::point(
                            color,
                            ctx.scale_factor as f32 * draggable_width,
                        ));
                        vertices.push(Vertex::new(p, shape));

                        if self.hovered == DragTarget::Point(id) {
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

    pub fn render_buttons(
        &mut self,
        ctx: &Context,
        bounds: Bounds,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        self.graph_buttons.render(ctx, bounds, draw_quad);
    }
}

#[derive(Default)]
struct Button {
    hovered: bool,
    pressed: bool,
}

impl Button {
    fn fill_color(&self) -> DVec4 {
        if self.pressed { [232; 3] } else { [237; 3] }.to_rgbaf64()
    }

    fn icon_color(&self) -> DVec4 {
        if self.pressed {
            [0; 3]
        } else if self.hovered {
            [24; 3]
        } else {
            [95; 3]
        }
        .to_rgbaf64()
    }
}

impl Button {
    fn update(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> (Response, bool) {
        let mut response = Response::default();

        let new_hovered =
            bounds.contains(ctx.cursor) && (self.pressed || !ctx.left_mouse_button_already_pressed);
        if set(&mut self.hovered, new_hovered) && !self.pressed {
            response.request_redraw();
        }
        let mut clicked = false;

        match event {
            Event::MouseInput(ElementState::Pressed, MouseButton::Left) if self.hovered => {
                self.pressed = true;
                response.consume_event();
                response.request_redraw();
            }
            Event::MouseInput(ElementState::Released, MouseButton::Left) if self.pressed => {
                self.pressed = false;
                response.request_redraw();
                clicked = self.hovered;
            }
            _ => {}
        }

        if self.pressed || self.hovered {
            response.cursor_mode = CursorMode::Icon(CursorIcon::Pointer);
        }

        (response, clicked)
    }
}

struct ViewportAnimation {
    start: Viewport,
    end: Viewport,
    start_time: f64,
    duration: f64,
}

impl ViewportAnimation {
    fn is_animating(&self, time: f64) -> bool {
        time - self.start_time < self.duration
    }

    fn get(&self, time: f64, bounds: Bounds) -> Viewport {
        if !self.is_animating(time) {
            return self.end;
        }

        let t = (time - self.start_time) / self.duration;
        let t = t * t * (10.0 + t * (-20.0 + t * (15.0 - 4.0 * t)));

        let start_size = self.start.size(bounds);
        let end_size = self.end.size(bounds);
        let r = end_size / start_size;
        let a = (r.ln() * t).map(f64::exp_m1);
        let mut size = (a + 1.0) * start_size;
        let mut center = mix(self.start.center, self.end.center, a / (r - 1.0));

        if !size.x.is_finite() || !center.x.is_finite() {
            size.x = mix(start_size.x, end_size.x, t);
            center.x = mix(self.start.center.x, self.end.center.x, t);
        }

        if !size.y.is_finite() || !center.y.is_finite() {
            size.y = mix(start_size.y, end_size.y, t);
            center.y = mix(self.start.center.y, self.end.center.y, t);
        }

        Viewport {
            center,
            width: size.x,
            height: (self.start.height.is_some() || self.end.height.is_some()).then_some(size.y),
        }
    }
}

struct GraphButtons {
    plus: Button,
    minus: Button,
    home: Button,
    viewport_animation: Option<ViewportAnimation>,
    home_button_showing: bool,
    home_button_showing_amount: AnimatedValue,
}

impl GraphButtons {
    fn new() -> Self {
        Self {
            plus: Default::default(),
            minus: Default::default(),
            home: Default::default(),
            viewport_animation: None,
            home_button_showing: false,
            home_button_showing_amount: AnimatedValue::new(0.0),
        }
    }

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        bounds: Bounds,
        viewport: &mut Viewport,
    ) -> (Response, bool) {
        let mut response = Response::default();

        if matches!(event, Event::MouseInput(ElementState::Pressed, _))
            && !bounds.contains(ctx.cursor)
        {
            return (response, false);
        }

        if event == &Event::AnimationFrame {
            if let Some(animation) = &self.viewport_animation {
                *viewport = animation.get(ctx.time, bounds);
                if animation.is_animating(ctx.time) {
                    response.request_redraw();
                } else {
                    self.viewport_animation = None;
                }
            }
            if self.home_button_showing_amount.is_animating(ctx.time) {
                response.request_redraw();
            }
        }

        let size = 37.0;
        let padding = 5.0;
        let stroke_width = 1.0; // hardcoded in quad.wgsl too
        let mut offset = dvec2(bounds.right() - size - padding, bounds.top() + padding);

        let (r, plus_clicked) = self.plus.update(
            ctx,
            event,
            Bounds {
                pos: offset,
                size: dvec2(size, size - stroke_width * 0.5),
            },
        );
        response = response.or(r);
        offset.y += size - stroke_width * 0.5;
        let (r, minus_clicked) = self.minus.update(
            ctx,
            event,
            Bounds {
                pos: offset,
                size: dvec2(size, size - stroke_width * 0.5),
            },
        );
        response = response.or(r);
        offset.y += size - stroke_width * 0.5 + padding;

        let home_clicked = if self.home_button_showing {
            let (r, home_clicked) = self.home.update(
                ctx,
                event,
                Bounds {
                    pos: offset,
                    size: DVec2::splat(size),
                },
            );
            response = response.or(r);
            home_clicked
        } else {
            self.home.hovered = false;
            false
        };

        let reference_viewport = match &self.viewport_animation {
            Some(animation) => animation.end,
            None => *viewport,
        };

        // check if we need to show the home button
        let home_viewport = Viewport::default();
        let pixels_off_center = reference_viewport.center.distance(home_viewport.center)
            / reference_viewport.width
            * bounds.size.x
            * ctx.scale_factor;

        if set(
            &mut self.home_button_showing,
            pixels_off_center > 0.4
                || home_viewport.width != reference_viewport.width
                || home_viewport.height != reference_viewport.height,
        ) {
            self.home_button_showing_amount.animate_towards(
                if self.home_button_showing { 1.0 } else { 0.0 },
                0.2,
                2,
                ctx.time,
            );
            response.request_redraw();
        }

        // start any necessary zoom animations
        let end = if plus_clicked {
            Some(Viewport {
                width: reference_viewport.width / 2.0,
                height: reference_viewport.height.map(|h| h / 2.0),
                ..reference_viewport
            })
        } else if minus_clicked {
            Some(Viewport {
                width: reference_viewport.width * 2.0,
                height: reference_viewport.height.map(|h| h * 2.0),
                ..reference_viewport
            })
        } else if home_clicked {
            Some(Viewport::default())
        } else {
            None
        };

        if let Some(end) = end {
            self.viewport_animation = Some(ViewportAnimation {
                start: *viewport,
                end,
                start_time: ctx.time,
                duration: 0.2,
            });
            response.request_redraw();
        }

        let any_button_hovered = self.plus.hovered || self.minus.hovered || self.home.hovered;

        (response, any_button_hovered)
    }

    fn render(&mut self, ctx: &Context, bounds: Bounds, draw_quad: &mut impl FnMut(Quad)) {
        let mut draw_quad = |quad: Quad| draw_quad(quad.clip(bounds));
        let size = 37.0;
        let padding = 5.0;
        let stroke_width = 1.0; // hardcoded in quad.wgsl
        let mut offset = dvec2(bounds.right() - size - padding, bounds.top() + padding);
        let shadow_color = (0, 0, 0, 0.055).to_rgbaf64();
        let shadow_radius = 5.0; // hardcoded in quad.wgsl

        // +- shadow
        draw_quad(Quad {
            kind: QuadKind::GraphButtonShadow,
            p0: offset - shadow_radius,
            p1: offset + dvec2(size, 2.0 * size - stroke_width) + shadow_radius,
            color: shadow_color,
            ..Default::default()
        });

        // +
        draw_quad(Quad {
            kind: QuadKind::GraphButtonUpper,
            p0: offset,
            p1: offset + size,
            color: self.plus.fill_color(),
            ..Default::default()
        });
        draw_quad(
            Quad::rectangle(
                offset + size / 2.0 - dvec2(1.25, 6.0),
                offset + size / 2.0 + dvec2(1.25, 6.0),
                self.plus.icon_color(),
            )
            .pixel_snap(ctx),
        );
        draw_quad(
            Quad::rectangle(
                offset + size / 2.0 - dvec2(6.0, 1.25),
                offset + size / 2.0 + dvec2(6.0, 1.25),
                self.plus.icon_color(),
            )
            .pixel_snap(ctx),
        );
        offset.y += size - stroke_width;

        // -
        draw_quad(Quad {
            kind: QuadKind::GraphButtonLower,
            p0: offset,
            p1: offset + size,
            color: self.minus.fill_color(),
            ..Default::default()
        });
        draw_quad(
            Quad::rectangle(
                offset + size / 2.0 - dvec2(6.0, 1.25),
                offset + size / 2.0 + dvec2(6.0, 1.25),
                self.minus.icon_color(),
            )
            .pixel_snap(ctx),
        );
        offset.y += size + padding;

        // home
        let home_animation = self.home_button_showing_amount.get(ctx.time);
        let home_opacity = home_animation.powi(2);
        let home_center = offset + size / 2.0;
        let home_size = home_animation;
        let mut draw_home_quad = |quad: Quad| {
            draw_quad(Quad {
                p0: (quad.p0 - home_center) * home_size + home_center,
                p1: (quad.p1 - home_center) * home_size + home_center,
                color: quad.color.with_opacity(home_opacity),
                ..quad
            })
        };
        draw_home_quad(Quad {
            kind: QuadKind::GraphButtonShadow,
            p0: offset - shadow_radius,
            p1: offset + size + shadow_radius,
            color: shadow_color,
            ..Default::default()
        });
        draw_home_quad(Quad {
            kind: QuadKind::GraphButton,
            p0: offset,
            p1: offset + size,
            color: self.home.fill_color(),
            ..Default::default()
        });
        draw_home_quad(Quad {
            kind: QuadKind::HomeIcon,
            p0: offset + dvec2(10.7, 12.3),
            p1: offset + dvec2(26.3, 24.7),
            color: self.home.icon_color(),
            ..Default::default()
        });
    }
}
