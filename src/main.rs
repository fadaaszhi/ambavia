use std::{f64, sync::Arc};

use glam::{dvec2, uvec2, vec2, DVec2, UVec2, Vec2};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, KeyEvent, Modifiers, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{CursorIcon, Window, WindowAttributes, WindowId},
};

fn main() -> Result<(), winit::error::EventLoopError> {
    struct AppRaw(Option<App>);

    impl winit::application::ApplicationHandler for AppRaw {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.0.is_none() {
                self.0 = Some(App::new(event_loop));
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            if let Some(app) = &mut self.0 {
                app.window_event(event_loop, window_id, event);
            }
        }
    }

    EventLoop::new()?.run_app(&mut AppRaw(None))
}

trait AsGlam {
    type G;
    fn as_glam(&self) -> Self::G;
}

impl AsGlam for PhysicalPosition<f64> {
    type G = DVec2;
    fn as_glam(&self) -> DVec2 {
        dvec2(self.x, self.y)
    }
}

impl AsGlam for PhysicalSize<u32> {
    type G = UVec2;
    fn as_glam(&self) -> UVec2 {
        uvec2(self.width, self.height)
    }
}

fn flip_y(v: DVec2) -> DVec2 {
    dvec2(v.x, -v.y)
}

trait Mix<I> {
    fn mix(self, other: Self, t: I) -> Self;
}

fn mix<T: Mix<I>, I>(x: T, y: T, t: I) -> T {
    x.mix(y, t)
}

impl Mix<f64> for f64 {
    fn mix(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }
}

impl Mix<f64> for DVec2 {
    fn mix(self, other: Self, t: f64) -> Self {
        self.lerp(other, t)
    }
}

fn unmix(t: f64, x: f64, y: f64) -> f64 {
    (t - x) / (y - x)
}

struct App {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    main_thing: MainThing,
}

impl App {
    fn new(event_loop: &ActiveEventLoop) -> App {
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Ambavia")
                        .with_inner_size(PhysicalSize::new(1440, 1080))
                        .with_theme(Some(winit::window::Theme::Light)),
                )
                .unwrap(),
        );
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("Failed to find an appropriate adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))
            .expect("Failed to create device");
        let size = window.inner_size().as_glam().max(UVec2::splat(1));
        let mut config = surface
            .get_default_config(&adapter, size.x, size.y)
            .unwrap();
        config.format = config.format.remove_srgb_suffix();
        surface.configure(&device, &config);
        let main_thing = MainThing::new(&device, &config);

        App {
            window,
            surface,
            config,
            device,
            queue,
            main_thing,
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let bounds = Bounds {
            pos: UVec2::ZERO,
            size: self.window.inner_size().as_glam(),
        };

        'update: {
            let my_event = match event.clone() {
                WindowEvent::KeyboardInput { event, .. } => Event::KeyboardInput(event),
                WindowEvent::ModifiersChanged(modifiers) => Event::ModifiersChanged(modifiers),
                WindowEvent::CursorMoved { position, .. } => Event::CursorMoved(position.as_glam()),
                WindowEvent::MouseWheel { delta, .. } => Event::MouseWheel(match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => vec2(x, y).as_dvec2() * 20.0,
                    winit::event::MouseScrollDelta::PixelDelta(delta) => delta.as_glam(),
                }),
                WindowEvent::MouseInput { state, button, .. } => Event::MouseInput(state, button),
                WindowEvent::PinchGesture { delta, .. } => Event::PinchGesture(delta),
                _ => break 'update,
            };
            let response = self.main_thing.update(&my_event, bounds);
            if response.requested_redraw {
                self.window.request_redraw();
            }
            self.window.set_cursor(response.cursor_icon);
        }

        match event {
            WindowEvent::Resized(new_size) => {
                self.config.width = new_size.width.max(1);
                self.config.height = new_size.height.max(1);
                self.surface.configure(&self.device, &self.config);
            }
            WindowEvent::RedrawRequested => {
                let surface_texture = self
                    .surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let surface_view = surface_texture.texture.create_view(&Default::default());
                let command_buffer = self.main_thing.render(
                    &self.device,
                    &self.queue,
                    &surface_view,
                    &self.config,
                    bounds,
                );
                self.queue.submit(command_buffer);
                self.window.pre_present_notify();
                surface_texture.present();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        };
    }
}

#[derive(Debug)]
pub enum Event {
    // Resized(UVec2),
    KeyboardInput(KeyEvent),
    ModifiersChanged(Modifiers),
    CursorMoved(DVec2),
    MouseWheel(DVec2),
    MouseInput(ElementState, MouseButton),
    PinchGesture(f64),
}

#[derive(Debug, Clone, Copy)]
struct Bounds {
    pos: UVec2,
    size: UVec2,
}

impl Bounds {
    fn left(&self) -> u32 {
        self.pos.x
    }

    fn right(&self) -> u32 {
        self.pos.x + self.size.x
    }

    fn top(&self) -> u32 {
        self.pos.y
    }

    fn bottom(&self) -> u32 {
        self.pos.y + self.size.y
    }

    fn is_empty(&self) -> bool {
        self.size.x == 0 || self.size.y == 0
    }

    fn contains(&self, position: DVec2) -> bool {
        (self.left() as f64 <= position.x && position.x < self.right() as f64)
            && (self.top() as f64 <= position.y && position.y < self.bottom() as f64)
    }
}

#[derive(Debug, Default)]
struct Response {
    consumed_event: bool,
    requested_redraw: bool,
    cursor_icon: CursorIcon,
}

impl Response {
    fn consume_event(&mut self) {
        self.consumed_event = true;
    }

    fn request_redraw(&mut self) {
        self.requested_redraw = true;
    }

    fn or_else(self, f: impl FnOnce() -> Response) -> Response {
        if self.consumed_event {
            self
        } else {
            let mut response = f();
            response.requested_redraw |= self.requested_redraw;
            if !response.consumed_event && self.cursor_icon != CursorIcon::Default {
                response.cursor_icon = self.cursor_icon;
            }
            response
        }
    }
}

struct MainThing {
    resizer_width: f64,
    resizer_position: f64,
    dragging: Option<f64>,
    cursor: DVec2,
    graph_paper: GraphPaper,
}

impl MainThing {
    fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> MainThing {
        MainThing {
            resizer_width: 50.0,
            resizer_position: 0.3,
            dragging: None,
            cursor: DVec2::ZERO,
            graph_paper: GraphPaper::new(device, config),
        }
    }

    fn update(&mut self, event: &Event, bounds: Bounds) -> Response {
        let mut response = Response::default();

        let l = bounds.left() as f64;
        let r = bounds.right() as f64;
        let mut x = mix(l, r, self.resizer_position);

        if let Event::CursorMoved(position) = event {
            if let Some(offset) = self.dragging {
                x = (self.cursor.x + offset).clamp(l, r);
                self.resizer_position = unmix(x, l, r);
                response.consume_event();
                response.request_redraw();
            }
            self.cursor = *position;
        }

        let offset = x - self.cursor.x;
        let hovering = offset.abs() <= self.resizer_width / 2.0;

        if let Event::MouseInput(state, MouseButton::Left) = event {
            match state {
                ElementState::Pressed if hovering => {
                    self.dragging = Some(offset);
                    response.consume_event();
                }
                ElementState::Released if self.dragging.is_some() => {
                    self.dragging = None;
                    response.consume_event();
                }
                _ => {}
            }
        }

        if hovering || self.dragging.is_some() {
            response.cursor_icon = CursorIcon::ColResize;

            // Really should be using TouchPhase here to not interrupt people
            // who started using these before we got hovered
            if matches!(event, Event::MouseWheel(_) | Event::PinchGesture(_)) {
                response.consume_event();
            }
        }

        response.or_else(|| {
            let x = x.round() as u32;
            let _left = Bounds {
                pos: bounds.pos,
                size: uvec2(x - bounds.left(), bounds.size.y),
            };
            let right = Bounds {
                pos: uvec2(x, bounds.pos.y),
                size: uvec2(bounds.right() - x, bounds.size.y),
            };
            self.graph_paper.update(event, right)
        })
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        config: &wgpu::SurfaceConfiguration,
        bounds: Bounds,
    ) -> Option<wgpu::CommandBuffer> {
        if bounds.is_empty() {
            return None;
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_thing_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        let x = mix(
            bounds.left() as f64,
            bounds.right() as f64,
            self.resizer_position,
        )
        .round() as u32;
        let _left = Bounds {
            pos: bounds.pos,
            size: uvec2(x - bounds.left(), bounds.size.y),
        };
        let right = Bounds {
            pos: uvec2(x, bounds.pos.y),
            size: uvec2(bounds.right() - x, bounds.size.y),
        };
        self.graph_paper
            .render(device, queue, view, config, &mut encoder, right);
        Some(encoder.finish())
        // todo!()
    }
}

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

struct GraphPaper {
    viewport: Viewport,
    cursor: DVec2,
    dragging: bool,

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

impl GraphPaper {
    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("graph_depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
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
                    resource: wgpu::BindingResource::Buffer(
                        uniforms_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        shapes_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        vertices_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        })
    }

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
            self.shapes_buffer = Self::shapes_buffer_with_capacity(device, self.shapes_capacity);
        }

        if vertices.len() > self.vertices_capacity {
            new_buffers = true;
            self.vertices_capacity = grow(self.vertices_capacity, vertices.len());
            self.vertices_buffer =
                Self::vertices_buffer_with_capacity(device, self.vertices_capacity);
        }

        if new_buffers {
            self.bind_group = Self::create_bind_group(
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

    fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> GraphPaper {
        let module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
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
        let depth_texture = Self::create_depth_texture(device, config.width, config.height);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("graph"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
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
            multisample: Default::default(),
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
        let shapes_buffer = Self::shapes_buffer_with_capacity(device, shapes_capacity);
        let vertices_capacity = 1;
        let vertices_buffer = Self::vertices_buffer_with_capacity(device, vertices_capacity);
        let bind_group = Self::create_bind_group(
            device,
            &layout,
            &uniforms_buffer,
            &shapes_buffer,
            &vertices_buffer,
        );
        GraphPaper {
            viewport: Default::default(),
            cursor: DVec2::ZERO,
            dragging: false,

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

    fn update(&mut self, event: &Event, bounds: Bounds) -> Response {
        let mut response = Response::default();

        let to_vp = |vp: &Viewport, p: DVec2| {
            flip_y(p - bounds.pos.as_dvec2() - 0.5 * bounds.size.as_dvec2()) / bounds.size.x as f64
                * vp.width
                + vp.center
        };
        let from_vp = |vp: &Viewport, p: DVec2| {
            flip_y(p - vp.center) / vp.width * bounds.size.x as f64
                + bounds.pos.as_dvec2()
                + 0.5 * bounds.size.as_dvec2()
        };
        let mut zoom = |amount: f64| {
            let origin = from_vp(&self.viewport, DVec2::ZERO);
            let p = if amount > 1.0 && (self.cursor - origin).abs().max_element() < 50.0 {
                origin
            } else {
                self.cursor
            };
            let p_vp = to_vp(&self.viewport, p);
            self.viewport.width /= amount;
            self.viewport.center += p_vp - to_vp(&self.viewport, p);
            response.request_redraw();
            response.consume_event();
        };

        match event {
            Event::MouseInput(ElementState::Pressed, MouseButton::Left)
                if bounds.contains(self.cursor) =>
            {
                self.dragging = true;
                response.consume_event();
            }
            Event::MouseInput(ElementState::Released, MouseButton::Left) if self.dragging => {
                self.dragging = false;
                response.consume_event();
            }
            Event::CursorMoved(position) => {
                if self.dragging {
                    let diff =
                        to_vp(&self.viewport, *position) - to_vp(&self.viewport, self.cursor);

                    self.viewport.center -= diff;
                    response.request_redraw();
                    response.consume_event();
                }

                self.cursor = *position;
            }
            Event::MouseWheel(delta) if bounds.contains(self.cursor) => {
                zoom((delta.y * 0.0015).exp2());
            }
            Event::PinchGesture(delta) if bounds.contains(self.cursor) => {
                zoom(delta.exp());
            }
            _ => {}
        }

        response
    }

    fn render(
        &mut self,
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

        let (shapes, vertices) = self.generate_geometry(bounds);

        if vertices.is_empty() {
            return;
        }

        if self.depth_texture.width() != config.width
            || self.depth_texture.height() != config.height
        {
            self.depth_texture = Self::create_depth_texture(device, config.width, config.height);
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
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
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
        pass.set_scissor_rect(bounds.pos.x, bounds.pos.y, bounds.size.x, bounds.size.y);
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

    fn generate_geometry(&self, bounds: Bounds) -> (Vec<Shape>, Vec<Vertex>) {
        let mut shapes = vec![];
        let mut vertices = vec![];
        let bounds_pos = bounds.pos.as_dvec2();
        let bounds_size = bounds.size.as_dvec2();
        let vp = &self.viewport;
        let vp_size = dvec2(vp.width, vp.width * bounds_size.y / bounds_size.x);

        let s = vp.width / bounds_size.x * 160.0;
        let (mut major, mut minor) = (f64::INFINITY, 0.0);
        for (a, b) in [(1.0, 5.0), (2.0, 4.0), (5.0, 5.0)] {
            let c = a * 10f64.powf((s / a).log10().ceil());
            if c < major {
                major = c;
                minor = c / b;
            }
        }

        let mut draw_grid = |step: f64, color: [f32; 4], width: f32| {
            let shape = shapes.len() as u32;
            shapes.push(Shape::line(color, width));
            let s = DVec2::splat(step);
            let a = (0.5 * vp_size / step).ceil();
            let n = 2 * a.as_uvec2() + 2;
            let b = flip_y(s / vp_size * bounds_size);
            let c = (0.5 - flip_y(vp.center.rem_euclid(s) + a * s) / vp_size) * bounds_size
                + bounds_pos;

            for i in 0..n.x {
                let x = (i as f64 * b.x + c.x).round() as f32;
                vertices.push(Vertex::BREAK);
                vertices.push(Vertex::new((x, bounds.top() as f32), shape));
                vertices.push(Vertex::new((x, bounds.bottom() as f32), shape));
            }

            for i in 0..n.y {
                let y = (i as f64 * b.y + c.y).round() as f32;
                vertices.push(Vertex::BREAK);
                vertices.push(Vertex::new((bounds.left() as f32, y), shape));
                vertices.push(Vertex::new((bounds.right() as f32, y), shape));
            }
        };

        draw_grid(minor, [0.88, 0.88, 0.88, 1.0], 2.0);
        draw_grid(major, [0.6, 0.6, 0.6, 1.0], 2.0);

        let to_frame = |p: DVec2| {
            flip_y(p - vp.center) / vp.width * bounds_size.x + 0.5 * bounds_size + bounds_pos
        };

        let origin = to_frame(DVec2::ZERO).floor().as_vec2() + 0.5;
        let shape = shapes.len() as u32;
        shapes.push(Shape::line([0.098, 0.098, 0.098, 1.0], 3.0));
        vertices.push(Vertex::new((origin.x, bounds.top() as f32), shape));
        vertices.push(Vertex::new((origin.x, bounds.bottom() as f32), shape));
        vertices.push(Vertex::BREAK);
        vertices.push(Vertex::new((bounds.left() as f32, origin.y), shape));
        vertices.push(Vertex::new((bounds.right() as f32, origin.y), shape));

        (shapes, vertices)
    }
}
