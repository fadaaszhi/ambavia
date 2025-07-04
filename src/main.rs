mod katex_font;
mod math_field;

use std::{f64, sync::Arc};

use ambavia::latex_parser::parse_latex;
use arboard::Clipboard;
use glam::{DVec2, UVec2, Vec2, dvec2, uvec2, vec2};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, KeyEvent, Modifiers, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::ModifiersState,
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

        fn exiting(&mut self, _: &ActiveEventLoop) {
            self.0 = None;
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

impl Mix<DVec2> for DVec2 {
    fn mix(self, other: Self, t: DVec2) -> Self {
        dvec2(mix(self.x, other.x, t.x), mix(self.y, other.y, t.y))
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
    context: Context,
    main_thing: MainThing,
}

struct Context {
    clipboard: Clipboard,
    prev_cursor: DVec2,
    cursor: DVec2,
    scale_factor: f64,
    modifiers: ModifiersState,
}

impl Context {
    fn update(&mut self, event: &WindowEvent) {
        // Should this happen every event or only when CursorMoved?
        self.prev_cursor = self.cursor;

        match &event {
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::CursorMoved { position, .. } => self.cursor = position.as_glam(),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = *scale_factor
            }
            _ => {}
        }
    }

    fn scale_width(&self, width: f64) -> u32 {
        (self.scale_factor * width).round().max(1.0) as u32
    }
}

fn snap(x: f64, w: u32) -> f64 {
    let a = 0.5 * (w % 2) as f64;
    (x - a).round() + a
}

impl App {
    fn new(event_loop: &ActiveEventLoop) -> App {
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Ambavia")
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
        let present_modes = surface.get_capabilities(&adapter).present_modes;
        if present_modes.contains(&wgpu::PresentMode::Mailbox) {
            config.present_mode = wgpu::PresentMode::Mailbox;
        }
        surface.configure(&device, &config);
        let clipboard = Clipboard::new().unwrap();
        let context = Context {
            clipboard,
            prev_cursor: DVec2::ZERO,
            cursor: DVec2::ZERO,
            scale_factor: window.scale_factor(),
            modifiers: Default::default(),
        };
        let main_thing = MainThing::new(&device, &queue, &config);

        App {
            window,
            surface,
            config,
            device,
            queue,
            context,
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

        self.context.update(&event);

        'update: {
            let my_event = match event.clone() {
                WindowEvent::KeyboardInput { event, .. } => Event::KeyboardInput(event),
                WindowEvent::ModifiersChanged(modifiers) => Event::ModifiersChanged(modifiers),
                WindowEvent::CursorMoved { .. } => Event::CursorMoved,
                WindowEvent::MouseWheel { delta, .. } => Event::MouseWheel(match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => vec2(x, y).as_dvec2() * 20.0,
                    winit::event::MouseScrollDelta::PixelDelta(delta) => delta.as_glam(),
                }),
                WindowEvent::MouseInput { state, button, .. } => Event::MouseInput(state, button),
                WindowEvent::PinchGesture { delta, .. } => Event::PinchGesture(delta),
                _ => break 'update,
            };
            let response = self.main_thing.update(&mut self.context, &my_event, bounds);
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
                    &self.context,
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
    CursorMoved,
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

    fn intersect(&self, other: &Bounds) -> Bounds {
        let p0 = uvec2(self.left(), self.top()).max(uvec2(other.left(), other.top()));
        let p1 = uvec2(self.right(), self.bottom()).min(uvec2(other.right(), other.bottom()));
        Bounds {
            pos: p0,
            size: p1.saturating_sub(p0),
        }
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

    fn or(self, other: Response) -> Response {
        Response {
            consumed_event: self.consumed_event | other.consumed_event,
            requested_redraw: self.requested_redraw | other.requested_redraw,
            cursor_icon: if self.consumed_event
                || !other.consumed_event && other.cursor_icon == CursorIcon::Default
            {
                self.cursor_icon
            } else {
                other.cursor_icon
            },
        }
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
    expression_list: expression_list::ExpressionList,
    graph_paper: graph::GraphPaper,
}

impl MainThing {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> MainThing {
        MainThing {
            resizer_width: 25.0,
            resizer_position: 0.3,
            dragging: None,
            expression_list: expression_list::ExpressionList::new(device, queue, config),
            graph_paper: graph::GraphPaper::new(device, config),
        }
    }

    fn update(&mut self, ctx: &mut Context, event: &Event, bounds: Bounds) -> Response {
        let mut response = Response::default();

        let l = bounds.left() as f64;
        let r = bounds.right() as f64;
        let mut x = mix(l, r, self.resizer_position);

        if let Event::CursorMoved = event {
            if let Some(offset) = self.dragging {
                x = (ctx.cursor.x + offset).clamp(l, r);
                self.resizer_position = unmix(x, l, r);
                response.consume_event();
                response.request_redraw();
            }
        }

        let offset = x - ctx.cursor.x;
        let hovering = offset.abs() <= ctx.scale_factor * self.resizer_width / 2.0;

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

        let x = x.round() as u32;
        let left = Bounds {
            pos: bounds.pos,
            size: uvec2(x - bounds.left(), bounds.size.y),
        };
        let right = Bounds {
            pos: uvec2(x, bounds.pos.y),
            size: uvec2(bounds.right() - x, bounds.size.y),
        };

        response
            .or_else(|| self.expression_list.update(ctx, event, left))
            .or_else(|| self.graph_paper.update(ctx, event, right))
    }

    fn render(
        &mut self,
        ctx: &Context,
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
        let left = Bounds {
            pos: bounds.pos,
            size: uvec2(x - bounds.left(), bounds.size.y),
        };
        let right = Bounds {
            pos: uvec2(x, bounds.pos.y),
            size: uvec2(bounds.right() - x, bounds.size.y),
        };
        self.expression_list
            .render(ctx, device, queue, view, config, &mut encoder, left);
        self.graph_paper
            .render(ctx, device, queue, view, config, &mut encoder, right);
        Some(encoder.finish())
    }
}

mod expression_list {
    use bytemuck::{Zeroable, offset_of};

    use crate::math_field::{MathField, Message};

    use super::*;

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
        uv: Vec2,
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
                                format: wgpu::VertexFormat::Float32x2,
                                offset: offset_of!(Vertex::zeroed(), Vertex, uv) as _,
                                shader_location: 1,
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

        pub fn update(&mut self, ctx: &mut Context, event: &Event, bounds: Bounds) -> Response {
            let mut response = Response::default();
            let mut y_offset = 0;
            let mut message = None;
            let padding = ctx.scale_width(Self::PADDING);

            for (i, expression) in self.expressions.iter_mut().enumerate() {
                let y_size = expression.size(ctx).y.ceil() as u32;
                let (r, m) = expression.update(
                    ctx,
                    event,
                    bounds.intersect(&Bounds {
                        pos: bounds.pos + uvec2(0, y_offset) + padding,
                        size: uvec2(bounds.size.x, y_size),
                    }),
                );

                if let Some(m) = m {
                    message = Some((i, m));
                }

                response = response.or(r);
                y_offset += y_size + 2 * padding + ctx.scale_width(Self::SEPARATOR_WIDTH);
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
            let draw_quad = &mut |p0: DVec2, p1: DVec2, uv0: DVec2, uv1: DVec2| {
                let p0 = p0.as_vec2();
                let p1 = p1.as_vec2();
                let uv0 = uv0.as_vec2();
                let uv1 = uv1.as_vec2();

                indices.push(vertices.len() as u32 + 0);
                indices.push(vertices.len() as u32 + 1);
                indices.push(vertices.len() as u32 + 2);
                indices.push(vertices.len() as u32 + 3);
                indices.push(0xffffffff);

                vertices.push(Vertex {
                    position: p0,
                    uv: uv0,
                });
                vertices.push(Vertex {
                    position: vec2(p1.x, p0.y),
                    uv: vec2(uv1.x, uv0.y),
                });
                vertices.push(Vertex {
                    position: vec2(p0.x, p1.y),
                    uv: vec2(uv0.x, uv1.y),
                });
                vertices.push(Vertex {
                    position: p1,
                    uv: uv1,
                });
            };
            let mut y_offset = 0;
            let padding = ctx.scale_width(Self::PADDING);

            for expression in &mut self.expressions {
                let y_size = expression.size(ctx).y.ceil() as u32;
                expression.render(
                    ctx,
                    bounds.intersect(&Bounds {
                        pos: bounds.pos + uvec2(0, y_offset) + padding,
                        size: uvec2(bounds.size.x, y_size),
                    }),
                    draw_quad,
                );
                y_offset += y_size + 2 * padding;

                let p0 = (bounds.pos + uvec2(0, y_offset)).as_dvec2();
                let p1 = uvec2(
                    bounds.right(),
                    bounds.top() + y_offset + ctx.scale_width(Self::SEPARATOR_WIDTH),
                )
                .as_dvec2();
                let uv = DVec2::splat(-3.0);
                draw_quad(p0, p1, uv, uv);

                y_offset += ctx.scale_width(Self::SEPARATOR_WIDTH);
            }

            {
                let p0 = uvec2(
                    bounds
                        .right()
                        .saturating_sub(ctx.scale_width(Self::SEPARATOR_WIDTH)),
                    bounds.top(),
                );
                let p1 = uvec2(bounds.right(), bounds.bottom());
                let uv = DVec2::splat(-3.0);
                draw_quad(p0.as_dvec2(), p1.as_dvec2(), uv, uv);
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
            pass.set_scissor_rect(bounds.pos.x, bounds.pos.y, bounds.size.x, bounds.size.y);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw_indexed(0..indices.len() as _, 0, 0..1);
        }
    }
}

mod graph {
    use super::*;

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

    pub struct GraphPaper {
        viewport: Viewport,
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
                self.vertices_buffer =
                    vertices_buffer_with_capacity(device, self.vertices_capacity);
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

        pub fn update(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> Response {
            let mut response = Response::default();

            let to_vp = |vp: &Viewport, p: DVec2| {
                flip_y(p - bounds.pos.as_dvec2() - 0.5 * bounds.size.as_dvec2())
                    / bounds.size.x as f64
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
                let p = if amount > 1.0
                    && (ctx.cursor - origin).abs().max_element() < 25.0 * ctx.scale_factor
                {
                    origin
                } else {
                    ctx.cursor
                };
                let p_vp = to_vp(&self.viewport, p);
                self.viewport.width /= amount;
                self.viewport.center += p_vp - to_vp(&self.viewport, p);
                response.request_redraw();
                response.consume_event();
            };

            match event {
                Event::MouseInput(ElementState::Pressed, MouseButton::Left)
                    if bounds.contains(ctx.cursor) =>
                {
                    self.dragging = true;
                    response.consume_event();
                }
                Event::MouseInput(ElementState::Released, MouseButton::Left) if self.dragging => {
                    self.dragging = false;
                    response.consume_event();
                }
                Event::CursorMoved => {
                    if self.dragging {
                        let diff = to_vp(&self.viewport, ctx.cursor)
                            - to_vp(&self.viewport, ctx.prev_cursor);

                        self.viewport.center -= diff;
                        response.request_redraw();
                        response.consume_event();
                    }
                }
                Event::MouseWheel(delta) if bounds.contains(ctx.cursor) => {
                    zoom((delta.y * 0.0015).exp2());
                }
                Event::PinchGesture(delta) if bounds.contains(ctx.cursor) => {
                    zoom(delta.exp());
                }
                _ => {}
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
            if bounds.is_empty() {
                return;
            }

            let (shapes, vertices) = self.generate_geometry(ctx, bounds);

            if vertices.is_empty() {
                return;
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

        fn generate_geometry(&self, ctx: &Context, bounds: Bounds) -> (Vec<Shape>, Vec<Vertex>) {
            let mut shapes = vec![];
            let mut vertices = vec![];
            let bounds_pos = bounds.pos.as_dvec2();
            let bounds_size = bounds.size.as_dvec2();
            let vp = &self.viewport;
            let vp_size = dvec2(vp.width, vp.width * bounds_size.y / bounds_size.x);

            let s = vp.width / bounds_size.x * (80.0 * ctx.scale_factor);
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
                let b = flip_y(s / vp_size * bounds_size);
                let c = (0.5 - flip_y(vp.center.rem_euclid(s) + a * s) / vp_size) * bounds_size
                    + bounds_pos;

                for i in 0..n.x {
                    let x = snap(i as f64 * b.x + c.x, width) as f32;
                    vertices.push(Vertex::BREAK);
                    vertices.push(Vertex::new((x, bounds.top() as f32), shape));
                    vertices.push(Vertex::new((x, bounds.bottom() as f32), shape));
                }

                for i in 0..n.y {
                    let y = snap(i as f64 * b.y + c.y, width) as f32;
                    vertices.push(Vertex::BREAK);
                    vertices.push(Vertex::new((bounds.left() as f32, y), shape));
                    vertices.push(Vertex::new((bounds.right() as f32, y), shape));
                }
            };

            draw_grid(minor, [0.88, 0.88, 0.88, 1.0], ctx.scale_width(1.0));
            draw_grid(major, [0.6, 0.6, 0.6, 1.0], ctx.scale_width(1.0));

            let to_frame = |p: DVec2| {
                flip_y(p - vp.center) / vp.width * bounds_size.x + 0.5 * bounds_size + bounds_pos
            };

            let w = ctx.scale_width(1.5);
            let origin = to_frame(DVec2::ZERO).map(|x| snap(x, w)).as_vec2();
            let shape = shapes.len() as u32;
            shapes.push(Shape::line([0.098, 0.098, 0.098, 1.0], w as f32));
            vertices.push(Vertex::new((origin.x, bounds.top() as f32), shape));
            vertices.push(Vertex::new((origin.x, bounds.bottom() as f32), shape));
            vertices.push(Vertex::BREAK);
            vertices.push(Vertex::new((bounds.left() as f32, origin.y), shape));
            vertices.push(Vertex::new((bounds.right() as f32, origin.y), shape));

            (shapes, vertices)
        }
    }
}
