mod expression_list;
mod graph;
mod katex_font;
mod label;
mod math_field;
mod quad_renderer;
mod timer;
mod ui;
mod utility;

use std::{f64, sync::Arc, time::Instant};

use glam::{DVec2, UVec2, dvec2, vec2};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{CursorIcon, Window, WindowAttributes, WindowId},
};

use crate::{
    ui::{Bounds, Context, CursorMode, Event, Response},
    utility::AsGlam,
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

struct App {
    events: Vec<(WindowEvent, f64)>,
    request_redraw: bool,
    window: Arc<Window>,
    graphics: AppGraphics,
    main_thing: MainThing,
    context: Context,
    start_time: Instant,
}

pub struct AppGraphics {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub surface: wgpu::Surface<'static>,
}

impl AppGraphics {
    fn new(window: &Arc<Window>) -> Self {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(Arc::clone(window)).unwrap();
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
        config.present_mode = wgpu::PresentMode::Fifo;
        let present_modes = surface.get_capabilities(&adapter).present_modes;
        if present_modes.contains(&wgpu::PresentMode::Mailbox) {
            config.present_mode = wgpu::PresentMode::Mailbox;
        }
        surface.configure(&device, &config);

        Self {
            device,
            queue,
            config,
            surface,
        }
    }
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.config.width = new_size.width.max(1);
        self.config.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.config);
    }
    fn get_surface_texture(&mut self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Suboptimal(tex)
            | wgpu::CurrentSurfaceTexture::Success(tex) => Some(tex),
            v @ (wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout) => {
                println!("surface.get_current_texture() returned {v:?}");
                None
            }
            v => todo!("handle wgpu surface error {v:?}"),
        }
    }
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

        let graphics = AppGraphics::new(&window);

        let main_thing = MainThing::new(&graphics);
        let context = Context::new(&window);
        App {
            events: vec![],
            request_redraw: false,
            window,
            context,
            main_thing,
            graphics,
            start_time: Instant::now(),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let time = self.start_time.elapsed().as_secs_f64();

        if true {
            // Pool events and only execute them on RedrawRequested to work around a
            // weird issue where taking more than 8ms during a non-RedrawRequested
            // event meant winit didn't give us any chance to redraw, making our app
            // appear frozen even though under the hood it was running fine. With
            // this method we get extraneous RedrawRequested events, but it's okay
            // because we only actually redraw when self.request_redraw is set to
            // true by fake_window_event.
            if event != WindowEvent::RedrawRequested {
                self.window.request_redraw();
                self.events.push((event, time));
                return;
            }

            for (event, time) in std::mem::take(&mut self.events).drain(..) {
                self.fake_window_event(event_loop, window_id, event, time);
            }

            if self.request_redraw {
                self.request_redraw = false;
                self.fake_window_event(event_loop, window_id, WindowEvent::RedrawRequested, time);
                // AnimationFrame might have requested redraw
                if self.request_redraw {
                    self.window.request_redraw();
                }
            }
        } else {
            self.fake_window_event(event_loop, window_id, event, time);
            if self.request_redraw {
                self.request_redraw = false;
                self.window.request_redraw();
            }
        }
    }

    fn update_main_thing(&mut self, event: &Event, bounds: Bounds) {
        let response = self.main_thing.update(&self.context, event, bounds);
        self.request_redraw |= response.requested_redraw;

        if matches!(event, Event::CursorMoved { .. }) {
            self.window.set_cursor_visible(true);
        }

        // TODO this isn't a good solution to setting cursor icon because sometimes
        // things won't respond to all events to remember to set their cursor icons
        match response.cursor_mode {
            CursorMode::NoPreference => {
                self.window.set_cursor(CursorIcon::Default);
            }
            CursorMode::Hidden => self.window.set_cursor_visible(false),
            CursorMode::Icon(icon) => {
                self.window.set_cursor(icon);
            }
        }
    }

    fn fake_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
        time: f64,
    ) {
        let previous_cursor = self.context.cursor;
        self.context.update(&event, time);
        let bounds = Bounds {
            pos: DVec2::ZERO,
            size: self.window.inner_size().as_glam().as_dvec2() / self.context.scale_factor,
        };

        'update: {
            let my_event = match event.clone() {
                WindowEvent::Resized { .. } => Event::Resized,
                WindowEvent::KeyboardInput { event, .. } => Event::KeyboardInput(event),
                // Ignore redundant CursorMoved events that happen sometimes (at least on macOS)
                WindowEvent::CursorMoved { .. } if previous_cursor != self.context.cursor => {
                    Event::CursorMoved { previous_cursor }
                }
                // Is this delta a physical size? Do we need to convert it to
                // logical? I think it's already logical because my trackpad
                // feels less sensitive when I decrease my Mac's scale factor.
                WindowEvent::MouseWheel { delta, .. } => Event::MouseWheel(match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => vec2(x, y).as_dvec2() * 60.0,
                    winit::event::MouseScrollDelta::PixelDelta(delta) => delta.as_glam(),
                }),
                WindowEvent::MouseInput { state, button, .. } => Event::MouseInput(state, button),
                WindowEvent::PinchGesture { delta, .. } => Event::PinchGesture(delta),
                _ => break 'update,
            };
            self.update_main_thing(&my_event, bounds);
        }

        match event {
            WindowEvent::Resized(new_size) => {
                self.graphics.resize(new_size);
            }
            WindowEvent::ScaleFactorChanged { .. } => self.request_redraw = true,
            WindowEvent::Occluded(false) => self.request_redraw = true,
            WindowEvent::RedrawRequested => {
                let Some(surface_texture) = self.graphics.get_surface_texture() else {
                    return;
                };

                // TODO bit gross that we are reaching into context here
                self.context.time = self.start_time.elapsed().as_secs_f64();
                self.update_main_thing(&Event::AnimationFrame, bounds);

                let surface_view = surface_texture.texture.create_view(&Default::default());
                let command_buffer =
                    self.main_thing
                        .render(&self.context, &self.graphics, &surface_view, bounds);
                self.graphics.queue.submit(command_buffer);
                self.window.pre_present_notify();
                surface_texture.present();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        };
    }
}

const RESIZER_WIDTH: f64 = 25.0;
const MIN_EXPRESSION_LIST_WIDTH: f64 = 320.0;

struct MainThing {
    raw_resizer_size: f64,
    clamped_resizer_size: f64,
    dragging: Option<f64>,
    quad_renderer: quad_renderer::QuadRenderer,
    expression_list: expression_list::ExpressionList,
    graph_paper: graph::GraphPaper,
}

impl MainThing {
    fn new(graphics: &AppGraphics) -> MainThing {
        MainThing {
            raw_resizer_size: f64::NAN,
            clamped_resizer_size: 0.0,
            dragging: None,
            quad_renderer: quad_renderer::QuadRenderer::new(graphics),
            expression_list: expression_list::ExpressionList::new(),
            graph_paper: graph::GraphPaper::new(graphics),
        }
    }

    fn update(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> Response {
        let mut response = Response::default();

        if let Event::CursorMoved { .. } = event
            && let Some(offset) = self.dragging
        {
            self.raw_resizer_size =
                (ctx.cursor.x + offset - bounds.left()).clamp(0.0, bounds.size.x);
            response.consume_event();
        }

        let resized = {
            let previous_clamped = self.clamped_resizer_size;
            let expression_list_width = if self.raw_resizer_size.is_finite() {
                self.raw_resizer_size.min(bounds.size.x)
            } else {
                bounds.size.x * 0.3
            };
            self.clamped_resizer_size = if expression_list_width == 0.0
                || expression_list_width >= MIN_EXPRESSION_LIST_WIDTH
            {
                expression_list_width
            } else if expression_list_width < RESIZER_WIDTH
                || MIN_EXPRESSION_LIST_WIDTH > bounds.size.x
            {
                0.0
            } else {
                MIN_EXPRESSION_LIST_WIDTH
            };
            previous_clamped != self.clamped_resizer_size
        };

        if resized {
            response.request_redraw();
        }

        let x = bounds.left() + self.clamped_resizer_size;
        let offset = x - ctx.cursor.x;
        let hovering = offset.abs() <= RESIZER_WIDTH / 2.0;

        if let Event::MouseInput(state, MouseButton::Left) = event {
            match state {
                ElementState::Pressed if hovering => {
                    self.dragging = Some(offset);
                    response.consume_event();
                }
                ElementState::Released if self.dragging.is_some() => {
                    self.dragging = None;
                    response.consume_event();
                    self.raw_resizer_size = self.clamped_resizer_size;
                }
                _ => {}
            }
        }

        if hovering || self.dragging.is_some() {
            response.cursor_mode = CursorMode::Icon(CursorIcon::ColResize);

            // Really should be using TouchPhase here to not interrupt people
            // who started using these before we got hovered
            if matches!(event, Event::MouseWheel(_) | Event::PinchGesture(_)) {
                response.consume_event();
            }
        }

        let x = ctx.round(x);
        let left = Bounds {
            pos: bounds.pos,
            size: dvec2(x - bounds.left(), bounds.size.y),
        };
        let right = Bounds {
            pos: dvec2(x, bounds.pos.y),
            size: dvec2(bounds.right() - x, bounds.size.y),
        };

        if resized {
            self.expression_list.update(ctx, &Event::Resized, left);
            self.graph_paper.update(ctx, &Event::Resized, right);
        }

        response.or_else(|| {
            let (r_graph, dragged_point) = self.graph_paper.update(ctx, event, right);

            if let Some((i, p)) = dragged_point {
                self.expression_list.point_dragged(i, p);
            }

            let (r_expression_list, geometry) = self.expression_list.update(ctx, event, left);

            if let Some((geometry, vm_vars)) = geometry {
                self.graph_paper.set_geometry(geometry, vm_vars);
            }

            r_graph.or(r_expression_list)
        })
    }

    fn render(
        &mut self,
        ctx: &Context,
        graphics: &AppGraphics,
        view: &wgpu::TextureView,
        bounds: Bounds,
    ) -> Option<wgpu::CommandBuffer> {
        if bounds.is_empty() {
            return None;
        }

        let mut encoder = graphics.device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_thing_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        let x = ctx.round(bounds.left() + self.clamped_resizer_size);
        let left = Bounds {
            pos: bounds.pos,
            size: dvec2(x - bounds.left(), bounds.size.y),
        };
        let right = Bounds {
            pos: dvec2(x, bounds.pos.y),
            size: dvec2(bounds.right() - x, bounds.size.y),
        };
        // Render graph paper first because it does a fullscreen MSAA resolve
        // which would otherwise overwrite the expression list
        self.graph_paper
            .render(ctx, graphics, view, &mut encoder, right);

        let mut indices = vec![];
        let mut vertices = vec![];
        let draw_quad = &mut |quad: quad_renderer::Quad| {
            quad.into_triangles(ctx, &mut vertices, &mut indices);
        };

        self.graph_paper.render_buttons(ctx, right, draw_quad);
        self.expression_list.render(ctx, left, draw_quad);
        self.quad_renderer
            .render(ctx, graphics, view, &mut encoder, &vertices, &indices);
        Some(encoder.finish())
    }
}
