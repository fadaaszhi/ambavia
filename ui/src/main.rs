mod expression_list;
mod graph;
mod katex_font;
mod math_field;
mod ui;
mod utility;

use std::{f64, sync::Arc};

use glam::{DVec2, UVec2, dvec2, vec2};
use winit::{
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{CursorIcon, Window, WindowAttributes, WindowId},
};

use crate::{
    ui::{Bounds, Context, Event, Response},
    utility::{AsGlam, mix, unmix},
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
    events: Vec<WindowEvent>,
    request_redraw: bool,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    context: Context,
    main_thing: MainThing,
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
        let context = Context::new(&window);
        let main_thing = MainThing::new(&device, &queue, &config);

        App {
            events: vec![],
            request_redraw: false,
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
        window_id: WindowId,
        event: WindowEvent,
    ) {
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
                self.events.push(event);
                return;
            }

            self.request_redraw = false;
            for event in std::mem::take(&mut self.events).drain(..) {
                self.fake_window_event(event_loop, window_id, event);
            }
            if self.request_redraw {
                self.fake_window_event(event_loop, window_id, WindowEvent::RedrawRequested);
            }
        } else {
            self.request_redraw = false;
            self.fake_window_event(event_loop, window_id, event);
            if self.request_redraw {
                self.window.request_redraw();
            }
        }
    }

    fn fake_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let previous_cursor = self.context.cursor;
        self.context.update(&event);
        let bounds = Bounds {
            pos: DVec2::ZERO,
            size: self.window.inner_size().as_glam().as_dvec2() / self.context.scale_factor,
        };

        'update: {
            let my_event = match event.clone() {
                WindowEvent::Resized { .. } => Event::Resized,
                WindowEvent::KeyboardInput { event, .. } => Event::KeyboardInput(event),
                WindowEvent::CursorMoved { .. } => Event::CursorMoved { previous_cursor },
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
            let response = self.main_thing.update(&self.context, &my_event, bounds);
            if response.requested_redraw {
                self.request_redraw = true;
                // self.window.request_redraw();
            }
            self.window.set_cursor(response.cursor_icon);
        }

        match event {
            WindowEvent::Resized(new_size) => {
                self.config.width = new_size.width.max(1);
                self.config.height = new_size.height.max(1);
                self.surface.configure(&self.device, &self.config);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.window.request_redraw();
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

    fn update(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> Response {
        let mut response = Response::default();

        let mut x = mix(bounds.left(), bounds.right(), self.resizer_position);
        let resized = if let Event::CursorMoved { .. } = event
            && let Some(offset) = self.dragging
        {
            x = (ctx.cursor.x + offset).clamp(bounds.left(), bounds.right());
            self.resizer_position = unmix(x, bounds.left(), bounds.right());
            response.consume_event();
            response.request_redraw();
            true
        } else {
            false
        };

        let offset = x - ctx.cursor.x;
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

            let (r_expression_list, points) = self.expression_list.update(ctx, event, left);

            if let Some(points) = points {
                self.graph_paper.set_geometry(points);
            }

            r_graph.or(r_expression_list)
        })
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

        let x = ctx.round(mix(bounds.left(), bounds.right(), self.resizer_position));
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
            .render(ctx, device, queue, view, config, &mut encoder, right);
        self.expression_list
            .render(ctx, device, queue, view, config, &mut encoder, left);
        Some(encoder.finish())
    }
}
