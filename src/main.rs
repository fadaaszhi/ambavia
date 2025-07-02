mod katex_font;
mod latex_editor;

use std::{f64, sync::Arc};

use ambavia::latex_parser::parse_latex;
use arboard::Clipboard;
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
}

impl Context {
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

        self.context.scale_factor = self.window.scale_factor();

        // Should this happen every event or only when CursorMoved?
        self.context.prev_cursor = self.context.cursor;

        if let WindowEvent::CursorMoved { position, .. } = &event {
            self.context.cursor = position.as_glam();
        }

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
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    dbg!((scale_factor, Event::ScaleFactorChanged)).1
                }
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
    ScaleFactorChanged,
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
    use std::{iter::zip, ops::Range};

    use bytemuck::{offset_of, Zeroable};
    use winit::keyboard::ModifiersState;

    use crate::latex_editor::{
        editor::{self, Node as ENode, Nodes as ENodes},
        layout::{self, Node as LNode, Nodes as LNodes},
    };

    use super::*;

    /// Specifies which structural component of a node to navigate to. Used for
    /// traversing into the different parts of an expression tree.
    #[derive(Debug, Clone, PartialEq)]
    enum NodeField {
        DelimitedGroup,
        SubSupSub,
        SubSupSup,
        SqrtRoot,
        SqrtArg,
        FracNum,
        FracDen,
        SumProdSub,
        SumProdSup,
    }

    use NodeField as Nf;

    /// Represents a path through an expression tree to reach a specific list of
    /// nodes. Each tuple contains the child index and which structural
    /// component to enter.
    type NodePath = Vec<(usize, NodeField)>;

    /// Represents a cursor position within an expression tree. The cursor sits
    /// between nodes.
    #[derive(Debug, Clone, PartialEq)]
    struct Cursor {
        /// The path through the expression tree to reach the target node list.
        path: NodePath,
        /// The position within the target node list. The cursor
        /// sits to the left of the `index`th node.
        index: usize,
    }

    impl From<(NodePath, usize)> for Cursor {
        fn from((path, index): (NodePath, usize)) -> Self {
            Self { path, index }
        }
    }

    /// Represents the extent of a user selection within a single node list. Can
    /// be either a cursor (zero-width) or a range of selected nodes.
    #[derive(Debug, PartialEq)]
    enum SelectionSpan {
        /// A cursor position between two nodes.
        Cursor(usize),
        /// A range of selected nodes, where `start < end`.
        Range(Range<usize>),
    }

    impl From<Range<usize>> for SelectionSpan {
        fn from(r: Range<usize>) -> Self {
            if r.is_empty() {
                Self::Cursor(r.start)
            } else {
                Self::Range(r)
            }
        }
    }

    impl SelectionSpan {
        fn as_range(&self) -> Range<usize> {
            match self {
                Self::Cursor(pos) => *pos..*pos,
                Self::Range(r) => r.clone(),
            }
        }
    }

    /// Represents the user's actual selection before normalization. The
    /// `anchor` and `focus` may be at different tree depths and in any order.
    struct UserSelection {
        /// Where the selection started (e.g., mouse down position or
        /// `Shift`+`Arrow` start)
        anchor: Cursor,
        /// Where the selection currently ends (e.g., current mouse position or
        /// `Shift`+`Arrow` end)
        focus: Cursor,
    }

    impl From<Cursor> for UserSelection {
        fn from(c: Cursor) -> Self {
            Self {
                anchor: c.clone(),
                focus: c,
            }
        }
    }

    /// Represents a normalized selection within the document tree. Unlike
    /// `UserSelection`, this is guaranteed to be within a single node list.
    struct Selection {
        /// The path to the node list containing the selection.
        path: NodePath,
        /// The selected range or cursor position within the node list.
        span: SelectionSpan,
    }

    impl UserSelection {
        fn normalize(&self) -> (NodePath, f64, f64) {
            let path: NodePath = zip(&self.anchor.path, &self.focus.path)
                .take_while(|(a, f)| a == f)
                .map(|(a, _)| a.clone())
                .collect();
            let anchor = self
                .anchor
                .path
                .get(path.len())
                .map_or(self.anchor.index as f64, |(i, _)| *i as f64 + 0.5);
            let focus = self
                .focus
                .path
                .get(path.len())
                .map_or(self.focus.index as f64, |(i, _)| *i as f64 + 0.5);
            (path, anchor, focus)
        }
    }

    impl From<&UserSelection> for Selection {
        fn from(s: &UserSelection) -> Self {
            let (path, anchor, focus) = s.normalize();
            let span =
                (anchor.min(focus).floor() as usize..anchor.max(focus).ceil() as usize).into();
            Self { path, span }
        }
    }

    trait Walkable {
        fn walk(self, path: &NodePath) -> Self;
    }

    impl Walkable for &mut ENodes {
        fn walk(self, path: &NodePath) -> Self {
            let mut nodes = self;
            for (index, field) in path {
                use ENode::*;
                nodes = match (&mut nodes[*index], field) {
                    (DelimitedGroup { inner, .. }, Nf::DelimitedGroup) => inner,
                    (SubSup { sub: Some(s), .. }, Nf::SubSupSub) => s,
                    (SubSup { sup: Some(s), .. }, Nf::SubSupSup) => s,
                    (Sqrt { root: Some(r), .. }, Nf::SqrtRoot) => r,
                    (Sqrt { arg, .. }, Nf::SqrtArg) => arg,
                    (Frac { num, .. }, Nf::FracNum) => num,
                    (Frac { den, .. }, Nf::FracDen) => den,
                    (SumProd { sub, .. }, Nf::SumProdSub) => sub,
                    (SumProd { sup, .. }, Nf::SumProdSup) => sup,
                    (node, field) => {
                        panic!("mismatched node/field:\n  node = {node:?}\n  field = {field:?}")
                    }
                };
            }
            nodes
        }
    }

    impl Walkable for &LNodes {
        fn walk(self, path: &NodePath) -> Self {
            let mut nodes = self;
            for (index, field) in path {
                use LNode::*;
                nodes = match (&nodes.nodes[*index].1, field) {
                    (DelimitedGroup { inner, .. }, Nf::DelimitedGroup) => inner,
                    (SubSup { sub: Some(s), .. }, Nf::SubSupSub) => s,
                    (SubSup { sup: Some(s), .. }, Nf::SubSupSup) => s,
                    (Sqrt { root: Some(r), .. }, Nf::SqrtRoot) => r,
                    (Sqrt { arg, .. }, Nf::SqrtArg) => arg,
                    (Frac { num, .. }, Nf::FracNum) => num,
                    (Frac { den, .. }, Nf::FracDen) => den,
                    (SumProd { sub, .. }, Nf::SumProdSub) => sub,
                    (SumProd { sup, .. }, Nf::SumProdSup) => sup,
                    (node, field) => {
                        panic!("mismatched node/field:\n  node = {node:?}\n  field = {field:?}")
                    }
                };
            }
            nodes
        }
    }

    struct Expression {
        latex: String,
        editor: ENodes,
        layout: LNodes,

        scale: f64,
        padding: f64,

        modifiers: ModifiersState,
        dragging: bool,
        selection: Option<UserSelection>,
    }

    impl Default for Expression {
        fn default() -> Self {
            Self {
                latex: Default::default(),
                editor: Default::default(),
                layout: Default::default(),
                scale: 20.0,
                padding: 16.0,
                modifiers: Default::default(),
                dragging: false,
                selection: None,
            }
        }
    }

    /// The message returned by an expression after it receives input.
    enum Message {
        /// Set the focus to the expression above this one.
        Up,
        /// Set the focus to the expression below this one.
        Down,
        /// Add an expression below this one.
        Add,
        /// Remove this expression.
        Remove,
    }

    fn get_hovered(mut nodes: &LNodes, mut path: NodePath, position: DVec2) -> Cursor {
        let index = 'index: loop {
            for (i, (bounds, node)) in nodes.nodes.iter().enumerate() {
                if position.x >= bounds.right() {
                    continue;
                }

                match node {
                    LNode::DelimitedGroup { .. } => todo!(),
                    LNode::SubSup { .. } => todo!(),
                    LNode::Sqrt { .. } => todo!(),
                    LNode::Frac { line, num, den } => {
                        if position.x < line.0.x {
                            break 'index i;
                        }
                        if position.x >= line.1.x {
                            break 'index i + 1;
                        }
                        nodes = if position.y < line.0.y {
                            path.push((i, Nf::FracNum));
                            num
                        } else {
                            path.push((i, Nf::FracDen));
                            den
                        };
                        continue 'index;
                    }
                    LNode::SumProd { .. } => todo!(),
                    LNode::Char(_) => {
                        if position.x < bounds.left() + 0.5 * bounds.width {
                            break 'index i;
                        }
                    }
                }
            }

            break nodes.nodes.len();
        };

        Cursor { path, index }
    }

    fn add_char(nodes: &mut ENodes, span: SelectionSpan, c: char) -> usize {
        let cursor = match span {
            SelectionSpan::Cursor(c) => c,
            SelectionSpan::Range(r) => {
                nodes.drain(r.clone());
                r.start
            }
        };
        nodes.insert(cursor, ENode::Char(c));
        let replacements = [
            ("cross", "×"),
            ("Gamma", "Γ"),
            ("Delta", "Δ"),
            ("Theta", "Θ"),
            ("Lambda", "Λ"),
            ("Xi", "Ξ"),
            ("Pi", "Π"),
            ("Sigma", "Σ"),
            ("Upsilon", "Υ"),
            ("Uψlon", "Υ"),
            ("Phi", "Φ"),
            ("Psi", "Ψ"),
            ("Omega", "Ω"),
            ("alpha", "α"),
            ("beta", "β"),
            ("gamma", "γ"),
            ("delta", "δ"),
            ("varepsilon", "ε"),
            ("vareψlon", "ε"),
            ("zeta", "ζ"),
            ("vartheta", "ϑ"),
            ("theta", "θ"),
            ("eta", "η"),
            ("iota", "ι"),
            ("kappa", "κ"),
            ("lambda", "λ"),
            ("mu", "μ"),
            ("nu", "ν"),
            ("xi", "ξ"),
            ("varpi", "ϖ"),
            ("pi", "π"),
            ("varrho", "ϱ"),
            ("rho", "ρ"),
            ("varsigma", "ς"),
            ("sigma", "σ"),
            ("tau", "τ"),
            ("upsilon", "υ"),
            ("uψlon", "υ"),
            ("varphi", "φ"),
            ("chi", "χ"),
            ("psi", "ψ"),
            ("omega", "ω"),
            ("phi", "ϕ"),
            ("epsilon", "ϵ"),
            ("eψlon", "ϵ"),
            ("->", "→"),
            ("infty", "∞"),
            ("infinity", "∞"),
            ("<=", "≤"),
            (">=", "≥"),
            ("*", "⋅"),
        ];
        for (find, replace) in replacements {
            let char_count = find.chars().count();
            if cursor + 1 >= char_count
                && find
                    .chars()
                    .rev()
                    .enumerate()
                    .all(|(i, c)| nodes[cursor - i] == ENode::Char(c))
            {
                nodes.splice(
                    cursor + 1 - char_count..cursor + 1,
                    replace.chars().map(ENode::Char),
                );
                return cursor + 1 + replace.chars().count() - char_count;
            }
        }
        cursor + 1
    }

    impl Expression {
        fn from_latex(latex: &str) -> Result<Self, ambavia::latex_parser::ParseError> {
            let editor = editor::convert(&parse_latex(latex)?);
            let layout = layout::layout(&editor);
            Ok(Self {
                latex: latex.into(),
                editor,
                layout,
                ..Default::default()
            })
        }

        fn unfocus(&mut self) -> Response {
            let mut response = Response::default();
            if self.selection.is_some() {
                self.selection = None;
                response.request_redraw();
            }
            response
        }

        fn focus(&mut self) -> Response {
            let mut response = Response::default();
            if self.selection.is_none() {
                self.selection = Some(
                    Cursor {
                        path: vec![],
                        index: self.editor.len(),
                    }
                    .into(),
                );
                response.request_redraw();
            }
            response
        }

        fn has_focus(&self) -> bool {
            self.selection.is_some()
        }

        fn size(&self, ctx: &Context) -> DVec2 {
            let b = self.layout.bounds;
            ctx.scale_factor
                * (dvec2(1.0, 2.0) * self.padding + self.scale * dvec2(b.width, b.height + b.depth))
        }

        fn editor_updated(&mut self) {
            self.latex = editor::to_latex(&self.editor);
            self.layout = layout::layout(&self.editor);
        }

        fn set_cursor(&mut self, cursor: impl Into<Cursor>) {
            self.selection = Some(cursor.into().into());
        }

        fn update(
            &mut self,
            ctx: &mut Context,
            event: &Event,
            bounds: Bounds,
        ) -> (Response, Option<Message>) {
            let mut response = Response::default();
            let mut message = None;

            let hovered = (bounds.contains(ctx.cursor) || self.dragging).then(|| {
                let position = (ctx.cursor
                    - (bounds.pos.as_dvec2() + ctx.scale_factor * self.padding))
                    / (ctx.scale_factor * self.scale)
                    - dvec2(0.0, self.layout.bounds.height);
                get_hovered(&self.layout, vec![], position)
            });

            if hovered.is_some() {
                response.cursor_icon = CursorIcon::Text;
            }

            match event {
                Event::KeyboardInput(KeyEvent {
                    logical_key,
                    state: ElementState::Pressed,
                    ..
                }) if self.selection.is_some() => {
                    use winit::keyboard::{Key, NamedKey};
                    use ENode::*;
                    let Selection { mut path, span } = self.selection.as_ref().unwrap().into();

                    match &logical_key {
                        Key::Named(NamedKey::Enter) => {
                            message = Some(Message::Add);
                            response.consume_event();
                        }
                        Key::Named(NamedKey::Space) => {
                            let index = add_char(self.editor.walk(&path), span, ' ');
                            self.set_cursor((path, index));
                            self.editor_updated();
                            response.request_redraw();
                            response.consume_event();
                        }
                        Key::Named(NamedKey::ArrowLeft) => {
                            if self.modifiers.shift_key() {
                                let s = self.selection.as_mut().unwrap();
                                let (mut path, anchor, focus) = s.normalize();
                                let index = if focus % 1.0 == 0.5 {
                                    let mut index = focus.floor() as usize;
                                    if focus <= anchor && index > 0 {
                                        index -= 1
                                    }
                                    index
                                } else if anchor + 0.5 == focus {
                                    path.push(s.anchor.path[path.len()].clone());
                                    self.editor.walk(&path).len()
                                } else if focus > 0.0 {
                                    focus as usize - 1
                                } else if let Some((index, _)) = path.pop() {
                                    index
                                } else {
                                    0
                                };
                                s.focus = Cursor { path, index };
                            } else {
                                match span {
                                    SelectionSpan::Cursor(mut i) => {
                                        if i > 0 {
                                            i -= 1;
                                            match &self.editor.walk(&path)[i] {
                                                DelimitedGroup { .. } => todo!(),
                                                SubSup { .. } => todo!(),
                                                Sqrt { .. } => todo!(),
                                                Frac { num, .. } => {
                                                    path.push((i, Nf::FracNum));
                                                    let index = num.len();
                                                    self.set_cursor((path, index));
                                                }
                                                SumProd { .. } => todo!(),
                                                Char(_) => self.set_cursor((path, i)),
                                            }
                                        } else if let Some((index, field)) = path.pop() {
                                            match field {
                                                Nf::DelimitedGroup => todo!(),
                                                Nf::SubSupSub => todo!(),
                                                Nf::SubSupSup => todo!(),
                                                Nf::SqrtRoot => todo!(),
                                                Nf::SqrtArg => todo!(),
                                                Nf::FracNum | Nf::FracDen => {
                                                    self.set_cursor((path, index));
                                                }
                                                Nf::SumProdSub => todo!(),
                                                Nf::SumProdSup => todo!(),
                                            }
                                        }
                                    }
                                    SelectionSpan::Range(r) => self.set_cursor((path, r.start)),
                                }
                            }
                            response.request_redraw();
                            response.consume_event();
                        }
                        Key::Named(NamedKey::ArrowRight) => {
                            if self.modifiers.shift_key() {
                                let s = self.selection.as_mut().unwrap();
                                let (mut path, anchor, focus) = s.normalize();
                                let index = if focus % 1.0 == 0.5 {
                                    let mut index = focus.ceil() as usize;
                                    if focus >= anchor && index < self.editor.walk(&path).len() {
                                        index += 1
                                    }
                                    index
                                } else if anchor - 0.5 == focus {
                                    path.push(s.anchor.path[path.len()].clone());
                                    0
                                } else {
                                    let index = focus as usize;
                                    if index < self.editor.walk(&path).len() {
                                        index + 1
                                    } else if let Some((index, _)) = path.pop() {
                                        index + 1
                                    } else {
                                        self.editor.len()
                                    }
                                };
                                s.focus = Cursor { path, index };
                            } else {
                                match span {
                                    SelectionSpan::Cursor(i) => {
                                        let nodes = self.editor.walk(&path);
                                        if i < nodes.len() {
                                            match &nodes[i] {
                                                DelimitedGroup { .. } => todo!(),
                                                SubSup { .. } => todo!(),
                                                Sqrt { .. } => todo!(),
                                                Frac { .. } => {
                                                    path.push((i, Nf::FracNum));
                                                    self.set_cursor((path, 0));
                                                }
                                                SumProd { .. } => todo!(),
                                                Char(_) => self.set_cursor((path, i + 1)),
                                            }
                                        } else if let Some((index, field)) = path.pop() {
                                            match field {
                                                Nf::DelimitedGroup => todo!(),
                                                Nf::SubSupSub => todo!(),
                                                Nf::SubSupSup => todo!(),
                                                Nf::SqrtRoot => todo!(),
                                                Nf::SqrtArg => todo!(),
                                                Nf::FracNum | Nf::FracDen => {
                                                    self.set_cursor((path, index + 1));
                                                }
                                                Nf::SumProdSub => todo!(),
                                                Nf::SumProdSup => todo!(),
                                            }
                                        }
                                    }
                                    SelectionSpan::Range(r) => self.set_cursor((path, r.end)),
                                }
                            }
                            response.consume_event();
                            response.request_redraw();
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            if self.modifiers.shift_key() {
                                let s = self.selection.as_mut().unwrap();
                                let (mut path, anchor, focus) = s.normalize();
                                let index = if focus > anchor
                                    || anchor % 1.0 == 0.0
                                    || focus == anchor && focus % 1.0 == 0.0
                                {
                                    let index = self.editor.walk(&path).len();
                                    if focus.ceil() < index as f64 || path.is_empty() {
                                        index
                                    } else {
                                        path.pop().unwrap().0 + 1
                                    }
                                } else if focus + 0.5 == anchor {
                                    path.push(s.anchor.path[path.len()].clone());
                                    0
                                } else {
                                    anchor.floor() as usize
                                };
                                s.focus = Cursor { path, index };
                            } else {
                                let i = span.as_range().end;

                                'stuff: {
                                    let nodes = self.editor.walk(&path);
                                    if i < nodes.len() {
                                        match nodes[i] {
                                            DelimitedGroup { .. } => todo!(),
                                            SubSup { .. } => todo!(),
                                            Sqrt { .. } => todo!(),
                                            Frac { .. } => {
                                                path.push((i, Nf::FracDen));
                                                self.set_cursor((path, 0));
                                                response.request_redraw();
                                                break 'stuff;
                                            }
                                            SumProd { .. } => todo!(),
                                            Char(_) => {}
                                        }
                                    }

                                    if i > 0 {
                                        match &nodes[i - 1] {
                                            DelimitedGroup { .. } => todo!(),
                                            SubSup { .. } => todo!(),
                                            Sqrt { .. } => todo!(),
                                            Frac { den, .. } => {
                                                path.push((i - 1, Nf::FracDen));
                                                let index = den.len();
                                                self.set_cursor((path, index));
                                                response.request_redraw();
                                                break 'stuff;
                                            }
                                            SumProd { .. } => todo!(),
                                            Char(_) => {}
                                        }
                                    }

                                    let nodes = self.layout.walk(&path);
                                    let x = nodes
                                        .nodes
                                        .get(i)
                                        .map_or(nodes.bounds.right(), |(b, _)| b.left());

                                    loop {
                                        if let Some((index, field)) = path.pop() {
                                            match field {
                                                Nf::DelimitedGroup => todo!(),
                                                Nf::SubSupSub => todo!(),
                                                Nf::SubSupSup => todo!(),
                                                Nf::SqrtRoot => todo!(),
                                                Nf::SqrtArg => todo!(),
                                                Nf::FracNum => {
                                                    path.push((index, Nf::FracDen));
                                                    break;
                                                }
                                                Nf::FracDen => {}
                                                Nf::SumProdSub => todo!(),
                                                Nf::SumProdSup => todo!(),
                                            }
                                        } else {
                                            message = Some(Message::Down);
                                            break 'stuff;
                                        }
                                    }

                                    self.set_cursor(get_hovered(
                                        self.layout.walk(&path),
                                        path,
                                        dvec2(x, -f64::INFINITY),
                                    ));
                                }
                            }
                            response.request_redraw();
                            response.consume_event();
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            if self.modifiers.shift_key() {
                                let s = self.selection.as_mut().unwrap();
                                let (mut path, anchor, focus) = s.normalize();
                                let index = if focus < anchor
                                    || anchor % 1.0 == 0.0
                                    || focus == anchor && focus % 1.0 == 0.0
                                {
                                    let index = focus.floor() as usize;
                                    if index > 0 || path.is_empty() {
                                        0
                                    } else {
                                        path.pop().unwrap().0
                                    }
                                } else if focus - 0.5 == anchor {
                                    path.push(s.anchor.path[path.len()].clone());
                                    self.editor.walk(&path).len()
                                } else {
                                    anchor.ceil() as usize
                                };
                                s.focus = Cursor { path, index };
                            } else {
                                let i = span.as_range().end;

                                'stuff: {
                                    let nodes = self.editor.walk(&path);
                                    if i < nodes.len() {
                                        match nodes[i] {
                                            DelimitedGroup { .. } => todo!(),
                                            SubSup { .. } => todo!(),
                                            Sqrt { .. } => todo!(),
                                            Frac { .. } => {
                                                path.push((i, Nf::FracNum));
                                                self.set_cursor((path, 0));
                                                response.request_redraw();
                                                break 'stuff;
                                            }
                                            SumProd { .. } => todo!(),
                                            Char(_) => {}
                                        }
                                    }

                                    if i > 0 {
                                        match &nodes[i - 1] {
                                            DelimitedGroup { .. } => todo!(),
                                            SubSup { .. } => todo!(),
                                            Sqrt { .. } => todo!(),
                                            Frac { num, .. } => {
                                                path.push((i - 1, Nf::FracNum));
                                                let index = num.len();
                                                self.set_cursor((path, index));
                                                response.request_redraw();
                                                break 'stuff;
                                            }
                                            SumProd { .. } => todo!(),
                                            Char(_) => {}
                                        }
                                    }

                                    let nodes = self.layout.walk(&path);
                                    let x = nodes
                                        .nodes
                                        .get(i)
                                        .map_or(nodes.bounds.right(), |(b, _)| b.left());

                                    loop {
                                        if let Some((index, field)) = path.pop() {
                                            match field {
                                                Nf::DelimitedGroup => todo!(),
                                                Nf::SubSupSub => todo!(),
                                                Nf::SubSupSup => todo!(),
                                                Nf::SqrtRoot => todo!(),
                                                Nf::SqrtArg => todo!(),
                                                Nf::FracNum => {}
                                                Nf::FracDen => {
                                                    path.push((index, Nf::FracNum));
                                                    break;
                                                }
                                                Nf::SumProdSub => todo!(),
                                                Nf::SumProdSup => todo!(),
                                            }
                                        } else {
                                            message = Some(Message::Up);
                                            break 'stuff;
                                        }
                                    }

                                    self.set_cursor(get_hovered(
                                        self.layout.walk(&path),
                                        path,
                                        dvec2(x, f64::INFINITY),
                                    ));
                                }
                            }
                            response.request_redraw();
                            response.consume_event();
                        }
                        Key::Named(NamedKey::Backspace) => {
                            match span {
                                SelectionSpan::Cursor(mut i) => {
                                    let nodes = self.editor.walk(&path);
                                    if i > 0 {
                                        i -= 1;
                                        match &nodes[i] {
                                            DelimitedGroup { .. } => todo!(),
                                            SubSup { .. } => todo!(),
                                            Sqrt { .. } => todo!(),
                                            Frac { den, .. } => {
                                                path.push((i, Nf::FracDen));
                                                let index = den.len();
                                                self.set_cursor((path, index));
                                            }
                                            SumProd { .. } => todo!(),
                                            Char(_) => {
                                                nodes.remove(i);
                                                self.editor_updated();
                                                self.set_cursor((path, i));
                                            }
                                        }
                                    } else if let Some((index, field)) = path.pop() {
                                        let nodes = self.editor.walk(&path);
                                        match field {
                                            Nf::DelimitedGroup => todo!(),
                                            Nf::SubSupSub => todo!(),
                                            Nf::SubSupSup => todo!(),
                                            Nf::SqrtRoot => todo!(),
                                            Nf::SqrtArg => todo!(),
                                            Nf::FracNum | Nf::FracDen => {
                                                let Frac { num, den } = nodes.remove(index) else {
                                                    unreachable!()
                                                };
                                                let i = if field == Nf::FracNum {
                                                    index
                                                } else {
                                                    index + num.len()
                                                };
                                                nodes.splice(
                                                    index..index,
                                                    num.into_iter().chain(den),
                                                );
                                                self.editor_updated();
                                                self.set_cursor((path, i));
                                            }
                                            Nf::SumProdSub => todo!(),
                                            Nf::SumProdSup => todo!(),
                                        }
                                    } else if self.editor.is_empty() {
                                        message = Some(Message::Remove);
                                    }
                                }
                                SelectionSpan::Range(r) => {
                                    self.editor.walk(&path).drain(r.clone());
                                    self.editor_updated();
                                    self.set_cursor((path, r.start));
                                }
                            }

                            response.consume_event();
                            response.request_redraw();
                        }
                        Key::Named(NamedKey::Delete) => {
                            match span {
                                SelectionSpan::Cursor(i) => {
                                    let nodes = self.editor.walk(&path);
                                    if i < nodes.len() {
                                        match &nodes[i] {
                                            DelimitedGroup { .. } => todo!(),
                                            SubSup { .. } => todo!(),
                                            Sqrt { .. } => todo!(),
                                            Frac { .. } => {
                                                path.push((i, Nf::FracNum));
                                                self.set_cursor((path, 0));
                                            }
                                            SumProd { .. } => todo!(),
                                            Char(_) => {
                                                nodes.remove(i);
                                                self.editor_updated();
                                                self.set_cursor((path, i));
                                            }
                                        }
                                    } else if let Some((index, field)) = path.pop() {
                                        let nodes = self.editor.walk(&path);
                                        match field {
                                            Nf::DelimitedGroup => todo!(),
                                            Nf::SubSupSub => todo!(),
                                            Nf::SubSupSup => todo!(),
                                            Nf::SqrtRoot => todo!(),
                                            Nf::SqrtArg => todo!(),
                                            Nf::FracNum | Nf::FracDen => {
                                                let Frac { num, den } = nodes.remove(index) else {
                                                    unreachable!()
                                                };
                                                let i = if field == Nf::FracNum {
                                                    index + num.len()
                                                } else {
                                                    index + num.len() + den.len()
                                                };
                                                nodes.splice(
                                                    index..index,
                                                    num.into_iter().chain(den),
                                                );
                                                self.editor_updated();
                                                self.set_cursor((path, i));
                                            }
                                            Nf::SumProdSub => todo!(),
                                            Nf::SumProdSup => todo!(),
                                        }
                                    }
                                }
                                SelectionSpan::Range(r) => {
                                    self.editor.walk(&path).drain(r.clone());
                                    self.editor_updated();
                                    self.set_cursor((path, r.start));
                                }
                            }
                            response.consume_event();
                            response.request_redraw();
                        }
                        Key::Character(c) => match c.as_str().chars().next() {
                            Some('a')
                                if self.modifiers.control_key() || self.modifiers.super_key() =>
                            {
                                self.selection = Some(UserSelection {
                                    anchor: Cursor {
                                        path: vec![],
                                        index: 0,
                                    },
                                    focus: Cursor {
                                        path: vec![],
                                        index: self.editor.len(),
                                    },
                                });
                                response.consume_event();
                                response.request_redraw();
                            }
                            Some('c')
                                if self.modifiers.control_key() || self.modifiers.super_key() =>
                            {
                                if let SelectionSpan::Range(r) = span {
                                    let latex = editor::to_latex(&self.editor.walk(&path)[r]);
                                    if let Err(e) = ctx.clipboard.set_text(latex) {
                                        eprintln!("failed to set clipboard contents: {e}");
                                    }
                                }
                                response.consume_event();
                            }
                            Some('x')
                                if self.modifiers.control_key() || self.modifiers.super_key() =>
                            {
                                if let SelectionSpan::Range(r) = span {
                                    let nodes = self.editor.walk(&path);
                                    let latex = editor::to_latex(&nodes[r.clone()]);
                                    if let Err(e) = ctx.clipboard.set_text(latex) {
                                        eprintln!("failed to set clipboard contents: {e}");
                                    } else {
                                        nodes.drain(r.clone());
                                        self.editor_updated();
                                        self.set_cursor((path, r.start));
                                        response.request_redraw();
                                    }
                                }
                                response.consume_event();
                            }
                            Some('v')
                                if self.modifiers.control_key() || self.modifiers.super_key() =>
                            {
                                let latex = ctx.clipboard.get_text().unwrap_or_default();

                                match parse_latex(&latex) {
                                    Ok(tree) => {
                                        let nodes = editor::convert(&tree);
                                        let r = span.as_range();
                                        let pasted_len = nodes.len();
                                        self.editor.walk(&path).splice(r.clone(), nodes);
                                        self.editor_updated();
                                        self.set_cursor((path, r.start + pasted_len));
                                        response.request_redraw();
                                    }
                                    Err(e) => eprintln!("parse_latex error: {e:?}"),
                                }
                                response.consume_event();
                            }
                            Some('/') => {
                                let nodes = self.editor.walk(&path);
                                let r = match span {
                                    SelectionSpan::Cursor(c) => {
                                        nodes[0..c]
                                            .iter()
                                            .enumerate()
                                            .rev()
                                            .find_map(|(i, n)| {
                                                matches!(
                                                    n,
                                                    SumProd { .. }
                                                        | Char(
                                                            '+' | '-'
                                                                | '*'
                                                                | '='
                                                                | '<'
                                                                | '>'
                                                                | ','
                                                                | ':'
                                                                | '×'
                                                                | '→'
                                                                | '≤'
                                                                | '≥'
                                                                | '⋅'
                                                        )
                                                )
                                                .then(|| i + 1)
                                            })
                                            .unwrap_or(0)..c
                                    }
                                    SelectionSpan::Range(r) => r,
                                };

                                let num = nodes.drain(r.clone()).collect();
                                nodes.insert(r.start, Frac { num, den: vec![] });
                                self.editor_updated();
                                path.push((
                                    r.start,
                                    if r.is_empty() {
                                        Nf::FracNum
                                    } else {
                                        Nf::FracDen
                                    },
                                ));
                                self.set_cursor((path, 0));
                                response.consume_event();
                                response.request_redraw();
                            }
                            Some(
                                c @ ('0'..='9'
                                | 'A'..='Z'
                                | 'a'..='z'
                                | '.'
                                | '+'
                                | '-'
                                | '*'
                                | '='
                                | '<'
                                | '>'
                                | ','
                                | ':'
                                | '!'
                                | '%'
                                | '\''),
                            ) => {
                                let index = add_char(self.editor.walk(&path), span, c);
                                self.set_cursor((path, index));
                                self.editor_updated();
                                response.request_redraw();
                                response.consume_event();
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                Event::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
                Event::CursorMoved if self.dragging => {
                    if let Some(hovered) = hovered {
                        self.selection.as_mut().unwrap().focus = hovered;
                    } else {
                        eprintln!("how did we get here?");
                    }
                    response.consume_event();
                    response.request_redraw();
                }
                Event::MouseInput(ElementState::Pressed, MouseButton::Left) => {
                    if let Some(hovered) = hovered {
                        self.dragging = true;
                        self.selection = Some(hovered.into());
                        response.consume_event();
                        response.request_redraw();
                    } else if self.selection.is_some() {
                        self.selection = None;
                        response.request_redraw();
                    }
                }
                Event::MouseInput(ElementState::Released, MouseButton::Left) if self.dragging => {
                    self.dragging = false;
                    response.consume_event();
                }
                _ => {}
            }

            (response, message)
        }

        fn render(
            &self,
            ctx: &Context,
            bounds: Bounds,
            draw_quad: &mut impl FnMut(DVec2, DVec2, DVec2, DVec2),
        ) {
            let transform = &|p| {
                bounds.pos.as_dvec2()
                    + ctx.scale_factor
                        * (self.padding + self.scale * (p + dvec2(0.0, self.layout.bounds.height)))
            };
            if let Some(selection) = &self.selection {
                draw_selection(ctx, &self.layout, selection.into(), transform, draw_quad);
            }
            draw_latex(ctx, &self.layout, transform, draw_quad);
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

    fn draw_selection(
        ctx: &Context,
        nodes: &LNodes,
        selection: Selection,
        transform: &impl Fn(DVec2) -> DVec2,
        draw_quad: &mut impl FnMut(DVec2, DVec2, DVec2, DVec2),
    ) {
        let nodes = nodes.walk(&selection.path);
        match selection.span {
            SelectionSpan::Cursor(index) => {
                let position = nodes.nodes.get(index).map_or(
                    nodes.bounds.position + dvec2(nodes.bounds.width, 0.0),
                    |(b, _)| b.position,
                );
                let b = layout::Bounds::default();
                let p0 = transform(position - dvec2(0.0, nodes.bounds.scale * b.height));
                let p1 = transform(position + dvec2(0.0, nodes.bounds.scale * b.depth));
                let w = ctx.scale_width(1.0);
                let x = snap(p0.x, w);
                let p0 = dvec2(x - w as f64 / 2.0, p0.y.floor());
                let p1 = dvec2(x + w as f64 / 2.0, p1.y.floor() + 1.0);
                let uv = DVec2::splat(-1.0);
                draw_quad(p0, p1, uv, uv);
            }
            SelectionSpan::Range(r) => {
                for (b, _) in &nodes.nodes[r] {
                    let p0 = transform(b.top_left()).floor();
                    let p1 = transform(b.bottom_right()).ceil();
                    let uv = DVec2::splat(-2.0);
                    draw_quad(p0, p1, uv, uv);
                }
            }
        }
    }

    fn draw_latex(
        ctx: &Context,
        nodes: &LNodes,
        transform: &impl Fn(DVec2) -> DVec2,
        draw_quad: &mut impl FnMut(DVec2, DVec2, DVec2, DVec2),
    ) {
        for (_, node) in &nodes.nodes {
            match node {
                LNode::DelimitedGroup { .. } => todo!(),
                LNode::SubSup { .. } => todo!(),
                LNode::Sqrt { .. } => todo!(),
                LNode::Frac { line, num, den } => {
                    let l0 = transform(line.0);
                    let l1 = transform(line.1);
                    let w = ctx.scale_width(1.0);
                    let y = snap(l0.y, w);
                    draw_quad(
                        dvec2(l0.x.floor(), y - w as f64 / 2.0),
                        dvec2(l1.x.ceil(), y + w as f64 / 2.0),
                        DVec2::splat(-1.0),
                        DVec2::splat(-1.0),
                    );
                    draw_latex(ctx, num, transform, draw_quad);
                    draw_latex(ctx, den, transform, draw_quad);
                }
                LNode::SumProd { .. } => todo!(),
                LNode::Char(g) => {
                    let p0 = transform(dvec2(g.plane.left, g.plane.top));
                    let p1 = transform(dvec2(g.plane.right, g.plane.bottom));
                    let uv0 = dvec2(g.atlas.left, g.atlas.top);
                    let uv1 = dvec2(g.atlas.right, g.atlas.bottom);
                    draw_quad(p0, p1, uv0, uv1);
                }
            }
        }
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

            Self {
                expressions: vec![Expression::from_latex(r"\frac{a}{\frac{b}{c}+\alpha\operatorname{count}\frac{\frac{4}{7}}{\frac{4}{\frac{4}{4}}}}+\frac{\sin\frac{5}{6}}{6}").unwrap()],

                pipeline,
                vertex_buffer,
                index_buffer,
                uniforms_buffer,
                bind_group,
            }
        }

        const SEPARATOR_WIDTH: f64 = 1.0;

        pub fn update(&mut self, ctx: &mut Context, event: &Event, bounds: Bounds) -> Response {
            let mut response = Response::default();
            let mut y_offset = 0;
            let mut message = None;

            for (i, expression) in self.expressions.iter_mut().enumerate() {
                let y_size = expression.size(ctx).y.ceil() as u32;
                let (r, m) = expression.update(
                    ctx,
                    event,
                    bounds.intersect(&Bounds {
                        pos: bounds.pos + uvec2(0, y_offset),
                        size: uvec2(bounds.size.x, y_size),
                    }),
                );

                if let Some(m) = m {
                    message = Some((i, m));
                }

                response = response.or(r);
                y_offset += y_size + ctx.scale_width(Self::SEPARATOR_WIDTH);
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
                            self.expressions.push(Expression::default());
                        }

                        response = response
                            .or(self.expressions[i].unfocus())
                            .or(self.expressions[i + 1].focus());
                        response.request_redraw();
                    }
                    Message::Add => {
                        self.expressions.insert(i + 1, Expression::default());
                        response = response
                            .or(self.expressions[i].unfocus())
                            .or(self.expressions[i + 1].focus());
                        response.request_redraw();
                    }
                    Message::Remove => {
                        response = response.or(self.expressions[i].unfocus());
                        self.expressions.remove(i);

                        if self.expressions.is_empty() {
                            self.expressions.push(Expression::default());
                        }

                        response = response.or(self.expressions[i.saturating_sub(1)].focus());
                        response.request_redraw();
                    }
                }
            }

            if self.expressions.last().unwrap().has_focus() {
                self.expressions.push(Expression::default());
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

            for expression in &mut self.expressions {
                let y_size = expression.size(ctx).y.ceil() as u32;
                expression.render(
                    ctx,
                    bounds.intersect(&Bounds {
                        pos: bounds.pos + uvec2(0, y_offset),
                        size: uvec2(bounds.size.x, y_size),
                    }),
                    draw_quad,
                );
                y_offset += y_size;

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
