use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

use arboard::Clipboard;
use glam::{DVec2, DVec4, dvec2, dvec4};
use winit::{
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::ModifiersState,
    window::{CursorIcon, Window},
};

use crate::utility::{AsGlam, mix};

pub struct Context {
    clipboard: Arc<Mutex<Option<Clipboard>>>,
    /// The number of seconds elapsed since the application started
    pub time: f64,
    /// The cursor's current logical position
    pub cursor: DVec2,
    /// The window's scale factor
    pub scale_factor: f64,
    pub modifiers: ModifiersState,
}

impl Context {
    pub fn new(window: &Window) -> Self {
        Self {
            clipboard: Arc::new(Mutex::new(None)),
            time: 0.0,
            cursor: DVec2::ZERO,
            scale_factor: window.scale_factor(),
            modifiers: Default::default(),
        }
    }

    pub fn update(&mut self, event: &WindowEvent, time: f64) {
        self.time = time;

        match &event {
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = position.as_glam() / self.scale_factor
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = *scale_factor
            }
            _ => {}
        }
    }

    pub fn clipboard<T, F>(&self, f: F) -> Result<T, arboard::Error>
    where
        F: FnOnce(&mut Clipboard) -> T,
    {
        let mut clipboard = self.clipboard.lock().unwrap();
        Ok(f(match clipboard.as_mut() {
            Some(clipboard) => clipboard,
            None => clipboard.insert(Clipboard::new()?),
        }))
    }

    pub fn get_clipboard_text(&self) -> Result<String, arboard::Error> {
        self.clipboard(|c| c.get_text())?
    }

    pub fn set_clipboard_text<'a, T>(&self, text: T) -> Result<(), arboard::Error>
    where
        T: Into<Cow<'a, str>>,
    {
        self.clipboard(|c| c.set_text(text))?
    }

    /// Round a logical value to an integer physical value, returning a logical
    /// value.
    pub fn round(&self, x: f64) -> f64 {
        (x * self.scale_factor).round() / self.scale_factor
    }

    /// Floor a logical value to an integer physical value, returning a logical
    /// value.
    pub fn floor(&self, x: f64) -> f64 {
        (x * self.scale_factor).floor() / self.scale_factor
    }

    /// Ceil a logical value to an integer physical value, returning a logical
    /// value.
    pub fn ceil(&self, x: f64) -> f64 {
        (x * self.scale_factor).ceil() / self.scale_factor
    }

    /// Round a logical value to an integer physical value greater than 0,
    /// returning a logical value.
    pub fn round_nonzero(&self, x: f64) -> f64 {
        (x * self.scale_factor).round().max(1.0) / self.scale_factor
    }

    /// Round a logical value to an integer physical value greater than 0,
    /// returning a physical value.
    pub fn round_nonzero_as_physical(&self, x: f64) -> u32 {
        (x * self.scale_factor).round().max(1.0) as u32
    }

    pub fn to_physical(&self, logical: Bounds) -> Bounds {
        Bounds {
            pos: logical.pos * self.scale_factor,
            size: logical.size * self.scale_factor,
        }
    }

    pub fn set_scissor_rect(&self, pass: &mut wgpu::RenderPass, logical: Bounds) {
        let b = self.to_physical(logical);
        let q = b.pos.max(DVec2::ZERO).round();
        let s = ((b.size + b.pos).round() - q).max(DVec2::ZERO).round();
        pass.set_scissor_rect(q.x as u32, q.y as u32, s.x as u32, s.y as u32);
    }
}

#[derive(Debug, PartialEq)]
pub enum Event {
    Resized,
    /// Sent once right before every redraw. If you request a redraw during this
    /// event, it will trigger an additional redraw next frame.
    AnimationFrame,
    KeyboardInput(KeyEvent),
    CursorMoved {
        /// The cursor's previous logical position
        previous_cursor: DVec2,
    },
    MouseWheel(DVec2),
    MouseInput(ElementState, MouseButton),
    PinchGesture(f64),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Bounds {
    pub pos: DVec2,
    pub size: DVec2,
}

impl Bounds {
    pub fn left(&self) -> f64 {
        self.pos.x
    }

    pub fn right(&self) -> f64 {
        self.pos.x + self.size.x
    }

    pub fn top(&self) -> f64 {
        self.pos.y
    }

    pub fn bottom(&self) -> f64 {
        self.pos.y + self.size.y
    }

    pub fn is_empty(&self) -> bool {
        self.size.x <= 0.0 || self.size.y <= 0.0
    }

    pub fn contains(&self, position: DVec2) -> bool {
        (self.left() <= position.x && position.x < self.right())
            && (self.top() <= position.y && position.y < self.bottom())
    }

    pub fn union(self, other: Bounds) -> Bounds {
        let pos = self.pos.min(other.pos);
        Bounds {
            pos,
            size: (self.pos + self.size).max(other.pos + other.size) - pos,
        }
    }
}

#[derive(Debug, Default, PartialEq)]
pub enum CursorMode {
    #[default]
    NoPreference,
    Hidden,
    Icon(CursorIcon),
}

#[derive(Debug, Default)]
pub struct Response {
    pub consumed_event: bool,
    pub requested_redraw: bool,
    pub cursor_mode: CursorMode,
}

impl Response {
    pub fn consume_event(&mut self) {
        self.consumed_event = true;
    }

    pub fn request_redraw(&mut self) {
        self.requested_redraw = true;
    }

    /// Use when combining responses from elements that aren't nested, that way
    /// other elements still have the chance to unfocus themselves when one
    /// consumes a mouse down event.
    #[must_use]
    pub fn or(self, other: Response) -> Response {
        Response {
            consumed_event: self.consumed_event | other.consumed_event,
            requested_redraw: self.requested_redraw | other.requested_redraw,
            cursor_mode: if self.consumed_event
                || !other.consumed_event && self.cursor_mode != CursorMode::NoPreference
            {
                self.cursor_mode
            } else {
                other.cursor_mode
            },
        }
    }

    /// Use when combining responses from nested elements. Or don't, idk
    #[must_use]
    pub fn or_else(self, other: impl FnOnce() -> Response) -> Response {
        if self.consumed_event {
            self
        } else {
            self.or(other())
        }
    }
}

#[derive(PartialEq)]
pub enum ClickOrDrag {
    None,
    Clicked,
    Dragged,
}

impl ClickOrDrag {
    pub fn was_clicked(self) -> bool {
        self == ClickOrDrag::Clicked
    }

    pub fn was_dragged(self) -> bool {
        self == ClickOrDrag::Dragged
    }
}

/// Tracks whether an interaction with an object was a click or a drag.
#[derive(Default)]
pub enum ClickDragTracker {
    #[default]
    None,
    Pressed(DVec2),
    Dragging,
}

impl ClickDragTracker {
    const DRAG_THRESHOLD: f64 = 2.0;

    pub fn is_pressed(&self) -> bool {
        matches!(self, ClickDragTracker::Pressed(_))
    }

    pub fn is_dragging(&self) -> bool {
        matches!(self, ClickDragTracker::Dragging)
    }

    /// Call on mouse down and pass it the current cursor position
    pub fn press(&mut self, cursor: DVec2) {
        *self = ClickDragTracker::Pressed(cursor);
    }

    /// Call on mouse move and pass it the new cursor position.
    /// Returns `true` if the user is dragging.
    pub fn drag(&mut self, cursor: DVec2) -> bool {
        if let ClickDragTracker::Pressed(start) = self
            && start.distance(cursor) >= Self::DRAG_THRESHOLD
        {
            *self = ClickDragTracker::Dragging;
        }
        self.is_dragging()
    }

    /// Call on mouse up. Returns what type of interaction it was.
    pub fn release(&mut self) -> ClickOrDrag {
        match std::mem::take(self) {
            ClickDragTracker::None => ClickOrDrag::None,
            ClickDragTracker::Pressed(_) => ClickOrDrag::Clicked,
            ClickDragTracker::Dragging => ClickOrDrag::Dragged,
        }
    }
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

pub const PRIMARY_COLOR: [u8; 3] = [47, 114, 220];

pub trait Color {
    fn to_rgbaf64(self) -> DVec4;

    fn with_opacity(self, opacity: f64) -> DVec4
    where
        Self: Sized,
    {
        let mut color = self.to_rgbaf64();
        color.w *= opacity;
        color
    }
}

impl Color for DVec4 {
    fn to_rgbaf64(self) -> DVec4 {
        self
    }
}

impl Color for (f64, f64, f64, f64) {
    fn to_rgbaf64(self) -> DVec4 {
        self.into()
    }
}

impl Color for [f64; 3] {
    fn to_rgbaf64(self) -> DVec4 {
        dvec4(self[0], self[1], self[2], 1.0)
    }
}

impl Color for (f64, f64, f64) {
    fn to_rgbaf64(self) -> DVec4 {
        dvec4(self.0, self.1, self.2, 1.0)
    }
}

impl Color for (u8, u8, u8, f64) {
    fn to_rgbaf64(self) -> DVec4 {
        dvec4(
            self.0 as f64 / 255.0,
            self.1 as f64 / 255.0,
            self.2 as f64 / 255.0,
            self.3,
        )
    }
}

impl Color for (u8, u8, u8) {
    fn to_rgbaf64(self) -> DVec4 {
        (self.0, self.1, self.2, 1.0).to_rgbaf64()
    }
}

impl Color for [u8; 3] {
    fn to_rgbaf64(self) -> DVec4 {
        (self[0], self[1], self[2]).to_rgbaf64()
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
}

pub enum QuadKind {
    Rectangle,
    Pill,
    MsdfGlyph,
    AlphaGradientU,
    AlphaGradientV2,
    OutputValueBox,
    SliderPausedButton,
    SliderPlayingButton,
}

pub struct AnimatedValue {
    duration: f64,
    start_time: f64,
    target: f64,
    a: f64,
    b: f64,
    n: i32,
}

impl AnimatedValue {
    pub fn new(value: f64) -> AnimatedValue {
        AnimatedValue {
            duration: -f64::INFINITY,
            start_time: -f64::INFINITY,
            target: value,
            a: 0.0,
            b: 0.0,
            n: 0,
        }
    }

    pub fn get(&self, current_time: f64) -> f64 {
        let t = current_time - self.start_time;
        if t >= self.duration {
            return self.target;
        }

        self.target + (self.duration - t).powi(self.n + 1) * (self.a + self.b * t)
    }

    pub fn is_animating(&self, current_time: f64) -> bool {
        current_time - self.start_time < self.duration
    }

    /// Starts an animation towards `target` over the next `duration` seconds
    /// while matching the current position and velocity and ensuring the first
    /// `n` derivatives are zero at `target` by using a degree `n+2` polynomial.
    pub fn animate_towards(&mut self, target: f64, duration: f64, n: u32, current_time: f64) {
        let p = self.get(current_time);
        let t = current_time - self.start_time;
        let v = if t < self.duration {
            (self.duration - t).powi(self.n)
                * (self.a + self.duration * self.b - (self.n as f64 + 2.0) * (self.a + self.b * t))
        } else {
            0.0
        };
        let c = 1.0 / duration;
        self.n = n as i32;
        let e = c.powi(self.n + 1);
        self.a = (p - target) * e;
        self.b = (self.n + 1) as f64 * self.a * c + v * e;
        self.duration = duration;
        self.start_time = current_time;
        self.target = target;
    }
}
