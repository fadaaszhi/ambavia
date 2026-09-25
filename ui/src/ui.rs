use std::{
    borrow::Cow,
    cell::RefCell,
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

use arboard::Clipboard;
use glam::{DVec2, DVec4, dvec4};
use winit::{
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::ModifiersState,
    window::{CursorIcon, Window},
};

use crate::utility::{AsGlam, set};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RedrawRequest {
    absolute_time_bits: u64,
    id: u64,
}

#[derive(Debug, Default)]
pub struct RedrawRequests {
    next_id: u64,
    requests: BTreeSet<RedrawRequest>,
}

impl RedrawRequests {
    fn request_redraw_at(&mut self, absolute_time: f64) -> RedrawRequest {
        let absolute_time_bits = absolute_time.max(0.0).to_bits();
        let id = self.next_id;
        self.next_id += 1;
        let request = RedrawRequest {
            absolute_time_bits,
            id,
        };
        self.requests.insert(request.clone());
        request
    }

    fn is_request_pending(&self, request: &RedrawRequest) -> bool {
        self.requests.contains(request)
    }

    fn cancel_request(&mut self, request: RedrawRequest) {
        self.requests.remove(&request);
    }

    /// Removes all requests that have absolute_time <= time
    fn remove_completed(&mut self, time: f64) {
        if (0.0..f64::INFINITY).contains(&time) {
            self.requests = self.requests.split_off(&RedrawRequest {
                absolute_time_bits: time.to_bits() + 1,
                id: 0,
            })
        }
    }

    fn get_min_absolute_time(&self) -> Option<f64> {
        self.requests
            .first()
            .map(|r| f64::from_bits(r.absolute_time_bits))
    }
}

pub struct Context {
    clipboard: Arc<Mutex<Option<Clipboard>>>,
    /// The number of seconds elapsed since the application started
    pub time: f64,
    /// The cursor's current logical position
    pub cursor: DVec2,
    left_mouse_button_pressed: bool,
    pub left_mouse_button_already_pressed: bool,
    /// The window's scale factor
    pub scale_factor: f64,
    pub modifiers: ModifiersState,
    // refcell because i'm lazy
    redraw_requests: RefCell<RedrawRequests>,
}

impl Context {
    pub fn new(window: &Window) -> Self {
        Self {
            clipboard: Arc::new(Mutex::new(None)),
            time: 0.0,
            cursor: DVec2::ZERO,
            left_mouse_button_pressed: false,
            left_mouse_button_already_pressed: false,
            scale_factor: window.scale_factor(),
            modifiers: Default::default(),
            redraw_requests: RefCell::default(),
        }
    }

    pub fn update(&mut self, event: &WindowEvent, time: f64) {
        self.time = time;
        self.left_mouse_button_already_pressed = self.left_mouse_button_pressed;

        match &event {
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = position.as_glam() / self.scale_factor
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = *scale_factor
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.left_mouse_button_pressed = state.is_pressed();
                self.left_mouse_button_already_pressed &= state.is_pressed();
            }
            _ => {}
        }
    }

    // TODO unify Response::request_redraw and Context::request_redraw_after
    pub fn request_redraw_after(&self, seconds: f64) -> RedrawRequest {
        self.redraw_requests
            .borrow_mut()
            .request_redraw_at(self.time + seconds)
    }

    pub fn cancel_redraw_request(&self, request: RedrawRequest) {
        self.redraw_requests.borrow_mut().cancel_request(request)
    }

    pub fn is_redraw_request_pending(&self, request: &RedrawRequest) -> bool {
        self.redraw_requests.borrow().is_request_pending(request)
    }

    pub fn remove_completed_redraw_requests(&mut self) {
        self.redraw_requests.get_mut().remove_completed(self.time);
    }

    pub fn get_next_requested_redraw_time(&self) -> Option<f64> {
        self.redraw_requests.borrow().get_min_absolute_time()
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

    /// Round a logical point to an integer physical point, returning a logical
    /// point.
    pub fn roundp(&self, p: DVec2) -> DVec2 {
        p.map(|x| self.round(x))
    }

    /// Round the corners of a logical bound to integer physical values,
    /// returning a logical bound.
    pub fn roundb(&self, logical: Bounds) -> Bounds {
        let top_left = self.roundp(logical.pos);
        let bottom_left = self.roundp(logical.pos + logical.size);
        Bounds {
            pos: top_left,
            size: bottom_left - top_left,
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
    /// Sent once right before every redraw. If you request a redraw during this
    /// event, it will trigger an additional redraw next frame.
    AnimationFrame,
    ModifiersChanged,
    KeyboardInput(KeyEvent),
    CursorMoved {
        /// The cursor's previous logical position
        previous_cursor: DVec2,
    },
    MouseWheel(DVec2),
    MouseInput(ElementState, MouseButton),
    PinchGesture(f64),
}

impl Event {
    pub fn is_animation_frame(&self) -> bool {
        matches!(self, Event::AnimationFrame)
    }
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

    pub fn grow(self, radius: f64) -> Bounds {
        Bounds {
            pos: self.pos - radius,
            size: self.size + 2.0 * radius,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum CursorMode {
    #[default]
    NoPreference,
    Hidden,
    Icon(CursorIcon),
}

#[derive(Debug, Default, Clone, Copy)]
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

impl Color for ([u8; 3], f64) {
    fn to_rgbaf64(self) -> DVec4 {
        (self.0[0], self.0[1], self.0[2], self.1).to_rgbaf64()
    }
}

pub const fn rgb(r: u8, g: u8, b: u8) -> DVec4 {
    dvec4(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0, 1.0)
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

#[derive(Default)]
pub struct Button {
    pub hovered: bool,
    pub pressed: bool,
}

impl Button {
    pub fn state(&self) -> usize {
        if self.pressed {
            2
        } else if self.hovered {
            1
        } else {
            0
        }
    }
}

impl Button {
    pub fn update(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> (Response, bool) {
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
