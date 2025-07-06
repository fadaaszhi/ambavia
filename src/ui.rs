use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

use arboard::Clipboard;
use glam::DVec2;
use winit::{
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::ModifiersState,
    window::{CursorIcon, Window},
};

use crate::utility::AsGlam;

pub struct Context {
    clipboard: Arc<Mutex<Option<Clipboard>>>,
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
            cursor: DVec2::ZERO,
            scale_factor: window.scale_factor(),
            modifiers: Default::default(),
        }
    }

    pub fn update(&mut self, event: &WindowEvent) {
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

#[derive(Debug)]
pub enum Event {
    Resized,
    KeyboardInput(KeyEvent),
    CursorMoved {
        /// The cursor's previous logical position
        previous_cursor: DVec2,
    },
    MouseWheel(DVec2),
    MouseInput(ElementState, MouseButton),
    PinchGesture(f64),
}

#[derive(Debug, Clone, Copy)]
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
}

#[derive(Debug, Default)]
pub struct Response {
    pub consumed_event: bool,
    pub requested_redraw: bool,
    pub cursor_icon: CursorIcon,
}

impl Response {
    pub fn consume_event(&mut self) {
        self.consumed_event = true;
    }

    pub fn request_redraw(&mut self) {
        self.requested_redraw = true;
    }

    pub fn or(self, other: Response) -> Response {
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

    pub fn or_else(self, f: impl FnOnce() -> Response) -> Response {
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

pub enum QuadKind {
    MsdfGlyph(DVec2, DVec2),
    TranslucentMsdfGlyph(DVec2, DVec2),
    BlackBox,
    TranslucentBlackBox,
    HighlightBox,
    GrayBox,
    TransparentToWhiteGradient,
}

impl QuadKind {
    pub fn index(&self) -> u32 {
        use QuadKind::*;
        match self {
            MsdfGlyph(..) => 0,
            TranslucentMsdfGlyph(..) => 1,
            BlackBox => 2,
            TranslucentBlackBox => 3,
            HighlightBox => 4,
            GrayBox => 5,
            TransparentToWhiteGradient => 6,
        }
    }
}
