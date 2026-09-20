use std::fmt::Write;
use std::iter::zip;
use std::ops::DerefMut;
use std::{collections::HashMap, ops::Deref};

use derive_more::{Add, From, Into, Sub};
use glam::{DVec2, DVec4, dvec2, dvec4};
use parse::name_resolver::PropertyIndex;
use typed_index_collections::{TiVec, ti_vec};
use winit::{
    event::{ElementState, MouseButton},
    window::CursorIcon,
};

use crate::katex_font::Font;
use crate::label::{Label, render_label};
use crate::quad_renderer::{Quad, QuadKind};
use crate::ui::{AnimatedValue, Button, ClickDragTracker, Color, PRIMARY_COLOR};
use crate::utility::FiniteExt;
use crate::{
    graph::{Geometry, GeometryKind},
    math_field::{Cursor, Interactiveness, MathField, Message, UserSelection},
    ui::{Bounds, Context, CursorMode, Event, Response},
    utility::{max, mix, set, union, unmix},
};
use eval::{
    compiler::compile_assignments,
    math::{apply_slider, apply_slider_step},
    vm::{self, Vm},
};
use parse::{
    analyze_expression_list::{ExpressionResult, PlotKind, analyze_expression_list},
    ast,
    ast_parser::{parse_standalone_expression, parse_statement},
    latex_parser::parse_latex,
    latex_tree::{self, Bracket},
    name_resolver::{Domain, ExpressionIndex, ExpressionListEntry, Slider as NrSlider},
    type_checker::Type,
};

#[derive(Debug, Default, Clone, Copy, PartialEq)]
enum UnderlineState {
    #[default]
    None,
    Hovered,
    Focussed,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Underline {
    state: UnderlineState,
    error: bool,
}

impl Underline {
    fn update(
        &mut self,
        ctx: &Context,
        field: &MathField,
        field_bounds: Bounds,
        response: &mut Response,
    ) {
        let new = if field.has_focus() {
            UnderlineState::Focussed
        } else if field_bounds.contains(ctx.cursor) {
            UnderlineState::Hovered
        } else {
            UnderlineState::None
        };
        if set(&mut self.state, new) {
            response.request_redraw();
        }
    }

    fn render(&self, field_bounds: Bounds, draw_quad: &mut impl FnMut(Quad)) {
        let thickness = if self.state != UnderlineState::None || self.error {
            2.0
        } else {
            1.0
        };
        let top_left = field_bounds.pos + dvec2(0.0, field_bounds.size.y - 1.0);
        let bottom_right = top_left + dvec2(field_bounds.size.x, thickness);
        let color = match self.state {
            _ if self.error => [225, 88, 85],
            UnderlineState::None | UnderlineState::Hovered => [180; 3],
            UnderlineState::Focussed => PRIMARY_COLOR,
        };
        draw_quad(Quad::rectangle(top_left, bottom_right, color));
    }
}

struct InlineField {
    field: MathField,
    underline: Underline,
    do_underline: bool,
    min_width: f64,
    max_width: f64,
}

impl Deref for InlineField {
    type Target = MathField;

    fn deref(&self) -> &Self::Target {
        &self.field
    }
}

impl DerefMut for InlineField {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.field
    }
}

impl InlineField {
    const DEFAULT_FIELD_SCALE: f64 = 15.7;

    fn new(placeholder: &str) -> Self {
        let mut field = MathField::default();
        field.set_placeholder(
            &placeholder
                .chars()
                .map(latex_tree::Node::Char)
                .collect::<Vec<_>>(),
        );
        field.scale = Self::DEFAULT_FIELD_SCALE;
        field.left_padding = 0.22;
        field.bottom_padding = 0.25;
        field.top_padding = 0.25;
        Self {
            field,
            underline: Default::default(),
            do_underline: true,
            min_width: 35.0,
            max_width: 70.0,
        }
    }

    // TODO think about whether we should be accepting `ctx` and doing the ceiling
    // ourselves or if we should instead leave it as the caller's responsibility
    fn expression_size(&self, ctx: &Context, clamp: bool) -> DVec2 {
        let mut size = self.field.expression_size().map(|s| ctx.ceil(s));
        if clamp {
            size.x = size.x.clamp(self.min_width, self.max_width);
        }
        size
    }

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        bounds: Bounds,
    ) -> (Response, Option<Message>) {
        let (mut response, message) = self.field.update(ctx, event, bounds, None);
        self.underline
            .update(ctx, &self.field, bounds, &mut response);
        (response, message)
    }

    fn render(&mut self, ctx: &Context, bounds: Bounds, draw_quad: &mut impl FnMut(Quad)) {
        self.field.render(ctx, bounds, draw_quad);
        if self.do_underline {
            self.underline.state = if self.field.has_focus() {
                UnderlineState::Focussed
            } else if bounds.contains(ctx.cursor) {
                UnderlineState::Hovered
            } else {
                UnderlineState::None
            };
            self.underline.render(bounds, draw_quad);
        }
    }
}

const SLIDER_SOFT_MIN_DEFAULT: f64 = -10.0;
const SLIDER_SOFT_MAX_DEFAULT: f64 = 10.0;
const SLIDER_STEP_DEFAULT: f64 = 0.0;
const PARAMETRIC_DOMAIN_MIN_DEFAULT: f64 = 0.0;
const PARAMETRIC_DOMAIN_MAX_DEFAULT: f64 = 1.0;

#[derive(Default)]
struct GutterButton {
    hovered: bool,
    click_tracker: ClickDragTracker,
}

impl GutterButton {
    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        hovered: bool,
        on_click: impl FnOnce(&mut Response),
    ) -> Response {
        let mut response = Response::default();

        if set(&mut self.hovered, hovered) {
            response.request_redraw();
        }

        match event {
            Event::MouseInput(ElementState::Pressed, MouseButton::Left) if self.hovered => {
                self.click_tracker.press(ctx.cursor);
                response.consume_event();
                response.request_redraw();
            }
            Event::CursorMoved { .. } => {
                if self.click_tracker.drag(ctx.cursor) {
                    response.request_redraw();
                }
            }
            Event::MouseInput(ElementState::Released, MouseButton::Left)
                if self.click_tracker.release().was_clicked() =>
            {
                on_click(&mut response);
                response.request_redraw();
            }
            _ => {}
        }

        if self.hovered {
            response.cursor_mode = CursorMode::Icon(CursorIcon::Pointer);
        }

        response
    }

    fn state_pressed(&self) -> usize {
        if self.click_tracker.is_pressed() {
            2
        } else if self.hovered {
            1
        } else {
            0
        }
    }
}

fn draw_popup_container(bounds: Bounds, arrow: Option<Bounds>, draw_quad: &mut impl FnMut(Quad)) {
    let shadow_radius = 10.0; // hardcoded in quad.wgsl
    let shadow_offset = dvec2(0.0, 5.0);
    draw_quad(Quad {
        kind: QuadKind::PopupShadow,
        p0: bounds.pos - shadow_radius + shadow_offset,
        p1: bounds.pos + bounds.size + shadow_radius + shadow_offset,
        color: dvec4(0.0, 0.0, 0.0, 0.2),
        ..Default::default()
    });

    draw_quad(Quad::from_bounds(
        bounds,
        QuadKind::PopupBackground,
        [255; 3],
    ));

    if let Some(arrow) = arrow {
        draw_quad(Quad::from_bounds(arrow, QuadKind::PopupArrow, [255; 3]));
    }
}

struct SliderUi {
    value: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    dragging: Option<f64>,
    point_hovered: bool,
    point_hover_animation: AnimatedValue,
    name: String,
    name_field: MathField,
    step_label: Label<'static>,

    animated_value: f64,
    expected_value: Option<f64>,
    play_button: GutterButton,
    mode_button: GutterButton,

    is_popup_open: bool,
    animation_mode_label: Label<'static>,
    mode_radio_buttons: [Button; 4],
    speed_label: Label<'static>,
    decrease_speed_button: Button,
    increase_speed_button: Button,
    should_close_popup: bool,
}

struct SliderEditMinNameMaxLayout {
    min_field: Bounds,
    name: Bounds,
    max_field: Bounds,
}

struct SliderEditLayout {
    min_name_max: Option<SliderEditMinNameMaxLayout>,
    step_label: Bounds,
    step_field: Bounds,
    bounds: Bounds,
}

struct SliderBarLayout {
    min_field: Bounds,
    max_field: Bounds,
    bar_left: f64,
    bar_right: f64,
    bar_hitbox: Bounds,
    point_hitbox: Bounds,
    point: DVec2,
    point_radius: f64,
    bounds: Bounds,
}

enum SliderLayout {
    Edit(SliderEditLayout),
    Bar(SliderBarLayout),
}

struct SliderGutterLayout {
    play_button_center: DVec2,
    play_button_radius: f64,
    play_button: Bounds,
    mode_button: Bounds,
    mode_button_hitbox: Bounds,
}

struct SliderPopupLayout {
    animation_mode_cursor: DVec2,
    mode_buttons: [Bounds; 4],
    mode_button_hitboxes: [Bounds; 4],
    speed_cursor: DVec2,
    decrease_speed_button: Bounds,
    increase_speed_button: Bounds,
    arrow: Option<Bounds>,
    bounds: Bounds,
}

impl SliderUi {
    const SLIDER_BAR_RADIUS: f64 = 3.0;
    const SLIDER_TICK_RADIUS: f64 = Self::SLIDER_BAR_RADIUS / 3.0;
    const SLIDER_STEP_TICKS_THRESHOLD: f64 = 0.03;
    const SLIDER_POINT_RADIUS: f64 = 11.0;

    fn new(name: String) -> Self {
        SliderUi {
            value: Some(0.0),
            min: Some(SLIDER_SOFT_MIN_DEFAULT),
            max: Some(SLIDER_SOFT_MAX_DEFAULT),
            step: Some(SLIDER_STEP_DEFAULT),
            dragging: None,
            point_hovered: false,
            point_hover_animation: AnimatedValue::new(0.0),
            name_field: create_le_name_le(&name),
            name,
            step_label: Label::new("Step:", 15.7, Font::MainRegular),

            animated_value: 0.0,
            expected_value: None,
            play_button: Default::default(),
            mode_button: Default::default(),

            is_popup_open: false,
            animation_mode_label: Label::new("Animation Mode", 16.0, Font::MainRegular),
            mode_radio_buttons: Default::default(),
            speed_label: Label::new("Speed", 16.0, Font::MainRegular),
            decrease_speed_button: Default::default(),
            increase_speed_button: Default::default(),
            should_close_popup: false,
        }
    }

    fn set_fields(
        &mut self,
        slider: &mut Slider,
        new_name: &str,
        new_value: Option<f64>,
        new_min: Option<f64>,
        new_max: Option<f64>,
        new_step: Option<f64>,
    ) {
        if set(&mut self.name, new_name) {
            self.name_field = create_le_name_le(new_name);
        }

        self.value = new_value;
        self.step = new_step;

        for (old, field, new) in [
            (&mut self.min, &mut slider.hard_min, new_min),
            (&mut self.max, &mut slider.hard_max, new_max),
        ] {
            if set(old, new)
                && let Some(new) = new
            {
                let mut latex = vec![];
                number_to_latex(&mut latex, new);
                field.0.set_placeholder(&latex);
            }
        }
    }

    fn layout(
        &mut self,
        ctx: &Context,
        padding: f64,
        top_left: DVec2,
        width: f64,
        field_has_focus: bool,
        slider: &mut Slider,
    ) -> SliderLayout {
        let is_play_indefinitely = slider.loop_mode == SliderLoopMode::PlayIndefinitely;
        let is_slider_edit_shown = field_has_focus
            || slider.hard_min.0.has_focus()
            || slider.hard_max.0.has_focus()
            || slider.step.0.has_focus()
            || slider.hard_min.0.underline.error
            || slider.hard_max.0.underline.error
            || slider.step.0.underline.error
            || is_play_indefinitely;

        for field in [&mut slider.hard_max, &mut slider.hard_min] {
            field.0.use_placeholder_if_empty = !is_slider_edit_shown;
            field.0.grayed = !is_slider_edit_shown;
            field.0.do_underline = is_slider_edit_shown;
            field.0.scale = if is_slider_edit_shown {
                InlineField::DEFAULT_FIELD_SCALE
            } else {
                12.9
            };
        }

        let min_field_size = slider.hard_min.0.expression_size(ctx, is_slider_edit_shown);
        let max_field_size = slider.hard_max.0.expression_size(ctx, is_slider_edit_shown);

        if is_slider_edit_shown {
            // TODO currently uses a dumb way to calculate layout for PlayIndefinitely, make it more efficient
            let name_size = self.name_field.expression_size().map(|s| ctx.ceil(s));
            let step_label_size = self.step_label.size();
            let step_field_size = slider.step.0.expression_size(ctx, true);

            let height = if is_play_indefinitely {
                max([step_label_size.y, step_field_size.y])
            } else {
                max([
                    min_field_size.y,
                    name_size.y,
                    max_field_size.y,
                    step_label_size.y,
                    step_field_size.y,
                ])
            };

            let min_field = Bounds {
                pos: dvec2(
                    top_left.x + padding,
                    top_left.y + (height - min_field_size.y) / 2.0,
                ),
                size: min_field_size,
            };
            let name = Bounds {
                pos: dvec2(min_field.right(), top_left.y + (height - name_size.y) / 2.0),
                size: name_size,
            };
            let max_field = Bounds {
                pos: dvec2(name.right(), top_left.y + (height - max_field_size.y) / 2.0),
                size: max_field_size,
            };
            let step_label = Bounds {
                pos: dvec2(
                    if is_play_indefinitely {
                        top_left.x + padding
                    } else {
                        max_field.right() + 11.0
                    },
                    top_left.y + (height - step_label_size.y) / 2.0,
                ),
                size: step_label_size,
            };
            let step_field = Bounds {
                pos: dvec2(
                    step_label.right() + 1.4,
                    top_left.y + (height - step_field_size.y) / 2.0,
                ),
                size: step_field_size,
            };

            let bounds = if is_play_indefinitely {
                union([step_label, step_field])
            } else {
                union([min_field, name, max_field, step_label, step_field])
            };

            SliderLayout::Edit(SliderEditLayout {
                min_name_max: (!is_play_indefinitely).then_some(SliderEditMinNameMaxLayout {
                    min_field,
                    name,
                    max_field,
                }),
                step_label,
                step_field,
                bounds,
            })
        } else {
            let point_radius = ctx.round_nonzero(Self::SLIDER_POINT_RADIUS);
            let height = max([point_radius * 2.0, min_field_size.y, max_field_size.y]);

            let min_field = Bounds {
                pos: top_left + dvec2(0.5 * padding, height / 2.0 - min_field_size.y / 2.0),
                size: min_field_size,
            };
            let max_field = Bounds {
                pos: top_left
                    + dvec2(
                        width - 0.5 * padding - max_field_size.x,
                        height / 2.0 - max_field_size.y / 2.0,
                    ),
                size: max_field_size,
            };

            let bar_left = min_field.right() + 0.8 * padding;
            let bar_right = max_field.left() - 0.8 * padding;

            let (Some(ref mut value), Some(ref mut min), Some(ref mut max)) =
                (self.value, self.min, self.max)
            else {
                unreachable!("only None if error in which case slider edit shown")
            };

            let point = dvec2(
                mix(
                    bar_left,
                    bar_right,
                    unmix(*value, *min, *max).clamp(0.0, 1.0),
                ),
                top_left.y + height / 2.0,
            );
            let bar_hitbox = Bounds {
                pos: dvec2(bar_left, point.y - point_radius),
                size: dvec2(bar_right - bar_left, 2.0 * point_radius),
            };
            let point_hitbox_size = dvec2(3.0, 2.0) * point_radius;
            let point_hitbox = Bounds {
                pos: point - point_hitbox_size / 2.0,
                size: point_hitbox_size,
            };
            let bounds = Bounds {
                pos: top_left,
                size: dvec2(width, height),
            };

            SliderLayout::Bar(SliderBarLayout {
                min_field,
                max_field,
                bar_left,
                bar_right,
                bar_hitbox,
                point_hitbox,
                point,
                point_radius,
                bounds,
            })
        }
    }

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        padding: f64,
        top_left: DVec2,
        width: f64,
        field_has_focus: bool,
        slider: &mut Slider,
    ) -> (Response, Option<f64>, Option<Message>, Bounds) {
        let mut result = match self.layout(ctx, padding, top_left, width, field_has_focus, slider) {
            SliderLayout::Edit(layout) => self.update_slider_edit(ctx, event, slider, layout),
            SliderLayout::Bar(layout) => self.update_slider_bar(ctx, event, slider, layout),
        };

        if self.should_close_popup {
            self.should_close_popup = false;
            self.is_popup_open = false;
            result.0.request_redraw();
        }

        result
    }

    fn update_slider_edit(
        &mut self,
        ctx: &Context,
        event: &Event,
        slider: &mut Slider,
        l: SliderEditLayout,
    ) -> (Response, Option<f64>, Option<Message>, Bounds) {
        let mut response = Response::default();
        let mut message = None;

        if let Some(l) = &l.min_name_max {
            let (min_response, mut min_message) = slider.hard_min.0.update(ctx, event, l.min_field);
            let (max_response, mut max_message) = slider.hard_max.0.update(ctx, event, l.max_field);

            match min_message {
                Some(Message::ContentsChanged { .. }) => {
                    slider.soft_min = SLIDER_SOFT_MIN_DEFAULT;
                    if !slider.hard_min.0.is_empty() {
                        slider.hard_min.1 =
                            parse_standalone_expression(&slider.hard_min.0.to_latex());
                    }
                }
                Some(Message::Left) => min_message = None,
                Some(Message::Right) => {
                    min_message = None;
                    slider.hard_min.0.unfocus();
                    slider.hard_max.0.select_all();
                }
                Some(Message::Up | Message::Down | Message::Add) => {
                    slider.hard_min.0.unfocus();
                }
                Some(Message::Remove) => {
                    min_message = None;
                    if set(&mut slider.soft_min, SLIDER_SOFT_MIN_DEFAULT) {
                        min_message = Some(Message::ContentsChanged { user_driven: true });
                    }
                }
                None => {}
            }

            match max_message {
                Some(Message::ContentsChanged { .. }) => {
                    slider.soft_max = SLIDER_SOFT_MAX_DEFAULT;
                    if !slider.hard_max.0.is_empty() {
                        slider.hard_max.1 =
                            parse_standalone_expression(&slider.hard_max.0.to_latex());
                    }
                }
                Some(Message::Left) => {
                    max_message = None;
                    slider.hard_max.0.unfocus();
                    slider.hard_min.0.select_all();
                }
                Some(Message::Right) => {
                    max_message = None;
                    slider.hard_max.0.unfocus();
                    slider.step.0.select_all();
                }
                Some(Message::Up | Message::Down | Message::Add) => {
                    slider.hard_max.0.unfocus();
                }
                Some(Message::Remove) => {
                    max_message = None;
                    if set(&mut slider.soft_max, SLIDER_SOFT_MAX_DEFAULT) {
                        max_message = Some(Message::ContentsChanged { user_driven: true });
                    }
                }
                None => {}
            }

            response = response.or(min_response).or(max_response);
            message = message.or(min_message).or(max_message);
        }

        let (step_response, mut step_message) = slider.step.0.update(ctx, event, l.step_field);

        match step_message {
            Some(Message::ContentsChanged { .. }) => {
                if slider.step.0.is_empty() {
                    slider.step.1 = Ok(ast::Expression::Number(0.0));
                } else {
                    slider.step.1 = parse_standalone_expression(&slider.step.0.to_latex());
                }
            }
            Some(Message::Left) if l.min_name_max.is_some() => {
                step_message = None;
                slider.step.0.unfocus();
                slider.hard_max.0.select_all();
            }
            Some(Message::Left | Message::Right | Message::Remove) => step_message = None,
            Some(Message::Up | Message::Down | Message::Add) => {
                slider.step.0.unfocus();
            }
            None => {}
        }

        if slider.step.0.has_focus() && set(&mut slider.is_playing, false) {
            response.request_redraw();
        }

        let mut new_value = None;

        if slider.loop_mode != SliderLoopMode::PlayIndefinitely
            || self.value.is_none()
            || self.step.is_none()
        {
            // We didn't update the slider animation so next animation step it shouldn't apply any delta time
            slider.previous_update_time = None;
        } else if event.is_animation_frame()
            && slider.is_playing
            && let (Some(value), Some(step)) = (&mut self.value, self.step)
        {
            // animated_value is the raw unstepped value used to maintain
            // correct timing. check if it got invalidated by something like
            // an action updating the slider value
            if let Some(expected) = self.expected_value
                && apply_slider(expected, f64::NAN, f64::NAN, step)
                    != apply_slider(*value, f64::NAN, f64::NAN, step)
            {
                self.animated_value = *value;
            }

            let dt = ctx.time - slider.previous_update_time.unwrap_or(ctx.time);
            let speed = 4.0 / slider.animation_period * if step == 0.0 { 1.0 } else { step.abs() };
            self.animated_value = (self.animated_value + speed * dt).if_finite_else(*value);

            // TODO round value to fewest required decimal places based on animation period,framerate,step
            if set(
                value,
                apply_slider(self.animated_value, f64::NAN, f64::NAN, step),
            ) {
                new_value = Some(*value);
                self.expected_value = Some(*value);
            }

            // TODO make sliders with a step only request an animation frame when
            // they actually need to change using ctx.request_redraw_after
            slider.previous_update_time = Some(ctx.time);
            response.request_redraw();
        }

        response = response.or(step_response);
        message = message.or(step_message);
        (response, new_value, message, l.bounds)
    }

    fn update_slider_bar(
        &mut self,
        ctx: &Context,
        event: &Event,
        slider: &mut Slider,
        l: SliderBarLayout,
    ) -> (Response, Option<f64>, Option<Message>, Bounds) {
        let (Some(ref mut value), Some(ref mut min), Some(ref mut max), Some(ref mut step)) =
            (self.value, self.min, self.max, self.step)
        else {
            unreachable!("only None if error in which case slider edit shown")
        };
        let mut response = Response::default();
        let new_point_hovered = l.point_hitbox.contains(ctx.cursor);
        let bar_hovered = l.bar_hitbox.contains(ctx.cursor);
        let new_slider_min_hovered = l.min_field.contains(ctx.cursor);
        let new_slider_max_hovered = l.max_field.contains(ctx.cursor);

        let mut new_value = None;
        let original_value = *value;
        let mut should_update_soft_bounds = false;

        let mut update_value_based_on_slider = |offset: f64| {
            // Not using `.clamp()` because it panics if sidebar is resized too small
            let point_x = (ctx.cursor.x + offset).max(l.bar_left).min(l.bar_right);
            *value = mix(*min, *max, unmix(point_x, l.bar_left, l.bar_right));
            *value = apply_slider(*value, *min, *max, *step);
            new_value = Some(*value);
            should_update_soft_bounds = true;
            response.consume_event();
            response.request_redraw();
        };

        match event {
            // drag point
            Event::CursorMoved { .. } if self.dragging.is_some() => {
                update_value_based_on_slider(self.dragging.unwrap());
            }
            Event::MouseInput(ElementState::Pressed, MouseButton::Left) => {
                if new_point_hovered {
                    // start dragging point
                    self.dragging = Some(l.point.x - ctx.cursor.x);
                    should_update_soft_bounds = true;
                    slider.is_playing = false;
                    response.consume_event();
                } else if bar_hovered {
                    // teleport point then start dragging
                    update_value_based_on_slider(0.0);
                    self.dragging = Some(0.0);
                    slider.is_playing = false;
                } else if new_slider_min_hovered {
                    // select min field
                    slider.hard_min.0.select_all();
                    slider.is_playing = false;
                    response.consume_event();
                    response.request_redraw()
                } else if new_slider_max_hovered {
                    // select max field
                    slider.hard_max.0.select_all();
                    slider.is_playing = false;
                    response.consume_event();
                    response.request_redraw()
                }
            }
            Event::MouseInput(ElementState::Released, MouseButton::Left)
                if self.dragging.is_some() =>
            {
                self.dragging = None;
                response.consume_event();
            }
            Event::AnimationFrame => {
                if self.point_hover_animation.is_animating(ctx.time) {
                    response.request_redraw();
                }

                if slider.is_playing {
                    // animated_value is the raw unstepped value used to maintain
                    // correct timing. check if it got invalidated by something like
                    // an action updating the slider value
                    if let Some(expected) = self.expected_value
                        && apply_slider(expected, *min, *max, *step)
                            != apply_slider(*value, *min, *max, *step)
                    {
                        self.animated_value = *value;
                    }

                    let dt = ctx.time - slider.previous_update_time.unwrap_or(ctx.time);
                    let (smin, smax, speed) = if slider.loop_mode == SliderLoopMode::LoopForward {
                        let speed = 1.0 - *step / (*max - *min + *step);
                        (*min - *step / 2.0, *max + *step / 2.0, speed)
                    } else {
                        (*min, *max, 1.0)
                    };
                    let x = unmix(self.animated_value.clamp(smin, smax), smin, smax);
                    let y = x + slider.play_direction * dt * speed / slider.animation_period;
                    let z = match slider.loop_mode {
                        SliderLoopMode::LoopForwardReverse => {
                            slider.play_direction *= 1.0 - y.rem_euclid(2.0).floor() * 2.0;
                            1.0 - (y.rem_euclid(2.0) - 1.0).abs()
                        }
                        SliderLoopMode::LoopForward => y.rem_euclid(1.0),
                        SliderLoopMode::PlayOnce => y.clamp(0.0, 1.0),
                        SliderLoopMode::PlayIndefinitely => {
                            unreachable!("handled in update_slider_edit")
                        }
                    };
                    self.animated_value = mix(smin, smax, z).if_finite_else(*value);

                    // TODO round value to fewest required decimal places based on animation period,framerate,max-min,step
                    if set(value, apply_slider(self.animated_value, *min, *max, *step)) {
                        new_value = Some(*value);
                        self.expected_value = Some(*value);
                    }

                    should_update_soft_bounds = true;
                    // TODO make sliders with a step only request an animation frame when
                    // they actually need to change using ctx.request_redraw_after
                    if *min == *max
                        || slider.loop_mode == SliderLoopMode::PlayOnce
                            && self.animated_value >= *max
                    {
                        slider.previous_update_time = None;
                    } else {
                        slider.previous_update_time = Some(ctx.time);
                        response.request_redraw();
                    }
                }
            }
            _ => {}
        }

        if should_update_soft_bounds {
            slider.soft_min = slider.soft_min.min(*value).min(original_value);
            slider.soft_max = slider.soft_max.max(*value).max(original_value);
        }

        if set(
            &mut self.point_hovered,
            new_point_hovered || self.dragging.is_some(),
        ) {
            let current = self.point_hover_animation.get(ctx.time);
            let target = if self.point_hovered { 1.0 } else { 0.0 };
            let duration = (current - target).abs().sqrt() * 0.2;
            self.point_hover_animation
                .animate_towards(target, duration, 2, ctx.time);
            response.request_redraw();
        }

        #[cfg(not(windows))]
        let (grab, grabbing) = (CursorIcon::Grab, CursorIcon::Grabbing);

        // https://github.com/rust-windowing/winit/issues/1043
        #[cfg(windows)]
        let (grab, grabbing) = (CursorIcon::EwResize, CursorIcon::EwResize);

        if self.dragging.is_some() {
            response.cursor_mode = CursorMode::Icon(grabbing);
        } else if self.point_hovered {
            response.cursor_mode = CursorMode::Icon(grab);
        } else if bar_hovered || new_slider_min_hovered || new_slider_max_hovered {
            response.cursor_mode = CursorMode::Icon(CursorIcon::Pointer);
        }

        (response, new_value, None, l.bounds)
    }

    fn render(
        &mut self,
        ctx: &Context,
        padding: f64,
        top_left: DVec2,
        width: f64,
        field_has_focus: bool,
        slider: &mut Slider,
        draw_quad: &mut impl FnMut(Quad),
    ) -> f64 {
        match self.layout(ctx, padding, top_left, width, field_has_focus, slider) {
            SliderLayout::Edit(layout) => self.render_edit(ctx, slider, draw_quad, layout),
            SliderLayout::Bar(layout) => self.render_bar(ctx, slider, draw_quad, layout),
        }
    }

    fn render_edit(
        &mut self,
        ctx: &Context,
        slider: &mut Slider,
        draw_quad: &mut impl FnMut(Quad),
        l: SliderEditLayout,
    ) -> f64 {
        if let Some(l) = &l.min_name_max {
            slider.hard_min.0.render(ctx, l.min_field, draw_quad);
            self.name_field.render(ctx, l.name, draw_quad);
            slider.hard_max.0.render(ctx, l.max_field, draw_quad);
        }
        self.step_label
            .render_from_top_left(l.step_label.pos, [0; 3], draw_quad);
        slider.step.0.render(ctx, l.step_field, draw_quad);

        l.bounds.size.y
    }

    fn render_bar(
        &mut self,
        ctx: &Context,
        slider: &mut Slider,
        draw_quad: &mut impl FnMut(Quad),
        l: SliderBarLayout,
    ) -> f64 {
        let (Some(min), Some(max), Some(step)) = (self.min, self.max, self.step) else {
            unreachable!("only None if error in which case slider edit shown")
        };
        let opacity = if min == max { 0.3 } else { 1.0 };
        let bar_radius = ctx.round_nonzero(Self::SLIDER_BAR_RADIUS);
        let tick_radius = ctx.round_nonzero(Self::SLIDER_TICK_RADIUS);

        // slider bar
        draw_quad(Quad::pill(
            (l.bar_left, l.point.y - bar_radius),
            (l.bar_right, l.point.y + bar_radius),
            [0.9; 3].with_opacity(opacity),
        ));

        // step ticks on slider bar
        if step.abs() >= (max - min) * Self::SLIDER_STEP_TICKS_THRESHOLD {
            let n = (((max - min) / step.abs()).ceil() as u32).max(1) - 1;
            for i in 1..=n {
                let value = min + step * i as f64;
                let tick = dvec2(
                    mix(l.bar_left, l.bar_right, unmix(value, min, max)),
                    l.point.y,
                );
                draw_quad(Quad::pill(
                    tick - tick_radius,
                    tick + tick_radius,
                    [1.0; 3].with_opacity(opacity),
                ));
            }
        }

        // zero tick on slider bar
        if (min..=max).contains(&0.0) {
            let tick = dvec2(
                mix(l.bar_left, l.bar_right, unmix(0.0, min, max)),
                l.point.y,
            );
            draw_quad(Quad::pill(
                tick - tick_radius,
                tick + tick_radius,
                (0, 0, 0, 0.35).with_opacity(opacity),
            ));
        }

        // slider point
        draw_quad(Quad::pill(
            l.point - l.point_radius,
            l.point + l.point_radius,
            PRIMARY_COLOR.with_opacity(0.25).with_opacity(opacity),
        ));
        let inner_radius = mix(
            bar_radius,
            l.point_radius,
            self.point_hover_animation.get(ctx.time),
        );
        draw_quad(
            Quad::pill(
                l.point - inner_radius,
                l.point + inner_radius,
                PRIMARY_COLOR.with_opacity(opacity),
            )
            .pixel_snap(ctx),
        );

        // min/max field
        slider.hard_min.0.render(ctx, l.min_field, draw_quad);
        slider.hard_max.0.render(ctx, l.max_field, draw_quad);

        l.bounds.size.y
    }

    fn layout_gutter(
        &self,
        ctx: &Context,
        bounds: Bounds,
        slider: &Slider,
    ) -> Option<SliderGutterLayout> {
        if slider.loop_mode != SliderLoopMode::PlayIndefinitely {
            let (Some(_value), Some(min), Some(max), Some(_step)) =
                (self.value, self.min, self.max, self.step)
            else {
                return None;
            };

            // TODO find a way to not repeat this validity check
            if min > max {
                return None;
            }
        } else if self.value.is_none() || self.step.is_none() {
            return None;
        }

        let play_button_center = bounds.pos + bounds.size.x * dvec2(0.5, 0.752);
        let play_button_radius = 0.392 * bounds.size.x;
        let play_button = ctx.roundb(Bounds {
            pos: play_button_center - play_button_radius,
            size: DVec2::splat(play_button_radius * 2.0),
        });
        let mode_button_center = bounds.pos + bounds.size.x * dvec2(0.5, 1.588);
        let mode_button_size = bounds.size.x * dvec2(0.39, 0.355);
        let mode_button = ctx.roundb(Bounds {
            pos: mode_button_center - mode_button_size / 2.0,
            size: mode_button_size,
        });
        let mode_button_hitbox_size = 1.5 * mode_button_size;
        let mode_button_hitbox = Bounds {
            pos: mode_button_center - mode_button_hitbox_size / 2.0,
            size: mode_button_hitbox_size,
        };

        Some(SliderGutterLayout {
            play_button_center,
            play_button_radius,
            play_button,
            mode_button,
            mode_button_hitbox,
        })
    }

    fn update_gutter(
        &mut self,
        ctx: &Context,
        event: &Event,
        slider: &mut Slider,
        bounds: Bounds,
    ) -> Response {
        let mut response = Response::default();
        let Some(l) = self.layout_gutter(ctx, bounds, slider) else {
            return response;
        };

        response = response.or(self.play_button.update(
            ctx,
            event,
            ctx.cursor.distance(l.play_button_center) <= l.play_button_radius,
            |response| {
                let max = self.max.expect("play button only shows when no error");
                slider.is_playing = !(slider.is_playing
                    && (slider.loop_mode != SliderLoopMode::PlayOnce || self.animated_value < max));
                if slider.is_playing {
                    let value = self.value.expect("play button only shows if no error");
                    slider.soft_min = slider.soft_min.min(value);
                    slider.soft_max = slider.soft_max.max(value);
                    slider.previous_update_time = Some(ctx.time);
                    if slider.loop_mode == SliderLoopMode::PlayOnce && value >= max {
                        self.animated_value =
                            self.min.expect("play button only shows when no error");
                        self.expected_value = None;
                    } else {
                        self.animated_value = value;
                        self.expected_value = Some(value);
                    }
                }
                response.request_redraw();
            },
        ));

        response = response.or(self.mode_button.update(
            ctx,
            event,
            l.mode_button_hitbox.contains(ctx.cursor),
            |response| {
                self.is_popup_open ^= true;

                response.request_redraw();
            },
        ));
        response
    }

    fn render_gutter(
        &mut self,
        ctx: &Context,
        bounds: Bounds,
        expression_is_focussed: bool,
        slider: &mut Slider,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        let Some(l) = self.layout_gutter(ctx, bounds, slider) else {
            return;
        };
        let mut render = |button: &GutterButton, kind: QuadKind, bounds: Bounds| {
            draw_quad(Quad {
                kind,
                p0: bounds.pos,
                p1: bounds.pos + bounds.size,
                color: if expression_is_focussed {
                    (255, 255, 255, [0.9, 1.0, 1.0][button.state_pressed()])
                } else {
                    (0, 0, 0, [0.5, 0.7, 0.9][button.state_pressed()])
                }
                .to_rgbaf64(),
                ..Default::default()
            })
        };
        render(
            &self.play_button,
            if slider.is_playing
                && (slider.loop_mode != SliderLoopMode::PlayOnce
                    || self.animated_value
                        < self.max.expect("play button only shows when no error"))
            {
                QuadKind::SliderPlayingButton
            } else {
                QuadKind::SliderPausedButton
            },
            l.play_button,
        );
        render(&self.mode_button, slider.loop_mode.into(), l.mode_button);
    }

    fn layout_popup(
        &mut self,
        ctx: &Context,
        expression_list_bounds: Bounds,
        gutter_bounds: Bounds,
        slider: &Slider,
    ) -> Option<SliderPopupLayout> {
        if !self.is_popup_open {
            return None;
        }
        let Some(g) = self.layout_gutter(ctx, gutter_bounds, slider) else {
            self.is_popup_open = false;
            return None;
        };

        let padding = 13.0;
        let border_width = 1.0; // hardcoded in quad.wgsl
        let mode_button_size = dvec2(30.0, 26.0); // including borders
        const N_MODES: usize = 4;
        let mode_buttons_width =
            mode_button_size.x * N_MODES as f64 - (N_MODES - 1) as f64 * border_width;
        let speed_button_size = dvec2(26.0, 26.0);
        let speed_number_width = 45.0;
        let speed_buttons_width = speed_button_size.x * 2.0 + speed_number_width;

        let width = max([
            self.animation_mode_label.size().x,
            mode_buttons_width,
            self.speed_label.size().x,
            speed_buttons_width,
        ]) + 2.0 * padding;

        let arrow_tip = g.mode_button.pos + g.mode_button.size * dvec2(1.0, 0.5) + dvec2(3.0, 0.0);

        let top = arrow_tip.y - 19.0;
        let animation_mode_cursor_y = top + padding + self.animation_mode_label.scale;
        let mode_buttons_y = animation_mode_cursor_y + 9.0;
        let speed_cursor_y = mode_buttons_y + mode_button_size.y + 11.0 + self.speed_label.scale;
        let speed_buttons_y = speed_cursor_y + 9.0;
        let bottom = speed_buttons_y + speed_button_size.y + padding;

        let clearance = 9.0;
        let bottom1 = bottom.min(expression_list_bounds.bottom() - clearance);
        let top1 = top + bottom1 - bottom;
        let top2 = top1.max(expression_list_bounds.top() + clearance);
        let bottom2 = bottom1 + top2 - top1;

        let offset_y = top2 - top;
        let top = top2;
        let bottom = bottom2;

        let arrow_half_height = 9.0;
        let arrow_width = arrow_half_height;
        let corner_radius = 6.0; // hardcoded in quad.wgsl
        let is_arrow_shown = top + corner_radius < arrow_tip.y - arrow_half_height
            && arrow_tip.y + arrow_half_height < bottom - corner_radius;

        let arrow = is_arrow_shown.then_some(Bounds {
            pos: arrow_tip + dvec2(0.0, -arrow_half_height),
            size: dvec2(arrow_width + border_width, 2.0 * arrow_half_height),
        });

        let left = arrow_tip.x + arrow_width;
        let animation_mode_cursor = dvec2(left + padding, animation_mode_cursor_y + offset_y);
        let mode_buttons = std::array::from_fn::<_, N_MODES, _>(|i| Bounds {
            pos: dvec2(
                left + padding + i as f64 * (mode_button_size.x - border_width),
                mode_buttons_y + offset_y,
            ),
            size: mode_button_size,
        });
        let mode_button_hitboxes = mode_buttons.map(|b| Bounds {
            pos: b.pos + dvec2(border_width * 0.5, 0.0),
            size: b.size - dvec2(border_width, 0.0),
        });
        let speed_cursor = dvec2(left + padding, speed_cursor_y + offset_y);
        let decrease_speed_button = ctx.roundb(Bounds {
            pos: dvec2(left + padding, speed_buttons_y + offset_y),
            size: speed_button_size,
        });
        let increase_speed_button = ctx.roundb(Bounds {
            pos: dvec2(
                decrease_speed_button.right() + speed_number_width,
                decrease_speed_button.top(),
            ),
            size: speed_button_size,
        });

        let bounds = ctx.roundb(Bounds {
            pos: dvec2(left, top),
            size: dvec2(width, bottom - top),
        });

        Some(SliderPopupLayout {
            animation_mode_cursor,
            mode_buttons: mode_buttons.map(|b| ctx.roundb(b)),
            mode_button_hitboxes,
            speed_cursor,
            decrease_speed_button,
            increase_speed_button,
            arrow,
            bounds,
        })
    }

    fn decreased_increased_slider_speeds(animation_period: f64) -> (Option<f64>, Option<f64>) {
        let slider_speeds = [
            0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0,
        ];
        if animation_period <= 0.0 {
            (slider_speeds.last().cloned(), None)
        } else {
            let current_speed = 4.0 / animation_period;
            let decreased_speed = slider_speeds.into_iter().rev().find(|s| *s < current_speed);
            let increased_speed = slider_speeds.into_iter().find(|s| *s > current_speed);
            (decreased_speed, increased_speed)
        }
    }

    fn update_popup(
        &mut self,
        ctx: &Context,
        event: &Event,
        slider: &mut Slider,
        expression_list_bounds: Bounds,
        gutter_bounds: Bounds,
    ) -> Response {
        let mut response = Response::default();
        let Some(l) = self.layout_popup(ctx, expression_list_bounds, gutter_bounds, slider) else {
            return response;
        };

        for (i, (button, hitbox)) in
            zip(&mut self.mode_radio_buttons, l.mode_button_hitboxes).enumerate()
        {
            let (r, clicked) = button.update(ctx, event, hitbox);
            response = response.or(r);

            if clicked {
                let original_loop_mode = slider.loop_mode;
                slider.loop_mode = [
                    SliderLoopMode::LoopForwardReverse,
                    SliderLoopMode::LoopForward,
                    SliderLoopMode::PlayOnce,
                    SliderLoopMode::PlayIndefinitely,
                ][i];
                slider.play_direction = 1.0;

                if slider.loop_mode == SliderLoopMode::PlayIndefinitely {
                    slider.hard_min.0.clear();
                    slider.hard_max.0.clear();
                } else if slider.is_playing {
                    self.animated_value = self.min.expect("button only shows if no error");
                    self.expected_value = None;

                    if original_loop_mode == SliderLoopMode::PlayIndefinitely {
                        // TODO fix soft bounds resetting, it's not working because
                        // somewhere later they are getting set based on stale value
                        slider.soft_min = SLIDER_SOFT_MIN_DEFAULT;
                        slider.soft_max = SLIDER_SOFT_MAX_DEFAULT;
                    }
                }
                response.request_redraw();
            }
        }

        let (decreased, increased) =
            Self::decreased_increased_slider_speeds(slider.animation_period);

        if let Some(decreased) = decreased {
            let (r, clicked) =
                self.decrease_speed_button
                    .update(ctx, event, l.decrease_speed_button);
            response = response.or(r);
            if clicked {
                slider.animation_period = 4.0 / decreased;
            }
        }

        if let Some(increased) = increased {
            let (r, clicked) =
                self.increase_speed_button
                    .update(ctx, event, l.increase_speed_button);
            response = response.or(r);
            if clicked {
                slider.animation_period = 4.0 / increased;
            }
        }

        // TODO figure out more robust way to make popups steal the correct inputs from things beneath them
        if l.bounds.contains(ctx.cursor) {
            let mut r = Response::default();
            if matches!(
                event,
                Event::MouseInput(ElementState::Pressed, _)
                    | Event::PinchGesture(_)
                    | Event::MouseWheel(_)
            ) {
                r.consume_event();
            }
            r.cursor_mode = CursorMode::Icon(CursorIcon::Default);
            response = response.or(r)
        } else if matches!(event, Event::MouseInput(ElementState::Pressed, _))
            && let Some(l) = self.layout_gutter(ctx, gutter_bounds, slider)
            && !l.mode_button_hitbox.contains(ctx.cursor)
        {
            // don't just set self.is_popup_open = true because then we might
            // accidentally reopen if we had clicked on mode button
            self.should_close_popup = true;
            response.request_redraw();
        }

        response
    }

    fn render_popup(
        &mut self,
        ctx: &Context,
        slider: &Slider,
        expression_list_bounds: Bounds,
        gutter_bounds: Bounds,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        let Some(l) = self.layout_popup(ctx, expression_list_bounds, gutter_bounds, slider) else {
            return;
        };

        draw_popup_container(l.bounds, l.arrow, draw_quad);

        self.animation_mode_label
            .render_from_cursor(l.animation_mode_cursor, [90; 3], draw_quad);

        let mut order = [
            (0, SliderLoopMode::LoopForwardReverse),
            (1, SliderLoopMode::LoopForward),
            (2, SliderLoopMode::PlayOnce),
            (3, SliderLoopMode::PlayIndefinitely),
        ];
        order.sort_by_key(|(i, mode)| {
            if *mode == slider.loop_mode {
                4
            } else {
                self.mode_radio_buttons[*i].state()
            }
        });

        for (i, mode) in order {
            let b = &l.mode_buttons[i];
            draw_quad(Quad {
                kind: match (i, mode == slider.loop_mode) {
                    (0, true) => QuadKind::PopupRadioSelectedLeft,
                    (0, false) => QuadKind::PopupRadioLeft,
                    (3, true) => QuadKind::PopupRadioSelectedRight,
                    (3, false) => QuadKind::PopupRadioRight,
                    (_, true) => QuadKind::PopupRadioSelectedMiddle,
                    (_, false) => QuadKind::PopupRadioMiddle,
                },
                p0: b.pos,
                p1: b.pos + b.size,
                color: if mode == slider.loop_mode {
                    PRIMARY_COLOR
                } else {
                    [[255, 245, 204][self.mode_radio_buttons[i].state()]; 3]
                }
                .to_rgbaf64(),
                ..Default::default()
            });
            draw_quad(
                Quad {
                    kind: mode.into(),
                    p0: b.pos + b.size * dvec2(0.234, 0.244),
                    p1: b.pos + b.size * dvec2(0.766, 0.804),
                    color: if mode == slider.loop_mode {
                        PRIMARY_COLOR
                    } else {
                        [[38, 0, 0][self.mode_radio_buttons[i].state()]; 3]
                    }
                    .to_rgbaf64(),
                    ..Default::default()
                }
                .pixel_snap(ctx),
            );
        }

        self.speed_label
            .render_from_cursor(l.speed_cursor, [90; 3], draw_quad);

        let (decreased, increased) =
            Self::decreased_increased_slider_speeds(slider.animation_period);

        draw_quad(Quad {
            kind: QuadKind::PopupButton,
            p0: l.decrease_speed_button.pos,
            p1: l.decrease_speed_button.pos + l.decrease_speed_button.size,
            color: if decreased.is_some() {
                [[255, 245, 204][self.decrease_speed_button.state()]; 3].to_rgbaf64()
            } else {
                [255; 3].with_opacity(0.25)
            },
            ..Default::default()
        });

        draw_quad(Quad {
            kind: QuadKind::IncreaseSliderSpeedIcon, // x mirrored
            p0: l.decrease_speed_button.pos + l.decrease_speed_button.size * dvec2(0.615, 0.337),
            p1: l.decrease_speed_button.pos + l.decrease_speed_button.size * dvec2(0.325, 0.663),
            color: if decreased.is_some() {
                [[102, 34, 0][self.decrease_speed_button.state()]; 3].to_rgbaf64()
            } else {
                [102; 3].with_opacity(0.25)
            },
            ..Default::default()
        });

        let mut speed_number = (4.0 / slider.animation_period).to_string();
        speed_number.push('×');
        let label = Label::new(&speed_number, 15.0, Font::MainRegular);
        let size = label.size();
        let middle = dvec2(
            (l.decrease_speed_button.right() + l.increase_speed_button.left()) / 2.0,
            l.decrease_speed_button.pos.y + l.decrease_speed_button.size.y / 2.0,
        );
        label.render_from_top_left(middle - size / 2.0, [0; 3], draw_quad);

        draw_quad(Quad {
            kind: QuadKind::PopupButton,
            p0: l.increase_speed_button.pos,
            p1: l.increase_speed_button.pos + l.increase_speed_button.size,
            color: if increased.is_some() {
                [[255, 245, 204][self.increase_speed_button.state()]; 3].to_rgbaf64()
            } else {
                [255; 3].with_opacity(0.25)
            },
            ..Default::default()
        });

        draw_quad(Quad {
            kind: QuadKind::IncreaseSliderSpeedIcon,
            p0: l.increase_speed_button.pos + l.increase_speed_button.size * dvec2(0.385, 0.337),
            p1: l.increase_speed_button.pos + l.increase_speed_button.size * dvec2(0.675, 0.663),
            color: if increased.is_some() {
                [[102, 34, 0][self.increase_speed_button.state()]; 3].to_rgbaf64()
            } else {
                [102; 3].with_opacity(0.25)
            },
            ..Default::default()
        });
    }
}

struct FieldUi(MathField);

impl FieldUi {
    fn new(latex: &[latex_tree::Node]) -> FieldUi {
        let mut field = MathField::from(latex);
        field.interactiveness = Interactiveness::Select;
        field.scale = 18.0;
        field.left_padding = 0.22;
        field.right_padding = 0.4;
        field.bottom_padding = 0.19;
        field.top_padding = 0.25;
        FieldUi(field)
    }

    fn layout(&self, ctx: &Context, top_left: DVec2, width: f64, padding: f64) -> Bounds {
        let size = self.0.expression_size().map(|s| ctx.ceil(s));
        let right = top_left.x + width - 0.5 * padding;
        let left = (right - size.x).max(top_left.x + padding);
        Bounds {
            pos: dvec2(left, top_left.y),
            size: dvec2(right - left, size.y),
        }
    }

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        top_left: DVec2,
        width: f64,
        padding: f64,
    ) -> (Response, Option<f64>, Option<Message>, Bounds) {
        let bounds = self.layout(ctx, top_left, width, padding);
        let (mut response, _) = self.0.update(ctx, event, bounds, None);

        if let Some(UserSelection { anchor, focus }) = self.0.get_selection() {
            let mut clamp = |mut cursor: Cursor| {
                let index = cursor
                    .path
                    .first_mut()
                    .map_or(&mut cursor.index, |(index, _)| index);
                if *index == 0 {
                    *index = 1;
                    response.request_redraw();
                }
                cursor
            };
            self.0
                .set_selection((clamp(anchor.clone()), clamp(focus.clone())));
        }

        (response, None, None, bounds)
    }

    fn render(
        &mut self,
        ctx: &Context,
        padding: f64,
        top_left: DVec2,
        width: f64,
        draw_quad: &mut impl FnMut(Quad),
    ) -> f64 {
        let bounds = self.layout(ctx, top_left, width, padding);
        draw_quad(Quad {
            kind: QuadKind::OutputValueBox,
            p0: bounds.pos,
            p1: bounds.pos + bounds.size,
            color: dvec4(0.96, 0.96, 0.96, 1.0),
            ..Default::default()
        });
        self.0.render(ctx, bounds, draw_quad);
        bounds.size.y
    }
}

// TODO turn this into general purpose IntervalUi that can be reused inside
// SliderUi
struct ParametricDomainUi {
    name: String,
    name_field: MathField,
}

struct ParametricDomainLayout {
    min_field: Bounds,
    name: Bounds,
    max_field: Bounds,
    bounds: Bounds,
}

impl ParametricDomainUi {
    fn new(name: String) -> Self {
        Self {
            name_field: create_le_name_le(&name),
            name,
        }
    }

    fn set_name(&mut self, new_name: &str) {
        if set(&mut self.name, new_name) {
            self.name_field = create_le_name_le(new_name);
        }
    }

    fn layout(
        &mut self,
        ctx: &Context,
        padding: f64,
        top_left: DVec2,
        domain: &mut ParametricDomain,
    ) -> ParametricDomainLayout {
        let min_field_size = domain.min.0.expression_size(ctx, true);
        let name_size = self.name_field.expression_size().map(|s| ctx.ceil(s));
        let max_field_size = domain.max.0.expression_size(ctx, true);

        let height = max([min_field_size.y, name_size.y, max_field_size.y]);

        let min_field = Bounds {
            pos: dvec2(
                top_left.x + padding,
                top_left.y + (height - min_field_size.y) / 2.0,
            ),
            size: min_field_size,
        };
        let name = Bounds {
            pos: dvec2(min_field.right(), top_left.y + (height - name_size.y) / 2.0),
            size: name_size,
        };
        let max_field = Bounds {
            pos: dvec2(name.right(), top_left.y + (height - max_field_size.y) / 2.0),
            size: max_field_size,
        };

        let bounds = union([min_field, name, max_field]);

        ParametricDomainLayout {
            min_field,
            name,
            max_field,
            bounds,
        }
    }

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        padding: f64,
        top_left: DVec2,
        domain: &mut ParametricDomain,
    ) -> (Response, Option<f64>, Option<Message>, Bounds) {
        let l = self.layout(ctx, padding, top_left, domain);

        let (min_response, mut min_message) = domain.min.0.update(ctx, event, l.min_field);
        let (max_response, mut max_message) = domain.max.0.update(ctx, event, l.max_field);

        match min_message {
            Some(Message::ContentsChanged { .. }) => {
                let mut latex = domain.min.0.to_latex();
                if latex.is_empty() {
                    latex = domain.min.0.get_placeholder();
                }
                domain.min.1 = parse_standalone_expression(&latex)
            }
            Some(Message::Left | Message::Remove) => min_message = None,
            Some(Message::Right) => {
                min_message = None;
                domain.min.0.unfocus();
                domain.max.0.select_all();
            }
            Some(Message::Up | Message::Down | Message::Add) => domain.min.0.unfocus(),
            None => {}
        }

        match max_message {
            Some(Message::ContentsChanged { .. }) => {
                let mut latex = domain.max.0.to_latex();
                if latex.is_empty() {
                    latex = domain.max.0.get_placeholder();
                }
                domain.max.1 = parse_standalone_expression(&latex)
            }
            Some(Message::Left) => {
                max_message = None;
                domain.max.0.unfocus();
                domain.min.0.select_all();
            }
            Some(Message::Right | Message::Remove) => max_message = None,
            Some(Message::Up | Message::Down | Message::Add) => domain.max.0.unfocus(),
            None => {}
        }

        let response = min_response.or(max_response);
        let message = min_message.or(max_message);
        (response, None, message, l.bounds)
    }

    fn render(
        &mut self,
        ctx: &Context,
        padding: f64,
        top_left: DVec2,
        domain: &mut ParametricDomain,
        draw_quad: &mut impl FnMut(Quad),
    ) -> f64 {
        let l = self.layout(ctx, padding, top_left, domain);
        domain.min.0.render(ctx, l.min_field, draw_quad);
        self.name_field.render(ctx, l.name, draw_quad);
        domain.max.0.render(ctx, l.max_field, draw_quad);
        l.bounds.size.y
    }
}

#[derive(Default)]
enum OutputUi {
    #[default]
    None,
    Slider(SliderUi),
    Field(FieldUi),
    ParametricDomain(ParametricDomainUi),
}

/// Creates a MathField initialized with ≤name≤
fn create_le_name_le(name: &str) -> MathField {
    use latex_tree::Node;

    let mut latex = vec![];
    latex.push(Node::CtrlSeq("le"));
    let mut parts = name.split("_");
    let a = parts.next().unwrap();
    if a.chars().count() > 1 {
        latex.push(Node::CtrlSeq(a));
    } else {
        latex.push(Node::Char(a.chars().next().unwrap()));
    }
    if let Some(b) = parts.next() {
        let b = b.strip_prefix("{").unwrap().strip_suffix("}").unwrap();
        latex.push(Node::SubSup {
            sub: Some(b.chars().map(Node::Char).collect()),
            sup: None,
        });
    }
    latex.push(Node::CtrlSeq("le"));

    let mut field = MathField::from(&latex);
    field.interactiveness = Interactiveness::None;
    field.scale = 18.0;

    field
}

fn number_to_latex(nodes: &mut Vec<latex_tree::Node>, mut x: f64) {
    use latex_tree::Node::{self, Char as C};

    if x.is_nan() {
        nodes.push(Node::Frac {
            num: vec![C('0')],
            den: vec![C('0')],
        });
        return;
    }

    if x.is_sign_negative() {
        nodes.push(C('-'));
        x = -x;
    }

    if x.is_infinite() {
        nodes.push(Node::CtrlSeq("infty"));
        return;
    }

    let mut buffer = ryu::Buffer::new();
    let mut s = buffer.format_finite(x).split('e');
    let m = s.next().unwrap();
    nodes.extend(m.strip_suffix(".0").unwrap_or(m).chars().map(C));

    if let Some(e) = s.next() {
        nodes.extend([
            Node::CtrlSeq("times"),
            C('1'),
            C('0'),
            Node::SubSup {
                sub: None,
                sup: Some(e.chars().map(C).collect()),
            },
        ]);
    }
}

fn color_to_latex(nodes: &mut Vec<latex_tree::Node>, r: f64, g: f64, b: f64) {
    use latex_tree::Node::Char as C;
    // Not using DelimitedGroup because that increases the height compared to Char('(')
    nodes.extend([C('r'), C('g'), C('b'), C('(')]);
    let f = |nodes: &mut _, x: f64| {
        number_to_latex(
            nodes,
            (x.if_finite_else(0.0).clamp(0.0, 1.0) * 255.0).round(),
        )
    };
    f(nodes, r);
    nodes.push(C(','));
    f(nodes, g);
    nodes.push(C(','));
    f(nodes, b);
    nodes.push(C(')'));
}

impl OutputUi {
    fn set_slider(
        &mut self,
        slider: &mut Slider,
        name: &str,
        value: Option<f64>,
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
    ) {
        let ui = match self {
            OutputUi::Slider(ui) => ui,
            _ => {
                slider.is_playing = false;
                *self = OutputUi::Slider(SliderUi::new(name.into()));
                match self {
                    OutputUi::Slider(ui) => ui,
                    _ => unreachable!(),
                }
            }
        };

        ui.set_fields(slider, name, value, min, max, step);
    }

    fn set_parametric_domain(&mut self, name: &str) {
        // If there was an already existing parametric domain then we just need to update its name
        if let OutputUi::ParametricDomain(ui) = self {
            ui.set_name(name);
            return;
        }
        // Otherwise we need to create a whole new parametric domain UI
        *self = OutputUi::ParametricDomain(ParametricDomainUi::new(name.into()))
    }

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        padding: f64,
        top_left: DVec2,
        width: f64,
        field_has_focus: bool,
        slider: &mut Slider,
        parametric_domain: &mut ParametricDomain,
    ) -> (Response, Option<f64>, Option<Message>, Bounds) {
        match self {
            OutputUi::None => (Response::default(), None, None, Bounds::default()),
            OutputUi::Slider(ui) => ui.update(
                ctx,
                event,
                padding,
                top_left,
                width,
                field_has_focus,
                slider,
            ),
            OutputUi::Field(ui) => ui.update(ctx, event, top_left, width, padding),
            OutputUi::ParametricDomain(ui) => {
                ui.update(ctx, event, padding, top_left, parametric_domain)
            }
        }
    }

    fn render(
        &mut self,
        ctx: &Context,
        padding: f64,
        top_left: DVec2,
        width: f64,
        field_has_focus: bool,
        slider: &mut Slider,
        parametric_domain: &mut ParametricDomain,
        draw_quad: &mut impl FnMut(Quad),
    ) -> f64 {
        match self {
            OutputUi::None => 0.0,
            OutputUi::Slider(ui) => ui.render(
                ctx,
                padding,
                top_left,
                width,
                field_has_focus,
                slider,
                draw_quad,
            ),
            OutputUi::Field(ui) => ui.render(ctx, padding, top_left, width, draw_quad),
            OutputUi::ParametricDomain(ui) => {
                ui.render(ctx, padding, top_left, parametric_domain, draw_quad)
            }
        }
    }
}

#[derive(Debug, Default)]
enum OutputData {
    #[default]
    None,
    Error(String),
    DraggablePoint(Geometry),
    Geometry(Vec<Geometry>),
}

#[derive(Default)]
struct Output {
    ui: OutputUi,
    data: OutputData,
}

impl Output {
    const NONE: Output = Output {
        ui: OutputUi::None,
        data: OutputData::None,
    };

    fn new_error(error: String) -> Output {
        Output {
            ui: OutputUi::None,
            data: OutputData::Error(error),
        }
    }
}

/// If `expr` is a numeric literal then this returns its value, otherwise it returns `None`.
fn get_numeric_literal(expr: &parse::ast::Expression) -> Option<f64> {
    match expr {
        parse::ast::Expression::Number(x) => Some(*x),
        parse::ast::Expression::Op {
            operation: parse::op::OpName::Neg,
            args: arguments,
        } => Some(-get_numeric_literal(
            arguments.first().expect("neg should have one argument"),
        )?),
        _ => None,
    }
}

#[derive(Default, Clone, Copy, PartialEq)]
enum SliderLoopMode {
    #[default]
    LoopForwardReverse,
    LoopForward,
    PlayOnce,
    PlayIndefinitely,
}

impl From<SliderLoopMode> for QuadKind {
    fn from(value: SliderLoopMode) -> Self {
        match value {
            SliderLoopMode::LoopForwardReverse => QuadKind::LoopForwardReverseIcon,
            SliderLoopMode::LoopForward => QuadKind::LoopForwardIcon,
            SliderLoopMode::PlayOnce => QuadKind::PlayOnceIcon,
            SliderLoopMode::PlayIndefinitely => QuadKind::PlayIndefinitelyIcon,
        }
    }
}

/// If the hard bound is empty, then the soft bound is used.
struct Slider {
    hard_min: (InlineField, Result<parse::ast::Expression, String>),
    soft_min: f64,
    hard_max: (InlineField, Result<parse::ast::Expression, String>),
    soft_max: f64,
    step: (InlineField, Result<parse::ast::Expression, String>),
    is_playing: bool,
    /// `None` means that the next time the slider animates it will use a delta time of zero
    previous_update_time: Option<f64>,
    /// `1.0` or `-1.0`
    play_direction: f64,
    /// In seconds
    animation_period: f64,
    loop_mode: SliderLoopMode,
    /// This is what is displayed to the user when a slider is shown instead of
    /// the actual math field. It's to handle desync between the actual value vs
    /// clamped slider value, e.g., when slider bounds get animated.
    // TODO fix this ugly solution, it's annoying having to maintain both fake_field and real field
    fake_field: MathField,
    fake_field_value: f64,
}

type ParametricDomain = Domain<(InlineField, Result<parse::ast::Expression, String>)>;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum LineStyle {
    #[default]
    Solid,
    Dashed,
    Dotted,
}

impl From<LineStyle> for QuadKind {
    fn from(value: LineStyle) -> Self {
        match value {
            LineStyle::Solid => QuadKind::LineStyleSolidIcon,
            LineStyle::Dashed => QuadKind::LineStyleDashedIcon,
            LineStyle::Dotted => QuadKind::LineStyleDottedIcon,
        }
    }
}

const LINE_WIDTH_DEFAULT: f64 = 2.5;
const LINE_OPACITY_DEFAULT: f64 = 1.0;

struct LineAppearance {
    enabled: Option<bool>,
    style: LineStyle,
    width: (InlineField, Result<ast::Expression, String>),
    opacity: (InlineField, Result<ast::Expression, String>),
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum PointStyle {
    #[default]
    Point,
    Open,
    Cross,
    Square,
    Plus,
    Triangle,
    Diamond,
    Star,
}

impl PointStyle {
    fn gutter_quad_kind(self) -> QuadKind {
        match self {
            PointStyle::Point => QuadKind::GutterPointPointIcon,
            PointStyle::Open => QuadKind::GutterPointOpenIcon,
            PointStyle::Cross => QuadKind::GutterPointCrossIcon,
            PointStyle::Square => QuadKind::GutterPointSquareIcon,
            PointStyle::Plus => QuadKind::GutterPointPlusIcon,
            PointStyle::Triangle => QuadKind::GutterPointTriangleIcon,
            PointStyle::Diamond => QuadKind::GutterPointDiamondIcon,
            PointStyle::Star => QuadKind::GutterPointStarIcon,
        }
    }

    fn popup_quad_kind(self) -> QuadKind {
        match self {
            PointStyle::Point => QuadKind::PointStylePointIcon,
            PointStyle::Open => QuadKind::PointStyleOpenIcon,
            PointStyle::Cross => QuadKind::PointStyleCrossIcon,
            PointStyle::Square => QuadKind::PointStyleSquareIcon,
            PointStyle::Plus => QuadKind::PointStylePlusIcon,
            PointStyle::Triangle => QuadKind::PointStyleTriangleIcon,
            PointStyle::Diamond => QuadKind::PointStyleDiamondIcon,
            PointStyle::Star => QuadKind::PointStyleStarIcon,
        }
    }
}

const POINT_SIZE_DEFAULT: f64 = 8.0;
const POINT_OPACITY_DEFAULT: f64 = 1.0;

struct PointAppearance {
    enabled: Option<bool>,
    style: PointStyle,
    size: (InlineField, Result<ast::Expression, String>),
    opacity: (InlineField, Result<ast::Expression, String>),
}

const FILL_OPACITY_DEFAULT: f64 = 0.4;

struct FillAppearance {
    enabled: Option<bool>,
    opacity: (InlineField, Result<ast::Expression, String>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DragMode {
    X,
    Y,
    XY,
}

struct DragAppearance {
    enabled: Option<bool>,
    mode: DragMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ExpressionStyleKind {
    None,
    Equality,
    Point,
    PointList,
    DraggablePoint,
    Polygon,
    Parametric,
}

struct ExpressionStyle {
    changed: bool,
    kind: ExpressionStyleKind,
    hidden: bool,
    color: DVec4,
    color_latex: (InlineField, Result<ast::Expression, String>),
    line: LineAppearance,
    point: PointAppearance,
    fill: FillAppearance,
    drag: DragAppearance,
}

fn create_with_placeholder(placeholder: f64) -> (InlineField, Result<ast::Expression, String>) {
    (
        InlineField::new(&placeholder.to_string()),
        Ok(ast::Expression::Number(placeholder)),
    )
}

impl Default for ExpressionStyle {
    fn default() -> Self {
        let create_with_placeholder = |placeholder: f64| {
            let mut y = create_with_placeholder(placeholder);
            y.0.min_width = 42.0;
            y.0.max_width = 60.0;
            y
        };
        let color = EXPRESSION_COLORS[0];
        let mut color_latex = (
            InlineField::new(""),
            Ok(ast::Expression::Call {
                callee: "rgb".into(),
                args: vec![
                    ast::Expression::Number(1.0),
                    ast::Expression::Number(2.0),
                    ast::Expression::Number(3.0),
                ],
            }),
        );
        let mut nodes = vec![];
        color_to_latex(&mut nodes, color.x, color.y, color.z);
        color_latex.0.set_placeholder(&nodes);
        color_latex.0.min_width = 135.0;
        color_latex.0.max_width = 183.0;

        Self {
            changed: true,
            kind: ExpressionStyleKind::None,
            hidden: false,
            color,
            color_latex,
            line: LineAppearance {
                enabled: None,
                style: Default::default(),
                width: create_with_placeholder(LINE_WIDTH_DEFAULT),
                opacity: create_with_placeholder(LINE_OPACITY_DEFAULT),
            },
            point: PointAppearance {
                enabled: None,
                style: Default::default(),
                size: create_with_placeholder(POINT_SIZE_DEFAULT),
                opacity: create_with_placeholder(POINT_OPACITY_DEFAULT),
            },
            fill: FillAppearance {
                enabled: None,
                opacity: create_with_placeholder(FILL_OPACITY_DEFAULT),
            },
            drag: DragAppearance {
                enabled: None,
                mode: DragMode::XY,
            },
        }
    }
}

enum StylePopupSectionKind {
    Line,
    Point,
    Fill,
    Drag,
}

impl ExpressionStyle {
    fn set_color(&mut self, color: DVec4) {
        self.color = color;
        self.color_latex.0.clear();
        let mut nodes = vec![];
        color_to_latex(&mut nodes, self.color.x, self.color.y, self.color.z);
        self.color_latex.0.set_placeholder(&nodes);
    }

    fn popup_order(&self) -> &'static [StylePopupSectionKind] {
        use StylePopupSectionKind as K;
        match self.kind {
            ExpressionStyleKind::None => &[],
            ExpressionStyleKind::Equality => &[K::Line],
            ExpressionStyleKind::Point => &[K::Point],
            ExpressionStyleKind::PointList => &[K::Point, K::Line],
            ExpressionStyleKind::DraggablePoint => &[K::Point, K::Drag],
            ExpressionStyleKind::Polygon => &[K::Line, K::Fill],
            ExpressionStyleKind::Parametric => &[K::Line, K::Fill],
        }
    }

    fn line_enabled(&self) -> bool {
        match self.kind {
            ExpressionStyleKind::None => false,
            ExpressionStyleKind::Equality => self.line.enabled.unwrap_or(true),
            ExpressionStyleKind::Point => false,
            ExpressionStyleKind::PointList => self.line.enabled.unwrap_or(false),
            ExpressionStyleKind::DraggablePoint => false,
            ExpressionStyleKind::Polygon => self.line.enabled.unwrap_or(true),
            ExpressionStyleKind::Parametric => self.line.enabled.unwrap_or(true),
        }
    }

    fn point_enabled(&self) -> bool {
        match self.kind {
            ExpressionStyleKind::None => false,
            ExpressionStyleKind::Equality => false,
            ExpressionStyleKind::Point => self.point.enabled.unwrap_or(true),
            ExpressionStyleKind::PointList => self.point.enabled.unwrap_or(true),
            ExpressionStyleKind::DraggablePoint => self.point.enabled.unwrap_or(true),
            ExpressionStyleKind::Polygon => false,
            ExpressionStyleKind::Parametric => false,
        }
    }

    fn fill_enabled(&self) -> bool {
        match self.kind {
            ExpressionStyleKind::None => false,
            ExpressionStyleKind::Equality => false,
            ExpressionStyleKind::Point => false,
            ExpressionStyleKind::PointList => false,
            ExpressionStyleKind::DraggablePoint => false,
            ExpressionStyleKind::Polygon => self.fill.enabled.unwrap_or(true),
            ExpressionStyleKind::Parametric => self.fill.enabled.unwrap_or(false),
        }
    }

    fn drag_enabled(&self) -> bool {
        match self.kind {
            ExpressionStyleKind::None => false,
            ExpressionStyleKind::Equality => false,
            ExpressionStyleKind::Point => false,
            ExpressionStyleKind::PointList => false,
            ExpressionStyleKind::DraggablePoint => self.drag.enabled.unwrap_or(true),
            ExpressionStyleKind::Polygon => false,
            ExpressionStyleKind::Parametric => false,
        }
    }
}

struct StyleGutter {
    toggle_button: GutterButton,
    color: Vec<DVec4>,

    is_popup_open: bool,
    should_close_popup: bool,

    lines_label: Label<'static>,
    line_enabled_button: Button,
    line_style_radio_buttons: [Button; 3],

    points_label: Label<'static>,
    point_enabled_button: Button,
    point_style_radio_buttons: [Button; 8],

    fill_label: Label<'static>,
    fill_enabled_button: Button,

    drag_label: Label<'static>,
    drag_enabled_button: Button,
    drag_mode_radio_buttons: [Button; 3],

    color_buttons: [Button; N_EXPRESSION_COLORS],
}

struct StyleGutterLayout {
    toggle_button_center: DVec2,
    toggle_button_radius: f64,
    toggle_button: Bounds,
}

struct StylePopupTitleLayout {
    top: f64,
    cursor: DVec2,
    toggle_bar: Bounds,
    toggle_point: Bounds,
    toggle_hitbox: Bounds,
}

struct StylePopupIconAndFieldLayout {
    icon: Bounds,
    field: Bounds,
}

struct StylePopupLineLayout {
    style_buttons: [Bounds; 3],
    style_button_hitboxes: [Bounds; 3],
    opacity: StylePopupIconAndFieldLayout,
    width: StylePopupIconAndFieldLayout,
}

struct StylePopupPointLayout {
    style_buttons: [Bounds; 8],
    style_button_hitboxes: [Bounds; 8],
    opacity: StylePopupIconAndFieldLayout,
    size: StylePopupIconAndFieldLayout,
}

struct StylePopupFillLayout {
    opacity: StylePopupIconAndFieldLayout,
}

struct StylePopupDragLayout {
    mode_buttons: [Bounds; 3],
    mode_button_hitboxes: [Bounds; 3],
}

struct StylePopupColorLayout {
    top: f64,
    color_buttons: [Bounds; N_EXPRESSION_COLORS],
    color_latex: StylePopupIconAndFieldLayout,
}

struct StylePopupLayout {
    line: Option<(usize, StylePopupTitleLayout, Option<StylePopupLineLayout>)>,
    point: Option<(usize, StylePopupTitleLayout, Option<StylePopupPointLayout>)>,
    fill: Option<(usize, StylePopupTitleLayout, Option<StylePopupFillLayout>)>,
    drag: Option<(usize, StylePopupTitleLayout, Option<StylePopupDragLayout>)>,
    color: Option<StylePopupColorLayout>,
    arrow: Option<Bounds>,
    bounds: Bounds,
}

impl StyleGutter {
    fn layout_gutter(
        &self,
        ctx: &Context,
        bounds: Bounds,
        style: &ExpressionStyle,
    ) -> Option<StyleGutterLayout> {
        if style.popup_order().is_empty() {
            return None;
        }

        let toggle_button_center = bounds.pos + bounds.size.x * dvec2(0.5, 0.752);
        let toggle_button_radius = 0.392 * bounds.size.x;
        let toggle_button = ctx.roundb(Bounds {
            pos: toggle_button_center - toggle_button_radius,
            size: DVec2::splat(toggle_button_radius * 2.0),
        });

        Some(StyleGutterLayout {
            toggle_button_center,
            toggle_button_radius,
            toggle_button,
        })
    }

    fn update_gutter(
        &mut self,
        ctx: &Context,
        event: &Event,
        bounds: Bounds,
        style: &mut ExpressionStyle,
    ) -> Response {
        let mut response = Response::default();
        let Some(l) = self.layout_gutter(ctx, bounds, style) else {
            return response;
        };
        response = response.or(self.toggle_button.update(
            ctx,
            event,
            l.toggle_button_center.distance(ctx.cursor) <= l.toggle_button_radius,
            |response| {
                if ctx.modifiers.shift_key() || self.is_popup_open {
                    self.is_popup_open ^= true;
                    response.request_redraw();
                    return;
                }

                if !style.line_enabled() && !style.point_enabled() && !style.fill_enabled() {
                    style.line.enabled = None;
                    style.point.enabled = None;
                    style.fill.enabled = None;
                } else {
                    style.hidden ^= true;
                }

                style.changed = true;
                response.request_redraw();
            },
        ));

        if self.should_close_popup {
            self.should_close_popup = false;
            self.is_popup_open = false;
            response.request_redraw();
        }

        response
    }

    fn render_gutter(
        &mut self,
        ctx: &Context,
        bounds: Bounds,
        expression_has_focus: bool,
        style: &ExpressionStyle,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        let Some(l) = self.layout_gutter(ctx, bounds, style) else {
            return;
        };

        if style.hidden || !style.line_enabled() && !style.point_enabled() && !style.fill_enabled()
        {
            draw_quad(Quad::from_bounds(
                l.toggle_button,
                QuadKind::ExpressionHiddenIcon,
                if expression_has_focus {
                    let opacity = [0.6f64, 0.7, 0.8][self.toggle_button.state_pressed()];
                    ([255u8; 3], opacity)
                } else {
                    let opacity = [0.17, 0.23, 0.32][self.toggle_button.state_pressed()];
                    ([0; 3], opacity)
                },
            ));
        } else {
            let x = [1.0, 0.95, 0.9][self.toggle_button.state_pressed()];
            let darken = dvec4(x, x, x, 1.0);

            for (i, &color) in self.color.iter().enumerate() {
                draw_quad(
                    Quad::from_bounds(
                        l.toggle_button,
                        QuadKind::ExpressionShownIcon,
                        color * darken,
                    )
                    .clip(Bounds {
                        pos: dvec2(
                            mix(
                                l.toggle_button.left(),
                                l.toggle_button.right(),
                                i as f64 / self.color.len() as f64,
                            ),
                            l.toggle_button.pos.y,
                        ),
                        size: l.toggle_button.size / dvec2(self.color.len() as f64, 1.0),
                    }),
                );
            }

            let white = DVec4::ONE * darken;

            if style.kind == ExpressionStyleKind::PointList && style.line_enabled() {
                draw_quad(Quad::from_bounds(
                    l.toggle_button,
                    QuadKind::LinesIcon,
                    white,
                ));
            }

            if style.kind == ExpressionStyleKind::Polygon && style.fill_enabled() {
                draw_quad(Quad::from_bounds(
                    l.toggle_button,
                    QuadKind::PolygonFilledIcon,
                    white.with_opacity(0.5),
                ));
            }

            if style.kind == ExpressionStyleKind::Parametric && style.fill_enabled() {
                draw_quad(Quad::from_bounds(
                    l.toggle_button,
                    QuadKind::SineFilledIcon,
                    white.with_opacity(0.5),
                ));
            }

            let mut draw = |kind| draw_quad(Quad::from_bounds(l.toggle_button, kind, white));

            match style.kind {
                ExpressionStyleKind::Equality | ExpressionStyleKind::Parametric
                    if style.line_enabled() =>
                {
                    draw(match style.line.style {
                        LineStyle::Solid => QuadKind::SineSolidIcon,
                        LineStyle::Dashed => QuadKind::SineDashedIcon,
                        LineStyle::Dotted => QuadKind::SineDottedIcon,
                    })
                }
                ExpressionStyleKind::Point => draw(style.point.style.gutter_quad_kind()),
                ExpressionStyleKind::PointList if style.point_enabled() => {
                    draw(QuadKind::PointsIcon)
                }
                ExpressionStyleKind::DraggablePoint => {
                    draw(match (style.drag_enabled(), style.drag.mode) {
                        (false, _) => style.point.style.gutter_quad_kind(),
                        (true, DragMode::X) => QuadKind::GutterDragXIcon,
                        (true, DragMode::Y) => QuadKind::GutterDragYIcon,
                        (true, DragMode::XY) => QuadKind::GutterDragXYIcon,
                    })
                }
                ExpressionStyleKind::Polygon if style.line_enabled() => {
                    draw(match style.line.style {
                        LineStyle::Solid => QuadKind::PolygonSolidIcon,
                        LineStyle::Dashed => QuadKind::PolygonDashedIcon,
                        LineStyle::Dotted => QuadKind::PolygonDottedIcon,
                    })
                }
                _ => {}
            };
        }
    }

    fn layout_popup(
        &mut self,
        ctx: &Context,
        expression_list_bounds: Bounds,
        gutter_bounds: Bounds,
        style: &ExpressionStyle,
    ) -> Option<StylePopupLayout> {
        if !self.is_popup_open {
            return None;
        }
        let Some(g) = self.layout_gutter(ctx, gutter_bounds, style) else {
            self.is_popup_open = false;
            return None;
        };

        let line_enabled = !style.hidden && style.line_enabled();
        let point_enabled = !style.hidden && style.point_enabled();
        let fill_enabled = !style.hidden && style.fill_enabled();
        let drag_enabled = !style.hidden && style.drag_enabled();

        let border_width = 1.0;
        let popup_width = 220.0 + 2.0 * border_width;
        let side_padding = 10.0 + border_width;
        let title_height = 38.0;

        let arrow_tip =
            (g.toggle_button.pos + g.toggle_button.size * dvec2(0.95, 0.5)).map(|x| ctx.round(x));

        let mut next_y = arrow_tip.y - 19.0;
        let top = next_y;
        next_y += border_width;

        let icon_height = 14.0;
        struct IconAndFieldY {
            field_size: DVec2,
            icon_y: f64,
            field_y: f64,
        }

        let icon_and_field_y = |next_y: &mut f64, field: &InlineField| {
            let field_size = field.expression_size(ctx, true);
            let height = max([icon_height, field_size.y]);
            let icon_y = *next_y + (height - icon_height) / 2.0;
            let field_y = *next_y + (height - field_size.y) / 2.0;
            *next_y += height;
            IconAndFieldY {
                field_size,
                icon_y,
                field_y,
            }
        };

        struct LineSectionY {
            style_buttons_y: f64,
            opacity: IconAndFieldY,
            width: IconAndFieldY,
        }
        struct PointSectionY {
            style_buttons_y: f64,
            opacity: IconAndFieldY,
            size: IconAndFieldY,
        }

        struct FillSectionY {
            opacity: IconAndFieldY,
        }

        struct DragSectionY {
            mode_buttons_y: f64,
        }

        struct ColorSectionY {
            color_buttons_y: f64,
            padding: f64,
            color_latex: IconAndFieldY,
        }
        let n_color_buttons_per_row = 6;
        let color_button_size = 30.0;

        let mut line_section = None;
        let mut point_section = None;
        let mut fill_section = None;
        let mut drag_section = None;
        let mut color_section = None;

        for (i, kind) in style.popup_order().iter().enumerate() {
            if i > 0 {
                next_y += border_width;
            }
            let top = next_y;
            next_y += title_height;
            let mut do_color = false;
            use StylePopupSectionKind as K;
            match kind {
                K::Line => {
                    let contents = line_enabled.then(|| {
                        do_color = true;
                        let style_buttons_y = next_y - 2.0;
                        let opacity = icon_and_field_y(&mut next_y, &style.line.opacity.0);
                        next_y += 8.0;
                        let width = icon_and_field_y(&mut next_y, &style.line.width.0);
                        next_y += 10.0;
                        LineSectionY {
                            style_buttons_y,
                            opacity,
                            width,
                        }
                    });
                    line_section = Some((i, top, contents));
                }
                K::Point => {
                    let contents = point_enabled.then(|| {
                        do_color = true;
                        let style_buttons_y = next_y - 2.0;
                        let opacity = icon_and_field_y(&mut next_y, &style.point.opacity.0);
                        next_y += 8.0;
                        let size = icon_and_field_y(&mut next_y, &style.point.size.0);
                        next_y += 10.0;
                        PointSectionY {
                            style_buttons_y,
                            opacity,
                            size,
                        }
                    });
                    point_section = Some((i, top, contents));
                }
                K::Fill => {
                    let contents = fill_enabled.then(|| {
                        do_color = true;
                        let opacity = icon_and_field_y(&mut next_y, &style.fill.opacity.0);
                        next_y += 10.0;
                        FillSectionY { opacity }
                    });
                    fill_section = Some((i, top, contents));
                }
                K::Drag => {
                    let contents = drag_enabled.then(|| {
                        let mode_buttons_y = next_y - 2.0;
                        next_y += 38.0;
                        DragSectionY { mode_buttons_y }
                    });
                    drag_section = Some((i, top, contents));
                }
            }

            if do_color && color_section.is_none() {
                next_y += border_width;
                let top = next_y;
                let contents = {
                    next_y += 10.0;
                    let color_buttons_y = next_y;
                    let padding = (popup_width - 2.0 * (side_padding) - color_button_size)
                        / (n_color_buttons_per_row as f64 - 1.0)
                        - color_button_size;
                    let n_rows = N_EXPRESSION_COLORS.div_ceil(n_color_buttons_per_row);
                    next_y += color_button_size * n_rows as f64 + padding * (n_rows - 1) as f64;
                    next_y += 10.0;
                    let color_latex = icon_and_field_y(&mut next_y, &style.color_latex.0);
                    next_y += 10.0;
                    ColorSectionY {
                        color_buttons_y,
                        padding,
                        color_latex,
                    }
                };
                color_section = Some((top, contents));
            }
        }

        let bottom = next_y + border_width;

        let clearance = 9.0;
        let bottom1 = bottom.min(expression_list_bounds.bottom() - clearance);
        let top1 = top + bottom1 - bottom;
        let top2 = top1.max(expression_list_bounds.top() + clearance);
        let bottom2 = bottom1 + top2 - top1;

        let offset_y = top2 - top;
        let top = top2;
        let bottom = bottom2;

        let arrow_half_height = 9.0;
        let arrow_width = arrow_half_height;
        let corner_radius = 6.0; // hardcoded in quad.wgsl
        let is_arrow_shown = top + corner_radius < arrow_tip.y - arrow_half_height
            && arrow_tip.y + arrow_half_height < bottom - corner_radius;

        let arrow = (is_arrow_shown).then_some(Bounds {
            pos: arrow_tip + dvec2(0.0, -arrow_half_height),
            size: dvec2(arrow_width + border_width, 2.0 * arrow_half_height),
        });

        let left = arrow_tip.x + arrow_width;

        let do_title = |top: f64, scale: f64, enabled: bool| {
            let cursor = dvec2(
                left + side_padding,
                top + (title_height + scale / 2.0) / 2.0 + offset_y,
            );
            let toggle_center_y = top + title_height / 2.0;
            let bar_size = dvec2(30.0, 10.0);
            let toggle_bar = Bounds {
                pos: dvec2(
                    left + popup_width - side_padding - bar_size.x,
                    toggle_center_y + offset_y - bar_size.y / 2.0,
                ),
                size: bar_size,
            };
            let point_size = DVec2::splat(18.0);
            let toggle_point = Bounds {
                pos: dvec2(
                    if enabled {
                        toggle_bar.right() - point_size.x
                    } else {
                        toggle_bar.left()
                    },
                    toggle_center_y + offset_y - point_size.y / 2.0,
                ),
                size: point_size,
            };
            let hitbox_expansion = dvec2(5.0, 7.0);
            let toggle_hitbox = Bounds {
                pos: dvec2(
                    left + popup_width - side_padding - bar_size.x - hitbox_expansion.x,
                    toggle_center_y + offset_y - bar_size.y / 2.0 - hitbox_expansion.y,
                ),
                size: bar_size + hitbox_expansion * 2.0,
            };
            StylePopupTitleLayout {
                top: top + offset_y,
                cursor,
                toggle_bar,
                toggle_point,
                toggle_hitbox,
            }
        };

        let do_icon_and_field = |s: IconAndFieldY| {
            let icon = ctx.roundb(Bounds {
                pos: dvec2(left + side_padding, s.icon_y + offset_y),
                size: DVec2::splat(icon_height),
            });
            let field = ctx.roundb(Bounds {
                pos: dvec2(icon.right() + 3.0, s.field_y + offset_y),
                size: s.field_size,
            });
            StylePopupIconAndFieldLayout { icon, field }
        };

        let line = line_section.map(|(i, top, contents)| {
            let title = do_title(top, self.lines_label.scale, line_enabled);
            let contents = contents.map(|s| {
                let style_button_size = dvec2(33.0, 30.0);
                let style_buttons = std::array::from_fn::<_, 3, _>(|i| Bounds {
                    pos: dvec2(
                        left + popup_width - side_padding - border_width
                            + (i as f64 - 3.0) * (style_button_size.x - border_width),
                        s.style_buttons_y + offset_y,
                    ),
                    size: style_button_size,
                });
                let style_button_hitboxes = style_buttons.map(|b| Bounds {
                    pos: b.pos + dvec2(border_width * 0.5, 0.0),
                    size: b.size - dvec2(border_width, 0.0),
                });

                StylePopupLineLayout {
                    style_buttons,
                    style_button_hitboxes,
                    opacity: do_icon_and_field(s.opacity),
                    width: do_icon_and_field(s.width),
                }
            });
            (i, title, contents)
        });

        let point = point_section.map(|(i, top, contents)| {
            let title = do_title(top, self.points_label.scale, point_enabled);
            let contents = contents.map(|s| {
                let style_button_size = dvec2(28.0, 28.0);
                let style_buttons = std::array::from_fn::<_, 8, _>(|i| {
                    let x = i % 4;
                    let y = i / 4;
                    Bounds {
                        pos: dvec2(
                            left + popup_width - side_padding - border_width
                                + (x as f64 - 4.0) * (style_button_size.x - border_width),
                            s.style_buttons_y
                                + (y as f64) * (style_button_size.y - border_width)
                                + offset_y,
                        ),
                        size: style_button_size,
                    }
                });
                let style_button_hitboxes = style_buttons.map(|b| Bounds {
                    pos: b.pos + border_width * 0.5,
                    size: b.size - border_width,
                });

                StylePopupPointLayout {
                    style_buttons,
                    style_button_hitboxes,
                    opacity: do_icon_and_field(s.opacity),
                    size: do_icon_and_field(s.size),
                }
            });
            (i, title, contents)
        });

        let fill = fill_section.map(|(i, top, contents)| {
            let title = do_title(top, self.fill_label.scale, fill_enabled);
            let contents = contents.map(|s| StylePopupFillLayout {
                opacity: do_icon_and_field(s.opacity),
            });
            (i, title, contents)
        });

        let drag = drag_section.map(|(i, top, contents)| {
            let title = do_title(top, self.drag_label.scale, drag_enabled);
            let contents = contents.map(|s| {
                let mode_button_size = dvec2(33.0, 30.0);
                let mode_buttons = std::array::from_fn::<_, 3, _>(|i| Bounds {
                    pos: dvec2(
                        left + side_padding + i as f64 * (mode_button_size.x - border_width),
                        s.mode_buttons_y + offset_y,
                    ),
                    size: mode_button_size,
                });
                let mode_button_hitboxes = mode_buttons.map(|b| Bounds {
                    pos: b.pos + dvec2(border_width * 0.5, 0.0),
                    size: b.size - dvec2(border_width, 0.0),
                });

                StylePopupDragLayout {
                    mode_buttons,
                    mode_button_hitboxes,
                }
            });
            (i, title, contents)
        });

        let color = color_section.map(|(top, c)| {
            let corner = dvec2(left + side_padding, c.color_buttons_y + offset_y);
            let color_buttons = std::array::from_fn::<_, N_EXPRESSION_COLORS, _>(|i| {
                let x = i % n_color_buttons_per_row;
                let y = i / n_color_buttons_per_row;
                ctx.roundb(Bounds {
                    pos: corner + dvec2(x as f64, y as f64) * (color_button_size + c.padding),
                    size: DVec2::splat(color_button_size),
                })
            });
            let color_latex = do_icon_and_field(c.color_latex);

            StylePopupColorLayout {
                top: top + offset_y,
                color_buttons,
                color_latex,
            }
        });

        let bounds = Bounds {
            pos: dvec2(left, top),
            size: dvec2(popup_width, bottom - top),
        };

        Some(StylePopupLayout {
            line,
            point,
            fill,
            drag,
            color,
            arrow,
            bounds,
        })
    }

    fn update_popup(
        &mut self,
        ctx: &Context,
        event: &Event,
        expression_list_bounds: Bounds,
        gutter_bounds: Bounds,
        style: &mut ExpressionStyle,
    ) -> (Response, Option<Message>) {
        let mut response = Response::default();
        let mut message = None;
        let Some(l) = self.layout_popup(ctx, expression_list_bounds, gutter_bounds, style) else {
            return (response, message);
        };

        let update_title = |l: &StylePopupTitleLayout,
                            button: &mut Button,
                            hidden: &mut bool,
                            enabled: &mut Option<bool>,
                            currently_enabled: bool,
                            changed: &mut bool,
                            response: &mut Response| {
            let (r, clicked) = button.update(ctx, event, l.toggle_hitbox);
            *response = response.or(r);
            if clicked {
                if *hidden {
                    *hidden = false;
                    if !currently_enabled {
                        *enabled = Some(true);
                    }
                } else {
                    *enabled = Some(!currently_enabled);
                }
                *changed = true;
                response.request_redraw();
            }
        };

        let parse = |field: &mut (InlineField, _)| {
            if !field.0.is_empty() {
                field.1 = parse_standalone_expression(&field.0.to_latex());
            }
        };

        if let Some((_, title, contents)) = &l.line {
            let enabled = style.line_enabled();
            update_title(
                title,
                &mut self.line_enabled_button,
                &mut style.hidden,
                &mut style.line.enabled,
                enabled,
                &mut style.changed,
                &mut response,
            );

            if let Some(l) = contents {
                for (i, (button, hitbox)) in
                    zip(&mut self.line_style_radio_buttons, l.style_button_hitboxes).enumerate()
                {
                    let (r, clicked) = button.update(ctx, event, hitbox);
                    response = response.or(r);

                    if clicked {
                        style.line.style =
                            [LineStyle::Solid, LineStyle::Dashed, LineStyle::Dotted][i];
                        style.changed = true;
                        response.request_redraw();
                    }
                }

                let (r, m_opacity) = style.line.opacity.0.update(ctx, event, l.opacity.field);
                response = response.or(r);
                let (r, m_width) = style.line.width.0.update(ctx, event, l.width.field);
                response = response.or(r);

                match m_opacity {
                    Some(Message::Down) => {
                        style.line.opacity.0.unfocus();
                        style.line.width.0.focus();
                        response.request_redraw();
                    }
                    Some(Message::ContentsChanged { .. }) => {
                        parse(&mut style.line.opacity);
                        message = message.or(m_opacity);
                    }
                    _ => {}
                }

                match m_width {
                    Some(Message::Up) => {
                        style.line.opacity.0.focus();
                        style.line.width.0.unfocus();
                        response.request_redraw();
                    }
                    Some(Message::ContentsChanged { .. }) => {
                        parse(&mut style.line.width);
                        message = message.or(m_width);
                    }
                    _ => {}
                }
            }
        }

        if let Some((_, title, contents)) = &l.point {
            let enabled = style.point_enabled();
            update_title(
                title,
                &mut self.point_enabled_button,
                &mut style.hidden,
                &mut style.point.enabled,
                enabled,
                &mut style.changed,
                &mut response,
            );

            if let Some(l) = contents {
                for (i, (button, hitbox)) in
                    zip(&mut self.point_style_radio_buttons, l.style_button_hitboxes).enumerate()
                {
                    let (r, clicked) = button.update(ctx, event, hitbox);
                    response = response.or(r);

                    if clicked {
                        style.point.style = [
                            PointStyle::Point,
                            PointStyle::Open,
                            PointStyle::Cross,
                            PointStyle::Square,
                            PointStyle::Plus,
                            PointStyle::Triangle,
                            PointStyle::Diamond,
                            PointStyle::Star,
                        ][i];
                        style.changed = true;
                        response.request_redraw();
                    }
                }

                let (r, m_opacity) = style.point.opacity.0.update(ctx, event, l.opacity.field);
                response = response.or(r);
                let (r, m_size) = style.point.size.0.update(ctx, event, l.size.field);
                response = response.or(r);

                match m_opacity {
                    Some(Message::Down) => {
                        style.point.opacity.0.unfocus();
                        style.point.size.0.focus();
                        response.request_redraw();
                    }
                    Some(Message::ContentsChanged { .. }) => {
                        parse(&mut style.point.opacity);
                        message = message.or(m_opacity);
                    }
                    _ => {}
                }

                match m_size {
                    Some(Message::Up) => {
                        style.point.opacity.0.focus();
                        style.point.size.0.unfocus();
                        response.request_redraw();
                    }
                    Some(Message::ContentsChanged { .. }) => {
                        parse(&mut style.point.size);
                        message = message.or(m_size);
                    }
                    _ => {}
                }
            }
        }

        if let Some((_, title, contents)) = &l.fill {
            let enabled = style.fill_enabled();
            update_title(
                title,
                &mut self.fill_enabled_button,
                &mut style.hidden,
                &mut style.fill.enabled,
                enabled,
                &mut style.changed,
                &mut response,
            );

            if let Some(l) = contents {
                let (r, m_opacity) = style.fill.opacity.0.update(ctx, event, l.opacity.field);
                response = response.or(r);

                if matches!(m_opacity, Some(Message::ContentsChanged { .. })) {
                    parse(&mut style.fill.opacity);
                    message = message.or(m_opacity);
                }
            }
        }

        if let Some((_, title, contents)) = &l.drag {
            let enabled = style.drag_enabled();
            update_title(
                title,
                &mut self.drag_enabled_button,
                &mut style.hidden,
                &mut style.drag.enabled,
                enabled,
                &mut style.changed,
                &mut response,
            );

            if let Some(l) = contents {
                for (i, (button, hitbox)) in
                    zip(&mut self.drag_mode_radio_buttons, l.mode_button_hitboxes).enumerate()
                {
                    let (r, clicked) = button.update(ctx, event, hitbox);
                    response = response.or(r);

                    if clicked {
                        style.drag.mode = [DragMode::X, DragMode::Y, DragMode::XY][i];
                        style.changed = true;
                        response.request_redraw();
                    }
                }
            }
        }

        if let Some(l) = &l.color {
            for (i, (button, hitbox)) in zip(&mut self.color_buttons, l.color_buttons).enumerate() {
                let (r, clicked) = button.update(ctx, event, hitbox);
                response = response.or(r);

                if clicked {
                    style.set_color(EXPRESSION_COLORS[i]);
                    style.changed = true;
                    message = message.or(Some(Message::ContentsChanged { user_driven: true }));
                    response.request_redraw();
                }
            }

            let (r, m_color) = style.color_latex.0.update(ctx, event, l.color_latex.field);
            response = response.or(r);

            if matches!(m_color, Some(Message::ContentsChanged { .. })) {
                parse(&mut style.color_latex);
                message = message.or(m_color);
            }
        }

        // TODO figure out more robust way to make popups steal the correct inputs from things beneath them
        if l.bounds.contains(ctx.cursor) {
            let mut r = Response::default();
            if matches!(
                event,
                Event::MouseInput(ElementState::Pressed, _)
                    | Event::PinchGesture(_)
                    | Event::MouseWheel(_)
            ) {
                r.consume_event();
            }
            r.cursor_mode = CursorMode::Icon(CursorIcon::Default);
            response = response.or(r)
        } else if matches!(event, Event::MouseInput(ElementState::Pressed, _))
            && let Some(l) = self.layout_gutter(ctx, gutter_bounds, style)
            && l.toggle_button_center.distance(ctx.cursor) > l.toggle_button_radius
        {
            // don't just set self.is_popup_open = true because then we might
            // accidentally reopen if we had clicked on toggle button
            self.should_close_popup = true;
            response.request_redraw();
        }

        (response, message)
    }

    fn render_popup(
        &mut self,
        ctx: &Context,
        expression_list_bounds: Bounds,
        gutter_bounds: Bounds,
        style: &mut ExpressionStyle,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        let Some(l) = self.layout_popup(ctx, expression_list_bounds, gutter_bounds, style) else {
            return;
        };

        draw_popup_container(l.bounds, l.arrow, draw_quad);

        let popup_bounds = l.bounds;

        fn render_separator(popup_bounds: Bounds, top: f64, draw_quad: &mut impl FnMut(Quad)) {
            draw_quad(Quad::rectangle(
                (popup_bounds.left() + 1.0, top - 1.0),
                (popup_bounds.right() - 1.0, top),
                [226; 3],
            ));
        }

        fn render_title<T>(
            (index, l, contents): &(usize, StylePopupTitleLayout, Option<T>),
            button: &Button,
            popup_bounds: Bounds,
            label: &Label,
            draw_quad: &mut impl FnMut(Quad),
        ) {
            if *index > 0 {
                render_separator(popup_bounds, l.top, draw_quad);
            }

            label.render_from_cursor(l.cursor, [0; 3], draw_quad);

            draw_quad(Quad::from_bounds(l.toggle_bar, QuadKind::Pill, [221; 3]));
            let shadow_radius = 2.0; // hardcoded in quad.wgsl
            let shadow_offset = dvec2(0.0, 2.0);
            draw_quad(Quad {
                kind: QuadKind::PopupToggleShadow,
                p0: l.toggle_point.pos - shadow_radius + shadow_offset,
                p1: l.toggle_point.pos + l.toggle_point.size + shadow_radius + shadow_offset,
                color: dvec4(0.0, 0.0, 0.0, 0.2),
                ..Default::default()
            });
            draw_quad(Quad {
                kind: QuadKind::PopupToggleShadow,
                p0: l.toggle_point.pos - shadow_radius,
                p1: l.toggle_point.pos + l.toggle_point.size + shadow_radius,
                color: dvec4(0.0, 0.0, 0.0, 0.2),
                ..Default::default()
            });
            draw_quad(Quad::from_bounds(l.toggle_point, QuadKind::Pill, {
                let darken = [1.0, 0.96, 0.91][button.state()];
                if contents.is_some() {
                    [102; 3]
                } else {
                    [245; 3]
                }
                .to_rgbaf64()
                    * dvec4(darken, darken, darken, 1.0)
            }));
        }

        fn render_icon_and_field(
            ctx: &Context,
            l: &StylePopupIconAndFieldLayout,
            icon: QuadKind,
            field: &mut InlineField,
            draw_quad: &mut impl FnMut(Quad),
        ) {
            let icon_bounds = if icon == QuadKind::PaintBucketIcon {
                l.icon.grow(l.icon.size.x * 0.07)
            } else {
                l.icon
            };
            draw_quad(Quad::from_bounds(icon_bounds, icon, [148; 3]));
            field.render(ctx, l.field, draw_quad);
        }

        if let Some(l) = &l.line {
            render_title(
                l,
                &self.line_enabled_button,
                popup_bounds,
                &self.lines_label,
                draw_quad,
            );

            if let Some(l) = &l.2 {
                let mut order = [
                    (0, LineStyle::Solid),
                    (1, LineStyle::Dashed),
                    (2, LineStyle::Dotted),
                ];
                order.sort_by_key(|(i, s)| {
                    if *s == style.line.style {
                        4
                    } else {
                        self.line_style_radio_buttons[*i].state()
                    }
                });

                for (i, s) in order {
                    let b = &l.style_buttons[i];
                    draw_quad(Quad {
                        kind: match (i, s == style.line.style) {
                            (0, true) => QuadKind::PopupRadioSelectedLeft,
                            (0, false) => QuadKind::PopupRadioLeft,
                            (2, true) => QuadKind::PopupRadioSelectedRight,
                            (2, false) => QuadKind::PopupRadioRight,
                            (_, true) => QuadKind::PopupRadioSelectedMiddle,
                            (_, false) => QuadKind::PopupRadioMiddle,
                        },
                        p0: b.pos,
                        p1: b.pos + b.size,
                        color: if s == style.line.style {
                            PRIMARY_COLOR
                        } else {
                            [[255, 245, 204][self.line_style_radio_buttons[i].state()]; 3]
                        }
                        .to_rgbaf64(),
                        ..Default::default()
                    });
                    let center = b.pos + b.size / 2.0;
                    let size = b.size.min_element() * 0.583;
                    draw_quad(
                        Quad {
                            kind: s.into(),
                            p0: center - size / 2.0,
                            p1: center + size / 2.0,
                            color: if s == style.line.style {
                                PRIMARY_COLOR
                            } else {
                                [[148, 40, 0][self.line_style_radio_buttons[i].state()]; 3]
                            }
                            .to_rgbaf64(),
                            ..Default::default()
                        }
                        .pixel_snap(ctx),
                    );
                }

                render_icon_and_field(
                    ctx,
                    &l.opacity,
                    QuadKind::OpacityIcon,
                    &mut style.line.opacity.0,
                    draw_quad,
                );
                render_icon_and_field(
                    ctx,
                    &l.width,
                    QuadKind::ThicknessIcon,
                    &mut style.line.width.0,
                    draw_quad,
                );
            }
        }

        if let Some(l) = &l.point {
            render_title(
                l,
                &self.point_enabled_button,
                popup_bounds,
                &self.points_label,
                draw_quad,
            );

            if let Some(l) = &l.2 {
                let mut order = [
                    (0, PointStyle::Point),
                    (1, PointStyle::Open),
                    (2, PointStyle::Cross),
                    (3, PointStyle::Square),
                    (4, PointStyle::Plus),
                    (5, PointStyle::Triangle),
                    (6, PointStyle::Diamond),
                    (7, PointStyle::Star),
                ];
                order.sort_by_key(|(i, s)| {
                    if *s == style.point.style {
                        4
                    } else {
                        self.point_style_radio_buttons[*i].state()
                    }
                });

                for (i, s) in order {
                    let b = &l.style_buttons[i];
                    draw_quad(Quad {
                        kind: match (i, s == style.point.style) {
                            (0, true) => QuadKind::PopupRadioSelectedTopLeft,
                            (0, false) => QuadKind::PopupRadioTopLeft,
                            (3, true) => QuadKind::PopupRadioSelectedTopRight,
                            (3, false) => QuadKind::PopupRadioTopRight,
                            (4, true) => QuadKind::PopupRadioSelectedBottomLeft,
                            (4, false) => QuadKind::PopupRadioBottomLeft,
                            (7, true) => QuadKind::PopupRadioSelectedBottomRight,
                            (7, false) => QuadKind::PopupRadioBottomRight,
                            (_, true) => QuadKind::PopupRadioSelectedMiddle,
                            (_, false) => QuadKind::PopupRadioMiddle,
                        },
                        p0: b.pos,
                        p1: b.pos + b.size,
                        color: if s == style.point.style {
                            PRIMARY_COLOR
                        } else {
                            [[255, 245, 204][self.point_style_radio_buttons[i].state()]; 3]
                        }
                        .to_rgbaf64(),
                        ..Default::default()
                    });
                    let center = b.pos + b.size / 2.0;
                    let size = b.size.min_element() * 0.4;
                    draw_quad(
                        Quad {
                            kind: s.popup_quad_kind(),
                            p0: center - size / 2.0,
                            p1: center + size / 2.0,
                            color: if s == style.point.style {
                                PRIMARY_COLOR
                            } else {
                                [[148, 40, 0][self.point_style_radio_buttons[i].state()]; 3]
                            }
                            .to_rgbaf64(),
                            ..Default::default()
                        }
                        .pixel_snap(ctx),
                    );
                }

                render_icon_and_field(
                    ctx,
                    &l.opacity,
                    QuadKind::OpacityIcon,
                    &mut style.point.opacity.0,
                    draw_quad,
                );
                render_icon_and_field(
                    ctx,
                    &l.size,
                    QuadKind::ThicknessIcon,
                    &mut style.point.size.0,
                    draw_quad,
                );
            }
        }

        if let Some(l) = &l.fill {
            render_title(
                l,
                &self.fill_enabled_button,
                popup_bounds,
                &self.fill_label,
                draw_quad,
            );

            if let Some(l) = &l.2 {
                render_icon_and_field(
                    ctx,
                    &l.opacity,
                    QuadKind::OpacityIcon,
                    &mut style.fill.opacity.0,
                    draw_quad,
                );
            }
        }

        if let Some(l) = &l.drag {
            render_title(
                l,
                &self.drag_enabled_button,
                popup_bounds,
                &self.drag_label,
                draw_quad,
            );

            if let Some(l) = &l.2 {
                let mut order = [(0, DragMode::X), (1, DragMode::Y), (2, DragMode::XY)];
                order.sort_by_key(|(i, s)| {
                    if *s == style.drag.mode {
                        4
                    } else {
                        self.drag_mode_radio_buttons[*i].state()
                    }
                });

                for (i, s) in order {
                    let b = &l.mode_buttons[i];
                    draw_quad(Quad {
                        kind: match (i, s == style.drag.mode) {
                            (0, true) => QuadKind::PopupRadioSelectedLeft,
                            (0, false) => QuadKind::PopupRadioLeft,
                            (2, true) => QuadKind::PopupRadioSelectedRight,
                            (2, false) => QuadKind::PopupRadioRight,
                            (_, true) => QuadKind::PopupRadioSelectedMiddle,
                            (_, false) => QuadKind::PopupRadioMiddle,
                        },
                        p0: b.pos,
                        p1: b.pos + b.size,
                        color: if s == style.drag.mode {
                            PRIMARY_COLOR
                        } else {
                            [[255, 245, 204][self.drag_mode_radio_buttons[i].state()]; 3]
                        }
                        .to_rgbaf64(),
                        ..Default::default()
                    });
                    let center = b.pos + b.size / 2.0;
                    let size = b.size.min_element() * 0.9;
                    draw_quad(
                        Quad {
                            kind: match s {
                                DragMode::X => QuadKind::PopupDragXIcon,
                                DragMode::Y => QuadKind::PopupDragYIcon,
                                DragMode::XY => QuadKind::PopupDragXYIcon,
                            },
                            p0: center - size / 2.0,
                            p1: center + size / 2.0,
                            color: if s == style.drag.mode {
                                PRIMARY_COLOR
                            } else {
                                [[148, 40, 0][self.drag_mode_radio_buttons[i].state()]; 3]
                            }
                            .to_rgbaf64(),
                            ..Default::default()
                        }
                        .pixel_snap(ctx),
                    );
                }
            }
        }

        if let Some(l) = &l.color {
            render_separator(popup_bounds, l.top, draw_quad);

            for (i, (button, &bounds)) in
                zip(&self.color_buttons, l.color_buttons.iter()).enumerate()
            {
                draw_quad(Quad {
                    p0: bounds.pos - 2.0,
                    p1: bounds.pos + bounds.size + 2.0,
                    kind: QuadKind::PopupColorSwatchHighlight,
                    color: ([0; 3], [0.0, 0.1, 0.2][button.state()]).to_rgbaf64(),
                    ..Default::default()
                });
                let color = EXPRESSION_COLORS[i % EXPRESSION_COLORS.len()];
                draw_quad(Quad::from_bounds(bounds, QuadKind::PopupColorSwatch, color));
                if color == style.color
                    && (style.color_latex.0.is_empty() || style.color_latex.0.underline.error)
                {
                    let center = bounds.pos + bounds.size / 2.0;
                    let size = bounds.size.min_element() * dvec2(1.0, 0.75) * 0.6;
                    let brightness = if color.dot(dvec4(0.2126, 0.7152, 0.0722, 0.0)) > 0.65 {
                        69
                    } else {
                        255
                    };
                    draw_quad(
                        Quad {
                            p0: center - size / 2.0,
                            p1: center + size / 2.0,
                            kind: QuadKind::TickIcon,
                            color: ([brightness; 3], 1.0).to_rgbaf64(),
                            ..Default::default()
                        }
                        .pixel_snap(ctx),
                    );
                }
            }

            render_icon_and_field(
                ctx,
                &l.color_latex,
                QuadKind::PaintBucketIcon,
                &mut style.color_latex.0,
                draw_quad,
            );
        }
    }
}

struct Expression {
    field: MathField,
    slider: Slider,
    parametric_domain: ParametricDomain,
    style: ExpressionStyle,
    ast: Option<Result<parse::ast::Statement, String>>,
    output: Output,
    style_gutter: StyleGutter,
    delete_button: Button,
    /// The cached height from the last update or render, or `None` if it's never
    /// been calculated before.
    height: Option<f64>,
}

fn create_slider_latex<'a>(name_equal_field: &MathField, value: f64) -> latex_tree::Nodes<'a> {
    use latex_tree::Node::Char as C;
    let name = name_equal_field
        .to_latex()
        .iter()
        .take_while(|n| n != &&C('='))
        .cloned()
        .collect::<Vec<_>>();
    let mut latex = name;
    latex.push(C('='));
    latex.extend(value.to_string().chars().map(C));
    latex
}

impl Expression {
    const PADDING: f64 = 16.0;

    fn new(color: DVec4) -> Expression {
        let mut style = ExpressionStyle::default();
        style.set_color(color);
        Expression {
            field: Default::default(),
            slider: Slider {
                hard_min: create_with_placeholder(SLIDER_SOFT_MIN_DEFAULT),
                soft_min: SLIDER_SOFT_MIN_DEFAULT,
                hard_max: create_with_placeholder(SLIDER_SOFT_MAX_DEFAULT),
                soft_max: SLIDER_SOFT_MAX_DEFAULT,
                step: (InlineField::new(""), Ok(ast::Expression::Number(0.0))),
                is_playing: false,
                previous_update_time: Some(0.0),
                play_direction: 1.0,
                animation_period: 4.0,
                loop_mode: Default::default(),
                fake_field: Default::default(),
                fake_field_value: 0.0,
            },
            parametric_domain: Domain {
                min: create_with_placeholder(PARAMETRIC_DOMAIN_MIN_DEFAULT),
                max: create_with_placeholder(PARAMETRIC_DOMAIN_MAX_DEFAULT),
            },
            style,
            ast: None,
            output: Default::default(),
            style_gutter: StyleGutter {
                toggle_button: Default::default(),
                color: Default::default(),

                is_popup_open: false,
                should_close_popup: false,

                lines_label: Label::new("Lines", 16.0, Font::MainRegular),
                line_enabled_button: Default::default(),
                line_style_radio_buttons: Default::default(),

                points_label: Label::new("Points", 16.0, Font::MainRegular),
                point_enabled_button: Default::default(),
                point_style_radio_buttons: Default::default(),

                fill_label: Label::new("Fill", 16.0, Font::MainRegular),
                fill_enabled_button: Default::default(),

                drag_label: Label::new("Drag", 16.0, Font::MainRegular),
                drag_enabled_button: Default::default(),
                drag_mode_radio_buttons: Default::default(),

                color_buttons: Default::default(),
            },
            delete_button: Default::default(),
            height: None,
        }
    }

    /// Returns the expression's height from the previous update or render. The height
    /// is guessed if the expression has never been updated or rendered before. If
    /// possible, you should prefer using the height directly after calling update/render
    /// to avoid the value being stale (e.g., the height will be wrong if a slider UI
    /// was just added and there was no update/render afterwards).
    fn height(&self) -> f64 {
        self.height
            .unwrap_or_else(|| 2.0 * Self::PADDING + self.field.expression_size().y)
    }

    fn from_latex(latex: &[latex_tree::Node], color: DVec4) -> Self {
        let mut e = Expression::new(color);
        e.set_latex(latex);
        e
    }

    const DELETE_BUTTON_PADDING: f64 = 7.0;
    const DELETE_BUTTON_SIZE: f64 = 18.0;

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        top_left: DVec2,
        width: f64,
        is_last: bool,
    ) -> (Response, Option<Message>) {
        let mut response = Response::default();
        let mut message = None;
        let mut height = 0.0;

        let use_fake_field = matches!(self.output.ui, OutputUi::Slider { .. });
        let field = match use_fake_field {
            true => &self.slider.fake_field,
            false => &self.field,
        };
        // TODO why do we round here if later uses end up multiplying by fractional amount anyway
        let padding = ctx.round(Self::PADDING);
        height += padding;
        let field_bounds = Bounds {
            pos: top_left + dvec2(padding, height),
            size: dvec2(
                width - padding - 2.0 * Self::DELETE_BUTTON_PADDING - Self::DELETE_BUTTON_SIZE,
                ctx.ceil(field.expression_size().y),
            ),
        };
        height += field_bounds.size.y;
        // Bounds used for testing if field got clicked on (field_bounds + padding included)
        let field_hit_test_bounds = Bounds {
            pos: top_left,
            size: dvec2(width, field_bounds.size.y + padding * 1.5),
        };
        height += 0.5 * padding;
        let (output_response, new_value, output_message, output_bounds) = self.output.ui.update(
            ctx,
            event,
            padding,
            top_left + dvec2(0.0, height),
            width,
            field.has_focus(),
            &mut self.slider,
            &mut self.parametric_domain,
        );

        // Update new value from slider
        if let Some(value) = new_value {
            self.set_latex(&create_slider_latex(&self.field, value));
            // TODO distinguish between value changed because user dragged
            // slider (user driven) vs slider animation (not user driven)
            message = Some(Message::ContentsChanged { user_driven: false });
        }

        response = response.or(output_response);
        height += output_bounds.size.y;
        height += 0.5 * padding;

        let mut delete_response = Response::default();

        if !is_last {
            let delete_button_hitbox_size = 45.0;
            let delete_button_hitbox = Bounds {
                pos: dvec2(top_left.x + width - delete_button_hitbox_size, top_left.y),
                size: DVec2::splat(delete_button_hitbox_size),
            };

            let delete_clicked;
            (delete_response, delete_clicked) =
                self.delete_button.update(ctx, event, delete_button_hitbox);
            response = response.or(delete_response);

            if delete_clicked {
                message = Some(Message::Remove);
                response.request_redraw();
            }
        }

        let field = match use_fake_field {
            true => &mut self.slider.fake_field,
            false => &mut self.field,
        };
        let (field_response, field_message) = if delete_response.consumed_event {
            (Response::default(), None)
        } else {
            field.update(ctx, event, field_bounds, Some(field_hit_test_bounds))
        };

        if field.has_focus() {
            self.slider.is_playing = false;
        }

        if !field.has_focus() && use_fake_field {
            self.field.unfocus();
        }
        if matches!(field_message, Some(Message::ContentsChanged { .. })) {
            if use_fake_field {
                self.field = self.slider.fake_field.clone();
            }
            self.parse_ast();
            if let Some(Ok(ast::Statement::Assignment { value, .. })) = &self.ast
                && let Some(value) = get_numeric_literal(value)
            {
                self.slider.fake_field_value = value;
                if !use_fake_field {
                    // We just became a slider. Transfer control over to fake field
                    self.slider.fake_field = self.field.clone();
                }

                if let OutputUi::Slider(SliderUi { min, max, step, .. }) = self.output.ui {
                    // maybe using `offset` instead of unconditionally
                    // using `min` reduces floating-point error?
                    let offset = if !self.slider.hard_min.0.is_empty()
                        && let Some(min) = min
                    {
                        min
                    } else {
                        0.0
                    };

                    if min.is_some_and(|min| value < min) {
                        self.slider.hard_min.0.clear();
                    }
                    if max.is_some_and(|max| value > max) {
                        self.slider.hard_max.0.clear();
                    }
                    if max.is_none_or(|max| value != max)
                        && let Some(step) = step
                        && value != apply_slider_step(value, offset, step, f64::round)
                    {
                        self.slider.step.0.clear();
                    }
                }
            }
        }
        if message.is_none() {
            message = field_message;
        }

        response = response.or(field_response);

        // Maybe the parametric domain or slider settings got changed
        if let Some(m) = output_message {
            message = match m {
                Message::ContentsChanged { .. } | Message::Down | Message::Add => Some(m),
                Message::Left | Message::Right | Message::Remove => unreachable!(),
                Message::Up => {
                    self.focus();
                    None
                }
            };
        }

        // Adjust height after field was updated
        height += ctx.ceil(self.field.expression_size().y) - field_bounds.size.y;
        self.height = Some(height);
        (response, message)
    }

    fn update_gutter(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> Response {
        if let OutputUi::Slider(ui) = &mut self.output.ui {
            return ui.update_gutter(ctx, event, &mut self.slider, bounds);
        }
        self.style_gutter
            .update_gutter(ctx, event, bounds, &mut self.style)
    }

    fn update_popup(
        &mut self,
        ctx: &Context,
        event: &Event,
        expression_list_bounds: Bounds,
        gutter_bounds: Bounds,
    ) -> (Response, Option<Message>) {
        if let OutputUi::Slider(ui) = &mut self.output.ui {
            let response = ui.update_popup(
                ctx,
                event,
                &mut self.slider,
                expression_list_bounds,
                gutter_bounds,
            );
            return (response, None);
        }
        self.style_gutter.update_popup(
            ctx,
            event,
            expression_list_bounds,
            gutter_bounds,
            &mut self.style,
        )
    }

    fn set_latex(&mut self, latex: &[latex_tree::Node]) {
        self.field = MathField::from(latex);
        self.parse_ast();

        if let Some(Ok(ast::Statement::Assignment { value, .. })) = &self.ast
            && let Some(value) = get_numeric_literal(value)
        {
            self.slider.fake_field_value = value;
            self.slider.fake_field = self.field.clone();
        }
    }

    fn parse_ast(&mut self) {
        let latex = self.field.to_latex();
        self.ast = latex
            .iter()
            .any(|n| n != &latex_tree::Node::Char(' '))
            .then(|| parse_statement(&latex));
    }

    fn focus(&mut self) {
        self.field.focus();
        self.slider.fake_field.focus();
        self.slider.is_playing = false;
    }

    fn unfocus(&mut self) {
        self.field.unfocus();
        self.slider.fake_field.unfocus();
        self.slider.hard_min.0.unfocus();
        self.slider.hard_max.0.unfocus();
        self.slider.step.0.unfocus();
        self.parametric_domain.min.0.unfocus();
        self.parametric_domain.max.0.unfocus();
    }

    fn has_focus(&self) -> bool {
        self.field.has_focus()
            || match &self.output.ui {
                OutputUi::None | OutputUi::Field(_) => false,
                OutputUi::Slider(_) => {
                    self.slider.fake_field.has_focus()
                        || self.slider.hard_min.0.has_focus()
                        || self.slider.hard_max.0.has_focus()
                        || self.slider.step.0.has_focus()
                }
                OutputUi::ParametricDomain(_) => {
                    self.parametric_domain.min.0.has_focus()
                        || self.parametric_domain.max.0.has_focus()
                }
            }
    }

    fn render(
        &mut self,
        ctx: &Context,
        top_left: DVec2,
        width: f64,
        is_last: bool,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        let mut height = 0.0;

        let use_fake_field = matches!(self.output.ui, OutputUi::Slider { .. });
        let field = match use_fake_field {
            true => &self.slider.fake_field,
            false => &self.field,
        };
        let padding = ctx.round(Self::PADDING);
        height += padding;
        let field_bounds = Bounds {
            pos: top_left + dvec2(padding, height),
            size: dvec2(
                width - padding - 2.0 * Self::DELETE_BUTTON_PADDING - Self::DELETE_BUTTON_SIZE,
                ctx.ceil(field.expression_size().y),
            ),
        };
        height += field_bounds.size.y;
        height += 0.5 * padding;
        height += self.output.ui.render(
            ctx,
            padding,
            top_left + dvec2(0.0, height),
            width,
            field.has_focus(),
            &mut self.slider,
            &mut self.parametric_domain,
            draw_quad,
        );
        height += 0.5 * padding;

        let field = match use_fake_field {
            true => &mut self.slider.fake_field,
            false => &mut self.field,
        };
        field.render(ctx, field_bounds, draw_quad);

        if !is_last {
            let delete_button_bounds = Bounds {
                pos: dvec2(
                    top_left.x + width - Self::DELETE_BUTTON_PADDING - Self::DELETE_BUTTON_SIZE,
                    top_left.y + Self::DELETE_BUTTON_PADDING,
                ),
                size: DVec2::splat(Self::DELETE_BUTTON_SIZE),
            };
            draw_quad(Quad::from_bounds(
                ctx.roundb(delete_button_bounds),
                QuadKind::CrossIcon,
                [[204, 102, 51][self.delete_button.state()]; 3],
            ));
        }

        self.height = Some(height);
    }

    fn render_gutter(
        &mut self,
        ctx: &Context,
        bounds: Bounds,
        has_focus: bool,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        if let OutputUi::Slider(ui) = &mut self.output.ui {
            ui.render_gutter(ctx, bounds, has_focus, &mut self.slider, draw_quad);
        }
        self.style_gutter
            .render_gutter(ctx, bounds, has_focus, &self.style, draw_quad);
    }

    fn render_popup(
        &mut self,
        ctx: &Context,
        expression_list_bounds: Bounds,
        gutter_bounds: Bounds,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        if let OutputUi::Slider(ui) = &mut self.output.ui {
            ui.render_popup(
                ctx,
                &self.slider,
                expression_list_bounds,
                gutter_bounds,
                draw_quad,
            );
        }
        self.style_gutter.render_popup(
            ctx,
            expression_list_bounds,
            gutter_bounds,
            &mut self.style,
            draw_quad,
        );
    }
}

#[derive(Debug, Clone, Copy, From, Into, Add, Sub, PartialEq, PartialOrd)]
pub struct ExpressionId(usize);

pub struct ExpressionList {
    expressions: TiVec<ExpressionId, Expression>,
    expressions_changed: bool,
    redraw_geometry: bool,
    dragged_expression: Option<(ClickDragTracker, ExpressionId, f64)>,
    next_color: usize,
    scroll: f64,
    height: f64,
    vm_vars: vm::Vars,
}

const N_EXPRESSION_COLORS: usize = 6;
const EXPRESSION_COLORS: [DVec4; N_EXPRESSION_COLORS] = [
    dvec4(0.78, 0.267, 0.25, 1.0),
    dvec4(0.176, 0.44, 0.7, 1.0),
    dvec4(0.204, 0.52, 0.263, 1.0),
    dvec4(0.98, 0.494, 0.098, 1.0),
    dvec4(0.376, 0.26, 0.65, 1.0),
    dvec4(0.0, 0.0, 0.0, 1.0),
];

fn get_default_expression_color(i: usize) -> DVec4 {
    // skip orange
    let p = [0, 1, 2, 4, 5];
    EXPRESSION_COLORS[p[i % p.len()]]
}

impl ExpressionList {
    pub fn new() -> Self {
        let expressions = [];
        let mut next_color = 0;
        let expressions = expressions
            .iter()
            .chain(Some(&""))
            .chain(expressions.is_empty().then_some(&""))
            .map(|s| {
                let color = get_default_expression_color(next_color);
                next_color += 1;
                Expression::from_latex(parse_latex(s).unwrap().as_slice(), color)
            })
            .collect();
        Self {
            expressions,
            expressions_changed: true,
            redraw_geometry: true,
            dragged_expression: None,
            next_color,
            scroll: 0.0,
            height: 0.0,
            vm_vars: Default::default(),
        }
    }

    pub fn point_dragged(&mut self, i: ExpressionId, p: DVec2) {
        use parse::latex_tree::Node::{self, Char as C};
        let name = self.expressions[i]
            .field
            .to_latex()
            .iter()
            .take_while(|n| n != &&C('='))
            .cloned()
            .collect::<Vec<_>>();
        let mut latex = name;
        latex.push(C('='));
        let mut inner = vec![];
        inner.extend(p.x.to_string().chars().map(C));
        inner.push(C(','));
        inner.extend(p.y.to_string().chars().map(C));
        latex.push(Node::DelimitedGroup {
            left: Bracket::Paren,
            right: Bracket::Paren,
            inner,
        });
        self.expressions[i].set_latex(&latex);
        self.expressions_changed = true;
    }

    // Positive `delta` moves the expressions down
    fn scroll(&mut self, ctx: &Context, delta: f64) {
        const SCROLL_EXTRA: f64 = 80.0;
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
        let expressions_height = self.expressions
            [..ExpressionId(self.expressions.len().max(1) - 1)] // ignore faded
            .iter()
            .map(|e| e.height() + separator_width)
            .sum::<f64>();
        self.scroll = (self.scroll - delta)
            .min(SCROLL_EXTRA + expressions_height - self.height)
            .max(0.0);
    }

    const SCROLL_PADDING: f64 = 25.0;

    fn scroll_y_into_view(&mut self, ctx: &Context, y: f64) {
        self.scroll(
            ctx,
            (self.height - Self::SCROLL_PADDING - (y - self.scroll)).min(0.0),
        );
        self.scroll(ctx, (Self::SCROLL_PADDING - (y - self.scroll)).max(0.0));
    }

    fn scroll_into_view(&mut self, ctx: &Context, i: ExpressionId) {
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
        let top = self.expressions[ExpressionId(0)..i]
            .iter()
            .map(|e| e.height() + separator_width)
            .sum::<f64>();
        let bottom = top + self.expressions[i].height();
        self.scroll(
            ctx,
            (self.height - Self::SCROLL_PADDING - (bottom - self.scroll)).min(0.0),
        );
        self.scroll(ctx, (Self::SCROLL_PADDING - (top - self.scroll)).max(0.0));
    }

    fn new_expression(&mut self) -> Expression {
        let color = get_default_expression_color(self.next_color);
        self.next_color += 1;
        Expression::new(color)
    }

    const SEPARATOR_WIDTH: f64 = 1.0;
    const GUTTER_WIDTH: f64 = 37.0;

    pub fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        bounds: Bounds,
    ) -> (Response, Option<(Vec<Geometry>, vm::Vars)>) {
        self.height = bounds.size.y;
        let mut response = Response::default();

        if let Some((drag_tracker, i, offset)) = &mut self.dragged_expression {
            if drag_tracker.drag(ctx.cursor) {
                #[cfg(not(windows))]
                let grabbing = CursorIcon::Grabbing;

                // https://github.com/rust-windowing/winit/issues/1043
                #[cfg(windows)]
                let grabbing = CursorIcon::NsResize;

                response.cursor_mode = CursorMode::Icon(grabbing);
            }

            match event {
                Event::MouseInput(ElementState::Released, MouseButton::Left) => {
                    if drag_tracker.release().was_dragged() {
                        response.request_redraw();
                    }
                    self.dragged_expression = None;
                }
                Event::CursorMoved { .. } if drag_tracker.drag(ctx.cursor) => {
                    let i_top = ctx.cursor.y + *offset - (bounds.pos.y - self.scroll);
                    let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
                    let mut new_i = 0;
                    let mut top = 0.0;

                    for (j, expression) in self.expressions.iter_enumerated() {
                        if j == *i {
                            continue;
                        }
                        let middle = top + expression.height() / 2.0;
                        if i_top < middle {
                            break;
                        }
                        top += expression.height() + separator_width;
                        new_i += 1;
                    }

                    // Don't try putting it past the last expression which is just
                    // for the faded-away visual
                    let new_i = ExpressionId(new_i.min(self.expressions.len() - 2));

                    let offset_geometry_id = |expression: &mut Expression, amount: isize| {
                        if let OutputData::DraggablePoint(Geometry {
                            kind:
                                GeometryKind::Point {
                                    draggable: Some((id, _)),
                                    ..
                                },
                            ..
                        }) = &mut expression.output.data
                        {
                            id.0 = (id.0 as isize + amount) as usize;
                        }
                    };

                    if new_i > *i {
                        self.expressions[*i..=new_i].rotate_left(1.into());
                        for expression in &mut self.expressions[*i..new_i] {
                            offset_geometry_id(expression, -1);
                        }
                    } else if new_i < *i {
                        self.expressions[new_i..=*i].rotate_right(1.into());
                        for expression in &mut self.expressions[new_i + 1.into()..=*i] {
                            offset_geometry_id(expression, 1);
                        }
                    }

                    offset_geometry_id(
                        &mut self.expressions[new_i],
                        new_i.0 as isize - i.0 as isize,
                    );

                    self.redraw_geometry |= set(i, new_i);
                    // TODO keep it scrolling even when cursor isn't moving and make it FPS-independent
                    self.scroll_y_into_view(ctx, ctx.cursor.y - (bounds.pos.y - self.scroll));

                    response.consume_event();
                    response.request_redraw();
                }
                _ => {}
            }
        }

        match event {
            Event::MouseWheel(delta)
                if bounds.contains(ctx.cursor) && delta.abs().y >= delta.x.abs() =>
            {
                self.scroll(ctx, delta.y);
                response.consume_event();
                response.request_redraw();
            }
            _ => {
                let mut message = None;
                let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
                let gutter_width = ctx.round_nonzero(Self::GUTTER_WIDTH);
                let expression_width = bounds.size.x - 2.0 * separator_width - gutter_width;
                let expression_left = bounds.pos.x + gutter_width + separator_width;
                let mut expression_top = bounds.pos.y - self.scroll;
                let expressions_len = self.expressions.len();
                let mut original_focus = None;

                for (i, expression) in self.expressions.iter_mut_enumerated() {
                    let has_focus = expression.has_focus();

                    if has_focus {
                        original_focus = Some(i);
                    }

                    let is_last = i.0 == expressions_len - 1;
                    let (r, m) = expression.update(
                        ctx,
                        event,
                        dvec2(expression_left, expression_top),
                        expression_width,
                        is_last,
                    );
                    response = response.or(r);
                    message = message.or(m.map(|m| (i, m)));

                    let gutter_response = expression.update_gutter(
                        ctx,
                        event,
                        Bounds {
                            pos: dvec2(bounds.left(), expression_top),
                            size: dvec2(gutter_width, expression.height()),
                        },
                    );

                    if gutter_response.consumed_event && has_focus {
                        // We would've just lost focus after we clicked on gutter, refocus if so
                        if has_focus {
                            // TODO add this back but make it not let user edit slider without it pausing
                            // expression.focus();
                        }
                    }

                    response = response.or(gutter_response);

                    if set(&mut expression.style.changed, false) {
                        self.redraw_geometry = true;
                    }

                    let drag_bounds = Bounds {
                        pos: dvec2(bounds.left(), expression_top),
                        size: dvec2(gutter_width, expression.height()) + separator_width,
                    };

                    if !is_last && drag_bounds.contains(ctx.cursor) {
                        let mut drag_response = Response::default();

                        if event == &Event::MouseInput(ElementState::Pressed, MouseButton::Left) {
                            self.dragged_expression = Some((
                                if response.consumed_event {
                                    ClickDragTracker::Pressed(ctx.cursor)
                                } else {
                                    ClickDragTracker::Dragging
                                },
                                i,
                                expression_top - ctx.cursor.y,
                            ));
                            drag_response.consume_event();
                            drag_response.request_redraw();

                            // We would've just lost focus after we started dragging, refocus if so
                            if has_focus {
                                // TODO add this back but make it not let user edit slider without it pausing
                                // expression.focus();
                            }
                        }

                        #[cfg(not(windows))]
                        let (grab, grabbing) = (CursorIcon::Grab, CursorIcon::Grabbing);

                        // https://github.com/rust-windowing/winit/issues/1043
                        #[cfg(windows)]
                        let (grab, grabbing) = (CursorIcon::NsResize, CursorIcon::NsResize);

                        drag_response.cursor_mode =
                            CursorMode::Icon(if self.dragged_expression.is_some() {
                                grabbing
                            } else {
                                grab
                            });

                        response = response.or(drag_response);
                    }

                    expression_top += expression.height() + separator_width;
                }

                if let Some((i, m)) = message {
                    match m {
                        Message::ContentsChanged { user_driven } => {
                            self.expressions_changed = true;
                            if user_driven {
                                self.scroll_into_view(ctx, i);
                            }
                        }
                        Message::Left | Message::Right => {}
                        Message::Up => {
                            if i.0 > 0 {
                                self.expressions[i].unfocus();
                                self.expressions[i - 1.into()].focus();
                                response.request_redraw();
                            }
                        }
                        Message::Down => {
                            if i.0 == self.expressions.len() - 1 {
                                let expression = self.new_expression();
                                self.expressions.push(expression);
                            }
                            self.expressions[i].unfocus();
                            self.expressions[i + 1.into()].focus();
                            response.request_redraw();
                        }
                        Message::Add => {
                            self.expressions_changed = true;
                            let expression = self.new_expression();
                            self.expressions.insert(i + 1.into(), expression);
                            self.expressions[i].unfocus();
                            self.expressions[i + 1.into()].focus();
                            response.request_redraw();
                        }
                        Message::Remove => {
                            let had_focus = self.expressions[i].has_focus();
                            self.expressions.remove(i);
                            self.expressions_changed = true;
                            if self.expressions.len() < 2 {
                                let expression = self.new_expression();
                                self.expressions.push(expression);
                            }
                            if had_focus {
                                self.expressions[ExpressionId(i.0.saturating_sub(1))].focus();
                            }
                            response.request_redraw();
                        }
                    }
                }

                if self.expressions.last().unwrap().has_focus() {
                    let expression = self.new_expression();
                    self.expressions.push(expression);
                    response.request_redraw();
                }

                let new_focus = self
                    .expressions
                    .iter_enumerated()
                    .find_map(|(i, e)| e.has_focus().then_some(i));
                self.redraw_geometry |= self.expressions_changed || original_focus != new_focus;

                if let Some(i) = new_focus
                    && original_focus != new_focus
                {
                    self.scroll_into_view(ctx, i);
                }

                if set(&mut self.expressions_changed, false) {
                    use latex_tree::Node::{self, Char as C};
                    let point2 = |nodes: &mut Vec<Node>, x: f64, y: f64| {
                        let mut inner = vec![];
                        number_to_latex(&mut inner, x);
                        inner.push(C(','));
                        number_to_latex(&mut inner, y);
                        nodes.push(Node::DelimitedGroup {
                            left: Bracket::Paren,
                            right: Bracket::Paren,
                            inner,
                        });
                    };
                    let point3 = |nodes: &mut Vec<Node>, x: f64, y: f64, z: f64| {
                        let mut inner = vec![];
                        number_to_latex(&mut inner, x);
                        inner.push(C(','));
                        number_to_latex(&mut inner, y);
                        inner.push(C(','));
                        number_to_latex(&mut inner, z);
                        nodes.push(Node::DelimitedGroup {
                            left: Bracket::Paren,
                            right: Bracket::Paren,
                            inner,
                        });
                    };

                    let mut ei_to_oi: TiVec<ExpressionIndex, ExpressionId> = ti_vec![];
                    let mut list: TiVec<ExpressionIndex, _> = ti_vec![];
                    let mut properties: TiVec<PropertyIndex, _> = ti_vec![];
                    struct ExpressionProperties {
                        line_width: Option<PropertyIndex>,
                        line_opacity: Option<PropertyIndex>,
                        point_size: Option<PropertyIndex>,
                        point_opacity: Option<PropertyIndex>,
                        fill_opacity: Option<PropertyIndex>,
                        color: Option<PropertyIndex>,
                    }
                    let mut oi_to_pi: TiVec<ExpressionId, _> = ti_vec![];

                    for (i, e) in self.expressions.iter_mut_enumerated() {
                        e.style.kind = ExpressionStyleKind::None;

                        fn push<'a>(
                            properties: &mut TiVec<PropertyIndex, &'a ast::Expression>,
                            property: &'a mut (InlineField, Result<ast::Expression, String>),
                        ) -> Option<PropertyIndex> {
                            property.0.underline.error = false;
                            if property.0.is_empty() {
                                None
                            } else {
                                match property.1.as_ref() {
                                    Ok(p) => Some(properties.push_and_get_key(p)),
                                    Err(_) => {
                                        property.0.underline.error = true;
                                        None
                                    }
                                }
                            }
                        }
                        oi_to_pi.push(ExpressionProperties {
                            line_width: push(&mut properties, &mut e.style.line.width),
                            line_opacity: push(&mut properties, &mut e.style.line.opacity),
                            point_size: push(&mut properties, &mut e.style.point.size),
                            point_opacity: push(&mut properties, &mut e.style.point.opacity),
                            fill_opacity: push(&mut properties, &mut e.style.fill.opacity),
                            color: push(&mut properties, &mut e.style.color_latex),
                        });
                        let ast = match &e.ast {
                            Some(Ok(ast)) => ast,
                            Some(Err(err)) => {
                                e.output = Output::new_error(format!("parse error: {err}"));
                                continue;
                            }
                            None => {
                                e.output = Output::NONE;
                                continue;
                            }
                        };
                        let mut slider = None;
                        if let parse::ast::Statement::Assignment { value, .. } = ast {
                            if get_numeric_literal(value).is_some() {
                                e.output.data = OutputData::None;
                                let [min, max, step] =
                                    [&e.slider.hard_min, &e.slider.hard_max, &e.slider.step].map(
                                        |f| (!f.0.is_empty()).then(|| f.1.as_ref().ok()).flatten(),
                                    );
                                slider = Some(NrSlider { min, max, step });
                            } else if let parse::ast::Expression::Op {
                                operation: parse::op::OpName::Point,
                                args: arguments,
                            } = value
                                && arguments.len() == 2
                                && let Some(x) = get_numeric_literal(&arguments[0])
                                && let Some(y) = get_numeric_literal(&arguments[1])
                            {
                                let mut latex = vec![C('=')];
                                point2(&mut latex, x, y);
                                e.output = Output {
                                    ui: OutputUi::None,
                                    data: OutputData::DraggablePoint(Geometry {
                                        original_width: 8.0,
                                        width: 8.0,
                                        color: e.style.color.as_vec4().into(),
                                        line_style: Default::default(),
                                        point_style: Default::default(),
                                        kind: GeometryKind::Point {
                                            p: dvec2(x, y),
                                            draggable: Some((i, DragMode::XY)),
                                        },
                                    }),
                                };
                            } else {
                                e.output.data = OutputData::None;
                            }
                        } else {
                            e.output.data = OutputData::None;
                        }
                        list.push(ExpressionListEntry {
                            expression: ast,
                            parametric_domain: Domain {
                                min: match &e.parametric_domain.min.1 {
                                    Ok(ast) => ast,
                                    Err(_) => &ast::Expression::Number(0.0),
                                },
                                max: match &e.parametric_domain.max.1 {
                                    Ok(ast) => ast,
                                    Err(_) => &ast::Expression::Number(1.0),
                                },
                            },
                            slider,
                        });
                        ei_to_oi.push(i);
                    }

                    let builtin_constants = [
                        ("pi", std::f64::consts::PI),
                        ("tau", std::f64::consts::TAU),
                        ("e", std::f64::consts::E),
                        ("infty", f64::INFINITY),
                    ];
                    let analysis = analyze_expression_list(
                        &list,
                        &builtin_constants.map(|(name, _)| name),
                        &properties,
                        false,
                    );

                    let mut function_id_map = HashMap::new();
                    let (program, mut functions, var_indices) = compile_assignments(
                        analysis.constants.iter().map(|&i| &analysis.assignments[i]),
                        analysis.results.iter_enumerated().filter_map(|(i, r)| {
                            let ExpressionResult::Plot {
                                parameters,
                                assignments,
                                ..
                            } = r
                            else {
                                return None;
                            };
                            function_id_map.insert(i, function_id_map.len());
                            Some((
                                parameters.iter().cloned(),
                                assignments.iter().map(|&i| &analysis.assignments[i]),
                            ))
                        }),
                        analysis
                            .builtin_constants
                            .values()
                            .map(|&id| (id, Type::Number)),
                    );
                    let mut functions = function_id_map
                        .into_iter()
                        .map(|(id, i)| (id, std::mem::take(&mut functions[i])))
                        .collect::<HashMap<_, _>>();
                    let mut vm = Vm::new(
                        &program,
                        Default::default(),
                        analysis
                            .builtin_constants
                            .values()
                            .map(|id| var_indices[id]),
                    );

                    for (name, value) in builtin_constants {
                        vm.vars[var_indices[&analysis.builtin_constants[name]]] =
                            vm::Value::Number(value);
                    }

                    vm.run(false);

                    'results_loop: for (ei, r) in analysis.results.into_iter_enumerated() {
                        let i = ei_to_oi[ei];
                        let expression = &mut self.expressions[i];
                        let output = &mut expression.output;

                        if let OutputData::Error(_) = output.data {
                            continue 'results_loop;
                        }

                        let pi = &oi_to_pi[i];

                        enum Property<T> {
                            Single(T),
                            List(Vec<T>),
                        }
                        impl<T: Copy> Property<T> {
                            fn get(&self, index: usize) -> Option<T> {
                                match self {
                                    Property::Single(x) => Some(*x),
                                    Property::List(xs) => xs.get(index).cloned(),
                                }
                            }
                        }
                        let get_number_property =
                            |index: Option<PropertyIndex>,
                             default: f64,
                             min: f64,
                             max: f64,
                             field: &mut (InlineField, _)| {
                                let Some(pi) = index else {
                                    return Property::Single(default as f32);
                                };
                                let Some(value) =
                                    analysis.properties[pi].as_ref().ok().and_then(|(id, ty)| {
                                        match *ty {
                                            Type::Number => vm.vars[var_indices[id]]
                                                .clone()
                                                .number()
                                                .into_finite()
                                                .map(
                                                    |x| Property::Single(x.clamp(min, max) as f32),
                                                ),
                                            Type::NumberList => vm.vars[var_indices[id]]
                                                .clone()
                                                .list()
                                                .borrow()
                                                .iter()
                                                .map(|x| {
                                                    x.into_finite()
                                                        .map(|x| x.clamp(min, max) as f32)
                                                })
                                                .collect::<Option<_>>()
                                                .map(Property::List),
                                            Type::EmptyList => Some(Property::List(vec![])),
                                            _ => None,
                                        }
                                    })
                                else {
                                    field.0.underline.error = true;
                                    return Property::Single(default as f32);
                                };
                                value
                            };

                        let line_width = get_number_property(
                            pi.line_width,
                            LINE_WIDTH_DEFAULT,
                            0.0,
                            f64::INFINITY,
                            &mut expression.style.line.width,
                        );
                        let line_opacity = get_number_property(
                            pi.line_opacity,
                            LINE_OPACITY_DEFAULT,
                            0.0,
                            1.0,
                            &mut expression.style.line.opacity,
                        );
                        let point_size = get_number_property(
                            pi.point_size,
                            POINT_SIZE_DEFAULT,
                            0.0,
                            f64::INFINITY,
                            &mut expression.style.point.size,
                        );
                        let point_opacity = get_number_property(
                            pi.point_opacity,
                            POINT_OPACITY_DEFAULT,
                            0.0,
                            1.0,
                            &mut expression.style.point.opacity,
                        );
                        let fill_opacity = get_number_property(
                            pi.fill_opacity,
                            FILL_OPACITY_DEFAULT,
                            0.0,
                            1.0,
                            &mut expression.style.fill.opacity,
                        );
                        let color = 'color: {
                            let index = pi.color;
                            let default = expression.style.color.as_vec4().to_array();
                            let Some(pi) = index else {
                                break 'color Property::Single(default);
                            };
                            let Some(value) = analysis.properties[pi].as_ref().ok().and_then(
                                |(id, ty)| match *ty {
                                    Type::Color => {
                                        let v = var_indices[id];
                                        let r = vm.vars[v + 0.into()].clone().number();
                                        let g = vm.vars[v + 1.into()].clone().number();
                                        let b = vm.vars[v + 2.into()].clone().number();
                                        Some(Property::Single([r as f32, g as f32, b as f32, 1.0]))
                                    }
                                    Type::ColorList => Some(Property::List(
                                        vm.vars[var_indices[id]]
                                            .clone()
                                            .list()
                                            .borrow()
                                            .chunks(3)
                                            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, 1.0])
                                            .collect(),
                                    )),
                                    Type::EmptyList => Some(Property::List(vec![])),
                                    _ => None,
                                },
                            ) else {
                                expression.style.color_latex.0.underline.error = true;
                                break 'color Property::Single(default);
                            };
                            value
                        };
                        let apply_opacity = |mut color: [f32; 4], opacity: f32| {
                            color[3] *= opacity;
                            color
                        };
                        expression.style_gutter.color = match &color {
                            Property::Single(c) => vec![c.map(|x| x as f64).into()],
                            Property::List(cs) if !cs.is_empty() => {
                                let n = cs.len().min(10);
                                (0..n)
                                    .map(|i| {
                                        cs[(i as f64 / n as f64 * cs.len() as f64) as usize]
                                            .map(|x| x as f64)
                                            .into()
                                    })
                                    .collect()
                            }
                            _ => vec![[0; 3].to_rgbaf64()],
                        };

                        match r {
                            ExpressionResult::None => *output = Output::NONE,
                            ExpressionResult::Err(e) => {
                                *output = Output::new_error(format!("analysis error: {e}"))
                            }
                            ExpressionResult::Value(id, ty)
                            | ExpressionResult::Plot { value: id, ty, .. } => {
                                let mut nodes = vec![C('=')];

                                let mut geometry = vec![];
                                let make_point = |x: f64, y: f64, i: usize| match (
                                    point_size.get(i),
                                    point_opacity.get(i),
                                    color.get(i),
                                ) {
                                    (Some(size), Some(opacity), Some(color)) => Some(Geometry {
                                        original_width: size,
                                        width: size,
                                        color: apply_opacity(color, opacity),
                                        line_style: Default::default(),
                                        point_style: Default::default(),
                                        kind: GeometryKind::Point {
                                            p: dvec2(x, y),
                                            draggable: None,
                                        },
                                    }),
                                    _ => None,
                                };
                                let list_limit = 10;

                                if let ExpressionResult::Plot {
                                    ref kind,
                                    value,
                                    ref parameters,
                                    ..
                                } = r
                                {
                                    let kind = match kind {
                                        PlotKind::Normal => {
                                            expression.style.kind = ExpressionStyleKind::Equality;
                                            PlotKind::Normal
                                        }
                                        PlotKind::Inverse => {
                                            expression.style.kind = ExpressionStyleKind::Equality;
                                            PlotKind::Inverse
                                        }
                                        PlotKind::Parametric(d) => {
                                            output.ui.set_parametric_domain(
                                                &analysis.freevars[&parameters[0]],
                                            );

                                            let min = match &expression.parametric_domain.min.1 {
                                                Ok(_) => match &d.min {
                                                    Ok(id) => match vm.vars[var_indices[id]]
                                                        .clone()
                                                        .number()
                                                    {
                                                        x if x.is_finite() => Ok(x),
                                                        _ => Err(
                                                            "value error: domain bound should be finite".into(),
                                                        ),
                                                    },
                                                    Err(e) => Err(format!("analysis error: {e}")),
                                                },
                                                Err(e) => Err(format!("parse error: {e}")),
                                            };
                                            let max = match &expression.parametric_domain.max.1 {
                                                Ok(_) => match &d.max {
                                                    Ok(id) => match vm.vars[var_indices[id]]
                                                        .clone()
                                                        .number()
                                                    {
                                                        x if x.is_finite() => Ok(x),
                                                        _ => Err(
                                                            "value error: domain bound should be finite".into(),
                                                        ),
                                                    },
                                                    Err(e) => Err(format!("analysis error: {e}")),
                                                },
                                                Err(e) => Err(format!("parse error: {e}")),
                                            };
                                            expression.parametric_domain.min.0.underline.error =
                                                min.is_err();
                                            expression.parametric_domain.max.0.underline.error =
                                                max.is_err();

                                            let (min, max) = match (min, max) {
                                                (Ok(min), Ok(max)) => (min, max),
                                                (min, max) => {
                                                    output.data = OutputData::Error(
                                                        [("min", min), ("max", max)]
                                                            .into_iter()
                                                            .filter_map(|(n, m)| {
                                                                m.err().map(|e| {
                                                                    format!("(parametric {n}) {e}")
                                                                })
                                                            })
                                                            .collect::<Vec<_>>()
                                                            .join("\n"),
                                                    );
                                                    continue 'results_loop;
                                                }
                                            };

                                            if min > max {
                                                output.data = OutputData::Error("invalid domain limits: min should be less than max".into());
                                                expression
                                                    .parametric_domain
                                                    .min
                                                    .0
                                                    .underline
                                                    .error = true;
                                                expression
                                                    .parametric_domain
                                                    .max
                                                    .0
                                                    .underline
                                                    .error = true;
                                                continue 'results_loop;
                                            }

                                            expression.style.kind = ExpressionStyleKind::Parametric;
                                            PlotKind::Parametric(Domain { min, max })
                                        }
                                        PlotKind::Implicit => {
                                            expression.style.kind = ExpressionStyleKind::Equality;
                                            PlotKind::Implicit
                                        }
                                    };
                                    output.data =
                                        OutputData::Geometry(
                                            match (
                                                line_width.get(0),
                                                line_opacity.get(0),
                                                color.get(0),
                                            ) {
                                                (Some(width), Some(opacity), Some(color)) => {
                                                    Some(Geometry {
                                                        original_width: width,
                                                        width,
                                                        color: apply_opacity(color, opacity),
                                                        line_style: Default::default(),
                                                        point_style: Default::default(),
                                                        kind: GeometryKind::Plot {
                                                            kind,
                                                            inputs: parameters
                                                                .iter()
                                                                .map(|p| var_indices[p])
                                                                .collect(),
                                                            output: var_indices[&value],
                                                            instructions: functions
                                                                .remove(&ei)
                                                                .unwrap(),
                                                        },
                                                    })
                                                }
                                                _ => None,
                                            }
                                            .into_iter()
                                            .collect(),
                                        );
                                }

                                if match r {
                                    ExpressionResult::Plot {
                                        ref kind,
                                        ref parameters,
                                        ..
                                    } => {
                                        !matches!(kind, PlotKind::Parametric(_))
                                            && !parameters.is_empty()
                                    }
                                    _ => false,
                                } {
                                    output.ui = OutputUi::None;
                                }

                                if match r {
                                    ExpressionResult::Plot { parameters, .. } => {
                                        parameters.is_empty()
                                    }
                                    _ => true,
                                } {
                                    let v = var_indices[&id];
                                    match ty {
                                        Type::Number => {
                                            number_to_latex(&mut nodes, vm.vars[v].clone().number())
                                        }
                                        Type::NumberList => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, x) in a.borrow().as_slice().iter().enumerate() {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    number_to_latex(&mut inner, *x);
                                                } else {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                    break;
                                                }
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Point2 => {
                                            expression.style.kind = ExpressionStyleKind::Point;
                                            let x = vm.vars[v].clone().number();
                                            let y = vm.vars[v + 1.into()].clone().number();
                                            geometry.extend(make_point(x, y, 0));
                                            point2(&mut nodes, x, y);
                                        }
                                        Type::Point2List => {
                                            expression.style.kind = ExpressionStyleKind::PointList;
                                            let a = vm.vars[v].clone().list();
                                            let a = a.borrow();

                                            if let Some(width) = line_width.get(0)
                                                && let Some(opacity) = line_opacity.get(0)
                                                && let Some(color) = color.get(0)
                                            {
                                                geometry.push(Geometry {
                                                    original_width: width,
                                                    width,
                                                    color: apply_opacity(color, opacity),
                                                    line_style: Default::default(),
                                                    point_style: Default::default(),
                                                    kind: GeometryKind::Line(
                                                        a.chunks(2)
                                                            .map(|p| dvec2(p[0], p[1]))
                                                            .collect(),
                                                    ),
                                                });
                                            }

                                            let mut inner = vec![];
                                            for (i, &[x, y]) in a.as_chunks().0.iter().enumerate() {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    point2(&mut inner, x, y);
                                                } else if i == list_limit {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                }
                                                geometry.extend(make_point(x, y, i));
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Point3 => {
                                            let x = vm.vars[v].clone().number();
                                            let y = vm.vars[v + 1.into()].clone().number();
                                            let z = vm.vars[v + 2.into()].clone().number();
                                            point3(&mut nodes, x, y, z);
                                        }
                                        Type::Point3List => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, &[x, y, z]) in
                                                a.borrow().as_chunks().0.iter().enumerate()
                                            {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    point3(&mut inner, x, y, z);
                                                } else if i == list_limit {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                }
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Polygon => {
                                            expression.style.kind = ExpressionStyleKind::Polygon;
                                            let a = vm.vars[v].clone().list();
                                            let a = a.borrow();

                                            if let Some(opacity) = fill_opacity.get(0)
                                                && let Some(color) = color.get(0)
                                            {
                                                geometry.push(Geometry {
                                                    original_width: 0.0,
                                                    width: 0.0,
                                                    color: apply_opacity(color, opacity),
                                                    line_style: Default::default(),
                                                    point_style: Default::default(),
                                                    kind: GeometryKind::Fill(
                                                        a.chunks(2)
                                                            .map(|p| dvec2(p[0], p[1]))
                                                            .collect(),
                                                    ),
                                                });
                                            }

                                            if let Some(width) = line_width.get(0)
                                                && let Some(opacity) = line_opacity.get(0)
                                                && let Some(color) = color.get(0)
                                            {
                                                geometry.push(Geometry {
                                                    original_width: width,
                                                    width,
                                                    color: apply_opacity(color, opacity),
                                                    line_style: Default::default(),
                                                    point_style: Default::default(),
                                                    kind: GeometryKind::Line(
                                                        a.chunks(2)
                                                            .chain(a.chunks(2).next())
                                                            .map(|p| dvec2(p[0], p[1]))
                                                            .collect(),
                                                    ),
                                                });
                                            }
                                        }
                                        Type::PolygonList => {
                                            expression.style.kind = ExpressionStyleKind::Polygon;
                                            let a = vm.vars[v].clone().polygon_list();
                                            geometry.extend(
                                                a.borrow().iter().enumerate().flat_map(|(i, a)| {
                                                    let a = a.borrow();
                                                    let fill =
                                                        match (fill_opacity.get(i), color.get(i)) {
                                                            (Some(opacity), Some(color)) => {
                                                                Some(Geometry {
                                                                    original_width: 0.0,
                                                                    width: 0.0,
                                                                    color: apply_opacity(
                                                                        color, opacity,
                                                                    ),
                                                                    line_style: Default::default(),
                                                                    point_style: Default::default(),
                                                                    kind: GeometryKind::Fill(
                                                                        a.chunks(2)
                                                                            .map(|p| {
                                                                                dvec2(p[0], p[1])
                                                                            })
                                                                            .collect(),
                                                                    ),
                                                                })
                                                            }
                                                            _ => None,
                                                        };
                                                    let line = match (
                                                        line_width.get(i),
                                                        line_opacity.get(i),
                                                        color.get(i),
                                                    ) {
                                                        (
                                                            Some(width),
                                                            Some(opacity),
                                                            Some(color),
                                                        ) => Some(Geometry {
                                                            original_width: width,
                                                            width,
                                                            color: apply_opacity(color, opacity),
                                                            line_style: Default::default(),
                                                            point_style: Default::default(),
                                                            kind: GeometryKind::Line(
                                                                a.chunks(2)
                                                                    .chain(a.chunks(2).take(
                                                                        if a.len() > 2 {
                                                                            1
                                                                        } else {
                                                                            0
                                                                        },
                                                                    ))
                                                                    .map(|p| dvec2(p[0], p[1]))
                                                                    .collect(),
                                                            ),
                                                        }),
                                                        _ => None,
                                                    };
                                                    [fill, line].into_iter().flatten()
                                                }),
                                            );
                                        }
                                        Type::Color => {
                                            let r = vm.vars[v].clone().number();
                                            let g = vm.vars[v + 1.into()].clone().number();
                                            let b = vm.vars[v + 2.into()].clone().number();
                                            color_to_latex(&mut nodes, r, g, b);
                                        }
                                        Type::ColorList => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, &[r, g, b]) in
                                                a.borrow().as_chunks().0.iter().enumerate()
                                            {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    color_to_latex(&mut inner, r, g, b);
                                                } else if i == list_limit {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                }
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Bool | Type::BoolList => unreachable!(),
                                        Type::EmptyList => nodes.push(Node::DelimitedGroup {
                                            left: Bracket::Square,
                                            right: Bracket::Square,
                                            inner: vec![],
                                        }),
                                    }

                                    if ty.as_single() == Type::Polygon {
                                        output.ui = OutputUi::None;
                                    } else if !matches!(output.data, OutputData::DraggablePoint(_))
                                    {
                                        output.ui = OutputUi::Field(FieldUi::new(&nodes));
                                    }
                                    if let OutputData::None = output.data {
                                        output.data = OutputData::Geometry(geometry);
                                    }

                                    if let OutputData::DraggablePoint(Geometry { kind, .. }) =
                                        &output.data
                                    {
                                        expression.style.kind = ExpressionStyleKind::DraggablePoint;
                                        output.data = match (
                                            point_size.get(0),
                                            point_opacity.get(0),
                                            color.get(0),
                                        ) {
                                            (Some(size), Some(opacity), Some(color)) => {
                                                OutputData::DraggablePoint(Geometry {
                                                    original_width: size,
                                                    width: size,
                                                    color: apply_opacity(color, opacity),
                                                    line_style: Default::default(),
                                                    point_style: Default::default(),
                                                    kind: kind.clone(),
                                                })
                                            }
                                            _ => OutputData::None,
                                        };
                                    }
                                }
                            }
                            ExpressionResult::Slider { value, slider } => {
                                let mut error_msg = String::new();
                                let [min, max, step] = [
                                    ("min", &mut expression.slider.hard_min, &slider.min),
                                    ("max", &mut expression.slider.hard_max, &slider.max),
                                    ("step", &mut expression.slider.step, &slider.step),
                                ]
                                .map(|(name, field, result)| {
                                    field.0.underline.error = false;

                                    if field.0.is_empty() {
                                        return Ok(None);
                                    }

                                    let nl = if error_msg.is_empty() { "" } else { "\n" };

                                    if let Err(e) = &field.1 {
                                        write!(
                                            &mut error_msg,
                                            "{nl}(slider {name}) parse error: {e}"
                                        )
                                        .unwrap();
                                        field.0.underline.error = true;
                                        return Err(());
                                    }

                                    let id = match result.as_ref().unwrap() {
                                        Ok(id) => id,
                                        Err(e) => {
                                            write!(
                                                &mut error_msg,
                                                "{nl}(slider {name}) analysis error: {e}"
                                            )
                                            .unwrap();
                                            field.0.underline.error = true;
                                            return Err(());
                                        }
                                    };

                                    let value = vm.vars[var_indices[id]].clone().number();
                                    if !value.is_finite() {
                                        write!(
                                            &mut error_msg,
                                            "{nl}invalid slider {name}: value should be finite"
                                        )
                                        .unwrap();
                                        field.0.underline.error = true;
                                        return Err(());
                                    }

                                    Ok(Some(value))
                                });

                                if !error_msg.is_empty() {
                                    output.data = OutputData::Error(error_msg);
                                } else if let (Ok(Some(min)), Ok(Some(max))) = (min, max)
                                    && min > max
                                {
                                    output.data = OutputData::Error(
                                        "invalid slider limits: min should be less than max".into(),
                                    );
                                    expression.slider.hard_min.0.underline.error = true;
                                    expression.slider.hard_max.0.underline.error = true;
                                }

                                let value =
                                    value.map(|id| vm.vars[var_indices[&id]].clone().number());
                                let slider_min = min.ok().map(|min| {
                                    min.unwrap_or(apply_slider_step(
                                        value.unwrap_or(0.0).min(expression.slider.soft_min),
                                        0.0,
                                        step.ok().flatten().unwrap_or(SLIDER_STEP_DEFAULT),
                                        f64::floor,
                                    ))
                                });
                                let slider_max = max.ok().map(|max| {
                                    max.unwrap_or({
                                        let max =
                                            value.unwrap_or(0.0).max(expression.slider.soft_max);
                                        if let Ok(Some(step)) = step {
                                            let offset = min.ok().flatten().unwrap_or(0.0);
                                            apply_slider_step(max, offset, step, f64::ceil)
                                        } else {
                                            max
                                        }
                                    })
                                });

                                let Some(Ok(ast::Statement::Assignment { name, .. })) =
                                    &expression.ast
                                else {
                                    unreachable!("slider always has a name")
                                };

                                output.ui.set_slider(
                                    &mut expression.slider,
                                    name,
                                    value,
                                    slider_min,
                                    slider_max,
                                    step.ok().map(|step| step.unwrap_or(SLIDER_STEP_DEFAULT)),
                                );
                            }
                        }
                    }

                    self.vm_vars = vm.vars;

                    let mut has_error = false;
                    for (i, e) in self.expressions.iter().enumerate() {
                        if let OutputData::Error(e) = &e.output.data {
                            println!("expression {} {e}", i + 1);
                            has_error = true;
                        }
                    }
                    if has_error {
                        println!();
                    }
                }
            }
        }

        for expression in &mut self.expressions {
            if let OutputUi::Slider(SliderUi { value, .. }) = expression.output.ui
                && let Some(value) = value
                && expression.slider.fake_field_value != value
                && !expression.slider.fake_field.has_focus()
            {
                expression.slider.fake_field_value = value;
                expression.slider.fake_field =
                    MathField::from(&create_slider_latex(&expression.field, value));
            }
        }

        let mut geometry = None;

        if set(&mut self.redraw_geometry, false) {
            let mut regular_geometry = vec![];
            let mut draggable_points = vec![];
            let mut focussed_geometry = vec![];

            for e in &self.expressions {
                if e.style.hidden {
                    continue;
                }
                let style = |g: &mut Geometry| {
                    g.original_width = g.width;
                    g.line_style = e.style.line.style;
                    g.point_style = e.style.point.style;
                };
                let focus = |g: &mut Geometry| {
                    g.width *= match g.kind {
                        GeometryKind::Line(_) | GeometryKind::Plot { .. } => 1.4,
                        GeometryKind::Point { .. } => 1.2,
                        GeometryKind::Fill(_) => 1.0,
                    };
                    g.color[3] = match g.kind {
                        GeometryKind::Line(_) | GeometryKind::Plot { .. } => 1.0,
                        GeometryKind::Point { .. } | GeometryKind::Fill(_) => {
                            // Same as blending over itself, as if it was rendered twice
                            1.0 - (1.0 - g.color[3]).powi(2)
                        }
                    };
                };
                match &e.output.data {
                    OutputData::DraggablePoint(g) if e.style.point_enabled() => {
                        let mut g = g.clone();
                        style(&mut g);
                        let GeometryKind::Point { draggable, .. } = &mut g.kind else {
                            unreachable!();
                        };
                        if e.style.drag_enabled() {
                            let Some((_, mode)) = draggable else {
                                unreachable!()
                            };
                            *mode = e.style.drag.mode;
                            if e.has_focus() {
                                g.width *= 1.15;
                                draggable_points.push(g);
                            } else {
                                draggable_points.push(g);
                            }
                        } else {
                            *draggable = None;
                            if e.has_focus() {
                                focus(&mut g);
                                focussed_geometry.push(g);
                            } else {
                                regular_geometry.push(g);
                            }
                        }
                    }
                    OutputData::Geometry(geometry) => {
                        let geometry = geometry
                            .iter()
                            .filter(|g| match g.kind {
                                GeometryKind::Line(_) | GeometryKind::Plot { .. } => {
                                    e.style.line_enabled()
                                }
                                GeometryKind::Point { .. } => e.style.point_enabled(),
                                GeometryKind::Fill(_) => e.style.fill_enabled(),
                            })
                            .cloned()
                            .map(|mut g| {
                                style(&mut g);
                                g
                            });
                        if e.has_focus() {
                            for mut g in geometry {
                                focus(&mut g);
                                focussed_geometry.push(g);
                            }
                        } else {
                            regular_geometry.extend(geometry);
                        }
                    }
                    _ => {}
                }
            }

            regular_geometry.append(&mut draggable_points);
            regular_geometry.append(&mut focussed_geometry);

            geometry = Some((regular_geometry, self.vm_vars.clone()));
        }

        if response.requested_redraw {
            // If something wanted a redraw then some heights probably got
            // altered so it would be good to reclamp the scroll
            self.scroll(ctx, 0.0);
        }

        (response, geometry)
    }

    pub fn render(&mut self, ctx: &Context, bounds: Bounds, draw_quad: &mut impl FnMut(Quad)) {
        if bounds.size.x == 0.0 || bounds.size.y == 0.0 {
            return;
        }

        // shadow cast on graph by expression list
        draw_quad(Quad {
            kind: QuadKind::AlphaGradientU2,
            p0: dvec2(bounds.right() + 6.0, bounds.top()),
            p1: dvec2(bounds.right(), bounds.bottom()),
            color: (0, 0, 0, 0.11).to_rgbaf64(),
            ..Default::default()
        });

        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
        let gutter_width = ctx.round_nonzero(Self::GUTTER_WIDTH);
        let expression_width = bounds.size.x - 2.0 * separator_width - gutter_width;
        let expression_left = bounds.pos.x + gutter_width + separator_width;
        let mut expression_top = bounds.pos.y - self.scroll;
        let expressions_len = self.expressions.len();

        let separator_color = [216; 3];
        let gutter_color = [238; 3];

        // separator between expression list and graph
        draw_quad(Quad::rectangle(
            (bounds.right() - separator_width, bounds.top()),
            (bounds.right(), bounds.bottom()),
            separator_color,
        ));

        for (i, expression) in self.expressions.iter_mut().enumerate() {
            // Don't draw last faded expression if expressions are being reordered
            if i == expressions_len - 1
                && let Some((ClickDragTracker::Dragging, _, _)) = self.dragged_expression
            {
                continue;
            }

            let has_focus = expression.has_focus();
            let focus_color_or = |color| if has_focus { PRIMARY_COLOR } else { color };
            let is_being_dragged = match &self.dragged_expression {
                Some((ClickDragTracker::Dragging, j, _)) => j.0 == i,
                _ => false,
            };
            let is_last = i == expressions_len - 1;
            let expression_bottom;

            if is_being_dragged {
                // We will render dragged expression on top afterwards
                expression_bottom = expression_top + expression.height();

                if i < expressions_len - 2 {
                    // top separator for next expression
                    draw_quad(Quad::rectangle(
                        (bounds.left(), expression_bottom),
                        (bounds.right(), expression_bottom + separator_width),
                        separator_color,
                    ));
                }
            } else {
                expression.render(
                    ctx,
                    dvec2(expression_left, expression_top),
                    expression_width,
                    is_last,
                    draw_quad,
                );
                expression_bottom = expression_top + expression.height();
                // gutter separator
                draw_quad(Quad::rectangle(
                    (bounds.left() + gutter_width, expression_top),
                    (
                        bounds.left() + gutter_width + separator_width,
                        expression_bottom,
                    ),
                    focus_color_or(separator_color),
                ));
                // gutter fill
                draw_quad(Quad::rectangle(
                    (bounds.left(), expression_top),
                    (bounds.left() + gutter_width, expression_bottom),
                    focus_color_or(gutter_color),
                ));

                // gutter contents
                expression.render_gutter(
                    ctx,
                    Bounds {
                        pos: dvec2(bounds.left(), expression_top),
                        size: dvec2(gutter_width, expression.height()),
                    },
                    has_focus,
                    draw_quad,
                );

                // expression number
                render_label(
                    &(i + 1).to_string(),
                    dvec2(bounds.left() + 2.1, expression_top + 10.6),
                    11.4,
                    if has_focus {
                        (255, 255, 255, 1.0)
                    } else {
                        (0, 0, 0, 0.75)
                    },
                    Font::MainRegular,
                    draw_quad,
                );

                if !is_last {
                    if has_focus {
                        // replace separators with thicker focus color when focussed
                        // top separator
                        draw_quad(Quad::rectangle(
                            (bounds.left(), expression_top - separator_width),
                            (
                                bounds.right(),
                                expression_top + if i == 0 { 2.0 } else { 1.0 } * separator_width,
                            ),
                            focus_color_or(separator_color),
                        ));

                        // expression list/graph separator
                        draw_quad(Quad::rectangle(
                            (bounds.right() - 2.0 * separator_width, expression_top),
                            (bounds.right(), expression_bottom),
                            focus_color_or(separator_color),
                        ));
                    }

                    // bottom separator
                    draw_quad(Quad::rectangle(
                        (
                            bounds.left(),
                            expression_bottom - if has_focus { separator_width } else { 0.0 },
                        ),
                        (bounds.right(), expression_bottom + separator_width),
                        focus_color_or(separator_color),
                    ));
                }
            }

            if is_last && !is_being_dragged {
                // fade away gradient for last expression
                draw_quad(Quad {
                    kind: QuadKind::AlphaGradientV2,
                    p0: dvec2(bounds.left(), expression_top),
                    p1: dvec2(
                        bounds.left() + gutter_width + separator_width,
                        expression_bottom,
                    ),
                    color: DVec4::ONE,
                    ..Default::default()
                });
            }

            expression_top += expression.height() + separator_width;
        }

        // draw the expression being currently dragged on top
        if let Some((ClickDragTracker::Dragging, i, offset)) = self.dragged_expression {
            let expression_top = offset + ctx.cursor.y;
            let expression = &mut self.expressions[i];

            // background fill
            draw_quad(Quad::rectangle(
                (expression_left, expression_top),
                (
                    expression_left + expression_width,
                    expression_top + expression.height(),
                ),
                [255; 3],
            ));

            expression.render(
                ctx,
                dvec2(expression_left, expression_top),
                expression_width,
                false,
                draw_quad,
            );
            let expression_bottom = expression_top + expression.height();

            // gutter fill
            draw_quad(Quad::rectangle(
                (bounds.left(), expression_top),
                (
                    bounds.left() + gutter_width + separator_width,
                    expression_bottom,
                ),
                PRIMARY_COLOR,
            ));
            // top separator
            draw_quad(Quad::rectangle(
                (bounds.left(), expression_top - separator_width),
                (bounds.right(), expression_top + separator_width),
                PRIMARY_COLOR,
            ));
            // bottom separator
            draw_quad(Quad::rectangle(
                (bounds.left(), expression_bottom - separator_width),
                (bounds.right(), expression_bottom + separator_width),
                PRIMARY_COLOR,
            ));
            // side separator
            draw_quad(Quad::rectangle(
                (bounds.right() - 2.0 * separator_width, expression_top),
                (bounds.right(), expression_bottom),
                PRIMARY_COLOR,
            ));

            // gutter contents
            expression.render_gutter(
                ctx,
                Bounds {
                    pos: dvec2(bounds.left(), expression_top),
                    size: dvec2(gutter_width, expression.height()),
                },
                true,
                draw_quad,
            );

            let shadow_height = 12.0;
            let color = dvec4(0.0, 0.0, 0.0, 0.22);
            // top shadow
            draw_quad(Quad {
                kind: QuadKind::AlphaGradientV2,
                p0: dvec2(
                    bounds.left(),
                    expression_top - separator_width - shadow_height,
                ),
                p1: dvec2(bounds.right(), expression_top - separator_width),
                color,
                ..Default::default()
            });
            // bottom shadow
            draw_quad(Quad {
                kind: QuadKind::AlphaGradientV2,
                p0: dvec2(
                    bounds.left(),
                    expression_bottom + separator_width + shadow_height,
                ),
                p1: dvec2(bounds.right(), expression_bottom + separator_width),
                color,
                ..Default::default()
            });
        }
    }

    pub fn update_popup(&mut self, ctx: &Context, event: &Event, bounds: Bounds) -> Response {
        let mut response = Response::default();
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
        let gutter_width = ctx.round_nonzero(Self::GUTTER_WIDTH);
        let mut expression_top = bounds.pos.y - self.scroll;

        for expression in &mut self.expressions {
            let (r, m) = expression.update_popup(
                ctx,
                event,
                bounds,
                Bounds {
                    pos: dvec2(bounds.left(), expression_top),
                    size: dvec2(gutter_width, expression.height()),
                },
            );
            response = response.or(r);
            if matches!(m, Some(Message::ContentsChanged { .. })) {
                self.expressions_changed = true;
            }
            if set(&mut expression.style.changed, false) {
                self.redraw_geometry = true;
            }
            expression_top += expression.height() + separator_width;
        }

        response
    }

    pub fn render_popup(
        &mut self,
        ctx: &Context,
        bounds: Bounds,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
        let gutter_width = ctx.round_nonzero(Self::GUTTER_WIDTH);
        let mut expression_top = bounds.pos.y - self.scroll;

        for expression in &mut self.expressions {
            expression.render_popup(
                ctx,
                bounds,
                Bounds {
                    pos: dvec2(bounds.left(), expression_top),
                    size: dvec2(gutter_width, expression.height()),
                },
                draw_quad,
            );
            expression_top += expression.height() + separator_width;
        }
    }
}
