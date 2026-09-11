use std::fmt::Write;
use std::ops::DerefMut;
use std::{collections::HashMap, ops::Deref};

use bytemuck::{Zeroable, offset_of};
use derive_more::{Add, From, Into, Sub};
use glam::{DVec2, DVec4, Vec2, dvec2, dvec4, uvec2, vec2};
use typed_index_collections::{TiVec, ti_vec};
use winit::{
    event::{ElementState, MouseButton},
    window::CursorIcon,
};

use crate::katex_font::Font;
use crate::label::{Label, render_label};
use crate::ui::{ClickDragTracker, Color, PRIMARY_COLOR};
use crate::{
    AppGraphics,
    graph::{Geometry, GeometryKind},
    math_field::{Cursor, Interactiveness, MathField, Message, UserSelection},
    ui::{Bounds, Context, CursorMode, Event, Quad, QuadKind, Response},
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
        self.field.render(ctx, bounds, draw_quad);
    }
}

const SLIDER_SOFT_MIN_DEFAULT: f64 = -10.0;
const SLIDER_SOFT_MAX_DEFAULT: f64 = 10.0;
const SLIDER_STEP_DEFAULT: f64 = 0.0;
const PARAMETRIC_DOMAIN_MIN_DEFAULT: f64 = 0.0;
const PARAMETRIC_DOMAIN_MAX_DEFAULT: f64 = 1.0;

struct SliderUi {
    value: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    dragging: Option<f64>,
    point_hovered: bool,
    name: String,
    name_field: MathField,
    step_label: Label<'static>,

    play_button_hovered: bool,
    play_button_click_tracker: ClickDragTracker,
    animated_value: f64,
    expected_value: f64,
}

struct SliderEditLayout {
    min_field: Bounds,
    name: Bounds,
    max_field: Bounds,
    step_label: Bounds,
    step_field: Bounds,
    bounds: Bounds,
}

struct SliderBarLayout {
    min_field: Bounds,
    max_field: Bounds,
    bar_left: f64,
    bar_right: f64,
    point_bounds: Bounds,
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
            name_field: create_le_name_le(&name),
            name,
            step_label: Label::new("Step:", 15.7, Font::MainRegular),

            play_button_hovered: false,
            play_button_click_tracker: Default::default(),
            animated_value: 0.0,
            expected_value: 0.0,
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
        let is_slider_edit_shown = field_has_focus
            || slider.hard_min.0.has_focus()
            || slider.hard_max.0.has_focus()
            || slider.step.0.has_focus()
            || slider.hard_min.0.underline.error
            || slider.hard_max.0.underline.error
            || slider.step.0.underline.error;

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
            let name_size = self.name_field.expression_size().map(|s| ctx.ceil(s));
            let step_label_size = self.step_label.size();
            let step_field_size = slider.step.0.expression_size(ctx, true);

            let height = max([
                min_field_size.y,
                name_size.y,
                max_field_size.y,
                step_label_size.y,
                step_field_size.y,
            ]);

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
                    max_field.right() + 11.0,
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

            let bounds = union([min_field, name, max_field, step_label, step_field]);

            SliderLayout::Edit(SliderEditLayout {
                min_field,
                name,
                max_field,
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
            let point_bounds = Bounds {
                pos: point - point_radius,
                size: DVec2::splat(2.0 * point_radius),
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
                point_bounds,
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
        let result = match self.layout(ctx, padding, top_left, width, field_has_focus, slider) {
            SliderLayout::Edit(layout) => self.update_slider_edit(ctx, event, slider, layout),
            SliderLayout::Bar(layout) => self.update_slider_bar(ctx, event, slider, layout),
        };

        if event == &Event::AnimationFrame {
            slider.previous_update_time = ctx.time;
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
        let (min_response, mut min_message) = slider.hard_min.0.update(ctx, event, l.min_field);
        let (max_response, mut max_message) = slider.hard_max.0.update(ctx, event, l.max_field);
        let (step_response, mut step_message) = slider.step.0.update(ctx, event, l.step_field);

        match min_message {
            Some(Message::ContentsChanged { .. }) => {
                slider.soft_min = SLIDER_SOFT_MIN_DEFAULT;
                if !slider.hard_min.0.is_empty() {
                    slider.hard_min.1 = parse_standalone_expression(&slider.hard_min.0.to_latex());
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
                    slider.hard_max.1 = parse_standalone_expression(&slider.hard_max.0.to_latex());
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

        match step_message {
            Some(Message::ContentsChanged { .. }) => {
                if slider.step.0.is_empty() {
                    slider.step.1 = Ok(ast::Expression::Number(0.0));
                } else {
                    slider.step.1 = parse_standalone_expression(&slider.step.0.to_latex());
                }
            }
            Some(Message::Left) => {
                step_message = None;
                slider.step.0.unfocus();
                slider.hard_max.0.select_all();
            }
            Some(Message::Right | Message::Remove) => step_message = None,
            Some(Message::Up | Message::Down | Message::Add) => {
                slider.step.0.unfocus();
            }
            None => {}
        }

        let response = min_response.or(max_response).or(step_response);
        let message = min_message.or(max_message).or(step_message);
        (response, None, message, l.bounds)
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
        let new_point_hovered = l.point_bounds.contains(ctx.cursor);
        let new_slider_min_hovered = l.min_field.contains(ctx.cursor);
        let new_slider_max_hovered = l.max_field.contains(ctx.cursor);

        let mut new_value = None;
        let original_value = *value;
        let mut should_update_soft_bounds = false;

        match event {
            // drag point
            Event::CursorMoved { .. } if self.dragging.is_some() => {
                let offset = self.dragging.unwrap();
                // Not using `.clamp()` because it panics if sidebar is resized too small
                let point_x = (ctx.cursor.x + offset).max(l.bar_left).min(l.bar_right);
                *value = mix(*min, *max, unmix(point_x, l.bar_left, l.bar_right));
                *value = apply_slider(*value, *min, *max, *step);
                new_value = Some(*value);
                should_update_soft_bounds = true;
                response.consume_event();
                response.request_redraw();
            }
            Event::MouseInput(ElementState::Pressed, MouseButton::Left) => {
                if new_point_hovered {
                    // start dragging point
                    self.dragging = Some(l.point.x - ctx.cursor.x);
                    should_update_soft_bounds = true;
                    slider.is_playing = false;
                    response.consume_event();
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
            Event::AnimationFrame if slider.is_playing => {
                // animated_value is the raw unstepped value used to maintain
                // correct timing. check if it got invalidated by something like
                // an action updating the slider value
                if apply_slider(self.expected_value, *min, *max, *step)
                    != apply_slider(*value, *min, *max, *step)
                {
                    println!("unsynced!");
                    self.animated_value = *value;
                }

                let dt = ctx.time - slider.previous_update_time;
                let x = unmix(self.animated_value, *min, *max);
                let y = x + slider.play_direction * dt / slider.animation_period;
                let z = 1.0 - (y.rem_euclid(2.0) - 1.0).abs();
                slider.play_direction *= 1.0 - y.rem_euclid(2.0).floor() * 2.0;
                self.animated_value = mix(*min, *max, z);

                // TODO round value to fewest required decimal places based on animatino period,framerate,max-min,step
                if set(value, apply_slider(self.animated_value, *min, *max, *step)) {
                    new_value = Some(*value);
                    self.expected_value = *value;
                }

                should_update_soft_bounds = true;
                // TODO make sliders with a step only request an animation frame when
                // they actually need to change. we'd need to use ControlFlow::WaitUntil
                // or something and add a new method response.request_redraw_at(Instant)
                response.request_redraw();
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
        } else if new_slider_min_hovered || new_slider_max_hovered {
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
        slider.hard_min.0.render(ctx, l.min_field, draw_quad);
        self.name_field.render(ctx, l.name, draw_quad);
        slider.hard_max.0.render(ctx, l.max_field, draw_quad);
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
        let inner_radius = if self.point_hovered {
            l.point_radius
        } else {
            bar_radius
        };
        draw_quad(Quad::pill(
            l.point - inner_radius,
            l.point + inner_radius,
            PRIMARY_COLOR.with_opacity(opacity),
        ));

        // min/max field
        slider.hard_min.0.render(ctx, l.min_field, draw_quad);
        slider.hard_max.0.render(ctx, l.max_field, draw_quad);

        l.bounds.size.y
    }

    fn layout_gutter(&self, ctx: &Context, bounds: Bounds) -> Option<SliderGutterLayout> {
        // TODO when slider modes are implemented, "play indefinitely" only requires step to be Some
        let (Some(_value), Some(min), Some(max), Some(_step)) =
            (self.value, self.min, self.max, self.step)
        else {
            return None;
        };

        // TODO find a way to not repeat this validity check
        if min > max {
            return None;
        }

        let play_button_center = bounds.pos + bounds.size.x * dvec2(0.5, 0.752);
        let play_button_radius = 0.392 * bounds.size.x;
        let round = |p: DVec2| p.map(|x| ctx.round(x));
        let p0 = round(play_button_center - play_button_radius);
        let p1 = round(play_button_center + play_button_radius);
        let play_button = Bounds {
            pos: p0,
            size: p1 - p0,
        };
        Some(SliderGutterLayout {
            play_button_center,
            play_button_radius,
            play_button,
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
        let Some(l) = self.layout_gutter(ctx, bounds) else {
            return response;
        };

        match event {
            Event::MouseInput(ElementState::Pressed, MouseButton::Left)
                if self.play_button_hovered =>
            {
                self.play_button_click_tracker.press(ctx.cursor);
                response.consume_event();
                response.request_redraw();
            }
            Event::CursorMoved { .. } => {
                if set(
                    &mut self.play_button_hovered,
                    ctx.cursor.distance(l.play_button_center) <= l.play_button_radius,
                ) {
                    response.request_redraw();
                }

                if self.play_button_click_tracker.drag(ctx.cursor) {
                    response.request_redraw();
                }
            }
            Event::MouseInput(ElementState::Released, MouseButton::Left)
                if self.play_button_click_tracker.release().was_clicked() =>
            {
                slider.is_playing ^= true;
                if slider.is_playing {
                    let value = self.value.expect("play button only shows if no error");
                    slider.soft_min = slider.soft_min.min(value);
                    slider.soft_max = slider.soft_max.max(value);
                    slider.previous_update_time = ctx.time;
                    self.animated_value = value;
                    self.expected_value = value;
                }
                response.request_redraw();
            }
            _ => {}
        }

        if self.play_button_hovered {
            response.cursor_mode = CursorMode::Icon(CursorIcon::Pointer);
        }

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
        let Some(l) = self.layout_gutter(ctx, bounds) else {
            return;
        };
        draw_quad(Quad {
            kind: if slider.is_playing {
                QuadKind::SliderPlayingButton
            } else {
                QuadKind::SliderPausedButton
            },
            p0: l.play_button.pos,
            p1: l.play_button.pos + l.play_button.size,
            color: if expression_is_focussed {
                let opacity =
                    if self.play_button_hovered || self.play_button_click_tracker.is_pressed() {
                        1.0
                    } else {
                        0.9
                    };
                (255, 255, 255, opacity)
            } else {
                let opacity = if self.play_button_click_tracker.is_pressed() {
                    0.9
                } else if self.play_button_hovered {
                    0.7
                } else {
                    0.5
                };
                (0, 0, 0, opacity)
            }
            .to_rgbaf64(),
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

    fn set_domain(&mut self, name: &str) {
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

/// If the hard bound is empty, then the soft bound is used.
struct Slider {
    hard_min: (InlineField, Result<parse::ast::Expression, String>),
    soft_min: f64,
    hard_max: (InlineField, Result<parse::ast::Expression, String>),
    soft_max: f64,
    step: (InlineField, Result<parse::ast::Expression, String>),
    is_playing: bool,
    previous_update_time: f64,
    /// `1.0` or `-1.0`
    play_direction: f64,
    /// In seconds
    animation_period: f64,
    /// This is what is displayed to the user when a slider is shown instead of
    /// the actual math field. It's to handle desync between the actual value vs
    /// clamped slider value, e.g., when slider bounds get animated.
    // TODO fix this ugly solution, it's annoying having to maintain both fake_field and real field
    fake_field: MathField,
    fake_field_value: f64,
}

type ParametricDomain = Domain<(InlineField, Result<parse::ast::Expression, String>)>;

struct Expression {
    field: MathField,
    color: [f32; 4],
    slider: Slider,
    parametric_domain: ParametricDomain,
    ast: Option<Result<parse::ast::Statement, String>>,
    output: Output,
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

    fn new(color: [f32; 4]) -> Expression {
        Expression {
            field: Default::default(),
            color,
            slider: Slider {
                hard_min: (
                    InlineField::new(&SLIDER_SOFT_MIN_DEFAULT.to_string()),
                    Ok(ast::Expression::Number(SLIDER_SOFT_MIN_DEFAULT)),
                ),
                soft_min: SLIDER_SOFT_MIN_DEFAULT,
                hard_max: (
                    InlineField::new(&SLIDER_SOFT_MAX_DEFAULT.to_string()),
                    Ok(ast::Expression::Number(SLIDER_SOFT_MAX_DEFAULT)),
                ),
                soft_max: SLIDER_SOFT_MAX_DEFAULT,
                step: (InlineField::new(""), Ok(ast::Expression::Number(0.0))),
                is_playing: false,
                previous_update_time: 0.0,
                play_direction: 1.0,
                animation_period: 4.0,
                fake_field: Default::default(),
                fake_field_value: 0.0,
            },
            parametric_domain: Domain {
                min: (
                    InlineField::new(&PARAMETRIC_DOMAIN_MIN_DEFAULT.to_string()),
                    Ok(ast::Expression::Number(PARAMETRIC_DOMAIN_MIN_DEFAULT)),
                ),
                max: (
                    InlineField::new(&PARAMETRIC_DOMAIN_MAX_DEFAULT.to_string()),
                    Ok(ast::Expression::Number(PARAMETRIC_DOMAIN_MAX_DEFAULT)),
                ),
            },
            ast: None,
            output: Default::default(),
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

    fn from_latex(latex: &[latex_tree::Node], color: [f32; 4]) -> Self {
        let mut e = Expression::new(color);
        e.set_latex(latex);
        e
    }

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        top_left: DVec2,
        width: f64,
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
            size: dvec2(width - padding * 1.5, ctx.ceil(field.expression_size().y)),
        };
        height += field_bounds.size.y;
        // Bounds used for testing if field got clicked on (field_bounds + padding included)
        let field_hit_test_bounds = Bounds {
            pos: top_left,
            size: field_bounds.size + padding * 1.5,
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

        let field = match use_fake_field {
            true => &mut self.slider.fake_field,
            false => &mut self.field,
        };
        let (field_response, field_message) =
            field.update(ctx, event, field_bounds, Some(field_hit_test_bounds));

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
        match &mut self.output.ui {
            OutputUi::Slider(ui) => ui.update_gutter(ctx, event, &mut self.slider, bounds),
            _ => Response::default(),
        }
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
            size: dvec2(width - padding * 1.5, ctx.ceil(field.expression_size().y)),
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

        self.height = Some(height);
    }

    fn render_gutter(
        &mut self,
        ctx: &Context,
        bounds: Bounds,
        has_focus: bool,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        match &mut self.output.ui {
            OutputUi::Slider(ui) => {
                ui.render_gutter(ctx, bounds, has_focus, &mut self.slider, draw_quad)
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy, From, Into, Add, Sub, PartialEq, PartialOrd)]
pub struct ExpressionId(usize);

pub struct ExpressionList {
    expressions: TiVec<ExpressionId, Expression>,
    expressions_changed: bool,
    dragged_expression: Option<(ClickDragTracker, ExpressionId, f64)>,
    next_color: usize,
    scroll: f64,
    height: f64,
    vm_vars: vm::Vars,

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
    scale_factor: f32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Vertex {
    position: Vec2,
    color: [u8; 4],
    kind: u32,
    uv: [u16; 2],
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

const EXPRESSION_COLORS: &[[f32; 4]] = &[
    [0.780, 0.267, 0.251, 1.0],
    [0.176, 0.439, 0.702, 1.0],
    [0.204, 0.522, 0.263, 1.0],
    [0.376, 0.259, 0.651, 1.0],
    [0.0, 0.0, 0.0, 1.0],
];

impl ExpressionList {
    pub fn new(
        AppGraphics {
            device,
            queue,
            config,
            ..
        }: &AppGraphics,
    ) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("latex.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("latex_bind_group_layout"),
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
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
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
                            format: wgpu::VertexFormat::Unorm8x4,
                            offset: offset_of!(Vertex::zeroed(), Vertex, color) as _,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: offset_of!(Vertex::zeroed(), Vertex, kind) as _,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm16x2,
                            offset: offset_of!(Vertex::zeroed(), Vertex, uv) as _,
                            shader_location: 3,
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
            cache: None,
            multiview_mask: None,
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

        let expressions = [];
        let mut next_color = 0;
        let expressions = expressions
            .iter()
            .chain(Some(&""))
            .chain(expressions.is_empty().then_some(&""))
            .map(|s| {
                let color = EXPRESSION_COLORS[next_color % EXPRESSION_COLORS.len()];
                next_color += 1;
                Expression::from_latex(parse_latex(s).unwrap().as_slice(), color)
            })
            .collect();
        Self {
            expressions,
            expressions_changed: true,
            dragged_expression: None,
            next_color,
            scroll: 0.0,
            height: 0.0,
            vm_vars: Default::default(),

            pipeline,
            vertex_buffer,
            index_buffer,
            uniforms_buffer,
            bind_group,
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
        let color = EXPRESSION_COLORS[self.next_color % EXPRESSION_COLORS.len()];
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
        let mut redraw_geometry = false;

        if let Some((drag_tracker, i, offset)) = &mut self.dragged_expression {
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
                                    draggable: Some(id),
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

                    redraw_geometry |= set(i, new_i);
                    // TODO keep it scrolling even when cursor isn't moving and make it FPS-independent
                    self.scroll_y_into_view(ctx, ctx.cursor.y - (bounds.pos.y - self.scroll));

                    #[cfg(not(windows))]
                    let grabbing = CursorIcon::Grabbing;

                    // https://github.com/rust-windowing/winit/issues/1043
                    #[cfg(windows)]
                    let grabbing = CursorIcon::NsResize;

                    response.cursor_mode = CursorMode::Icon(grabbing);
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

                    let (r, m) = expression.update(
                        ctx,
                        event,
                        dvec2(expression_left, expression_top),
                        expression_width,
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

                    let drag_bounds = Bounds {
                        pos: dvec2(bounds.left(), expression_top),
                        size: dvec2(gutter_width, expression.height()) + separator_width,
                    };

                    if i.0 != expressions_len - 1 && drag_bounds.contains(ctx.cursor) {
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
                            self.expressions.remove(i);
                            self.expressions_changed = true;
                            if self.expressions.is_empty() {
                                let expression = self.new_expression();
                                self.expressions.push(expression);
                            }
                            self.expressions[ExpressionId(i.0.saturating_sub(1))].focus();
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
                redraw_geometry |= self.expressions_changed || original_focus != new_focus;

                if let Some(i) = new_focus
                    && original_focus != new_focus
                {
                    self.scroll_into_view(ctx, i);
                }

                if self.expressions_changed {
                    use latex_tree::Node::{self, Char as C};
                    let line_width = 2.5;
                    let fill_opacity = 0.4;
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

                    for (i, e) in self.expressions.iter_mut_enumerated() {
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
                                        width: 8.0,
                                        color: e.color,
                                        kind: GeometryKind::Point {
                                            p: dvec2(x, y),
                                            draggable: Some(i),
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

                        match r {
                            ExpressionResult::None => *output = Output::NONE,
                            ExpressionResult::Err(e) => {
                                *output = Output::new_error(format!("analysis error: {e}"))
                            }
                            ExpressionResult::Value(id, ty)
                            | ExpressionResult::Plot { value: id, ty, .. } => {
                                let mut nodes = vec![C('=')];

                                let color = expression.color;
                                let mut geometry = vec![];
                                let mut draw_point = |x: f64, y: f64| {
                                    geometry.push(Geometry {
                                        width: 8.0,
                                        color,
                                        kind: GeometryKind::Point {
                                            p: dvec2(x, y),
                                            draggable: None,
                                        },
                                    });
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
                                        PlotKind::Normal => PlotKind::Normal,
                                        PlotKind::Inverse => PlotKind::Inverse,
                                        PlotKind::Parametric(d) => {
                                            output
                                                .ui
                                                .set_domain(&analysis.freevars[&parameters[0]]);

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

                                            PlotKind::Parametric(Domain { min, max })
                                        }
                                        PlotKind::Implicit => PlotKind::Implicit,
                                    };
                                    output.data = OutputData::Geometry(vec![Geometry {
                                        width: line_width,
                                        color,
                                        kind: GeometryKind::Plot {
                                            kind,
                                            inputs: parameters
                                                .iter()
                                                .map(|p| var_indices[p])
                                                .collect(),
                                            output: var_indices[&value],
                                            instructions: functions.remove(&ei).unwrap(),
                                        },
                                    }]);
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
                                            let x = vm.vars[v].clone().number();
                                            let y = vm.vars[v + 1.into()].clone().number();
                                            draw_point(x, y);
                                            point2(&mut nodes, x, y);
                                        }
                                        Type::Point2List => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, &[x, y]) in
                                                a.borrow().as_chunks().0.iter().enumerate()
                                            {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    point2(&mut inner, x, y);
                                                } else if i == list_limit {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                }
                                                draw_point(x, y);
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
                                            let a = vm.vars[v].clone().list();
                                            let a = a.borrow();
                                            let fill = Geometry {
                                                width: line_width,
                                                color: [color[0], color[1], color[2], fill_opacity],
                                                kind: GeometryKind::Fill(
                                                    a.chunks(2)
                                                        .map(|p| dvec2(p[0], p[1]))
                                                        .collect(),
                                                ),
                                            };
                                            let line = Geometry {
                                                width: line_width,
                                                color,
                                                kind: GeometryKind::Line(
                                                    a.chunks(2)
                                                        .chain(a.chunks(2).next())
                                                        .map(|p| dvec2(p[0], p[1]))
                                                        .collect(),
                                                ),
                                            };
                                            geometry.extend([fill, line]);
                                        }
                                        Type::PolygonList => {
                                            let a = vm.vars[v].clone().polygon_list();
                                            geometry.extend(a.borrow().iter().flat_map(|a| {
                                                let a = a.borrow();
                                                let fill = Geometry {
                                                    width: line_width,
                                                    color: [
                                                        color[0],
                                                        color[1],
                                                        color[2],
                                                        fill_opacity,
                                                    ],
                                                    kind: GeometryKind::Fill(
                                                        a.chunks(2)
                                                            .map(|p| dvec2(p[0], p[1]))
                                                            .collect(),
                                                    ),
                                                };
                                                let line = Geometry {
                                                    width: line_width,
                                                    color,
                                                    kind: GeometryKind::Line(
                                                        a.chunks(2)
                                                            .chain(a.chunks(2).take(
                                                                if a.len() > 2 { 1 } else { 0 },
                                                            ))
                                                            .map(|p| dvec2(p[0], p[1]))
                                                            .collect(),
                                                    ),
                                                };
                                                [fill, line]
                                            }));
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

        if redraw_geometry {
            let mut regular_geometry = vec![];
            let mut draggable_points = vec![];
            let mut focussed_geometry = vec![];

            for e in &self.expressions {
                match &e.output.data {
                    OutputData::DraggablePoint(p) => {
                        let mut p = p.clone();
                        if e.has_focus() {
                            p.width *= 1.15;
                            draggable_points.push(p);
                        } else {
                            draggable_points.push(p);
                        }
                    }
                    OutputData::Geometry(geometry) => {
                        if e.has_focus() {
                            for mut g in geometry.iter().cloned() {
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
                                focussed_geometry.push(g);
                            }
                        } else {
                            regular_geometry.extend_from_slice(geometry);
                        }
                    }
                    _ => {}
                }
            }

            regular_geometry.append(&mut draggable_points);
            regular_geometry.append(&mut focussed_geometry);

            geometry = Some((regular_geometry, self.vm_vars.clone()));
        }

        self.expressions_changed = false;

        if response.requested_redraw {
            // If something wanted a redraw then some heights probably got
            // altered so it would be good to reclamp the scroll
            self.scroll(ctx, 0.0);
        }

        (response, geometry)
    }

    pub fn render(
        &mut self,
        ctx: &Context,
        AppGraphics {
            device,
            queue,
            config,
            ..
        }: &AppGraphics,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        bounds: Bounds,
    ) {
        let mut indices = vec![];
        let mut vertices = vec![];
        let draw_quad = &mut |quad: Quad| {
            let kind = quad.kind as u32;
            let p0 = (ctx.scale_factor * quad.p0).as_vec2();
            let p1 = (ctx.scale_factor * quad.p1).as_vec2();
            let to_unorm = |x: f64, s: f64| (x.clamp(0.0, 1.0) * s).round();
            let uv0 = quad.uv0.to_array().map(|x| to_unorm(x, 65535.0) as u16);
            let uv1 = quad.uv1.to_array().map(|x| to_unorm(x, 65535.0) as u16);
            let color = quad.color.to_array().map(|x| to_unorm(x, 255.0) as u8);

            indices.push(vertices.len() as u32);
            indices.push(vertices.len() as u32 + 1);
            indices.push(vertices.len() as u32 + 2);
            indices.push(vertices.len() as u32 + 3);
            indices.push(0xffffffff);

            vertices.push(Vertex {
                position: p0,
                color,
                kind,
                uv: uv0,
            });
            vertices.push(Vertex {
                position: vec2(p1.x, p0.y),
                color,
                kind,
                uv: [uv1[0], uv0[1]],
            });
            vertices.push(Vertex {
                position: vec2(p0.x, p1.y),
                color,
                kind,
                uv: [uv0[0], uv1[1]],
            });
            vertices.push(Vertex {
                position: p1,
                color,
                kind,
                uv: uv1,
            });
        };

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

                if i < expressions_len - 1 {
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

            if i == expressions_len - 1 && !is_being_dragged {
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
                scale_factor: ctx.scale_factor as f32,
            }]),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("latex"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        ctx.set_scissor_rect(&mut pass, bounds);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw_indexed(0..indices.len() as _, 0, 0..1);
    }
}
